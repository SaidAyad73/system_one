"""Capability Resolver: evaluates computational compatibility against TypeSafe and OpenAI Decisions documentation."""

from __future__ import annotations

from typing import Any

from .models import (
     APIProvider,
    CapabilityReport,
    ClassifiedRequest,
    CompatibilityLevel,
    DATA_URL_RE,
    OPENAI_QUESTION_TYPES,
    TYPESAFE_IMAGE_FORMATS,
    TYPESAFE_QUESTION_TYPES,
)


class CapabilityResolver:
    """Determines native and cross-API translation capabilities without performing routing or translation."""

    def evaluate_native_capability(self, classified: ClassifiedRequest) -> CapabilityReport:
        """Evaluates whether the request's original destination can compute it natively."""
        destination = classified.original_destination
        if destination == APIProvider.TYPESAFE:
            return self._evaluate_typesafe_native(classified)
        if destination == APIProvider.OPENAI_DECISIONS:
            return self._evaluate_openai_native(classified)
        return CapabilityReport(
            provider=APIProvider.UNKNOWN,
            level=CompatibilityLevel.UNKNOWN,
            can_execute=False,
            reasons=("Original destination API could not be identified.",),
        )

    def evaluate_translation_capability(
        self,
        classified: ClassifiedRequest,
        target: APIProvider,
    ) -> CapabilityReport:
        """Evaluates whether a request can be translated to and computed by the alternative API."""
        if target == APIProvider.OPENAI_DECISIONS and classified.original_destination == APIProvider.TYPESAFE:
            return self._evaluate_typesafe_to_openai_translation(classified)
        if target == APIProvider.TYPESAFE and classified.original_destination == APIProvider.OPENAI_DECISIONS:
            return self._evaluate_openai_to_typesafe_translation(classified)
        return CapabilityReport(
            provider=target,
            level=CompatibilityLevel.UNKNOWN,
            can_execute=False,
            reasons=("Translation capability cannot be determined for unknown provider pair.",),
        )

    # -------------------------------------------------------------------------
    # Native TypeSafe capability evaluation
    # -------------------------------------------------------------------------

    def _evaluate_typesafe_native(self, classified: ClassifiedRequest) -> CapabilityReport:
        method = classified.raw_request.method.upper()
        path = classified.path.rstrip("/") or "/"

        if method == "GET" and path == "/v1/models":
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.FULL,
                can_execute=True,
                reasons=("GET /v1/models is natively supported by TypeSafe.",),
            )

        if method == "GET" and path == "/v1/organization/usage/decisions":
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=("Organization decisions usage endpoint is an OpenAI-only endpoint.",),
                unsupported_operations=("GET /v1/organization/usage/decisions",),
            )

        if method != "POST" or path not in ("/v1/systemone", "/"):
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNKNOWN,
                can_execute=False,
                reasons=(f"Unrecognized endpoint or method for TypeSafe: {method} {path}.",),
            )

        if classified.is_multipart:
            has_images_header = self._has_typesafe_images_header(classified)
            if not has_images_header:
                return CapabilityReport(
                    provider=APIProvider.TYPESAFE,
                    level=CompatibilityLevel.UNSUPPORTED,
                    can_execute=False,
                    reasons=("Multipart image requests require TypeSafe-Danger-Zone-Images: true.",),
                    constraints_violated=("missing_danger_zone_images_header",),
                )
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.FULL,
                can_execute=True,
                reasons=("Multipart request with TypeSafe-Danger-Zone-Images header.",),
            )

        if classified.json_parse_error is not None or not isinstance(classified.parsed_json, dict):
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNKNOWN,
                can_execute=False,
                reasons=("Request body is not a valid JSON object; deferring validation to upstream API.",),
            )

        body: dict[str, Any] = classified.parsed_json

        if "state" not in body or "questions" not in body:
            if "input" in body and isinstance(body.get("questions"), list):
                return CapabilityReport(
                    provider=APIProvider.TYPESAFE,
                    level=CompatibilityLevel.UNSUPPORTED,
                    can_execute=False,
                    reasons=("Payload uses OpenAI Decisions schema (input + questions list) rather than TypeSafe schema.",),
                    constraints_violated=("missing_state_or_questions_map",),
                )
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNKNOWN,
                can_execute=False,
                reasons=("Required TypeSafe fields 'state' or 'questions' are missing.",),
            )

        state = body["state"]
        if state is None or isinstance(state, (bool, int, float)):
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=("TypeSafe 'state' must be a string, JSON object, or JSON array.",),
                constraints_violated=("invalid_state_type",),
            )

        questions = body["questions"]
        if not isinstance(questions, dict):
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=("TypeSafe 'questions' must be a JSON object mapping question IDs to questions.",),
                constraints_violated=("questions_not_object",),
            )

        if len(questions) < 1:
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=("TypeSafe 'questions' requires at least 1 question (minProperties: 1).",),
                constraints_violated=("empty_questions",),
            )

        reasons: list[str] = []
        unsupported_ops: list[str] = []
        unsupported_params: list[str] = []
        constraints_violated: list[str] = []

        has_images_header = self._has_typesafe_images_header(classified)
        assets = body.get("assets")
        has_valid_image_assets = False

        if assets is not None:
            if not isinstance(assets, dict) or len(assets) < 1:
                reasons.append("TypeSafe 'assets' must be a non-empty object when present.")
                constraints_violated.append("invalid_assets_map")
            else:
                if not has_images_header:
                    reasons.append("TypeSafe image assets require header 'TypeSafe-Danger-Zone-Images: true'.")
                    constraints_violated.append("missing_danger_zone_images_header")
                for asset_key, asset_val in assets.items():
                    if not isinstance(asset_val, dict):
                        reasons.append(f"Asset '{asset_key}' must be an object.")
                        constraints_violated.append(f"assets.{asset_key}.not_object")
                        continue
                    asset_type = asset_val.get("type")
                    if asset_type != "image":
                        reasons.append(f"Asset '{asset_key}' has unsupported type '{asset_type}' (only 'image' is supported).")
                        unsupported_ops.append(f"asset_type:{asset_type}")
                        continue
                    if "url" in asset_val or "image_url" in asset_val:
                        reasons.append(f"Asset '{asset_key}' uses a remote URL; TypeSafe requires base64 or multipart part.")
                        unsupported_params.append(f"assets.{asset_key}.image_url")
                        constraints_violated.append("remote_image_url_unsupported")
                        continue
                    part = asset_val.get("part")
                    data_b64 = asset_val.get("data_base64")
                    fmt = (asset_val.get("format") or "").lower()
                    if part and not classified.is_multipart:
                        reasons.append(f"Asset '{asset_key}' specifies multipart 'part' in a non-multipart JSON request.")
                        constraints_violated.append("multipart_asset_in_json_request")
                    elif data_b64:
                        if fmt not in TYPESAFE_IMAGE_FORMATS:
                            reasons.append(
                                f"Asset '{asset_key}' has unsupported image format '{fmt}'; TypeSafe supports png, jpg, jpeg."
                            )
                            constraints_violated.append("unsupported_image_format")
                        else:
                            has_valid_image_assets = True
                    else:
                        reasons.append(f"Asset '{asset_key}' is missing 'data_base64' or 'part'.")
                        constraints_violated.append("missing_asset_data")

        for q_id, q in questions.items():
            if not isinstance(q, dict):
                return CapabilityReport(
                    provider=APIProvider.TYPESAFE,
                    level=CompatibilityLevel.UNKNOWN,
                    can_execute=False,
                    reasons=(f"Question '{q_id}' is not a JSON object.",),
                )
            q_type = q.get("type")
            if q_type not in TYPESAFE_QUESTION_TYPES:
                if q_type == "predicate":
                    reasons.append(f"Question '{q_id}' uses OpenAI type 'predicate' instead of TypeSafe 'noul'.")
                    unsupported_ops.append("question_type:predicate")
                    continue
                return CapabilityReport(
                    provider=APIProvider.TYPESAFE,
                    level=CompatibilityLevel.UNKNOWN,
                    can_execute=False,
                    reasons=(f"Question '{q_id}' has unrecognized type '{q_type}'.",),
                )

            if q_type == "choice":
                criteria = q.get("criteria")
                if not isinstance(criteria, dict):
                    reasons.append(f"Choice question '{q_id}' requires 'criteria' as an object map.")
                    constraints_violated.append(f"questions.{q_id}.criteria_not_object")
                elif len(criteria) < 1 or len(criteria) > 255:
                    reasons.append(
                        f"Choice question '{q_id}' has {len(criteria)} options; TypeSafe supports 1 to 255 choices."
                    )
                    constraints_violated.append("choice_criteria_count_out_of_range")

            elif q_type == "score":
                criteria = q.get("criteria")
                if not isinstance(criteria, list):
                    reasons.append(f"Score question '{q_id}' requires 'criteria' as an ordered list.")
                    constraints_violated.append(f"questions.{q_id}.criteria_not_list")
                elif len(criteria) < 1 or len(criteria) > 10:
                    reasons.append(
                        f"Score question '{q_id}' has {len(criteria)} levels; TypeSafe supports 1 to 10 levels."
                    )
                    constraints_violated.append("score_criteria_count_out_of_range")

            elif q_type == "box":
                if not has_images_header:
                    reasons.append(f"Box question '{q_id}' requires header 'TypeSafe-Danger-Zone-Images: true'.")
                    constraints_violated.append("missing_danger_zone_images_header")
                if not has_valid_image_assets:
                    reasons.append(f"Box question '{q_id}' requires an image asset in 'assets'.")
                    constraints_violated.append("box_without_image_asset")

        if reasons or unsupported_ops or unsupported_params or constraints_violated:
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=tuple(reasons),
                unsupported_operations=tuple(unsupported_ops),
                unsupported_parameters=tuple(unsupported_params),
                constraints_violated=tuple(constraints_violated),
            )

        return CapabilityReport(
            provider=APIProvider.TYPESAFE,
            level=CompatibilityLevel.FULL,
            can_execute=True,
            reasons=("Request satisfies all documented TypeSafe System One native capabilities and constraints.",),
        )

    # -------------------------------------------------------------------------
    # Native OpenAI Decisions capability evaluation
    # -------------------------------------------------------------------------

    def _evaluate_openai_native(self, classified: ClassifiedRequest) -> CapabilityReport:
        method = classified.raw_request.method.upper()
        path = classified.path.rstrip("/") or "/"

        if method == "GET" and path == "/v1/organization/usage/decisions":
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.FULL,
                can_execute=True,
                reasons=("GET /v1/organization/usage/decisions is natively supported by OpenAI Decisions API.",),
            )

        if method == "GET" and path == "/v1/models":
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=("GET /v1/models is a TypeSafe endpoint, not an OpenAI Decisions endpoint.",),
                unsupported_operations=("GET /v1/models",),
            )

        if method != "POST" or path not in ("/v1/decisions", "/"):
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNKNOWN,
                can_execute=False,
                reasons=(f"Unrecognized endpoint or method for OpenAI Decisions API: {method} {path}.",),
            )

        if classified.is_multipart:
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=("OpenAI Decisions API does not support multipart/form-data requests.",),
                constraints_violated=("multipart_unsupported",),
            )

        if classified.json_parse_error is not None or not isinstance(classified.parsed_json, dict):
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNKNOWN,
                can_execute=False,
                reasons=("Request body is not a valid JSON object; deferring validation to upstream API.",),
            )

        body: dict[str, Any] = classified.parsed_json

        if "input" not in body or "questions" not in body:
            if "state" in body and isinstance(body.get("questions"), dict):
                return CapabilityReport(
                    provider=APIProvider.OPENAI_DECISIONS,
                    level=CompatibilityLevel.UNSUPPORTED,
                    can_execute=False,
                    reasons=("Payload uses TypeSafe schema (state + questions map) rather than OpenAI Decisions schema.",),
                    constraints_violated=("missing_input_or_questions_list",),
                )
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNKNOWN,
                can_execute=False,
                reasons=("Required OpenAI Decisions fields 'input' or 'questions' are missing.",),
            )

        reasons: list[str] = []
        unsupported_ops: list[str] = []
        unsupported_params: list[str] = []
        constraints_violated: list[str] = []

        shared_input = body["input"]
        if isinstance(shared_input, dict):
            reasons.append(
                "OpenAI Decisions 'input' must be a string or an array of messages/tool items; "
                "raw JSON objects must be wrapped in a user message 'input_json' part."
            )
            constraints_violated.append("raw_json_object_in_input")
        elif isinstance(shared_input, list):
            if len(shared_input) < 1:
                reasons.append("OpenAI Decisions 'input' array cannot be empty.")
                constraints_violated.append("empty_input_array")
            else:
                self._validate_openai_input_items(
                    shared_input,
                    reasons=reasons,
                    unsupported_ops=unsupported_ops,
                    unsupported_params=unsupported_params,
                    constraints_violated=constraints_violated,
                )
        elif not isinstance(shared_input, str):
            reasons.append("OpenAI Decisions 'input' must be a string or array of items.")
            constraints_violated.append("invalid_input_type")

        questions = body["questions"]
        if not isinstance(questions, list):
            reasons.append("OpenAI Decisions 'questions' must be an array of question objects.")
            constraints_violated.append("questions_not_list")
        else:
            if len(questions) < 1 or len(questions) > 400:
                reasons.append(
                    f"OpenAI Decisions API supports 1 to 400 questions per request; received {len(questions)}."
                )
                constraints_violated.append("question_count_out_of_range")

            seen_names: set[str] = set()
            for idx, q in enumerate(questions):
                if not isinstance(q, dict):
                    return CapabilityReport(
                        provider=APIProvider.OPENAI_DECISIONS,
                        level=CompatibilityLevel.UNKNOWN,
                        can_execute=False,
                        reasons=(f"Question at index {idx} is not a JSON object.",),
                    )
                q_name = q.get("name")
                if not isinstance(q_name, str) or not q_name:
                    reasons.append(f"Question at index {idx} is missing a non-empty string 'name'.")
                    constraints_violated.append(f"questions[{idx}].missing_name")
                elif q_name in seen_names:
                    reasons.append(f"Duplicate question name '{q_name}'; OpenAI requires unique question names.")
                    constraints_violated.append("duplicate_question_name")
                else:
                    seen_names.add(q_name)

                q_type = q.get("type")
                if q_type not in OPENAI_QUESTION_TYPES:
                    if q_type in ("noul", "box"):
                        reasons.append(f"Question '{q_name or idx}' uses TypeSafe type '{q_type}'.")
                        unsupported_ops.append(f"question_type:{q_type}")
                        continue
                    return CapabilityReport(
                        provider=APIProvider.OPENAI_DECISIONS,
                        level=CompatibilityLevel.UNKNOWN,
                        can_execute=False,
                        reasons=(f"Question '{q_name or idx}' has unrecognized type '{q_type}'.",),
                    )

                instructions = q.get("instructions")
                if instructions is None or (isinstance(instructions, str) and instructions == ""):
                    reasons.append(f"Question '{q_name or idx}' is missing required 'instructions'.")
                    constraints_violated.append(f"questions[{idx}].missing_instructions")
                elif isinstance(instructions, dict):
                    reasons.append(
                        f"Question '{q_name or idx}' has a structured object for 'instructions'; "
                        "OpenAI requires a string or array of input_text/input_image parts."
                    )
                    constraints_violated.append(f"questions[{idx}].structured_instructions")
                elif isinstance(instructions, list):
                    if len(instructions) < 1:
                        reasons.append(f"Question '{q_name or idx}' has an empty 'instructions' array.")
                        constraints_violated.append(f"questions[{idx}].empty_instructions")
                    for p_idx, part in enumerate(instructions):
                        if not isinstance(part, dict):
                            reasons.append(
                                f"Question '{q_name or idx}' instructions[{p_idx}] is not a content part object."
                            )
                            constraints_violated.append(f"questions[{idx}].instructions_not_parts")
                            continue
                        p_type = part.get("type")
                        if p_type not in ("input_text", "input_image"):
                            reasons.append(
                                f"Question '{q_name or idx}' instructions[{p_idx}] has unsupported part type '{p_type}'."
                            )
                            unsupported_params.append(f"questions[{idx}].instructions[{p_idx}].type")
                        if "prompt_cache_breakpoint" in part:
                            reasons.append(
                                f"Question '{q_name or idx}' instructions[{p_idx}] contains 'prompt_cache_breakpoint', "
                                "which is unsupported in question instructions."
                            )
                            unsupported_params.append(
                                f"questions[{idx}].instructions[{p_idx}].prompt_cache_breakpoint"
                            )
                            constraints_violated.append("breakpoint_in_question_instructions")
                        if p_type == "input_image" and "file_id" in part:
                            reasons.append(
                                f"Question '{q_name or idx}' instructions[{p_idx}] uses unsupported 'file_id'."
                            )
                            unsupported_params.append(f"questions[{idx}].instructions[{p_idx}].file_id")

                if q_type == "predicate":
                    if "criteria" in q and q["criteria"] is not None:
                        reasons.append(f"Predicate question '{q_name or idx}' does not support 'criteria'.")
                        unsupported_params.append(f"questions[{idx}].criteria")

                elif q_type == "choice":
                    choices = q.get("choices")
                    if not isinstance(choices, list) or len(choices) < 1:
                        reasons.append(f"Choice question '{q_name or idx}' requires a non-empty 'choices' array.")
                        constraints_violated.append(f"questions[{idx}].invalid_choices")
                    else:
                        seen_values: set[str] = set()
                        for c_idx, c in enumerate(choices):
                            if not isinstance(c, dict) or not isinstance(c.get("value"), str):
                                reasons.append(
                                    f"Choice question '{q_name or idx}' choices[{c_idx}] requires a string 'value'."
                                )
                                constraints_violated.append(f"questions[{idx}].choices[{c_idx}].invalid_value")
                                continue
                            val = c["value"]
                            if val in seen_values:
                                reasons.append(
                                    f"Choice question '{q_name or idx}' has duplicate choice value '{val}'."
                                )
                                constraints_violated.append(f"questions[{idx}].duplicate_choice_value")
                            seen_values.add(val)
                            desc = c.get("description")
                            if desc is not None and not isinstance(desc, str):
                                reasons.append(
                                    f"Choice question '{q_name or idx}' choices[{c_idx}].description must be a string."
                                )
                                constraints_violated.append(f"questions[{idx}].structured_choice_description")

                elif q_type == "score":
                    levels = q.get("levels")
                    if not isinstance(levels, list):
                        reasons.append(f"Score question '{q_name or idx}' requires a 'levels' array.")
                        constraints_violated.append(f"questions[{idx}].levels_not_list")
                    elif len(levels) < 2 or len(levels) > 10:
                        reasons.append(
                            f"Score question '{q_name or idx}' has {len(levels)} levels; "
                            "OpenAI Decisions API requires 2 to 10 ordered levels."
                        )
                        constraints_violated.append("score_levels_count_out_of_range")
                    else:
                        for l_idx, lvl in enumerate(levels):
                            if not isinstance(lvl, dict) or not isinstance(lvl.get("label"), str):
                                reasons.append(
                                    f"Score question '{q_name or idx}' levels[{l_idx}] requires a string 'label'."
                                )
                                constraints_violated.append(f"questions[{idx}].levels[{l_idx}].invalid_label")
                                continue
                            desc = lvl.get("description")
                            if desc is not None and not isinstance(desc, str):
                                reasons.append(
                                    f"Score question '{q_name or idx}' levels[{l_idx}].description must be a string."
                                )
                                constraints_violated.append(f"questions[{idx}].structured_level_description")

        if reasons or unsupported_ops or unsupported_params or constraints_violated:
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=tuple(reasons),
                unsupported_operations=tuple(unsupported_ops),
                unsupported_parameters=tuple(unsupported_params),
                constraints_violated=tuple(constraints_violated),
            )

        return CapabilityReport(
            provider=APIProvider.OPENAI_DECISIONS,
            level=CompatibilityLevel.FULL,
            can_execute=True,
            reasons=("Request satisfies all documented OpenAI Decisions native capabilities and constraints.",),
        )

    def _validate_openai_input_items(
        self,
        items: list[Any],
        *,
        reasons: list[str],
        unsupported_ops: list[str],
        unsupported_params: list[str],
        constraints_violated: list[str],
    ) -> None:
        seen_call_ids: set[str] = set()
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                reasons.append(f"Input item at index {idx} must be a JSON object.")
                constraints_violated.append(f"input[{idx}].not_object")
                continue

            item_type = item.get("type")
            role = item.get("role")

            if item_type in ("input_audio", "audio", "input_file", "file", "item_reference"):
                reasons.append(f"Input item at index {idx} uses unsupported item type '{item_type}'.")
                unsupported_ops.append(f"input_item:{item_type}")
                continue

            if "file_id" in item:
                reasons.append(f"Input item at index {idx} uses unsupported 'file_id'.")
                unsupported_params.append(f"input[{idx}].file_id")
                continue

            if item_type in ("function_call", "custom_tool_call"):
                call_id = item.get("call_id")
                if isinstance(call_id, str) and call_id:
                    seen_call_ids.add(call_id)
                else:
                    reasons.append(f"Tool call at input[{idx}] is missing a 'call_id'.")
                    constraints_violated.append(f"input[{idx}].missing_call_id")
                continue

            if item_type in ("function_call_output", "custom_tool_call_output"):
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or call_id not in seen_call_ids:
                    reasons.append(
                        f"Tool output at input[{idx}] references unmatched or missing call_id '{call_id}'."
                    )
                    constraints_violated.append(f"input[{idx}].unmatched_call_id")
                output_content = item.get("output")
                if isinstance(output_content, list):
                    for o_idx, part in enumerate(output_content):
                        if isinstance(part, dict):
                            p_type = part.get("type")
                            if p_type == "input_json":
                                reasons.append(
                                    f"Tool output at input[{idx}].output[{o_idx}] cannot contain 'input_json' "
                                    "(input_json is only supported in user messages)."
                                )
                                constraints_violated.append("input_json_outside_user_message")
                            if "file_id" in part:
                                reasons.append(f"Tool output at input[{idx}].output[{o_idx}] uses 'file_id'.")
                                unsupported_params.append(f"input[{idx}].output[{o_idx}].file_id")
                continue

            if role in ("user", "system", "developer", "assistant"):
                content = item.get("content")
                if isinstance(content, list):
                    for c_idx, part in enumerate(content):
                        if not isinstance(part, dict):
                            continue
                        p_type = part.get("type")
                        if p_type in ("input_audio", "audio", "input_file", "file", "item_reference"):
                            reasons.append(f"Input[{idx}].content[{c_idx}] uses unsupported part type '{p_type}'.")
                            unsupported_ops.append(f"content_part:{p_type}")
                        if "file_id" in part:
                            reasons.append(f"Input[{idx}].content[{c_idx}] uses unsupported 'file_id'.")
                            unsupported_params.append(f"input[{idx}].content[{c_idx}].file_id")
                        if p_type == "input_json" and role != "user":
                            reasons.append(
                                f"Input[{idx}].content[{c_idx}] uses 'input_json' in a '{role}' message; "
                                "'input_json' is only supported in user messages."
                            )
                            constraints_violated.append("input_json_outside_user_message")
                        if role in ("system", "developer") and p_type not in ("input_text", "text"):
                            reasons.append(
                                f"Input[{idx}].content[{c_idx}] in '{role}' message has type '{p_type}'; "
                                "system and developer messages support text only."
                            )
                            constraints_violated.append("non_text_in_system_or_developer_message")
                        if role == "assistant" and p_type not in ("output_text", "refusal", "input_text", "text"):
                            reasons.append(
                                f"Input[{idx}].content[{c_idx}] in 'assistant' message has unsupported type '{p_type}'."
                            )
                            constraints_violated.append("invalid_assistant_content_type")

    # -------------------------------------------------------------------------
    # Cross-API Translation Capability: TypeSafe -> OpenAI Decisions
    # -------------------------------------------------------------------------

    def _evaluate_typesafe_to_openai_translation(self, classified: ClassifiedRequest) -> CapabilityReport:
        method = classified.raw_request.method.upper()
        path = classified.path.rstrip("/") or "/"

        if method != "POST" or path not in ("/v1/systemone", "/"):
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=(f"Endpoint {method} {path} cannot be translated to OpenAI Decisions API.",),
                unsupported_operations=(f"{method} {path}",),
            )

        if classified.is_multipart:
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=("Multipart form-data requests cannot be translated to OpenAI Decisions JSON API.",),
                unsupported_operations=("multipart/form-data",),
            )

        if classified.json_parse_error is not None or not isinstance(classified.parsed_json, dict):
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNKNOWN,
                can_execute=False,
                reasons=("Malformed JSON cannot be translated.",),
            )

        body: dict[str, Any] = classified.parsed_json
        if "state" not in body or not isinstance(body.get("questions"), dict):
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNKNOWN,
                can_execute=False,
                reasons=("Request does not match TypeSafe schema.",),
            )

        state = body["state"]
        if state is None or isinstance(state, (bool, int, float)):
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=("Invalid 'state' type for OpenAI Decisions API.",),
                constraints_violated=("invalid_state_type",),
            )

        questions: dict[str, Any] = body["questions"]
        if len(questions) < 1 or len(questions) > 400:
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=(f"OpenAI Decisions API requires 1 to 400 questions; TypeSafe request has {len(questions)}.",),
                constraints_violated=("question_count_out_of_range",),
            )

        unsupported_reasons: list[str] = []
        partial_reasons: list[str] = []
        unsupported_ops: list[str] = []
        unsupported_params: list[str] = []
        constraints_violated: list[str] = []

        assets = body.get("assets")
        if assets is not None:
            if not isinstance(assets, dict) or len(assets) < 1:
                unsupported_reasons.append("Invalid 'assets' map.")
                constraints_violated.append("invalid_assets_map")
            else:
                for asset_key, asset_val in assets.items():
                    if not isinstance(asset_val, dict):
                        unsupported_reasons.append(f"Asset '{asset_key}' is not an object.")
                        continue
                    asset_type = asset_val.get("type")
                    if asset_type != "image":
                        unsupported_reasons.append(f"Asset '{asset_key}' has unsupported type '{asset_type}'.")
                        unsupported_ops.append(f"asset_type:{asset_type}")
                        continue
                    if asset_val.get("part"):
                        unsupported_reasons.append(
                            f"Asset '{asset_key}' uses multipart 'part' without inline base64 data."
                        )
                        unsupported_params.append(f"assets.{asset_key}.part")
                        continue
                    has_b64 = bool(asset_val.get("data_base64") and asset_val.get("format"))
                    has_url = bool(asset_val.get("image_url") or asset_val.get("url"))
                    if not has_b64 and not has_url:
                        unsupported_reasons.append(f"Asset '{asset_key}' lacks base64 image data.")
                        constraints_violated.append("missing_asset_data")

        for q_id, q in questions.items():
            if not isinstance(q, dict):
                return CapabilityReport(
                    provider=APIProvider.OPENAI_DECISIONS,
                    level=CompatibilityLevel.UNKNOWN,
                    can_execute=False,
                    reasons=(f"Question '{q_id}' is not an object.",),
                )
            q_type = q.get("type")
            if q_type == "box":
                unsupported_reasons.append(
                    f"Question '{q_id}' uses 'box' primitive, which has no equivalent in OpenAI Decisions API."
                )
                unsupported_ops.append("question_type:box")
                continue

            if q_type not in ("noul", "choice", "score"):
                unsupported_reasons.append(f"Question '{q_id}' has unsupported question type '{q_type}'.")
                unsupported_ops.append(f"question_type:{q_type}")
                continue

            instructions = q.get("instructions")
            if instructions is None or (isinstance(instructions, str) and instructions == ""):
                unsupported_reasons.append(
                    f"Question '{q_id}' omits 'instructions', which is required by OpenAI Decisions API."
                )
                constraints_violated.append(f"questions.{q_id}.missing_instructions")
            elif isinstance(instructions, (dict, list)):
                partial_reasons.append(
                    f"Question '{q_id}' uses structured (object/array) 'instructions', "
                    "which OpenAI Decisions API only accepts as text/image parts."
                )
                unsupported_params.append(f"questions.{q_id}.instructions")

            if q_type == "noul":
                if q.get("criteria") is not None:
                    partial_reasons.append(
                        f"Noul question '{q_id}' specifies 'criteria' (true/false definitions), "
                        "which cannot be represented in OpenAI 'predicate' without discarding or fabricating prompts."
                    )
                    unsupported_params.append(f"questions.{q_id}.criteria")

            elif q_type == "choice":
                criteria = q.get("criteria")
                if not isinstance(criteria, dict) or len(criteria) < 1:
                    unsupported_reasons.append(f"Choice question '{q_id}' must have at least 1 option in 'criteria'.")
                    constraints_violated.append(f"questions.{q_id}.empty_criteria")
                else:
                    for opt_label, opt_desc in criteria.items():
                        if opt_desc is not None and not isinstance(opt_desc, str):
                            partial_reasons.append(
                                f"Choice question '{q_id}' option '{opt_label}' has a structured description, "
                                "which OpenAI 'choices[].description' does not support."
                            )
                            unsupported_params.append(f"questions.{q_id}.criteria.{opt_label}")

            elif q_type == "score":
                criteria = q.get("criteria")
                if not isinstance(criteria, list) or len(criteria) < 2 or len(criteria) > 10:
                    unsupported_reasons.append(
                        f"Score question '{q_id}' has {len(criteria) if isinstance(criteria, list) else 0} levels; "
                        "OpenAI Decisions API requires 2 to 10 levels."
                    )
                    constraints_violated.append("score_levels_count_out_of_range")
                else:
                    for l_idx, item in enumerate(criteria):
                        if isinstance(item, str):
                            continue
                        if isinstance(item, dict) and isinstance(item.get("label"), str) and set(item.keys()) <= {"label", "description"}:
                            continue
                        partial_reasons.append(
                            f"Score question '{q_id}' criteria[{l_idx}] is a custom structured object/array "
                            "that cannot be losslessly mapped to OpenAI 'levels[i].label/description'."
                        )
                        unsupported_params.append(f"questions.{q_id}.criteria[{l_idx}]")

        if unsupported_reasons or unsupported_ops or constraints_violated:
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=tuple(unsupported_reasons + partial_reasons),
                unsupported_operations=tuple(unsupported_ops),
                unsupported_parameters=tuple(unsupported_params),
                constraints_violated=tuple(constraints_violated),
            )

        if partial_reasons or unsupported_params:
            return CapabilityReport(
                provider=APIProvider.OPENAI_DECISIONS,
                level=CompatibilityLevel.PARTIAL,
                can_execute=False,
                reasons=tuple(partial_reasons),
                unsupported_parameters=tuple(unsupported_params),
            )

        return CapabilityReport(
            provider=APIProvider.OPENAI_DECISIONS,
            level=CompatibilityLevel.FULL,
            can_execute=True,
            reasons=("TypeSafe request can be translated losslessly to OpenAI Decisions API.",),
        )

    # -------------------------------------------------------------------------
    # Cross-API Translation Capability: OpenAI Decisions -> TypeSafe
    # -------------------------------------------------------------------------

    def _evaluate_openai_to_typesafe_translation(self, classified: ClassifiedRequest) -> CapabilityReport:
        method = classified.raw_request.method.upper()
        path = classified.path.rstrip("/") or "/"

        if method != "POST" or path not in ("/v1/decisions", "/"):
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=(f"Endpoint {method} {path} cannot be translated to TypeSafe API.",),
                unsupported_operations=(f"{method} {path}",),
            )

        if classified.json_parse_error is not None or not isinstance(classified.parsed_json, dict):
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNKNOWN,
                can_execute=False,
                reasons=("Malformed JSON cannot be translated.",),
            )

        body: dict[str, Any] = classified.parsed_json
        if "input" not in body or not isinstance(body.get("questions"), list):
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNKNOWN,
                can_execute=False,
                reasons=("Request does not match OpenAI Decisions schema.",),
            )

        unsupported_reasons: list[str] = []
        partial_reasons: list[str] = []
        unsupported_ops: list[str] = []
        unsupported_params: list[str] = []
        constraints_violated: list[str] = []

        shared_input = body["input"]
        if shared_input is None or isinstance(shared_input, (bool, int, float)):
            unsupported_reasons.append("Invalid 'input' type for TypeSafe 'state'.")
            constraints_violated.append("invalid_input_type")
        elif isinstance(shared_input, list):
            if len(shared_input) < 1:
                unsupported_reasons.append("Input array cannot be empty.")
                constraints_violated.append("empty_input_array")
            self._check_openai_input_for_typesafe(
                shared_input,
                unsupported_reasons=unsupported_reasons,
                partial_reasons=partial_reasons,
                unsupported_ops=unsupported_ops,
                unsupported_params=unsupported_params,
                constraints_violated=constraints_violated,
            )

        questions: list[Any] = body["questions"]
        if len(questions) < 1:
            unsupported_reasons.append("At least 1 question is required.")
            constraints_violated.append("empty_questions")

        seen_names: set[str] = set()
        for idx, q in enumerate(questions):
            if not isinstance(q, dict):
                return CapabilityReport(
                    provider=APIProvider.TYPESAFE,
                    level=CompatibilityLevel.UNKNOWN,
                    can_execute=False,
                    reasons=(f"Question at index {idx} is not an object.",),
                )
            q_name = q.get("name")
            if not isinstance(q_name, str) or not q_name:
                unsupported_reasons.append(f"Question at index {idx} is missing a valid string 'name'.")
                constraints_violated.append(f"questions[{idx}].missing_name")
            elif q_name in seen_names:
                unsupported_reasons.append(
                    f"Duplicate question name '{q_name}' cannot be represented in TypeSafe 'questions' map without overwriting."
                )
                constraints_violated.append("duplicate_question_name")
            else:
                seen_names.add(q_name)

            q_type = q.get("type")
            if q_type not in ("predicate", "choice", "score"):
                unsupported_reasons.append(f"Question '{q_name or idx}' has unsupported type '{q_type}'.")
                unsupported_ops.append(f"question_type:{q_type}")
                continue

            instructions = q.get("instructions")
            if isinstance(instructions, list):
                for p_idx, part in enumerate(instructions):
                    if not isinstance(part, dict):
                        continue
                    if "prompt_cache_breakpoint" in part:
                        partial_reasons.append(
                            f"Question '{q_name or idx}' instructions[{p_idx}] contains 'prompt_cache_breakpoint'."
                        )
                        unsupported_params.append(
                            f"questions[{idx}].instructions[{p_idx}].prompt_cache_breakpoint"
                        )
                    p_type = part.get("type")
                    if p_type == "input_image":
                        self._check_openai_image_part_for_typesafe(
                            part,
                            location=f"questions[{idx}].instructions[{p_idx}]",
                            unsupported_reasons=unsupported_reasons,
                            unsupported_params=unsupported_params,
                            constraints_violated=constraints_violated,
                        )
                    elif p_type in ("input_audio", "audio", "input_file", "file"):
                        unsupported_reasons.append(
                            f"Question '{q_name or idx}' instructions[{p_idx}] uses unsupported part '{p_type}'."
                        )
                        unsupported_ops.append(f"content_part:{p_type}")

            if q_type == "choice":
                choices = q.get("choices")
                if not isinstance(choices, list) or len(choices) < 1 or len(choices) > 255:
                    unsupported_reasons.append(
                        f"Choice question '{q_name or idx}' has {len(choices) if isinstance(choices, list) else 0} choices; "
                        "TypeSafe supports 1 to 255 choices."
                    )
                    constraints_violated.append("choice_criteria_count_out_of_range")
                else:
                    seen_vals: set[str] = set()
                    for c_idx, c in enumerate(choices):
                        if not isinstance(c, dict) or not isinstance(c.get("value"), str):
                            unsupported_reasons.append(
                                f"Choice question '{q_name or idx}' choices[{c_idx}] is missing a string 'value'."
                            )
                            constraints_violated.append(f"questions[{idx}].choices[{c_idx}].invalid_value")
                            continue
                        if c["value"] in seen_vals:
                            unsupported_reasons.append(
                                f"Choice question '{q_name or idx}' has duplicate value '{c['value']}'."
                            )
                            constraints_violated.append("duplicate_choice_value")
                        seen_vals.add(c["value"])

            elif q_type == "score":
                levels = q.get("levels")
                if not isinstance(levels, list) or len(levels) < 1 or len(levels) > 10:
                    unsupported_reasons.append(
                        f"Score question '{q_name or idx}' has {len(levels) if isinstance(levels, list) else 0} levels; "
                        "TypeSafe supports 1 to 10 levels."
                    )
                    constraints_violated.append("score_criteria_count_out_of_range")
                else:
                    for l_idx, lvl in enumerate(levels):
                        if not isinstance(lvl, dict) or not isinstance(lvl.get("label"), str):
                            unsupported_reasons.append(
                                f"Score question '{q_name or idx}' levels[{l_idx}] is missing a string 'label'."
                            )
                            constraints_violated.append(f"questions[{idx}].levels[{l_idx}].invalid_label")

        if unsupported_reasons or unsupported_ops or constraints_violated:
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.UNSUPPORTED,
                can_execute=False,
                reasons=tuple(unsupported_reasons + partial_reasons),
                unsupported_operations=tuple(unsupported_ops),
                unsupported_parameters=tuple(unsupported_params),
                constraints_violated=tuple(constraints_violated),
            )

        if partial_reasons or unsupported_params:
            return CapabilityReport(
                provider=APIProvider.TYPESAFE,
                level=CompatibilityLevel.PARTIAL,
                can_execute=False,
                reasons=tuple(partial_reasons),
                unsupported_parameters=tuple(unsupported_params),
            )

        return CapabilityReport(
            provider=APIProvider.TYPESAFE,
            level=CompatibilityLevel.FULL,
            can_execute=True,
            reasons=("OpenAI Decisions request can be translated losslessly to TypeSafe API.",),
        )

    def _check_openai_input_for_typesafe(
        self,
        items: list[Any],
        *,
        unsupported_reasons: list[str],
        partial_reasons: list[str],
        unsupported_ops: list[str],
        unsupported_params: list[str],
        constraints_violated: list[str],
    ) -> None:
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in ("input_audio", "audio", "input_file", "file", "item_reference"):
                unsupported_reasons.append(f"Input[{idx}] uses unsupported item type '{item_type}'.")
                unsupported_ops.append(f"input_item:{item_type}")
                continue
            if "file_id" in item:
                unsupported_reasons.append(f"Input[{idx}] uses unsupported 'file_id'.")
                unsupported_params.append(f"input[{idx}].file_id")
                continue

            content_list = item.get("content") if "content" in item else item.get("output")
            if isinstance(content_list, list):
                for c_idx, part in enumerate(content_list):
                    if not isinstance(part, dict):
                        continue
                    p_type = part.get("type")
                    if p_type in ("input_audio", "audio", "input_file", "file", "item_reference"):
                        unsupported_reasons.append(
                            f"Input[{idx}].content[{c_idx}] uses unsupported part type '{p_type}'."
                        )
                        unsupported_ops.append(f"content_part:{p_type}")
                    if "file_id" in part:
                        unsupported_reasons.append(f"Input[{idx}].content[{c_idx}] uses unsupported 'file_id'.")
                        unsupported_params.append(f"input[{idx}].content[{c_idx}].file_id")
                    if "prompt_cache_breakpoint" in part:
                        partial_reasons.append(
                            f"Input[{idx}].content[{c_idx}] contains 'prompt_cache_breakpoint', "
                            "which has no equivalent in TypeSafe 'state'."
                        )
                        unsupported_params.append(f"input[{idx}].content[{c_idx}].prompt_cache_breakpoint")
                    if p_type == "input_image":
                        self._check_openai_image_part_for_typesafe(
                            part,
                            location=f"input[{idx}].content[{c_idx}]",
                            unsupported_reasons=unsupported_reasons,
                            unsupported_params=unsupported_params,
                            constraints_violated=constraints_violated,
                        )

    def _check_openai_image_part_for_typesafe(
        self,
        part: dict[str, Any],
        *,
        location: str,
        unsupported_reasons: list[str],
        unsupported_params: list[str],
        constraints_violated: list[str],
    ) -> None:
        if "file_id" in part:
            unsupported_reasons.append(f"{location} uses 'file_id', which is unsupported.")
            unsupported_params.append(f"{location}.file_id")
            return
        image_url = part.get("image_url")
        if not isinstance(image_url, str) or not image_url:
            unsupported_reasons.append(f"{location} is missing a valid 'image_url'.")
            constraints_violated.append(f"{location}.missing_image_url")
            return
        if image_url.startswith("http://") or image_url.startswith("https://"):
            unsupported_reasons.append(
                f"{location} uses a remote HTTP(S) image URL; TypeSafe 'assets' only accepts base64 or multipart image data."
            )
            unsupported_params.append(f"{location}.image_url")
            constraints_violated.append("remote_image_url_unsupported_on_typesafe")
            return
        match = DATA_URL_RE.match(image_url)
        if not match:
            unsupported_reasons.append(f"{location} has an unrecognized image_url format.")
            constraints_violated.append("invalid_data_url")
            return
        fmt = match.group(1).lower()
        if fmt not in TYPESAFE_IMAGE_FORMATS:
            unsupported_reasons.append(
                f"{location} uses image format '{fmt}'; TypeSafe only supports png, jpg, and jpeg."
            )
            constraints_violated.append("unsupported_image_format")

    @staticmethod
    def _has_typesafe_images_header(classified: ClassifiedRequest) -> bool:
        val = classified.raw_request.get_header("TypeSafe-Danger-Zone-Images")
        return val is not None and val.strip().lower() == "true"
