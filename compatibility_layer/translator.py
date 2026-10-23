"""Translation Middleware: translates prompt inputs (state/assets/questions <-> input/questions) and response answers.

All non-prompt parameters (such as `model`, caching fields, `usage`, etc.) are passed through unmodified.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlsplit

from .models import (
    APIProvider,
    DATA_URL_RE,
    RawHTTPRequest,
    RawHTTPResponse,
    RequestTranslationError,
    ResponseTranslationError,
    TYPESAFE_IMAGE_FORMATS,
)


@dataclass(frozen=True)
class QuestionTranslationMeta:
    """Stores original question metadata needed to reconstruct response fields losslessly."""

    name: str
    original_type: str
    choice_order: tuple[str, ...] = ()
    score_criteria: tuple[Any, ...] = ()
    score_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationContext:
    """Captures question order and rubric legends across the request/response lifecycle."""

    source_provider: APIProvider
    target_provider: APIProvider
    questions_meta: Mapping[str, QuestionTranslationMeta] = field(default_factory=dict)
    question_order: tuple[str, ...] = ()


class TranslationMiddleware:
    """Bidirectional translator between TypeSafe System One and OpenAI Decisions API.

    Translates only prompt inputs (`state`/`assets`/`questions` <-> `input`/`questions`) and
    response `answers`. Passes all other request/response parameters (`model`, `usage`, etc.)
    through unmodified. Never silently discards unsupported question fields or fabricates answers.
    """

    # -------------------------------------------------------------------------
    # Direction 1: TypeSafe -> OpenAI Decisions
    # -------------------------------------------------------------------------

    def translate_typesafe_to_openai_request(
        self,
        request: RawHTTPRequest,
    ) -> tuple[RawHTTPRequest, TranslationContext]:
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception as exc:
            raise RequestTranslationError(
                f"Failed to parse TypeSafe request JSON: {exc}",
                cause=exc,
            ) from exc

        if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict):
            raise RequestTranslationError(
                "TypeSafe request must be a JSON object containing 'state' and a 'questions' map."
            )

        state = body["state"]
        assets = body.get("assets")
        questions_map: dict[str, Any] = body["questions"]

        if len(questions_map) < 1 or len(questions_map) > 400:
            raise RequestTranslationError(
                f"Cannot translate {len(questions_map)} questions to OpenAI Decisions API (supports 1..400)."
            )

        asset_data_urls = self._extract_typesafe_assets_as_data_urls(assets) if assets is not None else {}
        question_referenced_assets: set[str] = set()
        for q_val in questions_map.values():
            if isinstance(q_val, dict) and isinstance(q_val.get("instructions"), str):
                for ph in asset_data_urls:
                    if ph in q_val["instructions"]:
                        question_referenced_assets.add(ph)

        openai_input = self._translate_state_to_openai_input(
            state,
            asset_data_urls,
            question_referenced_assets=question_referenced_assets,
        )

        openai_questions: list[dict[str, Any]] = []
        questions_meta: dict[str, QuestionTranslationMeta] = {}
        question_order: list[str] = []

        for q_id, q in questions_map.items():
            if not isinstance(q, dict):
                raise RequestTranslationError(f"Question '{q_id}' must be a JSON object.")

            q_type = q.get("type")
            if q_type == "box":
                raise RequestTranslationError(
                    f"Question '{q_id}' uses 'box' primitive, which is unsupported by OpenAI Decisions API.",
                    unsupported_operations=("question_type:box",),
                )
            if q_type not in ("noul", "choice", "score"):
                raise RequestTranslationError(
                    f"Question '{q_id}' has unsupported type '{q_type}'.",
                    unsupported_operations=(f"question_type:{q_type}",),
                )

            raw_instructions = q.get("instructions")
            if raw_instructions is None or (isinstance(raw_instructions, str) and raw_instructions == ""):
                raise RequestTranslationError(
                    f"Question '{q_id}' is missing 'instructions', which is required by OpenAI Decisions API.",
                    unsupported_parameters=(f"questions.{q_id}.instructions",),
                )
            if isinstance(raw_instructions, (dict, list)):
                raise RequestTranslationError(
                    f"Question '{q_id}' has structured 'instructions' that cannot be represented in OpenAI without fabrication.",
                    unsupported_parameters=(f"questions.{q_id}.instructions",),
                )

            translated_instructions = self._inject_assets_into_instructions(raw_instructions, asset_data_urls)

            # Pass through any extra question-level parameters unmodified
            extra_q_params = {
                k: v for k, v in q.items() if k not in ("type", "instructions", "criteria")
            }

            if q_type == "noul":
                if q.get("criteria") is not None:
                    raise RequestTranslationError(
                        f"Noul question '{q_id}' includes 'criteria', which OpenAI 'predicate' does not support.",
                        unsupported_parameters=(f"questions.{q_id}.criteria",),
                    )
                openai_q: dict[str, Any] = {
                    **extra_q_params,
                    "type": "predicate",
                    "name": q_id,
                    "instructions": translated_instructions,
                }
                openai_questions.append(openai_q)
                questions_meta[q_id] = QuestionTranslationMeta(name=q_id, original_type="noul")
                question_order.append(q_id)

            elif q_type == "choice":
                criteria = q.get("criteria")
                if not isinstance(criteria, dict) or len(criteria) < 1:
                    raise RequestTranslationError(
                        f"Choice question '{q_id}' requires a non-empty 'criteria' map."
                    )
                choices_list: list[dict[str, Any]] = []
                choice_keys: list[str] = []
                for label, desc in criteria.items():
                    if desc is not None and not isinstance(desc, str):
                        raise RequestTranslationError(
                            f"Choice question '{q_id}' option '{label}' has a structured description unsupported by OpenAI.",
                            unsupported_parameters=(f"questions.{q_id}.criteria.{label}",),
                        )
                    choice_entry: dict[str, Any] = {"value": label}
                    if desc is not None:
                        choice_entry["description"] = desc
                    choices_list.append(choice_entry)
                    choice_keys.append(label)

                openai_q = {
                    **extra_q_params,
                    "type": "choice",
                    "name": q_id,
                    "instructions": translated_instructions,
                    "choices": choices_list,
                }
                openai_questions.append(openai_q)
                questions_meta[q_id] = QuestionTranslationMeta(
                    name=q_id,
                    original_type="choice",
                    choice_order=tuple(choice_keys),
                )
                question_order.append(q_id)

            elif q_type == "score":
                criteria = q.get("criteria")
                if not isinstance(criteria, list) or len(criteria) < 2 or len(criteria) > 10:
                    raise RequestTranslationError(
                        f"Score question '{q_id}' requires 2..10 levels for OpenAI Decisions API."
                    )
                levels_list: list[dict[str, Any]] = []
                for l_idx, item in enumerate(criteria):
                    if isinstance(item, str):
                        levels_list.append({"label": item})
                    elif (
                        isinstance(item, dict)
                        and isinstance(item.get("label"), str)
                        and set(item.keys()) <= {"label", "description"}
                    ):
                        lvl_entry: dict[str, Any] = {"label": item["label"]}
                        if item.get("description") is not None:
                            if not isinstance(item["description"], str):
                                raise RequestTranslationError(
                                    f"Score question '{q_id}' criteria[{l_idx}].description must be a string.",
                                    unsupported_parameters=(f"questions.{q_id}.criteria[{l_idx}]",),
                                )
                            lvl_entry["description"] = item["description"]
                        levels_list.append(lvl_entry)
                    else:
                        raise RequestTranslationError(
                            f"Score question '{q_id}' criteria[{l_idx}] cannot be represented in OpenAI 'levels'.",
                            unsupported_parameters=(f"questions.{q_id}.criteria[{l_idx}]",),
                        )

                openai_q = {
                    **extra_q_params,
                    "type": "score",
                    "name": q_id,
                    "instructions": translated_instructions,
                    "levels": levels_list,
                }
                openai_questions.append(openai_q)
                questions_meta[q_id] = QuestionTranslationMeta(
                    name=q_id,
                    original_type="score",
                    score_criteria=tuple(criteria),
                )
                question_order.append(q_id)

        # Build translated payload preserving all other top-level parameters (e.g. model) unmodified
        translated_body: dict[str, Any] = {}
        for k, v in body.items():
            if k in ("state", "assets", "questions"):
                continue
            translated_body[k] = v
        translated_body["input"] = openai_input
        translated_body["questions"] = openai_questions

        # Preserve headers (dropping only TypeSafe-Danger-Zone-Images if it was used for assets)
        translated_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("content-length", "typesafe-danger-zone-images")
        }
        translated_headers["Content-Type"] = "application/json"

        query = urlsplit(request.url).query
        target_url = f"/v1/decisions?{query}" if query else "/v1/decisions"

        translated_request = RawHTTPRequest(
            method=request.method,
            url=target_url,
            headers=translated_headers,
            body=json.dumps(translated_body).encode("utf-8"),
            destination_hint=APIProvider.OPENAI_DECISIONS,
        )
        context = TranslationContext(
            source_provider=APIProvider.TYPESAFE,
            target_provider=APIProvider.OPENAI_DECISIONS,
            questions_meta=questions_meta,
            question_order=tuple(question_order),
        )
        return translated_request, context

    def translate_openai_to_typesafe_response(
        self,
        response: RawHTTPResponse,
        context: TranslationContext,
    ) -> RawHTTPResponse:
        try:
            body = json.loads(response.body.decode("utf-8"))
        except Exception as exc:
            raise ResponseTranslationError(
                f"Failed to parse upstream OpenAI response JSON: {exc}",
                raw_upstream_response=response,
                cause=exc,
            ) from exc

        if not isinstance(body, dict) or not isinstance(body.get("answers"), list):
            raise ResponseTranslationError(
                "Upstream OpenAI response does not contain an 'answers' list.",
                raw_upstream_response=response,
            )

        typesafe_answers: dict[str, Any] = {}
        for ans in body["answers"]:
            if not isinstance(ans, dict):
                raise ResponseTranslationError(
                    "Invalid answer object in OpenAI response.",
                    raw_upstream_response=response,
                )
            ans_type = ans.get("type")
            ans_name = ans.get("name")
            if not isinstance(ans_name, str) or not ans_name:
                raise ResponseTranslationError(
                    "Answer in OpenAI response is missing 'name'.",
                    raw_upstream_response=response,
                )

            if ans_type == "refusal":
                raise ResponseTranslationError(
                    f"OpenAI returned a 'refusal' for question '{ans_name}', which has no representation in TypeSafe SystemOneResponse.",
                    raw_upstream_response=response,
                )

            meta = context.questions_meta.get(ans_name)

            if ans_type == "predicate":
                if "probability" not in ans or not isinstance(ans["probability"], (int, float)):
                    raise ResponseTranslationError(
                        f"Predicate answer '{ans_name}' is missing numeric 'probability'.",
                        raw_upstream_response=response,
                    )
                extra_ans = {k: v for k, v in ans.items() if k not in ("type", "name", "probability")}
                typesafe_answers[ans_name] = {
                    **extra_ans,
                    "type": "noul",
                    "noul": ans["probability"],
                }

            elif ans_type == "choice":
                probs_list = ans.get("probabilities")
                if not isinstance(probs_list, list) or "choice" not in ans or "confidence" not in ans:
                    raise ResponseTranslationError(
                        f"Choice answer '{ans_name}' is missing required fields.",
                        raw_upstream_response=response,
                    )
                probs_map: dict[str, float] = {}
                for item in probs_list:
                    if not isinstance(item, dict) or "value" not in item or "probability" not in item:
                        raise ResponseTranslationError(
                            f"Choice answer '{ans_name}' has malformed probability entry.",
                            raw_upstream_response=response,
                        )
                    probs_map[str(item["value"])] = item["probability"]

                extra_ans = {
                    k: v
                    for k, v in ans.items()
                    if k not in ("type", "name", "choice", "confidence", "probabilities")
                }
                typesafe_answers[ans_name] = {
                    **extra_ans,
                    "type": "choice",
                    "choice": ans["choice"],
                    "confidence": ans["confidence"],
                    "probabilities": probs_map,
                }

            elif ans_type == "score":
                probs_list = ans.get("probabilities")
                if not isinstance(probs_list, list) or "score" not in ans or "confidence" not in ans:
                    raise ResponseTranslationError(
                        f"Score answer '{ans_name}' is missing required fields.",
                        raw_upstream_response=response,
                    )
                probs_map_score: dict[str, float] = {}
                legend_map: dict[str, Any] = {}
                for item in probs_list:
                    if not isinstance(item, dict) or "value" not in item or "probability" not in item:
                        raise ResponseTranslationError(
                            f"Score answer '{ans_name}' has malformed probability entry.",
                            raw_upstream_response=response,
                        )
                    idx_int = int(item["value"])
                    idx_str = str(idx_int)
                    probs_map_score[idx_str] = item["probability"]
                    if meta and idx_int < len(meta.score_criteria):
                        legend_map[idx_str] = meta.score_criteria[idx_int]
                    elif "label" in item:
                        legend_map[idx_str] = item["label"]

                extra_ans = {
                    k: v
                    for k, v in ans.items()
                    if k not in ("type", "name", "score", "confidence", "probabilities")
                }
                typesafe_answers[ans_name] = {
                    **extra_ans,
                    "type": "score",
                    "score": ans["score"],
                    "confidence": ans["confidence"],
                    "legend": legend_map,
                    "probabilities": probs_map_score,
                }

            else:
                raise ResponseTranslationError(
                    f"Unsupported OpenAI answer type '{ans_type}' for question '{ans_name}'.",
                    raw_upstream_response=response,
                )

        # Pass through all non-answers top-level response fields (model, usage, etc.) unmodified
        translated_body: dict[str, Any] = {}
        for k, v in body.items():
            if k == "answers":
                continue
            translated_body[k] = v
        translated_body["answers"] = typesafe_answers

        response_headers = {
            k: v for k, v in response.headers.items() if k.lower() != "content-length"
        }
        response_headers["Content-Type"] = "application/json"

        return RawHTTPResponse(
            status_code=response.status_code,
            headers=response_headers,
            body=json.dumps(translated_body).encode("utf-8"),
            from_upstream=True,
            upstream_provider=response.upstream_provider,
        )

    # -------------------------------------------------------------------------
    # Direction 2: OpenAI Decisions -> TypeSafe
    # -------------------------------------------------------------------------

    def translate_openai_to_typesafe_request(
        self,
        request: RawHTTPRequest,
    ) -> tuple[RawHTTPRequest, TranslationContext]:
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception as exc:
            raise RequestTranslationError(
                f"Failed to parse OpenAI Decisions request JSON: {exc}",
                cause=exc,
            ) from exc

        if not isinstance(body, dict) or "input" not in body or not isinstance(body.get("questions"), list):
            raise RequestTranslationError(
                "OpenAI Decisions request must be a JSON object containing 'input' and a 'questions' list."
            )

        shared_input = body["input"]
        questions_list: list[Any] = body["questions"]
        if len(questions_list) < 1:
            raise RequestTranslationError("OpenAI Decisions request must contain at least 1 question.")

        assets: dict[str, dict[str, Any]] = {}
        asset_counter = 0

        def register_data_url_image(image_url: str, location: str) -> str:
            nonlocal asset_counter
            if image_url.startswith("http://") or image_url.startswith("https://"):
                raise RequestTranslationError(
                    f"Remote image URL at {location} cannot be translated to TypeSafe 'assets' (base64 required).",
                    unsupported_parameters=(f"{location}.image_url",),
                )
            match = DATA_URL_RE.match(image_url)
            if not match:
                raise RequestTranslationError(
                    f"Invalid image data URL at {location}.",
                    unsupported_parameters=(f"{location}.image_url",),
                )
            fmt = match.group(1).lower()
            if fmt not in TYPESAFE_IMAGE_FORMATS:
                raise RequestTranslationError(
                    f"Image format '{fmt}' at {location} is unsupported by TypeSafe (supports png, jpg, jpeg).",
                    unsupported_parameters=(f"{location}.image_url",),
                )
            b64_data = re.sub(r"\s+", "", match.group(2))
            placeholder = f"{{{{asset_image_{asset_counter}}}}}"
            asset_counter += 1
            assets[placeholder] = {
                "type": "image",
                "format": fmt,
                "data_base64": b64_data,
            }
            return placeholder

        typesafe_state = self._translate_openai_input_to_state(shared_input, register_data_url_image)

        typesafe_questions: dict[str, Any] = {}
        questions_meta: dict[str, QuestionTranslationMeta] = {}
        question_order: list[str] = []

        for idx, q in enumerate(questions_list):
            if not isinstance(q, dict):
                raise RequestTranslationError(f"Question at index {idx} must be a JSON object.")

            q_name = q.get("name")
            if not isinstance(q_name, str) or not q_name:
                raise RequestTranslationError(f"Question at index {idx} is missing a string 'name'.")
            if q_name in typesafe_questions:
                raise RequestTranslationError(
                    f"Duplicate question name '{q_name}' cannot be mapped to TypeSafe 'questions' object."
                )

            q_type = q.get("type")
            if q_type not in ("predicate", "choice", "score"):
                raise RequestTranslationError(
                    f"Question '{q_name}' has unsupported type '{q_type}'.",
                    unsupported_operations=(f"question_type:{q_type}",),
                )

            raw_instructions = q.get("instructions")
            translated_instructions = self._translate_openai_instructions_to_typesafe(
                raw_instructions,
                location=f"questions[{idx}].instructions",
                register_image=register_data_url_image,
            )

            extra_q_params = {
                k: v
                for k, v in q.items()
                if k not in ("type", "name", "instructions", "choices", "levels")
            }

            if q_type == "predicate":
                ts_q: dict[str, Any] = {
                    **extra_q_params,
                    "type": "noul",
                }
                if translated_instructions is not None:
                    ts_q["instructions"] = translated_instructions
                typesafe_questions[q_name] = ts_q
                questions_meta[q_name] = QuestionTranslationMeta(
                    name=q_name,
                    original_type="predicate",
                )
                question_order.append(q_name)

            elif q_type == "choice":
                choices = q.get("choices")
                if not isinstance(choices, list) or len(choices) < 1 or len(choices) > 255:
                    raise RequestTranslationError(
                        f"Choice question '{q_name}' has {len(choices) if isinstance(choices, list) else 0} choices; "
                        "TypeSafe supports 1..255 choices."
                    )
                criteria_map: dict[str, Any] = {}
                choice_keys: list[str] = []
                for c_idx, c in enumerate(choices):
                    if not isinstance(c, dict) or not isinstance(c.get("value"), str):
                        raise RequestTranslationError(
                            f"Choice question '{q_name}' choices[{c_idx}] requires a string 'value'."
                        )
                    val = c["value"]
                    if val in criteria_map:
                        raise RequestTranslationError(
                            f"Choice question '{q_name}' has duplicate choice value '{val}'."
                        )
                    criteria_map[val] = c.get("description")
                    choice_keys.append(val)

                ts_q = {
                    **extra_q_params,
                    "type": "choice",
                    "criteria": criteria_map,
                }
                if translated_instructions is not None:
                    ts_q["instructions"] = translated_instructions
                typesafe_questions[q_name] = ts_q
                questions_meta[q_name] = QuestionTranslationMeta(
                    name=q_name,
                    original_type="choice",
                    choice_order=tuple(choice_keys),
                )
                question_order.append(q_name)

            elif q_type == "score":
                levels = q.get("levels")
                if not isinstance(levels, list) or len(levels) < 1 or len(levels) > 10:
                    raise RequestTranslationError(
                        f"Score question '{q_name}' requires 1..10 levels for TypeSafe."
                    )
                criteria_list: list[Any] = []
                labels_list: list[str] = []
                for l_idx, lvl in enumerate(levels):
                    if not isinstance(lvl, dict) or not isinstance(lvl.get("label"), str):
                        raise RequestTranslationError(
                            f"Score question '{q_name}' levels[{l_idx}] requires a string 'label'."
                        )
                    label = lvl["label"]
                    labels_list.append(label)
                    if "description" in lvl and lvl["description"] is not None:
                        criteria_list.append({"label": label, "description": lvl["description"]})
                    else:
                        criteria_list.append(label)

                ts_q = {
                    **extra_q_params,
                    "type": "score",
                    "criteria": criteria_list,
                }
                if translated_instructions is not None:
                    ts_q["instructions"] = translated_instructions
                typesafe_questions[q_name] = ts_q
                questions_meta[q_name] = QuestionTranslationMeta(
                    name=q_name,
                    original_type="score",
                    score_criteria=tuple(criteria_list),
                    score_labels=tuple(labels_list),
                )
                question_order.append(q_name)

        # Pass through all non-prompt parameters (model, prompt_cache_key, etc.) unmodified
        translated_body: dict[str, Any] = {}
        for k, v in body.items():
            if k in ("input", "questions"):
                continue
            translated_body[k] = v
        translated_body["state"] = typesafe_state
        translated_body["questions"] = typesafe_questions
        if assets:
            translated_body["assets"] = assets

        translated_headers = {
            k: v for k, v in request.headers.items() if k.lower() != "content-length"
        }
        translated_headers["Content-Type"] = "application/json"
        if assets:
            translated_headers["TypeSafe-Danger-Zone-Images"] = "true"

        query = urlsplit(request.url).query
        target_url = f"/v1/systemone?{query}" if query else "/v1/systemone"

        translated_request = RawHTTPRequest(
            method=request.method,
            url=target_url,
            headers=translated_headers,
            body=json.dumps(translated_body).encode("utf-8"),
            destination_hint=APIProvider.TYPESAFE,
        )
        context = TranslationContext(
            source_provider=APIProvider.OPENAI_DECISIONS,
            target_provider=APIProvider.TYPESAFE,
            questions_meta=questions_meta,
            question_order=tuple(question_order),
        )
        return translated_request, context

    def translate_typesafe_to_openai_response(
        self,
        response: RawHTTPResponse,
        context: TranslationContext,
    ) -> RawHTTPResponse:
        try:
            body = json.loads(response.body.decode("utf-8"))
        except Exception as exc:
            raise ResponseTranslationError(
                f"Failed to parse upstream TypeSafe response JSON: {exc}",
                raw_upstream_response=response,
                cause=exc,
            ) from exc

        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise ResponseTranslationError(
                "Upstream TypeSafe response does not contain an 'answers' object.",
                raw_upstream_response=response,
            )

        ts_answers: dict[str, Any] = body["answers"]
        openai_answers: list[dict[str, Any]] = []

        ordered_names = list(context.question_order) if context.question_order else list(ts_answers.keys())
        for q_name in ordered_names:
            if q_name not in ts_answers:
                raise ResponseTranslationError(
                    f"Upstream TypeSafe response is missing answer for question '{q_name}'.",
                    raw_upstream_response=response,
                )
            ans = ts_answers[q_name]
            if not isinstance(ans, dict):
                raise ResponseTranslationError(
                    f"Invalid answer object for question '{q_name}' in TypeSafe response.",
                    raw_upstream_response=response,
                )

            ans_type = ans.get("type")
            meta = context.questions_meta.get(q_name)

            if ans_type == "noul":
                if "noul" not in ans or not isinstance(ans["noul"], (int, float)):
                    raise ResponseTranslationError(
                        f"Noul answer '{q_name}' is missing numeric 'noul' field.",
                        raw_upstream_response=response,
                    )
                extra_ans = {k: v for k, v in ans.items() if k not in ("type", "noul")}
                openai_answers.append(
                    {
                        **extra_ans,
                        "type": "predicate",
                        "name": q_name,
                        "probability": ans["noul"],
                    }
                )

            elif ans_type == "choice":
                probs_map = ans.get("probabilities")
                if not isinstance(probs_map, dict) or "choice" not in ans or "confidence" not in ans:
                    raise ResponseTranslationError(
                        f"Choice answer '{q_name}' is missing required fields.",
                        raw_upstream_response=response,
                    )
                keys_order = (
                    list(meta.choice_order)
                    if meta and meta.choice_order
                    else list(probs_map.keys())
                )
                probs_list: list[dict[str, Any]] = []
                for k in keys_order:
                    if k not in probs_map:
                        raise ResponseTranslationError(
                            f"Choice answer '{q_name}' is missing probability for option '{k}'.",
                            raw_upstream_response=response,
                        )
                    probs_list.append({"value": k, "probability": probs_map[k]})

                extra_ans = {
                    k: v
                    for k, v in ans.items()
                    if k not in ("type", "choice", "confidence", "probabilities")
                }
                openai_answers.append(
                    {
                        **extra_ans,
                        "type": "choice",
                        "name": q_name,
                        "choice": ans["choice"],
                        "probabilities": probs_list,
                        "confidence": ans["confidence"],
                    }
                )

            elif ans_type == "score":
                probs_map = ans.get("probabilities")
                legend_map = ans.get("legend", {})
                if not isinstance(probs_map, dict) or "score" not in ans or "confidence" not in ans:
                    raise ResponseTranslationError(
                        f"Score answer '{q_name}' is missing required fields.",
                        raw_upstream_response=response,
                    )
                probs_list_score: list[dict[str, Any]] = []
                sorted_indices = sorted(probs_map.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
                for idx_key in sorted_indices:
                    idx_int = int(idx_key)
                    if meta and idx_int < len(meta.score_labels):
                        label = meta.score_labels[idx_int]
                    else:
                        legend_entry = legend_map.get(str(idx_key), str(idx_key))
                        label = (
                            legend_entry.get("label", str(legend_entry))
                            if isinstance(legend_entry, dict)
                            else str(legend_entry)
                        )
                    probs_list_score.append(
                        {
                            "value": idx_int,
                            "label": label,
                            "probability": probs_map[idx_key],
                        }
                    )

                extra_ans = {
                    k: v
                    for k, v in ans.items()
                    if k not in ("type", "score", "confidence", "legend", "probabilities")
                }
                openai_answers.append(
                    {
                        **extra_ans,
                        "type": "score",
                        "name": q_name,
                        "score": ans["score"],
                        "probabilities": probs_list_score,
                        "confidence": ans["confidence"],
                    }
                )

            else:
                raise ResponseTranslationError(
                    f"Unsupported TypeSafe answer type '{ans_type}' for question '{q_name}'.",
                    raw_upstream_response=response,
                )

        # Pass through all non-answers top-level fields (model, usage, etc.) unmodified
        translated_body: dict[str, Any] = {}
        for k, v in body.items():
            if k == "answers":
                continue
            translated_body[k] = v
        translated_body["answers"] = openai_answers

        response_headers = {
            k: v for k, v in response.headers.items() if k.lower() != "content-length"
        }
        response_headers["Content-Type"] = "application/json"

        return RawHTTPResponse(
            status_code=response.status_code,
            headers=response_headers,
            body=json.dumps(translated_body).encode("utf-8"),
            from_upstream=True,
            upstream_provider=response.upstream_provider,
        )

    # -------------------------------------------------------------------------
    # Internal helpers for prompt/input translation
    # -------------------------------------------------------------------------

    def _extract_typesafe_assets_as_data_urls(self, assets: Any) -> dict[str, str]:
        if not isinstance(assets, dict):
            raise RequestTranslationError("TypeSafe 'assets' must be a JSON object.")
        result: dict[str, str] = {}
        for key, asset in assets.items():
            if not isinstance(asset, dict) or asset.get("type") != "image":
                raise RequestTranslationError(
                    f"Asset '{key}' is not a valid image asset.",
                    unsupported_operations=(f"asset:{key}",),
                )
            if asset.get("part"):
                raise RequestTranslationError(
                    f"Asset '{key}' uses multipart 'part', which cannot be translated to OpenAI JSON.",
                    unsupported_parameters=(f"assets.{key}.part",),
                )
            fmt = (asset.get("format") or "").lower()
            b64 = asset.get("data_base64")
            if not b64 or fmt not in TYPESAFE_IMAGE_FORMATS:
                raise RequestTranslationError(
                    f"Asset '{key}' requires 'data_base64' and format in png/jpg/jpeg.",
                    unsupported_parameters=(f"assets.{key}",),
                )
            result[key] = f"data:image/{fmt};base64,{b64}"
        return result

    def _translate_state_to_openai_input(
        self,
        state: Any,
        asset_data_urls: Mapping[str, str],
        *,
        question_referenced_assets: set[str] | None = None,
    ) -> Any:
        q_refs = question_referenced_assets or set()
        state_only_assets = {k: v for k, v in asset_data_urls.items() if k not in q_refs}
        if isinstance(state, str):
            has_state_ph = any(ph in state for ph in asset_data_urls)
            if not has_state_ph and not state_only_assets:
                return state
            parts = self._expand_placeholders_to_parts(
                state,
                asset_data_urls,
                unreferenced_pool=state_only_assets,
            )
            return [{"role": "user", "content": parts}]

        if isinstance(state, (dict, list)):
            content_parts: list[dict[str, Any]] = [{"type": "input_json", "data": state}]
            for data_url in state_only_assets.values():
                content_parts.append({"type": "input_image", "image_url": data_url})
            return [{"role": "user", "content": content_parts}]

        raise RequestTranslationError("Unsupported TypeSafe 'state' value.")

    def _inject_assets_into_instructions(
        self,
        instructions: str,
        asset_data_urls: Mapping[str, str],
    ) -> Any:
        if not asset_data_urls:
            return instructions
        has_placeholder = any(ph in instructions for ph in asset_data_urls)
        if not has_placeholder:
            return instructions
        return self._expand_placeholders_to_parts(instructions, asset_data_urls, unreferenced_pool={})

    @staticmethod
    def _expand_placeholders_to_parts(
        text: str,
        asset_data_urls: Mapping[str, str],
        *,
        unreferenced_pool: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        remaining = text
        used_placeholders: set[str] = set()

        while remaining:
            earliest_idx = None
            earliest_ph = None
            for ph in asset_data_urls:
                idx = remaining.find(ph)
                if idx != -1 and (earliest_idx is None or idx < earliest_idx):
                    earliest_idx = idx
                    earliest_ph = ph

            if earliest_idx is None or earliest_ph is None:
                if remaining:
                    parts.append({"type": "input_text", "text": remaining})
                break

            prefix = remaining[:earliest_idx]
            if prefix:
                parts.append({"type": "input_text", "text": prefix})
            parts.append({"type": "input_image", "image_url": asset_data_urls[earliest_ph]})
            used_placeholders.add(earliest_ph)
            remaining = remaining[earliest_idx + len(earliest_ph) :]

        for ph, data_url in unreferenced_pool.items():
            if ph not in used_placeholders:
                parts.append({"type": "input_image", "image_url": data_url})

        return parts

    def _translate_openai_input_to_state(
        self,
        shared_input: Any,
        register_image: Any,
    ) -> Any:
        if isinstance(shared_input, (str, dict)):
            return shared_input

        if not isinstance(shared_input, list):
            raise RequestTranslationError("OpenAI 'input' must be a string, object, or list.")

        # Single user message unwrap when possible
        if len(shared_input) == 1 and isinstance(shared_input[0], dict) and shared_input[0].get("role") == "user":
            content = shared_input[0].get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
                part = content[0]
                if "prompt_cache_breakpoint" in part:
                    raise RequestTranslationError(
                        "Cannot translate 'prompt_cache_breakpoint' inside input part to TypeSafe without dropping it.",
                        unsupported_parameters=("input[0].content[0].prompt_cache_breakpoint",),
                    )
                if part.get("type") == "input_text" and isinstance(part.get("text"), str):
                    return part["text"]
                if part.get("type") == "input_json" and "data" in part:
                    return part["data"]

        translated_items: list[Any] = []
        for idx, item in enumerate(shared_input):
            if not isinstance(item, dict):
                translated_items.append(item)
                continue
            item_type = item.get("type")
            if item_type in ("input_audio", "audio", "input_file", "file", "item_reference") or "file_id" in item:
                raise RequestTranslationError(
                    f"Unsupported OpenAI input item at input[{idx}].",
                    unsupported_operations=(f"input_item:{item_type or 'file_id'}",),
                )
            item_copy = dict(item)
            for content_key in ("content", "output"):
                if isinstance(item_copy.get(content_key), list):
                    new_parts: list[Any] = []
                    for c_idx, part in enumerate(item_copy[content_key]):
                        if not isinstance(part, dict):
                            new_parts.append(part)
                            continue
                        if "prompt_cache_breakpoint" in part:
                            raise RequestTranslationError(
                                f"Cannot translate 'prompt_cache_breakpoint' at input[{idx}].{content_key}[{c_idx}].",
                                unsupported_parameters=(
                                    f"input[{idx}].{content_key}[{c_idx}].prompt_cache_breakpoint",
                                ),
                            )
                        p_type = part.get("type")
                        if p_type in ("input_audio", "audio", "input_file", "file", "item_reference") or "file_id" in part:
                            raise RequestTranslationError(
                                f"Unsupported content part at input[{idx}].{content_key}[{c_idx}].",
                                unsupported_operations=(f"content_part:{p_type or 'file_id'}",),
                            )
                        if p_type == "input_image":
                            ph = register_image(
                                part.get("image_url", ""),
                                f"input[{idx}].{content_key}[{c_idx}]",
                            )
                            new_parts.append({"type": "input_text", "text": ph})
                        else:
                            new_parts.append(part)
                    item_copy[content_key] = new_parts
            translated_items.append(item_copy)

        return translated_items

    def _translate_openai_instructions_to_typesafe(
        self,
        instructions: Any,
        *,
        location: str,
        register_image: Any,
    ) -> Any:
        if instructions is None or isinstance(instructions, (str, dict)):
            return instructions
        if isinstance(instructions, list):
            segments: list[str] = []
            for p_idx, part in enumerate(instructions):
                if not isinstance(part, dict):
                    raise RequestTranslationError(f"Invalid instruction part at {location}[{p_idx}].")
                if "prompt_cache_breakpoint" in part:
                    raise RequestTranslationError(
                        f"Cannot translate 'prompt_cache_breakpoint' at {location}[{p_idx}].",
                        unsupported_parameters=(f"{location}[{p_idx}].prompt_cache_breakpoint",),
                    )
                p_type = part.get("type")
                if p_type == "input_text" and isinstance(part.get("text"), str):
                    segments.append(part["text"])
                elif p_type == "input_image":
                    ph = register_image(part.get("image_url", ""), f"{location}[{p_idx}]")
                    segments.append(ph)
                else:
                    raise RequestTranslationError(
                        f"Unsupported instruction part type '{p_type}' at {location}[{p_idx}].",
                        unsupported_operations=(f"instruction_part:{p_type}",),
                    )
            return " ".join(s for s in segments if s)
        raise RequestTranslationError(f"Unsupported instructions format at {location}.")
