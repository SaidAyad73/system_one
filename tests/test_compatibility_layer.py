"""Comprehensive automated tests for the TypeSafe <-> OpenAI Decisions API Compatibility Layer."""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from typing import Any

from compatibility_layer import (
    APIProvider,
    CapabilityResolver,
    CompatibilityLevel,
    DiagnosticsRecorder,
    ExecutionPath,
    OpenAIDecisionsAdapter,
    RawHTTPRequest,
    RawHTTPResponse,
    RequestInterceptor,
    RequestTranslationError,
    ResponseTranslationError,
    TrafficRouter,
    TranslationMiddleware,
    TypeSafeAdapter,
    UpstreamTransportError,
    create_http_proxy_server,
)


class MockUpstreamRecorder:
    """Records outbound requests and returns pre-configured RawHTTPResponse objects."""

    def __init__(self, default_response: RawHTTPResponse | None = None) -> None:
        self.requests: list[RawHTTPRequest] = []
        self.next_response: RawHTTPResponse = default_response or RawHTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b"{}",
        )
        self.raise_error: Exception | None = None

    def __call__(self, request: RawHTTPRequest) -> RawHTTPResponse:
        self.requests.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        return self.next_response


class CompatibilityLayerTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.ts_mock = MockUpstreamRecorder()
        self.oa_mock = MockUpstreamRecorder()
        self.diagnostics = DiagnosticsRecorder()
        self.router = TrafficRouter(
            typesafe_adapter=TypeSafeAdapter(
                base_url="https://api.typesafe.ai",
                api_key="ts-secret-test-key-12345",
                transport=self.ts_mock,
            ),
            openai_adapter=OpenAIDecisionsAdapter(
                base_url="https://api.openai.com",
                api_key="sk-secret-openai-key-67890",
                transport=self.oa_mock,
            ),
            diagnostics=self.diagnostics,
        )

    # -------------------------------------------------------------------------
    # 1 & 11 & 12: Native TypeSafe execution without translation (byte-for-byte preservation)
    # -------------------------------------------------------------------------

    def test_native_typesafe_execution_without_translation_preserves_exact_bytes_and_headers(self) -> None:
        # Deliberately non-standard JSON whitespace and key ordering to verify zero re-serialization
        raw_req_body = (
            b'{\n  "model" : "jev-latest" ,\n'
            b'  "state": "My card was charged twice.",\n'
            b'  "questions": {\n'
            b'    "is_duplicate": {"type": "noul", "instructions": "Is this a duplicate charge?"},\n'
            b'    "dept": {"type": "choice", "instructions": "Pick team", "criteria": {"billing": "Payments", "other": "Other"}},\n'
            b'    "severity": {"type": "score", "instructions": "Rate severity", "criteria": ["Low", "Medium", "High"]}\n'
            b'  }\n}'
        )
        raw_resp_body = (
            b'{\n  "model": "jev-1.13.0",\n'
            b'  "answers": {\n'
            b'    "is_duplicate": {"type": "noul", "noul": 0.96},\n'
            b'    "dept": {"type": "choice", "choice": "billing", "confidence": 0.91, "probabilities": {"billing": 0.95, "other": 0.05}},\n'
            b'    "severity": {"type": "score", "score": 1.43, "confidence": 0.35, "legend": {"0": "Low", "1": "Medium", "2": "High"}, "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43}}\n'
            b'  },\n'
            b'  "usage": {"input_tokens": 42, "output_tokens": 12}\n}'
        )
        self.ts_mock.next_response = RawHTTPResponse(
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "X-TypeSafe-Request-Id": "req_ts_999",
                "X-Custom-Upstream-Header": "kept-verbatim",
            },
            body=raw_resp_body,
        )

        req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={
                "Authorization": "Bearer ts-caller-token-abc",
                "Content-Type": "application/json",
                "X-Caller-Trace": "trace-1",
            },
            body=raw_req_body,
        )
        resp = self.router.handle(req)

        self.assertEqual(len(self.ts_mock.requests), 1)
        self.assertEqual(len(self.oa_mock.requests), 0)
        # Verify exact request bytes and headers were forwarded untouched
        self.assertEqual(self.ts_mock.requests[0].body, raw_req_body)
        self.assertEqual(self.ts_mock.requests[0].headers["X-Caller-Trace"], "trace-1")
        # Verify exact response bytes, status code, and headers were returned untouched
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.body, raw_resp_body)
        self.assertEqual(resp.headers["X-TypeSafe-Request-Id"], "req_ts_999")
        self.assertEqual(resp.headers["X-Custom-Upstream-Header"], "kept-verbatim")
        self.assertEqual(self.diagnostics.events[-1].execution_path, ExecutionPath.NATIVE_TYPESAFE)

    # -------------------------------------------------------------------------
    # 3 & 11 & 12: Native OpenAI Decisions execution without translation
    # -------------------------------------------------------------------------

    def test_native_openai_execution_without_translation_preserves_exact_bytes_and_headers(self) -> None:
        raw_req_body = (
            b'{"model":"gpt-6-luna","input":"Customer charged twice.","questions":['
            b'{"type":"predicate","name":"dup","instructions":"Duplicate charge?"}]}'
        )
        raw_resp_body = b'{"answers":[{"type":"predicate","name":"dup","probability":0.92}]}'
        self.oa_mock.next_response = RawHTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json", "x-request-id": "req_oa_123"},
            body=raw_resp_body,
        )

        req = RawHTTPRequest(
            method="POST",
            url="https://api.openai.com/v1/decisions",
            headers={"Authorization": "Bearer sk-caller-key", "Content-Type": "application/json"},
            body=raw_req_body,
        )
        resp = self.router.handle(req)

        self.assertEqual(len(self.oa_mock.requests), 1)
        self.assertEqual(len(self.ts_mock.requests), 0)
        self.assertEqual(self.oa_mock.requests[0].body, raw_req_body)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.body, raw_resp_body)
        self.assertEqual(resp.headers["x-request-id"], "req_oa_123")
        self.assertEqual(self.diagnostics.events[-1].execution_path, ExecutionPath.NATIVE_OPENAI)

    # -------------------------------------------------------------------------
    # 2 & 7: TypeSafe -> OpenAI Decisions fallback & successful response translation
    # -------------------------------------------------------------------------

    def test_typesafe_to_openai_fallback_when_choices_exceed_255(self) -> None:
        # TypeSafe caps choice.criteria at 255 entries; OpenAI choices has no 255 cap.
        many_choices = {f"opt_{i}": f"Description for option {i}" for i in range(260)}
        ts_payload = {
            "model": "jev-latest",
            "custom_param": "passed-unmodified",
            "state": {"ticket": "Order #102", "amount": 49},
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Select category",
                    "criteria": many_choices,
                },
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Is the ticket urgent?",
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate severity",
                    "criteria": ["Cosmetic", "Degraded", "Blocking"],
                },
            },
        }

        oa_probs = [
            {"value": f"opt_{i}", "probability": (0.8 if i == 0 else 0.2 / 259)}
            for i in range(260)
        ]
        oa_upstream_response = {
            "model": "gpt-6-luna",
            "answers": [
                {
                    "type": "choice",
                    "name": "category",
                    "choice": "opt_0",
                    "probabilities": oa_probs,
                    "confidence": 0.88,
                },
                {
                    "type": "predicate",
                    "name": "is_urgent",
                    "probability": 0.91,
                },
                {
                    "type": "score",
                    "name": "severity",
                    "score": 1.43,
                    "probabilities": [
                        {"value": 0, "label": "Cosmetic", "probability": 0.0},
                        {"value": 1, "label": "Degraded", "probability": 0.57},
                        {"value": 2, "label": "Blocking", "probability": 0.43},
                    ],
                    "confidence": 0.35,
                },
            ],
            "usage": {"input_tokens": 120, "output_tokens": 15},
        }
        self.oa_mock.next_response = RawHTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json", "x-request-id": "oa_fallback_1"},
            body=json.dumps(oa_upstream_response).encode("utf-8"),
        )

        req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={"Content-Type": "application/json"},
            body=json.dumps(ts_payload).encode("utf-8"),
        )
        resp = self.router.handle(req)

        # Routed to OpenAI Decisions API
        self.assertEqual(len(self.ts_mock.requests), 0)
        self.assertEqual(len(self.oa_mock.requests), 1)
        self.assertEqual(
            self.diagnostics.events[-1].execution_path,
            ExecutionPath.TRANSLATED_TYPESAFE_TO_OPENAI,
        )

        sent_oa_json = json.loads(self.oa_mock.requests[0].body.decode("utf-8"))
        # Verify model and custom_param were passed unmodified!
        self.assertEqual(sent_oa_json["model"], "jev-latest")
        self.assertEqual(sent_oa_json["custom_param"], "passed-unmodified")
        # Verify structured JSON state was mapped to input_json inside user message
        self.assertEqual(
            sent_oa_json["input"],
            [{"role": "user", "content": [{"type": "input_json", "data": {"ticket": "Order #102", "amount": 49}}]}],
        )
        self.assertEqual(len(sent_oa_json["questions"]), 3)
        self.assertEqual(sent_oa_json["questions"][0]["type"], "choice")
        self.assertEqual(len(sent_oa_json["questions"][0]["choices"]), 260)

        # Verify translated response is in TypeSafe SystemOneResponse format
        translated_ts_resp = resp.json()
        self.assertEqual(translated_ts_resp["model"], "gpt-6-luna")
        self.assertEqual(translated_ts_resp["usage"], {"input_tokens": 120, "output_tokens": 15})
        self.assertEqual(translated_ts_resp["answers"]["is_urgent"], {"type": "noul", "noul": 0.91})
        self.assertEqual(translated_ts_resp["answers"]["category"]["type"], "choice")
        self.assertEqual(translated_ts_resp["answers"]["category"]["choice"], "opt_0")
        self.assertEqual(len(translated_ts_resp["answers"]["category"]["probabilities"]), 260)
        self.assertEqual(
            translated_ts_resp["answers"]["severity"],
            {
                "type": "score",
                "score": 1.43,
                "confidence": 0.35,
                "legend": {"0": "Cosmetic", "1": "Degraded", "2": "Blocking"},
                "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43},
            },
        )

    def test_typesafe_to_openai_fallback_for_image_assets_without_danger_zone_header(self) -> None:
        # TypeSafe requires TypeSafe-Danger-Zone-Images: true for image assets; OpenAI supports base64 images natively.
        ts_payload = {
            "model": "jev-latest",
            "state": "Inspect this product photo: {{item_photo}}",
            "assets": {
                "{{item_photo}}": {
                    "type": "image",
                    "format": "png",
                    "data_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                }
            },
            "questions": {
                "damaged": {
                    "type": "noul",
                    "instructions": "Is the item visibly damaged?",
                }
            },
        }
        self.oa_mock.next_response = RawHTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {"answers": [{"type": "predicate", "name": "damaged", "probability": 0.84}]}
            ).encode("utf-8"),
        )

        req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={"Content-Type": "application/json"},  # No TypeSafe-Danger-Zone-Images header
            body=json.dumps(ts_payload).encode("utf-8"),
        )
        resp = self.router.handle(req)

        self.assertEqual(len(self.oa_mock.requests), 1)
        sent_oa = json.loads(self.oa_mock.requests[0].body.decode("utf-8"))
        self.assertEqual(sent_oa["input"][0]["role"], "user")
        self.assertEqual(sent_oa["input"][0]["content"][0], {"type": "input_text", "text": "Inspect this product photo: "})
        self.assertTrue(
            sent_oa["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
        )
        self.assertEqual(resp.json()["answers"]["damaged"], {"type": "noul", "noul": 0.84})

    def test_question_specific_image_asset_scoped_to_question_instructions_and_query_preserved(self) -> None:
        ts_payload = {
            "model": "jev-latest",
            "state": "Shared reference text only.",
            "assets": {
                "{{candidate_img}}": {
                    "type": "image",
                    "format": "jpeg",
                    "data_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/",
                }
            },
            "questions": {
                "candidate_match": {
                    "type": "noul",
                    "instructions": "Does candidate match: {{candidate_img}}",
                }
            },
        }
        self.oa_mock.next_response = RawHTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {"answers": [{"type": "predicate", "name": "candidate_match", "probability": 0.77}]}
            ).encode("utf-8"),
        )
        req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone?trace_id=abc",
            headers={"Content-Type": "application/json"},
            body=json.dumps(ts_payload).encode("utf-8"),
        )
        resp = self.router.handle(req)

        self.assertEqual(self.oa_mock.requests[-1].url, "https://api.openai.com/v1/decisions?trace_id=abc")
        sent_oa = json.loads(self.oa_mock.requests[-1].body.decode("utf-8"))
        # Shared input stays a plain string because the image asset was referenced only in question instructions
        self.assertEqual(sent_oa["input"], "Shared reference text only.")
        self.assertEqual(
            sent_oa["questions"][0]["instructions"],
            [
                {"type": "input_text", "text": "Does candidate match: "},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/"},
            ],
        )
        self.assertEqual(resp.json()["answers"]["candidate_match"], {"type": "noul", "noul": 0.77})

    # -------------------------------------------------------------------------
    # 4 & 7: OpenAI Decisions -> TypeSafe fallback & successful response translation
    # -------------------------------------------------------------------------

    def test_openai_to_typesafe_fallback_when_questions_exceed_400(self) -> None:
        # OpenAI Decisions API caps questions at 400; TypeSafe has no 400 cap.
        questions = [
            {
                "type": "predicate",
                "name": f"cond_{i}",
                "instructions": f"Does condition {i} hold?",
            }
            for i in range(405)
        ]
        oa_payload = {
            "model": "gpt-6-luna",
            "prompt_cache_key": "caller-partition-1",
            "input": "Customer report text.",
            "questions": questions,
        }
        ts_answers = {
            f"cond_{i}": {"type": "noul", "noul": round((i % 10) / 10.0, 2)}
            for i in range(405)
        }
        self.ts_mock.next_response = RawHTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "jev-1.13.0",
                    "answers": ts_answers,
                    "usage": {"input_tokens": 500, "output_tokens": 405},
                }
            ).encode("utf-8"),
        )

        req = RawHTTPRequest(
            method="POST",
            url="https://api.openai.com/v1/decisions",
            headers={"Content-Type": "application/json"},
            body=json.dumps(oa_payload).encode("utf-8"),
        )
        resp = self.router.handle(req)

        self.assertEqual(len(self.oa_mock.requests), 0)
        self.assertEqual(len(self.ts_mock.requests), 1)
        self.assertEqual(
            self.diagnostics.events[-1].execution_path,
            ExecutionPath.TRANSLATED_OPENAI_TO_TYPESAFE,
        )

        sent_ts = json.loads(self.ts_mock.requests[0].body.decode("utf-8"))
        # Non-prompt parameters (model, prompt_cache_key) are passed unmodified!
        self.assertEqual(sent_ts["model"], "gpt-6-luna")
        self.assertEqual(sent_ts["prompt_cache_key"], "caller-partition-1")
        self.assertEqual(sent_ts["state"], "Customer report text.")
        self.assertEqual(len(sent_ts["questions"]), 405)

        oa_resp = resp.json()
        self.assertEqual(len(oa_resp["answers"]), 405)
        self.assertEqual(oa_resp["answers"][0], {"type": "predicate", "name": "cond_0", "probability": 0.0})
        self.assertEqual(oa_resp["answers"][7], {"type": "predicate", "name": "cond_7", "probability": 0.7})

    def test_openai_to_typesafe_fallback_for_single_level_score_and_raw_object_input(self) -> None:
        # OpenAI requires 2..10 score levels and rejects raw dict `input`; TypeSafe supports 1..10 levels and dict `state`.
        oa_payload = {
            "model": "gpt-6-luna",
            "input": {"ticket_id": "T-55", "body": "Export fails in Safari."},
            "questions": [
                {
                    "type": "score",
                    "name": "severity",
                    "instructions": "Rate severity",
                    "levels": [{"label": "Minor", "description": "No functionality lost"}],
                },
                {
                    "type": "choice",
                    "name": "team",
                    "instructions": "Pick team",
                    "choices": [
                        {"value": "frontend", "description": "Browser issues"},
                        {"value": "backend", "description": "Server issues"},
                    ],
                },
            ],
        }
        self.ts_mock.next_response = RawHTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "jev-1.13.0",
                    "answers": {
                        "severity": {
                            "type": "score",
                            "score": 0.0,
                            "confidence": 1.0,
                            "legend": {"0": {"label": "Minor", "description": "No functionality lost"}},
                            "probabilities": {"0": 1.0},
                        },
                        "team": {
                            "type": "choice",
                            "choice": "frontend",
                            "confidence": 0.9,
                            "probabilities": {"frontend": 0.95, "backend": 0.05},
                        },
                    },
                    "usage": {"input_tokens": 30, "output_tokens": 5},
                }
            ).encode("utf-8"),
        )

        req = RawHTTPRequest(
            method="POST",
            url="https://api.openai.com/v1/decisions",
            headers={"Content-Type": "application/json"},
            body=json.dumps(oa_payload).encode("utf-8"),
        )
        resp = self.router.handle(req)

        self.assertEqual(len(self.ts_mock.requests), 1)
        oa_resp = resp.json()
        self.assertEqual(
            oa_resp["answers"][0],
            {
                "type": "score",
                "name": "severity",
                "score": 0.0,
                "probabilities": [{"value": 0, "label": "Minor", "probability": 1.0}],
                "confidence": 1.0,
            },
        )
        self.assertEqual(
            oa_resp["answers"][1],
            {
                "type": "choice",
                "name": "team",
                "choice": "frontend",
                "probabilities": [
                    {"value": "frontend", "probability": 0.95},
                    {"value": "backend", "probability": 0.05},
                ],
                "confidence": 0.9,
            },
        )

    # -------------------------------------------------------------------------
    # 5: Requests believed to be unsupported by both APIs -> forwarded unchanged to original
    # -------------------------------------------------------------------------

    def test_unsupported_by_both_apis_forwards_unchanged_to_original_typesafe(self) -> None:
        # Score with 12 levels is unsupported by both TypeSafe (max 10) and OpenAI (max 10).
        raw_body = json.dumps(
            {
                "model": "jev-latest",
                "state": "Some state",
                "questions": {
                    "bad_score": {
                        "type": "score",
                        "instructions": "Rate on 12 levels",
                        "criteria": [f"Level {i}" for i in range(12)],
                    }
                },
            }
        ).encode("utf-8")
        upstream_422_body = b'{"detail":[{"loc":["body","questions","bad_score","criteria"],"msg":"List should have at most 10 items","type":"too_long"}]}'
        self.ts_mock.next_response = RawHTTPResponse(
            status_code=422,
            headers={"Content-Type": "application/json"},
            body=upstream_422_body,
        )

        req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={"Content-Type": "application/json"},
            body=raw_body,
        )
        resp = self.router.handle(req)

        # Forwarded unchanged to TypeSafe; OpenAI never called
        self.assertEqual(len(self.ts_mock.requests), 1)
        self.assertEqual(len(self.oa_mock.requests), 0)
        self.assertEqual(self.ts_mock.requests[0].body, raw_body)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.body, upstream_422_body)
        self.assertEqual(
            self.diagnostics.events[-1].execution_path,
            ExecutionPath.FALLBACK_UNSUPPORTED_TO_ORIGINAL,
        )

    def test_unsupported_by_both_apis_forwards_unchanged_to_original_openai(self) -> None:
        # file_id in input_image is unsupported by OpenAI Decisions API and unsupported by TypeSafe.
        raw_body = json.dumps(
            {
                "model": "gpt-6-luna",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_image", "file_id": "file-abc123"}],
                    }
                ],
                "questions": [{"type": "predicate", "name": "q1", "instructions": "Check image"}],
            }
        ).encode("utf-8")
        upstream_400_body = b'{"error":{"message":"file_id is not supported in Decisions API","type":"invalid_request_error"}}'
        self.oa_mock.next_response = RawHTTPResponse(
            status_code=400,
            headers={"Content-Type": "application/json"},
            body=upstream_400_body,
        )

        req = RawHTTPRequest(
            method="POST",
            url="https://api.openai.com/v1/decisions",
            headers={"Content-Type": "application/json"},
            body=raw_body,
        )
        resp = self.router.handle(req)

        self.assertEqual(len(self.oa_mock.requests), 1)
        self.assertEqual(len(self.ts_mock.requests), 0)
        self.assertEqual(self.oa_mock.requests[0].body, raw_body)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.body, upstream_400_body)
        self.assertEqual(
            self.diagnostics.events[-1].execution_path,
            ExecutionPath.FALLBACK_UNSUPPORTED_TO_ORIGINAL,
        )

    # -------------------------------------------------------------------------
    # 6: Unknown or ambiguous compatibility -> forwarded unchanged to original
    # -------------------------------------------------------------------------

    def test_unknown_compatibility_malformed_json_or_unrecognized_primitive_forwards_unchanged(self) -> None:
        # Case A: Malformed JSON body sent to OpenAI
        corrupt_body = b'{"model": "gpt-6-luna", "input": "unterminated...'
        self.oa_mock.next_response = RawHTTPResponse(
            status_code=400,
            headers={"Content-Type": "application/json"},
            body=b'{"error":{"message":"Invalid JSON"}}',
        )
        req1 = RawHTTPRequest(
            method="POST",
            url="https://api.openai.com/v1/decisions",
            headers={"Content-Type": "application/json"},
            body=corrupt_body,
        )
        resp1 = self.router.handle(req1)
        self.assertEqual(self.oa_mock.requests[-1].body, corrupt_body)
        self.assertEqual(resp1.status_code, 400)
        self.assertEqual(
            self.diagnostics.events[-1].execution_path,
            ExecutionPath.FALLBACK_UNKNOWN_TO_ORIGINAL,
        )

        # Case B: Unrecognized future question primitive sent to TypeSafe
        future_primitive_body = json.dumps(
            {
                "model": "jev-latest",
                "state": "Some state",
                "questions": {"q1": {"type": "future_ranking_primitive", "instructions": "Rank items"}},
            }
        ).encode("utf-8")
        self.ts_mock.next_response = RawHTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"model":"jev-2.0","answers":{"q1":{"type":"future_ranking_primitive"}}}',
        )
        req2 = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={"Content-Type": "application/json"},
            body=future_primitive_body,
        )
        resp2 = self.router.handle(req2)
        self.assertEqual(self.ts_mock.requests[-1].body, future_primitive_body)
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(
            self.diagnostics.events[-1].execution_path,
            ExecutionPath.FALLBACK_UNKNOWN_TO_ORIGINAL,
        )

    # -------------------------------------------------------------------------
    # 8: Raw upstream errors after translation
    # -------------------------------------------------------------------------

    def test_raw_upstream_error_after_translation_is_returned_without_normalization(self) -> None:
        # TypeSafe request with >255 choices translates to OpenAI, and OpenAI returns HTTP 429 rate-limit error
        many_choices = {f"c_{i}": f"Choice {i}" for i in range(260)}
        ts_req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "jev-latest",
                    "state": "Hello",
                    "questions": {
                        "q_choice": {
                            "type": "choice",
                            "instructions": "Select one",
                            "criteria": many_choices,
                        }
                    },
                }
            ).encode("utf-8"),
        )
        raw_openai_429_body = (
            b'{\n  "error": {\n    "message": "Rate limit reached for gpt-6-luna",\n'
            b'    "type": "rate_limit_error",\n    "code": "rate_limit_exceeded"\n  }\n}'
        )
        self.oa_mock.next_response = RawHTTPResponse(
            status_code=429,
            headers={
                "Content-Type": "application/json",
                "Retry-After": "15",
                "x-request-id": "req_oa_err_429",
            },
            body=raw_openai_429_body,
        )

        resp = self.router.handle(ts_req)

        # Must preserve OpenAI's exact 429 status, raw body, and headers without translating to TypeSafe error format
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(resp.body, raw_openai_429_body)
        self.assertEqual(resp.headers["Retry-After"], "15")
        self.assertEqual(resp.headers["x-request-id"], "req_oa_err_429")
        self.assertTrue(resp.from_upstream)
        self.assertEqual(resp.upstream_provider, APIProvider.OPENAI_DECISIONS)

    # -------------------------------------------------------------------------
    # 9: Raw errors when translation is not required
    # -------------------------------------------------------------------------

    def test_raw_upstream_error_when_no_translation_required(self) -> None:
        raw_ts_401 = b'{"detail":"Invalid API key provided"}'
        self.ts_mock.next_response = RawHTTPResponse(
            status_code=401,
            headers={"Content-Type": "application/json", "WWW-Authenticate": "Bearer"},
            body=raw_ts_401,
        )

        req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "jev-latest",
                    "state": "Test",
                    "questions": {"q1": {"type": "noul", "instructions": "Yes or no?"}},
                }
            ).encode("utf-8"),
        )
        resp = self.router.handle(req)

        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.body, raw_ts_401)
        self.assertEqual(resp.headers["WWW-Authenticate"], "Bearer")

    # -------------------------------------------------------------------------
    # 10: Middleware failures before an upstream response exists & on untranslatable refusal
    # -------------------------------------------------------------------------

    def test_middleware_failure_before_upstream_response_is_distinct_from_upstream_error(self) -> None:
        # Simulate transport connection failure before any HTTP response exists
        self.ts_mock.raise_error = ConnectionResetError("Connection reset by peer")

        req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "jev-latest",
                    "state": "Test",
                    "questions": {"q1": {"type": "noul", "instructions": "Yes or no?"}},
                }
            ).encode("utf-8"),
        )
        resp = self.router.handle(req, raise_middleware_errors=False)

        self.assertFalse(resp.from_upstream)
        self.assertIsNone(resp.upstream_provider)
        self.assertEqual(resp.get_header("X-Compatibility-Error-Origin"), "middleware")
        self.assertEqual(resp.get_header("X-Compatibility-Error-Stage"), "upstream_transport")
        err_json = resp.json()
        self.assertEqual(err_json["middleware_error"]["origin"], "compatibility_middleware")
        self.assertEqual(err_json["middleware_error"]["type"], "UpstreamTransportError")

        # Also verify raise_middleware_errors=True raises UpstreamTransportError directly
        with self.assertRaises(UpstreamTransportError):
            self.router.handle(req, raise_middleware_errors=True)

    def test_openai_refusal_answer_during_response_translation_surfaces_middleware_error_with_raw_upstream(
        self,
    ) -> None:
        # OpenAI can return type="refusal" in answers; TypeSafe has no refusal schema and must not fabricate a probability.
        many_choices = {f"c_{i}": f"Choice {i}" for i in range(260)}
        ts_req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "jev-latest",
                    "state": "Sensitive request",
                    "questions": {
                        "q_choice": {
                            "type": "choice",
                            "instructions": "Select one",
                            "criteria": many_choices,
                        }
                    },
                }
            ).encode("utf-8"),
        )
        raw_refusal_body = b'{"answers":[{"type":"refusal","name":"q_choice"}]}'
        self.oa_mock.next_response = RawHTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=raw_refusal_body,
        )

        with self.assertRaises(ResponseTranslationError) as ctx:
            self.router.handle(ts_req, raise_middleware_errors=True)

        self.assertIsNotNone(ctx.exception.raw_upstream_response)
        assert ctx.exception.raw_upstream_response is not None
        self.assertEqual(ctx.exception.raw_upstream_response.body, raw_refusal_body)

    # -------------------------------------------------------------------------
    # 13: Partial compatibility and unsupported parameters
    # -------------------------------------------------------------------------

    def test_partial_compatibility_noul_with_criteria_and_structured_choice_descriptions(self) -> None:
        interceptor = RequestInterceptor()
        resolver = CapabilityResolver()
        translator = TranslationMiddleware()

        # TypeSafe request with noul.criteria (true/false definitions) and >255 choices
        many_choices = {f"c_{i}": f"Desc {i}" for i in range(260)}
        ts_req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "jev-latest",
                    "state": "Customer contacted support 3 times.",
                    "questions": {
                        "is_repeat": {
                            "type": "noul",
                            "instructions": "Has customer contacted support before?",
                            "criteria": {
                                "true": "Mentions prior contact",
                                "false": "First time contact",
                            },
                        },
                        "big_choice": {
                            "type": "choice",
                            "instructions": "Select category",
                            "criteria": many_choices,
                        },
                    },
                }
            ).encode("utf-8"),
        )

        classified = interceptor.classify(ts_req)
        alt_report = resolver.evaluate_translation_capability(classified, APIProvider.OPENAI_DECISIONS)
        self.assertEqual(alt_report.level, CompatibilityLevel.PARTIAL)
        self.assertFalse(alt_report.can_execute)
        self.assertIn("questions.is_repeat.criteria", alt_report.unsupported_parameters)

        # Direct translation attempt refuses to silently discard `criteria`
        with self.assertRaises(RequestTranslationError) as err_ctx:
            translator.translate_typesafe_to_openai_request(ts_req)
        self.assertIn("questions.is_repeat.criteria", err_ctx.exception.unsupported_parameters)

        # Router forwards unchanged to TypeSafe rather than silently dropping `criteria`
        self.ts_mock.next_response = RawHTTPResponse(status_code=422, headers={}, body=b'{"detail":[]}')
        resp = self.router.handle(ts_req)
        self.assertEqual(len(self.ts_mock.requests), 1)
        self.assertEqual(len(self.oa_mock.requests), 0)
        self.assertEqual(resp.status_code, 422)

    def test_partial_compatibility_openai_prompt_cache_breakpoint_in_input_part(self) -> None:
        interceptor = RequestInterceptor()
        resolver = CapabilityResolver()
        translator = TranslationMiddleware()

        # OpenAI request with >400 questions AND a prompt_cache_breakpoint on an input_text part
        oa_req = RawHTTPRequest(
            method="POST",
            url="https://api.openai.com/v1/decisions",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "gpt-6-luna",
                    "prompt_cache_options": {"mode": "explicit"},
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Shared prefix evidence",
                                    "prompt_cache_breakpoint": {"mode": "explicit"},
                                }
                            ],
                        }
                    ],
                    "questions": [
                        {"type": "predicate", "name": f"q_{i}", "instructions": "Valid?"}
                        for i in range(401)
                    ],
                }
            ).encode("utf-8"),
        )

        classified = interceptor.classify(oa_req)
        alt_report = resolver.evaluate_translation_capability(classified, APIProvider.TYPESAFE)
        self.assertEqual(alt_report.level, CompatibilityLevel.PARTIAL)
        self.assertFalse(alt_report.can_execute)
        self.assertIn(
            "input[0].content[0].prompt_cache_breakpoint",
            alt_report.unsupported_parameters,
        )

        with self.assertRaises(RequestTranslationError):
            translator.translate_openai_to_typesafe_request(oa_req)

    # -------------------------------------------------------------------------
    # 14: Edge cases identified in the documentation & observability credential redaction
    # -------------------------------------------------------------------------

    def test_documented_edge_cases_box_question_and_tool_call_matching(self) -> None:
        interceptor = RequestInterceptor()
        resolver = CapabilityResolver()

        # Edge Case 1: TypeSafe `box` question WITH image asset and TypeSafe-Danger-Zone-Images: true is FULL native
        box_req_valid = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={
                "Content-Type": "application/json",
                "TypeSafe-Danger-Zone-Images": "true",
            },
            body=json.dumps(
                {
                    "model": "jev-latest",
                    "state": "Find defect in {{img}}",
                    "assets": {
                        "{{img}}": {
                            "type": "image",
                            "format": "jpg",
                            "data_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/",
                        }
                    },
                    "questions": {"defect_box": {"type": "box", "instructions": "Locate the scratch"}},
                }
            ).encode("utf-8"),
        )
        c_box = interceptor.classify(box_req_valid)
        self.assertEqual(resolver.evaluate_native_capability(c_box).level, CompatibilityLevel.FULL)
        # And box is UNSUPPORTED on OpenAI Decisions API
        self.assertEqual(
            resolver.evaluate_translation_capability(c_box, APIProvider.OPENAI_DECISIONS).level,
            CompatibilityLevel.UNSUPPORTED,
        )

        # Edge Case 2: OpenAI tool output with unmatched `call_id` is UNSUPPORTED natively on OpenAI,
        # but falls back to TypeSafe structured `state` array
        unmatched_tool_req = RawHTTPRequest(
            method="POST",
            url="https://api.openai.com/v1/decisions",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "gpt-6-luna",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_missing_999",
                            "output": "Order status: shipped",
                        }
                    ],
                    "questions": [{"type": "predicate", "name": "shipped", "instructions": "Has it shipped?"}],
                }
            ).encode("utf-8"),
        )
        c_tool = interceptor.classify(unmatched_tool_req)
        self.assertEqual(resolver.evaluate_native_capability(c_tool).level, CompatibilityLevel.UNSUPPORTED)
        self.assertEqual(
            resolver.evaluate_translation_capability(c_tool, APIProvider.TYPESAFE).level,
            CompatibilityLevel.FULL,
        )

        # Edge Case 3: OpenAI `input_image` with base64 PNG translates to TypeSafe `assets` + `TypeSafe-Danger-Zone-Images: true`
        translator = TranslationMiddleware()
        oa_img_req = RawHTTPRequest(
            method="POST",
            url="https://api.openai.com/v1/decisions",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "gpt-6-luna",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Reference photo:"},
                                {
                                    "type": "input_image",
                                    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                                },
                            ],
                        }
                    ],
                    "questions": [
                        {
                            "type": "predicate",
                            "name": "match",
                            "instructions": "Is the item intact?",
                        }
                    ],
                }
            ).encode("utf-8"),
        )
        ts_translated_req, _ = translator.translate_openai_to_typesafe_request(oa_img_req)
        self.assertEqual(ts_translated_req.get_header("TypeSafe-Danger-Zone-Images"), "true")
        ts_translated_body = json.loads(ts_translated_req.body.decode("utf-8"))
        self.assertIn("{{asset_image_0}}", ts_translated_body["assets"])
        self.assertEqual(ts_translated_body["assets"]["{{asset_image_0}}"]["format"], "png")

    def test_diagnostics_redacts_authorization_and_api_keys(self) -> None:
        req = RawHTTPRequest(
            method="POST",
            url="https://api.typesafe.ai/v1/systemone",
            headers={
                "Authorization": "Bearer ts-live-secret-token-999999",
                "X-Api-Key": "sk-secret-key-111111",
                "Content-Type": "application/json",
            },
            body=json.dumps(
                {
                    "model": "jev-latest",
                    "state": "Hello",
                    "questions": {"q1": {"type": "noul", "instructions": "Yes?"}},
                }
            ).encode("utf-8"),
        )
        self.router.handle(req)
        event = self.diagnostics.events[-1]
        self.assertEqual(event.sanitized_headers["Authorization"], "[REDACTED]")
        self.assertEqual(event.sanitized_headers["X-Api-Key"], "[REDACTED]")

    def test_http_proxy_server_end_to_end(self) -> None:
        self.ts_mock.next_response = RawHTTPResponse(
            status_code=200,
            headers={"Content-Type": "application/json", "X-Upstream": "typesafe"},
            body=b'{"model":"jev-1.13.0","answers":{"q1":{"type":"noul","noul":0.88}},"usage":{"input_tokens":10,"output_tokens":2}}',
        )
        server = create_http_proxy_server(self.router, host="127.0.0.1", port=0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = json.dumps(
                {
                    "model": "jev-latest",
                    "state": "Proxy test",
                    "questions": {"q1": {"type": "noul", "instructions": "Does proxy work?"}},
                }
            ).encode("utf-8")
            http_req = urllib.request.Request(
                url=f"http://127.0.0.1:{port}/v1/systemone",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(http_req, timeout=5.0) as http_resp:
                self.assertEqual(http_resp.status, 200)
                self.assertEqual(http_resp.headers.get("X-Upstream"), "typesafe")
                resp_data = json.loads(http_resp.read().decode("utf-8"))
                self.assertEqual(resp_data["answers"]["q1"]["noul"], 0.88)
        finally:
            server.shutdown()
            server.server_close()

    @unittest.skipUnless(
        os.environ.get("RUN_LIVE_INTEGRATION_TESTS") == "1"
        and (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("OPENAI_API_KEY")),
        "Live integration tests require RUN_LIVE_INTEGRATION_TESTS=1 and API credentials.",
    )
    def test_live_integration_when_credentials_available(self) -> None:
        live_router = TrafficRouter()
        if os.environ.get("TYPESAFE_API_KEY"):
            req = RawHTTPRequest(
                method="POST",
                url="https://api.typesafe.ai/v1/systemone",
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {
                        "model": "jev-latest",
                        "state": "The customer says: I was charged twice for order #100.",
                        "questions": {
                            "duplicate_charge": {
                                "type": "noul",
                                "instructions": "Is the customer reporting a duplicate charge?",
                            }
                        },
                    }
                ).encode("utf-8"),
            )
            resp = live_router.handle(req)
            self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
