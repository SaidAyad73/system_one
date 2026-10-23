# TypeSafe ↔ OpenAI Decisions API Compatibility Layer & Traffic Router

A modular, transparent compatibility middleware and traffic router bridging the **TypeSafe System One API** (`POST https://api.typesafe.ai/v1/systemone`) and the **OpenAI Decisions API** (`POST https://api.openai.com/v1/decisions`).

---

## 1. Summary of Capabilities and Limitations Discovered in Documentation

### TypeSafe System One API (`https://docs.typesafe.ai`)
- **Endpoints**:
  - `POST /v1/systemone` — Evaluates shared `state` against one or more named `questions`.
  - `GET /v1/models` — Lists available model names and aliases (e.g., `jev-latest`, `jev-preview`, `jev-1.13.0`).
- **Headers**:
  - `Authorization: Bearer <TYPESAFE_API_KEY>`
  - `Content-Type: application/json` or `multipart/form-data` (first part named `request` containing JSON, followed by uniquely named image parts).
  - `TypeSafe-Danger-Zone-Images: true` — Required whenever image `assets` or `box` questions are included.
  - `TypeSafe-Danger-Zone-Long-Context: true` — Enables long-context processing.
- **Inputs (`state` & `assets`)**:
  - `state` (required): Accepts a `string`, a structured JSON `object`, or a JSON `array`.
  - `assets` (optional): Non-empty object mapping placeholder keys (referenced in `state`, `instructions`, or `criteria`) to image assets (`type: "image"`, `format: "png" | "jpg" | "jpeg"`, with either `data_base64` or a multipart `part` name).
  - Does **not** support remote HTTP(S) image URLs, audio, or video.
- **Questions & Outputs (`questions` & `answers`)**:
  - `questions` is a JSON **object (map)** keyed by caller-defined question IDs (`minProperties: 1`; no upper limit in the OpenAPI schema).
  - `noul`: Binary yes/no proposition (`type: "noul"`, optional `instructions` as `string | object | array | null`, optional `criteria: {"true": ..., "false": ...}`). Returns `{"type": "noul", "noul": <0..1>}`.
  - `choice`: Categorical selection (`type: "choice"`, optional `instructions`, required `criteria` object map of **1–255** `{label: description}` entries where descriptions may be strings, objects, arrays, or `null`). Returns `{"type": "choice", "choice": label, "confidence": <0..1>, "probabilities": {label: prob}}`.
  - `score`: Rubric rating (`type: "score"`, optional `instructions`, required `criteria` ordered list of **1–10** level descriptions). Returns `{"type": "score", "score": <weighted_index>, "confidence": <0..1>, "legend": {"0": ...}, "probabilities": {"0": prob}}`.
  - `box`: Spatial bounding-box localization in images (`type: "box"`, optional `instructions`, requires image asset + `TypeSafe-Danger-Zone-Images: true`). Returns `{"type": "box", "box": {"top": ..., "bottom": ..., "left": ..., "right": ...}, "confidence": <number | null>}`.

### OpenAI Decisions API (`https://developers.openai.com/api/docs/guides/decisions`)
- **Endpoints**:
  - `POST /v1/decisions` — Evaluates shared `input` against `1–400` typed `questions`.
  - `GET /v1/organization/usage/decisions` — Organization usage reporting (requires admin key with `api.usage.read`).
- **Inputs (`input`)**:
  - `input` (required): Either a `string` or a non-empty ordered `array` of conversation items:
    - Messages with `role` (`user`, `system`, `developer`, `assistant`):
      - `user` messages support `input_text`, `input_image` (inline `data:image/...;base64,...` or public `http(s)://` URLs; `file_id` is unsupported), and `input_json` (`{"type": "input_json", "data": <object | array>}`, supported **only** in user messages).
      - `system` and `developer` messages support text only.
      - `assistant` messages support output text or refusals.
    - Function and custom tool calls paired in order with tool outputs via `call_id` (tool outputs support text and images).
  - Files (`file_id`), audio, video, and item references are unsupported.
- **Prompt Caching**:
  - Supports `prompt_cache_key`, `prompt_cache_options` (`{"mode": "explicit"}`), and `prompt_cache_breakpoint` (`{"mode": "explicit"}`) on `input_text` and `input_image` parts inside shared `input` (breakpoints are unsupported inside `questions[].instructions`).
- **Questions & Outputs (`questions` & `answers`)**:
  - `questions` is a JSON **array** of **1–400** question objects, each requiring a unique `name` and required `instructions` (`string` or non-empty array of `input_text` / `input_image` parts).
  - `predicate`: Binary condition (`type: "predicate"`, `name`, `instructions`). Returns `{"type": "predicate", "name": ..., "probability": <0..1>}`.
  - `choice`: Categorical selection (`type: "choice"`, `name`, `instructions`, `choices`: array of `{"value": str, "description"?: str}` with distinct values and no 255-option limit). Returns `{"type": "choice", "name": ..., "choice": val, "probabilities": [{"value": val, "probability": prob}], "confidence": <0..1>}`.
  - `score`: Rubric rating (`type: "score"`, `name`, `instructions`, `levels`: ordered array of **2–10** `{"label": str, "description"?: str}`). Returns `{"type": "score", "name": ..., "score": <weighted_index>, "probabilities": [{"value": 0, "label": ..., "probability": prob}], "confidence": <0..1>}`.
  - `refusal` (Response only): An answer in `answers` may be `{"type": "refusal", "name": ...}` instead of a typed result.

---

## 2. Architecture and Routing Decision Flow

### Modular Components (`compatibility_layer/`)
Each architectural concern in Section 5 is implemented in its own dedicated module:
- [compatibility_layer/models.py](compatibility_layer/models.py) — Immutable request/response types (`RawHTTPRequest`, `RawHTTPResponse`), compatibility enums (`CompatibilityLevel`, `ExecutionPath`, `APIProvider`), routing decisions, and distinct `MiddlewareError` classes.
- [compatibility_layer/interceptor.py](compatibility_layer/interceptor.py) — **Request Interceptor** (`RequestInterceptor`): Classifies incoming requests by destination (`TYPESAFE`, `OPENAI_DECISIONS`, `UNKNOWN`) and inspects JSON safely without mutating raw request bytes.
- [compatibility_layer/capability_resolver.py](compatibility_layer/capability_resolver.py) — **Capability Resolver** (`CapabilityResolver`): Evaluates native capability and cross-API translation capability (`FULL`, `PARTIAL`, `UNSUPPORTED`, `UNKNOWN`) purely from documented operational and structural constraints.
- [compatibility_layer/translator.py](compatibility_layer/translator.py) — **Translation Middleware** (`TranslationMiddleware`): Bidirectionally translates **only** prompt inputs (`state`/`assets`/`questions` ↔ `input`/`questions`) and response `answers`, passing all other parameters (`model`, caching fields, `usage`, custom fields) through **unmodified**.
- [compatibility_layer/adapters.py](compatibility_layer/adapters.py) — **API Adapters** (`TypeSafeAdapter`, `OpenAIDecisionsAdapter`): Communicate with upstream endpoints while preserving raw HTTP status codes, headers, and body bytes on both 2xx and 4xx/5xx responses.
- [compatibility_layer/forwarder.py](compatibility_layer/forwarder.py) — **Transparent Response Forwarder** (`TransparentForwarder`): Forwards native and fallback requests and responses byte-for-byte without parsing, re-serializing, or normalizing.
- [compatibility_layer/diagnostics.py](compatibility_layer/diagnostics.py) — **Diagnostics** (`DiagnosticsRecorder`): Records routing decisions, translation outcomes, and middleware errors while redacting `Authorization` headers, Bearer tokens, and API keys.
- [compatibility_layer/router.py](compatibility_layer/router.py) — **Traffic Router** (`TrafficRouter` & `create_http_proxy_server`): Orchestrates routing, fallback translation, and transparent error handling.

### Routing Decision Flow

```
Incoming RawHTTPRequest
        │
        ▼
[1. Request Interceptor] ──► Identify original_destination (TYPESAFE | OPENAI_DECISIONS | UNKNOWN)
        │
        ▼
[2. Capability Resolver] ──► evaluate_native_capability(request)
        │
        ├─► Level is FULL or UNKNOWN ──► [Transparent Forwarder]
        │                                  Forward raw request unchanged to original API
        │                                  Return raw upstream response/error unchanged
        ▼
Level is UNSUPPORTED
        │
        ▼
[3. Capability Resolver] ──► evaluate_translation_capability(request, alternative_api)
        │
        ├─► Level is FULL (can_execute=True)
        │         │
        │         ▼
        │   [4. Translation Middleware: Request]
        │     Translate prompt inputs (state/assets/questions <-> input/questions)
        │     Pass all other parameters (model, etc.) unmodified
        │         ├─► Local error ──► Report MiddlewareError (origin="compatibility_middleware")
        │         ▼
        │   [5. Alternative API Adapter]
        │         ├─► Transport failure ──► Report UpstreamTransportError (origin="compatibility_middleware")
        │         ├─► Non-2xx status ──► [Transparent Forwarder] Return RAW upstream error unchanged
        │         ▼
        │   [6. Translation Middleware: Response]
        │     Translate answers back to original format; pass other fields unmodified
        │         ├─► Local error (e.g., OpenAI refusal) ──► Report ResponseTranslationError (with raw upstream attached)
        │         ▼
        │   Return translated 2xx response
        │
        └─► Level is PARTIAL, UNSUPPORTED, or UNKNOWN
                  │
                  ▼
            [Transparent Forwarder]
              Forward original request unchanged to original API
              Return raw upstream response/error unchanged
```

---

## 3. Known Limitations and Features That Cannot Be Translated Faithfully

| Feature / Operation | Source API | Target API | Classification | Handling |
|---|---|---|---|---|
| `box` spatial localization primitive | TypeSafe | OpenAI Decisions | `UNSUPPORTED` | Never translated; forwarded unchanged to TypeSafe. |
| `noul` with `criteria: {"true": ..., "false": ...}` | TypeSafe | OpenAI Decisions | `PARTIAL` | OpenAI `predicate` lacks `criteria`; translator refuses to silently discard `criteria`. |
| Structured object/array `instructions` or `choice.criteria` descriptions | TypeSafe | OpenAI Decisions | `PARTIAL` | OpenAI requires string `instructions` (or text/image parts) and string `choice.description`; never silently flattened or discarded. |
| Multipart `part` image references without inline `data_base64` | TypeSafe | OpenAI Decisions | `UNSUPPORTED` | OpenAI Decisions only accepts JSON bodies with `input_image` URLs/data URLs. |
| Remote `http(s)://` `image_url` in `input_image` | OpenAI Decisions | TypeSafe | `UNSUPPORTED` | TypeSafe `assets` requires inline `data_base64` (or multipart `part`) in `png`, `jpg`, or `jpeg`. |
| `prompt_cache_breakpoint` inside `input` parts or `questions[].instructions` | OpenAI Decisions | TypeSafe | `PARTIAL` / `UNSUPPORTED` | Per-part cache breakpoints inside `input` cannot be represented inside TypeSafe `state` strings without dropping the breakpoint property. |
| OpenAI `refusal` answer (`{"type": "refusal", "name": "..."}`) | OpenAI Decisions | TypeSafe | `ResponseTranslationError` | TypeSafe `SystemOneResponse` requires typed answers (`noul`, `choice`, `score`, `box`); middleware refuses to fabricate a score/probability and exposes `ResponseTranslationError` with the raw upstream response attached. |
| `GET /v1/organization/usage/decisions` & `GET /v1/models` | Provider-specific | Alternative API | `UNSUPPORTED` | Provider-specific GET endpoints are never translated across providers. |

---

## 4. Setup Instructions, Configuration, and Example Requests

### Requirements & Running Tests
- Python 3.10+ (uses only the Python standard library; no third-party runtime packages required).
- Set environment variables when communicating with live upstream APIs:
  ```powershell
  $env:TYPESAFE_API_KEY = "ts-..."
  $env:OPENAI_API_KEY = "sk-..."
  ```
- Run the automated test suite:
  ```powershell
  python -m unittest discover -s tests -v
  ```

### Example Usage (In-Process Router & HTTP Proxy)

```python
import json
from compatibility_layer import RawHTTPRequest, TrafficRouter, create_http_proxy_server

router = TrafficRouter()

# 1. Native TypeSafe request (forwarded unchanged to TypeSafe)
native_ts_req = RawHTTPRequest(
    method="POST",
    url="https://api.typesafe.ai/v1/systemone",
    headers={"Content-Type": "application/json"},
    body=json.dumps({
        "model": "jev-latest",
        "state": "My card was charged twice.",
        "questions": {
            "duplicate_charge": {
                "type": "noul",
                "instructions": "Is the customer reporting a duplicate charge?",
            }
        },
    }).encode("utf-8"),
)

# 2. OpenAI Decisions request with >400 questions (automatically translated to TypeSafe and back)
fallback_oa_req = RawHTTPRequest(
    method="POST",
    url="https://api.openai.com/v1/decisions",
    headers={"Content-Type": "application/json"},
    body=json.dumps({
        "model": "gpt-6-luna",
        "input": "Customer report text.",
        "questions": [
            {"type": "predicate", "name": f"q_{i}", "instructions": f"Condition {i}?"}
            for i in range(405)
        ],
    }).encode("utf-8"),
)

# Optional: Run as a local HTTP proxy server on port 8088
# server = create_http_proxy_server(router, host="127.0.0.1", port=8088)
# server.serve_forever()
```

### Future Work (`TODO`)
- `TODO`: Add optional multipart parser to extract raw binary image parts into `data_base64` when translating `multipart/form-data` TypeSafe requests to OpenAI Decisions API.
- `TODO`: Add optional async ASGI/aiohttp transport adapter for high-concurrency streaming proxy deployments.
