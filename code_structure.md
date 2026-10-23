# Code Structure & Concerns Overview

```text
system_one/
├── README.md
├── code_structure.md
├── compatibility_layer/
│   ├── __init__.py
│   ├── models.py
│   ├── interceptor.py
│   ├── capability_resolver.py
│   ├── translator.py
│   ├── adapters.py
│   ├── forwarder.py
│   ├── diagnostics.py
│   └── router.py
└── tests/
    └── test_compatibility_layer.py
```

## Module Responsibilities and Core Concerns

- **[compatibility_layer/models.py](compatibility_layer/models.py)**
  - **Responsibility**: Defines immutable value objects, enums, and the middleware error hierarchy.
  - **Core Concerns**: `APIProvider`, `CompatibilityLevel` (`FULL`, `PARTIAL`, `UNSUPPORTED`, `UNKNOWN`), `ExecutionPath`, `RawHTTPRequest`, `RawHTTPResponse`, `ClassifiedRequest`, `CapabilityReport`, `RoutingDecision`, `DiagnosticEvent`, and `MiddlewareError` (`RequestTranslationError`, `ResponseTranslationError`, `UpstreamTransportError`).

- **[compatibility_layer/interceptor.py](compatibility_layer/interceptor.py)**
  - **Responsibility**: Request Interceptor that classifies incoming HTTP traffic without mutating raw bytes.
  - **Core Concerns**: Destination identification (`TYPESAFE`, `OPENAI_DECISIONS`, `UNKNOWN`) via explicit hints, URL host/path, headers, or JSON body shape; safe non-mutating JSON inspection.

- **[compatibility_layer/capability_resolver.py](compatibility_layer/capability_resolver.py)**
  - **Responsibility**: Capability Resolver that evaluates native execution capability and cross-API translation capability from documented constraints.
  - **Core Concerns**: Evaluates endpoints, input/evidence structures (`state`/`assets` vs `input`/`input_json`/`input_image`/tool `call_id` matching), question primitives (`noul`/`predicate`, `choice`, `score`, `box`), and cardinality bounds (`1..400` questions, `1..255` choices, `1..10` vs `2..10` score levels). Never mutates or blocks on `model` or non-prompt parameters.

- **[compatibility_layer/translator.py](compatibility_layer/translator.py)**
  - **Responsibility**: Translation Middleware performing bidirectional translation between TypeSafe and OpenAI Decisions schemas.
  - **Core Concerns**: Translates only prompt inputs (`state`/`assets`/`questions` ↔ `input`/`questions`) and response `answers`, passing all other parameters (`model`, `usage`, caching fields, custom attributes) through unmodified. Refuses to silently drop unsupported fields (`noul.criteria`, `prompt_cache_breakpoint` in parts, `box`) or fabricate answers (OpenAI `refusal`).

- **[compatibility_layer/adapters.py](compatibility_layer/adapters.py)**
  - **Responsibility**: Upstream HTTP adapters for TypeSafe (`https://api.typesafe.ai`) and OpenAI Decisions (`https://api.openai.com`).
  - **Core Concerns**: Injectable `TransportCallable`, URL resolution, optional API key injection when omitted by caller, and raw status/header/body preservation on both 2xx and 4xx/5xx responses.

- **[compatibility_layer/forwarder.py](compatibility_layer/forwarder.py)**
  - **Responsibility**: Transparent Response Forwarder.
  - **Core Concerns**: Forwards native and fallback requests and upstream responses/errors byte-for-byte without parsing, re-serializing, or normalizing.

- **[compatibility_layer/diagnostics.py](compatibility_layer/diagnostics.py)**
  - **Responsibility**: Observability and diagnostic event recording.
  - **Core Concerns**: Records routing, capability, translation, and error telemetry while redacting `Authorization` headers, Bearer tokens, and API keys (`sk-*`, `ts-*`).

- **[compatibility_layer/router.py](compatibility_layer/router.py)**
  - **Responsibility**: Traffic Router and optional local HTTP proxy server.
  - **Core Concerns**: Orchestrates the 6-step routing flow (`NATIVE_TYPESAFE`, `NATIVE_OPENAI`, `TRANSLATED_TYPESAFE_TO_OPENAI`, `TRANSLATED_OPENAI_TO_TYPESAFE`, `FALLBACK_UNSUPPORTED_TO_ORIGINAL`, `FALLBACK_UNKNOWN_TO_ORIGINAL`).

- **[compatibility_layer/__init__.py](compatibility_layer/__init__.py)**
  - **Responsibility**: Public package interface exporting all components and types.

- **[tests/test_compatibility_layer.py](tests/test_compatibility_layer.py)**
  - **Responsibility**: Automated unit and integration test suite covering all 14 required validation scenarios.
