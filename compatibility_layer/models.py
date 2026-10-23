"""Core data structures, enums, and error types for the TypeSafe <-> OpenAI Decisions compatibility layer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional

DATA_URL_RE = re.compile(r"^data:image/([a-zA-Z0-9+.-]+);base64,([A-Za-z0-9+/=\s]+)$")
TYPESAFE_IMAGE_FORMATS: frozenset[str] = frozenset({"png", "jpg", "jpeg"})
TYPESAFE_QUESTION_TYPES: frozenset[str] = frozenset({"noul", "choice", "score", "box"})
OPENAI_QUESTION_TYPES: frozenset[str] = frozenset({"predicate", "choice", "score"})


class APIProvider(str, Enum):
    """Identifies the target or source API provider."""

    TYPESAFE = "typesafe"
    OPENAI_DECISIONS = "openai_decisions"
    UNKNOWN = "unknown"


class CompatibilityLevel(str, Enum):
    """Distinguishes full compatibility, partial compatibility, unsupported operations, and unknown states."""

    FULL = "full"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class ExecutionPath(str, Enum):
    """Routing decision outcomes."""

    NATIVE_TYPESAFE = "native_typesafe"
    NATIVE_OPENAI = "native_openai"
    TRANSLATED_TYPESAFE_TO_OPENAI = "translated_typesafe_to_openai"
    TRANSLATED_OPENAI_TO_TYPESAFE = "translated_openai_to_typesafe"
    FALLBACK_UNSUPPORTED_TO_ORIGINAL = "fallback_unsupported_to_original"
    FALLBACK_UNKNOWN_TO_ORIGINAL = "fallback_unknown_to_original"


@dataclass(frozen=True)
class RawHTTPRequest:
    """Immutable HTTP request representation preserving exact raw bytes and headers."""

    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    destination_hint: Optional[APIProvider] = None

    def get_header(self, name: str, default: Optional[str] = None) -> Optional[str]:
        target = name.lower()
        for k, v in self.headers.items():
            if k.lower() == target:
                return v
        return default


@dataclass(frozen=True)
class RawHTTPResponse:
    """Immutable HTTP response representation preserving exact raw bytes, status code, and headers."""

    status_code: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    from_upstream: bool = True
    upstream_provider: Optional[APIProvider] = None

    def get_header(self, name: str, default: Optional[str] = None) -> Optional[str]:
        target = name.lower()
        for k, v in self.headers.items():
            if k.lower() == target:
                return v
        return default

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


# Injectable callable type for upstream HTTP transport: (RawHTTPRequest) -> RawHTTPResponse
TransportCallable = Callable[[RawHTTPRequest], RawHTTPResponse]


@dataclass(frozen=True)
class ClassifiedRequest:
    """Result produced by the Request Interceptor without mutating the raw request."""

    raw_request: RawHTTPRequest
    original_destination: APIProvider
    path: str
    query: str
    content_type: str
    is_json: bool
    is_multipart: bool
    parsed_json: Optional[Any] = None
    json_parse_error: Optional[str] = None


@dataclass(frozen=True)
class CapabilityReport:
    """Detailed capability evaluation for a single target API."""

    provider: APIProvider
    level: CompatibilityLevel
    can_execute: bool
    reasons: tuple[str, ...] = ()
    unsupported_operations: tuple[str, ...] = ()
    unsupported_parameters: tuple[str, ...] = ()
    constraints_violated: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoutingDecision:
    """Complete routing decision combining native and alternative capability evaluations."""

    original_destination: APIProvider
    target_destination: APIProvider
    execution_path: ExecutionPath
    requires_translation: bool
    native_capability: CapabilityReport
    alternative_capability: Optional[CapabilityReport] = None
    rationale: str = ""


class MiddlewareError(Exception):
    """Base error for failures originating inside the compatibility middleware itself.

    Explicitly separated from upstream API responses so callers never mistake a local
    middleware failure for an upstream TypeSafe or OpenAI error.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        unsupported_parameters: tuple[str, ...] = (),
        unsupported_operations: tuple[str, ...] = (),
        raw_upstream_response: Optional[RawHTTPResponse] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.unsupported_parameters = tuple(unsupported_parameters)
        self.unsupported_operations = tuple(unsupported_operations)
        self.raw_upstream_response = raw_upstream_response
        self.cause = cause

    def to_http_response(self, status_code: int = 502) -> RawHTTPResponse:
        payload: dict[str, Any] = {
            "middleware_error": {
                "origin": "compatibility_middleware",
                "stage": self.stage,
                "type": self.__class__.__name__,
                "message": self.message,
            }
        }
        if self.unsupported_parameters:
            payload["middleware_error"]["unsupported_parameters"] = list(self.unsupported_parameters)
        if self.unsupported_operations:
            payload["middleware_error"]["unsupported_operations"] = list(self.unsupported_operations)
        if self.raw_upstream_response is not None:
            payload["middleware_error"]["upstream_status_code"] = self.raw_upstream_response.status_code
            try:
                payload["middleware_error"]["raw_upstream_body"] = self.raw_upstream_response.body.decode("utf-8")
            except UnicodeDecodeError:
                payload["middleware_error"]["raw_upstream_body_hex"] = self.raw_upstream_response.body.hex()

        body_bytes = json.dumps(payload).encode("utf-8")
        return RawHTTPResponse(
            status_code=status_code,
            headers={
                "Content-Type": "application/json",
                "X-Compatibility-Error-Origin": "middleware",
                "X-Compatibility-Error-Stage": self.stage,
            },
            body=body_bytes,
            from_upstream=False,
            upstream_provider=None,
        )


class RequestTranslationError(MiddlewareError):
    """Raised when a request cannot be faithfully translated without dropping or fabricating fields."""

    def __init__(
        self,
        message: str,
        *,
        unsupported_parameters: tuple[str, ...] = (),
        unsupported_operations: tuple[str, ...] = (),
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            stage="request_translation",
            unsupported_parameters=unsupported_parameters,
            unsupported_operations=unsupported_operations,
            cause=cause,
        )


class ResponseTranslationError(MiddlewareError):
    """Raised when a 2xx upstream response cannot be faithfully translated into the caller's expected format."""

    def __init__(
        self,
        message: str,
        *,
        raw_upstream_response: Optional[RawHTTPResponse] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            stage="response_translation",
            raw_upstream_response=raw_upstream_response,
            cause=cause,
        )


class UpstreamTransportError(MiddlewareError):
    """Raised when the local HTTP transport fails before any upstream HTTP response is received."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None) -> None:
        super().__init__(message, stage="upstream_transport", cause=cause)


@dataclass(frozen=True)
class DiagnosticEvent:
    """Sanitized observability event recording routing, capability, and translation outcomes."""

    event_type: str
    original_destination: APIProvider
    target_destination: APIProvider
    execution_path: ExecutionPath
    requires_translation: bool
    native_compatibility: CompatibilityLevel
    alternative_compatibility: Optional[CompatibilityLevel]
    upstream_status_code: Optional[int] = None
    middleware_error_stage: Optional[str] = None
    notes: tuple[str, ...] = ()
    sanitized_headers: Mapping[str, str] = field(default_factory=dict)
