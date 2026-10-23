"""Public interface for the TypeSafe <-> OpenAI Decisions API Compatibility Layer."""

from .adapters import (
    APIAdapter,
    OpenAIDecisionsAdapter,
    TypeSafeAdapter,
    default_urllib_transport,
)
from .capability_resolver import CapabilityResolver
from .diagnostics import DiagnosticsRecorder, sanitize_headers, sanitize_text
from .forwarder import TransparentForwarder
from .interceptor import RequestInterceptor, classify_request
from .models import (
    APIProvider,
    CapabilityReport,
    ClassifiedRequest,
    CompatibilityLevel,
    DiagnosticEvent,
    ExecutionPath,
    MiddlewareError,
    RawHTTPRequest,
    RawHTTPResponse,
    RequestTranslationError,
    ResponseTranslationError,
    RoutingDecision,
    TransportCallable,
    UpstreamTransportError,
)
from .router import TrafficRouter, create_http_proxy_server
from .translator import (
    QuestionTranslationMeta,
    TranslationContext,
    TranslationMiddleware,
)

__all__ = [
    "APIAdapter",
    "APIProvider",
    "CapabilityReport",
    "CapabilityResolver",
    "ClassifiedRequest",
    "CompatibilityLevel",
    "DiagnosticEvent",
    "DiagnosticsRecorder",
    "ExecutionPath",
    "MiddlewareError",
    "OpenAIDecisionsAdapter",
    "QuestionTranslationMeta",
    "RawHTTPRequest",
    "RawHTTPResponse",
    "RequestInterceptor",
    "RequestTranslationError",
    "ResponseTranslationError",
    "RoutingDecision",
    "TrafficRouter",
    "TranslationContext",
    "TranslationMiddleware",
    "TransparentForwarder",
    "TransportCallable",
    "TypeSafeAdapter",
    "UpstreamTransportError",
    "classify_request",
    "create_http_proxy_server",
    "default_urllib_transport",
    "sanitize_headers",
    "sanitize_text",
]
