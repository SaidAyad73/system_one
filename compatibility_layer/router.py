"""Traffic Router: determines execution path and coordinates native forwarding or bidirectional translation."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from .adapters import APIAdapter, OpenAIDecisionsAdapter, TypeSafeAdapter
from .capability_resolver import CapabilityResolver
from .diagnostics import DiagnosticsRecorder
from .forwarder import TransparentForwarder
from .interceptor import RequestInterceptor
from .models import (
    APIProvider,
    CompatibilityLevel,
    ExecutionPath,
    MiddlewareError,
    RawHTTPRequest,
    RawHTTPResponse,
    RoutingDecision,
)
from .translator import TranslationMiddleware


class TrafficRouter:
    """Routes requests between TypeSafe System One and OpenAI Decisions API.

    Algorithm for every request:
    1. Identify original destination via RequestInterceptor.
    2. Evaluate whether the original destination can compute the operation natively via CapabilityResolver.
    3. If native capability is FULL (or UNKNOWN), forward unchanged and return response unchanged.
    4. If native capability is UNSUPPORTED, evaluate whether the alternative API can compute it (FULL).
    5. If the alternative API can compute it, translate request, execute on alternative API, and
       translate a 2xx response back to the original format (returning any upstream error raw).
    6. If neither API appears capable (or translation is PARTIAL/UNSUPPORTED/UNKNOWN), forward the
       original request to its original destination unchanged and return its response/error unchanged.
    """

    def __init__(
        self,
        *,
        typesafe_adapter: Optional[APIAdapter] = None,
        openai_adapter: Optional[APIAdapter] = None,
        interceptor: Optional[RequestInterceptor] = None,
        capability_resolver: Optional[CapabilityResolver] = None,
        translator: Optional[TranslationMiddleware] = None,
        forwarder: Optional[TransparentForwarder] = None,
        diagnostics: Optional[DiagnosticsRecorder] = None,
        default_unknown_destination: APIProvider = APIProvider.TYPESAFE,
    ) -> None:
        self.typesafe_adapter = typesafe_adapter or TypeSafeAdapter()
        self.openai_adapter = openai_adapter or OpenAIDecisionsAdapter()
        self.interceptor = interceptor or RequestInterceptor()
        self.capability_resolver = capability_resolver or CapabilityResolver()
        self.translator = translator or TranslationMiddleware()
        self.forwarder = forwarder or TransparentForwarder()
        self.diagnostics = diagnostics or DiagnosticsRecorder()
        self.default_unknown_destination = default_unknown_destination

    def plan_route(self, request: RawHTTPRequest) -> RoutingDecision:
        """Evaluates a request and returns the RoutingDecision without executing it."""
        classified = self.interceptor.classify(request)
        original = classified.original_destination

        if original == APIProvider.UNKNOWN:
            fallback_target = self.default_unknown_destination
            native_cap = self.capability_resolver.evaluate_native_capability(classified)
            return RoutingDecision(
                original_destination=APIProvider.UNKNOWN,
                target_destination=fallback_target,
                execution_path=ExecutionPath.FALLBACK_UNKNOWN_TO_ORIGINAL,
                requires_translation=False,
                native_capability=native_cap,
                alternative_capability=None,
                rationale="Original destination could not be determined; forwarding unchanged to default destination.",
            )

        native_cap = self.capability_resolver.evaluate_native_capability(classified)

        # Native API can compute the operation
        if native_cap.level == CompatibilityLevel.FULL and native_cap.can_execute:
            exec_path = (
                ExecutionPath.NATIVE_TYPESAFE
                if original == APIProvider.TYPESAFE
                else ExecutionPath.NATIVE_OPENAI
            )
            return RoutingDecision(
                original_destination=original,
                target_destination=original,
                execution_path=exec_path,
                requires_translation=False,
                native_capability=native_cap,
                alternative_capability=None,
                rationale="Original destination natively supports the operation; forwarding unchanged.",
            )

        # Compatibility cannot be determined confidently -> prefer forwarding to original destination unchanged
        if native_cap.level == CompatibilityLevel.UNKNOWN:
            return RoutingDecision(
                original_destination=original,
                target_destination=original,
                execution_path=ExecutionPath.FALLBACK_UNKNOWN_TO_ORIGINAL,
                requires_translation=False,
                native_capability=native_cap,
                alternative_capability=None,
                rationale="Native compatibility is unknown; forwarding unchanged to original destination.",
            )

        # Original destination cannot compute natively -> check alternative API
        alternative = (
            APIProvider.OPENAI_DECISIONS
            if original == APIProvider.TYPESAFE
            else APIProvider.TYPESAFE
        )
        alt_cap = self.capability_resolver.evaluate_translation_capability(classified, alternative)

        if alt_cap.level == CompatibilityLevel.FULL and alt_cap.can_execute:
            exec_path = (
                ExecutionPath.TRANSLATED_TYPESAFE_TO_OPENAI
                if original == APIProvider.TYPESAFE
                else ExecutionPath.TRANSLATED_OPENAI_TO_TYPESAFE
            )
            return RoutingDecision(
                original_destination=original,
                target_destination=alternative,
                execution_path=exec_path,
                requires_translation=True,
                native_capability=native_cap,
                alternative_capability=alt_cap,
                rationale=(
                    f"Original destination ({original.value}) cannot compute the request natively, "
                    f"while alternative ({alternative.value}) supports full translation."
                ),
            )

        # Neither API can handle the operation (or translation is PARTIAL/UNSUPPORTED/UNKNOWN)
        return RoutingDecision(
            original_destination=original,
            target_destination=original,
            execution_path=ExecutionPath.FALLBACK_UNSUPPORTED_TO_ORIGINAL,
            requires_translation=False,
            native_capability=native_cap,
            alternative_capability=alt_cap,
            rationale=(
                f"Neither {original.value} nor {alternative.value} can fully compute the operation; "
                f"forwarding unchanged to original destination ({original.value})."
            ),
        )

    def handle(
        self,
        request: RawHTTPRequest,
        *,
        raise_middleware_errors: bool = False,
    ) -> RawHTTPResponse:
        """Routes and executes the request, returning a RawHTTPResponse.

        - If `raise_middleware_errors=True`, local MiddlewareError exceptions are raised directly.
        - If `raise_middleware_errors=False` (default for HTTP proxy mode), local MiddlewareError
          exceptions are converted into a distinct middleware error HTTP response (`from_upstream=False`)
          that never impersonates an upstream API error.
        """
        decision = self.plan_route(request)

        # Path 1: No translation required (native execution, unknown compatibility, or unsupported by both)
        if not decision.requires_translation:
            adapter = self._get_adapter(decision.target_destination)
            try:
                response = self.forwarder.forward_request_unchanged(request, adapter)
            except MiddlewareError as exc:
                self.diagnostics.record(
                    event_type="middleware_failure",
                    decision=decision,
                    request_headers=request.headers,
                    middleware_error_stage=exc.stage,
                    notes=(exc.message,),
                )
                if raise_middleware_errors:
                    raise
                return exc.to_http_response()

            self.diagnostics.record(
                event_type="forwarded_unchanged",
                decision=decision,
                request_headers=request.headers,
                upstream_status_code=response.status_code,
                notes=(decision.rationale, *decision.native_capability.reasons),
            )
            return response

        # Path 2: Translation required
        try:
            if decision.execution_path == ExecutionPath.TRANSLATED_TYPESAFE_TO_OPENAI:
                translated_req, ctx = self.translator.translate_typesafe_to_openai_request(request)
                alt_adapter = self.openai_adapter
                upstream_resp = alt_adapter.send(translated_req, preserve_exact_request=False)

                # If upstream returns an error status, return the raw upstream error unchanged!
                if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
                    self.diagnostics.record(
                        event_type="translated_upstream_error",
                        decision=decision,
                        request_headers=request.headers,
                        upstream_status_code=upstream_resp.status_code,
                        notes=("Returning raw upstream error from OpenAI Decisions API without translation.",),
                    )
                    return self.forwarder.forward_response_unchanged(upstream_resp)

                final_resp = self.translator.translate_openai_to_typesafe_response(upstream_resp, ctx)
            else:
                translated_req, ctx = self.translator.translate_openai_to_typesafe_request(request)
                alt_adapter = self.typesafe_adapter
                upstream_resp = alt_adapter.send(translated_req, preserve_exact_request=False)

                # If upstream returns an error status, return the raw upstream error unchanged!
                if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
                    self.diagnostics.record(
                        event_type="translated_upstream_error",
                        decision=decision,
                        request_headers=request.headers,
                        upstream_status_code=upstream_resp.status_code,
                        notes=("Returning raw upstream error from TypeSafe API without translation.",),
                    )
                    return self.forwarder.forward_response_unchanged(upstream_resp)

                final_resp = self.translator.translate_typesafe_to_openai_response(upstream_resp, ctx)

        except MiddlewareError as exc:
            self.diagnostics.record(
                event_type="middleware_failure",
                decision=decision,
                request_headers=request.headers,
                upstream_status_code=(
                    exc.raw_upstream_response.status_code if exc.raw_upstream_response is not None else None
                ),
                middleware_error_stage=exc.stage,
                notes=(exc.message,),
            )
            if raise_middleware_errors:
                raise
            return exc.to_http_response()

        self.diagnostics.record(
            event_type="translated_success",
            decision=decision,
            request_headers=request.headers,
            upstream_status_code=final_resp.status_code,
            notes=(decision.rationale,),
        )
        return final_resp

    def _get_adapter(self, provider: APIProvider) -> APIAdapter:
        if provider == APIProvider.OPENAI_DECISIONS:
            return self.openai_adapter
        return self.typesafe_adapter


def create_http_proxy_server(
    router: TrafficRouter,
    host: str = "127.0.0.1",
    port: int = 8088,
) -> HTTPServer:
    """Creates a lightweight standard-library HTTP proxy server backed by TrafficRouter."""

    class ProxyRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._handle_all("GET")

        def do_POST(self) -> None:
            self._handle_all("POST")

        def _handle_all(self, method: str) -> None:
            content_len = int(self.headers.get("Content-Length", "0") or "0")
            raw_body = self.rfile.read(content_len) if content_len > 0 else b""
            req_headers = {k: v for k, v in self.headers.items()}
            raw_req = RawHTTPRequest(
                method=method,
                url=self.path,
                headers=req_headers,
                body=raw_body,
            )
            resp = router.handle(raw_req, raise_middleware_errors=False)
            self.send_response(resp.status_code)
            for hk, hv in resp.headers.items():
                if hk.lower() in ("transfer-encoding", "content-length"):
                    continue
                self.send_header(hk, hv)
            self.send_header("Content-Length", str(len(resp.body)))
            self.end_headers()
            if resp.body:
                self.wfile.write(resp.body)

        def log_message(self, format: str, *args: object) -> None:
            # Suppress default stdout logging to avoid leaking URLs/headers
            return

    return HTTPServer((host, port), ProxyRequestHandler)
