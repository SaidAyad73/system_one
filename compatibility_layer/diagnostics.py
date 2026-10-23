"""Diagnostics: records routing and translation decisions without modifying traffic or leaking secrets."""

from __future__ import annotations

import re
from typing import Callable, Mapping, Optional

from .models import (
    CompatibilityLevel,
    DiagnosticEvent,
    RoutingDecision,
)

_SENSITIVE_HEADER_NAMES = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "openai-api-key",
    "typesafe-api-key",
    "cookie",
    "set-cookie",
}

_BEARER_TOKEN_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-+/=]+", re.IGNORECASE)
_SK_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{6,}|ts-[A-Za-z0-9_\-]{6,})\b")


def sanitize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Redacts credentials and authorization headers for safe observability."""
    sanitized: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _SENSITIVE_HEADER_NAMES or "key" in k.lower() or "token" in k.lower() or "secret" in k.lower():
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = sanitize_text(v)
    return sanitized


def sanitize_text(text: str) -> str:
    """Scrubs Bearer tokens and API key patterns from diagnostic strings."""
    scrubbed = _BEARER_TOKEN_RE.sub(r"\1[REDACTED]", text)
    return _SK_KEY_RE.sub("[REDACTED_KEY]", scrubbed)


class DiagnosticsRecorder:
    """Collects sanitized diagnostic events without mutating requests or responses."""

    def __init__(self, sink: Optional[Callable[[DiagnosticEvent], None]] = None) -> None:
        self._events: list[DiagnosticEvent] = []
        self._sink = sink

    @property
    def events(self) -> tuple[DiagnosticEvent, ...]:
        return tuple(self._events)

    def clear(self) -> None:
        self._events.clear()

    def record(
        self,
        *,
        event_type: str,
        decision: RoutingDecision,
        request_headers: Mapping[str, str],
        upstream_status_code: Optional[int] = None,
        middleware_error_stage: Optional[str] = None,
        notes: tuple[str, ...] = (),
    ) -> DiagnosticEvent:
        alt_level: Optional[CompatibilityLevel] = (
            decision.alternative_capability.level if decision.alternative_capability is not None else None
        )
        safe_notes = tuple(sanitize_text(n) for n in notes)
        event = DiagnosticEvent(
            event_type=event_type,
            original_destination=decision.original_destination,
            target_destination=decision.target_destination,
            execution_path=decision.execution_path,
            requires_translation=decision.requires_translation,
            native_compatibility=decision.native_capability.level,
            alternative_compatibility=alt_level,
            upstream_status_code=upstream_status_code,
            middleware_error_stage=middleware_error_stage,
            notes=safe_notes,
            sanitized_headers=sanitize_headers(request_headers),
        )
        self._events.append(event)
        if self._sink is not None:
            self._sink(event)
        return event
