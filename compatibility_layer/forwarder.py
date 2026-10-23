"""Transparent Response Forwarder: preserves upstream requests, responses, and errors byte-for-byte."""

from __future__ import annotations

from .adapters import APIAdapter
from .models import RawHTTPRequest, RawHTTPResponse


class TransparentForwarder:
    """Forwards native/fallback requests unchanged and returns upstream responses/errors untouched."""

    def forward_request_unchanged(
        self,
        request: RawHTTPRequest,
        adapter: APIAdapter,
    ) -> RawHTTPResponse:
        """Sends the raw request to the target adapter without parsing, re-serializing, or normalizing."""
        response = adapter.send(request, preserve_exact_request=True)
        return self.forward_response_unchanged(response)

    @staticmethod
    def forward_response_unchanged(response: RawHTTPResponse) -> RawHTTPResponse:
        """Returns the upstream response (whether 2xx or 4xx/5xx error) without interpreting or modifying it."""
        return response
