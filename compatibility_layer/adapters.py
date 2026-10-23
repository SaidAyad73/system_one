"""API Adapters: communicate with upstream TypeSafe and OpenAI Decisions APIs while preserving raw responses."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urljoin, urlsplit

from .models import (
    APIProvider,
    RawHTTPRequest,
    RawHTTPResponse,
    TransportCallable,
    UpstreamTransportError,
)


def default_urllib_transport(request: RawHTTPRequest, *, timeout: float = 30.0) -> RawHTTPResponse:
    """Default standard-library HTTP transport that preserves raw status codes, headers, and body bytes."""
    req = urllib.request.Request(
        url=request.url,
        data=request.body if request.body else None,
        headers=dict(request.headers),
        method=request.method.upper(),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read()
            resp_headers = {k: v for k, v in resp.headers.items()}
            return RawHTTPResponse(
                status_code=resp.status,
                headers=resp_headers,
                body=raw_body,
                from_upstream=True,
                upstream_provider=request.destination_hint,
            )
    except urllib.error.HTTPError as http_err:
        # Upstream returned a 4xx/5xx HTTP response: preserve status code, headers, and raw body unchanged.
        raw_err_body = http_err.read() if http_err.fp is not None else b""
        err_headers = {k: v for k, v in http_err.headers.items()} if http_err.headers else {}
        return RawHTTPResponse(
            status_code=http_err.code,
            headers=err_headers,
            body=raw_err_body,
            from_upstream=True,
            upstream_provider=request.destination_hint,
        )
    except Exception as exc:
        raise UpstreamTransportError(
            f"Transport failure communicating with upstream '{request.url}': {exc}",
            cause=exc,
        ) from exc


class APIAdapter:
    """Base adapter for an upstream API provider."""

    def __init__(
        self,
        *,
        provider: APIProvider,
        base_url: str,
        api_key: Optional[str] = None,
        transport: Optional[TransportCallable] = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._transport: TransportCallable = transport or default_urllib_transport

    def send(self, request: RawHTTPRequest, *, preserve_exact_request: bool = False) -> RawHTTPResponse:
        """Dispatches a request to the upstream API.

        When `preserve_exact_request=True`, the request body and headers are sent without modification
        (only resolving a relative path against `base_url` if the URL has no scheme/host, and injecting
        a configured `Authorization` header only if the incoming request omitted one).
        """
        resolved_url = self._resolve_url(request.url)
        headers = dict(request.headers)
        if self.api_key and request.get_header("Authorization") is None:
            headers["Authorization"] = f"Bearer {self.api_key}"

        outbound = RawHTTPRequest(
            method=request.method,
            url=resolved_url,
            headers=headers if not preserve_exact_request or len(headers) != len(request.headers) else request.headers,
            body=request.body,
            destination_hint=self.provider,
        )
        try:
            response = self._transport(outbound)
        except UpstreamTransportError:
            raise
        except Exception as exc:
            raise UpstreamTransportError(
                f"Upstream transport failed for {self.provider.value}: {exc}",
                cause=exc,
            ) from exc

        return RawHTTPResponse(
            status_code=response.status_code,
            headers=response.headers,
            body=response.body,
            from_upstream=response.from_upstream,
            upstream_provider=self.provider,
        )

    def _resolve_url(self, url: str) -> str:
        split = urlsplit(url)
        if split.scheme and split.netloc:
            return url
        path = url if url.startswith("/") else f"/{url}"
        return urljoin(f"{self.base_url}/", path.lstrip("/"))


class TypeSafeAdapter(APIAdapter):
    """Adapter for the TypeSafe System One API (https://api.typesafe.ai)."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.typesafe.ai",
        api_key: Optional[str] = None,
        transport: Optional[TransportCallable] = None,
    ) -> None:
        super().__init__(
            provider=APIProvider.TYPESAFE,
            base_url=base_url,
            api_key=api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY"),
            transport=transport,
        )


class OpenAIDecisionsAdapter(APIAdapter):
    """Adapter for the OpenAI Decisions API (https://api.openai.com)."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com",
        api_key: Optional[str] = None,
        transport: Optional[TransportCallable] = None,
    ) -> None:
        super().__init__(
            provider=APIProvider.OPENAI_DECISIONS,
            base_url=base_url,
            api_key=api_key if api_key is not None else os.environ.get("OPENAI_API_KEY"),
            transport=transport,
        )
