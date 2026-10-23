"""Request Interceptor: receives and classifies incoming HTTP requests without mutating raw bytes."""

from __future__ import annotations

import json
from typing import Any, Optional
from urllib.parse import urlsplit

from .models import APIProvider, ClassifiedRequest, RawHTTPRequest


class RequestInterceptor:
    """Inspects an incoming RawHTTPRequest to determine its original destination and structural shape."""

    def classify(self, request: RawHTTPRequest) -> ClassifiedRequest:
        split = urlsplit(request.url)
        host = (split.hostname or "").lower()
        path = split.path or "/"
        query = split.query or ""

        content_type_raw = request.get_header("Content-Type", "") or ""
        content_type = content_type_raw.split(";")[0].strip().lower()
        is_multipart = content_type.startswith("multipart/form-data")
        is_json = content_type in ("application/json", "") or content_type.endswith("+json")

        parsed_json: Optional[Any] = None
        json_parse_error: Optional[str] = None

        if request.body and not is_multipart:
            try:
                parsed_json = json.loads(request.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                json_parse_error = str(exc)

        destination = self._resolve_destination(
            request=request,
            host=host,
            path=path,
            parsed_json=parsed_json,
        )

        return ClassifiedRequest(
            raw_request=request,
            original_destination=destination,
            path=path,
            query=query,
            content_type=content_type,
            is_json=is_json and json_parse_error is None,
            is_multipart=is_multipart,
            parsed_json=parsed_json,
            json_parse_error=json_parse_error,
        )

    def _resolve_destination(
        self,
        *,
        request: RawHTTPRequest,
        host: str,
        path: str,
        parsed_json: Optional[Any],
    ) -> APIProvider:
        if request.destination_hint is not None and request.destination_hint != APIProvider.UNKNOWN:
            return request.destination_hint

        if "typesafe.ai" in host:
            return APIProvider.TYPESAFE
        if "openai.com" in host:
            return APIProvider.OPENAI_DECISIONS

        normalized_path = path.rstrip("/") or "/"
        if normalized_path in ("/v1/systemone", "/v1/models"):
            return APIProvider.TYPESAFE
        if normalized_path in ("/v1/decisions", "/v1/organization/usage/decisions"):
            return APIProvider.OPENAI_DECISIONS

        for header_name in request.headers:
            if header_name.lower().startswith("typesafe-"):
                return APIProvider.TYPESAFE

        if isinstance(parsed_json, dict):
            if "state" in parsed_json and isinstance(parsed_json.get("questions"), dict):
                return APIProvider.TYPESAFE
            if "input" in parsed_json and isinstance(parsed_json.get("questions"), list):
                return APIProvider.OPENAI_DECISIONS

        return APIProvider.UNKNOWN


def classify_request(request: RawHTTPRequest) -> ClassifiedRequest:
    """Convenience function wrapping RequestInterceptor.classify."""
    return RequestInterceptor().classify(request)
