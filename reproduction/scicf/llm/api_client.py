"""Dependency-free client for a private OpenAI-compatible chat API."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


Transport = Callable[
    [urllib.request.Request, float], Tuple[int, Mapping[str, str], bytes]
]

_ALLOWED_CREDENTIAL_KEYS = {
    "SCICF_LLM_API_URL",
    "SCICF_LLM_API_KEY",
    "SCICF_LLM_MODEL_ID",
    "SCICF_LLM_MODEL_REVISION",
    "SCICF_LLM_PROVIDER_ID",
    "SCICF_LLM_API_KEY_HEADER",
    "SCICF_LLM_API_KEY_PREFIX",
    "SCICF_LLM_ALLOW_INSECURE_HTTP",
    "SCICF_LLM_INCLUDE_SEED",
    "SCICF_LLM_JSON_MODE",
    "SCICF_LLM_MAX_TOKENS_FIELD",
    "SCICF_LLM_THINKING",
}
_REQUIRED_CREDENTIAL_KEYS = {
    "SCICF_LLM_API_URL",
    "SCICF_LLM_API_KEY",
    "SCICF_LLM_MODEL_ID",
    "SCICF_LLM_MODEL_REVISION",
    "SCICF_LLM_PROVIDER_ID",
}
_HEADER_NAME = re.compile(r"^[A-Za-z0-9-]+$")


class APITransportError(RuntimeError):
    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def _boolean(value: str, key: str, default: bool) -> bool:
    if value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError("{} must be true or false".format(key))


def _read_credentials_file(path: Path) -> Dict[str, str]:
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("API credentials path must be a regular file")
    if metadata.st_uid != os.getuid():
        raise ValueError("API credentials file must be owned by the Slurm user")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("API credentials file permissions must be 0600 or stricter")
    if metadata.st_size > 65536:
        raise ValueError("API credentials file is unexpectedly large")

    values: Dict[str, str] = {}
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                "invalid credentials line {}; expected KEY=VALUE".format(line_number)
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in _ALLOWED_CREDENTIAL_KEYS:
            raise ValueError("unsupported credentials key: {}".format(key))
        if key in values:
            raise ValueError("duplicate credentials key: {}".format(key))
        if "\r" in value or "\n" in value:
            raise ValueError("credential values cannot contain newlines")
        values[key] = value
    missing = sorted(key for key in _REQUIRED_CREDENTIAL_KEYS if not values.get(key))
    if missing:
        raise ValueError("missing API settings: {}".format(", ".join(missing)))
    return values


@dataclass(frozen=True)
class APISettings:
    endpoint: str
    api_key: str
    model_id: str
    model_revision: str
    provider_id: str
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    allow_insecure_http: bool = False
    include_seed: bool = True
    json_mode: bool = False
    max_tokens_field: str = "max_tokens"
    thinking: str = "omit"

    @classmethod
    def from_private_file(cls, path: Path) -> "APISettings":
        values = _read_credentials_file(path)
        settings = cls(
            endpoint=values["SCICF_LLM_API_URL"],
            api_key=values["SCICF_LLM_API_KEY"],
            model_id=values["SCICF_LLM_MODEL_ID"],
            model_revision=values["SCICF_LLM_MODEL_REVISION"],
            provider_id=values["SCICF_LLM_PROVIDER_ID"],
            api_key_header=values.get("SCICF_LLM_API_KEY_HEADER", "Authorization"),
            api_key_prefix=values.get("SCICF_LLM_API_KEY_PREFIX", "Bearer"),
            allow_insecure_http=_boolean(
                values.get("SCICF_LLM_ALLOW_INSECURE_HTTP", ""),
                "SCICF_LLM_ALLOW_INSECURE_HTTP",
                False,
            ),
            include_seed=_boolean(
                values.get("SCICF_LLM_INCLUDE_SEED", ""),
                "SCICF_LLM_INCLUDE_SEED",
                True,
            ),
            json_mode=_boolean(
                values.get("SCICF_LLM_JSON_MODE", ""),
                "SCICF_LLM_JSON_MODE",
                False,
            ),
            max_tokens_field=values.get(
                "SCICF_LLM_MAX_TOKENS_FIELD", "max_tokens"
            ),
            thinking=values.get("SCICF_LLM_THINKING", "omit").strip().lower(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("SCICF_LLM_API_URL must be an absolute HTTP(S) URL")
        if parsed.scheme != "https" and not self.allow_insecure_http:
            raise ValueError(
                "plain HTTP is disabled; explicitly set "
                "SCICF_LLM_ALLOW_INSECURE_HTTP=true only for a trusted private network"
            )
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("API URL cannot contain credentials, query, or fragment")
        if not _HEADER_NAME.fullmatch(self.api_key_header):
            raise ValueError("invalid API key header name")
        if "\r" in self.api_key_prefix or "\n" in self.api_key_prefix:
            raise ValueError("API key prefix cannot contain newlines")
        if self.max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("unsupported max-token request field")
        if self.thinking not in {"omit", "enabled", "disabled"}:
            raise ValueError("SCICF_LLM_THINKING must be omit, enabled, or disabled")
        for name, value in (
            ("api_key", self.api_key),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("provider_id", self.provider_id),
        ):
            if not value:
                raise ValueError("{} is required".format(name))

    def public_identity(self) -> Dict[str, Any]:
        return {
            "type": "openai-compatible-api",
            "provider_id": self.provider_id,
            "endpoint_sha256": hashlib.sha256(
                self.endpoint.encode("utf-8")
            ).hexdigest(),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "include_seed": self.include_seed,
            "json_mode": self.json_mode,
            "max_tokens_field": self.max_tokens_field,
            "thinking": self.thinking,
        }


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    provider_response_id: Optional[str]
    system_fingerprint: Optional[str]
    response_sha256: str
    transport_retries_used: int


def _default_transport(
    request: urllib.request.Request, timeout: float
) -> Tuple[int, Mapping[str, str], bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (
                int(response.getcode()),
                dict(response.headers.items()),
                response.read(),
            )
    except urllib.error.HTTPError as error:
        retryable = error.code in {408, 409, 425, 429} or error.code >= 500
        raise APITransportError(
            "API returned HTTP {}".format(error.code), retryable=retryable
        )
    except urllib.error.URLError as error:
        raise APITransportError(
            "API network request failed: {}".format(type(error.reason).__name__),
            retryable=True,
        )


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("API usage token fields must be non-negative integers")
    return value


class OpenAICompatibleClient:
    def __init__(
        self,
        settings: APISettings,
        transport: Transport = _default_transport,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.transport = transport
        self.sleeper = sleeper

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
        seed: int,
        timeout_seconds: float,
        transport_retries: int,
    ) -> ChatCompletion:
        payload: Dict[str, Any] = {
            "model": self.settings.model_id,
            "messages": [dict(item) for item in messages],
            "temperature": 0.0,
            self.settings.max_tokens_field: max_tokens,
        }
        if self.settings.include_seed:
            payload["seed"] = seed
        if self.settings.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.settings.thinking != "omit":
            payload["thinking"] = {"type": self.settings.thinking}
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        authentication = self.settings.api_key
        if self.settings.api_key_prefix:
            authentication = "{} {}".format(
                self.settings.api_key_prefix, self.settings.api_key
            )
        request = urllib.request.Request(
            self.settings.endpoint,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                self.settings.api_key_header: authentication,
            },
            method="POST",
        )

        response_body = b""
        retries_used = 0
        for attempt in range(transport_retries + 1):
            try:
                status_code, _, response_body = self.transport(
                    request, timeout_seconds
                )
                if status_code < 200 or status_code >= 300:
                    raise APITransportError(
                        "API returned HTTP {}".format(status_code),
                        retryable=status_code in {408, 409, 425, 429}
                        or status_code >= 500,
                    )
                retries_used = attempt
                break
            except APITransportError as error:
                if not error.retryable or attempt >= transport_retries:
                    raise
                self.sleeper(min(float(2**attempt), 8.0))
        try:
            envelope = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("API response is not valid UTF-8 JSON") from error
        if not isinstance(envelope, dict):
            raise ValueError("API response must be a JSON object")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("API response has no choices")
        first = choices[0]
        if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
            raise ValueError("API response choice has no message object")
        content = first["message"].get("content")
        if not isinstance(content, str):
            raise ValueError("API response message content must be a string")
        usage = envelope.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("API usage must be an object when present")
        provider_response_id = envelope.get("id")
        system_fingerprint = envelope.get("system_fingerprint")
        return ChatCompletion(
            content=content,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
            provider_response_id=(
                provider_response_id if isinstance(provider_response_id, str) else None
            ),
            system_fingerprint=(
                system_fingerprint if isinstance(system_fingerprint, str) else None
            ),
            response_sha256=hashlib.sha256(response_body).hexdigest(),
            transport_retries_used=retries_used,
        )
