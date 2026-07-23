"""Minimal OpenRouter chat-completions adapter over httpx.

Owns request construction, an httpx timeout budget, typed errors, usage
extraction, and prompt caching (``cache_control`` on the stable system prefix).
The provider JSON shape stays inside this module; consumers depend on the typed
``Prompt`` / ``Usage`` / error surface, never on httpx or the raw response.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV = "OPENROUTER_MODEL"


class OpenRouterError(RuntimeError):
    """Base class for adapter failures; consumers can catch this for any of them."""


class OpenRouterConfigurationError(OpenRouterError):
    """Raised when the adapter cannot be configured from the environment."""


class OpenRouterRequestError(OpenRouterError):
    """Raised when the HTTP request to OpenRouter fails."""


class OpenRouterResponseError(OpenRouterError):
    """Raised when OpenRouter returns an unexpected response shape."""


@dataclass(frozen=True)
class Prompt:
    """A system/user pair; ``system`` is the cacheable stable prefix."""

    system: str
    user: str


@dataclass(frozen=True)
class Usage:
    """Adapter-owned per-call usage summary, normalized from the provider block."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost: float | None = None


@dataclass(frozen=True)
class OpenRouterConfig:
    """Resolved adapter settings.

    ``provider_preferences`` is passed through verbatim as the request body's
    ``provider`` field (OpenRouter provider routing: ``order``, ``only``,
    ``ignore``, ``allow_fallbacks``, ``sort``, ...). The mapping is untyped on
    purpose — routing fields belong to OpenRouter's schema, not this adapter —
    and ``None`` (the default) leaves request bodies byte-identical to
    previous releases.

    ``max_tokens`` is the default output cap sent as the request body's
    ``max_tokens`` field; individual calls can override it. ``None`` (the
    default) omits the field entirely.
    """

    api_key: str
    model: str = DEFAULT_MODEL
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    provider_preferences: dict[str, Any] | None = None
    max_tokens: int | None = None


def load_config(
    *,
    api_key: str | None = None,
    model: str | None = None,
    provider_preferences: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> OpenRouterConfig:
    """Resolve configuration from explicit values or the environment."""

    resolved_key = api_key or os.getenv(OPENROUTER_API_KEY_ENV)
    if not resolved_key:
        raise OpenRouterConfigurationError(f"{OPENROUTER_API_KEY_ENV} is required")
    resolved_model = model or os.getenv(OPENROUTER_MODEL_ENV) or DEFAULT_MODEL
    return OpenRouterConfig(
        api_key=resolved_key,
        model=resolved_model,
        provider_preferences=provider_preferences,
        max_tokens=max_tokens,
    )


class OpenRouterClient:
    """Callable adapter: ``client(prompt) -> str``, recording the last call's usage.

    After each call, ``last_usage`` holds the normalized :class:`Usage` and
    ``last_raw_usage`` holds the provider's raw ``usage`` block (for cache
    verification). Both reset to ``None`` at the start of every call.
    """

    def __init__(self, config: OpenRouterConfig, *, http_client: Any | None = None) -> None:
        self._config = config
        self._http_client = http_client
        self.last_usage: Usage | None = None
        self.last_raw_usage: dict[str, Any] | None = None

    @property
    def config(self) -> OpenRouterConfig:
        return self._config

    def __call__(self, prompt: Prompt, *, max_tokens: int | None = None) -> str:
        self.last_usage = None
        self.last_raw_usage = None
        payload = self._post(prompt, max_tokens=max_tokens)
        content = _extract_content(payload)
        self.last_usage = _extract_usage(payload)
        raw_usage = payload.get("usage")
        self.last_raw_usage = raw_usage if isinstance(raw_usage, dict) else None
        return content

    def stream(self, prompt: Prompt, *, max_tokens: int | None = None) -> Iterator[str]:
        self.last_usage = None
        self.last_raw_usage = None
        body = self._request_body(prompt, stream=True, max_tokens=max_tokens)
        headers = self._headers()
        if self._http_client is None:
            timeout = httpx.Timeout(self._config.request_timeout_seconds)
            with httpx.Client(timeout=timeout) as http_client:
                yield from self._stream(http_client, headers, body)
            return
        yield from self._stream(self._http_client, headers, body)

    def _post(self, prompt: Prompt, *, max_tokens: int | None = None) -> dict[str, Any]:
        body = self._request_body(prompt, max_tokens=max_tokens)
        headers = self._headers()
        if self._http_client is None:
            timeout = httpx.Timeout(self._config.request_timeout_seconds)
            with httpx.Client(timeout=timeout) as http_client:
                return _request(http_client, headers, body)
        return _request(self._http_client, headers, body)

    def _request_body(
        self,
        prompt: Prompt,
        *,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": _messages_for_prompt(prompt),
            # Opt in to OpenRouter usage accounting so the response carries cost
            # and the cache-token breakdown; without it both come back empty.
            "usage": {"include": True},
        }
        if self._config.provider_preferences is not None:
            body["provider"] = dict(self._config.provider_preferences)
        resolved_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        if resolved_max_tokens is not None:
            body["max_tokens"] = resolved_max_tokens
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

    def _stream(self, client: Any, headers: dict[str, str], body: dict[str, Any]) -> Iterator[str]:
        terminal_payload: dict[str, Any] | None = None
        try:
            with client.stream(
                "POST",
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers=headers,
                json=body,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except ValueError as exc:
                        raise OpenRouterResponseError("OpenRouter returned invalid JSON") from exc
                    if not isinstance(payload, dict):
                        raise OpenRouterResponseError("OpenRouter returned a non-object payload")
                    terminal_payload = payload
                    raw_usage = payload.get("usage")
                    if isinstance(raw_usage, dict):
                        self.last_usage = _extract_usage(payload)
                        self.last_raw_usage = raw_usage
                    content = _extract_stream_delta(payload)
                    if content is not None:
                        yield content
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            raise OpenRouterRequestError("OpenRouter request failed") from exc
        if terminal_payload is not None:
            raw_usage = terminal_payload.get("usage")
            if isinstance(raw_usage, dict):
                self.last_usage = _extract_usage(terminal_payload)
                self.last_raw_usage = raw_usage
        if self.last_usage is None:
            self.last_usage = Usage()


def build_client(config: OpenRouterConfig, *, http_client: Any | None = None) -> OpenRouterClient:
    """Build a client from an explicit config."""

    return OpenRouterClient(config, http_client=http_client)


def load_client(
    *,
    api_key: str | None = None,
    model: str | None = None,
    http_client: Any | None = None,
    provider_preferences: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> OpenRouterClient:
    """Build a client from the environment (or explicit overrides)."""

    return build_client(
        load_config(
            api_key=api_key,
            model=model,
            provider_preferences=provider_preferences,
            max_tokens=max_tokens,
        ),
        http_client=http_client,
    )


def _messages_for_prompt(prompt: Prompt) -> list[dict[str, Any]]:
    """Build messages with ``cache_control`` on the stable system block."""

    messages: list[dict[str, Any]] = []
    if prompt.system:
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        )
    messages.append({"role": "user", "content": prompt.user})
    return messages


def _request(client: Any, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.post(OPENROUTER_CHAT_COMPLETIONS_URL, headers=headers, json=body)
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise OpenRouterRequestError("OpenRouter request failed") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenRouterResponseError("OpenRouter returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise OpenRouterResponseError("OpenRouter returned a non-object payload")
    return payload


def _extract_stream_delta(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise OpenRouterResponseError("OpenRouter returned an invalid stream chunk")
    delta = first_choice.get("delta")
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    if content is None:
        return None
    if not isinstance(content, str):
        raise OpenRouterResponseError("OpenRouter response content must be a string")
    return content


def _extract_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterResponseError(
            "OpenRouter response did not include assistant content"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise OpenRouterResponseError("OpenRouter response content must be a non-empty string")
    return content


def _extract_usage(payload: dict[str, Any]) -> Usage:
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, dict):
        return Usage()
    # OpenRouter normalizes usage across providers; cache token fields have been
    # observed both nested under prompt_tokens_details and at the top level. Read
    # nested first, fall back to top-level, so extraction is correct regardless.
    details = raw_usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        details = {}
    cached = details.get("cached_tokens", raw_usage.get("cached_tokens"))
    cache_write = details.get("cache_write_tokens", raw_usage.get("cache_write_tokens"))
    return Usage(
        prompt_tokens=_as_int(raw_usage.get("prompt_tokens")),
        completion_tokens=_as_int(raw_usage.get("completion_tokens")),
        total_tokens=_as_int(raw_usage.get("total_tokens")),
        cached_tokens=_as_int(cached),
        cache_write_tokens=_as_int(cache_write),
        cost=_resolve_cost(raw_usage),
    )


def _resolve_cost(raw_usage: dict[str, Any]) -> float | None:
    # Top-level `cost` is OpenRouter's own charge. Under BYOK it is 0 and the real
    # provider spend lives in `cost_details.upstream_inference_cost`. Prefer the
    # top-level cost when it is non-zero; otherwise fall back to upstream.
    cost = raw_usage.get("cost")
    if cost:
        return _as_float(cost)
    details = raw_usage.get("cost_details")
    if isinstance(details, dict) and details.get("upstream_inference_cost") is not None:
        return _as_float(details["upstream_inference_cost"])
    return _as_float(cost)


def _as_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _as_float(value: Any) -> float | None:
    return float(value) if value is not None else None
