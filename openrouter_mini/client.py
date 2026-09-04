"""Minimal OpenRouter chat-completions adapter over httpx.

Owns request construction, an httpx timeout budget, typed errors, usage
extraction, and prompt caching (``cache_control`` on the stable system prefix).
The provider JSON shape stays inside this module; consumers depend on the typed
``Prompt`` / ``Usage`` / error surface, never on httpx or the raw response.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, get_args

import httpx

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV = "OPENROUTER_MODEL"
OPENROUTER_REQUEST_DEADLINE_ENV = "OPENROUTER_REQUEST_DEADLINE_SECONDS"
ReasoningEffort = Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"]
REASONING_EFFORTS = frozenset(get_args(ReasoningEffort))


class OpenRouterError(RuntimeError):
    """Base class for adapter failures; consumers can catch this for any of them."""


class OpenRouterConfigurationError(OpenRouterError):
    """Raised when the adapter cannot be configured from the environment."""


class OpenRouterRequestError(OpenRouterError):
    """Raised when the HTTP request to OpenRouter fails."""


class OpenRouterDeadlineError(OpenRouterRequestError):
    """Raised when a request exceeds its configured wall-clock deadline.

    OpenRouter keeps some long requests alive with periodic bytes, so httpx's
    per-read timeout never fires while an upstream hangs. This is a stricter,
    opt-in wall-clock budget on top of that read timeout.
    """

    def __init__(self, deadline_seconds: float, elapsed_seconds: float) -> None:
        self.deadline_seconds = deadline_seconds
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"OpenRouter request exceeded the {deadline_seconds:g} s deadline "
            f"after {elapsed_seconds:.1f} s"
        )


class OpenRouterResponseError(OpenRouterError):
    """Raised when OpenRouter returns an unexpected response shape."""


@dataclass(frozen=True)
class Prompt:
    """A system/user pair; ``system`` is the cacheable stable prefix."""

    system: str
    user: str


@dataclass(frozen=True)
class ReasoningRequest:
    """A provider-neutral reasoning policy for one chat-completions request.

    Exactly one control is required: a named effort level or a positive
    reasoning-token budget. The adapter serializes this policy into
    OpenRouter's ``reasoning`` body field; callers never pass provider JSON.
    """

    effort: ReasoningEffort | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if (self.effort is None) == (self.max_tokens is None):
            raise ValueError("reasoning requires exactly one of effort or max_tokens")
        if self.effort is not None and (
            not isinstance(self.effort, str) or self.effort not in REASONING_EFFORTS
        ):
            raise ValueError(f"unsupported reasoning effort: {self.effort!r}")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("reasoning max_tokens must be a positive integer")


@dataclass(frozen=True)
class Usage:
    """Adapter-owned per-call usage summary, normalized from the provider block."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
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

    ``response_format`` is passed through verbatim as the request body's
    ``response_format`` field (OpenRouter's JSON-schema structured output,
    e.g. ``{"type": "json_schema", "json_schema": {...}}``). The mapping is
    untyped on purpose — its shape belongs to OpenRouter and the model, not
    this adapter — and ``None`` (the default) omits the field entirely.
    Individual calls can override it, like ``max_tokens``. Not every model
    supports or honors ``response_format``; consumers should gate its use per
    model. Like ``provider_preferences``, only the top-level mapping is
    copied before sending — nested values (e.g. the ``json_schema`` dict) are
    shared with the caller's object, so mutating them after the call is not
    isolated from what was sent.

    ``request_deadline_seconds`` is an opt-in wall-clock budget for the whole
    request (send through fully-read response), on top of
    ``request_timeout_seconds``'s per-read timeout. ``None`` (the default)
    leaves today's behavior unchanged: only the per-read timeout applies.
    """

    api_key: str
    model: str = DEFAULT_MODEL
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    provider_preferences: dict[str, Any] | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    request_deadline_seconds: float | None = None


def load_config(
    *,
    api_key: str | None = None,
    model: str | None = None,
    provider_preferences: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    deadline_seconds: float | None = None,
) -> OpenRouterConfig:
    """Resolve configuration from explicit values or the environment."""

    resolved_key = api_key or os.getenv(OPENROUTER_API_KEY_ENV)
    if not resolved_key:
        raise OpenRouterConfigurationError(f"{OPENROUTER_API_KEY_ENV} is required")
    resolved_model = model or os.getenv(OPENROUTER_MODEL_ENV) or DEFAULT_MODEL
    resolved_deadline = _resolve_deadline_seconds(deadline_seconds)
    return OpenRouterConfig(
        api_key=resolved_key,
        model=resolved_model,
        provider_preferences=provider_preferences,
        max_tokens=max_tokens,
        response_format=response_format,
        request_deadline_seconds=resolved_deadline,
    )


def _resolve_deadline_seconds(deadline_seconds: float | None) -> float | None:
    if deadline_seconds is not None:
        if deadline_seconds <= 0:
            raise OpenRouterConfigurationError(
                f"deadline_seconds must be positive, got {deadline_seconds!r}"
            )
        return deadline_seconds
    env_value = os.getenv(OPENROUTER_REQUEST_DEADLINE_ENV)
    if env_value is None:
        return None
    try:
        parsed = float(env_value)
    except ValueError as exc:
        raise OpenRouterConfigurationError(
            f"{OPENROUTER_REQUEST_DEADLINE_ENV} must be a number, got {env_value!r}"
        ) from exc
    if parsed <= 0:
        raise OpenRouterConfigurationError(
            f"{OPENROUTER_REQUEST_DEADLINE_ENV} must be positive, got {parsed!r}"
        )
    return parsed


class OpenRouterClient:
    """Callable adapter: ``client(prompt) -> str``, recording the last call's usage.

    After each call, ``last_usage`` holds the normalized :class:`Usage` and
    ``last_raw_usage`` holds the provider's raw ``usage`` block (for cache
    verification). Both reset to ``None`` at the start of every call.
    """

    def __init__(
        self,
        config: OpenRouterConfig,
        *,
        http_client: Any | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = config
        self._http_client = http_client
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self.last_usage: Usage | None = None
        self.last_raw_usage: dict[str, Any] | None = None

    @property
    def config(self) -> OpenRouterConfig:
        return self._config

    def __call__(
        self,
        prompt: Prompt,
        *,
        max_tokens: int | None = None,
        reasoning: ReasoningRequest | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self.last_usage = None
        self.last_raw_usage = None
        payload = self._post(
            prompt,
            max_tokens=max_tokens,
            reasoning=reasoning,
            response_format=response_format,
        )
        content = _extract_content(payload)
        self.last_usage = _extract_usage(payload)
        raw_usage = payload.get("usage")
        self.last_raw_usage = raw_usage if isinstance(raw_usage, dict) else None
        return content

    def stream(
        self,
        prompt: Prompt,
        *,
        max_tokens: int | None = None,
        reasoning: ReasoningRequest | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        self.last_usage = None
        self.last_raw_usage = None
        body = self._request_body(
            prompt,
            stream=True,
            max_tokens=max_tokens,
            reasoning=reasoning,
            response_format=response_format,
        )
        headers = self._headers()
        if self._http_client is None:
            timeout = httpx.Timeout(self._config.request_timeout_seconds)
            with httpx.Client(timeout=timeout) as http_client:
                yield from self._stream(http_client, headers, body)
            return
        yield from self._stream(self._http_client, headers, body)

    def _post(
        self,
        prompt: Prompt,
        *,
        max_tokens: int | None = None,
        reasoning: ReasoningRequest | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = self._request_body(
            prompt,
            max_tokens=max_tokens,
            reasoning=reasoning,
            response_format=response_format,
        )
        headers = self._headers()
        deadline = self._config.request_deadline_seconds
        if self._http_client is None:
            timeout = httpx.Timeout(self._config.request_timeout_seconds)
            with httpx.Client(timeout=timeout) as http_client:
                return _request(http_client, headers, body, deadline=deadline, clock=self._clock)
        return _request(self._http_client, headers, body, deadline=deadline, clock=self._clock)

    def _request_body(
        self,
        prompt: Prompt,
        *,
        stream: bool = False,
        max_tokens: int | None = None,
        reasoning: ReasoningRequest | None = None,
        response_format: dict[str, Any] | None = None,
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
        if reasoning is not None:
            body["reasoning"] = _reasoning_body(reasoning)
        resolved_response_format = (
            response_format if response_format is not None else self._config.response_format
        )
        if resolved_response_format is not None:
            body["response_format"] = dict(resolved_response_format)
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
        deadline = self._config.request_deadline_seconds
        started = self._clock()
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
                    _check_deadline(self._clock, started, deadline)
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
    response_format: dict[str, Any] | None = None,
) -> OpenRouterClient:
    """Build a client from the environment (or explicit overrides)."""

    return build_client(
        load_config(
            api_key=api_key,
            model=model,
            provider_preferences=provider_preferences,
            max_tokens=max_tokens,
            response_format=response_format,
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


def _reasoning_body(reasoning: ReasoningRequest) -> dict[str, str | int]:
    if reasoning.effort is not None:
        return {"effort": reasoning.effort}
    return {"max_tokens": reasoning.max_tokens}


def _check_deadline(
    clock: Callable[[], float], started: float, deadline: float | None
) -> None:
    if deadline is None:
        return
    elapsed = clock() - started
    if elapsed > deadline:
        raise OpenRouterDeadlineError(deadline, elapsed)


def _request(
    client: Any,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    deadline: float | None,
    clock: Callable[[], float],
) -> dict[str, Any]:
    started = clock()
    buffer = bytearray()
    try:
        with client.stream(
            "POST", OPENROUTER_CHAT_COMPLETIONS_URL, headers=headers, json=body
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                # Buffer the body before re-raising so `exc.response.text` on the
                # chained error stays readable for callers (actor-runtime #83).
                response.read()
                raise
            for chunk in response.iter_bytes():
                buffer += chunk
                _check_deadline(clock, started, deadline)
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise OpenRouterRequestError("OpenRouter request failed") from exc
    try:
        payload = json.loads(bytes(buffer))
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
    completion_details = raw_usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}
    return Usage(
        prompt_tokens=_as_int(raw_usage.get("prompt_tokens")),
        completion_tokens=_as_int(raw_usage.get("completion_tokens")),
        total_tokens=_as_int(raw_usage.get("total_tokens")),
        cached_tokens=_as_int(cached),
        cache_write_tokens=_as_int(cache_write),
        reasoning_tokens=_as_int(completion_details.get("reasoning_tokens")),
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
