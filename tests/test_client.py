from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError
from typing import get_args
from unittest import mock

import httpx

from openrouter_mini import (
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterConfigurationError,
    OpenRouterDeadlineError,
    OpenRouterRequestError,
    OpenRouterResponseError,
    Prompt,
    ReasoningEffort,
    ReasoningRequest,
    load_config,
)
from openrouter_mini.client import (
    OPENROUTER_CHAT_COMPLETIONS_URL,
    OPENROUTER_REQUEST_DEADLINE_ENV,
)


class _FakeClock:
    """A monotonic clock stub the fake response iterators can step by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeResponse:
    def __init__(
        self,
        payload,
        *,
        status_error: bool = False,
        bad_json: bool = False,
        lines=None,
        chunks=None,
        raise_request_error_at: int | None = None,
        clock: "_FakeClock | None" = None,
        step_seconds: float = 0.0,
        status_code: int = 500,
        error_body: bytes = b"",
    ) -> None:
        self._payload = payload
        self._status_error = status_error
        self._bad_json = bad_json
        self._lines = list(lines or [])
        if chunks is not None:
            self._chunks = list(chunks)
        elif bad_json:
            self._chunks = [b"not-json"]
        elif payload is not None:
            self._chunks = [json.dumps(payload).encode()]
        else:
            self._chunks = []
        self._raise_request_error_at = raise_request_error_at
        self._clock = clock
        self._step_seconds = step_seconds
        self._status_code = status_code
        self._error_body = error_body

    def raise_for_status(self) -> None:
        if self._status_error:
            request = httpx.Request("POST", OPENROUTER_CHAT_COMPLETIONS_URL)
            response = httpx.Response(
                self._status_code, request=request, content=self._error_body
            )
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def read(self) -> None:
        return None

    def json(self):
        if self._bad_json:
            raise ValueError("invalid json")
        return self._payload

    def iter_lines(self):
        for index, line in enumerate(self._lines):
            if self._raise_request_error_at is not None and index == self._raise_request_error_at:
                raise httpx.RequestError("boom")
            if self._clock is not None:
                self._clock.advance(self._step_seconds)
            yield line

    def iter_bytes(self):
        for index, chunk in enumerate(self._chunks):
            if self._raise_request_error_at is not None and index == self._raise_request_error_at:
                raise httpx.RequestError("boom")
            if self._clock is not None:
                self._clock.advance(self._step_seconds)
            yield chunk


class _FakeStreamContextManager:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def __enter__(self):
        return self._response

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeClient:
    def __init__(self, response=None, *, raise_request_error: bool = False) -> None:
        self._response = response
        self._raise_request_error = raise_request_error
        self.posted = None
        self.streamed = None

    def post(self, url, *, headers, json):
        self.posted = {"url": url, "headers": headers, "json": json}
        if self._raise_request_error:
            raise httpx.RequestError("boom")
        return self._response

    def stream(self, method, url, *, headers, json):
        # The non-streaming path (`_request`) now issues its POST via
        # `client.stream(...)` too (decision #3), so record it under both
        # `streamed` and the legacy `posted` shape existing assertions use.
        self.streamed = {"method": method, "url": url, "headers": headers, "json": json}
        self.posted = {"url": url, "headers": headers, "json": json}
        if self._raise_request_error:
            raise httpx.RequestError("boom")
        return _FakeStreamContextManager(self._response)


def _config(*, deadline_seconds: float | None = None) -> OpenRouterConfig:
    return OpenRouterConfig(
        api_key="key", model="test-model", request_deadline_seconds=deadline_seconds
    )


def _ok_payload(usage=None):
    payload = {"choices": [{"message": {"content": "hello"}}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


class OpenRouterClientTest(unittest.TestCase):
    def test_returns_content_and_records_usage(self) -> None:
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 7},
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80},
            "cost": 0.0012,
        }
        fake = _FakeClient(_FakeResponse(_ok_payload(usage)))
        client = OpenRouterClient(_config(), http_client=fake)

        result = client(Prompt(system="sys", user="usr"))

        self.assertEqual(result, "hello")
        self.assertEqual(client.last_usage.prompt_tokens, 100)
        self.assertEqual(client.last_usage.completion_tokens, 20)
        self.assertEqual(client.last_usage.cached_tokens, 80)
        self.assertEqual(client.last_usage.reasoning_tokens, 7)
        self.assertEqual(client.last_usage.cost, 0.0012)
        self.assertEqual(client.last_raw_usage, usage)

    def test_system_block_carries_cache_control(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="stable prefix", user="volatile"))

        messages = fake.posted["json"]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"][0]["text"], "stable prefix")
        self.assertEqual(messages[0]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(messages[1], {"role": "user", "content": "volatile"})
        self.assertEqual(fake.posted["json"]["model"], "test-model")

    def test_request_opts_into_usage_accounting(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertEqual(fake.posted["json"]["usage"], {"include": True})

    def test_provider_preferences_pass_through_verbatim(self) -> None:
        preferences = {"sort": "throughput", "allow_fallbacks": True}
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        config = OpenRouterConfig(
            api_key="key", model="test-model", provider_preferences=preferences
        )
        client = OpenRouterClient(config, http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertEqual(fake.posted["json"]["provider"], preferences)

    def test_no_provider_block_without_preferences(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertNotIn("provider", fake.posted["json"])

    def test_no_max_tokens_field_by_default(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertNotIn("max_tokens", fake.posted["json"])
        self.assertNotIn("reasoning", fake.posted["json"])

    def test_config_max_tokens_is_sent(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        config = OpenRouterConfig(api_key="key", model="test-model", max_tokens=4096)
        client = OpenRouterClient(config, http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertEqual(fake.posted["json"]["max_tokens"], 4096)

    def test_per_call_max_tokens_overrides_config(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        config = OpenRouterConfig(api_key="key", model="test-model", max_tokens=4096)
        client = OpenRouterClient(config, http_client=fake)

        client(Prompt(system="s", user="u"), max_tokens=8192)

        self.assertEqual(fake.posted["json"]["max_tokens"], 8192)

    def test_per_call_max_tokens_without_config_default(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"), max_tokens=2048)

        self.assertEqual(fake.posted["json"]["max_tokens"], 2048)

    def test_stream_carries_max_tokens(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            "data: [DONE]",
        ]
        fake = _FakeClient(_FakeResponse(None, lines=lines))
        client = OpenRouterClient(_config(), http_client=fake)

        "".join(client.stream(Prompt(system="s", user="u"), max_tokens=1024))

        self.assertEqual(fake.streamed["json"]["max_tokens"], 1024)

    def test_stream_omits_max_tokens_by_default(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            "data: [DONE]",
        ]
        fake = _FakeClient(_FakeResponse(None, lines=lines))
        client = OpenRouterClient(_config(), http_client=fake)

        "".join(client.stream(Prompt(system="s", user="u")))

        self.assertNotIn("max_tokens", fake.streamed["json"])
        self.assertNotIn("reasoning", fake.streamed["json"])

    def test_reasoning_effort_is_serialized_for_nonstreaming_call(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"), reasoning=ReasoningRequest(effort="high"))

        self.assertEqual(fake.posted["json"]["reasoning"], {"effort": "high"})

    def test_reasoning_effort_alias_owns_all_serialized_efforts(self) -> None:
        expected_efforts = ("max", "xhigh", "high", "medium", "low", "minimal", "none")

        self.assertEqual(get_args(ReasoningEffort), expected_efforts)
        for effort in get_args(ReasoningEffort):
            fake = _FakeClient(_FakeResponse(_ok_payload()))
            client = OpenRouterClient(_config(), http_client=fake)

            client(Prompt(system="s", user="u"), reasoning=ReasoningRequest(effort=effort))

            self.assertEqual(fake.posted["json"]["reasoning"], {"effort": effort})

    def test_reasoning_token_budget_is_serialized_for_stream(self) -> None:
        lines = ['data: {"choices":[{"delta":{"content":"hello"}}]}', "data: [DONE]"]
        fake = _FakeClient(_FakeResponse(None, lines=lines))
        client = OpenRouterClient(_config(), http_client=fake)

        "".join(
            client.stream(
                Prompt(system="s", user="u"),
                reasoning=ReasoningRequest(max_tokens=512),
            )
        )

        self.assertEqual(fake.streamed["json"]["reasoning"], {"max_tokens": 512})

    def test_reasoning_request_rejects_invalid_policies(self) -> None:
        invalid_policies = (
            {},
            {"effort": "high", "max_tokens": 512},
            {"max_tokens": 0},
            {"max_tokens": -1},
            {"max_tokens": True},
            {"effort": "unsupported"},
            {"effort": ["high"]},
        )

        for policy in invalid_policies:
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    ReasoningRequest(**policy)

    def test_reasoning_request_is_publicly_exported(self) -> None:
        request = ReasoningRequest(effort="minimal")

        self.assertEqual(request.effort, "minimal")
        with self.assertRaises(FrozenInstanceError):
            request.effort = "high"

    def test_load_config_carries_max_tokens(self) -> None:
        config = load_config(api_key="k", model="m", max_tokens=512)
        self.assertEqual(config.max_tokens, 512)

    def test_no_response_format_field_by_default(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertNotIn("response_format", fake.posted["json"])

    def test_config_response_format_is_sent(self) -> None:
        response_format = {"type": "json_schema", "json_schema": {"name": "schema"}}
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        config = OpenRouterConfig(
            api_key="key", model="test-model", response_format=response_format
        )
        client = OpenRouterClient(config, http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertEqual(fake.posted["json"]["response_format"], response_format)

    def test_per_call_response_format_overrides_config(self) -> None:
        config_format = {"type": "json_schema", "json_schema": {"name": "config"}}
        call_format = {"type": "json_schema", "json_schema": {"name": "call"}}
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        config = OpenRouterConfig(
            api_key="key", model="test-model", response_format=config_format
        )
        client = OpenRouterClient(config, http_client=fake)

        client(Prompt(system="s", user="u"), response_format=call_format)

        self.assertEqual(fake.posted["json"]["response_format"], call_format)

    def test_per_call_response_format_without_config_default(self) -> None:
        response_format = {"type": "json_schema", "json_schema": {"name": "schema"}}
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"), response_format=response_format)

        self.assertEqual(fake.posted["json"]["response_format"], response_format)

    def test_stream_carries_response_format(self) -> None:
        response_format = {"type": "json_schema", "json_schema": {"name": "schema"}}
        lines = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            "data: [DONE]",
        ]
        fake = _FakeClient(_FakeResponse(None, lines=lines))
        client = OpenRouterClient(_config(), http_client=fake)

        "".join(client.stream(Prompt(system="s", user="u"), response_format=response_format))

        self.assertEqual(fake.streamed["json"]["response_format"], response_format)

    def test_stream_omits_response_format_by_default(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            "data: [DONE]",
        ]
        fake = _FakeClient(_FakeResponse(None, lines=lines))
        client = OpenRouterClient(_config(), http_client=fake)

        "".join(client.stream(Prompt(system="s", user="u")))

        self.assertNotIn("response_format", fake.streamed["json"])

    def test_load_config_carries_response_format(self) -> None:
        response_format = {"type": "json_schema", "json_schema": {"name": "schema"}}
        config = load_config(api_key="k", model="m", response_format=response_format)
        self.assertEqual(config.response_format, response_format)

    def test_cost_falls_back_to_upstream_under_byok(self) -> None:
        usage = {
            "prompt_tokens": 3114,
            "cost": 0,
            "is_byok": True,
            "cost_details": {"upstream_inference_cost": 0.049662},
        }
        fake = _FakeClient(_FakeResponse(_ok_payload(usage)))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertEqual(client.last_usage.cost, 0.049662)

    def test_top_level_cost_wins_when_nonzero(self) -> None:
        usage = {"cost": 0.0012, "cost_details": {"upstream_inference_cost": 9.99}}
        fake = _FakeClient(_FakeResponse(_ok_payload(usage)))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertEqual(client.last_usage.cost, 0.0012)

    def test_top_level_cached_tokens_fallback(self) -> None:
        usage = {"prompt_tokens": 10, "cached_tokens": 4}
        fake = _FakeClient(_FakeResponse(_ok_payload(usage)))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertEqual(client.last_usage.cached_tokens, 4)

    def test_no_usage_block_yields_empty_usage(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload()))
        client = OpenRouterClient(_config(), http_client=fake)

        client(Prompt(system="s", user="u"))

        self.assertIsNone(client.last_usage.prompt_tokens)
        self.assertIsNone(client.last_raw_usage)

    def test_stream_yields_deltas_and_records_terminal_usage(self) -> None:
        usage = {
            "prompt_tokens": 3114,
            "completion_tokens": 222,
            "completion_tokens_details": {"reasoning_tokens": 91},
            "total_tokens": 3336,
            "prompt_tokens_details": {"cached_tokens": 80},
            "cost": 0,
            "cost_details": {"upstream_inference_cost": 0.049662},
        }
        lines = [
            'data: {"choices":[{"delta":{"content":"he"}}]}',
            'data: {"choices":[{"delta":{}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}],"usage":' + json.dumps(usage) + "}",
            "data: [DONE]",
        ]
        fake = _FakeClient(_FakeResponse(None, lines=lines))
        client = OpenRouterClient(_config(), http_client=fake)

        result = "".join(client.stream(Prompt(system="sys", user="usr")))

        self.assertEqual(result, "hello")
        self.assertEqual(fake.streamed["method"], "POST")
        self.assertEqual(fake.streamed["url"], OPENROUTER_CHAT_COMPLETIONS_URL)
        self.assertEqual(fake.streamed["json"]["stream"], True)
        self.assertEqual(fake.streamed["json"]["stream_options"], {"include_usage": True})
        self.assertEqual(fake.streamed["json"]["usage"], {"include": True})
        self.assertEqual(client.last_usage.prompt_tokens, 3114)
        self.assertEqual(client.last_usage.completion_tokens, 222)
        self.assertEqual(client.last_usage.total_tokens, 3336)
        self.assertEqual(client.last_usage.cached_tokens, 80)
        self.assertEqual(client.last_usage.reasoning_tokens, 91)
        self.assertEqual(client.last_usage.cost, 0.049662)
        self.assertEqual(client.last_raw_usage, usage)

    def test_stream_skips_chunks_without_delta_content(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"he"}}]}',
            'data: {"choices":[{"delta":{}}]}',
            'data: {"choices":[{"delta":{"content":null}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}]}',
            "data: [DONE]",
        ]
        fake = _FakeClient(_FakeResponse(None, lines=lines))
        client = OpenRouterClient(_config(), http_client=fake)

        self.assertEqual("".join(client.stream(Prompt(system="sys", user="usr"))), "hello")

    def test_stream_mid_stream_request_error_is_wrapped(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"he"}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}]}',
        ]
        fake = _FakeClient(_FakeResponse(None, lines=lines, raise_request_error_at=1))
        client = OpenRouterClient(_config(), http_client=fake)

        with self.assertRaises(OpenRouterRequestError):
            "".join(client.stream(Prompt(system="s", user="u")))

    def test_stream_malformed_chunk_raises_response_error(self) -> None:
        lines = ["data: not-json"]
        fake = _FakeClient(_FakeResponse(None, lines=lines))
        client = OpenRouterClient(_config(), http_client=fake)

        with self.assertRaises(OpenRouterResponseError):
            "".join(client.stream(Prompt(system="s", user="u")))

    def test_request_error_is_wrapped(self) -> None:
        fake = _FakeClient(raise_request_error=True)
        client = OpenRouterClient(_config(), http_client=fake)
        with self.assertRaises(OpenRouterRequestError):
            client(Prompt(system="s", user="u"))

    def test_http_status_error_is_wrapped(self) -> None:
        fake = _FakeClient(_FakeResponse(_ok_payload(), status_error=True))
        client = OpenRouterClient(_config(), http_client=fake)
        with self.assertRaises(OpenRouterRequestError):
            client(Prompt(system="s", user="u"))

    def test_invalid_json_raises_response_error(self) -> None:
        fake = _FakeClient(_FakeResponse(None, bad_json=True))
        client = OpenRouterClient(_config(), http_client=fake)
        with self.assertRaises(OpenRouterResponseError):
            client(Prompt(system="s", user="u"))

    def test_missing_content_raises_response_error(self) -> None:
        fake = _FakeClient(_FakeResponse({"choices": []}))
        client = OpenRouterClient(_config(), http_client=fake)
        with self.assertRaises(OpenRouterResponseError):
            client(Prompt(system="s", user="u"))

    def test_empty_content_raises_response_error(self) -> None:
        fake = _FakeClient(_FakeResponse({"choices": [{"message": {"content": "   "}}]}))
        client = OpenRouterClient(_config(), http_client=fake)
        with self.assertRaises(OpenRouterResponseError):
            client(Prompt(system="s", user="u"))

    def test_load_config_uses_explicit_values(self) -> None:
        config = load_config(api_key="explicit", model="m")
        self.assertEqual(config.api_key, "explicit")
        self.assertEqual(config.model, "m")

    def test_load_config_requires_api_key(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(OpenRouterConfigurationError):
                load_config()


class OpenRouterDeadlineTest(unittest.TestCase):
    """Covers actor-runtime #86: an opt-in wall-clock request deadline."""

    def test_nonstreaming_deadline_exceeded_raises_on_third_chunk(self) -> None:
        clock = _FakeClock()
        chunks = [b'{"choices"', b':[{"message"', b':{"content":"hi"}}]}']
        fake = _FakeClient(
            _FakeResponse(None, chunks=chunks, clock=clock, step_seconds=2.0)
        )
        client = OpenRouterClient(_config(deadline_seconds=5.0), http_client=fake, clock=clock)

        with self.assertRaises(OpenRouterDeadlineError) as ctx:
            client(Prompt(system="s", user="u"))

        err = ctx.exception
        self.assertEqual(err.deadline_seconds, 5.0)
        self.assertEqual(err.elapsed_seconds, 6.0)
        self.assertEqual(
            str(err),
            "OpenRouter request exceeded the 5 s deadline after 6.0 s",
        )
        self.assertIsInstance(err, OpenRouterRequestError)

    def test_nonstreaming_deadline_not_exceeded_by_fast_body(self) -> None:
        clock = _FakeClock()
        fake = _FakeClient(_FakeResponse(_ok_payload({"prompt_tokens": 5}), clock=clock))
        client = OpenRouterClient(_config(deadline_seconds=5.0), http_client=fake, clock=clock)

        result = client(Prompt(system="s", user="u"))

        self.assertEqual(result, "hello")
        self.assertEqual(client.last_usage.prompt_tokens, 5)

    def test_streaming_deadline_exceeded_raises_on_third_line(self) -> None:
        clock = _FakeClock()
        lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            'data: {"choices":[{"delta":{"content":"b"}}]}',
            'data: {"choices":[{"delta":{"content":"c"}}]}',
        ]
        fake = _FakeClient(
            _FakeResponse(None, lines=lines, clock=clock, step_seconds=2.0)
        )
        client = OpenRouterClient(_config(deadline_seconds=5.0), http_client=fake, clock=clock)

        with self.assertRaises(OpenRouterDeadlineError) as ctx:
            "".join(client.stream(Prompt(system="s", user="u")))

        err = ctx.exception
        self.assertEqual(err.deadline_seconds, 5.0)
        self.assertEqual(err.elapsed_seconds, 6.0)
        self.assertIsInstance(err, OpenRouterRequestError)

    def test_streaming_deadline_not_exceeded_by_fast_body(self) -> None:
        clock = _FakeClock()
        lines = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            "data: [DONE]",
        ]
        fake = _FakeClient(_FakeResponse(None, lines=lines, clock=clock))
        client = OpenRouterClient(_config(deadline_seconds=5.0), http_client=fake, clock=clock)

        result = "".join(client.stream(Prompt(system="s", user="u")))

        self.assertEqual(result, "hello")

    def test_load_config_explicit_deadline_wins(self) -> None:
        with mock.patch.dict(
            "os.environ", {OPENROUTER_REQUEST_DEADLINE_ENV: "45"}, clear=True
        ):
            config = load_config(api_key="k", deadline_seconds=30)
        self.assertEqual(config.request_deadline_seconds, 30)

    def test_load_config_reads_deadline_from_environment(self) -> None:
        with mock.patch.dict(
            "os.environ", {OPENROUTER_REQUEST_DEADLINE_ENV: "45"}, clear=True
        ):
            config = load_config(api_key="k")
        self.assertEqual(config.request_deadline_seconds, 45.0)

    def test_load_config_deadline_unset_stays_none(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            config = load_config(api_key="k")
        self.assertIsNone(config.request_deadline_seconds)

    def test_load_config_rejects_non_positive_or_unparsable_deadline(self) -> None:
        for env_value in ("0", "-1", "abc"):
            with self.subTest(env_value=env_value):
                with mock.patch.dict(
                    "os.environ", {OPENROUTER_REQUEST_DEADLINE_ENV: env_value}, clear=True
                ):
                    with self.assertRaises(OpenRouterConfigurationError):
                        load_config(api_key="k")

    def test_http_402_with_json_body_keeps_response_text_readable(self) -> None:
        body = b'{"error": "payment required"}'
        fake = _FakeClient(
            _FakeResponse(None, status_error=True, status_code=402, error_body=body)
        )
        client = OpenRouterClient(_config(), http_client=fake)

        with self.assertRaises(OpenRouterRequestError) as ctx:
            client(Prompt(system="s", user="u"))

        cause = ctx.exception.__cause__
        self.assertIsInstance(cause, httpx.HTTPStatusError)
        self.assertIn("payment required", cause.response.text)


if __name__ == "__main__":
    unittest.main()
