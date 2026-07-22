from __future__ import annotations

import json
import unittest
from unittest import mock

import httpx

from openrouter_mini import (
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterConfigurationError,
    OpenRouterRequestError,
    OpenRouterResponseError,
    Prompt,
    load_config,
)
from openrouter_mini.client import OPENROUTER_CHAT_COMPLETIONS_URL


class _FakeResponse:
    def __init__(
        self,
        payload,
        *,
        status_error: bool = False,
        bad_json: bool = False,
        lines=None,
        raise_request_error_at: int | None = None,
    ) -> None:
        self._payload = payload
        self._status_error = status_error
        self._bad_json = bad_json
        self._lines = list(lines or [])
        self._raise_request_error_at = raise_request_error_at

    def raise_for_status(self) -> None:
        if self._status_error:
            request = httpx.Request("POST", OPENROUTER_CHAT_COMPLETIONS_URL)
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self):
        if self._bad_json:
            raise ValueError("invalid json")
        return self._payload

    def iter_lines(self):
        for index, line in enumerate(self._lines):
            if self._raise_request_error_at is not None and index == self._raise_request_error_at:
                raise httpx.RequestError("boom")
            yield line


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
        self.streamed = {"method": method, "url": url, "headers": headers, "json": json}
        if self._raise_request_error:
            raise httpx.RequestError("boom")
        return _FakeStreamContextManager(self._response)


def _config() -> OpenRouterConfig:
    return OpenRouterConfig(api_key="key", model="test-model")


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


if __name__ == "__main__":
    unittest.main()
