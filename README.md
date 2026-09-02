# openrouter-mini

A deliberately small OpenRouter chat-completions adapter over `httpx`. It owns
request construction, a timeout budget, typed errors, normalized usage, and
prompt caching (`cache_control` on the stable system prefix). Prompt content,
multi-turn orchestration, tool calls, structured-output handling, retries, and
domain policy belong to the consuming project.

It exists so projects using OpenRouter do not maintain duplicate adapters. The
package uses general-purpose `httpx` instead of an SDK while keeping
OpenRouter-specific cache and cost extraction in one place.

> **Status:** public so related projects can pin it; maintained for those
> needs. The intentionally small scope is a feature. Forks welcome under MIT.

## Install

Pin the HTTPS tag tarball in a consumer `pyproject.toml`; this does not require
Git in the consumer image:

```toml
dependencies = [
  "openrouter-mini @ https://github.com/Addryc/openrouter-mini/archive/refs/tags/v0.6.0.tar.gz",
]
```

For local package development, use an editable install:

```bash
pip install -e /path/to/openrouter-mini
```

## Complete a request

`load_client()` reads `OPENROUTER_API_KEY` and the optional
`OPENROUTER_MODEL`. Pass explicit `api_key` or `model` values to override them.
`OpenRouterError` is the public base error; configuration, request, and
unexpected-response failures use its typed subclasses.

```python
from openrouter_mini import OpenRouterError, Prompt, load_client

try:
    client = load_client()
    text = client(Prompt(system="<stable prefix>", user="<current turn>"))
except OpenRouterError as exc:
    # Handle adapter configuration, request, or response failures.
    raise

print(text)
print(client.last_usage)
print(client.last_raw_usage)  # provider's raw usage block, when supplied
```

`last_usage` is an adapter-owned `Usage` summary. Each field can be `None` when
the provider omits it: `prompt_tokens`, `completion_tokens`, `total_tokens`,
`cached_tokens`, `cache_write_tokens`, `reasoning_tokens`, and `cost`.
`last_raw_usage` retains the raw provider `usage` mapping when present for
diagnostics such as cache verification. For BYOK responses with a zero top-level
cost, `cost` falls back to `cost_details.upstream_inference_cost` when supplied.

Requesting structured output? Models often wrap JSON in prose or markdown
fences even when asked for JSON alone; `extract_json_candidate` pulls the
likely JSON span out of the raw text before you parse it — see
[Structured output](#structured-output).

## Stream a request

`stream()` yields text deltas. Consume the iterator fully before reading final
`last_usage` or `last_raw_usage`: terminal streaming usage arrives at the end of
the stream.

```python
from openrouter_mini import Prompt, load_client

client = load_client()
for delta in client.stream(Prompt(system="<stable prefix>", user="<current turn>")):
    print(delta, end="", flush=True)

print(client.last_usage)
print(client.last_raw_usage)
```

## Prompt caching and provider routing

Non-empty `system` text is sent as a cacheable text block
(`cache_control: ephemeral`); `user` carries the changing turn. Reuse a stable
system prefix to make caching possible, then confirm real cache behavior from
`last_raw_usage` rather than a unit test.

Pass `provider_preferences` to forward an
[OpenRouter provider routing](https://openrouter.ai/docs/features/provider-routing)
mapping verbatim as the request's `provider` field:

```python
client = load_client(provider_preferences={"sort": "throughput", "allow_fallbacks": True})
```

The mapping is intentionally untyped because its fields belong to OpenRouter.
Omit it to omit the `provider` field; provider routing behavior remains
OpenRouter's responsibility.

## Output and reasoning controls

`max_tokens` is the output-token cap. Set it on `OpenRouterConfig` through
`load_client(max_tokens=...)`, or per call; a per-call value wins. Omit it and
the top-level `max_tokens` field is not sent.

```python
from openrouter_mini import Prompt, ReasoningRequest, load_client

client = load_client(max_tokens=4096)
text = client(Prompt(system="...", user="..."), max_tokens=8192)
```

For a single call, use the adapter-owned immutable `ReasoningRequest`, with
exactly one control: an effort from `ReasoningEffort` or a positive reasoning
token budget. The controls are mutually exclusive and serialize beneath the
request's `reasoning` field; omit `reasoning` to leave that field out.

```python
text = client(
    Prompt(system="...", user="..."),
    reasoning=ReasoningRequest(effort="high"),
)

# Or: ReasoningRequest(max_tokens=2048)
```

Reasoning tokens, when reported, are exposed as `last_usage.reasoning_tokens`
and are part of completion/output billing.

## Structured output

Pass `response_format` to request OpenRouter's JSON-schema structured output
for models that support it:

```python
schema = {
    "type": "json_schema",
    "json_schema": {"name": "reply", "schema": {"type": "object", "properties": {...}}},
}
client = load_client(response_format=schema)
text = client(Prompt(system="...", user="..."), response_format=schema)  # per-call overrides config
```

The mapping is forwarded verbatim as the request's `response_format` field
and is untyped on purpose — its shape belongs to OpenRouter and the model.
Omit it to omit the field; a per-call value wins over the config default,
like `max_tokens`. Not every model supports or honors `response_format`;
consumers must gate its use per model rather than relying on the adapter to
validate it. Schema validation, repair loops, and re-prompting on failure
remain the consumer's responsibility.

`extract_json_candidate(text)` strips a leading/trailing markdown fence, then
narrows to the span from the first `{`/`[` to the last `}`/`]`, returning the
stripped input unchanged when no bracket is found:

```python
from openrouter_mini import extract_json_candidate

candidate = extract_json_candidate(text)  # text: raw model output
data = json.loads(candidate)
```

It has no dependencies and does no validation; parsing and schema checking
are the consumer's responsibility.

## Develop

```bash
make test
```

Tests inject a fake HTTP client; no API key, live provider call, or network is
needed.
