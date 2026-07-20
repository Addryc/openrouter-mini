# openrouter-mini

A deliberately small OpenRouter chat-completions adapter over `httpx`. It owns the
HTTP request, a timeout budget, typed errors, usage extraction, and prompt caching
(`cache_control` on the stable system prefix) — and nothing else. Prompt content and
any domain logic stay in the consuming project.

It exists so my projects that call OpenRouter stop maintaining duplicate adapters.
It was chosen over the OpenAI SDK to keep the dependency footprint to
general-purpose `httpx` and to own the OpenRouter-specific usage/cost/cache
extraction (including the BYOK case, where the real spend lives in
`cost_details.upstream_inference_cost`).

> **Status:** public so my own projects can pin it; maintained for their needs.
> Use it freely under MIT, but the deliberately small scope (no streaming, no
> multi-turn, no tool calls) is a feature — expect feature requests to be
> declined. Forks welcome.

## Install

Pin to a tag from the consumer's `pyproject.toml`:

```toml
dependencies = [
  "openrouter-mini @ git+https://github.com/Addryc/openrouter-mini@v0.3.1",
]
```

For local development of the package itself, use an editable install:

```bash
pip install -e /path/to/openrouter-mini
```

## Usage

```python
from openrouter_mini import load_client, Prompt

client = load_client()  # reads OPENROUTER_API_KEY (+ optional OPENROUTER_MODEL)
text = client(Prompt(system="<stable, cacheable prefix>", user="<volatile turn>"))

print(client.last_usage)      # normalized Usage(prompt_tokens=..., cached_tokens=..., cost=...)
print(client.last_raw_usage)  # provider's raw usage block, for cache verification
```

The `system` text is sent as a cacheable block (`cache_control: ephemeral`); put the
large stable prefix there and the small changing content in `user` to benefit from
prompt caching. Caching only pays when the prefix is reused many times within the
short ephemeral window — verify real cache hits against `last_raw_usage`
(`cached_tokens > 0`) with a live, key-gated probe, not a unit test.

### Provider routing preferences

The same model id can vary widely in throughput depending on which upstream
provider serves it. Pass `provider_preferences` to forward an
[OpenRouter provider routing](https://openrouter.ai/docs/features/provider-routing)
object verbatim as the request's `provider` field:

```python
client = load_client(provider_preferences={"sort": "throughput", "allow_fallbacks": True})
```

The mapping is intentionally untyped — routing fields belong to OpenRouter's
schema, and new ones work without a library release. Omit it (the default) and
request bodies are byte-identical to previous releases. Prefer advisory ordering
(`sort`/`order` with fallbacks) over hard `only` pinning, which changes failure
semantics.

## Develop

```bash
make test   # or: python3 -m unittest discover -s tests
```

Tests are deterministic — they inject a fake HTTP client, so no live key or network
is needed.
