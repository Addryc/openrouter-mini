# openrouter-mini

A deliberately small OpenRouter chat-completions adapter over `httpx`. It owns the
HTTP request, a timeout budget, typed errors, usage extraction, and prompt caching
(`cache_control` on the stable system prefix) — and nothing else. Prompt content and
any domain logic stay in the consuming project.

It exists so projects that call OpenRouter (e.g. `solo-mud-platform`,
`lego-moc-builder`) stop maintaining duplicate adapters. See ADR 0005 in
solo-mud-platform for the rationale (chosen over the OpenAI SDK to keep the
dependency footprint to general-purpose `httpx` and avoid provider-coupling on the
OpenRouter-specific usage/cost/cache fields).

## Install

Pin to a tag from the consumer's `pyproject.toml`:

```toml
dependencies = [
  "openrouter-mini @ git+https://github.com/Addryc/openrouter-mini@v0.1.0",
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

## Develop

```bash
make test   # or: python3 -m unittest discover -s tests
```

Tests are deterministic — they inject a fake HTTP client, so no live key or network
is needed.
