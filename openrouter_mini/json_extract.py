"""Pull a JSON candidate span out of a model's raw text response.

Chat-completions models routinely wrap JSON in markdown fences or surrounding
prose even when asked for JSON alone. This lifts story-builder's
dependency-free extractor into the shared adapter so consumers requesting
``response_format`` structured output (or hand-rolled JSON prompts) share one
implementation instead of drifting copies. Parsing and schema validation of
the extracted candidate remain the consumer's responsibility.
"""

from __future__ import annotations


def extract_json_candidate(text: str) -> str:
    """Return the likely JSON span within ``text``.

    Strips a leading/trailing markdown fence, then narrows to the span from
    the first ``{``/``[`` to the last ``}``/``]``. Falls back to the stripped
    input when no bracket is found.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_markdown_fence(stripped)
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    start_candidates = [index for index in (stripped.find("{"), stripped.find("[")) if index != -1]
    if not start_candidates:
        return stripped
    start = min(start_candidates)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end >= start:
        return stripped[start : end + 1]
    return stripped


def _strip_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
