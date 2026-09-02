from __future__ import annotations

import unittest

from openrouter_mini import extract_json_candidate


class ExtractJsonCandidateTest(unittest.TestCase):
    def test_bare_object_is_returned_unchanged(self) -> None:
        self.assertEqual(extract_json_candidate('{"a": 1}'), '{"a": 1}')

    def test_bare_array_is_returned_unchanged(self) -> None:
        self.assertEqual(extract_json_candidate("[1, 2, 3]"), "[1, 2, 3]")

    def test_surrounding_whitespace_is_stripped(self) -> None:
        self.assertEqual(extract_json_candidate('  \n {"a": 1}\n  '), '{"a": 1}')

    def test_fenced_json_with_language_tag(self) -> None:
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(extract_json_candidate(text), '{"a": 1}')

    def test_fenced_json_without_language_tag(self) -> None:
        text = '```\n{"a": 1}\n```'
        self.assertEqual(extract_json_candidate(text), '{"a": 1}')

    def test_fence_without_closing_line(self) -> None:
        text = '```json\n{"a": 1}'
        self.assertEqual(extract_json_candidate(text), '{"a": 1}')

    def test_leading_prose_before_object(self) -> None:
        text = 'Sure, here is the JSON:\n{"a": 1}'
        self.assertEqual(extract_json_candidate(text), '{"a": 1}')

    def test_trailing_prose_after_object_is_not_trimmed(self) -> None:
        # A stripped candidate that already starts with `{`/`[` is returned as-is
        # without narrowing to the closing bracket, matching story-builder's ported
        # behavior: only a leading fence or leading prose triggers narrowing.
        text = '{"a": 1}\nLet me know if that helps!'
        self.assertEqual(extract_json_candidate(text), text)

    def test_leading_and_trailing_prose_around_array(self) -> None:
        text = "Here you go:\n[1, 2, 3]\nHope that works."
        self.assertEqual(extract_json_candidate(text), "[1, 2, 3]")

    def test_no_braces_returns_stripped_input(self) -> None:
        self.assertEqual(extract_json_candidate("  just prose, no JSON  "), "just prose, no JSON")

    def test_unmatched_open_bracket_returns_stripped_input(self) -> None:
        text = "prose { still prose"
        self.assertEqual(extract_json_candidate(text), text)

    def test_object_nested_inside_prose_fence(self) -> None:
        text = 'prefix\n```json\n{"a": {"b": 2}}\n```\nsuffix'
        self.assertEqual(extract_json_candidate(text), '{"a": {"b": 2}}')


if __name__ == "__main__":
    unittest.main()
