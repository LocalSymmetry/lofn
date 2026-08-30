#!/usr/bin/env python3
"""Regression checks for numbered-stub detection in validate_step.py."""

from __future__ import annotations

import re
import unittest


NUMBERED_STUB = re.compile(
    r"^\s*(?:(?:[-*]|\d+[.)])\s*)?line\s+\d+\s*(?::|[-–—]|$)",
    re.I | re.M,
)


class NumberedStubTest(unittest.TestCase):
    def test_provenance_line_number_is_allowed(self) -> None:
        text = "The personality archive begins at source line 327 and is absent."
        self.assertIsNone(NUMBERED_STUB.search(text))

    def test_bare_numbered_stub_is_rejected(self) -> None:
        self.assertIsNotNone(NUMBERED_STUB.search("Line 1: write this later"))

    def test_list_numbered_stub_is_rejected(self) -> None:
        self.assertIsNotNone(NUMBERED_STUB.search("- line 42 — unfinished lyric"))


if __name__ == "__main__":
    unittest.main()
