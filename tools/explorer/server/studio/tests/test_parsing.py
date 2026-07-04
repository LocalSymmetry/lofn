"""Tolerant JSON loader (vendored parsing.py) — messy model output round-trips."""
from __future__ import annotations

import pytest

from studio.parsing import (
    loads_tolerant,
    iter_json_substrings,
    select_best_json_candidate,
    validate_schema,
)


def test_direct_parse():
    assert loads_tolerant('{"a": 1}') == {"a": 1}


def test_trailing_comma_repair():
    assert loads_tolerant('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_raw_newline_in_string_repair():
    got = loads_tolerant('{"text": "line one\nline two"}')
    assert got["text"] == "line one\nline two"


def test_iter_substrings_finds_embedded_json():
    text = 'Here is the answer:\n{"pairs": [1,2,3]}\nThanks!'
    subs = list(iter_json_substrings(text))
    assert subs
    assert loads_tolerant(subs[0]) == {"pairs": [1, 2, 3]}


def test_select_best_candidate_from_fenced_output():
    raw = 'Sure!\n```json\n{"meta_prompt": "make art", "extra": 9}\n```\n'
    got = select_best_json_candidate(raw, {"meta_prompt": str})
    assert got == {"meta_prompt": "make art"}


def test_select_best_candidate_list_coercion():
    raw = '{"facets": {"0": "a", "1": "b"}}'
    got = select_best_json_candidate(raw, {"facets": "list[str]"})
    assert got == {"facets": ["a", "b"]}


def test_select_best_candidate_no_match_raises():
    with pytest.raises(ValueError):
        select_best_json_candidate("no json here at all", {"meta_prompt": str})


def test_validate_schema():
    assert validate_schema({"a": "x", "b": [1, 2]}, {"a": str, "b": [int]})
    assert not validate_schema({"a": 1}, {"a": str})
