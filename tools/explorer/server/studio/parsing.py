"""Tolerant JSON loader — cleaned port of the streamlit-era ``parsing.py``.

Lofn pipeline steps emit JSON (concepts, essence/facets, pair assignments) inside
messy model output — code fences, prose preamble, trailing commas, raw control
characters in strings, smart quotes. This module extracts and repairs that JSON
so later studio steps can parse step artifacts.

Ported from ``docs/reference/lofn-legacy-llm/parsing.py`` (the vendored fossil),
with the streamlit / ``lofjson`` / ``json_repair`` third-party dependencies
removed — this is a pure-stdlib version. The public surface the studio needs:

  * ``loads_tolerant(text)``        — best-effort parse of one JSON value.
  * ``iter_json_substrings(text)``  — yield balanced ``{...}`` / ``[...]`` blobs.
  * ``select_best_json_candidate(text, schema)`` — find the JSON object whose
    keys best match ``schema`` (``{"key": str, "facets": "list[str]"}``).
  * ``validate_schema(data, schema)`` — recursive type check.

No streamlit, no ``st.write`` debug side-effects: a ``debug`` flag routes messages
to the stdlib ``logging`` module instead.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

JSON = Union[dict, list, str, int, float, bool, None]

_log = logging.getLogger("studio.parsing")


# ---------------------------------------------------------------------------
# Small cleanup helpers.
# ---------------------------------------------------------------------------

def _minimal_cleanup(s: str) -> str:
    """Normalize newlines, strip BOM / nbsp, trim."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("﻿", "").replace(" ", " ")
    return s.strip()


_CODE_FENCE_RE = re.compile(r"```(?:json|javascript|js)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_code_fences(s: str) -> str:
    """Prefer fenced-block contents; otherwise strip stray fences."""
    blocks = _CODE_FENCE_RE.findall(s)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())
    return s.replace("```json", "").replace("```", "").strip()


def _maybe_unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1]
    return s


def _remove_trailing_commas(s: str) -> str:
    """Drop trailing commas before ``}`` / ``]`` (outside strings)."""
    out: List[str] = []
    stack: List[str] = []
    in_str = False
    esc = False
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            j = len(out) - 1
            while j >= 0 and out[j].isspace():
                j -= 1
            if j >= 0 and out[j] == ",":
                out.pop(j)
            if stack:
                stack.pop()
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _escape_control_chars_in_strings(s: str) -> str:
    """Escape raw control chars (newlines/tabs/etc.) inside JSON strings.

    JSON forbids literal control characters in double-quoted strings; some models
    emit them anyway. Walk the text and, only while inside a string, escape:
      ``\\n`` / ``\\r\\n`` / ``\\r`` -> ``\\n``, ``\\t`` -> ``\\t``, other
      controls -> ``\\u00XX``, and a stray inner ``"`` -> ``\\"``.
    """
    out: List[str] = []
    in_str = False
    esc = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            else:
                if ch == "\\":
                    out.append(ch)
                    esc = True
                elif ch == '"':
                    # A quote not followed by a JSON terminator is a literal
                    # inner quote (GPT-style ``"Hello "World""``) — escape it.
                    j = i + 1
                    while j < n and s[j].isspace():
                        j += 1
                    if j < n and s[j] not in ",:}]":
                        out.append('\\"')
                    else:
                        out.append(ch)
                        in_str = False
                elif ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    if i + 1 < n and s[i + 1] == "\n":
                        i += 1
                    out.append("\\n")
                elif ch == "\t":
                    out.append("\\t")
                elif ord(ch) < 0x20:
                    out.append("\\u%04x" % ord(ch))
                else:
                    out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                j = len(out) - 2
                if j < 0 or out[j] != "\\":
                    in_str = True
        i += 1
    return "".join(out).replace('\\"', '"')


# ---------------------------------------------------------------------------
# Candidate extraction.
# ---------------------------------------------------------------------------

def _match_json_end(text: str, start: int) -> Optional[int]:
    """Index of the ``}``/``]`` matching ``text[start]``, ignoring strings."""
    open_to_close = {"{": "}", "[": "]"}
    stack = [text[start]]
    i = start + 1
    in_str = False
    esc = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    return None
                if ch != open_to_close[stack[-1]]:
                    return None
                stack.pop()
                if not stack:
                    return i
        i += 1
    return None


def iter_json_substrings(text: str, max_candidates: int = 24) -> Iterable[str]:
    """Yield balanced JSON substrings (``{...}`` or ``[...]``) found in ``text``."""
    i = 0
    n = len(text)
    count = 0
    while i < n and count < max_candidates:
        ch = text[i]
        if ch in "{[":
            end = _match_json_end(text, i)
            if end is not None:
                yield text[i:end + 1]
                count += 1
                i = end + 1
                continue
        i += 1


# ---------------------------------------------------------------------------
# Tolerant loader.
# ---------------------------------------------------------------------------

def loads_tolerant(candidate: str, debug: bool = False) -> JSON:
    """Turn ``candidate`` into JSON via a ladder of gentle repairs.

    Order: direct parse -> escape-control-chars -> unquote outer literal ->
    strip-trailing-commas -> both repairs combined -> double-decode. Raises
    ``json.JSONDecodeError`` if every attempt fails.
    """
    c = candidate.strip()

    # 1) direct
    try:
        first = json.loads(c)
        if isinstance(first, str) and first.strip().startswith(("{", "[")):
            try:
                return json.loads(first)
            except Exception:
                pass
        return first
    except Exception:
        pass

    # 1b) escape raw control chars inside strings
    repaired_controls = _escape_control_chars_in_strings(c)
    if repaired_controls != c:
        try:
            first = json.loads(repaired_controls)
            if isinstance(first, str) and first.strip().startswith(("{", "[")):
                try:
                    return json.loads(first)
                except Exception:
                    pass
            return first
        except Exception:
            pass

    # 2) strip an outer quote-wrapped literal
    unq = _maybe_unquote(c)
    if unq != c:
        try:
            return json.loads(unq)
        except Exception:
            try:
                s = json.loads(c)
                if isinstance(s, str) and s.strip().startswith(("{", "[")):
                    return json.loads(s)
            except Exception:
                pass

    # 3) trailing commas
    repaired = _remove_trailing_commas(c)
    if repaired != c:
        try:
            first = json.loads(repaired)
            if isinstance(first, str) and first.strip().startswith(("{", "[")):
                try:
                    return json.loads(first)
                except Exception:
                    pass
            return first
        except Exception:
            pass

    # 3b) both repairs combined
    if repaired != c:
        repaired_both = _escape_control_chars_in_strings(repaired)
        if repaired_both != repaired:
            try:
                first = json.loads(repaired_both)
                if isinstance(first, str) and first.strip().startswith(("{", "[")):
                    try:
                        return json.loads(first)
                    except Exception:
                        pass
                return first
            except Exception:
                pass

    # 4) last resort: decodes-to-string-of-JSON
    try:
        s = json.loads(c)
        if isinstance(s, str) and s.strip().startswith(("{", "[")):
            return json.loads(s)
    except Exception:
        pass

    if debug:
        _log.warning("loads_tolerant: unable to parse candidate (len=%d)", len(c))
    raise json.JSONDecodeError("Unable to parse JSON candidate", c, 0)


# ---------------------------------------------------------------------------
# Schema scoring / normalization.
# ---------------------------------------------------------------------------

def _is_sequential_numeric_dict(d: dict) -> bool:
    if not d:
        return False
    keys = list(d.keys())
    if not all(str(k).isdigit() for k in keys):
        return False
    idx = sorted(int(k) for k in keys)
    return idx == list(range(len(keys)))


def _dict_to_list_by_numeric_keys(d: dict) -> list:
    return [d[str(i)] if str(i) in d else d[i] for i in range(len(d))]


def _normalize_to_schema(obj: JSON, schema: Dict[str, Union[type, str]]) -> Optional[dict]:
    """Return ``obj`` reduced to the keys in ``schema`` if satisfiable, else None.

    ``schema`` maps a key to ``str`` (a string value) or the sentinel
    ``"list[str]"`` (a list of strings, with gentle coercions from a numeric-keyed
    dict or a lone string).
    """
    if not isinstance(obj, (dict, list)):
        return None
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        return None

    out: Dict[str, Any] = {}
    for key, t in schema.items():
        if key not in obj:
            return None
        val = obj[key]
        if t is str:
            if isinstance(val, str):
                out[key] = val
            else:
                return None
        elif t == "list[str]":
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                out[key] = val
            elif isinstance(val, dict) and _is_sequential_numeric_dict(val):
                out[key] = [str(x) for x in _dict_to_list_by_numeric_keys(val)]
            elif isinstance(val, str):
                out[key] = [val]
            else:
                return None
        else:
            if isinstance(val, t):  # type: ignore[arg-type]
                out[key] = val
            else:
                return None
    return out


def select_best_json_candidate(
    raw_text: str,
    schema: Dict[str, Union[type, str]],
    expected_schema: Optional[dict] = None,
    debug: bool = False,
) -> dict:
    """Find the JSON object in ``raw_text`` that best matches ``schema``.

    If ``expected_schema`` is given, each candidate is additionally checked with
    ``validate_schema`` before being accepted. Raises ``ValueError`` if nothing
    matches.
    """
    text = _minimal_cleanup(_strip_code_fences(raw_text))

    def _maybe_return(norm: Optional[dict]) -> Optional[dict]:
        if norm is None:
            return None
        if expected_schema and not validate_schema(norm, expected_schema):
            return None
        return norm

    # 0) whole message is the JSON
    try:
        obj = loads_tolerant(text, debug)
        norm = _maybe_return(_normalize_to_schema(obj, schema))
        if norm is not None:
            return norm
    except Exception:
        if debug:
            _log.warning("select_best_json_candidate: whole-message parse failed")

    # 1) scan for balanced substrings, score by key overlap, take the largest match
    candidates = list(iter_json_substrings(text, max_candidates=24))
    key_hints = [k.lower() for k in schema.keys()]

    def key_score(s: str) -> int:
        t = s.lower()
        return sum(1 for k in key_hints if f'"{k}"' in t)

    candidates.sort(key=lambda s: (key_score(s), len(s)), reverse=True)

    parsed: List[Tuple[dict, int]] = []
    for cand in candidates:
        try:
            cleaned = (
                cand.replace("“", "").replace("”", "")
            )
            value = loads_tolerant(cleaned, debug)
        except Exception:
            continue
        norm = _maybe_return(_normalize_to_schema(value, schema))
        if norm is not None:
            parsed.append((norm, len(cand)))

    if parsed:
        parsed.sort(key=lambda t: t[1], reverse=True)
        return parsed[0][0]

    raise ValueError(
        "No JSON matching the expected schema was found "
        f"(saw {len(candidates)} JSON-like blocks)."
    )


def validate_schema(data: Any, schema: Any) -> bool:
    """Recursively validate JSON ``data`` against a nested type ``schema``."""
    if isinstance(schema, dict):
        if not isinstance(data, dict):
            return False
        for key, subschema in schema.items():
            if key not in data or not validate_schema(data[key], subschema):
                return False
    elif isinstance(schema, list):
        if not isinstance(data, list):
            return False
        subschema = schema[0] if schema else None
        for item in data:
            if not validate_schema(item, subschema):
                return False
    elif isinstance(schema, tuple):
        if not isinstance(data, schema):
            return False
    else:
        if not isinstance(data, schema):
            return False
    return True
