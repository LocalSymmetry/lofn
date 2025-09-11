from __future__ import annotations
import json
import re
import streamlit as st
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .lofjson import parse_with_repairs
from json_repair import repair_json

JSON = Union[dict, list, str, int, float, bool, None]

# --------- small helpers ---------

def _minimal_cleanup(s: str) -> str:
    # normalize newlines, strip BOM/nbsp, drop obvious log clutter
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\ufeff", "").replace("\u00a0", " ")
    return s.strip()

_CODE_FENCE_RE = re.compile(r"```(?:json|javascript|js)?\s*([\s\S]*?)```", re.IGNORECASE)

def _strip_code_fences(s: str) -> str:
    # If fenced blocks exist, prefer their content; otherwise just strip any stray fences.
    blocks = _CODE_FENCE_RE.findall(s)
    if blocks:
        # Return concatenated blocks (common when models emit multiple fenced snippets)
        return "\n\n".join(b.strip() for b in blocks if b.strip())
    # No explicit fences—just remove lone fences
    return s.replace("```json", "").replace("```", "").strip()

def _maybe_unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1]
    return s

def _remove_trailing_commas(s: str) -> str:
    """
    Remove trailing commas before } or ] outside strings (minimal 'repair' only).
    """
    out = []
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
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        # not in string
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            # if last non-space char is a comma, drop it
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
    """
    JSON cannot contain literal control characters (including raw newlines) inside
    double-quoted strings. GPT-5 sometimes emits them. This pass walks the text and,
    only while inside a JSON string, replaces:
      \n -> \\n,  \r\n -> \\n,  \r -> \\n,  \t -> \\t,
      other control chars -> \\u00XX
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
                if ch == '\\':
                    out.append(ch)
                    esc = True
                elif ch == '"':
                    # If the quote isn't followed by a typical string terminator
                    # (comma, colon, closing brace/bracket), treat it as a
                    # literal quote and escape it. GPT-5 occasionally emits
                    # unescaped quotes inside strings like: "Hello "World"".
                    j = i + 1
                    while j < n and s[j].isspace():
                        j += 1
                    if j < n and s[j] not in ',:}]':
                        out.append('\\"')
                    else:
                        out.append(ch)
                        in_str = False
                elif ch == '\n':
                    out.append('\\n')
                elif ch == '\r':
                    if i + 1 < n and s[i+1] == '\n':
                        i += 1
                    out.append('\\n')
                elif ch == '\t':
                    out.append('\\t')
                elif ord(ch) < 0x20:
                    out.append('\\u%04x' % ord(ch))
                else:
                    out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                j = len(out) - 2
                if j < 0 or out[j] != '\\':
                    in_str = True
        i += 1
    return ''.join(out).replace('\\"',"\"")

# --------- JSON candidate extraction ---------

def _match_json_end(text: str, start: int) -> Optional[int]:
    """
    Given text[start] in {'{','['}, return the index of the matching closing
    bracket, ignoring anything inside JSON strings.
    """
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
            elif ch == '\\':
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
                expected = open_to_close[stack[-1]]
                if ch != expected:
                    return None
                stack.pop()
                if not stack:
                    return i
        i += 1
    return None

def iter_json_substrings(text: str, max_candidates: int = 16) -> Iterable[str]:
    """
    Yield balanced JSON substrings ({...} or [...]) found anywhere in 'text'.
    """
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

# --------- tolerant JSON loader ---------

def _loads_tolerant(candidate: str, debug : bool = False) -> JSON:
    """
    Try several safe ways to turn a candidate string into JSON.
    No heavy 'repairs'—only gentle unquoting and trailing-comma removal.
    """
    c = candidate.strip()

    # 1) direct
    try:
        first = json.loads(c)
    except Exception:
        try:
            first = loads(repair_json(c))
        except Exception:
            pass
    try:
        if isinstance(first, str) and first.strip().startswith(("{", "[")):
            try:
                return json.loads(first)
            except Exception:
                pass
        return first
    except Exception:
        pass

    # 1b) repair: escape raw newlines/tabs/control chars inside quoted JSON strings
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

    # 2) if it's a quoted string literal containing JSON, remove outer quotes
    unq = _maybe_unquote(c)
    if unq != c:
        try:
            return json.loads(unq)
        except Exception:
            # Sometimes it is a JSON-encoded string of JSON: try twice.
            try:
                s = json.loads(c)
                if isinstance(s, str) and s.strip().startswith(("{", "[")):
                    return json.loads(s)
            except Exception:
                pass

    # 3) trailing commas (common LLM hiccup)
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

    # 3b) combine both repairs (common in GPT outputs)
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

    # 4) as last resort, if it decodes to a string, try to parse that string
    try:
        s = json.loads(c)
        if isinstance(s, str) and s.strip().startswith(("{", "[")):
            return json.loads(s)
    except Exception:
        pass

    try:
        return json.loads(repair_json(c))
    except Exception:
        try:
            if debug:
                st.error("JSONs are failing to load. Trying automated repairs...")
            return json_repair.loads(_escape_control_chars_in_strings(c))
        except Exception:
            try:
                return json_repair.loads(repair_json(_escape_control_chars_in_strings(_remove_trailing_commas(c))))
            except Exception: 
                pass   

    # give up
    raise json.JSONDecodeError("Unable to parse JSON candidate", c, 0)

# --------- schema scoring / normalization ---------

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
    """
    schema example:
      {"meta_prompt": str}
      {"personality_prompt": str}
      {"facets": "list[str]"}
    Returns a dict with only the required keys if they can be satisfied; else None.
    """
    if not isinstance(obj, (dict, list)):
        return None

    # unwrap single-item list
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]

    if not isinstance(obj, dict):
        return None

    out: Dict[str, Any] = {}
    for key, t in schema.items():
        if key not in obj:
            return None
        val = obj[key]
        if t == str or t is str:
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
                # gentle coercion: a single string → [string]
                out[key] = [val]
            else:
                return None
        else:
            # simple type
            if isinstance(val, t):  # type: ignore[arg-type]
                out[key] = val
            else:
                return None
    return out


def select_best_json_candidate(
    raw_text: str,
    schema: Dict[str, Union[type, str]],
    expected_schema: Dict | None = None,
    debug: bool = False,
) -> dict:
    """Find and return the best JSON object matching ``schema``.
    If ``expected_schema`` is provided, each parsed candidate is also validated
    against it before being returned. This allows us to try additional
    candidates when the first parse yields JSON that doesn't actually conform
    to the desired structure.
    """
    text = _minimal_cleanup(
        _strip_code_fences(
            raw_text.replace("""\'""", "\u0027").replace("""\\'""", "\u0027")
        )
    )

    def _maybe_return(norm: Optional[dict]) -> Optional[dict]:
        if norm is None:
            return None
        if expected_schema and not validate_schema(norm, expected_schema):
            return None
        return norm

    # 0) Direct parse: if whole message IS the JSON
    try:
        obj = _loads_tolerant(text, debug)
        norm = _maybe_return(_normalize_to_schema(obj, schema))
        if norm is not None:
            if debug:
                st.write("Parsed {text} \n\n to get {obj} \n\n and {norm}")
            return norm
    except Exception:
        if debug:
            st.write("Failed to parse {text} \n\n to get {obj} \n\n and {norm}")

    # 0b) Try robust repair-based parser
    try:
        repaired_obj, _, _ = parse_with_repairs(
            text.replace("""\n""", "").replace("""\\n""", ""),
            required_keys=list(schema.keys()),
        )
        norm = _maybe_return(_normalize_to_schema(repaired_obj, schema))
        if norm is not None:
            return norm
    except Exception:
        if debug:
          st.write("Failed to parse {text} \n\n to get {repaired_obj} \n\n and {norm}")

    # 1) Scan for JSON substrings (objects OR arrays)
    candidates = list(iter_json_substrings(text, max_candidates=24))

    key_hints = [k.lower() for k in schema.keys()]

    def key_score(s: str) -> int:
        t = s.lower()
        return sum(1 for k in key_hints if f'"{k}"' in t)

    candidates.sort(key=lambda s: (key_score(s), len(s)), reverse=True)

    parsed: List[Tuple[dict, int]] = []  # (normalized_obj, length)

    for cand in candidates:
        try:
            cleaned_cand = cand.replace('''\n''','').replace('''\\n''','').replace('''\\"''',"\"").replace("\n","").replace("""\'""","\u0027").replace("""\\'""","\u0027").replace("”","").replace("“","")
            value = _loads_tolerant(cleaned_cand)
            if debug:
                st.write('Attempted parsing for {value}, and recieved {cand} back')
        except Exception:
            if debug:
                st.write('Failed to parse {cleaned_cand} with _loads_tolerant')
            continue
            if debug:
                raise ValueError('Failed to parse {cleaned_cand} with _loads_tolerant')
        norm = _maybe_return(_normalize_to_schema(value, schema))
        if norm is not None:
            parsed.append((norm, len(cand)))

    if parsed:
        # pick the largest candidate (usually the full object)
        parsed.sort(key=lambda t: t[1], reverse=True)
        return parsed[0][0]

    # 2) If we couldn’t match the schema but *did* see JSON, return a helpful error
    raise ValueError(
        "No JSON matching the expected schema was found. "
        "Saw {} JSON-like blocks. "
        "Candidates: {}".format(len(candidates), candidates)
    )


def validate_schema(data: Any, schema: Any) -> bool:
    """Recursively validate JSON ``data`` against ``schema``."""
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

