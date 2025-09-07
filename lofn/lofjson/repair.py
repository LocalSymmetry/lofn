from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["parse_with_repairs", "parse_with_repairs_v2", "repair_json_string"]

def _strip_code_fences(text: str) -> str:
    return re.sub(r"```(?:json|JSON|python|[\w.+-]+)?\s*([\s\S]*?)```", r"\1", text)

def _strip_prefix_suffix_wrappers(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^\s*(?:json|JSON)\s*[:\n\\n]*", "", t)
    t = re.sub(r"^\s*(?:content|output)\s*=\s*", "", t)
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        inner = t[1:-1].strip()
        if inner.startswith("{") or inner.startswith("["):
            t = inner
    return t

def _extract_json_candidates(text: str) -> List[str]:
    cands: List[str] = []
    in_string = False
    escape = False
    quote_char: Optional[str] = None
    depth_brace = 0
    depth_bracket = 0
    start = None
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch in ('"', "'"):
            if not in_string:
                in_string = True
                quote_char = ch
            elif quote_char == ch:
                in_string = False
                quote_char = None
            continue
        if in_string:
            continue
        if ch == '{':
            if depth_brace == 0 and depth_bracket == 0:
                start = i
            depth_brace += 1
        elif ch == '}':
            if depth_brace > 0:
                depth_brace -= 1
                if depth_brace == 0 and depth_bracket == 0 and start is not None:
                    cands.append(text[start:i+1])
                    start = None
        elif ch == '[':
            if depth_brace == 0 and depth_bracket == 0:
                start = i
            depth_bracket += 1
        elif ch == ']':
            if depth_bracket > 0:
                depth_bracket -= 1
                if depth_bracket == 0 and depth_brace == 0 and start is not None:
                    cands.append(text[start:i+1])
                    start = None
    return cands

def _try_parse_json(candidate: str):
    try:
        return json.loads(candidate), None
    except Exception as e:
        return None, e

def _remove_json_comments(json_like: str) -> str:
    out: List[str] = []
    i = 0
    n = len(json_like)
    in_string = False
    quote_char: Optional[str] = None
    escape = False
    while i < n:
        ch = json_like[i]
        nxt = json_like[i+1] if i+1 < n else ''
        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            i += 1
            continue
        if ch in ('"', "'") and not in_string:
            in_string = True
            quote_char = ch
            out.append(ch)
            i += 1
            continue
        if in_string and ch == quote_char:
            in_string = False
            quote_char = None
            out.append(ch)
            i += 1
            continue
        if not in_string and ch == '/' and nxt == '/':
            i += 2
            while i < n and json_like[i] not in ('\n', '\r'):
                i += 1
            continue
        if not in_string and ch == '/' and nxt == '*':
            i += 2
            while i+1 < n and not (json_like[i] == '*' and json_like[i+1] == '/'):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return ''.join(out)

def _remove_trailing_commas(json_like: str) -> str:
    out: List[str] = []
    in_string = False
    escape = False
    quote_char: Optional[str] = None
    for i, ch in enumerate(json_like):
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            continue
        if ch in ('"', "'") and not in_string:
            in_string = True
            quote_char = ch
            out.append(ch)
            continue
        if in_string and ch == quote_char:
            in_string = False
            quote_char = None
            out.append(ch)
            continue
        if not in_string and ch == ',':
            j = i + 1
            while j < len(json_like) and json_like[j] in " \t\r\n":
                j += 1
            if j < len(json_like) and json_like[j] in '}]':
                continue
        out.append(ch)
    return ''.join(out)

def _replace_single_quotes(json_like: str) -> str:
    if json_like.count("'") > json_like.count('"') * 1.3:
        def repl(m):
            s = m.group(0)
            inner = s[1:-1].replace('"', '\\"').replace("\\'", "'")
            return f'"{inner}"'
        json_like = re.sub(r"'([^'\\]|\\.)*'", repl, json_like)
    json_like = re.sub(r"(?P<pre>[\{,\s])'(?P<key>[^'\n\r\\]+)'\s*:", r'\g<pre>"\g<key>":', json_like)
    return json_like

def _escape_control_chars_in_strings(json_like: str) -> str:
    CONTROL_MAP = {"\r": "\\r", "\n": "\\n", "\t": "\\t", "\b": "\\b", "\f": "\\f"}
    out: List[str] = []
    in_string = False
    escape = False
    quote_char: Optional[str] = None
    for ch in json_like:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            continue
        if ch in ('"', "'") and not in_string:
            in_string = True
            quote_char = ch
            out.append(ch)
            continue
        if in_string and ch == quote_char:
            in_string = False
            quote_char = None
            out.append(ch)
            continue
        if in_string and ch in CONTROL_MAP:
            out.append(CONTROL_MAP[ch])
            continue
        out.append(ch)
    return ''.join(out)

def _fix_identifiers(json_like: str) -> str:
    json_like = re.sub(r'(?<=[:\s\[,\{\(])None(?=[\s,\]}])', 'null', json_like)
    json_like = re.sub(r'(?<=[:\s\[,\{\(])True(?=[\s,\]}])', 'true', json_like)
    json_like = re.sub(r'(?<=[:\s\[,\{\(])False(?=[\s,\]}])', 'false', json_like)
    return json_like

def repair_json_string(json_like: str) -> str:
    json_like = _strip_code_fences(json_like)
    json_like = _strip_prefix_suffix_wrappers(json_like)
    json_like = _remove_json_comments(json_like)
    json_like = _replace_single_quotes(json_like)
    json_like = _fix_identifiers(json_like)
    json_like = _escape_control_chars_in_strings(json_like)
    json_like = _remove_trailing_commas(json_like)
    return json_like

def _find_with_required_keys(obj: Any, required_keys: Sequence[str], deep_scan: bool) -> Optional[dict]:
    if isinstance(obj, dict) and any(k in obj for k in required_keys):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            got = _find_with_required_keys(v, required_keys, deep_scan)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find_with_required_keys(v, required_keys, deep_scan)
            if got is not None:
                return got
    if deep_scan and isinstance(obj, str) and ('{' in obj or '[' in obj):
        inner_obj, _, _ = _parse_with_repairs_impl(obj, required_keys, deep_scan=False)
        if isinstance(inner_obj, dict) and any(k in inner_obj for k in required_keys):
            return inner_obj
    return None

def _parse_with_repairs_impl(text: str, required_keys: Sequence[str], deep_scan: bool = True):
    text_norm = _strip_prefix_suffix_wrappers(_strip_code_fences(text))
    candidates = _extract_json_candidates(text_norm)
    if text_norm.strip():
        candidates.insert(0, text_norm)
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        key = c[:120] + str(len(c))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    logs: List[Tuple[str, str, str]] = []
    best_obj: Optional[dict] = None
    best_repaired: Optional[str] = None
    for cand in uniq:
        repaired = repair_json_string(cand)
        obj, err = _try_parse_json(repaired)
        logs.append((cand[:200], "parsed" if err is None else f"error:{err}", repaired[:200]))
        if obj is not None:
            found = _find_with_required_keys(obj, required_keys, deep_scan)
            if found is not None:
                return found, repaired, logs
            if best_obj is None and isinstance(obj, dict):
                best_obj = obj
                best_repaired = repaired
    return best_obj, best_repaired, logs

def parse_with_repairs(raw_text: str, required_keys: Sequence[str] = (), deep_scan: bool = True):
    obj, repaired, logs = _parse_with_repairs_impl(raw_text, required_keys, deep_scan)
    return obj, repaired, logs

def parse_with_repairs_v2(raw_text: str, required_keys: Sequence[str] = (), deep_scan: bool = True):
    obj, repaired, _ = _parse_with_repairs_impl(raw_text, required_keys, deep_scan)
    return obj, repaired
