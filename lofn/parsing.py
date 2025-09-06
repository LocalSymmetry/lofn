from __future__ import annotations
import json
from typing import Any, Callable


def extract_first_json_object(text: str) -> str:
    """Grab the first balanced {...} block. No rewriting, no numbering of array items."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found.")
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unbalanced JSON braces.")


def parse_strict_json(text: str, validate: Callable[[Any], None] | None = None) -> Any:
    """Strictly parse ``text`` as JSON, optionally validating the result."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = json.loads(extract_first_json_object(text))
    if validate:
        validate(obj)
    return obj


def coerce_common_forms(obj: Any, expected_keys: set[str]) -> Any:
    """Perform gentle, idempotent coercions of common near-miss structures."""
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]
    if isinstance(obj, dict):
        synonyms = {"metaPrompt": "meta_prompt", "facetsList": "facets"}
        for k, v in list(obj.items()):
            if k in synonyms and synonyms[k] not in obj:
                obj[synonyms[k]] = v
                del obj[k]
    return obj


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
