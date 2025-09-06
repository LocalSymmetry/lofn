from __future__ import annotations
import json
from typing import Any, Callable, Iterable


def extract_json_objects(text: str) -> Iterable[str]:
    """Yield all substrings within ``text`` that are valid JSON objects."""

    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        yield candidate
                    except json.JSONDecodeError:
                        pass
                    break


def extract_first_json_object(text: str) -> str:
    """Return the first valid JSON object found within ``text``."""

    for obj in extract_json_objects(text):
        return obj
    raise ValueError("No valid JSON object found.")


def parse_strict_json(text: str, validate: Callable[[Any], None] | None = None) -> Any:
    """Strictly parse ``text`` as JSON, optionally validating the result.

    If multiple JSON objects are present, the first one that parses and passes
    ``validate`` (when supplied) is returned.
    """

    def try_parse(candidate: str) -> Any:
        obj = json.loads(candidate)
        if validate:
            validate(obj)
        return obj

    try:
        return try_parse(text)
    except (json.JSONDecodeError, ValueError):
        pass

    for candidate in extract_json_objects(text):
        try:
            return try_parse(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    raise ValueError("No valid JSON object found.")


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
