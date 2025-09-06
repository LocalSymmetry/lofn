from __future__ import annotations
import json
from typing import Any, Callable


def extract_first_json_object(text: str) -> str:
    """Return the first substring that is valid JSON.

    Some models (notably ``gpt-5`` via the Responses API) may prepend search
    citations or other text containing stray curly braces before the actual
    JSON payload.  The previous implementation grabbed the first balanced brace
    block without verifying that it was valid JSON, which meant a snippet such
    as ``{not json}`` would be returned and subsequently fail to parse.  To be
    more resilient we scan for every ``{"`` candidate and only return the first
    brace block that can be ``json.loads``ed successfully.
    """

    # Iterate over every potential opening brace and try to extract a valid
    # JSON object.  This gracefully skips over stray brace fragments that might
    # appear in search snippets or reasoning traces.
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
                        return candidate
                    except json.JSONDecodeError:
                        break  # not valid JSON, continue searching
    raise ValueError("No valid JSON object found.")


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
