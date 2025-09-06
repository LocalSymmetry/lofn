import sys
import os
import types

# Stub dependencies not available in the test environment
sys.modules['streamlit'] = types.SimpleNamespace(
    write=lambda *a, **k: None,
    code=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    session_state={},
    cache_data=lambda *a, **k: (lambda f: f),
)
sys.modules['requests'] = types.SimpleNamespace(
    post=lambda *a, **k: None,
    HTTPError=Exception,
    Response=object,
)

plotly_module = types.ModuleType("plotly")
graph_objects_module = types.ModuleType("plotly.graph_objects")
plotly_module.graph_objects = graph_objects_module
sys.modules['plotly'] = plotly_module
sys.modules['plotly.graph_objects'] = graph_objects_module

# Make repository root importable so ``lofn`` becomes a package
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from lofn.parsing import extract_first_json_object, parse_strict_json, coerce_common_forms


def test_extract_first_json_object():
    text = "prefix {\"a\":1} suffix {\"b\":2}"
    assert extract_first_json_object(text) == '{"a":1}'


def test_extract_skips_non_json_blocks():
    text = "pre {not json} post {\"ok\":1}"
    assert extract_first_json_object(text) == '{"ok":1}'


def test_parse_strict_json_direct_and_extract():
    assert parse_strict_json('{"x":1}') == {"x":1}
    text = "noise before {\"x\":2} trailing"
    assert parse_strict_json(text) == {"x":2}


def test_parse_skips_invalid_blocks():
    text = "intro {bad} middle {\"y\":3}"
    assert parse_strict_json(text) == {"y":3}


def test_parse_strict_json_validation():
    def validator(obj):
        if "foo" not in obj:
            raise ValueError("missing foo")
    result = parse_strict_json('{"foo":"bar"}', validate=validator)
    assert result == {"foo": "bar"}


def test_coerce_common_forms():
    obj = [{"metaPrompt": "x"}]
    coerced = coerce_common_forms(obj, {"meta_prompt"})
    assert coerced == {"meta_prompt": "x"}
