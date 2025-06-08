import sys
import os
import types

# Stub dependencies not available in the test environment
sys.modules['streamlit'] = types.SimpleNamespace(write=lambda *a, **k: None, code=lambda *a, **k: None)
sys.modules['requests'] = types.SimpleNamespace(post=lambda *a, **k: None)
sys.modules['json_repair'] = types.SimpleNamespace(repair_json=lambda s: s)

plotly_module = types.ModuleType("plotly")
graph_objects_module = types.ModuleType("plotly.graph_objects")
plotly_module.graph_objects = graph_objects_module
sys.modules['plotly'] = plotly_module
sys.modules['plotly.graph_objects'] = graph_objects_module

# Make repository root importable so ``lofn`` becomes a package
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

# Ensure ``config`` module resolves to ``lofn.config``
import importlib
config_module = importlib.import_module('lofn.config')
sys.modules['config'] = config_module

from lofn.helpers import parse_output


def test_preserve_newlines_and_quotes():
    output = "{'text': '{\"lyrics\": \"line1\\nline2\"}'}"
    data, error = parse_output(output, {'lyrics': str})
    assert error is None
    assert data == {'lyrics': 'line1\nline2'}


def test_extract_json_from_markdown_block():
    output = 'Here is data:\n```json\n{"song": "Can\'t Stop"}\n```\nDone.'
    data, error = parse_output(output, {'song': str})
    assert error is None
    assert data == {'song': "Can't Stop"}


def test_trailing_text_ignored():
    output = '{"foo": "bar"}\nSome trailing text.'
    data, error = parse_output(output, {'foo': str})
    assert error is None
    assert data == {'foo': 'bar'}
