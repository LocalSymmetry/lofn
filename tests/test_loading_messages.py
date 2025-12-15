import sys
import os
import types
import pytest

# Stub dependencies
streamlit_stub = types.SimpleNamespace(
    write=lambda *a, **k: None,
    code=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    session_state={},
    # cache_data decorator stub
    # Handles both @st.cache_data and @st.cache_data(...)
    cache_data=lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else lambda f: f
    ),
)
sys.modules['streamlit'] = streamlit_stub
sys.modules['requests'] = types.SimpleNamespace(
    post=lambda *a, **k: None,
    HTTPError=Exception,
    Response=object,
)
sys.modules['json_repair'] = types.SimpleNamespace(repair_json=lambda s: s)

# Stub plotly
plotly_module = types.ModuleType("plotly")
graph_objects_module = types.ModuleType("plotly.graph_objects")
plotly_module.graph_objects = graph_objects_module
sys.modules['plotly'] = plotly_module
sys.modules['plotly.graph_objects'] = graph_objects_module

# Make repo root importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

import importlib
config_module = importlib.import_module('lofn.config')
sys.modules['config'] = config_module
helpers_module = importlib.import_module('lofn.helpers')
helpers = importlib.reload(helpers_module)
get_loading_message = helpers.get_loading_message

def test_get_loading_message_returns_string():
    msg = get_loading_message("concepts")
    assert isinstance(msg, str)
    assert len(msg) > 0

def test_get_loading_message_valid_category():
    # We know "concepts" has specific messages
    msg = get_loading_message("concepts")
    assert isinstance(msg, str)
    assert len(msg) > 0

def test_get_loading_message_invalid_category_fallback():
    msg = get_loading_message("non_existent_category")
    assert isinstance(msg, str)
    assert len(msg) > 0
