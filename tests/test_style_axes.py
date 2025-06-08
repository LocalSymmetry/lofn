import sys
import os
import types

# Stub dependencies
streamlit_stub = types.SimpleNamespace(session_state={})
sys.modules['streamlit'] = streamlit_stub
sys.modules['requests'] = types.SimpleNamespace(post=lambda *a, **k: None)
sys.modules['json_repair'] = types.SimpleNamespace(repair_json=lambda s: s)

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
set_style_axes = helpers.set_style_axes


def test_set_style_axes_preserves_existing():
    streamlit_stub.session_state.clear()
    streamlit_stub.session_state['style_axes'] = {'A': 1}
    set_style_axes(True)
    assert streamlit_stub.session_state['style_axes'] == {'A': 1}


def test_set_style_axes_sets_none_when_missing():
    streamlit_stub.session_state.clear()
    set_style_axes(True)
    assert 'style_axes' in streamlit_stub.session_state
    assert streamlit_stub.session_state['style_axes'] is None


def test_set_style_axes_manual():
    streamlit_stub.session_state.clear()
    set_style_axes(False, {'B': 2})
    assert streamlit_stub.session_state['style_axes'] == {'B': 2}
