import sys
import os
import types

# Stub dependencies
streamlit_stub = types.SimpleNamespace(
    write=lambda *a, **k: None,
    code=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    session_state={},
    cache_data=lambda *a, **k: (lambda f: f),
)
sys.modules['streamlit'] = streamlit_stub
sys.modules['requests'] = types.SimpleNamespace(
    post=lambda *a, **k: None,
    HTTPError=Exception,
    Response=object,
)
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
from lofn.helpers import sample_music_genres


def test_sample_music_genres_returns_lines():
    data = sample_music_genres(min_count=5, max_count=5)
    lines = data.split('\n')
    assert len(lines) == 5
    assert all(line for line in lines)
