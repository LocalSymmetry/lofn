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

from lofn.helpers import async_to_sync_generator
import asyncio


async def dummy_stream():
    for token in ["a", "b", "c"]:
        yield token


def test_async_to_sync_generator():
    gen = async_to_sync_generator(dummy_stream())
    assert list(gen) == ["a", "b", "c"]
