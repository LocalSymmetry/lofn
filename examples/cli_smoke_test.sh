#!/bin/bash
# Basic CLI smoke test for Lofn

set -euo pipefail

python - <<'PY'
import os, sys, importlib, types
# Ensure repository root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

# Stub heavy dependencies if unavailable
sys.modules.setdefault('streamlit', types.SimpleNamespace(write=lambda *a, **k: None, session_state={} ))
sys.modules.setdefault('requests', types.SimpleNamespace(post=lambda *a, **k: None))
sys.modules.setdefault('json_repair', types.SimpleNamespace(repair_json=lambda s: s))

plotly_module = types.ModuleType("plotly")
graph_objects_module = types.ModuleType("plotly.graph_objects")
plotly_module.graph_objects = graph_objects_module
sys.modules.setdefault('plotly', plotly_module)
sys.modules.setdefault('plotly.graph_objects', graph_objects_module)

# Ensure 'config' resolves to lofn.config
config_module = importlib.import_module('lofn.config')
sys.modules['config'] = config_module

from lofn.helpers import sample_music_genres, sample_music_frames

# Generate small samples to ensure functions work
genres = sample_music_genres(min_count=2, max_count=2)
frames = sample_music_frames(min_count=2, max_count=2)

assert len(genres.splitlines()) == 2, "Unexpected genre count"
assert len(frames.splitlines()) == 2, "Unexpected frame count"

print("CLI smoke test passed")
PY

