import types
from fastapi import FastAPI
from pydantic import BaseModel

# Provide a minimal Streamlit stub so helper functions can run
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - tests stub this
    class _DummyStatus:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def write(self, *a, **k):
            pass
        def update(self, *a, **k):
            pass
    st = types.SimpleNamespace(
        status=lambda *a, **k: _DummyStatus(),
        write=lambda *a, **k: None,
        session_state={},
    )

from lofn.helpers import sample_music_genres, sample_music_frames

app = FastAPI(title="Lofn MCP Server")

class SampleRequest(BaseModel):
    min_count: int = 5
    max_count: int = 10

@app.post("/music/genres")
async def music_genres(req: SampleRequest):
    genres = sample_music_genres(req.min_count, req.max_count).splitlines()
    return {"genres": genres}

@app.post("/music/frames")
async def music_frames(req: SampleRequest):
    frames = sample_music_frames(req.min_count, req.max_count).splitlines()
    return {"frames": frames}
