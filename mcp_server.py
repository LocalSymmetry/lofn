import types
from fastapi import FastAPI
from pydantic import BaseModel

# Provide a minimal Streamlit stub so helper functions can run
try:  # pragma: no cover - optional dependency
    import streamlit as st  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - tests stub this
    class _Dummy:
        """Fallback object returning itself for any attribute or call."""

        def __init__(self, name: str = "") -> None:
            self._name = name

        def __call__(self, *args, **kwargs):
            if self._name == "columns" and args:
                return [_Dummy() for _ in range(args[0])]
            if self._name == "cache_data":
                def decorator(func):
                    return func

                return decorator
            return _Dummy()

        def __getattr__(self, name: str):
            return _Dummy(name)

        def __enter__(self):  # for context managers like st.status
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def write(self, *a, **k):
            pass

        def update(self, *a, **k):
            pass

    st = _Dummy()
    st.session_state = {}

from lofn.helpers import sample_music_genres, sample_music_frames
from lofn.llm_integration import (
    generate_concept_mediums,
    generate_video_concept_mediums,
    generate_image_prompts,
    generate_video_prompts,
    generate_music_prompts,
)

app = FastAPI(title="Lofn MCP Server")

class SampleRequest(BaseModel):
    min_count: int = 5
    max_count: int = 10


class ConceptRequest(BaseModel):
    text: str
    max_retries: int = 3
    temperature: float = 0.7
    model: str = "gpt-3.5-turbo-16k"
    reasoning_level: str = "medium"


class PromptRequest(ConceptRequest):
    concept: str
    medium: str

@app.post("/music/genres")
async def music_genres(req: SampleRequest):
    genres = sample_music_genres(req.min_count, req.max_count).splitlines()
    return {"genres": genres}

@app.post("/music/frames")
async def music_frames(req: SampleRequest):
    frames = sample_music_frames(req.min_count, req.max_count).splitlines()
    return {"frames": frames}


@app.post("/image/concepts")
async def image_concepts(req: ConceptRequest):
    pairs, style_axes, creativity = generate_concept_mediums(
        req.text,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
        style_axes=None,
        creativity_spectrum=None,
        reasoning_level=req.reasoning_level,
    )
    return {
        "pairs": pairs,
        "style_axes": style_axes,
        "creativity_spectrum": creativity,
    }


@app.post("/video/concepts")
async def video_concepts(req: ConceptRequest):
    pairs, style_axes, creativity = generate_video_concept_mediums(
        req.text,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
        style_axes=None,
        creativity_spectrum=None,
        reasoning_level=req.reasoning_level,
    )
    return {
        "pairs": pairs,
        "style_axes": style_axes,
        "creativity_spectrum": creativity,
    }


@app.post("/image/prompts")
async def image_prompts(req: PromptRequest):
    df = generate_image_prompts(
        req.text,
        req.concept,
        req.medium,
        req.max_retries,
        req.temperature,
        model=req.model,
        debug=False,
        style_axes=None,
        creativity_spectrum=None,
        reasoning_level=req.reasoning_level,
    )
    return {
        "revised_prompts": df["Revised Prompts"].tolist(),
        "synthesized_prompts": df["Synthesized Prompts"].tolist(),
    }


@app.post("/video/prompts")
async def video_prompts(req: PromptRequest):
    df = generate_video_prompts(
        req.text,
        req.concept,
        req.medium,
        req.max_retries,
        req.temperature,
        model=req.model,
        debug=False,
        style_axes=None,
        creativity_spectrum=None,
        reasoning_level=req.reasoning_level,
    )
    return {
        "revised_prompts": df["Revised Prompts"].tolist(),
        "synthesized_prompts": df["Synthesized Prompts"].tolist(),
    }


@app.post("/music/prompts")
async def music_prompts(req: ConceptRequest):
    music_prompt, lyrics_prompt, title = generate_music_prompts(
        req.text,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
        reasoning_level=req.reasoning_level,
    )
    return {
        "music_prompt": music_prompt,
        "lyrics_prompt": lyrics_prompt,
        "title": title,
    }
