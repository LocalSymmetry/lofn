import types
import os
import yaml
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

from lofn.helpers import sample_music_genres, sample_music_frames, read_prompt
from lofn.llm_integration import (
    generate_concept_mediums,
    generate_video_concept_mediums,
    generate_image_prompts,
    generate_video_prompts,
    generate_music_prompts,
    select_best_pairs,
    generate_meta_prompt,
    generate_panel_prompt,
    generate_personality_prompt,
)
from config import Config

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


def get_available_models():
    models = []
    if Config.OPENAI_API:
        models.extend([
            "gpt-4.1", "o4-mini", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o1",
            "o3-mini", "gpt-4.5-preview", "gpt-4o-mini", "gpt-4o",
            "o3-mini-2025-01-31", "o1-2024-12-17", "o1-preview", "o1-mini",
            "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "chatgpt-4o-latest",
            "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4",
        ])
    if Config.ANTHROPIC_API:
        models.extend([
            "claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514",
            "claude-opus-4-20250514", "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
        ])
    if Config.GOOGLE_API:
        models.extend([
            "gemini-2.5-pro-preview-05-06", "gemini-2.5-pro-preview-06-05",
            "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-exp-03-25",
            "gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-exp",
            "gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp",
            "gemini-1.5-flash-002", "gemini-1.5-pro-002", "gemini-exp-1206",
            "gemini-exp-1121", "gemini-exp-1114", "gemini-1.5-flash",
            "gemini-1.5-pro", "gemini-1.0-pro",
            "gemini-1.5-pro-exp-0827", "gemini-1.5-pro-exp-0801",
        ])
    if Config.POE_API:
        models.extend([
            "Poe-o1", "Poe-GPT-4.5", "Poe-o3-mini", "Poe-o3-mini-high",
            "Poe-o1-preview-128k", "Poe-o1-mini-128k", "Poe-Gemini-1.5-Pro-128k",
            "Poe-Llama-3.1-405B-FW-128k", "Poe-Gemini-1.5-Flash-128k",
            "Poe-GPT-4o-Mini-128k", "Poe-GPT-4o-128k", "Poe-Claude-3.5-Sonnet-200k",
            "Poe-Mistral-Large-2-128k", "Poe-Llama-3.2-11B-FW-131k",
            "Poe-Llama-3.2-90B-FW-131k", "Poe-Llama-3.1-8B-T-128k",
            "Poe-Llama-3.1-70B-FW-128k", "Poe-Llama-3.1-70B-T-128k",
            "Poe-Llama-3.1-8B-FW-128k", "Poe-GPT-4-Turbo-128k",
            "Poe-Claude-3-Opus-200k", "Poe-Claude-3-Sonnet-200k",
            "Poe-Claude-3-Haiku-200k", "Poe-Mixtral8x22b-Inst-FW",
            "Poe-Command-R", "Poe-Gemma-2-9b-T", "Poe-Mistral-Large-2",
            "Poe-Mistral-Medium", "Poe-Snowflake-Arctic-T", "Poe-RekaCore",
            "Poe-RekaFlash", "Poe-Command-R-Plus", "Poe-GPT-3.5-Turbo",
            "Poe-Mixtral-8x7B-Chat", "Poe-DeepSeek-Coder-33B-T",
            "Poe-CodeLlama-70B-T", "Poe-Qwen2-72B-Chat", "Poe-Qwen-72B-T",
            "Poe-Claude-2", "Poe-Google-PaLM", "Poe-Llama-3-8b-Groq",
            "Poe-Llama-3-8B-T", "Poe-Gemma-2-27b-T", "Poe-Assistant",
            "Poe-Claude-3.5-Sonnet", "Poe-GPT-4o-Mini", "Poe-GPT-4o",
            "Poe-Llama-3.1-405B-T", "Poe-Gemini-1.5-Flash",
            "Poe-Gemini-1.5-Pro", "Poe-Claude-3-Sonnet",
            "Poe-Claude-3-Haiku", "Poe-Claude-3-Opus", "Poe-Gemini-1.0-Pro",
            "Poe-Llama-3-70B-T", "Poe-Llama-3-70b-Inst-FW",
            "Poe-Llama-3.2-90B-FW-131k", "Poe-Llama-3.2-11B-FW-131k",
        ])
    priority = [
        "gemini-2.5-pro-preview-06-05",
        "claude-opus-4-20250514",
        "o3",
        "claude-sonnet-4-20250514",
    ]
    ordered = [m for m in priority if m in models]
    ordered.extend([m for m in models if m not in ordered])
    return ordered


def get_available_image_models():
    models = ["None"]
    if Config.FAL_API_KEY:
        models.extend([
            "fal-ai/flux-pro/v1.1-ultra", "fal-ai/flux-pro/v1.1", "fal-ai/recraft-v3",
            "fal-ai/omnigen-v1", "fal-ai/stable-diffusion-v35-large",
            "fal-ai/stable-diffusion-v35-medium", "fal-ai/flux-pro",
            "fal-ai/flux-realism", "fal-ai/flux-dev", "fal-ai/flux/schnell",
        ])
    if Config.IDEOGRAM_API_KEY:
        models.append("Ideogram")
    if Config.GOOGLE_PROJECT_ID:
        models.append("Google Imagen 3")
    if Config.OPENAI_API:
        models.append("DALL-E 3")
    if Config.POE_API:
        models.extend([
            "Poe-FLUX-pro-1.1-ultra", "Poe-FLUX-pro-1.1", "Poe-Imagen3",
            "Poe-StableDiffusion3.5-L", "Poe-FLUX-pro", "Poe-DALL-E-3",
            "Poe-Ideogram-v2", "Poe-Playground-v2.5", "Poe-Playground-v3",
            "Poe-Ideogram", "Poe-FLUX-dev", "Poe-FLUX-schnell",
            "Poe-LivePortrait", "Poe-StableDiffusion3", "Poe-SD3-Turbo",
            "Poe-StableDiffusionXL", "Poe-StableDiffusion3-2B",
            "Poe-SD3-Medium", "Poe-RealVisXL",
        ])
    return models


def get_available_video_models():
    models = []
    if Config.RUNWAYML_API_KEY:
        models.append("RunwayML Gen-2")
    return models


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


class CompetitionRequest(ConceptRequest):
    num_best_pairs: int = 3


@app.post("/image/competition")
async def image_competition(req: CompetitionRequest):
    pairs, _, _ = generate_concept_mediums(
        req.text,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
        style_axes=None,
        creativity_spectrum=None,
        reasoning_level=req.reasoning_level,
    )
    best = select_best_pairs(
        req.text,
        pairs,
        req.num_best_pairs,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
        reasoning_level=req.reasoning_level,
    )
    results = []
    for pair in best:
        df = generate_image_prompts(
            req.text,
            pair["concept"],
            pair["medium"],
            req.max_retries,
            req.temperature,
            model=req.model,
            debug=False,
            style_axes=None,
            creativity_spectrum=None,
            reasoning_level=req.reasoning_level,
        )
        results.append({
            "concept": pair["concept"],
            "medium": pair["medium"],
            "revised_prompts": df["Revised Prompts"].tolist(),
            "synthesized_prompts": df["Synthesized Prompts"].tolist(),
        })
    return {"results": results}


@app.post("/video/competition")
async def video_competition(req: CompetitionRequest):
    pairs, _, _ = generate_video_concept_mediums(
        req.text,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
        style_axes=None,
        creativity_spectrum=None,
        reasoning_level=req.reasoning_level,
    )
    best = select_best_pairs(
        req.text,
        pairs,
        req.num_best_pairs,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
        reasoning_level=req.reasoning_level,
    )
    results = []
    for pair in best:
        df = generate_video_prompts(
            req.text,
            pair["concept"],
            pair["medium"],
            req.max_retries,
            req.temperature,
            model=req.model,
            debug=False,
            style_axes=None,
            creativity_spectrum=None,
            reasoning_level=req.reasoning_level,
        )
        results.append({
            "concept": pair["concept"],
            "medium": pair["medium"],
            "revised_prompts": df["Revised Prompts"].tolist(),
            "synthesized_prompts": df["Synthesized Prompts"].tolist(),
        })
    return {"results": results}


@app.post("/music/competition")
async def music_competition(req: ConceptRequest):
    personality = generate_personality_prompt(
        req.text,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
        reasoning_level=req.reasoning_level,
    )
    panel = generate_panel_prompt(
        req.text,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
        reasoning_level=req.reasoning_level,
        personality_prompt=personality,
    )
    meta, frames, genres = generate_meta_prompt(
        req.text,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
        reasoning_level=req.reasoning_level,
        medium="music",
        personality_prompt=personality,
    )
    template = read_prompt("/lofn/prompts/music_overall_prompt_template.txt")
    input_text = (
        template.replace('{Meta-Prompt}', meta['meta_prompt'])
        .replace('{Panel-prompt}', panel)
        .replace('{Personality-prompt}', personality)
        .replace('{genres_list}', genres)
        .replace('{frames_list}', frames)
        .replace('{input}', req.text)
    )
    music_prompt, lyrics_prompt, title = generate_music_prompts(
        input_text,
        req.max_retries,
        req.temperature,
        req.model,
        debug=False,
    )
    return {
        "music_prompt": music_prompt,
        "lyrics_prompt": lyrics_prompt,
        "title": title,
        "personality": personality,
        "panel": panel,
        "meta_prompt": meta['meta_prompt'],
        "frames_list": frames,
        "genres_list": genres,
    }


@app.get("/models")
async def list_models():
    return {"models": get_available_models()}


@app.get("/image/models")
async def list_image_models():
    return {"models": get_available_image_models()}


@app.get("/video/models")
async def list_video_models():
    return {"models": get_available_video_models()}


@app.get("/personalities")
async def list_personalities():
    with open('/lofn/prompts/personalities.yaml', 'r') as f:
        data = yaml.safe_load(f)
    names = [p['name'] for p in data]
    return {"personalities": names}


@app.get("/panels")
async def list_panels():
    with open('/lofn/prompts/panels.yaml', 'r') as f:
        data = yaml.safe_load(f)
    names = [p['name'] for p in data]
    return {"panels": names}


@app.get("/prompts")
async def list_prompts():
    files = sorted([f for f in os.listdir('/lofn/prompts') if not f.startswith('.')])
    return {"prompts": files}


@app.get("/prompt/{name}")
async def get_prompt(name: str):
    path = os.path.join('/lofn/prompts', name)
    if not os.path.isfile(path):
        return {"error": "not found"}
    return {"prompt": read_prompt(path)}
