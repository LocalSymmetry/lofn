import sys
import os
import logging
from typing import List, Optional, Dict, Any

# Shim streamlit before importing anything that uses it
try:
    from lofn import st_shim
    sys.modules['streamlit'] = st_shim.st
except ImportError:
    pass # Should be in path

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yaml

# Import Lofn logic
from lofn.llm_integration import (
    generate_meta_prompt,
    generate_panel_prompt,
    generate_personality_prompt,
    generate_concept_mediums,
    generate_video_concept_mediums,
    generate_music_concept_mediums,
    generate_image_prompts,
    generate_video_prompts,
    generate_music_prompts,
    select_best_pairs,
    run_personality_chat,
    run_personality_image2video_chat,
)
from lofn.ui import LofnApp
from lofn.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lofn API", description="API for Lofn AI Art Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class ConceptRequest(BaseModel):
    input_text: str
    medium_type: str = "image" # image, video, music
    model: str
    temperature: float = 0.7
    max_retries: int = 3
    reasoning_level: str = "medium"
    competition_mode: bool = False
    personality_prompt: Optional[str] = None
    panel_prompt: Optional[str] = None
    images: Optional[List[str]] = None # Base64 data URLs
    style_axes: Optional[Dict[str, int]] = None
    creativity_spectrum: Optional[Dict[str, int]] = None

class ConceptResponse(BaseModel):
    concepts: List[Dict[str, Any]]
    style_axes: Optional[Dict[str, Any]]
    creativity_spectrum: Optional[Dict[str, Any]]
    meta_prompt: Optional[str] = None
    panel_prompt: Optional[str] = None
    personality_prompt: Optional[str] = None

class BestPairsRequest(BaseModel):
    input_text: str
    pairs: List[Dict[str, str]]
    num_best_pairs: int = 3
    model: str
    temperature: float = 0.7
    max_retries: int = 3
    reasoning_level: str = "medium"

class PromptRequest(BaseModel):
    input_text: str
    concept: str
    medium: str
    medium_type: str = "image"
    model: str
    temperature: float = 0.7
    max_retries: int = 3
    reasoning_level: str = "medium"
    style_axes: Optional[Dict[str, Any]] = None
    creativity_spectrum: Optional[Dict[str, Any]] = None
    images: Optional[List[str]] = None

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] # format: {"role": "user"|"assistant", "content": ...}
    personality_prompt: str
    model: str
    temperature: float = 0.7
    reasoning_level: str = "medium"
    images: Optional[List[str]] = None
    mode: str = "chat" # chat or image2video

class ConfigResponse(BaseModel):
    models: List[str]
    image_models: List[str]
    personalities: List[Dict[str, str]]
    panels: List[Dict[str, str]]

# --- Helpers ---

def get_app_instance():
    # Instantiate LofnApp to get configs/models
    # Since we shimmed st, this is safe-ish
    return LofnApp()

# --- Endpoints ---

@app.get("/api/config", response_model=ConfigResponse)
def get_config():
    lofn = get_app_instance()

    # Load personalities
    personalities = []
    try:
        with open('lofn/prompts/personalities.yaml', 'r') as f:
            personalities = yaml.safe_load(f)
        if os.path.exists('lofn/prompts/custom_personalities.yaml'):
             with open('lofn/prompts/custom_personalities.yaml', 'r') as f:
                custom = yaml.safe_load(f) or []
                personalities[:0] = custom
    except Exception as e:
        logger.error(f"Error loading personalities: {e}")

    # Load panels
    panels = []
    try:
        with open('lofn/prompts/panels.yaml', 'r') as f:
            panels = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading panels: {e}")

    return ConfigResponse(
        models=lofn.available_models,
        image_models=lofn.available_image_models,
        personalities=personalities or [],
        panels=panels or []
    )

@app.post("/api/generate/concepts", response_model=ConceptResponse)
def generate_concepts_endpoint(req: ConceptRequest):
    try:
        # Pre-generation: Meta, Panel, Personality if needed
        meta_prompt_text = None
        panel_prompt_text = req.panel_prompt
        personality_prompt_text = req.personality_prompt

        input_text = req.input_text
        images = req.images

        if req.competition_mode:
            # Generate Personality if missing
            if not personality_prompt_text:
                 personality_prompt_text = generate_personality_prompt(
                    req.input_text, req.max_retries, req.temperature, req.model,
                    debug=False, reasoning_level=req.reasoning_level, input_images=images
                )

            # Generate Meta Prompt
            meta_res, frames, styles = generate_meta_prompt(
                req.input_text, req.max_retries, req.temperature, req.model,
                debug=False, reasoning_level=req.reasoning_level,
                medium=req.medium_type, personality_prompt=personality_prompt_text,
                input_images=images
            )
            meta_prompt_text = meta_res.get('meta_prompt')

            # Generate Panel if missing
            if not panel_prompt_text:
                panel_prompt_text = generate_panel_prompt(
                    meta_prompt_text, req.max_retries, req.temperature, req.model,
                    debug=False, reasoning_level=req.reasoning_level,
                    personality_prompt=personality_prompt_text, input_images=images
                )

            # Construct enhanced input text
            template_path = 'lofn/prompts/overall_prompt_template.txt'
            if req.medium_type == 'video':
                template_path = 'lofn/prompts/video_overall_prompt_template.txt'
            elif req.medium_type == 'music':
                template_path = 'lofn/prompts/music_overall_prompt_template.txt'

            with open(template_path, 'r') as f:
                template = f.read()

            def join_list(l): return ", ".join(l) if isinstance(l, list) else str(l)

            image_context_str = "\n".join(images) if images else ""

            replacements = {
                '{Meta-Prompt}': meta_prompt_text or "",
                '{Panel-prompt}': panel_prompt_text or "",
                '{Personality-prompt}': personality_prompt_text or "",
                '{frames_list}': join_list(frames),
                '{input}': req.input_text,
                '{image_context}': image_context_str
            }
            if req.medium_type == 'video':
                replacements['{film_styles_list}'] = join_list(styles)
            elif req.medium_type == 'music':
                replacements['{genres_list}'] = join_list(styles)
            else:
                replacements['{art_styles_list}'] = join_list(styles)

            for k, v in replacements.items():
                template = template.replace(k, v)

            input_text = template
        else:
             if images:
                 input_text = f"{input_text}\n\nReference Images:\n" + "\n".join(images)

        # Generate Concepts
        if req.medium_type == 'video':
             concepts, style_axes, creativity_spectrum = generate_video_concept_mediums(
                input_text, req.max_retries, req.temperature, req.model,
                debug=False, style_axes=req.style_axes, creativity_spectrum=req.creativity_spectrum,
                reasoning_level=req.reasoning_level, input_images=images
            )
        elif req.medium_type == 'music':
             concepts, style_axes, creativity_spectrum = generate_music_concept_mediums(
                input_text, req.max_retries, req.temperature, req.model,
                debug=False, style_axes=req.style_axes, creativity_spectrum=req.creativity_spectrum,
                reasoning_level=req.reasoning_level, input_images=images
            )
        else:
            concepts, style_axes, creativity_spectrum = generate_concept_mediums(
                input_text, req.max_retries, req.temperature, req.model,
                debug=False, style_axes=req.style_axes, creativity_spectrum=req.creativity_spectrum,
                reasoning_level=req.reasoning_level, input_images=images
            )

        return ConceptResponse(
            concepts=concepts,
            style_axes=style_axes,
            creativity_spectrum=creativity_spectrum,
            meta_prompt=meta_prompt_text,
            panel_prompt=panel_prompt_text,
            personality_prompt=personality_prompt_text
        )

    except Exception as e:
        logger.exception("Error in generate_concepts")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/best_pairs", response_model=List[Dict[str, str]])
def select_best_pairs_endpoint(req: BestPairsRequest):
    try:
        best = select_best_pairs(
            req.input_text, req.pairs, req.num_best_pairs,
            req.max_retries, req.temperature, req.model,
            debug=False, reasoning_level=req.reasoning_level
        )
        return best
    except Exception as e:
        logger.exception("Error in select_best_pairs")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/prompts")
def generate_prompts_endpoint(req: PromptRequest):
    try:
        if req.medium_type == 'video':
            res = generate_video_prompts(
                req.input_text, req.concept, req.medium,
                req.max_retries, req.temperature, req.model,
                debug=False, style_axes=req.style_axes,
                creativity_spectrum=req.creativity_spectrum,
                reasoning_level=req.reasoning_level, input_images=req.images
            )
        elif req.medium_type == 'music':
            res = generate_music_prompts(
                req.input_text, req.concept, req.medium,
                req.max_retries, req.temperature, req.model,
                debug=False, style_axes=req.style_axes,
                creativity_spectrum=req.creativity_spectrum,
                reasoning_level=req.reasoning_level, input_images=req.images
            )
        else:
            res = generate_image_prompts(
                req.input_text, req.concept, req.medium,
                req.max_retries, req.temperature, req.model,
                debug=False, style_axes=req.style_axes,
                creativity_spectrum=req.creativity_spectrum,
                reasoning_level=req.reasoning_level, input_images=req.images
            )

        # Convert DataFrame to dict if necessary (image/video returns DF)
        if hasattr(res, 'to_dict'):
            return res.to_dict(orient='records')
        elif isinstance(res, dict):
             # Music returns a dict with lists
             return res
        return res

    except Exception as e:
        logger.exception("Error in generate_prompts")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        # Convert history to LangChain messages?
        # run_personality_chat expects a list of Messages.
        from langchain.schema import HumanMessage, AIMessage

        chat_history = []
        for msg in req.history:
            if msg['role'] == 'user':
                chat_history.append(HumanMessage(content=msg['content']))
            else:
                chat_history.append(AIMessage(content=msg['content']))

        # Handle images
        input_media = []
        if req.images:
            for img in req.images:
                input_media.append({"type": "image_url", "image_url": {"url": img}})

        if req.mode == 'image2video':
            response = run_personality_image2video_chat(
                req.personality_prompt, chat_history, req.message,
                model=req.model, temperature=req.temperature,
                reasoning_level=req.reasoning_level, debug=False,
                input_media=input_media
            )
        else:
            response = run_personality_chat(
                req.personality_prompt, chat_history, req.message,
                model=req.model, temperature=req.temperature,
                reasoning_level=req.reasoning_level, debug=False,
                input_media=input_media
            )

        return {"response": response}
    except Exception as e:
        logger.exception("Error in chat")
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend assets
if os.path.exists("frontend/dist/assets"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

# SPA catch-all
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="Not Found")

    file_path = os.path.join("frontend/dist", full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)

    index_path = "frontend/dist/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return "Frontend build not found. Please run 'npm run build' in 'frontend/'.", 404

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
