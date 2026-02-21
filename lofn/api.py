import threading
import queue
import json
import time
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from lofn import llm_integration, image_generation, ui
from lofn.mock_streamlit import StreamlitMock, patch_streamlit
from lofn.schemas import (
    ConceptRequest, PromptRequest, ImageGenerationRequest,
    StyleAxes, CreativitySpectrum
)

app = FastAPI(title="Lofn Creative Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def stream_generator(func, *args, **kwargs):
    """
    Runs the provided function in a separate thread while capturing
    Streamlit logs via StreamlitMock. Yields JSON strings (NDJSON).
    """
    mock = StreamlitMock()
    result_queue = queue.Queue()

    def target():
        try:
            with patch_streamlit(mock):
                res = func(*args, **kwargs)
                # If the function returns something that isn't JSON-serializable, we might have issues.
                # llm_integration functions usually return (data, style_axes, spectrum) or DataFrames.
                # We need to handle DataFrames.
                if hasattr(res, 'to_dict'): # DataFrame
                    res = res.to_dict(orient='records')
                elif isinstance(res, tuple):
                    # Convert tuple to list for JSON
                    res = list(res)
                    # Handle DataFrames inside tuple
                    new_res = []
                    for item in res:
                        if hasattr(item, 'to_dict'):
                            new_res.append(item.to_dict(orient='records'))
                        else:
                            new_res.append(item)
                    res = new_res

                result_queue.put({"type": "result", "content": res})
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            mock.error(error_msg)
            result_queue.put({"type": "exception", "content": str(e)})
        finally:
            result_queue.put(None) # Sentinel

    thread = threading.Thread(target=target)
    thread.start()

    while True:
        # 1. Drain the log queue
        while True:
            try:
                msg = mock.queue.get_nowait()
                yield json.dumps(msg) + "\n"
            except queue.Empty:
                break

        # 2. Check for result or sentinel
        try:
            res = result_queue.get_nowait()
            if res is None: # Sentinel received, thread finished
                break
            yield json.dumps(res) + "\n"
        except queue.Empty:
            pass

        if not thread.is_alive() and mock.queue.empty() and result_queue.empty():
            break

        time.sleep(0.05)

@app.post("/api/generate/concepts")
async def generate_concepts(req: ConceptRequest):
    kwargs = {
        "input_text": req.input_text,
        "max_retries": req.max_retries,
        "temperature": req.temperature,
        "model": req.model,
        "verbose": False,
        "debug": True,
        "aesthetics": llm_integration.aesthetics,
        "style_axes": req.style_axes,
        "creativity_spectrum": req.creativity_spectrum.dict() if req.creativity_spectrum else None,
        "medium": req.medium,
        "reasoning_level": req.reasoning_level,
        "input_images": req.input_images
    }

    return StreamingResponse(
        stream_generator(llm_integration.generate_concept_mediums, **kwargs),
        media_type="application/x-ndjson"
    )

@app.post("/api/generate/prompts")
async def generate_prompts(req: PromptRequest):
    # Common arguments
    kwargs = {
        "input_text": req.input_text,
        "max_retries": req.max_retries,
        "temperature": req.temperature,
        "model": req.model,
        "debug": True,
        "style_axes": req.style_axes,
        "creativity_spectrum": req.creativity_spectrum.dict() if req.creativity_spectrum else None,
        "reasoning_level": req.reasoning_level,
        "input_images": req.input_images
    }

    target_func = llm_integration.generate_image_prompts

    if req.medium == 'video':
        target_func = llm_integration.generate_video_prompts
        kwargs["concept"] = req.concept
        kwargs["medium"] = req.medium
    elif req.medium == 'music':
        target_func = llm_integration.generate_music_prompts
        kwargs["concept"] = req.concept
        kwargs["arrangement"] = req.medium # Map medium to arrangement for music
    elif req.medium == 'story':
        # Story not fully supported in this V1 API path as it needs essence
        # For now, we will fallback to image generation or raise error
        # But to be safe, let's try calling it with medium as medium
        target_func = llm_integration.generate_story_prompts
        kwargs["concept"] = req.concept
        kwargs["medium"] = req.medium
        kwargs["essence"] = "" # Mock essence
    else:
        # Default to image
        kwargs["concept"] = req.concept
        kwargs["medium"] = req.medium

    return StreamingResponse(
        stream_generator(target_func, **kwargs),
        media_type="application/x-ndjson"
    )

@app.post("/api/generate/image")
async def generate_image_api(req: ImageGenerationRequest):
    mock = StreamlitMock()

    def worker():
        with patch_streamlit(mock):
            # Populate session state with defaults and overrides
            key_prefix = req.image_model
            mock.session_state[f"{key_prefix}_num_images"] = req.num_images
            mock.session_state[f"{key_prefix}_image_size"] = req.image_size

            for k, v in req.extra_params.items():
                mock.session_state[f"{key_prefix}_{k}"] = v

            # Now call get_model_params
            params = image_generation.get_model_params(req.image_model)
            params['prompt'] = req.prompt

            # Call generate_image
            results = image_generation.generate_image(
                req.image_model,
                params,
                debug=True
            )
            return results

    return StreamingResponse(
        stream_generator(worker),
        media_type="application/x-ndjson"
    )

@app.get("/api/models")
def get_models():
    # Helper to get available models
    # We can instantiate LofnApp to get them or just replicate logic
    app_instance = ui.LofnApp()
    return {
        "llm_models": app_instance.available_models,
        "image_models": app_instance.available_image_models,
        "video_models": app_instance.get_available_video_models(),
    }

# Serve Frontend
frontend_path = "/lofn/frontend/dist"
if not os.path.exists(frontend_path):
    # Fallback to local relative path for development outside Docker
    frontend_path = os.path.join(os.path.dirname(__file__), "../frontend/dist")

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
else:
    print(f"Frontend path {frontend_path} not found. Running in API-only mode.")

# Serve Generated Media
os.makedirs("/images", exist_ok=True)
os.makedirs("/videos", exist_ok=True)
os.makedirs("/music", exist_ok=True)
app.mount("/images", StaticFiles(directory="/images"), name="images")
app.mount("/videos", StaticFiles(directory="/videos"), name="videos")
app.mount("/music", StaticFiles(directory="/music"), name="music")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
