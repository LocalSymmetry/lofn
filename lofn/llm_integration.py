# llm_integration.py

import streamlit as st
import openai
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None
import asyncio
import fastapi_poe as fp
import requests
import json
from langchain.chains.structured_output.base import create_structured_output_runnable
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic.experimental import ChatAnthropicTools
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chat_models.base import BaseChatModel
from langchain.schema import OutputParserException
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage, ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import PrivateAttr
from langchain_core.language_models.llms import LLM
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import imghdr
from config import Config
try:
    from helpers import (
        read_prompt,
        send_to_discord,
        display_facets,
        display_creativity_and_style_axes,
        sample_artistic_frames,
        sample_video_frames,
        sample_music_frames,
        sample_music_genres,
        sample_art_styles,
        sample_film_styles,
        compress_image_bytes,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for package import
    from .helpers import (
        read_prompt,
        send_to_discord,
        display_facets,
        display_creativity_and_style_axes,
        sample_artistic_frames,
        sample_video_frames,
        sample_music_frames,
        sample_music_genres,
        sample_art_styles,
        sample_film_styles,
        compress_image_bytes,
    )

from parsing import (
        select_best_json_candidate,
        validate_schema,
)
import plotly.graph_objects as go
import random
import numpy as np
import pandas as pd
import logging
import base64
import mimetypes
from PIL import Image
import io
try:
    from helpers import (
        display_temporary_results,
        display_temporary_results_no_expander,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for package import
    from .helpers import (
        display_temporary_results,
        display_temporary_results_no_expander,
    )
import openai  # For the advanced "o1" usage if needed
from openai import OpenAI
try:
    from o1_integration import *  # noqa: F401,F403
except ModuleNotFoundError:  # pragma: no cover - fallback for package import
    from .o1_integration import *  # noqa: F401,F403
from pathlib import Path
try:
    from utils.image_io import normalize_image_bytes, to_data_url
except ModuleNotFoundError:  # pragma: no cover - fallback for package import
    from .utils.image_io import normalize_image_bytes, to_data_url

class LofnError(Exception):
    """Custom exception class for Lofn-specific errors."""
    pass


logger = logging.getLogger(__name__)

@dataclass
class ImageAsset:
    """Simple container for inline image data."""

    data: bytes
    mime: str
    data_url: str


def _detect_mime(data: bytes) -> str:
    """Best-effort MIME type detection for binary image data."""

    kind = imghdr.what(None, h=data) or "png"
    # normalize to common standards
    if kind in ("jpg", "jpeg"):
        return "image/jpeg"
    if kind == "png":
        return "image/png"
    if kind == "webp":
        return "image/webp"
    return f"image/{kind}"


def to_data_url(data: bytes, mime: Optional[str] = None) -> str:
    """Convert raw bytes into a data URL."""

    mime = (mime or _detect_mime(data)).lower()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _to_jpeg_max1024(raw: bytes) -> Tuple[bytes, str]:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    scale = min(1.0, 1024 / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80, optimize=True)
    return buf.getvalue(), "image/jpeg"


def normalize_images(files) -> List[ImageAsset]:
    """Convert an iterable of uploaded files to ``ImageAsset`` objects."""

    assets: List[ImageAsset] = []
    for f in files or []:
        try:
            raw = f.read() if hasattr(f, "read") else bytes(f)
            data, mime = _to_jpeg_max1024(raw)
            assets.append(ImageAsset(data=data, mime=mime, data_url=to_data_url(data, mime)))
        except Exception:
            continue
    return assets


# ---------------------------------------------------------------------------
# Model capability guardrails
# ---------------------------------------------------------------------------

VISION_MODELS = {
    "openai": {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o3-pro", "o4-mini"},
    "anthropic": {
        "claude-3-7-sonnet-20250219", 
        "claude-sonnet-4-20250514", 
        "claude-opus-4-20250514", 
        "claude-3-5-sonnet-latest", 
        "claude-3-5-haiku-20241022", 
        "claude-3-5-sonnet-20241022", 
        "claude-3-5-sonnet-20240620", 
        "claude-3-opus-20240229", 
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307"
    },
    "google": {
        "gemini-2.5-pro", 
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-pro",
    },
    "openrouter_like_openai_schema": {"*"},
    "poe": {"*"},
}

def _supports_vision(model_name: str) -> bool:
    for models in VISION_MODELS.values():
        if "*" in models or model_name in models:
            return True
    return False


def _has_image_parts(msgs: List[HumanMessage]) -> bool:
    for m in msgs:
        content = getattr(m, "content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "file"):
                    return True
    return False


def ensure_vision_capable(provider: str, model: str, has_images: bool) -> None:
    if not has_images:
        return
    provider = provider.lower()
    if provider == "openai" and model.startswith(("o1", "o3")):
        raise ValueError(f"Model {model} does not support images. Choose gpt-4o / gpt-4o-mini.")
    if provider in ("openai", "anthropic", "google"):
        known = any(model.startswith(m) or model == m for m in VISION_MODELS[provider])
        if not known:
            raise ValueError(f"Selected {provider}:{model} may not support images.")


# ---------------------------------------------------------------------------
# Token counting helpers (best-effort, many return ``None``)
# ---------------------------------------------------------------------------


def count_tokens_openai(text: str) -> Optional[int]:
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except Exception:
        return None


def count_tokens_anthropic(text: str) -> Optional[int]:
    return None


def count_tokens_google_gemini(text: str) -> Optional[int]:
    return None


def count_tokens_openrouter(text: str) -> Optional[int]:
    return None


def count_tokens_poe(text: str) -> Optional[int]:
    return None


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------


TEXT_PART_KEYS = {"text", "input_text", "output_text"}


def _to_str(x: Any) -> str:
    """Conservatively convert arbitrary objects to strings."""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        chunks = []
        for item in x:
            if isinstance(item, str):
                chunks.append(item)
            else:
                chunks.append(json.dumps(item, ensure_ascii=False))
        return " ".join(chunks)
    return str(x)


def _normalize_responses_messages(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Ensure every message/content part matches the OpenAI Responses schema."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role") or m.get("speaker") or "user"
        content = m.get("content")
        parts: List[Dict[str, Any]] = []

        if isinstance(content, str):
            msg_type = "output_text" if role == "assistant" else "input_text"
            parts.append({"type": msg_type, "text": content})
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict):
                    ptype = p.get("type", "text")
                    if ptype in TEXT_PART_KEYS:
                        txt = _to_str(p.get("text", ""))
                        if ptype in ("output_text", "refusal"):
                            parts.append({"type": ptype, "text": txt})
                        else:
                            t = "output_text" if role == "assistant" else "input_text"
                            parts.append({"type": t, "text": txt})
                    elif ptype in ("image_url", "input_image"):
                        img = p.get("image_url")
                        detail = p.get("detail", "auto")
                        if isinstance(img, dict):
                            image_url = img
                        else:
                            image_url = {"url": img}
                        parts.append({"type": "input_image", "image_url": image_url, "detail": detail})
                    elif ptype in ("input_file", "file"):
                        fid = p.get("file_id") or p.get("id")
                        if fid:
                            parts.append({"type": "input_file", "file_id": fid})
                    else:
                        t = "output_text" if role == "assistant" else "input_text"
                        parts.append({"type": t, "text": _to_str(p)})
                else:
                    t = "output_text" if role == "assistant" else "input_text"
                    parts.append({"type": t, "text": _to_str(p)})
        else:
            t = "output_text" if role == "assistant" else "input_text"
            parts.append({"type": t, "text": _to_str(content)})

        for cp in parts:
            if cp.get("type") in {"input_text", "output_text", "refusal"}:
                cp["text"] = _to_str(cp.get("text", ""))

        out.append({"role": role, "content": parts})

    return out


def call_openai_gpt5_multimodal(
    model: str,
    user_text: str,
    image_blobs: Optional[List[bytes]] = None,
    system_text: Optional[str] = None,
    temperature: float = 0.7,
):
    """Send text plus optional images to OpenAI GPT-5 via the Responses API."""
    client = OpenAI()
    user_content = []

    if image_blobs:
        for blob in image_blobs:
            img_bytes, _ = normalize_image_bytes(blob)
            uploaded = client.files.create(
                file=io.BytesIO(img_bytes), purpose="vision"
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"file_id": uploaded.id},
                    "detail": "high",
                }
            )

    user_content.append({"type": "text", "text": user_text})

    messages = []
    if system_text:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_text}]}
        )
    messages.append({"role": "user", "content": user_content})

    messages = _normalize_responses_messages(messages)

    resp = client.responses.create(
        model=model, input=messages, temperature=temperature
    )
    return getattr(resp, "output_text", None) or str(resp)


def call_gemini_25p_multimodal(
    project_id: str,
    location: str,
    user_text: str,
    image_blobs: Optional[List[bytes]] = None,
    system_text: Optional[str] = None,
    temperature: float = 0.7,
):
    """Send text and images to Gemini 2.5 Pro via Vertex AI."""
    from vertexai.generative_models import GenerativeModel, Part, Content
    import vertexai

    vertexai.init(project=project_id, location=location)
    model = GenerativeModel("gemini-2.5-pro")

    parts: List[Part] = []
    if image_blobs:
        for blob in image_blobs:
            img_bytes, mime = normalize_image_bytes(blob)
            parts.append(Part.from_data(mime_type=mime, data=img_bytes))

    parts.append(Part.from_text(user_text))
    user_content = Content(role="user", parts=parts)

    sys = [Content(role="user", parts=[Part.from_text(system_text)])] if system_text else []
    resp = model.generate_content(
        sys + [user_content], generation_config={"temperature": temperature}
    )
    return resp.text


def call_openai_with_images(user_text: str, images: List[ImageAsset], model: str = "gpt-4o") -> Tuple[str, Optional[dict]]:
    from openai import OpenAI

    client = OpenAI()
    ensure_vision_capable("openai", model, bool(images))

    parts = [{"type": "text", "text": user_text}]
    for img in images:
        parts.append({"type": "image_url", "image_url": {"url": img.data_url}, "detail": "low"})

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": parts}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return text, usage.model_dump() if usage else None


def call_anthropic_with_images(user_text: str, images: List[ImageAsset], model: str = "claude-3-5-sonnet") -> Tuple[str, Optional[dict]]:
    import anthropic

    client = anthropic.Anthropic()
    ensure_vision_capable("anthropic", model, bool(images))

    content = [{"type": "text", "text": user_text}]
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.mime,
                "data": base64.b64encode(img.data).decode("utf-8"),
            },
        })

    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
    )
    out = "".join([b.text for b in msg.content if getattr(b, "type", None) == "text"])
    usage = getattr(msg, "usage", None)
    usage_dict = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens} if usage else None
    return out, usage_dict


def call_gemini_with_images(user_text: str, images: List[ImageAsset], model: str = "gemini-1.5-pro") -> Tuple[str, Optional[dict]]:
    import google.generativeai as genai

    genai.configure()
    ensure_vision_capable("google", model, bool(images))

    model_obj = genai.GenerativeModel(model)
    parts = [user_text] + [{"mime_type": img.mime, "data": img.data} for img in images]

    usage_dict = None
    try:
        tc = model_obj.count_tokens(parts)
        usage_dict = {"input_tokens": tc.total_tokens}
    except Exception:
        pass

    resp = model_obj.generate_content(parts)
    text = resp.text or ""
    return text, usage_dict


def call_openrouter_with_images(user_text: str, images: List[ImageAsset], model: str) -> Tuple[str, Optional[dict]]:
    import os

    ensure_vision_capable("openrouter_like_openai_schema", model, bool(images))

    parts = [{"type": "text", "text": user_text}] + [
        {"type": "image_url", "image_url": {"url": img.data_url}, "detail": "low"}
        for img in images
    ]

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
        "HTTP-Referer": os.environ.get("OR_APP_URL", "https://lofn.ai"),
        "X-Title": "LOFN Vision Chat",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
        "temperature": 0.2,
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage")
    return text, usage


def upload_to_cdn(data: bytes, mime: str) -> str:
    raise NotImplementedError


def poe_send_and_receive(text: str, model: str) -> str:
    raise NotImplementedError


def call_poe_with_images(user_text: str, images: List[ImageAsset], model: str) -> Tuple[str, Optional[dict]]:
    if images:
        urls = [f"[image]({upload_to_cdn(img.data, img.mime)})" for img in images]
        user_text = user_text + "\n\n" + "\n".join(urls)

    text = poe_send_and_receive(user_text, model=model)
    return text, None


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def chat_with_model(user_text: str, uploaded_images, provider: str, model_name: str) -> str:
    assets = normalize_images(uploaded_images) if uploaded_images else []
    provider = provider.lower()
    if provider == "openai":
        text, _ = call_openai_with_images(user_text, assets, model=model_name)
    elif provider == "anthropic":
        text, _ = call_anthropic_with_images(user_text, assets, model=model_name)
    elif provider == "google":
        text, _ = call_gemini_with_images(user_text, assets, model=model_name)
    elif provider == "openrouter":
        text, _ = call_openrouter_with_images(user_text, assets, model=model_name)
    elif provider == "poe":
        text, _ = call_poe_with_images(user_text, assets, model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return text

def prepare_image_messages(images: List) -> List[HumanMessage]:
    """Return ``HumanMessage`` objects with inline JPEG data URLs.

    Accepts a list of strings, bytes, or file-like objects. Images are
    compressed and inlined as ``data:`` URLs. Only the first five images are
    processed.
    """

    images = images[:5] if images else []
    messages: List[HumanMessage] = []

    for img in images:
        url = None
        mime = None

        try:
            if isinstance(img, str):
                url = img
                if img.startswith("data:"):
                    header, b64_data = img.split(",", 1)
                    mime = header.split(";")[0].split(":")[1]
                    data = base64.b64decode(b64_data)
                    if mime.startswith("image/"):
                        data, new_mime = compress_image_bytes(data)
                        if new_mime:
                            mime = new_mime
                    encoded = base64.b64encode(data).decode()
                    url = f"data:{mime};base64,{encoded}"
                else:
                    response = requests.get(img, timeout=5)
                    response.raise_for_status()
                    mime = response.headers.get("Content-Type", "image/jpeg")
                    data = response.content
                    if mime.startswith("image/"):
                        data, new_mime = compress_image_bytes(data)
                        if new_mime:
                            mime = new_mime
                    encoded = base64.b64encode(data).decode()
                    url = f"data:{mime};base64,{encoded}"
            else:
                raw = img.read() if hasattr(img, "read") else bytes(img)
                name = getattr(img, "name", "")
                import mimetypes

                mime, _ = mimetypes.guess_type(name)
                if mime and mime.startswith("image/"):
                    data, new_mime = compress_image_bytes(raw)
                    if new_mime:
                        mime = new_mime
                else:
                    data = raw
                    if not mime:
                        mime = "video/mp4" if raw[:4] == b"\x00\x00\x00\x18" else "application/octet-stream"
                encoded = base64.b64encode(data).decode()
                url = f"data:{mime};base64,{encoded}"

        except Exception:
            continue

        if mime and mime.startswith("image/"):
            messages.append(
                HumanMessage(
                    content=[{"type": "image_url", "image_url": {"url": url}}]
                )
            )
        elif mime and mime.startswith("video/"):
            messages.append(
                HumanMessage(
                    content=[
                        {
                            "type": "file",
                            "file": {"mime_type": mime, "b64_json": encoded},
                        }
                    ]
                )
            )

    return messages


def prepare_image_strings(images: List) -> List[str]:
    """Return data URLs for images prepared for model consumption.

    Mirrors :func:`prepare_image_messages` but returns plain strings instead of
    ``HumanMessage`` objects. The input may contain strings, bytes, or uploaded
    file objects.
    """

    urls = []
    for m in prepare_image_messages(images):
        part = m.content[0]
        if part["type"] == "image_url":
            image_data = part.get("image_url", {})
            if isinstance(image_data, dict):
                urls.append(image_data.get("url", ""))
            else:
                urls.append(image_data)
        elif part["type"] == "file":
            file_data = part.get("file", {})
            mime = file_data.get("mime_type", "")
            b64 = file_data.get("b64_json", "")
            if mime and b64:
                urls.append(f"data:{mime};base64,{b64}")
    return urls

# Load prompts
concept_system = read_prompt('/lofn/prompts/concept_system.txt')
prompt_system = read_prompt('/lofn/prompts/prompt_system.txt')
prompt_ending = read_prompt('/lofn/prompts/prompt_ending.txt')
concept_header_part1 = read_prompt('/lofn/prompts/concept_header.txt')
concept_header_part2 = read_prompt('/lofn/prompts/concept_header_pt2.txt')
prompt_header_part1 = read_prompt('/lofn/prompts/prompt_header.txt')
prompt_header_part2 = read_prompt('/lofn/prompts/prompt_header_pt2.txt')
essence_prompt_middle = read_prompt('/lofn/prompts/essence_prompt.txt')
concepts_prompt_middle = read_prompt('/lofn/prompts/concepts_prompt.txt')
artist_and_critique_prompt_middle = read_prompt('/lofn/prompts/artist_and_critique_prompt.txt')
medium_prompt_middle = read_prompt('/lofn/prompts/medium_prompt.txt')
refine_medium_prompt_middle = read_prompt('/lofn/prompts/refine_medium_prompt.txt')
facets_prompt_middle = read_prompt('/lofn/prompts/facets_prompt.txt')
aspects_traits_prompt_middle = read_prompt('/lofn/prompts/aspects_traits_prompts.txt')
midjourney_prompt_middle = read_prompt('/lofn/prompts/imagegen_prompt.txt')
artist_refined_prompt_middle = read_prompt('/lofn/prompts/artist_refined_prompt.txt')
revision_synthesis_prompt_middle = read_prompt('/lofn/prompts/revision_synthesis_prompt.txt')
dalle3_gen_prompt_middle = read_prompt('/lofn/prompts/dalle3_gen_prompt.txt')
dalle3_gen_prompt_nodiv_middle = read_prompt('/lofn/prompts/dalle3_gen_nodiv_prompt.txt')
meta_prompt_generation_prompt = read_prompt('/lofn/prompts/meta_prompt_generation.txt')
pair_selection_prompt = read_prompt('/lofn/prompts/pair_selection_prompt.txt')
panel_generation_prompt = read_prompt('/lofn/prompts/panel_generation_prompt.txt')
personality_generation_prompt = read_prompt('/lofn/prompts/personality_generation_prompt.txt')
personality_chat_template = read_prompt('/lofn/prompts/personality_chat_template.txt')
personality_image2video_template = read_prompt('/lofn/prompts/personality_image2video_template.txt')

# Video prompts
video_concept_header_part1 = read_prompt('/lofn/prompts/video_concept_header.txt')
video_concept_header_part2 = read_prompt('/lofn/prompts/video_concept_header_pt2.txt')
video_essence_prompt_middle = read_prompt('/lofn/prompts/video_essence_prompt.txt')
video_concepts_prompt_middle = read_prompt('/lofn/prompts/video_concepts_prompt.txt')
video_prompt_header_part1 = read_prompt('/lofn/prompts/video_prompt_header.txt')
video_prompt_header_part2 = read_prompt('/lofn/prompts/video_prompt_header_pt2.txt')
video_artist_and_critique_prompt_middle = read_prompt('/lofn/prompts/video_artist_and_critique_prompt.txt')
video_medium_prompt_middle = read_prompt('/lofn/prompts/video_medium_prompt.txt')
video_refine_medium_prompt_middle = read_prompt('/lofn/prompts/video_refine_medium_prompt.txt')
video_facets_prompt_middle = read_prompt('/lofn/prompts/video_facets_prompt.txt')
video_aspects_traits_prompt_middle = read_prompt('/lofn/prompts/video_aspects_traits_prompt.txt')
video_generation_prompt_middle = read_prompt('/lofn/prompts/video_generation_prompt.txt')
video_revision_synthesis_prompt_middle = read_prompt('/lofn/prompts/video_revision_synthesis_prompt.txt')
video_artist_refined_prompt = read_prompt('/lofn/prompts/video_artist_refined_prompt.txt')

# Music prompts
music_essence_prompt = read_prompt('/lofn/prompts/music_essence_prompt.txt')
music_creation_prompt = read_prompt('/lofn/prompts/music_creation_prompt.txt')
music_prompt_header_part1 = read_prompt('/lofn/prompts/music_prompt_header.txt')
music_prompt_header_part2 = read_prompt('/lofn/prompts/music_prompt_header_pt2.txt')
music_concept_header_part1 = read_prompt('/lofn/prompts/music_concept_header.txt')
music_concept_header_part2 = read_prompt('/lofn/prompts/music_concept_header_pt2.txt')


# Read aesthetics from the file
with open('/lofn/prompts/aesthetics.txt', 'r') as file:
    aesthetics = file.read().split(', ')

# Combine prompt parts
prompt_header = prompt_header_part1 + prompt_header_part2
concept_header = concept_header_part1 + concept_header_part2

video_prompt_header = video_prompt_header_part1 + video_prompt_header_part2
video_concept_header = video_concept_header_part1 + video_concept_header_part2
music_prompt_header = music_prompt_header_part1 + music_prompt_header_part2
music_concept_header = music_concept_header_part1 + music_concept_header_part2

# Construct full prompts
essence_prompt = concept_header + essence_prompt_middle + prompt_ending
concepts_prompt = concept_header + concepts_prompt_middle + prompt_ending
artist_and_critique_prompt = concept_header + artist_and_critique_prompt_middle + prompt_ending
medium_prompt = concept_header + medium_prompt_middle + prompt_ending
refine_medium_prompt = concept_header + refine_medium_prompt_middle + prompt_ending
facets_prompt = concept_header + facets_prompt_middle + prompt_ending
aspects_traits_prompt = prompt_header + aspects_traits_prompt_middle + prompt_ending
midjourney_prompt = prompt_header + midjourney_prompt_middle + prompt_ending
artist_refined_prompt = prompt_header + artist_refined_prompt_middle + prompt_ending
revision_synthesis_prompt = concept_header + revision_synthesis_prompt_middle + prompt_ending
dalle3_gen_prompt = dalle3_gen_prompt_middle + prompt_ending
dalle3_gen_nodiv_prompt = dalle3_gen_prompt_nodiv_middle + prompt_ending

# Video prompts
video_prompts = {
    'essence_and_facets': video_concept_header + video_essence_prompt_middle + prompt_ending,
    'concepts': video_concept_header + video_concepts_prompt_middle + prompt_ending,
    'artist_and_critique': video_concept_header + video_artist_and_critique_prompt_middle + prompt_ending,
    'medium': video_concept_header + video_medium_prompt_middle + prompt_ending,
    'refine_medium': video_concept_header + video_refine_medium_prompt_middle + prompt_ending,
    'facets': video_concept_header + video_facets_prompt_middle + prompt_ending,
    'aspects_traits': video_prompt_header + video_aspects_traits_prompt_middle + prompt_ending,
    'generation': video_prompt_header + video_generation_prompt_middle + prompt_ending,
    'revision_synthesis': video_concept_header + video_revision_synthesis_prompt_middle + prompt_ending,
    'artist_refined': video_prompt_header + video_artist_refined_prompt + prompt_ending,  # Add this line
}

# Music prompts
music_prompts = {
    'essence_and_facets': music_concept_header + read_prompt('/lofn/prompts/music_essence_prompt.txt') + prompt_ending,
    'concepts': music_concept_header + read_prompt('/lofn/prompts/music_concepts_prompt.txt') + prompt_ending,
    'artist_and_critique': music_concept_header + read_prompt('/lofn/prompts/music_artist_and_critique_prompt.txt') + prompt_ending,
    'medium': music_concept_header + read_prompt('/lofn/prompts/music_medium_prompt.txt') + prompt_ending,
    'refine_medium': music_concept_header + read_prompt('/lofn/prompts/music_refine_medium_prompt.txt') + prompt_ending,
    'facets': music_concept_header + read_prompt('/lofn/prompts/music_facets_prompt.txt') + prompt_ending,
    'song_guides': music_prompt_header + read_prompt('/lofn/prompts/music_song_guides_prompt.txt') + prompt_ending,
    'generation': music_prompt_header + read_prompt('/lofn/prompts/music_generation_prompt.txt') + prompt_ending,
    'artist_refined': music_prompt_header + read_prompt('/lofn/prompts/music_artist_refined_prompt.txt') + prompt_ending,
    'revision_synthesis': music_prompt_header + read_prompt('/lofn/prompts/music_revision_synthesis_prompt.txt') + prompt_ending,
}

# Image prompts (existing)
image_prompts = {
    'essence_and_facets': essence_prompt,
    'concepts': concepts_prompt,
    'artist_and_critique': artist_and_critique_prompt,
    'medium': medium_prompt,
    'refine_medium': refine_medium_prompt,
    'facets': facets_prompt,
    'aspects_traits': aspects_traits_prompt,
    'generation': midjourney_prompt,
    'revision_synthesis': revision_synthesis_prompt,
    'artist_refined': artist_refined_prompt
}

# Configuration mapping
prompt_configs = {
    'image': image_prompts,
    'video': video_prompts,
    'music': music_prompts
}

meta_prompt_schema = {
    'meta_prompt': str
}

panel_prompt_schema = {
    'panel_prompt': str
}

personality_prompt_schema = {
    'personality_prompt': str
}

essence_and_facets_schema = {
    "essence_and_facets": {
        "creativity_spectrum": {
            "literal": (int, float),
            "inventive": (int, float),
            "transformative": (int, float)
        },
        "essence": str,
        "facets": list,
        "style_axes": dict  # We can further specify inner keys if needed
    }
}

concepts_schema = {
    "concepts": [
        {"concept": str}
    ]
}

mediums_schema = {
    "mediums": [
        {"medium": str}
    ]
}

artist_refined_concepts_schema = {
    "artists": [
        {"artist": str}
    ],
    "refinedconcepts": [
        {"refinedconcept": str}
    ]
}

mediums_schema = mediums_schema = {
    "mediums": [
        {"medium": str}
    ]
}

refined_mediums_schema = {
    "refinedconcepts": [
        {"refinedconcept": str}
    ],
    "refinedmediums": [
        {"refinedmedium": str}
    ]
}

facets_schema = {
    "facets": [str]
}

artistic_guides_schema = {
    "artistic_guides": [
        {"artistic_guide": str}
    ]
}

image_gen_schema = {
    "image_gen_prompts": [
        {"image_gen_prompt": str}
    ]
}

video_gen_schema = {
    "video_prompts": [
        {"video_prompt": str}
    ]
}

artist_refined_schema =  {
    "artist_refined_prompts": [
        {"artist_refined_prompt": str}
    ]
}

revised_synthesized_schema = {
    "revised_prompts": [
        {"revised_prompt": str}
    ],
    "synthesized_prompts": [
        {"synthesized_prompt": str}
    ]
}

music_facets_schema = {
    "essence_and_facets": {
        "creativity_spectrum": {
            "literal": (int, float),
            "inventive": (int, float),
            "transformative": (int, float)
        },
        "essence": str,
        "facets": [str],
        "style_axes": dict
    }
}

music_gen_schema = {
    "music_prompt": str,
    "lyrics_prompt": str,
    "title": str
}


song_guides_schema = {
    "song_guides": [
        {"song_guide": str}
    ]
}

music_generation_schema = {
    "music_prompts": [
        {"music_prompt": str, "lyrics_prompt": str, "title": str}
    ]
}

# Schema for artist-refined music prompts
music_artist_refined_schema = {
    "musician_refined_prompts": [
        {"music_prompt": str, "lyrics_prompt": str, "title": str}
    ]
}

# Schema for final music prompt revisions and synthesis
music_revised_synthesized_schema = {
    "revised_prompts": [
        {"music_prompt": str, "lyrics_prompt": str, "title": str}
    ],
    "synthesized_prompts": [
        {"music_prompt": str, "lyrics_prompt": str, "title": str}
    ]
}

# Schema for panel voting on concept-medium pairs
best_pairs_schema = {
    "best_pairs": [
        {"concept": str, "medium": str}
    ]
}


@st.cache_data(persist=True)
def fetch_openrouter_models():
    api_key = Config.OPEN_ROUTER_API_KEY
    if not api_key:
        print("OpenRouter API key is not set.")
        return []

    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        models_data = response.json()
        return models_data.get('data', [])
    else:
        print(f"Failed to fetch models from OpenRouter API: {response.status_code} {response.text}")
        return []

class OpenRouterLLM(LLM):
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 32000
    api_key: str
    debug: bool = False

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Prepare messages for OpenRouter API
        messages = [{"role": "user", "content": prompt}]

        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if stop is not None:
            data["stop"] = stop

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
        )

        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.text}")

        if self.debug:
            st.write(f"OpenRouter API response: {response.text}")

        result = response.json()

        # Extract assistant's reply
        text = result["choices"][0]["message"]["content"]
        return text

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class ChatOpenAIWebSearch(BaseChatModel):
    model_name: str
    openai_api_key: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 4096
    _client: OpenAI = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str,
        openai_api_key: Optional[str],
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ):
        super().__init__(
            model_name=model_name,
            openai_api_key=openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._client = OpenAI(api_key=openai_api_key)

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        converted = []
        for m in messages:
            if isinstance(m, SystemMessage):
                converted.append(
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": m.content}],
                    }
                )
            elif isinstance(m, HumanMessage):
                converted.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": m.content}],
                    }
                )
            elif isinstance(m, AIMessage):
                converted.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": m.content}],
                    }
                )
            else:
                converted.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": m.content}],
                    }
                )
        return converted

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        input_messages = self._convert_messages(messages)
        input_messages = _normalize_responses_messages(input_messages)
        tool_type_candidates = ["web_search", "web_search_preview"]
        last_err = None
        for tool_type in tool_type_candidates:
            try:
                resp = self._client.responses.create(
                    model=self.model_name,
                    input=input_messages,
                    tools=[{"type": tool_type}],
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                text = resp.output_text
                gen = ChatGeneration(message=AIMessage(content=text))
                return ChatResult(generations=[gen])
            except Exception as e:
                last_err = e
        raise RuntimeError(f"OpenAI search failed: {last_err}")

    @property
    def _llm_type(self) -> str:
        return "openai-web-search"

class GeminiLLM(LLM):
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str
    generative_model: Any = None
    thinking_budget: int = 24576

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generative_model = genai.Client(api_key=self.api_key)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        # Added for the release of the new gen-ai package
        # if self.model_name.startswith('gemini-2.5-flash'):
        #     generation_config = genai.types.GenerateContentConfig(
        #         thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
        #         max_output_tokens=self.max_tokens,
        #         stop_sequences=stop or []
        #     )
        # else:

        # Use thinking budget for Gemini 2.5 models that support it
        use_search = self.model_name.startswith("gemini-2.5-pro")
        tool = types.Tool(google_search=types.GoogleSearch()) if use_search and types is not None else None
        if self.model_name.startswith("gemini-2.5-pro") or self.model_name.startswith("gemini-2.5-flash"):
            generation_config = genai.types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                ),
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                stop_sequences=stop or [],
                tools=[tool] if tool else None,
            )
        else:
            generation_config = genai.types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                stop_sequences=stop or [],
                tools=[tool] if tool else None,
            )

        response = self.generative_model.models.generate_content(
            contents=prompt,
            model=self.model_name,
            config=generation_config,
        )

        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

class PoeLLM(LLM):
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 4096

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("Stop sequences are not supported for Poe models.")

        message = fp.ProtocolMessage(role="user", content=prompt)

        async def get_response():
            response = ""
            async for partial in fp.get_bot_response(
                messages=[message],
                bot_name=self.model_name.split("-", 1)[1],
                api_key=self.api_key,
            ):
                if isinstance(partial, fp.PartialResponse):
                    response += partial.text
            return response

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(get_response())
        finally:
            loop.close()

    @property
    def _llm_type(self) -> str:
        return "poe"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def get_llm(model, temperature, OPENAI_API=None, ANTHROPIC_API=None, debug=False, reasoning_level="medium"):
    """
    Returns a language-model interface for Lofn based on the chosen `model`.
    Also includes logic to handle the new 'o1' models with max_completion_tokens.
    `reasoning_level` can be "low", "medium", or "high" to control how many tokens to let the model use.
    """
    if model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return O1ChatOpenAI(
            model_name=model,
            openai_api_key=OPENAI_API,
            reasoning_level=reasoning_level,
            debug=debug
        )
    # If the model is an OpenRouter model
    if model.startswith("OR-"):
        model_id = model[3:]  # Remove OR-.
        # Fetch models from OpenRouter to get the context length
        or_models = fetch_openrouter_models()
        # Find the model in the list
        model_data = next((m for m in or_models if m['id'] == model_id), None)
        if model_data:
            context_length = min(model_data.get('context_length', 32000) ,42000)
            # Subtract estimated tokens for input prompts and retries
            max_tokens = context_length - 10000
            # Ensure max_tokens is not negative
            max_tokens = min(max(max_tokens, 0), context_length)
            return OpenRouterLLM(
                model_name=model_id,
                api_key=Config.OPEN_ROUTER_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
                debug=debug
            )
        else:
            raise ValueError(f"Model {model_id} not found in OpenRouter models.")
    else:
        # Dictionary mapping models to their maximum token limits
        model_max_tokens = {
            # OpenAI models
            "gpt-5": 32768,
            "gpt-5-mini": 32768,
            "gpt-5-nano": 32768,
            "gpt-4.1": 32768,
            "gpt-4.1-mini": 32768,
            "gpt-4.1-nano": 32768,
            "o3": 100000,
            "o3-pro": 100000,
            "o4-mini": 100000,

            # Anthropic models
            "claude-3-7-sonnet-20250219": 32000,
            "claude-sonnet-4-20250514": 32000,
            "claude-opus-4-20250514": 32000,
            "claude-3-5-sonnet-latest": 8096,
            "claude-3-5-sonnet-20241022": 8096,
            "claude-3-5-haiku-20241022": 8096,
            "claude-3-5-sonnet-20240620": 4096,
            "claude-3-opus-20240229": 4096,
            "claude-3-sonnet-20240229": 4096,
            "claude-3-haiku-20240307": 4096,

            # Google models
            "gemini-2.5-pro": 120000,
            "gemini-2.5-flash": 120000,
            "gemini-2.5-flash-lite": 120000,
            "gemini-2.5-flash-lite-preview": 120000,
            "gemini-2.0-flash": 8191,
            "gemini-2.0-flash-lite": 8191,
            "gemini-2.0-flash-preview": 8191,
            "gemini-1.5-pro": 32768,
            "gemini-1.5-flash": 16384,

            # Poe models
            "Poe-Assistant": 32768,
            "Poe-App-Creator": 32768,
            "Poe-GPT-5": 32768,
            "Poe-GPT-5-mini": 32768,
            "Poe-GPT-5-nano": 32768,
            "Poe-GPT-4o": 128000,
            "Poe-GPT-4.1": 32768,
            "Poe-GPT-4.1-mini": 32768,
            "Poe-GPT-4.1-nano": 32768,
            "Poe-o3": 100000,
            "Poe-o3-pro": 100000,
            "Poe-o4-mini": 100000,
            "Poe-Claude-Opus-4.1": 32000,
            "Poe-Claude-Sonnet-4": 32000,
            "Poe-Gemini-2.5-Pro": 120000,
            "Poe-Gemini-2.5-Flash": 120000,
            "Poe-Gemini-2.5-Flash-Lite": 120000,
            "Poe-Gemini-2.5-Flash-Lite-Preview": 120000,
            "Poe-Gemini-2.0-Flash": 8191,
            "Poe-Gemini-2.0-Flash-Lite": 8191,
            "Poe-Gemini-2.0-Flash-Preview": 8191,
            "Poe-Gemini-1.5-Pro": 32768,
            "Poe-Gemini-1.5-Flash": 16384,
            "Poe-Grok-4": 128000,
            "Poe-Grok-3": 128000,
            "Poe-Grok-3-Mini": 128000,
            "Poe-GPT-OSS-120B-T": 32768,
            "Poe-DeepSeek-V3": 128000,
            "Poe-Deepseek-V3-FW": 128000,
            "Poe-Deepseek-R1": 164000,
            "Poe-Qwen2-72B-Chat": 32768,
            "Poe-Qwen2.5-VL-72B-T": 32000,
            "Poe-Qwen2.5-Coder-32B": 32768
        }

        # Get the maximum token limit for the selected model
        max_tokens = model_max_tokens.get(model, 4096)

        if model.startswith("claude"):
            if model in ["claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]:
                return ChatAnthropic(
                    model=model,
                    max_tokens=max_tokens,
                    thinking={"type": "enabled", "budget_tokens": 15000},
                    anthropic_api_key=Config.ANTHROPIC_API
                )
            else:
                return ChatAnthropic(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    anthropic_api_key=Config.ANTHROPIC_API
                )
        elif model.startswith("gemini"):
            return GeminiLLM(
                model_name=model,
                api_key=Config.GOOGLE_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif model.startswith("Poe-"):
            return PoeLLM(
                model_name=model,
                api_key=Config.POE_API,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif model.startswith("LOCAL-"):
            model_name = model[6:]
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=Config.LOCAL_LLM_API_KEY,
                openai_api_base=Config.LOCAL_LLM_API_BASE,
            )
        elif model.startswith("gpt-5"):
            return ChatOpenAIWebSearch(
                model_name=model,
                openai_api_key=Config.OPENAI_API,
                temperature=1,
                max_tokens=max_tokens
            )
        elif model.startswith("gpt-4.1"):
            return ChatOpenAI(
                model=model,
                temperature=1,
                max_completion_tokens=max_tokens,
                openai_api_key=Config.OPENAI_API,
            )
        elif model.startswith("gpt"):
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=Config.OPENAI_API,
            )
        elif model.startswith("o1"):
            return ChatOpenAI(
                model=model,
                openai_api_key=Config.OPENAI_API,
                temperature=1
            )
        elif model.startswith("o3"):
            return ChatOpenAI(
                model=model,
                openai_api_key=Config.OPENAI_API,
                temperature=1
            )
        elif model.startswith("o4"):
            return ChatOpenAI(
                model=model,
                openai_api_key=Config.OPENAI_API,
                temperature=1
            )
        else:
            raise LofnError(f"{model} not found!")


def _make_serializable(obj: Any) -> Any:
    """Convert possibly complex objects into JSON-serializable structures."""

    if isinstance(obj, HumanMessage):
        return {
            "content": obj.content,
            "additional_kwargs": getattr(obj, "additional_kwargs", {}),
        }
    if isinstance(obj, list):
        return [_make_serializable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    return obj


def run_llm_chain(chains, chain_name, args_dict, max_retries, model=None,
                  debug=None, expected_schema=None):
    chain = chains[chain_name]
    serializable = _make_serializable(args_dict) if args_dict is not None else None
    args_json = json.dumps(serializable, sort_keys=True) if serializable is not None else None
    output = run_chain_with_retries(
        chain, max_retries=max_retries, args_json=args_json, is_correction=False,
        model=model, debug=debug, expected_schema=expected_schema
    )
    if output is None:
        st.error(f"Failed to get valid JSON response after {max_retries} attempts.")
        return None
    return output


def run_llm_chain_raw(chains, chain_name, args_dict, max_retries, model=None,
                      debug=None, expected_schema=None):
    chain = chains[chain_name]
    output = run_chain_with_retries_raw(
        chain, max_retries=max_retries, args_dict=args_dict,
        model=model, debug=debug, expected_schema=expected_schema
    )
    if output is None:
        st.error(f"Failed to get valid JSON response after {max_retries} attempts.")
        return None
    return output

@st.cache_data(persist=True)
def run_chain_with_retries(
    _lang_chain, max_retries, args_json=None, is_correction=False, model=None,
    debug=False, expected_schema=None
):
    args_dict = json.loads(args_json) if args_json else {}
    output = None
    retry_count = 0
    while retry_count < max_retries:
        try:
            output = run_any_chain(
                _lang_chain, args_dict, is_correction, retry_count, model, debug, expected_schema
            )
            args_dict['output'] = output
            if debug:
                st.write(f"Raw output from LLM:\n{output}")

            def schema_validator(obj):
                if expected_schema and not validate_schema(obj, expected_schema):
                    raise ValueError("Parsed JSON does not match the expected schema.")
            try:
                simple_schema = {}
                if isinstance(expected_schema, dict):
                    for key, val in expected_schema.items():
                        if isinstance(val, list):
                            if val and val[0] == str:
                                simple_schema[key] = "list[str]"
                            else:
                                simple_schema[key] = list
                        elif isinstance(val, dict):
                            simple_schema[key] = dict
                        else:
                            simple_schema[key] = val
                parsed_output = select_best_json_candidate(
                    str(output), simple_schema, expected_schema, debug
                )
                if expected_schema and not validate_schema(parsed_output, expected_schema):
                    raise ValueError("Parsed JSON does not match the expected schema. Parsed Output {parsed_output}")
                if debug:
                    st.write("Successfully parsed JSON output")
                return parsed_output
            except Exception as e:
                st.write(f"Failed to parse or validate JSON: {e}")
                is_correction = True  # Use correction prompt in next iteration
                retry_count += 1
        except Exception as e:
            st.write(f"An error occurred in attempt {retry_count + 1}: {str(e)}")
            is_correction = True
            retry_count += 1
    if retry_count >= max_retries:
        st.write("Max retries reached. Exiting.")
    return None


def run_chain_with_retries_raw(
    _lang_chain, max_retries, args_dict=None, model=None, debug=False, expected_schema=None
):
    args_dict = args_dict or {}
    output = None
    retry_count = 0
    is_correction = False
    while retry_count < max_retries:
        try:
            output = run_any_chain(
                _lang_chain, args_dict, is_correction, retry_count, model, debug, expected_schema
            )
            args_dict['output'] = output
            if debug:
                st.write(f"Raw output from LLM:\n{output}")
            def schema_validator(obj):
                if expected_schema and not validate_schema(obj, expected_schema):
                    raise ValueError("Parsed JSON does not match the expected schema.")
            try:
                simple_schema = {}
                if isinstance(expected_schema, dict):
                    for key, val in expected_schema.items():
                        if isinstance(val, list):
                            if val and val[0] == str:
                                simple_schema[key] = "list[str]"
                            else:
                                simple_schema[key] = list
                        elif isinstance(val, dict):
                            simple_schema[key] = dict
                        else:
                            simple_schema[key] = val
                parsed_output = select_best_json_candidate(
                    str(output), simple_schema, expected_schema
                )
                schema_validator(parsed_output)
                if debug:
                    st.write("Successfully parsed JSON output")
                return parsed_output
            except Exception as e:
                st.write(f"Failed to parse or validate JSON: {e}")
                is_correction = True
                retry_count += 1
        except Exception as e:
            st.write(f"An error occurred in attempt {retry_count + 1}: {str(e)}")
            is_correction = True
            retry_count += 1
    if retry_count >= max_retries:
        st.write("Max retries reached. Exiting.")
    return None

def run_any_chain(chain, args_dict, is_correction, retry_count, model, debug=False, expected_schema = None):
    try:
        if is_correction:
            correction_prompt = f"""
            Attempt {retry_count + 1}: The previous response was not in the correct JSON format or did not conform to the expected schema.
            Please refer to the instructions provided earlier and respond with only the complete JSON output.
            Ensure all required fields are present and correctly formatted.
            Escape any special characters within JSON strings.
            Did you forget to label the inside of your arrays? Make sure the key inside the array is also present!
            Most JSON schemas we provide want the return as `"keys" : ["key": "Value 1",  "key": "Value 2", ... ]`. 
            A common error most LLM's make is misisng `refinedconcept` inside the `refinedconcepts` array when revising mediums (this might not be your step!).  
            To assist parsing, make sure the JSON is completely parsable by itself. Remove all newline characters, avoid double-quotes, use \\u0027 as the apostrphe and ensure all escapes satify JSON parsing requirements.
            Expected schema we want from you is in the instructions, and is validated by us through checking: {str(dict(expected_schema)).replace("{", "{{").replace("}","}}")}
            """
            # Preserve original input parameters
            corrected_args = args_dict.copy()
            corrected_args['input'] = (
                correction_prompt
                + "\n\nOriginal prompt:\n"
                + args_dict.get('input', '')
                + "\n\nPrevious response:\n"
                + args_dict.get('output', '')
            )

            if debug:
                st.write(f"Attempt {retry_count + 1}: Using correction prompt")
                st.write(f"Corrected input args: {corrected_args}")

            response = chain.invoke(corrected_args)
        else:
            if debug:
                st.write(f"Attempt {retry_count + 1}: Using original prompt")
                st.write(f"Input args: {args_dict}")
            response = chain.invoke(args_dict)

        if isinstance(response, dict):
            if 'text' in response:
                full_response = response['text']
            elif 'content' in response and isinstance(response['content'], list):
                # Anthropic Claude models may return a list of content blocks
                full_response = "".join(block.get('text', '') for block in response['content'])
            else:
                full_response = str(response)
        elif isinstance(response, str):
            full_response = response
        else:
            full_response = str(response)

        if debug:
            st.write(f"Full response from attempt {retry_count + 1}: {full_response}")

        return full_response

    except Exception as e:
        st.write(f"An error occurred in attempt {retry_count + 1}: {str(e)}")
        return None


# The following functions are for generating image concepts and prompts
def process_essence_and_facets(
    chains, input_text, max_retries, debug=False,
    style_axes=None, creativity_spectrum=None, model=None, image_context=None
):
    expected_schema = essence_and_facets_schema  # Defined earlier
    args = {"input": input_text}
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains, 'essence_and_facets', args, max_retries,
        model, debug, expected_schema=expected_schema
    )
    if parsed_output is None:
        return None, None, None

    if "essence_and_facets" in parsed_output:
        st.session_state.essence_and_facets_output = parsed_output
        if creativity_spectrum is None:
            creativity_spectrum = parsed_output["essence_and_facets"]["creativity_spectrum"]
            st.session_state.creativity_spectrum = creativity_spectrum
        if style_axes is None:
            style_axes = parsed_output["essence_and_facets"]["style_axes"]
            st.session_state.style_axes = style_axes
    else:
        st.error(f"Failed to process essence and facets: Unexpected output structure")
        return None, None, None
    return parsed_output, style_axes, creativity_spectrum

def process_concepts(
    chains, input_text, essence, facets, max_retries, debug=False, style_axes=None,
    creativity_spectrum=None, model=None, image_context=None
):
    expected_schema = concepts_schema
    args = {
        "input": input_text,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_transformative": creativity_spectrum['transformative'],
        "creativity_spectrum_inventive": creativity_spectrum['inventive'],
        "creativity_spectrum_literal": creativity_spectrum['literal'],
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'concepts',
        args,
        max_retries,
        model,
        debug,
        expected_schema=expected_schema
    )
    if parsed_output is None:
        st.error(f"Failed to process concepts")
        return None
    return parsed_output

def process_artist_and_refined_concepts(
    chains,
    input_text,
    essence,
    facets,
    concepts,
    max_retries,
    debug=False,
    style_axes=None,
    creativity_spectrum=None,
    model=None,
    image_context=None
):
    expected_schema = artist_refined_concepts_schema
    args = {
        "input": input_text,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_transformative": creativity_spectrum['transformative'],
        "creativity_spectrum_inventive": creativity_spectrum['inventive'],
        "creativity_spectrum_literal": creativity_spectrum['literal'],
        "concepts": [x['concept'] for x in concepts['concepts']]
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'artist_and_refined_concepts',
        args,
        max_retries,
        model,
        debug,
        expected_schema=expected_schema
    )
    if parsed_output is None:
        st.error(f"Failed to process artist and refined concepts")
        return None
    return parsed_output

def process_mediums(
    chains,
    input_text,
    essence,
    facets,
    refined_concepts,
    max_retries,
    debug=False,
    style_axes=None,
    creativity_spectrum=None,
    model=None,
    image_context=None
):
    expected_schema = mediums_schema
    args = {
        "input": input_text,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "refinedconcepts": [x['refinedconcept'] for x in refined_concepts['refinedconcepts']],
        "creativity_spectrum_transformative": creativity_spectrum['transformative'],
        "creativity_spectrum_inventive": creativity_spectrum['inventive'],
        "creativity_spectrum_literal": creativity_spectrum['literal'],
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'medium',
        args,
        max_retries,
        model,
        debug,
        expected_schema=expected_schema
    )
    if parsed_output is None:
        st.error(f"Failed to process mediums")
        return None
    return parsed_output

def process_refined_mediums(
    chains,
    input_text,
    essence,
    facets,
    mediums,
    artists,
    refined_concepts,
    max_retries,
    debug=False,
    style_axes=None,
    creativity_spectrum=None,
    model=None,
    image_context=None
):
    expected_schema = refined_mediums_schema
    args = {
        "input": input_text,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_transformative": creativity_spectrum['transformative'],
        "creativity_spectrum_inventive": creativity_spectrum['inventive'],
        "creativity_spectrum_literal": creativity_spectrum['literal'],
        "mediums": [x['medium'] for x in mediums['mediums']],
        "artists": artists,
        "refinedconcepts": [x['refinedconcept'] for x in refined_concepts['refinedconcepts']]
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'refine_medium',
        args,
        max_retries,
        model,
        debug,
        expected_schema=expected_schema
    )
    if parsed_output is None:
        st.error(f"Failed to process refined mediums")
        return None
    return parsed_output


def process_facets(
    chains,
    input_text,
    concept,
    medium,
    max_retries,
    debug=False,
    style_axes=None,
    creativity_spectrum=None,
    model=None,
    image_context=None
):
    expected_schema = facets_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": medium,
        "style_axes": style_axes
    }
    if creativity_spectrum is not None:
        args.update({
            "creativity_spectrum_transformative": creativity_spectrum['transformative'],
            "creativity_spectrum_inventive": creativity_spectrum['inventive'],
            "creativity_spectrum_literal": creativity_spectrum['literal'],
        })
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'facets',
        args,
        max_retries,
        model,
        debug,
        expected_schema=expected_schema
    )
    if parsed_output is None:
        st.error(f"Failed to process facets")
        return None
    return parsed_output

def process_artistic_guides(
    chains,
    input_text,
    concept,
    medium,
    facets,
    max_retries,
    debug=False,
    style_axes=None,
    model=None,
    image_context=None
):
    expected_schema = artistic_guides_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'aspects_traits',
        args,
        max_retries,
        model,
        debug,
        expected_schema=expected_schema
    )
    if parsed_output is None:
        st.error(f"Failed to process artistic guides")
        return None
    return parsed_output

def process_midjourney_prompts(
    chains,
    input_text,
    concept,
    medium,
    facets,
    artistic_guides,
    max_retries,
    debug=False,
    style_axes=None,
    model=None,
    image_context=None
):
    expected_schema = image_gen_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "artistic_guides": [x['artistic_guide'] for x in artistic_guides['artistic_guides']]
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'midjourney',
        args,
        max_retries,
        model,
        debug,
        expected_schema=expected_schema
    )
    if parsed_output is None:
        st.error(f"Failed to process midjourney prompts")
        return None
    if parsed_output.get('image_gen_prompts'):
        send_to_discord(
            [prompt['image_gen_prompt'] for prompt in parsed_output['image_gen_prompts']],
            premessage=f'Generated Prompts for {concept} in {medium}:'
        )
    return parsed_output

def process_artist_refined_prompts(
    chains,
    input_text,
    concept,
    medium,
    facets,
    image_gen_prompts,
    max_retries,
    debug=False,
    style_axes=None,
    model=None,
    image_context=None
):
    expected_schema = artist_refined_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "image_gen_prompts": [x['image_gen_prompt'] for x in image_gen_prompts['image_gen_prompts']]
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'artist_refined',
        args,
        max_retries,
        model,
        debug,
        expected_schema=expected_schema
    )
    if parsed_output is None:
        st.error(f"Failed to process artist refined prompts")
        return None
    if parsed_output.get('artist_refined_prompts'):
        send_to_discord(
            [prompt['artist_refined_prompt'] for prompt in parsed_output['artist_refined_prompts']],
            premessage=f'Artist-Refined Prompts for {concept} in {medium}:'
        )
    return parsed_output

def process_revised_synthesized_prompts(
    chains,
    input_text,
    concept,
    medium,
    facets,
    artist_refined_prompts,
    max_retries,
    debug=False,
    style_axes=None,
    model=None,
    image_context=None
):
    expected_schema = revised_synthesized_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "artist_refined_prompts": [x['artist_refined_prompt'] for x in artist_refined_prompts['artist_refined_prompts']]
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'revision_synthesis',
        args,
        max_retries,
        model,
        debug,
        expected_schema=expected_schema
    )
    if parsed_output is None:
        st.error(f"Failed to process revised synthesized prompts")
        return None
    if parsed_output.get('revised_prompts'):
        send_to_discord(
            [prompt['revised_prompt'] for prompt in parsed_output['revised_prompts']],
            premessage=f'Revised Prompts for {concept} in {medium}:'
        )
    if parsed_output.get('synthesized_prompts'):
        send_to_discord(
            [prompt['synthesized_prompt'] for prompt in parsed_output['synthesized_prompts']],
            premessage=f'Synthesized Prompts for {concept} in {medium}:'
        )
    return parsed_output

def generate_concept_mediums(
    input_text,
    max_retries,
    temperature,
    model="gpt-3.5-turbo-16k",
    verbose=False,
    debug=False,
    aesthetics=aesthetics,
    style_axes=None,
    creativity_spectrum=None,
    medium='image',
    reasoning_level="medium",
    input_images: Optional[List[str]] = None
):
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)
        # selected_aesthetics = random.sample(aesthetics, 100)
        # if "Poe" in model:
        #     selected_aesthetics = selected_aesthetics[:24]

        image_context = prepare_image_messages(input_images)

        # Determine max_tokens based on model's capacity. Newer OpenAI
        # models expose this value under `max_completion_tokens` (or
        # occasionally `max_output_tokens`). Fall back to the legacy
        # `max_tokens` attribute if needed.
        max_tokens = (
            llm._identifying_params.get('max_completion_tokens')
            or llm._identifying_params.get('max_output_tokens')
            or llm._identifying_params.get('max_tokens', 4096)
        )

        # Select the appropriate prompts based on the medium
        prompts = prompt_configs.get(medium)

        # Build chains using the selected prompts
        if model[0] == "o":
            chains = {
                'essence_and_facets': (
                    ChatPromptTemplate.from_messages([
                        MessagesPlaceholder("image_context"),
                        ("human", prompts['essence_and_facets'])
                    ])
                    | llm
                ),
                'concepts': (
                    ChatPromptTemplate.from_messages([
                        MessagesPlaceholder("image_context"),
                        ("human", prompts['concepts'])
                    ])
                    | llm
                ),
                'artist_and_refined_concepts': (
                    ChatPromptTemplate.from_messages([
                        MessagesPlaceholder("image_context"),
                        ("human", prompts['artist_and_critique'])
                    ])
                    | llm
                ),
                'medium': (
                    ChatPromptTemplate.from_messages([
                        MessagesPlaceholder("image_context"),
                        ("human", prompts['medium'])
                    ])
                    | llm
                ),
                'refine_medium': (
                    ChatPromptTemplate.from_messages([
                        MessagesPlaceholder("image_context"),
                        ("human", prompts['refine_medium'])
                    ])
                    | llm
                ),
            }
        else:
            chains = {
                'essence_and_facets': (
                    ChatPromptTemplate.from_messages([
                        ("system", concept_system),
                        MessagesPlaceholder("image_context"),
                        ("human", prompts['essence_and_facets'])
                    ])
                    | llm
                ),
                'concepts': (
                    ChatPromptTemplate.from_messages([
                        ("system", concept_system),
                        MessagesPlaceholder("image_context"),
                        ("human", prompts['concepts'])
                    ])
                    | llm
                ),
                'artist_and_refined_concepts': (
                    ChatPromptTemplate.from_messages([
                        ("system", concept_system),
                        MessagesPlaceholder("image_context"),
                        ("human", prompts['artist_and_critique'])
                    ])
                    | llm
                ),
                'medium': (
                    ChatPromptTemplate.from_messages([
                        ("system", concept_system),
                        MessagesPlaceholder("image_context"),
                        ("human", prompts['medium'])
                    ])
                    | llm
                ),
                'refine_medium': (
                    ChatPromptTemplate.from_messages([
                        ("system", concept_system),
                        MessagesPlaceholder("image_context"),
                        ("human", prompts['refine_medium'])
                    ])
                    | llm
                ),
            }

        with st.status("Generating Concepts and Mediums...", expanded=True) as status:
            # Step 1: Essence and Facets
            status.write("Generating Essence and Facets...")
            essence_and_facets, style_axes, creativity_spectrum = process_essence_and_facets(
                chains,
                input_text,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                creativity_spectrum=creativity_spectrum,
                model=model,
                image_context=image_context,
            )
            if essence_and_facets:
                spectrum = (
                    essence_and_facets["essence_and_facets"]["creativity_spectrum"]
                    if st.session_state.creativity_spectrum is None
                    else st.session_state.creativity_spectrum
                )
                display_creativity_and_style_axes(
                    spectrum,
                    essence_and_facets["essence_and_facets"]["style_axes"],
                )
                display_facets(essence_and_facets["essence_and_facets"]["facets"])
            
            # Step 2: Concepts
            status.write("Generating Concepts...")
            concepts = process_concepts(
                chains,
                input_text,
                essence_and_facets["essence_and_facets"]["essence"],
                essence_and_facets["essence_and_facets"]["facets"],
                max_retries,
                debug=debug,
                style_axes=style_axes,
                creativity_spectrum=creativity_spectrum,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Initial Concepts:")
                for i, concept in enumerate(concepts['concepts'], 1):
                    st.write(f"{i}. {concept['concept']}")
            
            # Step 3: Refined Concepts
            status.write("Refining Concepts...")
            artist_and_refined_concepts = process_artist_and_refined_concepts(
                chains,
                input_text,
                essence_and_facets["essence_and_facets"]["essence"],
                essence_and_facets["essence_and_facets"]["facets"],
                concepts,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                creativity_spectrum=creativity_spectrum,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Refined Concepts:")
                for i, concept in enumerate(artist_and_refined_concepts['refinedconcepts'], 1):
                    st.write(f"{i}. {concept['refinedconcept']}")
            
            # Step 4: Generating Mediums
            status.write("Generating Mediums...")
            mediums = process_mediums(
                chains,
                input_text,
                essence_and_facets["essence_and_facets"]["essence"],
                essence_and_facets["essence_and_facets"]["facets"],
                artist_and_refined_concepts,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                creativity_spectrum=creativity_spectrum,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Initial Mediums:")
                for i, medium in enumerate(mediums['mediums'], 1):
                    st.write(f"{i}. {medium['medium']}")
            
            # Step 5: Refining Mediums
            status.write("Refining Mediums...")
            refined_mediums = process_refined_mediums(
                chains,
                input_text,
                essence_and_facets["essence_and_facets"]["essence"],
                essence_and_facets["essence_and_facets"]["facets"],
                mediums,
                [x['artist'] for x in artist_and_refined_concepts['artists']],
                artist_and_refined_concepts,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                creativity_spectrum=creativity_spectrum,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Refined Concepts:")
                for i, concept in enumerate(refined_mediums['refinedconcepts'], 1):
                    st.write(f"{i}. {concept['refinedconcept']}")
                st.write("Refined Mediums:")
                for i, medium in enumerate(refined_mediums['refinedmediums'], 1):
                    st.write(f"{i}. {medium['refinedmedium']}")
    
            status.update(label="Generation Complete!", state="complete")

        refined_concepts = [x['refinedconcept'] for x in refined_mediums['refinedconcepts']]
        refined_mediums_list = [x['refinedmedium'] for x in refined_mediums['refinedmediums']]
        pair_size = min(len(refined_concepts), len(refined_mediums_list))
        refined_concepts = refined_concepts[:pair_size]
        refined_mediums_list = refined_mediums_list[:pair_size]
        concept_mediums = [{'concept': concept, 'medium': medium} for concept, medium in zip(refined_concepts, refined_mediums_list)]
        if debug:
            st.write(f"Pair size: {pair_size}")
            st.write(f"Full list: {concept_mediums}")
        send_to_discord(concept_mediums, content_type='concepts')
        return concept_mediums, style_axes, creativity_spectrum
        
    except Exception as e:
        raise LofnError(f"Error in concept generation: {str(e)}")

def generate_video_concept_mediums(
    input_text,
    max_retries,
    temperature,
    model="gpt-3.5-turbo-16k",
    verbose=False,
    debug=False,
    aesthetics=aesthetics,
    style_axes=None,
    creativity_spectrum=None,
    reasoning_level="medium",
    input_images: Optional[List[str]] = None
):
    return generate_concept_mediums(
        input_text,
        max_retries,
        temperature,
        model,
        verbose,
        debug,
        aesthetics,
        style_axes,
        creativity_spectrum,
        medium='video',
        reasoning_level=reasoning_level,
        input_images=input_images
    )

def generate_music_concept_mediums(
    input_text,
    max_retries,
    temperature,
    model="gpt-3.5-turbo-16k",
    verbose=False,
    debug=False,
    aesthetics=aesthetics,
    style_axes=None,
    creativity_spectrum=None,
    reasoning_level="medium",
    input_images: Optional[List[str]] = None,
):
    return generate_concept_mediums(
        input_text,
        max_retries,
        temperature,
        model,
        verbose,
        debug,
        aesthetics,
        style_axes,
        creativity_spectrum,
        medium='music',
        reasoning_level=reasoning_level,
        input_images=input_images
    )

def generate_simple_music_prompts(
    input_text,
    max_retries,
    temperature,
    model,
    debug=False,
    reasoning_level="medium"
):
    """
    Generates music prompts based on the user's input.

    Parameters:
        input_text (str): The user's idea or description.
        max_retries (int): Maximum number of retries for API calls.
        temperature (float): Sampling temperature.
        model (str): The model name to use.
        debug (bool): If True, prints additional debug information.

    Returns:
        tuple: (music_prompt str, lyrics_prompt str)
    """
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)



        # Determine max_tokens based on model's capacity. Check for the
        # newer `max_completion_tokens`/`max_output_tokens` fields used by
        # recent OpenAI models before falling back to `max_tokens`.
        max_tokens = (
            llm._identifying_params.get('max_completion_tokens')
            or llm._identifying_params.get('max_output_tokens')
            or llm._identifying_params.get('max_tokens', 4096)
        )

        if model[0] == "o":
            chain = (
                ChatPromptTemplate.from_messages([("human", music_essence_prompt)])
                | llm
            )
        else:
            chain = (
                ChatPromptTemplate.from_messages([("system", concept_system), ("human", music_essence_prompt)])
                | llm
            )
    
        output_essence = run_chain_with_retries(
            chain,
            args_json=json.dumps({"input": input_text}, sort_keys=True),
            max_retries=max_retries,
            model=model,
            debug=debug,
            expected_schema = music_facets_schema
        )
        spectrum = (
            output_essence["essence_and_facets"]["creativity_spectrum"]
            if st.session_state.creativity_spectrum is None
            else st.session_state.creativity_spectrum
        )
        display_creativity_and_style_axes(
            spectrum,
            output_essence["essence_and_facets"]["style_axes"],
        )
        display_facets(output_essence["essence_and_facets"]["facets"])

        gen_chain = (
            ChatPromptTemplate.from_messages([("human", music_creation_prompt)])
            | llm
        )

        # Run the chain with retries
        parsed_output = run_chain_with_retries(
            gen_chain,
            args_json=json.dumps({
                "input": input_text,
                "essence":output_essence["essence_and_facets"]["essence"],
                "facets":output_essence["essence_and_facets"]["facets"],
                "style_axes":output_essence["essence_and_facets"]["style_axes"]
            }, sort_keys=True),
            max_retries=max_retries,
            model=model,
            debug=debug,
            expected_schema = music_gen_schema
        )
        if debug:
            print(parsed_output)
        # Parse the output
        if parsed_output is not None:
            music_prompt = parsed_output['music_prompt']
            lyrics_prompt = parsed_output['lyrics_prompt']
            music_title = parsed_output['title']
            return music_prompt, lyrics_prompt, music_title
        else:
            st.error(f"Failed to generate or parse music prompts: {error}")
            return "", ""

    except Exception as e:
        logger.exception("Error generating music prompts: %s", e)
        raise e

@st.cache_data(persist=True)
def generate_meta_prompt(
    input_text,
    max_retries,
    temperature,
    model="gpt-3.5-turbo-16k",
    debug=False,
    reasoning_level="medium",
    medium="image",
    personality_prompt="",
    input_images: Optional[List[str]] = None,
):
    try:
        if medium == "music":
            frames_list = sample_music_frames()
            styles_list = sample_music_genres()
        elif medium == "video":
            frames_list = sample_video_frames()
            styles_list = sample_film_styles()
        else:
            frames_list = sample_artistic_frames()
            styles_list = sample_art_styles()
        image_context = prepare_image_messages(input_images)
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)
        if model[0] == "o":
            chain = (
                ChatPromptTemplate.from_messages([
                    MessagesPlaceholder("image_context"),
                    ("human", meta_prompt_generation_prompt),
                ])
                | llm
            )
        else:
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", concept_system),
                    MessagesPlaceholder("image_context"),
                    ("human", meta_prompt_generation_prompt),
                ])
                | llm
            )

        # Only pass the user's text as the main input. The personality prompt is
        # supplied separately in the prompt template via the
        # ``personality_prompt`` variable, so prepending it here would cause the
        # generated meta-prompt to over-index on personality and ignore the
        # user's request.
        full_input = input_text

        parsed_output = run_llm_chain_raw(
            {"meta": chain},
            "meta",
            {
                "input": full_input,
                "frames_list": frames_list,
                "styles_list": styles_list,
                "personality_prompt": personality_prompt,
                "image_context": image_context,
            },
            max_retries,
            model,
            debug,
            expected_schema=meta_prompt_schema,
        )
        return parsed_output, frames_list, styles_list
    except Exception as e:
        logger.exception("Error generating meta prompt: %s", e)
        raise e

@st.cache_data(persist=True)
def generate_panel_prompt(
    input_text,
    max_retries,
    temperature,
    model="gpt-3.5-turbo-16k",
    debug=False,
    reasoning_level="medium",
    personality_prompt="",
    input_images: Optional[List[str]] = None,
):
    """Generate an artistic panel description via the LLM."""
    try:
        image_context = prepare_image_messages(input_images)
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)
        if model[0] == "o":
            chain = (
                ChatPromptTemplate.from_messages([
                    MessagesPlaceholder("image_context"),
                    ("human", panel_generation_prompt),
                ])
                | llm
            )
        else:
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", concept_system),
                    MessagesPlaceholder("image_context"),
                    ("human", panel_generation_prompt),
                ])
                | llm
            )

        # Similar to meta-prompt generation, keep the user's request separate
        # from any optional personality prompt. The prompt template already has
        # a ``personality_prompt`` field, so including it in the ``input`` would
        # duplicate content and potentially drown out the user's text.
        full_input = input_text

        parsed_output = run_llm_chain_raw(
            {"panel": chain},
            "panel",
            {
                "input": full_input,
                "personality_prompt": personality_prompt,
                "image_context": image_context,
            },
            max_retries,
            model,
            debug,
            expected_schema=panel_prompt_schema,
        )
        if parsed_output is not None:
            return parsed_output.get("panel_prompt", "")
        else:
            st.error("Failed to generate or parse panel prompt")
            return ""
    except Exception as e:
        logger.exception("Error generating panel prompt: %s", e)
        raise e

@st.cache_data(persist=True)
def generate_personality_prompt(
    input_text,
    max_retries,
    temperature,
    model="gpt-3.5-turbo-16k",
    debug=False,
    reasoning_level="medium",
    input_images: Optional[List[str]] = None,
):
    """Generate a short personality description via the LLM."""
    try:
        image_context = prepare_image_messages(input_images)
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)
        if model[0] == "o":
            chain = (
                ChatPromptTemplate.from_messages([
                    MessagesPlaceholder("image_context"),
                    ("human", personality_generation_prompt),
                ])
                | llm
            )
        else:
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", concept_system),
                    MessagesPlaceholder("image_context"),
                    ("human", personality_generation_prompt),
                ])
                | llm
            )

        parsed_output = run_llm_chain_raw(
            {"personality": chain},
            "personality",
            {"input": input_text, "image_context": image_context},
            max_retries,
            model,
            debug,
            expected_schema=personality_prompt_schema,
        )
        if parsed_output is not None:
            return parsed_output.get("personality_prompt", "")
        else:
            st.error("Failed to generate or parse personality prompt")
            return ""
    except Exception as e:
        logger.exception("Error generating personality prompt: %s", e)
        raise e


def run_personality_chat(
    personality_prompt,
    chat_history,
    user_input,
    model="gpt-3.5-turbo-16k",
    temperature=0.7,
    reasoning_level="medium",
    debug=False,
    input_media: Optional[List[Dict[str, str]]] = None,
    system_prompt: str = personality_chat_template,
):
    """Run a free-form chat with a given personality using the COGNITION MATRIX template."""
    user_message = HumanMessage(
        content=[{"type": "text", "text": user_input}, *(input_media or [])]
    )
    if _has_image_parts(chat_history + [user_message]) and not _supports_vision(model):
        raise LofnError(f"{model} does not accept image inputs. Pick a vision model.")
    llm = get_llm(
        model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level
    )
    readme_path = Path('/workspace/lofn/README.md')
    lofn_readme = readme_path.read_text() if readme_path.exists() else ""
    system_text = system_prompt.replace("{personality}", personality_prompt).replace(
        "{lofn_readme}", lofn_readme
    )
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_text),
        MessagesPlaceholder("chat_history"),
    ])
    chain = prompt | llm
    if debug:
        logger.debug(
            "Personality chat request",
            extra={
                "personality_prompt": personality_prompt,
                "chat_history": chat_history,
                "user_input": user_input,
                "input_media": input_media,
            },
        )
    response = chain.invoke({"chat_history": chat_history + [user_message]})
    if debug:
        logger.debug("Personality chat response: %s", response)
    if isinstance(response, dict):
        return response.get("text", str(response))
    if hasattr(response, "content"):
        return response.content
    return str(response)


async def stream_personality_chat(
    personality_prompt,
    chat_history,
    user_input,
    model="gpt-3.5-turbo-16k",
    temperature=0.7,
    reasoning_level="medium",
    debug=False,
    input_media: Optional[List[Dict[str, str]]] = None,
    system_prompt: str = personality_chat_template,
):
    """Stream a free-form chat response with a given personality.

    This function mirrors ``run_personality_chat`` but yields tokens as they
    arrive from the underlying LLM.  If streaming is unavailable for the
    selected model it falls back to returning the full response in one chunk.
    """

    user_message = HumanMessage(
        content=[{"type": "text", "text": user_input}, *(input_media or [])]
    )
    if _has_image_parts(chat_history + [user_message]) and not _supports_vision(model):
        raise LofnError(f"{model} does not accept image inputs. Pick a vision model.")
    # Build the chain using the same prompt setup as the non-streaming version
    llm = get_llm(
        model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level
    )

    readme_path = Path('/workspace/lofn/README.md')
    lofn_readme = readme_path.read_text() if readme_path.exists() else ""
    system_text = system_prompt.replace("{personality}", personality_prompt).replace(
        "{lofn_readme}", lofn_readme
    )
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_text),
        MessagesPlaceholder("chat_history"),
    ])

    chain = prompt | llm
    inputs = {"chat_history": chat_history + [user_message]}

    callback = AsyncIteratorCallbackHandler()

    try:
        task = asyncio.create_task(chain.astream(inputs, callbacks=[callback]))

        async for token in callback.aiter():
            yield token

        await task
    except Exception:
        # If streaming isn't supported, fall back to the blocking call
        fallback = run_personality_chat(
            personality_prompt,
            chat_history,
            user_input,
            model=model,
            temperature=temperature,
            reasoning_level=reasoning_level,
            debug=debug,
            input_media=input_media,
            system_prompt=system_prompt,
        )
        yield fallback


def run_personality_image2video_chat(
    personality_prompt,
    chat_history,
    user_input,
    model="gpt-3.5-turbo-16k",
    temperature=0.7,
    reasoning_level="medium",
    debug=False,
    input_media: Optional[List[Dict[str, str]]] = None,
):
    """Run a chat using the image-to-video personality template."""
    if debug:
        logger.debug(
            "Image-to-video chat request",
            extra={
                "personality_prompt": personality_prompt,
                "chat_history": chat_history,
                "user_input": user_input,
                "input_media": input_media,
            },
        )
    return run_personality_chat(
        personality_prompt,
        chat_history,
        user_input,
        model=model,
        temperature=temperature,
        reasoning_level=reasoning_level,
        debug=debug,
        input_media=input_media,
        system_prompt=personality_image2video_template,
    )


def stream_personality_image2video_chat(
    personality_prompt,
    chat_history,
    user_input,
    model="gpt-3.5-turbo-16k",
    temperature=0.7,
    reasoning_level="medium",
    debug=False,
    input_media: Optional[List[Dict[str, str]]] = None,
):
    """Stream chat responses using the image-to-video personality template."""
    return stream_personality_chat(
        personality_prompt,
        chat_history,
        user_input,
        model=model,
        temperature=temperature,
        reasoning_level=reasoning_level,
        debug=debug,
        input_media=input_media,
        system_prompt=personality_image2video_template,
    )

@st.cache_data(persist=True)
def select_best_pairs(input_text, pairs, num_best_pairs, max_retries, temperature, model="gpt-3.5-turbo-16k", debug=False, reasoning_level="medium"):
    """Use the panel to vote on the best concept-medium pairs."""
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)
        if model[0] == "o":
            chain = (
                ChatPromptTemplate.from_messages([("human", pair_selection_prompt)])
                | llm
            )
        else:
            chain = (
                ChatPromptTemplate.from_messages([("system", concept_system), ("human", pair_selection_prompt)])
                | llm
            )
        pairs_json = json.dumps(pairs)
        args = {
            "input": input_text,
            "pairs": pairs_json
        }
        parsed_output = run_llm_chain({'vote': chain}, 'vote', args, max_retries, model, debug, expected_schema=best_pairs_schema)
        return parsed_output.get('best_pairs', [])
    except Exception as e:
        logger.exception("Error selecting best pairs: %s", e)
        raise e

def generate_image_prompts(input_text, concept, medium, max_retries, temperature, model="gpt-3.5-turbo-16k", debug=False, style_axes=None, creativity_spectrum=None, reasoning_level="medium", input_images: Optional[List[str]] = None):
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)

        image_context = prepare_image_messages(input_images)

        # Build chains using the selected prompts
        if model[0] == "o":
            chains = {
                'facets': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", facets_prompt)])
                    | llm
                ),
                'aspects_traits': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", aspects_traits_prompt)])
                    | llm
                ),
                'midjourney': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", midjourney_prompt)])
                    | llm
                ),
                'artist_refined': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", artist_refined_prompt)])
                    | llm
                ),
                'revision_synthesis': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", revision_synthesis_prompt)])
                    | llm
                )
            }
        else:
            chains = {
                'facets': (
                    ChatPromptTemplate.from_messages([("system", concept_system), MessagesPlaceholder("image_context"), ("human", facets_prompt)])
                    | llm
                ),
                'aspects_traits': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", aspects_traits_prompt)])
                    | llm
                ),
                'midjourney': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", midjourney_prompt)])
                    | llm
                ),
                'artist_refined': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", artist_refined_prompt)])
                    | llm
                ),
                'revision_synthesis': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", revision_synthesis_prompt)])
                    | llm
                )
            }

        with st.status(f"Generating Prompts for {concept} in {medium}...", expanded=True) as status:
            # Step 1: Generate Facets
            status.write("Generating Facets...")
            facets = process_facets(
                chains,
                input_text,
                concept,
                medium,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                creativity_spectrum=creativity_spectrum,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Facets:")
                st.write(facets['facets'])
            
            display_facets(facets['facets'])

            # Step 2: Create Artistic Guides
            status.write("Creating Artistic Guides...")
            artistic_guides = process_artistic_guides(
                chains,
                input_text,
                concept,
                medium,
                facets,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Artistic Guides:")
                for i, guide in enumerate(artistic_guides['artistic_guides'], 1):
                    st.write(f"{i}. {guide['artistic_guide']}")
            # display_temporary_results_no_expander(
            #    "Artistic Guides",
            #    [g['artistic_guide'] for g in artistic_guides['artistic_guides']]
            # )

            # Step 3: Generate Image Prompts
            status.write("Generating Image Prompts...")
            midjourney_prompts = process_midjourney_prompts(
                chains,
                input_text,
                concept,
                medium,
                facets,
                artistic_guides,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Image Generation Prompts:")
                for i, prompt in enumerate(midjourney_prompts['image_gen_prompts'], 1):
                    st.write(f"{i}. {prompt['image_gen_prompt']}")

            # Step 4: Refine Prompts
            status.write("Refining Prompts...")
            artist_refined_prompts = process_artist_refined_prompts(
                chains,
                input_text,
                concept,
                medium,
                facets,
                midjourney_prompts,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Artist Refined Prompts:")
                for i, prompt in enumerate(artist_refined_prompts['artist_refined_prompts'], 1):
                    st.write(f"{i}. {prompt['artist_refined_prompt']}")

            # Step 5: Synthesize Final Prompts
            status.write("Synthesizing Final Prompts...")
            revised_synthesized_prompts = process_revised_synthesized_prompts(
                chains,
                input_text,
                concept,
                medium,
                facets,
                artist_refined_prompts,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )

            status.update(label="Prompt Generation Complete!", state="complete")

        # Prepare the DataFrame of prompts
        df_prompts = pd.DataFrame({
            'Revised Prompts': [prompt['revised_prompt'] for prompt in revised_synthesized_prompts['revised_prompts']],
            'Synthesized Prompts': [prompt['synthesized_prompt'] for prompt in revised_synthesized_prompts['synthesized_prompts']]
        })    

        return df_prompts
    except Exception as e:
        raise LofnError(f"Error in prompt generation: {str(e)}")

def generate_all_prompts(input_text, concept_mediums, max_retries, temperature, model, debug, style_axes=None, creativity_spectrum=None):
    results = []
    total_pairs = len(concept_mediums)
    
    for i, pair in enumerate(concept_mediums):
        st.write(f"Generating prompts for pair {i+1}/{total_pairs}: {pair['concept']} in {pair['medium']}")
        df_prompts = generate_image_prompts(
            input_text,
            pair['concept'],
            pair['medium'],
            max_retries,
            temperature,
            model,
            debug=debug,
            style_axes=style_axes,
            creativity_spectrum=creativity_spectrum,
        )
        results.append(df_prompts)
        st.markdown("---")  # Add a separator between each pair's results
        
    return results


def generate_video_prompts(
    input_text,
    concept,
    medium,
    max_retries,
    temperature,
    model="gpt-3.5-turbo-16k",
    debug=False,
    style_axes=None,
    creativity_spectrum=None,
    reasoning_level="medium",
    input_images: Optional[List[str]] = None
):
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)
        # selected_aesthetics = random.sample(aesthetics, 100)
        # if "Poe" in model:
        #     selected_aesthetics = selected_aesthetics[:24]

        # Use video prompts
        prompts = prompt_configs.get('video')

        image_context = prepare_image_messages(input_images)

        # Build chains using the selected prompts
        if model[0] == "o":
            chains = {
                'facets': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", prompts['facets'])])
                    | llm
                ),
                'aspects_traits': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", prompts['aspects_traits'])])
                    | llm
                ),
                'generation': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", prompts['generation'])])
                    | llm
                ),
                'artist_refined': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", prompts['artist_refined'])])
                    | llm
                ),
                'revision_synthesis': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", prompts['revision_synthesis'])])
                    | llm
                )
            }
        else:
            chains = {
                'facets': (
                    ChatPromptTemplate.from_messages([("system", concept_system), MessagesPlaceholder("image_context"), ("human", prompts['facets'])])
                    | llm
                ),
                'aspects_traits': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", prompts['aspects_traits'])])
                    | llm
                ),
                'generation': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", prompts['generation'])])
                    | llm
                ),
                'artist_refined': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", prompts['artist_refined'])])
                    | llm
                ),
                'revision_synthesis': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", prompts['revision_synthesis'])])
                    | llm
                )
            }

        with st.status(f"Generating Video Prompts for {concept} in {medium}...", expanded=True) as status:
            # Step 1: Generate Facets
            status.write("Generating Facets...")
            facets = process_facets(
                chains,
                input_text,
                concept,
                medium,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                creativity_spectrum=creativity_spectrum,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Facets:")
                st.write(facets['facets'])

            display_facets(facets['facets'])

            # Step 2: Create Artistic Guides
            status.write("Creating Artistic Guides...")
            artistic_guides = process_artistic_guides(
                chains,
                input_text,
                concept,
                medium,
                facets,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Artistic Guides:")
                for i, guide in enumerate(artistic_guides['artistic_guides'], 1):
                    st.write(f"{i}. {guide['artistic_guide']}")

            # Step 3: Generate Video Prompts
            status.write("Generating Video Prompts...")
            video_prompts = process_video_prompts(
                chains,
                input_text,
                concept,
                medium,
                facets,
                artistic_guides,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Video Prompts:")
                for i, prompt in enumerate(video_prompts['video_prompts'], 1):
                    st.write(f"{i}. {prompt['video_prompt']}")

            # Step 4: Refine Prompts
            status.write("Refining Prompts...")
            artist_refined_prompts = process_video_artist_refined_prompts(
                chains,
                input_text,
                concept,
                medium,
                facets,
                video_prompts,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )
            if debug:
                st.write("Filmmaker Refined Prompts:")
                for i, prompt in enumerate(artist_refined_prompts['artist_refined_prompts'], 1):
                    st.write(f"{i}. {prompt['artist_refined_prompt']}")

            # Step 5: Synthesize Final Prompts
            status.write("Synthesizing Final Prompts...")
            revised_synthesized_prompts = process_revised_synthesized_prompts(
                chains,
                input_text,
                concept,
                medium,
                facets,
                artist_refined_prompts,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )

            status.update(label="Video Prompt Generation Complete!", state="complete")

        # Prepare the DataFrame of prompts
        df_prompts = pd.DataFrame({
            'Revised Prompts': [prompt['revised_prompt'] for prompt in revised_synthesized_prompts['revised_prompts']],
            'Synthesized Prompts': [prompt['synthesized_prompt'] for prompt in revised_synthesized_prompts['synthesized_prompts']]
        })

        return df_prompts

    except Exception as e:
        raise LofnError(f"Error in video prompt generation: {str(e)}")

def generate_music_prompts(
    input_text,
    concept,
    arrangement,
    max_retries,
    temperature,
    model="gpt-3.5-turbo-16k",
    debug=False,
    style_axes=None,
    creativity_spectrum=None,
    reasoning_level="medium",
    input_images: Optional[List[str]] = None
):
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)
        prompts = prompt_configs.get('music')

        image_context = prepare_image_messages(input_images)

        if model[0] == "o":
            chains = {
                'facets': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", prompts['facets'])])
                    | llm
                ),
                'song_guides': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", prompts['song_guides'])])
                    | llm
                ),
                'generation': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", prompts['generation'])])
                    | llm
                ),
                'artist_refined': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", prompts['artist_refined'])])
                    | llm
                ),
                'revision_synthesis': (
                    ChatPromptTemplate.from_messages([MessagesPlaceholder("image_context"), ("human", prompts['revision_synthesis'])])
                    | llm
                )
            }
        else:
            chains = {
                'facets': (
                    ChatPromptTemplate.from_messages([("system", concept_system), MessagesPlaceholder("image_context"), ("human", prompts['facets'])])
                    | llm
                ),
                'song_guides': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", prompts['song_guides'])])
                    | llm
                ),
                'generation': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", prompts['generation'])])
                    | llm
                ),
                'artist_refined': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", prompts['artist_refined'])])
                    | llm
                ),
                'revision_synthesis': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), MessagesPlaceholder("image_context"), ("human", prompts['revision_synthesis'])])
                    | llm
                )
            }

        with st.status(
            f"Generating Music Prompts for {concept} in {arrangement}...",
            expanded=True,
        ) as status:
            status.write("Generating Facets...")
            facets = process_facets(
                chains,
                input_text,
                concept,
                arrangement,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                creativity_spectrum=creativity_spectrum,
                model=model,
                image_context=image_context,
            )
            if facets is None:
                raise LofnError("Failed to generate facets")
            display_facets(facets["facets"])

            status.write("Creating Song Guides...")
            guides = process_song_guides(
                chains,
                input_text,
                concept,
                arrangement,
                facets,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )
            if guides is None:
                raise LofnError("Failed to generate song guides")

            status.write("Generating Music Prompts...")
            music_prompts = process_music_generation_prompts(
                chains,
                input_text,
                concept,
                arrangement,
                facets,
                guides,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )
            if music_prompts is None:
                raise LofnError("Failed to generate music prompts")

            status.write("Refining Prompts...")
            artist_refined = process_music_artist_refined_prompts(
                chains,
                input_text,
                concept,
                arrangement,
                facets,
                music_prompts,
                guides,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )
            if artist_refined is None:
                raise LofnError("Failed to refine music prompts")

            status.write("Synthesizing Final Prompts...")
            final_output = process_music_revision_synthesis(
                chains,
                input_text,
                concept,
                arrangement,
                facets,
                artist_refined,
                guides,
                max_retries,
                debug=debug,
                style_axes=style_axes,
                model=model,
                image_context=image_context,
            )
            if final_output is None:
                raise LofnError("Failed to synthesize final music prompts")

            status.update(label="Music Prompt Generation Complete!", state="complete")

        return final_output

    except Exception as e:
        raise LofnError(f"Error in music prompt generation: {str(e)}")

def process_video_prompts(
    chains,
    input_text,
    concept,
    medium,
    facets,
    artistic_guides,
    max_retries,
    debug=False,
    style_axes=None,
    model=None,
    image_context=None
):
    expected_schema = video_gen_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "artistic_guides": [x['artistic_guide'] for x in artistic_guides['artistic_guides']]
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'generation',
        args,
        max_retries,
        model,
        debug,
        expected_schema
    )
    
    if parsed_output is None:
        st.error(f"Failed to process video prompts")
        return None
    if parsed_output.get('video_prompts'):
        send_to_discord(
            [prompt['video_prompt'] for prompt in parsed_output['video_prompts']],
            premessage=f'Generated Video Prompts for {concept} in {medium}:'
        )
    return parsed_output



def process_video_artist_refined_prompts(
    chains,
    input_text,
    concept,
    medium,
    facets,
    video_prompts,
    max_retries,
    debug=False,
    style_axes=None,
    model=None,
    image_context=None
):
    expected_schema = artist_refined_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "video_gen_prompts": [x['video_prompt'] for x in video_prompts['video_prompts']]
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'artist_refined',
        args,
        max_retries,
        model,
        debug,
        expected_schema
    )
    if parsed_output is None:
        st.error(f"Failed to process filmmaker refined prompts")
        return None
    if parsed_output.get('artist_refined_prompts'):
        send_to_discord(
            [prompt['artist_refined_prompt'] for prompt in parsed_output['artist_refined_prompts']],
            premessage=f'Filmmaker-Refined Prompts for {concept} in {medium}:'
        )
    return parsed_output


# === Music Processing Functions ===


def process_song_guides(
    chains,
    input_text,
    concept,
    arrangement,
    facets,
    max_retries,
    debug=False,
    style_axes=None,
    model=None,
    image_context=None
):
    expected_schema = song_guides_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": arrangement,
        "facets": facets['facets'],
        "style_axes": style_axes,
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'song_guides',
        args,
        max_retries,
        model,
        debug,
        expected_schema=expected_schema
    )
    if parsed_output is None:
        st.error("Failed to process song guides")
        return None
    return parsed_output


def process_music_generation_prompts(
    chains,
    input_text,
    concept,
    arrangement,
    facets,
    song_guides,
    max_retries,
    debug=False,
    style_axes=None,
    model=None,
    image_context=None
):
    expected_schema = music_generation_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": arrangement,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "song_guides": [x['song_guide'] for x in song_guides['song_guides']]
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'generation',
        args,
        max_retries,
        model,
        debug,
        expected_schema
    )
    if parsed_output is None:
        st.error("Failed to process music prompts")
        return None
    return parsed_output


def process_music_artist_refined_prompts(
    chains,
    input_text,
    concept,
    arrangement,
    facets,
    music_prompts,
    song_guides,
    max_retries,
    debug=False,
    style_axes=None,
    model=None,
    image_context=None
):
    expected_schema = music_artist_refined_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": arrangement,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "song_prompts": music_prompts,
        "song_guides": song_guides
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'artist_refined',
        args,
        max_retries,
        model,
        debug,
        expected_schema
    )
    if parsed_output is None:
        st.error("Failed to process musician refined prompts")
        return None
    return parsed_output


def process_music_revision_synthesis(
    chains,
    input_text,
    concept,
    arrangement,
    facets,
    artist_refined_prompts,
    song_guides,
    max_retries,
    debug=False,
    style_axes=None,
    model=None,
    image_context=None
):
    expected_schema = music_revised_synthesized_schema
    args = {
        "input": input_text,
        "concept": concept,
        "medium": arrangement,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "artist_refined_prompts": artist_refined_prompts,
        "song_guides": song_guides
    }
    if image_context is not None:
        args["image_context"] = image_context
    parsed_output = run_llm_chain_raw(
        chains,
        'revision_synthesis',
        args,
        max_retries,
        model,
        debug,
        expected_schema
    )
    if parsed_output is None:
        st.error("Failed to process revised music prompts")
        return None
    return parsed_output
