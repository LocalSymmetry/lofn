# lofn_ui.py

import streamlit as st
import os
import requests
from requests.exceptions import RequestException
import time
from datetime import datetime
import json
import ast
from langchain.chains.structured_output.base import create_structured_output_runnable
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic.experimental import ChatAnthropicTools
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema import OutputParserException
import numpy as np
import fal_client
import os
import pandas as pd
from string import Template
import json
import re
import random
import openai
import functools
import plotly.graph_objects as go
import math
import json_repair
from dataclasses import dataclass, field
from typing import List, Optional, Union
import google.generativeai as genai
import fastapi_poe as fp
from modal import Image, Stub, asgi_app
import asyncio
from typing import AsyncIterable, List, Optional
import fastapi_poe as fp
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from PIL import Image
from io import BytesIO
from langchain_core.runnables import RunnablePassthrough

class LofnError(Exception):
    """Custom exception class for Lofn-specific errors."""
    pass

def fetch_and_save_image(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(f"/images/{filename}")
        return f"/images/{filename}"
    except Exception as e:
        st.error(f"Error fetching and saving image: {str(e)}")
        return None

# Read environment variables
OPENAI_API = os.environ.get('OPENAI_API', '')
ANTHROPIC_API = os.environ.get('ANTHROPIC_API', '')
GOOGLE_API = os.environ.get('GOOGLE_API', '')
POE_API = os.environ.get('POE_API', '')
webhook_url = os.environ.get('WEBHOOK_URL', '')

# Ensure the OpenAI API key is set
openai.api_key = os.environ.get('OPENAI_API', '')

class GeminiLLM(LLM):
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str
    generative_model: Any = None  # Changed from 'model' to 'generative_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=self.api_key)
        self.generative_model = genai.GenerativeModel(self.model_name)  # Use 'generative_model' instead of 'model'

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            stop_sequences=stop or []
        )

        response = self.generative_model.generate_content(  # Use 'generative_model' instead of 'model'
            prompt,
            generation_config=generation_config
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
            async for partial in fp.get_bot_response(messages=[message], bot_name=self.model_name.split("-", 1)[1], api_key=self.api_key):
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

def extract_video_url_from_response(response, debug):
    if debug:
        st.write("Attempting to extract video URL from response")
        st.write("Response content:")
        st.code(response)

    # Use a regular expression to find URLs in the response, including query parameters
    url_pattern = r'(https?://\S+\.(?:mp4|mov|avi|wmv)(?:\?\S*)?)'
    urls = re.findall(url_pattern, response)
    
    if debug:
        if urls:
            st.write(f"Found {len(urls)} potential video URL(s):")
            for url in urls:
                st.write(url)
        else:
            st.write("No video URLs found in the response")

    return urls[0] if urls else None

def save_video_locally(video_url, filename, directory='videos'):
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(video_url, timeout=60)  # Increased timeout for video download
            response.raise_for_status()
            
            # Ensure the directory exists
            os.makedirs(f'/{directory}', exist_ok=True)
            with open(f'/{directory}/{filename}', 'wb') as f:
                f.write(response.content)
            st.write(f"Video saved as /{directory}/{filename}")
            return
        except requests.exceptions.RequestException as e:
            st.write(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                st.write(f"Failed to download video after {max_retries} attempts")
                st.write(f"Full URL attempted: {video_url}")  # Log the full URL for debugging

def generate_poe_video(prompt: str, image_url: str, params: dict, debug: bool = False):
    try:
        if debug:
            st.write("Starting Poe video generation process")
            st.write(f"Prompt: {prompt}")
            st.write(f"Image URL: {image_url}")
            st.write("Parameters:")
            st.json(params)

        poe_model = "Pika"
        
        # Construct the Pika options
        pika_options = {
            "frameRate": params['frame_rate'],
            "aspectRatio": params['aspect_ratio'],
            "camera": {
                "zoom": params['camera_zoom'],
                "pan": params['camera_pan'],
                "tilt": params['camera_tilt'],
                "rotate": params['camera_rotate']
            },
            "parameters": {
                "motion": params['motion'],
                "guidanceScale": params['guidance_scale'],
                "negativePrompt": params['negative_prompt']
            }
        }
        
        # Construct the message
        message_content = f"""Generate a video based on this image: {image_url}
Prompt: {prompt}
Pika Options: {json.dumps(pika_options)}"""
        
        if debug:
            st.write("Full message content sent to Poe-Pika:")
            st.code(message_content)

        messages = [fp.ProtocolMessage(role="user", content=message_content)]
        
        async def get_responses():
            if debug:
                st.write("Sending request to Poe-Pika and awaiting response...")
            async for partial in fp.get_bot_response(messages=messages, bot_name=poe_model, api_key=POE_API):
                if isinstance(partial, fp.PartialResponse):
                    if debug:
                        st.write("Received partial response:")
                        st.write(partial.text)
                    yield partial.text

        response = asyncio.run(collect_responses(get_responses()))
        
        #if debug:
        st.write("Full response from Poe-Pika:")
        st.code(response)

        video_url = extract_video_url_from_response(response, debug)
        
        if video_url:
            # if debug:
            st.write(f"Extracted video URL: {video_url}")
            return video_url
        else:
            st.write("No video URL found in the Poe-Pika response.")
            if debug:
                st.write("Failed to extract video URL from the response.")
            return None
    except Exception as e:
        st.write(f"An error occurred while generating the video with Poe-Pika: {str(e)}")
        if debug:
            st.write("Exception details:")
            st.exception(e)
        return None

def render_image_controls(model: str):
    
    if model == "DALL-E 3" or model == "Poe-DALL-E-3":
        st.selectbox("Image Size", ["1024x1792", "1792x1024", "1024x1024",], key=f"{model}_image_size")
        st.selectbox("Quality", ["hd", "standard"], key=f"{model}_quality")
        st.selectbox("Style", ["vivid", "natural"], key=f"{model}_style")
    elif model in ["fal-ai/flux/schnell", "Poe-FLUX-schnell"]:
        st.selectbox("Image Size", ["portrait_16_9", "square_hd", "square", "portrait_4_3", "landscape_4_3", "landscape_16_9"], key=f"{model}_image_size")
        st.number_input("Inference Steps", min_value=1, max_value=12, value=12, key=f"{model}_inference_steps")
        st.checkbox("Enable Safety Checker", value=True, key=f"{model}_enable_safety_checker")
    elif model in ["fal-ai/flux/dev", "fal-ai/flux-pro", "Poe-FLUX-pro", "Poe-StableDiffusion3", "Poe-SD3-Turbo", "fal-ai/stable-diffusion-v3", "Poe-FLUX-dev"]:
        st.selectbox("Image Size", ["portrait_16_9",  "square_hd", "square", "portrait_4_3", "landscape_4_3", "landscape_16_9"], key=f"{model}_image_size")
        st.number_input("Inference Steps", min_value=1, max_value=50, value=50, key=f"{model}_inference_steps")
        st.number_input("Guidance Scale", min_value=0.0, max_value=20.0, value=3.5, step=0.1, key=f"{model}_guidance_scale")
        st.checkbox("Enable Safety Checker", value=True, key=f"{model}_enable_safety_checker")
    elif model in [ "fal-ai/fast-sdxl", "fal-ai/playground-v25", "Poe-StableDiffusionXL", "Poe-StableDiffusion3-2B", "Poe-SD3-Medium"]:
        st.selectbox("Image Size", ["1024x1024", "512x512", "768x768", "512x768", "768x512"], key=f"{model}_image_size")
        st.text_area("Negative Prompt", key=f"{model}_negative_prompt")
    elif model in ["fal-ai/hyper-sdxl", "fal-ai/fast-sdxl", "fal-ai/playground-v25"]:
        st.selectbox("Image Size", ["1024x1024", "512x512", "768x768", "512x768", "768x512"], key=f"{model}_image_size")
        st.checkbox("Expand Prompt", value=False, key=f"{model}_expand_prompt")
        st.selectbox("Format", ["jpeg", "png"], key=f"{model}_format")
    elif model == "fal-ai/playground-v25":
        st.selectbox("Image Size", ["1024x1024", "512x512", "768x768", "512x768", "768x512"], key=f"{model}_image_size")
        st.number_input("Guidance Rescale", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="playground_guidance_rescale")
    elif model.startswith("Poe-"):
        # For other Poe models we don't have specific information about
        st.selectbox("Image Size", ["1024x1024", "512x512"], key=f"{model}_image_size")
    else:
        st.selectbox("Image Size", ["1024x1024", "512x512", "768x768", "512x768", "768x512"], key=f"{model}_image_size")
    
    st.number_input("Number of Images", min_value=1, max_value=10, value=1, key=f"{model}_num_images")


def get_model_params(model: str):
    base_params = {
        "num_images": st.session_state[f"{model}_num_images"]
    }

    image_size = st.session_state[f"{model}_image_size"]
    if model == "DALL-E 3" or ('flux' in model) or ('Flux' in model) or ('FLUX' in model):
        base_params["size"] = image_size
    else:
        # Convert size string to width and height
        width, height = map(int, image_size.split('x'))
        base_params["width"] = width
        base_params["height"] = height

    model_specific_params = {
        "DALL-E 3": {
            "model": "dall-e-3",
            "quality": st.session_state.get("dalle_quality", "hd"),
            "style": st.session_state.get("dalle_style", "vivid")
        },
        "fal-ai/flux/schnell": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 12),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "fal-ai/flux/dev": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 28),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "fal-ai/flux-pro": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 50),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "fal-ai/hyper-sdxl": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 28),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True),
            "expand_prompt": st.session_state.get(f"{model}_expand_prompt", False),
            "format": st.session_state.get(f"{model}_format", "jpeg")
        },
        "fal-ai/aura-flow": {
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 50),
            "expand_prompt": st.session_state.get(f"{model}_expand_prompt", True)
        },
        "fal-ai/stable-diffusion-v3": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 28),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 5.0),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True),
            "negative_prompt": st.session_state.get(f"{model}_negative_prompt", ""),
            "prompt_expansion": False
        },
        "fal-ai/fast-sdxl": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 28),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 7.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True),
            "negative_prompt": st.session_state.get(f"{model}_negative_prompt", ""),
            "expand_prompt": st.session_state.get(f"{model}_expand_prompt", False),
            "format": st.session_state.get(f"{model}_format", "jpeg")
        },
        "fal-ai/playground-v25": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 28),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.0),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True),
            "negative_prompt": st.session_state.get(f"{model}_negative_prompt", ""),
            "expand_prompt": st.session_state.get(f"{model}_expand_prompt", False),
            "format": st.session_state.get(f"{model}_format", "jpeg"),
            "guidance_rescale": st.session_state.get("playground_guidance_rescale", 0.0)
        }
    }


    params = base_params.copy()
    params.update(model_specific_params.get(model, {}))
    return params

def generate_fal_image(model_name, params, debug=False):
    try:
        if debug:
            st.write(f"Generating image with FAL model: {model_name}")
            st.write(f"Input parameters: {json.dumps(params, indent=2)}")

        # Base arguments common to all or most models
        arguments = {
            "prompt": params['prompt'],
            "num_images": params['num_images'],
        }

        if debug:
            st.write("Base arguments:")
            st.write(json.dumps(arguments, indent=2))

        # Add model-specific parameters
        if "flux" in model_name or "sdxl" in model_name:
            arguments.update({
                "num_inference_steps": params.get('num_inference_steps', 28),
                "guidance_scale": params.get('guidance_scale', 3.5),
                "enable_safety_checker": params.get('enable_safety_checker', True),
            })

        if "flux" in model_name or "Stable" in model_name or "SD3" in model_name or "stable in model_name":
            arguments.update({"image_size": params.get('image_size', 'square_hd')})
        else:
            arguments.update({
                "image_height": params.get('image_height', '1024'),
                "image_width": params.get('image_width', '1024')
            })
        
        if "stable-diffusion-v3" in model_name or "fast-sdxl" in model_name or "playground-v25" in model_name:
            arguments["negative_prompt"] = params.get('negative_prompt', "")

        if "hyper-sdxl" in model_name or "fast-sdxl" in model_name or "playground-v25" in model_name:
            arguments.update({
                "expand_prompt": params.get('expand_prompt', False),
                "format": params.get('format', 'jpeg'),
            })

        if "playground-v25" in model_name:
            arguments["guidance_rescale"] = params.get('guidance_rescale', 0.0)

        if "aura-flow" in model_name:
            arguments["expand_prompt"] = params.get('expand_prompt', True)

        if debug:
            st.write("Final arguments for FAL:")
            st.write(json.dumps(arguments, indent=2))

        # Submit the job to fal.ai
        if debug:
            st.write("Submitting job to FAL...")

        handler = fal_client.submit(model_name, arguments=arguments)
        
        if debug:
            st.write("Job submitted, waiting for result...")

        result = handler.get()

        if debug:
            st.write("Received result from FAL:")
            st.write(json.dumps(result, indent=2))

        # Check if the result contains images
        if 'images' in result and len(result['images']) > 0:
            image_urls = [image['url'] for image in result['images']]
            if debug:
                st.write(f"Generated {len(image_urls)} image(s):")
                for url in image_urls:
                    st.write(url)
            return image_urls
        else:
            st.write("No images were generated.")
            return None

    except Exception as e:
        st.write(f"An error occurred while generating the image with FAL: {str(e)}")
        if debug:
            st.write("Exception details:")
            st.exception(e)
        return None

def create_style_axes_chart(style_axes):
    categories = list(style_axes.keys())
    values = list(style_axes.values())
    
    fig = go.Figure(data=go.Scatterpolar(
      r=values,
      theta=categories,
      fill='toself'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 100]
        )),
      showlegend=False
    )
    
    return fig

def update_image_controls():
    # Clear previous image controls
    for key in list(st.session_state.keys()):
        if key.endswith(("_prompt", "_image_size", "_num_images", "_quality", "_style",
                         "_inference_steps", "_guidance_scale", "_enable_safety_checker",
                         "_negative_prompt", "_expand_prompt", "_format", "_guidance_rescale")):
            del st.session_state[key]

def generate_dalle_images(input, concept, medium, df_prompts, max_retries, temperature, model, debug, image_model):
    st.write(f"Generating images using {image_model}...")
    
    all_prompts = pd.concat([df_prompts['Revised Prompts'], df_prompts['Synthesized Prompts']])

    for index, prompt in enumerate(all_prompts):
        if debug:
            st.write(f"Generating image for prompt {index + 1}: {prompt}")
        
        params = get_model_params(image_model)
        params['prompt'] = prompt  # Override the prompt with the current one
        
        try:
            results = generate_image(image_model, params)
            
            if results:
                for i, result in enumerate(results):
                    try:
                        # Display the image
                        st.image(result, caption=f"Generated image {i+1} for {concept} in {medium} - Prompt: {prompt}")
                        
                        # Generate a title for the image
                        try:
                            title_data_json = generate_image_title(input, concept, medium, result, max_retries, temperature, model, debug)
                            title_data = json.loads(title_data_json)
                            st.write(f"Generated title: {title_data['title']}")
                            
                            # Display Instagram post information
                            st.subheader("Instagram Post")
                            st.write(f"Caption: {title_data['instagram_post']['caption']}")
                            st.write("Hashtags:")
                            st.write(", ".join(title_data['instagram_post']['hashtags']))
                            
                            st.subheader("SEO Keywords")
                            st.write(", ".join(title_data['seo_keywords']))
                        except Exception as e:
                            st.error(f"Error generating title and Instagram post: {str(e)}")
                            title_data = {"title": "Untitled", "instagram_post": {"caption": "", "hashtags": []}, "seo_keywords": []}
                    
                        # Save the image locally
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            prompt_type = "Revised" if index < len(df_prompts) else "Synthesized"
                            filename = f"{timestamp}_{model}_{concept[0:10]}_{medium[0:10]}_{prompt_type}_{index + 1}_{i + 1}.png"
                            save_image_locally(result, filename)
                        except Exception as e:
                            st.error(f"Error saving image locally: {str(e)}")

                        # Generate video using Pika through Poe if enabled
                        video_url = None
                        if enable_pika_video:
                            try:
                                pika_params = {
                                    'motion': pika_motion,
                                    'guidance_scale': pika_guidance_scale,
                                    'frame_rate': pika_frame_rate,
                                    'aspect_ratio': pika_aspect_ratio,
                                    'negative_prompt': pika_negative_prompt,
                                    'camera_zoom': pika_camera_zoom,
                                    'camera_pan': pika_camera_pan,
                                    'camera_tilt': pika_camera_tilt,
                                    'camera_rotate': pika_camera_rotate
                                }
                                video_url = generate_poe_video(prompt, result, pika_params, debug)
                                if video_url:
                                    video_filename = f"{timestamp}_{model}_{concept[0:10]}_{medium[0:10]}_{prompt_type}_{index + 1}_{i + 1}.mp4"
                                    save_video_locally(video_url, video_filename)
                            except Exception as e:
                                st.error(f"Error generating or saving video: {str(e)}")
                                video_url = None

                        # Save metadata
                        try:
                            metadata = {
                                "timestamp": timestamp,
                                "style_axes": st.session_state.style_axes,
                                "creativity_spectrum": st.session_state.creativity_spectrum,
                                "concept": concept,
                                "medium": medium,
                                "prompt_type": prompt_type,
                                "prompt_index": index + 1,
                                "image_index": i + 1,
                                "prompt": prompt,
                                "title": title_data['title'],
                                "instagram_post": title_data['instagram_post'],
                                "seo_keywords": title_data['seo_keywords'],
                                "image_url": result,
                                "video_url": video_url,
                                "image_model": image_model,
                                "model": model,
                                "image_filename": filename,
                                "video_filename": video_filename if video_url else None
                            }
                            if enable_pika_video:
                                metadata["pika_params"] = pika_params
                            save_metadata(metadata)
                        except Exception as e:
                            st.error(f"Error saving metadata: {str(e)}")
                    except Exception as e:
                        st.error(f"Error processing generated image {i+1}: {str(e)}")
            else:
                st.write(f"Failed to generate image for prompt {index + 1}.")
        except Exception as e:
            st.error(f"Error generating image for prompt {index + 1}: {str(e)}")

    st.write(f"{image_model} image generation and video generation complete.")

def generate_image(model: str, params: dict):
    if model == "DALL-E 3":
        return [generate_image_dalle3(params)]
    elif model.startswith("fal-ai/"):
        return generate_fal_image(model, params)
    elif model.startswith("Poe-"):
        return generate_poe_image(model, params, debug)
    else:
        st.write(f"Unsupported model: {model}")
        return None

def save_metadata(metadata):
    # Ensure the metadata directory exists
    os.makedirs('/metadata', exist_ok=True)
    
    # Create a filename for the metadata
    metadata_filename = f"/metadata/{metadata['timestamp']}_{metadata['model'][0:10]}_{metadata['concept'][0:10]}_{metadata['medium'][0:10]}_{metadata['prompt_type']}_{metadata['prompt_index']}.json"
    
    # Ensure all data is JSON serializable
    def json_serializable(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    # Save the metadata as a JSON file
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2, default=json_serializable)
    
    st.write(f"Metadata saved as {metadata_filename}")

def generate_poe_image(model: str, params: dict, debug: bool = False):
    try:
        poe_model = model.split("-", 1)[1]  # Remove "Poe-" prefix
        
        if debug:
            st.write(f"Generating image with Poe model: {poe_model}")
            st.write(f"Parameters: {json.dumps(params, indent=2)}")
        
        # Construct the prompt with settings
        prompt = params['prompt']
        size = params.get('size', '1792x1024')
        suffix = ""
        
        # Add model-specific options
        if poe_model == "DALL-E-3":
            if size == "1792x1024":
                suffix += '--aspect 7:4'
            elif size == "1024x1024":
                suffix += '--aspect 1:1'
            else: 
                suffix += '--aspect 4:7'
        elif poe_model in ["FLUX-schnell", "FLUX-pro", "FLUX-dev", "StableDiffusion3", "SD3-Turbo", "StableDiffusionXL", "StableDiffusion3-2B", "SD3-Medium"]:
            if size in ["square_hd", "square"]:
                suffix += '--aspect 1:1'
            elif size == "portrait_4_3":
                suffix += '--aspect 3:4'
            elif size == "portrait_16_9":
                suffix += '--aspect 9:16'
            elif size == "landscape_4_3":
                suffix += '--aspect 4:3'
            else: 
                suffix += '--aspect 16:9'
        elif poe_model == "Playground-v2.5":
            options['guidance_scale'] = params.get('guidance_scale', 7.5)
            options['negative_prompt'] = params.get('negative_prompt', '')
            suffix += '--aspect 16:9'
        elif poe_model == "Ideogram":
            options['style_preset'] = params.get('style_preset', 'default')
            suffix += '--aspect 16:9'
        elif poe_model == "LivePortrait":
            options['video_length'] = params.get('video_length', 3)
            suffix += '--aspect 16:9'
        elif poe_model == "RealVisXL":
            options['guidance_scale'] = params.get('guidance_scale', 7.5)
            options['negative_prompt'] = params.get('negative_prompt', '')
            suffix += '--aspect 16:9'
        full_prompt = f"""{prompt} {suffix}"""

        if debug:
            st.write("Full prompt sent to Poe:")
            st.code(full_prompt)

        messages = [fp.ProtocolMessage(role="user", content=full_prompt)]
        
        async def get_responses():
            async for partial in fp.get_bot_response(messages=messages, bot_name=poe_model, api_key=POE_API):
                if isinstance(partial, fp.PartialResponse):
                    yield partial.text

        response = asyncio.run(collect_responses(get_responses()))
        
        if debug:
            st.write("Raw response from Poe:")
            st.code(response)

        image_url = extract_image_url_from_response(response, debug)
        
        if image_url:
            #if debug:
            st.write(f"Extracted image URL: {image_url}")
            return [image_url]  # Return a list of URLs to maintain consistency with other image generation functions
        else:
            st.write("No image URL found in the Poe response.")
            return None
    except Exception as e:
        st.write(f"An error occurred while generating the image with Poe: {str(e)}")
        if debug:
            st.write("Exception details:")
            st.exception(e)
        return None

async def collect_responses(response_generator):
    full_response = ""
    async for partial_response in response_generator:
        full_response += partial_response
    return full_response

def extract_image_url_from_response(response, debug):
    if debug:
        st.write("Attempting to extract image URL from response")
        st.write("Response content:")
        st.code(response)

    # Use a regular expression to find URLs in the response, including all query parameters
    url_pattern = r'(https?://\S+\.(?:jpg|jpeg|png|gif)(?:\?\S*)?)'
    urls = re.findall(url_pattern, response)
    
    if debug:
        if urls:
            st.write(f"Found {len(urls)} potential image URL(s):")
            for url in urls:
                st.write(url)
        else:
            st.write("No image URLs found in the response")

    return urls[0] if urls else None

def generate_image_dalle3(params):
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=params['prompt'],
            size=params['size'],
            quality=params['quality'],
            style=params['style'],
            n=params['num_images']
        )
        return response.data[0].url
    except Exception as e:
        st.write(f"An error occurred while generating the image with DALL-E 3: {str(e)}")
        return None

def save_image_locally(image_url, filename, directory='images'):
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            # No headers needed for the signed URL
            response = requests.get(image_url, timeout=3)  # Add a timeout
            response.raise_for_status()  # This will raise an exception for HTTP errors

            # Ensure the directory exists
            os.makedirs(f'/{directory}', exist_ok=True)
            with open(f'/{directory}/{filename}', 'wb') as f:
                f.write(response.content)
            st.write(f"Image saved as /{directory}/{filename}")
            return
        except requests.exceptions.RequestException as e:
            st.write(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                st.write(f"Failed to download image after {max_retries} attempts")
                st.write(f"Full URL attempted: {image_url}")  # Log the full URL for debugging

def refresh_image_url(concept, medium, prompt):
    # This function should regenerate the image using the Poe API
    # and return a new URL with fresh access tokens
    new_image_url = generate_poe_image("Poe-DALL-E-3", {
        "prompt": prompt,
        "size": '1024x1792',
        "num_images": 1
    })
    return new_image_url[0] if new_image_url else None

def get_image_with_retry(image_url, concept, medium, prompt):
    try:
        response = requests.head(image_url, timeout=10)
        response.raise_for_status()
        return image_url
    except requests.exceptions.RequestException:
        st.write("Image URL expired or inaccessible. Refreshing...")
        return refresh_image_url(concept, medium, prompt)

@st.cache_data(persist=True)
def generate_image_title(input, concept, medium, image, max_retries, temperature, model, debug=False):
    llm = get_llm(model, temperature, OPENAI_API, ANTHROPIC_API)

    chain = (
        ChatPromptTemplate.from_messages([("human", image_title_prompt)])
        | llm
    )
    
    output = run_chain_with_retries(chain, args_dict={
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": st.session_state.essence_and_facets_output['essence_and_facets']['facets'],
        "image": image
    }, max_retries=max_retries, debug=debug)

    if debug:
        st.write("Output from run_chain_with_retries:")
        st.write(output)

    if output is None:
        return json.dumps({"title": "Untitled", "instagram_post": {"caption": "", "hashtags": []}, "seo_keywords": []})

    try:
        # If output is already a dict, convert it to JSON string
        if isinstance(output, dict):
            return json.dumps(output)
        # If output is a string, try to parse it as JSON and then convert back to string
        parsed_output = json.loads(output)
        return json.dumps(parsed_output)
    except json.JSONDecodeError:
        st.error("Failed to parse JSON output from title generation")
        return json.dumps({"title": "Untitled", "instagram_post": {"caption": "", "hashtags": []}, "seo_keywords": []})

def read_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()

def extract_json_from_text(output):
    # List of potential JSON patterns
    patterns = [
        r"'text':\s*'(json\n)?(.*?)'?\s*}$",  # Original pattern
        r'```json\n(.*?)```',  # Code block format
        r'json\s*(\{.*\})',  # JSON prefixed with "json"
        r'json\n\s*(\{.*\})',  # JSON prefixed with "json\n"       
        r'\{.*\}'  # Any JSON-like structure
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, output, re.DOTALL)
        if matches:
            # If it's a tuple (from capturing groups), join all parts
            if isinstance(matches[0], tuple):
                return ''.join(matches[0])
            return matches[0]
    
    return None

def repair_json(json_string):
    try:
        return json_repair.repair_json(json_string)
    except Exception as e:
        st.write(f"Failed to repair JSON: {e}")
        return json_string

def extract_json_from_text(output):
    # Look for JSON within the 'text' field
    text_pattern = r"'text':\s*'(.*?)\s*(?:(?<!\\)'\s*}|$)"
    text_match = re.search(text_pattern, output, re.DOTALL)
    if text_match:
        text_content = text_match.group(1)
        # Now look for JSON within the extracted text content
        json_pattern = r'({.*})'
        json_match = re.search(json_pattern, text_content, re.DOTALL)
        if json_match:
            return json_match.group(1)
    # If not found in 'text' field, look for JSON in the entire output
    json_pattern = r'({.*})'
    json_match = re.search(json_pattern, output, re.DOTALL)
    if json_match:
        return json_match.group(1)
    return None

def clean_json_string(json_string):
    if json_string is None:
        return None
    # Remove newlines and extra backslashes
    json_string = json_string.replace('\\n', '').replace('\\\\', '\\')
    # Remove leading/trailing whitespace
    json_string = json_string.strip()
    # Replace remaining escaped quotes and clean up any leftover single quotes or backslashes
    json_string = json_string.replace('\\"', '"').replace("'", "").replace("\\", "'")
    return json_string


def parse_output(output, debug=False):
    try:
        if debug:
            st.write("Original output:")
            st.write(output)
        json_string = extract_json_from_text(output)
        if json_string is None:
            if debug:
                st.write("Failed to extract JSON-like structure from the output.")
            return None, "No JSON-like structure found in the output."

        json_string = clean_json_string(json_string.split("response_metadata")[0])
        if debug:
            st.write("Extracted and cleaned JSON string:")
            st.write(json_string)

        # Attempt to parse the cleaned JSON
        try: 
            parsed_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            st.write("Error decoding JSON. Attemping automated repairs first.")
            parsed_json = json.loads(repair_json(json_string))
        if debug:
            st.write("Successfully parsed JSON:")
            st.write(parsed_json)
        return parsed_json, None

    except json.JSONDecodeError as e:
        error_message = f"JSON parsing error: {str(e)}"
        st.write(error_message)

        if 'json_string' in locals():
            st.write("Problematic JSON string:")
            st.write(json_string)
        else:
            st.write("No JSON string was extracted.")
        return None, error_message

    except Exception as e:
        error_message = f"JSON parsing error: {str(e)}"
        print(error_message)
        if 'json_string' in locals():
            print("Problematic JSON string:")
            print(json_string)
        else:
            print("No JSON string was extracted.")
        return None, error_message

def display_creativity_spectrum(creativity_spectrum):
    labels = ['Grounded', 'Creative', 'Wild']
    values = [creativity_spectrum['grounded'], creativity_spectrum['creative'], creativity_spectrum['wild']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker_colors=colors)])
    fig.update_layout(
        title_text="Creativity Spectrum",
        height=300  # Adjust this value to change the height of the chart
    )
    st.plotly_chart(fig, use_container_width=True)

def display_style_axes(style_axes):
    st.subheader("Style Axes")
    
    # Create the radar chart (as before)
    radar_fig = create_style_axes_chart(st.session_state.style_axes)
    
    # Create a radial bar chart
    axes = list(style_axes.keys())
    values = list(style_axes.values())
    
    radial_bar_fig = go.Figure()

    radial_bar_fig.add_trace(go.Barpolar(
        r=values,
        theta=axes,
        marker_color=values,
        marker_cmin=0,
        marker_cmax=100,
        marker_colorscale="Viridis",
        opacity=0.8
    ))

    radial_bar_fig.update_layout(
        title_text="Style Axes (Lofn Determined)",
        polar=dict(
            radialaxis=dict(range=[0, 100], showticklabels=True, ticks=''),
            angularaxis=dict(showticklabels=True, ticks='')
        ),
        height=500,
        showlegend=False
    )

    # Display both charts side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(radar_fig, use_container_width=True, title_text="Style Axes (Used Values)")
    with col2:
        st.plotly_chart(radial_bar_fig, use_container_width=True)

def display_facets(facets):
    st.subheader("Facets")
    cols = st.columns(len(facets))
    for i, facet in enumerate(facets):
        with cols[i]:
            st.markdown(f"**Facet {i+1}**")
            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>{facet}</div>", unsafe_allow_html=True)

def display_generation_progress(step, total_steps, process_name):
    progress = st.progress(0)
    status = st.empty()
    for i in range(total_steps):
        if i < step:
            progress.progress((i + 1) / total_steps)
        elif i == step:
            status.text(f"{process_name} - Step {i+1}/{total_steps}: {get_step_name(process_name, i)}")
            progress.progress((i + 1) / total_steps)
        else:
            break

def display_generation_results(title, data, is_dataframe=False):
    st.subheader(title)
    if is_dataframe:
        st.dataframe(data)
    elif isinstance(data, list):
        for i, item in enumerate(data, 1):
            if isinstance(item, dict):
                for key, value in item.items():
                    st.markdown(f"**{i}. {key}:** {value}")
            else:
                st.markdown(f"**{i}.** {item}")
            st.markdown("---")
    elif isinstance(data, dict):
        for key, value in data.items():
            st.markdown(f"**{key}:** {value}")
    else:
        st.write(data)

def create_mini_dashboard(pairs):
    cols = 3
    selected_pairs = []
    for i in range(0, len(pairs), cols):
        row = st.columns(cols)
        for j in range(cols):
            if i + j < len(pairs):
                pair = pairs[i + j]
                with row[j]:
                    st.markdown(f"**Pair {i+j+1}**")
                    with st.expander("View Details", expanded=True):
                        st.markdown(f"**Concept:** {pair['concept']}")
                        st.markdown(f"**Medium:** {pair['medium']}")
                    if st.checkbox(f"Select Pair {i+j+1}", key=f"pair_{i+j}"):
                        selected_pairs.append(i+j)
                    st.markdown("---")
    return selected_pairs

def get_step_name(process_name, step):
    steps = {
        "Concept Medium Generation": [
            "Generating Essence and Facets",
            "Generating Concepts",
            "Refining Concepts",
            "Generating Mediums",
            "Refining Mediums",
            "Reviewing and Shuffling"
        ],
        "Prompt Generation": [
            "Generating Facets",
            "Creating Artistic Guides",
            "Generating Image Prompts",
            "Refining Prompts",
            "Synthesizing Final Prompts"
        ]
    }
    return steps[process_name][step]

def display_temporary_results(title, data, is_dataframe=False):
    with st.expander(title, expanded=True):
        if is_dataframe:
            st.dataframe(data)
        elif isinstance(data, list):
            for item in data:
                st.write(item)
        elif isinstance(data, dict):
            for key, value in data.items():
                st.write(f"{key}: {value}")
        else:
            st.write(data)

# Initialize session state
def initialize_session_state():
    default_values = {
        'concept_mediums': None,
        'pairs_to_try': [0],
        'button_clicked': False,
        'webhook_url': webhook_url,
        'send_to_discord': True,
        'use_default_webhook': True,
        'concept_manual_mode': False,
        'essence_and_facets_output': None,
        'concepts_output': None,
        'artist_and_refined_concepts_output': None,
        'medium_output': None,
        'refined_medium_output': None,
        'shuffled_review_output': None,
        'proceed_concepts_clicked': False,
        'proceed_artist_refined_clicked': False,
        'proceed_mediums_clicked': False,
        'proceed_refined_mediums_clicked': False,
        'proceed_shuffled_reviews_clicked': False,
        'complete_all_steps_clicked': False,
        'image_model':'Poe-FLUX-pro'
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

concept_system = read_prompt('/lofn/prompts/concept_system.txt')

prompt_system = read_prompt('/lofn/prompts/prompt_system.txt')

prompt_ending = read_prompt('/lofn/prompts/prompt_ending.txt')

concept_header_part1 = read_prompt('/lofn/prompts/concept_header.txt')

concept_header_part2 = read_prompt('/lofn/prompts/concept_header_pt2.txt')

prompt_header_part1 = read_prompt("/lofn/prompts/prompt_header.txt")

prompt_header_part2 = read_prompt("/lofn/prompts/prompt_header_pt2.txt")

essence_prompt_middle = read_prompt("/lofn/prompts/essence_prompt.txt")

concepts_prompt_middle = read_prompt("/lofn/prompts/concepts_prompt.txt")

artist_and_critique_prompt_middle = read_prompt("/lofn/prompts/artist_and_critique_prompt.txt")

medium_prompt_middle = read_prompt("/lofn/prompts/medium_prompt.txt")

refine_medium_prompt_middle = read_prompt("/lofn/prompts/refine_medium_prompt.txt")

facets_prompt_middle = read_prompt("/lofn/prompts/facets_prompt.txt")

aspects_traits_prompt_middle = read_prompt("/lofn/prompts/aspects_traits_prompts.txt")

midjourney_prompt_middle = read_prompt("/lofn/prompts/imagegen_prompt.txt")

artist_refined_prompt_middle = read_prompt("/lofn/prompts/artist_refined_prompt.txt")

revision_synthesis_prompt_middle = read_prompt("/lofn/prompts/revision_synthesis_prompt.txt")

dalle3_gen_prompt_middle = read_prompt("/lofn/prompts/dalle3_gen_prompt.txt")

dalle3_gen_prompt_nodiv_middle = read_prompt("/lofn/prompts/dalle3_gen_nodiv_prompt.txt")

image_title_prompt_middle = read_prompt("/lofn/prompts/image_title_prompt.txt")

# Read aesthetics from the file
with open('/lofn/prompts/aesthetics.txt', 'r') as file:
    aesthetics = file.read().split(', ')

# Randomly select 50 aesthetics
# concept_selected_aesthetics = random.sample(aesthetics, 100)
# concept_aesthetics_string = ', '.join(concept_selected_aesthetics)

# prompt_selected_aesthetics = random.sample(aesthetics, 100)
# prompt_aesthetics_string = ', '.join(prompt_selected_aesthetics)

prompt_header = prompt_header_part1 + prompt_header_part2

concept_header = concept_header_part1 + concept_header_part2

essence_prompt = concept_header + essence_prompt_middle + prompt_ending 

concepts_prompt = concept_header + concepts_prompt_middle + prompt_ending

artist_and_critique_prompt = concept_header + artist_and_critique_prompt_middle + prompt_ending

# Prompt template for choosing medium for each artist-refined concept
medium_prompt = concept_header + medium_prompt_middle + prompt_ending

# Prompt template for refining the medium choice based on artist's critique
refine_medium_prompt = concept_header + refine_medium_prompt_middle + prompt_ending

facets_prompt = concept_header + facets_prompt_middle + prompt_ending

# Defining the prompt template for List of Aspects and Traits
aspects_traits_prompt = prompt_header + aspects_traits_prompt_middle + prompt_ending

# Defining the prompt template for Midjourney Prompts
midjourney_prompt = prompt_header + midjourney_prompt_middle + prompt_ending


# Defining the prompt template for Artist Selection and Refined Prompts
artist_refined_prompt = prompt_header + artist_refined_prompt_middle + prompt_ending

revision_synthesis_prompt = concept_header + revision_synthesis_prompt_middle + prompt_ending

dalle3_gen_prompt = dalle3_gen_prompt_middle + prompt_ending

dalle3_gen_nodiv_prompt = dalle3_gen_prompt_nodiv_middle + prompt_ending

image_title_prompt = prompt_header + image_title_prompt_middle + prompt_ending 


def get_llm(model, temperature, openai_api_key=OPENAI_API, anthropic_api_key=ANTHROPIC_API, google_api_key=GOOGLE_API, poe_api_key=POE_API):
    # Dictionary mapping models to their maximum token limits
    model_max_tokens = {
        # OpenAI models
        "gpt-4o-mini": 4096,
        "gpt-4o": 4096,
        "gpt-4o-2024-08-06": 8192,
        "gpt-3.5-turbo": 4096,
        "gpt-4-turbo": 4096,
        "gpt-4": 8192,

        # Anthropic models
        "claude-3-5-sonnet-20240620": 4096,
        "claude-3-opus-20240229": 4096,
        "claude-3-sonnet-20240229": 4096,
        "claude-3-haiku-20240307": 4096,

        # Google models
        "gemini-1.5-flash": 16384,
        "gemini-1.5-pro": 32768,
        "gemini-1.0-pro": 32768,

        # Poe models
        "Poe-Assistant": 4096,  # FIXME: Verify token limit
        "Poe-Claude-3.5-Sonnet": 4096,
        "Poe-GPT-4o-Mini": 4096,
        "Poe-GPT-4o": 8192,
        "Poe-Llama-3.1-405B-T": 4096,  # FIXME: Verify token limit
        "Poe-Gemini-1.5-Flash": 16384,
        "Poe-Gemini-1.5-Pro": 32768,
        "Poe-Llama-3.1-8B-T-128k": 128000,
        "Poe-Llama-3.1-70B-FW-128k": 128000,
        "Poe-Llama-3.1-70B-T-128k": 128000,
        "Poe-Llama-3.1-8B-FW-128k": 128000,
        "Poe-Llama-3-70b-Groq": 4096,  # FIXME: Verify token limit
        "Poe-Gemma-2-27b-T": 4096,  # FIXME: Verify token limit
        "Poe-Claude-3-Sonnet": 4096,
        "Poe-Claude-3-Haiku": 4096,
        "Poe-Claude-3-Opus": 4096,
        "Poe-Gemini-1.5-Flash-128k": 128000,
        "Poe-Gemini-1.5-Pro-128k": 128000,
        "Poe-Gemini-1.0-Pro": 32768,
        "Poe-Llama-3-70B-T": 4096,  # FIXME: Verify token limit
        "Poe-Llama-3-70b-Inst-FW": 4096,  # FIXME: Verify token limit
        "Poe-Mixtral8x22b-Inst-FW": 4096,  # FIXME: Verify token limit
        "Poe-Command-R": 4096,  # FIXME: Verify token limit
        "Poe-Gemma-2-9b-T": 4096,  # FIXME: Verify token limit
        "Poe-Mistral-Large-2": 4096,  # FIXME: Verify token limit
        "Poe-Mistral-Medium": 4096,  # FIXME: Verify token limit
        "Poe-Snowflake-Arctic-T": 4096,  # FIXME: Verify token limit
        "Poe-RekaCore": 4096,  # FIXME: Verify token limit
        "Poe-RekaFlash": 4096,  # FIXME: Verify token limit
        "Poe-Command-R-Plus": 4096,  # FIXME: Verify token limit
        "Poe-GPT-3.5-Turbo": 4096,
        "Poe-Mixtral-8x7B-Chat": 4096,  # FIXME: Verify token limit
        "Poe-DeepSeek-Coder-33B-T": 4096,  # FIXME: Verify token limit
        "Poe-CodeLlama-70B-T": 4096,  # FIXME: Verify token limit
        "Poe-Qwen2-72B-Chat": 4096,  # FIXME: Verify token limit
        "Poe-Qwen-72B-T": 4096,  # FIXME: Verify token limit
        "Poe-Claude-2": 100000,
        "Poe-Google-PaLM": 4096,  # FIXME: Verify token limit
        "Poe-Llama-3-8b-Groq": 4096,  # FIXME: Verify token limit
        "Poe-Llama-3-8B-T": 4096,  # FIXME: Verify token limit
        "Poe-Gemma-Instruct-7B-T": 4096,  # FIXME: Verify token limit
        "Poe-MythoMax-L2-13B": 4096,  # FIXME: Verify token limit
        "Poe-Code-Llama-34b": 4096,  # FIXME: Verify token limit
        "Poe-Code-Llama-13b": 4096,  # FIXME: Verify token limit
        "Poe-Solar-Mini": 4096,  # FIXME: Verify token limit
        "Poe-GPT-3.5-Turbo-Instruct": 4096,
        "Poe-GPT-3.5-Turbo-Raw": 4096,
        "Poe-Claude-instant": 100000,
        "Poe-Mixtral-8x7b-Groq": 4096,  # FIXME: Verify token limit
        "Poe-Mistral-7B-v0.3-T": 4096,  # FIXME: Verify token limit
    }

    # Get the maximum token limit for the selected model
    max_tokens = model_max_tokens.get(model, 4096)  # Default to 4096 if not specified

    if model.startswith("claude"):
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens, anthropic_api_key=anthropic_api_key)
    if model.startswith("gemini"):
        return GeminiLLM(model_name=model, api_key=google_api_key, temperature=temperature, max_tokens=model_max_tokens.get(model, 4096))
    elif model.startswith("Poe-"):
        return PoeLLM(model_name=model, api_key=poe_api_key, temperature=temperature, max_tokens=max_tokens)
    else:
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, openai_api_key=openai_api_key)

def run_chain_with_retries(_lang_chain, max_retries, args_dict=None, is_correction=False, debug=False):
    output = None
    retry_count = 0
    while retry_count < max_retries:
        try:
            output = run_any_chain(_lang_chain, args_dict, is_correction, retry_count, debug)
            args_dict['output'] = output
            if debug:
                st.write(f"Raw output from LLM:\n{output}")
            
            # Parse the output
            parsed_output, error = parse_output(str(output), debug)
            if parsed_output is not None:
                if debug:
                    st.write("Successfully parsed JSON output")
                return parsed_output
            else:
                st.write(f"Failed to parse JSON: {error}")
                raise ValueError(f"Invalid JSON: {error}")
        except Exception as e:
            st.write(f"An error occurred in attempt {retry_count + 1}: {str(e)}")
            retry_count += 1
            is_correction = True  # Use correction prompt in next iteration
    if retry_count >= max_retries:
        st.write("Max retries reached. Exiting.")
    return None

def truncate_prompt(prompt: Union[str, List[dict]], max_tokens: int = 3000) -> Union[str, List[dict]]:
    """
    Truncate a prompt string or a list of message dicts to a maximum number of tokens.
    """
    if isinstance(prompt, str):
        words = prompt.split()
        if len(words) <= max_tokens:
            return prompt
        return ' '.join(words[-max_tokens:])
    elif isinstance(prompt, list):
        total_tokens = sum(len(m['content'].split()) for m in prompt)
        if total_tokens <= max_tokens:
            return prompt
        truncated_prompt = []
        remaining_tokens = max_tokens
        for message in reversed(prompt):
            content = message['content']
            words = content.split()
            if len(words) <= remaining_tokens:
                truncated_prompt.insert(0, message)
                remaining_tokens -= len(words)
            else:
                truncated_content = ' '.join(words[-remaining_tokens:])
                truncated_prompt.insert(0, {**message, 'content': truncated_content})
                break
        return truncated_prompt

def run_any_chain(chain, args_dict, is_correction, retry_count, debug=False):
    try:
        if is_correction:
            correction_prompt = f"""
            Attempt {retry_count + 1}: The previous response was not in the correct JSON format or was incomplete.
            Please refer to the instructions provided earlier and respond with only the complete JSON output.
            Ensure that all required fields are included and properly formatted according to the instructions.
            Escape any special characters within JSON strings including apostrophes. The last portion of your
            previous message is included. Please start from the last complete step using your previous output.
            """
            # Preserve original input parameters
            corrected_args = args_dict.copy()
            if "Poe" in model:
                corrected_args['input'] = correction_prompt + "\n\nOriginal query: " + args_dict.get('input', '') + "\n\n End of last output - start from here: " + args_dict.get('output', '')[-200:]
            else:
                corrected_args['input'] = correction_prompt + "\n\nOriginal query: " + args_dict.get('input', '') + "\n\n End of last output - start from here: " + args_dict.get('output', '')
            
            if debug:
                st.write(f"Attempt {retry_count + 1}: Using correction prompt")
                st.write(f"Corrected input args: {corrected_args}")
            response = chain.invoke(corrected_args)
        else:
            if debug:
                st.write(f"Attempt {retry_count + 1}: Using original prompt")
                st.write(f"Input args: {args_dict}")
            response = chain.invoke(args_dict)
        
        if isinstance(response, dict) and 'text' in response:
            full_response = response['text']
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



def send_to_discord(content, content_type='prompts', premessage=''):
    try:
        if st.session_state['send_to_discord']:
            if premessage:
                requests.post(st.session_state['webhook_url'], data=json.dumps({"content": f'INCOMING MESSAGE: {premessage}'}), headers={"Content-Type": "application/json"})
            
            if content_type == 'prompts':
                for prompt in content:
                    message = {"content": prompt}
                    requests.post(st.session_state['webhook_url'], data=json.dumps(message), headers={"Content-Type": "application/json"})
            elif content_type == 'concepts':
                requests.post(st.session_state['webhook_url'], data=json.dumps({"content": f'INCOMING MESSAGE for Patron\'s idea: {st.session_state.input}'}), headers={"Content-Type": "application/json"})
                for con_dict in content:
                    message = {"content": f'{{concept}} = {con_dict["concept"]} \n{{medium}} = {con_dict["medium"]}'}
                    requests.post(st.session_state['webhook_url'], data=json.dumps(message), headers={"Content-Type": "application/json"})
    except Exception as e:
        st.write(f"An error occurred while sending to Discord: {str(e)}")

def run_llm_chain(chains, chain_name, args_dict, max_retries):
    chain = chains[chain_name]
    output = run_chain_with_retries(chain, max_retries=max_retries, args_dict=args_dict, is_correction=False, debug=debug)
    
    if output is None:
        st.error(f"Failed to get valid JSON response after {max_retries} attempts.")
        return None
    
    return output

def process_essence_and_facets(chains, input, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'essence_and_facets', {"input": input}, max_retries)
    if parsed_output is None:
        return None
    
    if "essence_and_facets" in parsed_output:
        st.session_state.essence_and_facets_output = parsed_output
        st.session_state.creativity_spectrum = parsed_output["essence_and_facets"]["creativity_spectrum"]
        if st.session_state.auto_style:
            st.session_state.style_axes = parsed_output["essence_and_facets"]["style_axes"]
    else:
        st.error(f"Failed to process essence and facets: Unexpected output structure")
        return None
    return parsed_output

def process_concepts(chains, input, essence, facets, style_axes, creativity_spectrum, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'concepts', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_wild": creativity_spectrum['wild'],
        "creativity_spectrum_creative": creativity_spectrum['creative'],
        "creativity_spectrum_grounded": creativity_spectrum['grounded'],
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process concepts")
        return None
    return parsed_output


def process_artist_and_refined_concepts(chains, input, essence, facets, style_axes, creativity_spectrum, concepts, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'artist_and_refined_concepts', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_wild": creativity_spectrum['wild'],
        "creativity_spectrum_creative": creativity_spectrum['creative'],
        "creativity_spectrum_grounded": creativity_spectrum['grounded'],
        "concepts": [x['concept'] for x in concepts['concepts']]
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process artist and refined concepts")
        return None
    return parsed_output

def process_mediums(chains, input, essence, facets, style_axes, creativity_spectrum, refined_concepts, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'medium', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "refined_concepts": [x['refined_concept'] for x in refined_concepts['refined_concepts']],
        "creativity_spectrum_wild": creativity_spectrum['wild'],
        "creativity_spectrum_creative": creativity_spectrum['creative'],
        "creativity_spectrum_grounded": creativity_spectrum['grounded'],
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process mediums")
        return None
    return parsed_output

def process_refined_mediums(chains, input, essence, facets, style_axes, creativity_spectrum, mediums, artists, refined_concepts, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'refine_medium', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_wild": creativity_spectrum['wild'],
        "creativity_spectrum_creative": creativity_spectrum['creative'],
        "creativity_spectrum_grounded": creativity_spectrum['grounded'],
        "mediums": [x['medium'] for x in mediums['mediums']],
        "artists": artists,
        "refined_concepts": [x['refined_concept'] for x in refined_concepts['refined_concepts']]
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process refined mediums")
        return None
    return parsed_output

def process_shuffled_review(chains, input, essence, facets, style_axes, creativity_spectrum, mediums, artists, refined_concepts, max_retries, debug=False):
    review_artists = np.random.permutation(artists).tolist()
    parsed_output = run_llm_chain(chains, 'shuffled_review', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_wild": creativity_spectrum['wild'],
        "creativity_spectrum_creative": creativity_spectrum['creative'],
        "creativity_spectrum_grounded": creativity_spectrum['grounded'],
        "mediums": [x['medium'] for x in mediums['mediums']],
        "artists": review_artists,
        "refined_concepts": [x['refined_concept'] for x in refined_concepts['refined_concepts']]
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process shuffled review")
        return None
    return parsed_output

def process_facets(chains, input, concept, medium, style_axes, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'facets', {"input": input, "concept": concept, "medium": medium, "style_axes":style_axes}, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process facets")
        return None
    return parsed_output

def process_artistic_guides(chains, input, concept, medium, facets, style_axes, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'aspects_traits', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process artistic guides")
        return None
    return parsed_output

def process_midjourney_prompts(chains, input, concept, medium, facets, style_axes, artistic_guides, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'midjourney', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "artistic_guides": [x['artistic_guide'] for x in artistic_guides['artistic_guides']]
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process midjourney prompts")
        return None
    if parsed_output.get('image_gen_prompts'):
        send_to_discord([prompt['image_gen_prompt'] for prompt in parsed_output['image_gen_prompts']], premessage=f'Generated Prompts for {concept} in {medium}:')
    return parsed_output

def process_artist_refined_prompts(chains, input, concept, medium, facets, style_axes, image_gen_prompts, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'artist_refined', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "image_gen_prompts": [x['image_gen_prompt'] for x in image_gen_prompts['image_gen_prompts']]
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process artist refined prompts")
        return None
    if parsed_output.get('artist_refined_prompts'):
        send_to_discord([prompt['artist_refined_prompt'] for prompt in parsed_output['artist_refined_prompts']], premessage=f'Artist-Refined Prompts for {concept} in {medium}:')
    return parsed_output

def process_revised_synthesized_prompts(chains, input, concept, medium, facets, style_axes, artist_refined_prompts, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'revision_synthesis', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "artist_refined_prompts": [x['artist_refined_prompt'] for x in artist_refined_prompts['artist_refined_prompts']]
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process revised synthesized prompts")
        return None
    if parsed_output.get('revised_prompts'):
        send_to_discord([prompt['revised_prompt'] for prompt in parsed_output['revised_prompts']], premessage=f'Revised Prompts for {concept} in {medium}:')
    if parsed_output.get('synthesized_prompts'):
        send_to_discord([prompt['synthesized_prompt'] for prompt in parsed_output['synthesized_prompts']], premessage=f'Synthesized Prompts for {concept} in {medium}:')
    return parsed_output

def display_generation_progress(step, total_steps, process_name):
    progress = st.progress(0)
    status = st.empty()
    for i in range(total_steps):
        if i < step:
            progress.progress((i + 1) / total_steps)
        elif i == step:
            status.text(f"{process_name} - Step {i+1}/{total_steps}: {get_step_name(process_name, i)}")
            progress.progress((i + 1) / total_steps)
        else:
            break

def get_step_name(process_name, step):
    steps = {
        "Concept Medium Generation": [
            "Generating Essence and Facets",
            "Generating Concepts",
            "Refining Concepts",
            "Generating Mediums",
            "Refining Mediums",
            "Reviewing and Shuffling"
        ],
        "Prompt Generation": [
            "Generating Facets",
            "Creating Artistic Guides",
            "Generating Image Prompts",
            "Refining Prompts",
            "Synthesizing Final Prompts"
        ]
    }
    return steps[process_name][step]


@st.cache_data(persist=True)
def generate_concept_mediums(input, max_retries, temperature, model="gpt-3.5-turbo-16k", verbose=False, debug=False, aesthetics=aesthetics):
    try:
        llm = get_llm(model, temperature, OPENAI_API, ANTHROPIC_API)
        selected_aesthetics = random.sample(aesthetics, 100)
        if "Poe" in model:
            selected_aesthetics = selected_aesthetics[:24]

        chains = {
            'essence_and_facets': (
                ChatPromptTemplate.from_messages([
                    ("system", concept_system),
                    ("human", essence_prompt)
                ])
                | llm
            ),
            'concepts': (
                ChatPromptTemplate.from_messages([
                    ("system", concept_system),
                    ("human", concepts_prompt)
                ])
                | llm
            ),
            'artist_and_refined_concepts': (
                ChatPromptTemplate.from_messages([
                    ("system", concept_system),
                    ("human", artist_and_critique_prompt)
                ])
                | llm
            ),
            'medium': (
                ChatPromptTemplate.from_messages([
                    ("system", concept_system),
                    ("human", medium_prompt)
                ])
                | llm
            ),
            'refine_medium': (
                ChatPromptTemplate.from_messages([
                    ("system", concept_system),
                    ("human", refine_medium_prompt)
                ])
                | llm
            ),
            'shuffled_review': (
                ChatPromptTemplate.from_messages([
                    ("system", concept_system),
                    ("human", refine_medium_prompt)
                ])
                | llm
            )
        }

        with st.status("Generating Concepts and Mediums...", expanded=True) as status:
            # Step 1: Essence and Facets
            status.write("Generating Essence and Facets...")
            essence_and_facets = process_essence_and_facets(chains, input, max_retries, debug)
            if essence_and_facets:
                display_creativity_spectrum(essence_and_facets["essence_and_facets"]["creativity_spectrum"])
                display_facets(essence_and_facets["essence_and_facets"]["facets"])
                display_style_axes(essence_and_facets["essence_and_facets"]["style_axes"])
            
            # Step 2: Concepts
            status.write("Generating Concepts...")
            concepts = process_concepts(
                chains, 
                input, 
                essence_and_facets["essence_and_facets"]["essence"], 
                essence_and_facets["essence_and_facets"]["facets"], 
                st.session_state.style_axes, 
                st.session_state.creativity_spectrum, 
                max_retries, 
                debug)
            if debug:
                st.write("Initial Concepts:")
                for i, concept in enumerate(concepts['concepts'], 1):
                    st.write(f"{i}. {concept['concept']}")
            
            # Step 3: Refined Concepts
            status.write("Refining Concepts...")
            artist_and_refined_concepts = process_artist_and_refined_concepts(
                chains, 
                input, 
                essence_and_facets["essence_and_facets"]["essence"], 
                essence_and_facets["essence_and_facets"]["facets"],  
                st.session_state.style_axes,
                st.session_state.creativity_spectrum, 
                concepts,
                max_retries, 
                debug)
            if debug:
                st.write("Refined Concepts:")
                for i, concept in enumerate(artist_and_refined_concepts['refined_concepts'], 1):
                    st.write(f"{i}. {concept['refined_concept']}")
            
            # Step 4: Generating Mediums
            status.write("Generating Mediums...")
            mediums = process_mediums(
                chains, 
                input, 
                essence_and_facets["essence_and_facets"]["essence"], 
                essence_and_facets["essence_and_facets"]["facets"],  
                st.session_state.style_axes, 
                st.session_state.creativity_spectrum,
                artist_and_refined_concepts, 
                max_retries, 
                debug)
            if debug:
                st.write("Initial Mediums:")
                for i, medium in enumerate(mediums['mediums'], 1):
                    st.write(f"{i}. {medium['medium']}")
            
            # Step 5: Refining Mediums
            status.write("Refining Mediums...")
            refined_mediums = process_refined_mediums(
                chains, 
                input, 
                essence_and_facets["essence_and_facets"]["essence"], 
                essence_and_facets["essence_and_facets"]["facets"],  
                st.session_state.style_axes, 
                st.session_state.creativity_spectrum, 
                mediums, 
                [x['artist'] for x in artist_and_refined_concepts['artists']], 
                artist_and_refined_concepts, 
                max_retries, 
                debug)
            if debug:
                st.write("Refined Concepts:")
                for i, concept in enumerate(refined_mediums['refined_concepts'], 1):
                    st.write(f"{i}. {concept['refined_concept']}")
                st.write("Refined Mediums:")
                for i, medium in enumerate(refined_mediums['refined_mediums'], 1):
                    st.write(f"{i}. {medium['refined_medium']}")
            
            # Step 6: Shuffling and Reviewing
            status.write("Shuffling and Reviewing...")
            shuffled_review = process_shuffled_review(
                chains, 
                input, 
                essence_and_facets["essence_and_facets"]["essence"], 
                essence_and_facets["essence_and_facets"]["facets"],  
                st.session_state.style_axes,
                st.session_state.creativity_spectrum, 
                mediums, 
                [x['artist'] for x in artist_and_refined_concepts['artists']], 
                refined_mediums, 
                max_retries, 
                debug)

            status.update(label="Generation Complete!", state="complete")

        refined_concepts = [x['refined_concept'] for x in shuffled_review['refined_concepts']]
        refined_mediums = [x['refined_medium'] for x in shuffled_review['refined_mediums']]
        concept_mediums = [{'concept': concept, 'medium': medium} for concept, medium in zip(refined_concepts, refined_mediums)]
        
        send_to_discord(concept_mediums, content_type='concepts')
        return concept_mediums
    except Exception as e:
        raise LofnError(f"Error in concept generation: {str(e)}")        

@st.cache_data(persist=True)
def generate_prompts(input, concept, medium, max_retries, temperature, model="gpt-3.5-turbo-16k", debug=False, aesthetics=aesthetics, image_model="None"):
    try:
        llm = get_llm(model, temperature, OPENAI_API, ANTHROPIC_API)
        selected_aesthetics = random.sample(aesthetics, 100)
        if "Poe" in model:
            selected_aesthetics = selected_aesthetics[:24]


        chains = {
            'facets': (
                ChatPromptTemplate.from_messages([("system", concept_system), ("human", facets_prompt)])
                | llm
            ),
            'aspects_traits': (
                ChatPromptTemplate.from_messages([("system", prompt_system), ("human", aspects_traits_prompt)])
                | llm
            ),
            'midjourney': (
                ChatPromptTemplate.from_messages([("system", prompt_system), ("human", midjourney_prompt)])
                | llm
            ),
            'artist_refined': (
                ChatPromptTemplate.from_messages([("system", prompt_system), ("human", artist_refined_prompt)])
                | llm
            ),
            'revision_synthesis': (
                ChatPromptTemplate.from_messages([("system", prompt_system), ("human", revision_synthesis_prompt)])
                | llm
            )
        }
        with st.status(f"Generating Prompts for {concept} in {medium}...", expanded=True) as status:
            status.write("Generating Facets...")
            facets = process_facets(chains, input, concept, medium, st.session_state.style_axes, max_retries, debug)

            display_facets(facets['facets'])
            
            status.write("Creating Artistic Guides...")
            artistic_guides = process_artistic_guides(chains, input, concept, medium, facets, st.session_state.style_axes, max_retries, debug)
            if debug:
                st.write("Artistic Guides:")
                for i, guide in enumerate(artistic_guides['artistic_guides'], 1):
                    st.write(f"{i}. {guide['artistic_guide']}")
            
            status.write("Generating Image Prompts...")
            midjourney_prompts = process_midjourney_prompts(chains, input, concept, medium, facets, st.session_state.style_axes, artistic_guides, max_retries, debug)
            if debug:
                st.write("Midjourney Prompts:")
                for i, prompt in enumerate(midjourney_prompts['image_gen_prompts'], 1):
                    st.write(f"{i}. {prompt['image_gen_prompt']}")
            
            status.write("Refining Prompts...")
            artist_refined_prompts = process_artist_refined_prompts(chains, input, concept, medium, facets, st.session_state.style_axes, midjourney_prompts, max_retries, debug)
            if debug:
                st.write("Artist Refined Prompts:")
                for i, prompt in enumerate(artist_refined_prompts['artist_refined_prompts'], 1):
                    st.write(f"{i}. {prompt['artist_refined_prompt']}")
            
            status.write("Synthesizing Final Prompts...")
            revised_synthesized_prompts = process_revised_synthesized_prompts(chains, input, concept, medium, facets, st.session_state.style_axes, artist_refined_prompts, max_retries, debug)

            status.update(label="Prompt Generation Complete!", state="complete")

        # Display the final prompts
        st.subheader("Final Prompts")
        for i, (revised, synthesized) in enumerate(zip(revised_synthesized_prompts['revised_prompts'], revised_synthesized_prompts['synthesized_prompts']), 1):
            st.markdown(f"**Prompt Set {i}:**")
            st.markdown(f"*Revised:* {revised['revised_prompt']}")
            st.markdown(f"*Synthesized:* {synthesized['synthesized_prompt']}")
            st.markdown("---")

        df_prompts = pd.DataFrame({
            'Revised Prompts': [prompt['revised_prompt'] for prompt in revised_synthesized_prompts['revised_prompts']],
            'Synthesized Prompts': [prompt['synthesized_prompt'] for prompt in revised_synthesized_prompts['synthesized_prompts']]
        })    

        if image_model != 'None':
            generate_dalle_images(input, concept, medium, df_prompts, max_retries, temperature, model, debug, image_model)

        return revised_synthesized_prompts
    except Exception as e:
        raise LofnError(f"Error in prompt generation: {str(e)}")        

def generate_all_prompts(input, concept_mediums, max_retries, temperature, model, debug, aesthetics=aesthetics, image_model = "None"):
    results = []
    total_pairs = len(concept_mediums)
    
    for i, pair in enumerate(concept_mediums):
        st.write(f"Generating prompts for pair {i+1}/{total_pairs}: {pair['concept']} in {pair['medium']}")
        result = generate_prompts(input, pair['concept'], pair['medium'], max_retries, temperature, model, debug, aesthetics, image_model)
        results.append(result)
        st.markdown("---")  # Add a separator between each pair's results
        
    return results

st.set_page_config(page_title="Lofn - The AI Artist", page_icon=":art:", layout="wide")
with open("/lofn/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

initialize_session_state()

st.title("LOFN - The AI Artist")


st.sidebar.header('Style Personalization')
st.session_state.auto_style = st.sidebar.checkbox("Automatic Style", value=True)

if not st.session_state.auto_style:
    st.sidebar.subheader("Adjust Style Axes")
    style_axes = {
        "Abstraction vs. Realism": st.sidebar.slider("Abstraction vs. Realism (0: Abstract)", 0, 100, 50),
        "Emotional Valence": st.sidebar.slider("Emotional Valence (0: Negative)", 0, 100, 50),
        "Color Intensity": st.sidebar.slider("Color Intensity (0: Muted)", 0, 100, 50),
        "Symbolic Density": st.sidebar.slider("Symbolic Density (0: Literal)", 0, 100, 50),
        "Compositional Complexity": st.sidebar.slider("Compositional Complexity (0: Simple)", 0, 100, 50),
        "Textural Richness": st.sidebar.slider("Textural Richness (0: Smooth)", 0, 100, 50),
        "Symmetry vs. Asymmetry": st.sidebar.slider("Symmetry vs. Asymmetry (0: Asymmetrical)", 0, 100, 50),
        "Novelty": st.sidebar.slider("Novelty (0: Traditional)", 0, 100, 50),
        "Figure-Ground Relationship": st.sidebar.slider("Figure-Ground Relationship (0: Distinct)", 0, 100, 50),
        "Dynamic vs. Static": st.sidebar.slider("Dynamic vs. Static (0: Static)", 0, 100, 50)
    }
else:
    style_axes = None

if 'style_axes' not in st.session_state:
    st.session_state.style_axes = None

# Add style_axes to session state
st.session_state['style_axes'] = style_axes

st.sidebar.header('Image Generation Settings')

poe_models = [
    "Poe-Assistant", "Poe-Claude-3.5-Sonnet", "Poe-GPT-4o-Mini", "Poe-GPT-4o",
    "Poe-Llama-3.1-405B-T", "Poe-Gemini-1.5-Flash", "Poe-Gemini-1.5-Pro",
    "Poe-Llama-3.1-8B-T-128k", "Poe-Llama-3.1-70B-FW-128k", "Poe-Llama-3.1-70B-T-128k",
    "Poe-Llama-3.1-8B-FW-128k", "Poe-Llama-3-70b-Groq", "Poe-Gemma-2-27b-T",
    "Poe-Claude-3-Sonnet", "Poe-Claude-3-Haiku", "Poe-Claude-3-Opus",
    "Poe-Gemini-1.5-Flash-128k", "Poe-Gemini-1.5-Pro-128k", "Poe-Gemini-1.0-Pro",
    "Poe-Llama-3-70B-T", "Poe-Llama-3-70b-Inst-FW", "Poe-Mixtral8x22b-Inst-FW",
    "Poe-Command-R", "Poe-Gemma-2-9b-T", "Poe-Mistral-Large-2", "Poe-Mistral-Medium",
    "Poe-Snowflake-Arctic-T", "Poe-RekaCore", "Poe-RekaFlash", "Poe-Command-R-Plus",
    "Poe-GPT-3.5-Turbo", "Poe-Mixtral-8x7B-Chat", "Poe-DeepSeek-Coder-33B-T",
    "Poe-CodeLlama-70B-T", "Poe-Qwen2-72B-Chat", "Poe-Qwen-72B-T", "Poe-Claude-2",
    "Poe-Google-PaLM", "Poe-Llama-3-8b-Groq", "Poe-Llama-3-8B-T", "Poe-Gemma-Instruct-7B-T",
    "Poe-MythoMax-L2-13B", "Poe-Code-Llama-34b", "Poe-Code-Llama-13b", "Poe-Solar-Mini",
    "Poe-GPT-3.5-Turbo-Instruct", "Poe-GPT-3.5-Turbo-Raw", "Poe-Claude-instant",
    "Poe-Mixtral-8x7b-Groq", "Poe-Mistral-7B-v0.3-T"
]

model = st.sidebar.selectbox("Select language model", 
    ["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-08-06", "claude-3-5-sonnet-20240620", "gpt-3.5-turbo", 
     "claude-3-opus-20240229", "gpt-4-turbo", "claude-3-sonnet-20240229", 
     "claude-3-haiku-20240307", "gpt-4",
     "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"] + poe_models)

poe_image_models = [
    "Poe-Playground-v2.5", "Poe-Ideogram", "Poe-FLUX-dev",
    "Poe-FLUX-schnell", "Poe-LivePortrait", "Poe-StableDiffusion3",
    "Poe-SD3-Turbo", "Poe-StableDiffusionXL", "Poe-StableDiffusion3-2B",
    "Poe-SD3-Medium", "Poe-RealVisXL"
]
image_model = st.sidebar.selectbox(
    "Select image model",
    ["Poe-FLUX-pro", "Poe-DALL-E-3", "fal-ai/flux/schnell", "None", "DALL-E 3", "fal-ai/flux-pro", "fal-ai/flux/dev", 
     "fal-ai/aura-flow", "fal-ai/stable-diffusion-v3-medium", "fal-ai/fast-sdxl", 
     "fal-ai/hyper-sdxl", "fal-ai/playground-v25"] + poe_image_models,
    key="image_model",
    on_change=update_image_controls
)

st.sidebar.header('Pika Video Generation Settings')
enable_pika_video = st.sidebar.checkbox("Enable Pika Video Generation", value=False)

if enable_pika_video:
    st.sidebar.subheader("Pika Video Parameters")
    pika_motion = st.sidebar.slider("Motion", min_value=1, max_value=4, value=1)
    pika_guidance_scale = st.sidebar.slider("Guidance Scale", min_value=5, max_value=25, value=12)
    pika_frame_rate = st.sidebar.slider("Frame Rate", min_value=1, max_value=24, value=24)
    pika_aspect_ratio = st.sidebar.selectbox("Aspect Ratio", ["16:9", "9:16", "1:1", "5:2", "4:5", "4:3"])
    pika_negative_prompt = st.sidebar.text_area("Negative Prompt", "")
    pika_camera_zoom = st.sidebar.selectbox("Camera Zoom", [None, "in", "out"])
    pika_camera_pan = st.sidebar.selectbox("Camera Pan", [None, "left", "right"])
    pika_camera_tilt = st.sidebar.selectbox("Camera Tilt", [None, "up", "down"])
    pika_camera_rotate = st.sidebar.selectbox("Camera Rotate", [None, "cw", "ccw"])

# Sidebar settings
st.sidebar.header('Patron Input Features')

manual_input = st.sidebar.checkbox("Manually input Concept and Medium")
st.session_state['send_to_discord'] = st.sidebar.checkbox("Send to Discord", st.session_state['send_to_discord'])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, step=0.02)
debug = st.sidebar.checkbox("Debug Mode")
enable_diversity = st.sidebar.checkbox("Enable Forced Diversity", value=False)
max_retries = st.sidebar.slider("Maximum Retries", 0, 10, 3)

# Discord settings
st.sidebar.header('Discord Settings')
if st.sidebar.button("Toggle Webhook URL"):
    st.session_state['use_default_webhook'] = not st.session_state['use_default_webhook']
    st.session_state['webhook_url'] = webhook_url if st.session_state['use_default_webhook'] else ''

if st.session_state['use_default_webhook']:
    st.sidebar.text("Webhook URL: **********")
else:
    st.session_state['webhook_url'] = st.sidebar.text_input("Discord Webhook URL", st.session_state['webhook_url'])

with st.container():
    st.session_state.input = st.text_area("Describe your idea", "I want to capture the essence of a mysterious and powerful witch's familiar.")
    if image_model != "None":
        render_image_controls(image_model)

    if st.button("Generate Concepts"):
        st.session_state.button_clicked = True

    if st.session_state.button_clicked:
        try:
            st.write("Generating")
            st.session_state['concept_mediums'] = generate_concept_mediums(st.session_state.input, max_retries=max_retries, temperature=temperature, model=model, debug=debug)
        except LofnError as e:
            st.error(f"An error occurred during concept generation: {str(e)}")
            if debug:
                st.exception(e)
            st.warning("Please try again or enable debug mode for more information.")
            st.session_state.button_clicked = False

    if st.session_state['concept_mediums'] is not None:
        st.write("Concepts and Mediums Generated")
        
        # Display the final pairs using the mini-dashboard
        st.subheader("Concept and Medium Pairs")
        selected_pairs = create_mini_dashboard(st.session_state['concept_mediums'])
        
        if st.button("Generate Image Prompts"):
            for pair_i in selected_pairs:
                pair = st.session_state['concept_mediums'][pair_i]
                st.write(f"Generating prompts for Pair {pair_i + 1}:")
                st.markdown(f"*Concept:* {pair['concept']}")
                st.markdown(f"*Medium:* {pair['medium']}")
                try:
                    prompts = generate_prompts(st.session_state.input, pair['concept'], pair['medium'], model=model, debug=debug, max_retries=max_retries, temperature=temperature, aesthetics=aesthetics, image_model=image_model)
                    
                    dalle_prompt = dalle3_gen_prompt if enable_diversity else dalle3_gen_nodiv_prompt
                    st.code(dalle_prompt.format(
                        concept=pair['concept'], 
                        medium=pair['medium'], 
                        input=st.session_state.input,
                        input_prompts=[p['revised_prompt'] for p in prompts['revised_prompts']] + 
                                      [p['synthesized_prompt'] for p in prompts['synthesized_prompts']]
                    ))
                except LofnError as e:
                    st.error(f"An error occurred during prompt generation for Pair {pair_i + 1}: {str(e)}")
                    if debug:
                        st.exception(e)
                    st.warning("Please try again or enable debug mode for more information.")
        else:
            st.write("Ready to generate Image Prompts")

    if st.session_state['concept_mediums'] is not None and model != "gpt-4":
        if st.button("Generate All"):
            for i, pair in enumerate(st.session_state['concept_mediums']):
                st.write(f"Generating prompts for Pair {i + 1}:")
                st.markdown(f"*Concept:* {pair['concept']}")
                st.markdown(f"*Medium:* {pair['medium']}")
                prompts = generate_prompts(st.session_state.input, pair['concept'], pair['medium'], model=model, debug=debug, max_retries=max_retries, temperature=temperature, aesthetics=aesthetics, image_model=image_model)
                
                st.code(dalle3_gen_prompt.format(
                    concept=pair['concept'], 
                    medium=pair['medium'], 
                    input=st.session_state.input,
                    input_prompts=[p['revised_prompt'] for p in prompts['revised_prompts']] + 
                                  [p['synthesized_prompt'] for p in prompts['synthesized_prompts']]
                ))
    elif st.session_state['concept_mediums'] is None:
        st.write("Waiting to generate concepts")
    else:
        st.write("Generate All is disabled for GPT-4.")

with st.container():
    if manual_input:
        manual_concept = st.text_input("Enter your Concept")
        manual_medium = st.text_input("Enter your Medium")

        if st.button("Generate Image Prompts for Manual Input"):
            st.write("Generating")
            try:
                df_prompts_man = generate_prompts(st.session_state.input, manual_concept, manual_medium, model=model, debug=debug, max_retries=max_retries, temperature=temperature)
                st.write(f"Prompts for Concept: {manual_concept}, Medium: {manual_medium}")
                st.dataframe(df_prompts_man)
                if df_prompts_man is not None:
                    st.code(dalle3_gen_prompt.format(
                        concept=manual_concept, 
                        medium=manual_medium, 
                        input=st.session_state.input,
                        input_prompts = df_prompts_man['Revised Prompts'].tolist() + df_prompts_man['Synthesized Prompts'].tolist()
                    ))
                st.write("Image Prompts Complete")
                st.write("Generation complete!")
            except LofnError as e:
                st.error(f"An error occurred during manual prompt generation: {str(e)}")
                if debug:
                    st.exception(e)
                st.warning("Please try again or enable debug mode for more information.")    