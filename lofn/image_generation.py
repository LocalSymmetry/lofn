# image_generation.py
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
import fastapi_poe as fp
import fal_client
import os
import time
import asyncio
import streamlit as st
from config import Config
from helpers import *
import logging
logger = logging.getLogger(__name__)
from config import Config
import plotly.graph_objects as go
import pandas as pd
from llm_integration import *
from langchain.chains.structured_output.base import create_structured_output_runnable
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic.experimental import ChatAnthropicTools
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema import OutputParserException
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

image_title_schema = {
    "title": str,
    "instagram_caption": str,
    "instagram_hashtags": str,
    "seo_keywords": str
}

prompt_system = read_prompt('/lofn/prompts/prompt_system.txt')

prompt_ending = read_prompt('/lofn/prompts/prompt_ending.txt')

prompt_header_part1 = read_prompt("/lofn/prompts/prompt_header.txt")

prompt_header_part2 = read_prompt("/lofn/prompts/prompt_header_pt2.txt")

prompt_header = prompt_header_part1 + prompt_header_part2

image_title_prompt_middle = read_prompt("/lofn/prompts/image_title_prompt.txt")

image_title_prompt = prompt_header + image_title_prompt_middle + prompt_ending 

def generate_image(model: str, params: dict, OPENAI_API = Config.OPENAI_API, debug = False):
    # if model.startswith("runware:") or model.startswith("civitai:"):
    #     return generate_runware_image(params['prompt'], params)
    if model == "DALL-E 3":
        return [generate_image_dalle3(params, OPENAI_API, debug = debug)]
    elif model == "Google Imagen 3":
          return generate_google_imagen_image(params, debug=debug)
    elif model.startswith("fal-ai/"):
        return generate_fal_image(model, params, debug = debug)
    elif model.startswith("Poe-"):
        return generate_poe_image(model, params, debug = debug)
    elif model == "Ideogram":
        return generate_ideogram_image(params['prompt'], params, debug = debug)
    else:
        st.write(f"Unsupported model: {model}")
        return None

def save_image_locally(image_url, filename, directory='images'):
    max_retries = 3
    retry_delay = 1  # seconds

    # Ensure the directory exists
    os.makedirs(f'/{directory}', exist_ok=True)

    for attempt in range(max_retries):
        try:
            if image_url[0] != '/':
                # It's a URL; download the image
                response = requests.get(image_url, timeout=3)
                response.raise_for_status()  # Raise an exception for HTTP errors
                content = response.content
            else:
                # It's a local file path; read the file content
                with open(image_url, 'rb') as f:
                    content = f.read()

            # Save the content to the desired location
            with open(f'/{directory}/{filename}', 'wb') as f:
                f.write(content)

            st.write(f"Image saved as /{directory}/{filename}")
            return  # Exit the function after successful save
        except Exception as e:
            st.write(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                st.write(f"Failed to save image after {max_retries} attempts")
                st.write(f"Full URL or path attempted: {image_url}")


def generate_google_imagen_image(params, debug=False):
    try:
        vertexai.init(project=Config.GOOGLE_PROJECT_ID, location="us-central1")

        generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

        prompt = params['prompt']
        number_of_images = params.get('num_images', 1)
        image_size = params.get('image_size', '1:1')
        safety_filter_level = params.get('safety_filter_level', 'block_some')
        person_generation = params.get('person_generation', 'allow_all')
        add_watermark = params.get('add_watermark', False)

        images = generation_model.generate_images(
          prompt=prompt,
          number_of_images=number_of_images,
          aspect_ratio=image_size,
          safety_filter_level=safety_filter_level,
          # Other parameters as needed
        )

        # The images returned are PIL Image objects
        image_paths = []
        for i, image in enumerate(images):
            filename = f"google_imagen_{i}_{prompt[:40]}.png"
            image.save(f"/images/{filename}")
            image_paths.append(f"/images/{filename}")
        return image_paths

    except Exception as e:
        st.error(f"An error occurred while generating the image with Google Imagen 3: {str(e)}")
        if debug:
            st.exception(e)
        return None

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
        elif poe_model in ["Poe-Imagen3", "Poe-StableDiffusion3.5-L", "Playground-v3", "Playground-v2.5", "Ideogram", "Ideogram-v2", "FLUX-schnell", "FLUX-pro", "FLUX-dev", "StableDiffusion3", "SD3-Turbo", "StableDiffusionXL", "StableDiffusion3-2B", "SD3-Medium"]:
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
        elif poe_model == "Playground-v2.5" or poe_model == "Playground-v3":
            options['guidance_scale'] = params.get('guidance_scale', 7.5)
            options['negative_prompt'] = params.get('negative_prompt', '')
        elif poe_model == "Ideogram":
            options['style_preset'] = params.get('style_preset', 'default')
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
            async for partial in fp.get_bot_response(messages=messages, bot_name=poe_model, api_key=Config.POE_API):
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

def generate_image_dalle3(params, OPENAI_API = Config.OPENAI_API, debug = False):
    try:
        openai.api_key = OPENAI_API
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
def generate_dalle_images(input, concept, medium, df_prompts, max_retries, temperature, model, debug, image_model, style_axes, creativity_spectrum, OPENAI_API = Config.OPENAI_API, reasoning_level = 'medium'):
    if image_model == "None":
        st.write("Image generation skipped.")
        return
    st.write(f"Generating images using {image_model}...")
    
    all_prompts = pd.concat([df_prompts['Revised Prompts'], df_prompts['Synthesized Prompts']])

    for index, prompt in enumerate(all_prompts):
        if debug:
            st.write(f"Generating image for prompt {index + 1}: {prompt}")
        
        params = get_model_params(image_model)
        params['prompt'] = prompt  # Override the prompt with the current one
        
        try:
            results = generate_image(image_model, params, OPENAI_API, debug)
            
            if results:
                for i, result in enumerate(results):
                    try:
                        # Display the image
                        st.image(result, caption=f"Generated image {i+1} for {concept} in {medium}")
                        st.code(prompt, language='')
                        
                        # Generate a title for the image
                        try:
                            title_data_json = generate_image_title(
                                input, concept, medium, result, max_retries,
                                temperature, model, debug, reasoning_level
                            )
                            title_data = json.loads(title_data_json)
                            st.code(json.dumps(title_data, indent=2), language='json')

                            # Generate Runway video prompt
                            # runway_prompt_json  = generate_runway_prompt(input, concept, medium, result, prompt, style_axes, creativity_spectrum, max_retries, temperature, model, debug)
                            # st.subheader("Runway Gen-3 Alpha Video Prompt")
                            # st.code(runway_prompt_json['runway_prompt'], language="")
                        except Exception as e:
                            st.error(f"Error generating title and Instagram post: {str(e)}")
                            title_data = {"title": "Untitled", "instagram_caption": "", "instagram_hashtags": "", "seo_keywords": ""}
                    
                        # Save the image locally
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            prompt_type = "Revised" if index < len(df_prompts) else "Synthesized"
                            filename = f"{timestamp}_{model.replace('/','_')}_{concept[0:10]}_{medium[0:10]}_{prompt_type}_{index + 1}_{i + 1}.png"
                            save_image_locally(result, filename)
                        except Exception as e:
                            st.error(f"Error saving image locally: {str(e)}")

                        # Generate video using Pika through Poe if enabled
                        # video_url = None
                        # if enable_pika_video:
                        #     try:
                        #         pika_params = {
                        #             'motion': pika_motion,
                        #             'guidance_scale': pika_guidance_scale,
                        #             'frame_rate': pika_frame_rate,
                        #             'aspect_ratio': pika_aspect_ratio,
                        #             'negative_prompt': pika_negative_prompt,
                        #             'camera_zoom': pika_camera_zoom,
                        #             'camera_pan': pika_camera_pan,
                        #             'camera_tilt': pika_camera_tilt,
                        #             'camera_rotate': pika_camera_rotate
                        #         }
                        #         video_url = generate_poe_video(prompt, result, pika_params, debug)
                        #         if video_url:
                        #             video_filename = f"{timestamp}_{model}_{concept[0:10]}_{medium[0:10]}_{prompt_type}_{index + 1}_{i + 1}.mp4"
                        #             save_video_locally(video_url, video_filename)
                        #     except Exception as e:
                        #         st.error(f"Error generating or saving video: {str(e)}")
                        #         video_url = None

                        # Save metadata
                        try:
                            metadata = {
                                "timestamp": timestamp,
                                "style_axes": style_axes,
                                "creativity_spectrum": creativity_spectrum,
                                "concept": concept,
                                "medium": medium,
                                "prompt_type": prompt_type,
                                "prompt_index": index + 1,
                                "image_index": i + 1,
                                "prompt": prompt,
                                "title": title_data['title'],
                                "instagram_post": title_data['instagram_caption'],
                                "hashtags": title_data['instagram_hashtags'],
                                "seo_keywords": title_data['seo_keywords'],
                                "image_url": result,
                                "image_model": image_model,
                                "model": model,
                                "image_filename": filename,
                                "user_input": input
                                #"video_filename": video_filename if video_url else None
                            }
                            # if enable_pika_video:
                            #     metadata["pika_params"] = pika_params
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

def generate_image_title(input, concept, medium, image, max_retries, temperature, model, debug=False, reasoning_level = "medium"):
    # o1 takes forever.
    if model[0] in ['o', 'P']:
        inner_model = 'gpt-4o-mini'
    else:
        inner_model = model

    llm = get_llm(inner_model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, reasoning_level = reasoning_level)

    chain = (
        ChatPromptTemplate.from_messages([("human", image_title_prompt)])
        | llm
    )
    
    output = run_chain_with_retries(
        chain, 
        args_dict={
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": st.session_state.essence_and_facets_output['essence_and_facets']['facets'],
        "image": image
        },
        max_retries=max_retries, 
        debug=debug,
        expected_schema = image_title_schema)

    if debug:
        st.write("Output from run_chain_with_retries:")
        st.write(output)

    if output is None:
        return json.dumps({"title": "Untitled", "instagram_post": {"caption": "", "hashtags": []}, "seo_keywords": []})

    try:
        try:
            # If output is a string, try to parse it as JSON and then convert back to string
            parsed_output = json.loads(output)
            return json.dumps(parsed_output)
        except:
            return json.dumps(output)
    except json.JSONDecodeError:
        st.error("Failed to parse JSON output from title generation")
        return json.dumps({"title": "Untitled", "instagram_post": {"caption": "", "hashtags": []}, "seo_keywords": []})

def generate_ideogram_image(prompt, params, api_key=Config.IDEOGRAM_API_KEY, debug = False):
    url = "https://api.ideogram.ai/generate"
    
    payload = {
        "image_request": {
            "model": params["model"],
            "magic_prompt_option": params["magic_prompt_option"],
            "prompt": prompt,
            "aspect_ratio": params["image_size"],
            "style_type": params["style_type"],
            "seed": params["seed"]
        }
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "api-key": api_key
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        if 'data' in result and len(result['data']) > 0:
            return [image['url'] for image in result['data'] if image['is_image_safe']]
        else:
            st.error("No safe images were generated.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while generating the image with Ideogram: {str(e)}")
        return None

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
        if ("flux" in model_name and "ultra" not in model_name) or "omni" in model_name or "sdxl" in model_name or "stable" in model_name:
            arguments.update({
                "num_inference_steps": int(params.get('num_inference_steps', 28)),
                "guidance_scale": float(params.get('guidance_scale', 3.5)),
                "enable_safety_checker": params.get('enable_safety_checker', True),
            })

        if "ultra" in model_name:
            arguments.update({
                "aspect_ratio": params.get('aspect_ratio', "1:1"),
                "safety_tolerance": int(params.get('safety_tolerance', 6)),
                "raw_mode": params.get('raw_mode', False),
                "enable_safety_checker": params.get('enable_safety_checker', True),
            })


        if ("flux" in model_name and "ultra" not in model_name) or "Stable" in model_name or "omni" in model_name or "recraft" in model_name or "SD3" in model_name:
            arguments.update({"image_size": str(params.get('image_size', 'portrait_16_9'))})
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

        if "omni" or "ultra" in model_name:
            arguments["output_format"] = "png"

        if "recraft" in model_name:
            arguments["style"] = params.get('style', 'any')
            arguments["enable_safety_checker"] = params.get('enable_safety_checker', True)

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
            async for partial in fp.get_bot_response(messages=messages, bot_name=poe_model, api_key=Config.POE_API):
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

def get_model_params(model: str):
    base_params = {
        "num_images": st.session_state[f"{model}_num_images"],
    }

    image_size = st.session_state[f"{model}_image_size"]
    if model == "DALL-E 3" or ('flux' in model) or ('omnigen' in model) or ('recraft' in model) or ('Flux' in model) or ('FLUX' in model) or ('Ideogram' in model) or ('Playground' in model):
        base_params["size"] = image_size
        base_params["image_size"] = image_size
    else:
        try:
            # Convert size string to width and height
            width, height = map(int, image_size.split('x'))
            base_params["width"] = width
            base_params["height"] = height
        except:
            base_params["width"] = 1024
            base_params["height"] = 1024           

    model_specific_params = {
        "DALL-E 3": {
            "model": "dall-e-3",
            "quality": st.session_state.get(f"{model}_quality", "hd"),
            "style": st.session_state.get(f"{model}_style", "vivid")
        },
        "Google Imagen 3": {
            "image_size": st.session_state.get(f"{model}_image_size", "1:1"),
            "safety_filter_level": st.session_state.get(f"{model}_safety_filter_level", "block_some"),
            "person_generation": st.session_state.get(f"{model}_person_generation", "allow_all"),
            "add_watermark": st.session_state.get(f"{model}_add_watermark", False),
            "prompt": st.session_state.get(f"{model}_prompt"),
        },
        "fal-ai/flux/schnell": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 12),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "fal-ai/flux-dev": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 28),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "fal-ai/flux-pro": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 50),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "fal-ai/flux-pro/v1.1": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 50),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "fal-ai/flux-pro/v1.1-ultra": {
            "aspect_ratio": st.session_state.get(f"{model}_image_size", "1:1"),
            "safety_tolerance": st.session_state.get(f"{model}_safety_tolerance", 1),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True),
            "raw_mode": st.session_state.get(f"{model}_raw_mode", True)
        },
        "fal-ai/omnigen-v1": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 50),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "fal-ai/recraft-v3": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 50),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True),
            "style": st.session_state.get(f"{model}_recraft_style")
        },
        "fal-ai/stable-diffusion-v35-large": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 50),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "fal-ai/stable-diffusion-v35-medium": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 50),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "fal-ai/flux-realism": {
            "num_inference_steps": st.session_state.get(f"{model}_inference_steps", 50),
            "guidance_scale": st.session_state.get(f"{model}_guidance_scale", 3.5),
            "enable_safety_checker": st.session_state.get(f"{model}_enable_safety_checker", True)
        },
        "Ideogram": {
            "model": st.session_state.get(f"{model}_model", "V_2"),
            "magic_prompt_option": st.session_state.get(f"{model}_magic_prompt_option", "AUTO"),
            "style_type": st.session_state.get(f"{model}_style_type", "GENERAL"),
            "seed": st.session_state.get(f"{model}_seed", 0)
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

    if model.startswith("runware:") or model.startswith("civitai:"):
        params = base_params.copy()
        params.update({
            "model": model,
            "width": width,
            "height": height,
            "steps": st.session_state.get(f"{model}_inference_steps", 20),
            "CFGScale": st.session_state.get(f"{model}_guidance_scale", 7.5),
            "negative_prompt": st.session_state.get(f"{model}_negative_prompt", ""),
            "use_cache": True,
            "save_to_cache": True,
            "clip_skip": st.session_state.get(f"{model}_clip_skip", 0),
            "check_nsfw": st.session_state.get(f"{model}_check_nsfw", False),
            "use_prompt_weighting": st.session_state.get(f"{model}_use_prompt_weighting", False),
            "scheduler_id": st.session_state.get(f"{model}_scheduler_id", 1),
            "controlNet": get_controlnet_params(model) if st.session_state.get(f"{model}_use_controlnet", False) else None,
            "lora": get_lora_params(model) if st.session_state.get(f"{model}_use_lora", False) else None
        })
    return params

def render_image_controls(model: str):
    if model.startswith("runware:") or model.startswith("civitai:"):
        st.selectbox("Image Size", ["512x512", "512x768", "512x1024", "768x512", "1024x512", "704x512", "896x512", "512x896", "512x704", "1024x1024", "1344x768", "768x1344", "640x960", "960x640"], key=f"{model}_image_size")
        st.number_input("Inference Steps", min_value=1, max_value=100, value=20, key=f"{model}_inference_steps")
        st.number_input("Guidance Scale", min_value=0.0, max_value=30.0, value=7.5, step=0.1, key=f"{model}_guidance_scale")
        st.text_area("Negative Prompt", key=f"{model}_negative_prompt")
        st.number_input("CLIP Skip", min_value=0, max_value=2, value=0, key=f"{model}_clip_skip")
        st.checkbox("Check NSFW", value=False, key=f"{model}_check_nsfw")
        st.checkbox("Use Prompt Weighting", value=False, key=f"{model}_use_prompt_weighting")
        st.selectbox("Scheduler", ["Default", "DDIMScheduler", "DDPMScheduler", "PNDMScheduler", "LMSDiscreteScheduler", "EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler", "DPMSolverMultistepScheduler"], key=f"{model}_scheduler_id")
        
        st.checkbox("Use ControlNet", key=f"{model}_use_controlnet")
        if st.session_state.get(f"{model}_use_controlnet", False):
            st.selectbox("ControlNet Model", ["canny", "depth", "mlsd", "normalbae", "openpose", "tile", "seg", "lineart", "lineart_anime", "shuffle", "scribble", "softedge"], key=f"{model}_controlnet_model")
            st.file_uploader("ControlNet Guide Image", type=["png", "jpg", "jpeg"], key=f"{model}_controlnet_image")
            st.slider("ControlNet Weight", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"{model}_controlnet_weight")
            st.number_input("ControlNet Start Step", min_value=0, max_value=100, value=0, key=f"{model}_controlnet_start_step")
            st.number_input("ControlNet End Step", min_value=0, max_value=100, value=20, key=f"{model}_controlnet_end_step")
            st.selectbox("ControlNet Mode", ["balanced", "prompt", "controlnet"], key=f"{model}_controlnet_mode")

        st.checkbox("Use LoRA", key=f"{model}_use_lora")
        if st.session_state.get(f"{model}_use_lora", False):
            st.text_area("LoRA Models", placeholder="Enter one LoRA model per line in format: model_id:weight", key=f"{model}_lora_models")
    elif model == "DALL-E 3" or model == "Poe-DALL-E-3":
        st.selectbox("Image Size", ["1024x1792", "1792x1024", "1024x1024"], key=f"{model}_image_size")
        st.selectbox("Quality", ["hd", "standard"], key=f"{model}_quality")
        st.selectbox("Style", ["vivid", "natural"], key=f"{model}_style")
    elif model == "Ideogram":
        st.selectbox("Model", ["V_2", "V_2_TURBO", "V_1", "V_1_TURBO"], key=f"{model}_model")
        st.selectbox("Magic Prompt Option", ["OFF", "AUTO", "ON"], key=f"{model}_magic_prompt_option")
        st.selectbox("Aspect Ratio", [
                "ASPECT_9_16", "ASPECT_4_3", "ASPECT_3_4",  "ASPECT_1_1", "ASPECT_10_16", "ASPECT_16_10", "ASPECT_16_9",
                "ASPECT_3_2", "ASPECT_2_3","ASPECT_1_3", "ASPECT_3_1"
            ], key=f"{model}_image_size")
        
        st.selectbox("Style Type", ["GENERAL", "REALISTIC", "DESIGN", "RENDER_3D", "ANIME"], key=f"{model}_style_type")
        st.number_input("Seed", min_value=0, max_value=2147483647, value=0, key=f"{model}_seed", help="0 for random seed")
    elif model == "Google Imagen 3":
          st.selectbox("Aspect Ratio", ["1:1", "4:3", "3:4", "16:9", "9:16"], key=f"{model}_image_size")
          st.selectbox("Safety Filter Level", ["block_most", "block_some", "block_few"], key=f"{model}_safety_filter_level")
          st.selectbox("Person Generation", ["allow_all", "allow_adult", "dont_allow"], key=f"{model}_person_generation")
          st.checkbox("Add Watermark", value=False, key=f"{model}_add_watermark")
    elif model in ["fal-ai/flux/schnell", "Poe-FLUX-schnell"]:
        st.selectbox("Image Size", ["portrait_16_9", "square_hd", "square", "portrait_4_3", "landscape_4_3", "landscape_16_9"], key=f"{model}_image_size")
        st.number_input("Inference Steps", min_value=1, max_value=12, value=12, key=f"{model}_inference_steps")
        st.checkbox("Enable Safety Checker", value=True, key=f"{model}_enable_safety_checker")
    elif model in ["fal-ai/recraft-v3"]:
        st.selectbox("Image Size", ["portrait_4_3", "portrait_16_9",  "square_hd", "square", "landscape_4_3", "landscape_16_9"], key=f"{model}_image_size")
        st.selectbox("Generation Style", ["any", "realistic_image", "digital_illustration", "vector_illustration", "realistic_image/b_and_w", "realistic_image/hard_flash", "realistic_image/hdr", "realistic_image/natural_light", "realistic_image/studio_portrait", "realistic_image/enterprise", "realistic_image/motion_blur", "digital_illustration/pixel_art", "digital_illustration/hand_drawn", "digital_illustration/grain", "digital_illustration/infantile_sketch", "digital_illustration/2d_art_poster", "digital_illustration/handmade_3d", "digital_illustration/hand_drawn_outline", "digital_illustration/engraving_color", "digital_illustration/2d_art_poster_2", "vector_illustration/engraving", "vector_illustration/line_art", "vector_illustration/line_circuit", "vector_illustration/linocut"], key=f"{model}_recraft_style")
        st.checkbox("Enable Safety Checker", value=True, key=f"{model}_enable_safety_checker")
    elif model in ["fal-ai/flux-dev", "fal-ai/flux-realism","fal-ai/stable-diffusion-v35-medium", "fal-ai/omnigen-v1", "fal-ai/stable-diffusion-v35-large", "fal-ai/flux-pro", "fal-ai/flux-pro/v1.1", "Poe-FLUX-pro-1.1", "Poe-FLUX-pro", "Poe-Ideogram-v2", "Poe-Ideogram", "Poe-Imagen3", "Poe-StableDiffusion3.5-L", "Poe-StableDiffusion3", "Poe-SD3-Turbo", "fal-ai/stable-diffusion-v3", "Poe-FLUX-dev"]:
        st.selectbox("Image Size", ["portrait_4_3", "portrait_16_9",  "square_hd", "square", "landscape_4_3", "landscape_16_9"], key=f"{model}_image_size")
        st.number_input("Inference Steps", min_value=1, max_value=50, value=50, key=f"{model}_inference_steps")
        st.number_input("Guidance Scale", min_value=0.0, max_value=20.0, value=7.0, step=0.1, key=f"{model}_guidance_scale")
        st.checkbox("Enable Safety Checker", value=True, key=f"{model}_enable_safety_checker")
    elif model in ["fal-ai/flux-pro/v1.1-ultra"]:
        st.selectbox("Aspect Ratio", ["3:4", "4:3", "1:1", "9:16", "16:9", "9:21", "21:9"], key=f"{model}_image_size")
        st.checkbox("Enable Raw Mode", value=False, key=f"{model}_raw_mode")    
        st.checkbox("Enable Safety Checker", value=True, key=f"{model}_enable_safety_checker")    
        st.number_input("Safety Tolerance", min_value=1, max_value=6, value=6, key=f"{model}_safety_tolerance")
    elif model in ["fal-ai/fast-sdxl", "fal-ai/playground-v25", "Poe-StableDiffusionXL", "Poe-StableDiffusion3-2B", "Poe-SD3-Medium"]:
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
        st.selectbox("Image Size", ["1024x1024", "512x512"], key=f"{model}_image_size")
    else:
        st.selectbox("Image Size", ["1024x1024", "512x512", "768x768", "512x768", "768x512"], key=f"{model}_image_size")
    
    st.number_input("Number of Images", min_value=1, max_value=10, value=1, key=f"{model}_num_images")

def update_image_controls():
    # Clear previous image controls
    for key in list(st.session_state.keys()):
        if key.endswith(("_prompt", "_image_size", "_num_images", "_quality", "_style",
                         "_inference_steps", "_guidance_scale", "_enable_safety_checker",
                         "_negative_prompt", "_expand_prompt", "_format", "_guidance_rescale")):
            del st.session_state[key]

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

def parse_lora_input(lora_input: str):
    lora_list = []
    for line in lora_input.split('\n'):
        if line.strip():
            model_id, weight = line.strip().split(':')
            lora_list.append({"model": model_id.strip(), "weight": float(weight.strip())})
    return lora_list

def get_controlnet_params(model: str):
    return {
        "model": st.session_state.get(f"{model}_controlnet_model", ""),
        "guideImage": st.session_state.get(f"{model}_controlnet_image", ""),
        "weight": st.session_state.get(f"{model}_controlnet_weight", 1.0)
    }

def save_metadata(metadata):
    # Ensure the metadata directory exists
    os.makedirs('/metadata', exist_ok=True)
    
    # Create a filename for the metadata
    metadata_filename = f"/metadata/{metadata['timestamp']}_{metadata['model'][0:10].replace('/','_')}_{metadata['concept'][0:10]}_{metadata['medium'][0:10]}_{metadata['prompt_type']}_{metadata['prompt_index']}.json"
    
    def json_serializable(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    # Save the metadata as a JSON file
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2, default=json_serializable)
    
    st.write(f"Metadata saved as {metadata_filename}")