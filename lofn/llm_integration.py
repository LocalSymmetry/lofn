# llm_integration.py

import streamlit as st
import openai
import google.generativeai as genai
import asyncio
import fastapi_poe as fp
import requests
import json
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
from typing import Any, Dict, List, Optional
from config import Config
from helpers import (
    read_prompt,
    send_to_discord,
    parse_output,
    display_facets,
    display_creativity_spectrum,
    display_style_axes,
)
import plotly.graph_objects as go
import random
import numpy as np
import pandas as pd
import logging
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import openai  # For the advanced "o1" usage if needed
from o1_integration import *

class LofnError(Exception):
    """Custom exception class for Lofn-specific errors."""
    pass


logger = logging.getLogger(__name__)

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

# Video prompts
video_concept_header_part1 = read_prompt('/lofn/prompts/concept_header.txt')
video_concept_header_part2 = read_prompt('/lofn/prompts/concept_header_pt2.txt')
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


# Read aesthetics from the file
with open('/lofn/prompts/aesthetics.txt', 'r') as file:
    aesthetics = file.read().split(', ')

# Combine prompt parts
prompt_header = prompt_header_part1 + prompt_header_part2
concept_header = concept_header_part1 + concept_header_part2

video_prompt_header = video_prompt_header_part1 + video_prompt_header_part2
video_concept_header = video_concept_header_part1 + video_concept_header_part2

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
    'essence_and_facets': read_prompt('/lofn/prompts/music_essence_prompt.txt'),
    'creation': read_prompt('/lofn/prompts/music_creation_prompt.txt'),
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
    "lyrics_prompt": str
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

class GeminiLLM(LLM):
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str
    generative_model: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=self.api_key)
        self.generative_model = genai.GenerativeModel(self.model_name)

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

        response = self.generative_model.generate_content(
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
    if model.startswith("o1"):
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
            "o1": 100000,
            "o1-2024-12-17": 100000,
            "o1-preview": 32768,
            "o1-mini": 32768,
            "gpt-4o-mini": 4096,
            "gpt-4o": 4096,
            "gpt-4o-2024-08-06": 8192,
            "gpt-4o-2024-11-20": 8192,
            "chatgpt-4o-latest": 8192,
            "gpt-3.5-turbo": 4096,
            "gpt-4-turbo": 4096,
            "gpt-4": 8192,

            # Anthropic models
            "claude-3-5-sonnet-latest": 8096,
            "claude-3-5-sonnet-20241022": 8096,
            "claude-3-5-haiku-20241022": 8096,
            "claude-3-5-sonnet-20240620": 4096,
            "claude-3-opus-20240229": 4096,
            "claude-3-sonnet-20240229": 4096,
            "claude-3-haiku-20240307": 4096,

            # Google models
            "gemini-2.0-flash-exp": 8191,
            "gemini-2.0-flash-thinking-exp": 32768,
            "gemini-1.5-flash": 16384,
            "gemini-1.5-flash-002": 8191,
            "gemini-1.5-pro": 32768,
            "gemini-1.5-pro-002": 32768,
            "gemini-1.5-pro-exp-0801": 32768,
            "gemini-1.0-pro-exp-0827": 32768,
            "gemini-exp-1114": 32768,
            "gemini-exp-1121": 32768,
            "gemini-exp-1206": 1932768,

            # Poe models
            "Poe-o1": 128000,
            "Poe-o1-128k": 128000,
            "Poe-Assistant": 4096,  # FIXME: Verify token limit
            "Poe-Claude-3.5-Sonnet": 4096,
            "Poe-GPT-4o-Mini": 4096,
            "Poe-GPT-4o": 8192,
            "Poe-Llama-3.1-405B-T": 4096,  # FIXME: Verify token limit
            "Poe-Gemini-1.5-Flash": 16384,
            "Poe-Gemini-1.5-Pro": 32768,
            "Poe-Llama-3.2-11B-FW-131k": 128000,
            "Poe-Llama-3.2-90B-FW-131k": 128000,
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
            "Poe-Mistral-7B-v0.3-T": 4096
        }

        # Get the maximum token limit for the selected model
        max_tokens = model_max_tokens.get(model, 4096)

        if model.startswith("claude"):
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_api_key=Config.ANTHROPIC_API
            )
        elif model.startswith("gemini"):
            return GeminiLLM(
                model_name=model,
                api_key=Config.GOOGLE_API,
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
        elif model.startswith("gpt"):
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=Config.OPENAI_API
            )
        elif model.startswith("o1"):
            return ChatOpenAI(
                model=model,
                openai_api_key=Config.OPENAI_API,
                temperature=1
            )
        else:
            raise LofnError(f"{model} not found!")


def run_llm_chain(chains, chain_name, args_dict, max_retries, model=None,
                  debug=None, expected_schema=None):
    chain = chains[chain_name]
    output = run_chain_with_retries(
        chain, max_retries=max_retries, args_dict=args_dict, is_correction=False,
        model=model, debug=debug, expected_schema=expected_schema
    )

    if output is None:
        st.error(f"Failed to get valid JSON response after {max_retries} attempts.")
        return None

    return output

@st.cache_data(persist=True)
def run_chain_with_retries(
    _lang_chain, max_retries, args_dict=None, is_correction=False, model=None,
    debug=False, expected_schema=None
):
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

            # Parse the output, passing the expected schema
            parsed_output, error = parse_output(str(output), expected_schema, debug)
            if parsed_output is not None:
                if debug:
                    st.write("Successfully parsed JSON output")
                return parsed_output
            else:
                st.write(f"Failed to parse or validate JSON: {error}")
                is_correction = True  # Use correction prompt in next iteration
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


# The following functions are for generating image concepts and prompts
def process_essence_and_facets(
    chains, input_text, max_retries, debug=False,
    style_axes=None, creativity_spectrum=None, model=None
):
    expected_schema = essence_and_facets_schema  # Defined earlier
    parsed_output = run_llm_chain(
        chains, 'essence_and_facets', {"input": input_text}, max_retries,
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
    chains, input_text, essence, facets, max_retries, debug=False, style_axes=None, creativity_spectrum=None, model=None
):
    expected_schema = concepts_schema
    parsed_output = run_llm_chain(
        chains,
        'concepts',
        {
            "input": input_text,
            "essence": essence,
            "facets": facets,
            "style_axes": style_axes,
            "creativity_spectrum_transformative": creativity_spectrum['transformative'],
            "creativity_spectrum_inventive": creativity_spectrum['inventive'],
            "creativity_spectrum_literal": creativity_spectrum['literal'],
        },
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
    model=None
):
    expected_schema = artist_refined_concepts_schema
    parsed_output = run_llm_chain(
        chains,
        'artist_and_refined_concepts',
        {
            "input": input_text,
            "essence": essence,
            "facets": facets,
            "style_axes": style_axes,
            "creativity_spectrum_transformative": creativity_spectrum['transformative'],
            "creativity_spectrum_inventive": creativity_spectrum['inventive'],
            "creativity_spectrum_literal": creativity_spectrum['literal'],
            "concepts": [x['concept'] for x in concepts['concepts']]
        },
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
    model=None
):
    expected_schema = mediums_schema
    parsed_output = run_llm_chain(
        chains,
        'medium',
        {
            "input": input_text,
            "essence": essence,
            "facets": facets,
            "style_axes": style_axes,
            "refinedconcepts": [x['refinedconcept'] for x in refined_concepts['refinedconcepts']],
            "creativity_spectrum_transformative": creativity_spectrum['transformative'],
            "creativity_spectrum_inventive": creativity_spectrum['inventive'],
            "creativity_spectrum_literal": creativity_spectrum['literal'],
        },
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
    model=None
):
    expected_schema = refined_mediums_schema
    parsed_output = run_llm_chain(
        chains,
        'refine_medium',
        {
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
        },
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
    model=None
):
    expected_schema = facets_schema
    parsed_output = run_llm_chain(
        chains,
        'facets',
        {
            "input": input_text,
            "concept": concept,
            "medium": medium,
            "style_axes": style_axes
        },
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
    model=None
):
    expected_schema = artistic_guides_schema
    parsed_output = run_llm_chain(
        chains,
        'aspects_traits',
        {
            "input": input_text,
            "concept": concept,
            "medium": medium,
            "facets": facets['facets'],
            "style_axes": style_axes,
        },
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
    model=None
):
    expected_schema = image_gen_schema
    parsed_output = run_llm_chain(
        chains,
        'midjourney',
        {
            "input": input_text,
            "concept": concept,
            "medium": medium,
            "facets": facets['facets'],
            "style_axes": style_axes,
            "artistic_guides": [x['artistic_guide'] for x in artistic_guides['artistic_guides']]
        },
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
    model=None
):
    expected_schema = artist_refined_schema
    parsed_output = run_llm_chain(
        chains,
        'artist_refined',
        {
            "input": input_text,
            "concept": concept,
            "medium": medium,
            "facets": facets['facets'],
            "style_axes": style_axes,
            "image_gen_prompts": [x['image_gen_prompt'] for x in image_gen_prompts['image_gen_prompts']]
        },
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
    model=None
):
    expected_schema = revised_synthesized_schema
    parsed_output = run_llm_chain(
        chains,
        'revision_synthesis',
        {
            "input": input_text,
            "concept": concept,
            "medium": medium,
            "facets": facets['facets'],
            "style_axes": style_axes,
            "artist_refined_prompts": [x['artist_refined_prompt'] for x in artist_refined_prompts['artist_refined_prompts']]
        },
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
    reasoning_level="medium"
):
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)
        selected_aesthetics = random.sample(aesthetics, 100)
        if "Poe" in model:
            selected_aesthetics = selected_aesthetics[:24]

        # Determine max_tokens based on model's capacity
        max_tokens = llm._identifying_params.get('max_tokens', 4096)

        # Select the appropriate prompts based on the medium
        prompts = prompt_configs.get(medium)

        # Build chains using the selected prompts
        if model[0] == "o":
            chains = {
                'essence_and_facets': (
                    ChatPromptTemplate.from_messages([
                        ("human", prompts['essence_and_facets'])
                    ])
                    | llm
                ),
                'concepts': (
                    ChatPromptTemplate.from_messages([
                        ("human", prompts['concepts'])
                    ])
                    | llm
                ),
                'artist_and_refined_concepts': (
                    ChatPromptTemplate.from_messages([
                        ("human", prompts['artist_and_critique'])
                    ])
                    | llm
                ),
                'medium': (
                    ChatPromptTemplate.from_messages([
                        ("human", prompts['medium'])
                    ])
                    | llm
                ),
                'refine_medium': (
                    ChatPromptTemplate.from_messages([
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
                        ("human", prompts['essence_and_facets'])
                    ])
                    | llm
                ),
                'concepts': (
                    ChatPromptTemplate.from_messages([
                        ("system", concept_system),
                        ("human", prompts['concepts'])
                    ])
                    | llm
                ),
                'artist_and_refined_concepts': (
                    ChatPromptTemplate.from_messages([
                        ("system", concept_system),
                        ("human", prompts['artist_and_critique'])
                    ])
                    | llm
                ),
                'medium': (
                    ChatPromptTemplate.from_messages([
                        ("system", concept_system),
                        ("human", prompts['medium'])
                    ])
                    | llm
                ),
                'refine_medium': (
                    ChatPromptTemplate.from_messages([
                        ("system", concept_system),
                        ("human", prompts['refine_medium'])
                    ])
                    | llm
                ),
            }

        with st.status("Generating Concepts and Mediums...", expanded=True) as status:
            # Step 1: Essence and Facets
            status.write("Generating Essence and Facets...")
            essence_and_facets, style_axes, creativity_spectrum = process_essence_and_facets(
                chains, input_text, max_retries, debug, style_axes, creativity_spectrum, model
            )
            if essence_and_facets:
                if st.session_state.creativity_spectrum is None:
                    display_creativity_spectrum(essence_and_facets["essence_and_facets"]["creativity_spectrum"])
                else:
                    display_creativity_spectrum(st.session_state.creativity_spectrum)
                display_facets(essence_and_facets["essence_and_facets"]["facets"])
                display_style_axes(essence_and_facets["essence_and_facets"]["style_axes"])
            
            # Step 2: Concepts
            status.write("Generating Concepts...")
            concepts = process_concepts(
                chains,
                input_text,
                essence_and_facets["essence_and_facets"]["essence"],
                essence_and_facets["essence_and_facets"]["facets"],
                max_retries,
                debug,
                style_axes,
                creativity_spectrum,
                model
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
                debug,
                style_axes,
                creativity_spectrum,
                model
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
                debug,
                style_axes,
                creativity_spectrum,
                model
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
                debug,
                style_axes,
                creativity_spectrum,
                model
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
    reasoning_level="medium"
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
        reasoning_level=reasoning_level
    )

def generate_music_prompts(
    input_text,
    run_time,
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
        run_time (float): Desired length of the song in minutes.
        max_retries (int): Maximum number of retries for API calls.
        temperature (float): Sampling temperature.
        model (str): The model name to use.
        debug (bool): If True, prints additional debug information.

    Returns:
        tuple: (music_prompt str, lyrics_prompt str)
    """
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)



        # Determine max_tokens based on model's capacity
        max_tokens = llm._identifying_params.get('max_tokens', 4096)

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
            args_dict={"input": input_text, "run_time": run_time},
            max_retries=max_retries,
            model=model,
            debug=debug,
            expected_schema = music_facets_schema
        )
        if st.session_state.creativity_spectrum is None:
            display_creativity_spectrum(output_essence["essence_and_facets"]["creativity_spectrum"])
        else:
            display_creativity_spectrum(st.session_state.creativity_spectrum)
        display_facets(output_essence["essence_and_facets"]["facets"])
        display_style_axes(output_essence["essence_and_facets"]["style_axes"])

        gen_chain = (
            ChatPromptTemplate.from_messages([("human", music_creation_prompt)])
            | llm
        )

        # Run the chain with retries
        parsed_output = run_chain_with_retries(
            gen_chain,
            args_dict={
                "input": input_text, 
                "essence":output_essence["essence_and_facets"]["essence"], 
                "facets":output_essence["essence_and_facets"]["facets"], 
                "style_axes":output_essence["essence_and_facets"]["style_axes"],
                "run_time": run_time},
            max_retries=max_retries,
            model=model,
            debug=debug,
            expected_schema = music_gen_schema
        )
        if debug:
            print(parsed_output)
        # Parse the output
        # parsed_output, error = parse_output(str(gen_output), debug)
        # if debug:
        #     print(parsed_output)
        #     print(error)
        if parsed_output is not None:
            music_prompt = parsed_output['music_prompt']
            lyrics_prompt = parsed_output['lyrics_prompt']
            return music_prompt, lyrics_prompt
        else:
            st.error(f"Failed to generate or parse music prompts: {error}")
            return "", ""

    except Exception as e:
        logger.exception("Error generating music prompts: %s", e)
        raise e

def generate_image_prompts(input_text, concept, medium, max_retries, temperature, model="gpt-3.5-turbo-16k", debug=False, style_axes=None, creativity_spectrum=None, reasoning_level="medium"):
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)

        # Build chains using the selected prompts
        if model[0] == "o":
            chains = {
                'facets': (
                    ChatPromptTemplate.from_messages([("human", facets_prompt)])
                    | llm
                ),
                'aspects_traits': (
                    ChatPromptTemplate.from_messages([("human", aspects_traits_prompt)])
                    | llm
                ),
                'midjourney': (
                    ChatPromptTemplate.from_messages([("human", midjourney_prompt)])
                    | llm
                ),
                'artist_refined': (
                    ChatPromptTemplate.from_messages([("human", artist_refined_prompt)])
                    | llm
                ),
                'revision_synthesis': (
                    ChatPromptTemplate.from_messages([("human", revision_synthesis_prompt)])
                    | llm
                )
            }
        else:
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
            # Step 1: Generate Facets
            status.write("Generating Facets...")
            facets = process_facets(chains, input_text, concept, medium, max_retries, debug, style_axes, model)
            if debug:
                st.write("Facets:")
                st.write(facets['facets'])
            
            display_facets(facets['facets'])

            # Step 2: Create Artistic Guides
            status.write("Creating Artistic Guides...")
            artistic_guides = process_artistic_guides(chains, input_text, concept, medium, facets, max_retries, debug, style_axes, model)
            if debug:
                st.write("Artistic Guides:")
                for i, guide in enumerate(artistic_guides['artistic_guides'], 1):
                    st.write(f"{i}. {guide['artistic_guide']}")

            # Step 3: Generate Image Prompts
            status.write("Generating Image Prompts...")
            midjourney_prompts = process_midjourney_prompts(chains, input_text, concept, medium, facets, artistic_guides, max_retries, debug, style_axes, model)
            if debug:
                st.write("Image Generation Prompts:")
                for i, prompt in enumerate(midjourney_prompts['image_gen_prompts'], 1):
                    st.write(f"{i}. {prompt['image_gen_prompt']}")

            # Step 4: Refine Prompts
            status.write("Refining Prompts...")
            artist_refined_prompts = process_artist_refined_prompts(chains, input_text, concept, medium, facets, midjourney_prompts, max_retries, debug, style_axes, model)
            if debug:
                st.write("Artist Refined Prompts:")
                for i, prompt in enumerate(artist_refined_prompts['artist_refined_prompts'], 1):
                    st.write(f"{i}. {prompt['artist_refined_prompt']}")

            # Step 5: Synthesize Final Prompts
            status.write("Synthesizing Final Prompts...")
            revised_synthesized_prompts = process_revised_synthesized_prompts(chains, input_text, concept, medium, facets, artist_refined_prompts, max_retries, debug, style_axes, model)

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
        df_prompts = generate_prompts(input_text, pair['concept'], pair['medium'], max_retries, temperature, model, debug, style_axes, creativity_spectrum)
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
    reasoning_level="medium"
):
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API, debug, reasoning_level)
        selected_aesthetics = random.sample(aesthetics, 100)
        if "Poe" in model:
            selected_aesthetics = selected_aesthetics[:24]

        # Use video prompts
        prompts = prompt_configs.get('video')

        # Build chains using the selected prompts
        if model[0] == "o":
            chains = {
                'facets': (
                    ChatPromptTemplate.from_messages([("human", prompts['facets'])])
                    | llm
                ),
                'aspects_traits': (
                    ChatPromptTemplate.from_messages([("human", prompts['aspects_traits'])])
                    | llm
                ),
                'generation': (
                    ChatPromptTemplate.from_messages([("human", prompts['generation'])])
                    | llm
                ),
                'artist_refined': (
                    ChatPromptTemplate.from_messages([("human", prompts['artist_refined'])])
                    | llm
                ),
                'revision_synthesis': (
                    ChatPromptTemplate.from_messages([("human", prompts['revision_synthesis'])])
                    | llm
                )
            }
        else:
            chains = {
                'facets': (
                    ChatPromptTemplate.from_messages([("system", concept_system), ("human", prompts['facets'])])
                    | llm
                ),
                'aspects_traits': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), ("human", prompts['aspects_traits'])])
                    | llm
                ),
                'generation': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), ("human", prompts['generation'])])
                    | llm
                ),
                'artist_refined': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), ("human", prompts['artist_refined'])])
                    | llm
                ),
                'revision_synthesis': (
                    ChatPromptTemplate.from_messages([("system", prompt_system), ("human", prompts['revision_synthesis'])])
                    | llm
                )
            }

        with st.status(f"Generating Video Prompts for {concept} in {medium}...", expanded=True) as status:
            # Step 1: Generate Facets
            status.write("Generating Facets...")
            facets = process_facets(chains, input_text, concept, medium, max_retries, debug, style_axes, model)
            if debug:
                st.write("Facets:")
                st.write(facets['facets'])

            display_facets(facets['facets'])

            # Step 2: Create Artistic Guides
            status.write("Creating Artistic Guides...")
            artistic_guides = process_artistic_guides(chains, input_text, concept, medium, facets, max_retries, debug, style_axes, model)
            if debug:
                st.write("Artistic Guides:")
                for i, guide in enumerate(artistic_guides['artistic_guides'], 1):
                    st.write(f"{i}. {guide['artistic_guide']}")

            # Step 3: Generate Video Prompts
            status.write("Generating Video Prompts...")
            video_prompts = process_video_prompts(chains, input_text, concept, medium, facets, artistic_guides, max_retries, debug, style_axes, model)
            if debug:
                st.write("Video Prompts:")
                for i, prompt in enumerate(video_prompts['video_prompts'], 1):
                    st.write(f"{i}. {prompt['video_prompt']}")

            # Step 4: Refine Prompts
            status.write("Refining Prompts...")
            artist_refined_prompts = process_video_artist_refined_prompts(
                chains, input_text, concept, medium, facets, video_prompts, max_retries, debug, style_axes, model
            )
            if debug:
                st.write("Filmmaker Refined Prompts:")
                for i, prompt in enumerate(artist_refined_prompts['artist_refined_prompts'], 1):
                    st.write(f"{i}. {prompt['artist_refined_prompt']}")

            # Step 5: Synthesize Final Prompts
            status.write("Synthesizing Final Prompts...")
            revised_synthesized_prompts = process_revised_synthesized_prompts(chains, input_text, concept, medium, facets, artist_refined_prompts, max_retries, debug, style_axes, model)

            status.update(label="Video Prompt Generation Complete!", state="complete")

        # Prepare the DataFrame of prompts
        df_prompts = pd.DataFrame({
            'Revised Prompts': [prompt['revised_prompt'] for prompt in revised_synthesized_prompts['revised_prompts']],
            'Synthesized Prompts': [prompt['synthesized_prompt'] for prompt in revised_synthesized_prompts['synthesized_prompts']]
        })

        return df_prompts

    except Exception as e:
        raise LofnError(f"Error in video prompt generation: {str(e)}")

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
    model=None
):
    expected_schema = video_gen_schema
    parsed_output = run_llm_chain(
        chains,
        'generation',
        {
            "input": input_text,
            "concept": concept,
            "medium": medium,
            "facets": facets['facets'],
            "style_axes": style_axes,
            "artistic_guides": [x['artistic_guide'] for x in artistic_guides['artistic_guides']]
        },
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
    model=None
):
    expected_schema = artist_refined_schema
    parsed_output = run_llm_chain(
        chains,
        'artist_refined',
        {
            "input": input_text,
            "concept": concept,
            "medium": medium,
            "facets": facets['facets'],
            "style_axes": style_axes,
            "video_gen_prompts": [x['video_prompt'] for x in video_prompts['video_prompts']]
        },
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