# llm_integration.py
import streamlit as st
import openai
import google.generativeai as genai
import asyncio
import fastapi_poe as fp
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
from helpers import read_prompt
import plotly.graph_objects as go
import random
from typing import Any, Dict, List, Optional
from config import Config
from helpers import *
import logging
import numpy as np
import pandas as pd

class LofnError(Exception):
    """Custom exception class for Lofn-specific errors."""
    pass

logger = logging.getLogger(__name__)

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

def get_llm(model, temperature, OPENAI_API=None, ANTHROPIC_API=None):
    # Dictionary mapping models to their maximum token limits
    model_max_tokens = {
        # OpenAI models
        "o1-preview": 32768,
        "o1-mini": 32768,
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
        "gemini-1.5-pro": 32768,
        "gemini-1.5-pro-exp-0801": 32768,
        "gemini-1.0-pro-exp-0827": 32768,

        # Poe models
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
        "Poe-Mistral-7B-v0.3-T": 4096,  # FIXME: Verify token limit
    }

    # Get the maximum token limit for the selected model
    max_tokens = model_max_tokens.get(model, 4096)

    if model.startswith("claude"):
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens, anthropic_api_key=Config.ANTHROPIC_API)
    elif model.startswith("gemini"):
        return GeminiLLM(model_name=model, api_key=Config.GOOGLE_API, temperature=temperature, max_tokens=max_tokens)
    elif model.startswith("Poe-"):
        return PoeLLM(model_name=model, api_key=Config.POE_API, temperature=temperature, max_tokens=max_tokens)
    elif model.startswith("gpt"):
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, openai_api_key=Config.OPENAI_API)
    else:
        return ChatOpenAI(model=model, openai_api_key=Config.OPENAI_API, temperature=1)


def run_chain_with_retries(_lang_chain, max_retries, args_dict=None, is_correction=False, model = None, debug=False):
    output = None
    retry_count = 0
    while retry_count < max_retries:
        try:
            output = run_any_chain(_lang_chain, args_dict, is_correction, retry_count, model, debug)
            args_dict['output'] = output
            if debug:
                st.write(f"Raw output from LLM:\n{output}")
            
            # Parse the output - FIXME
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

def run_any_chain(chain, args_dict, is_correction, retry_count, model, debug=False):
    try:
        if is_correction:
            correction_prompt = f"""
            Attempt {retry_count + 1}: The previous response was not in the correct JSON format or was incomplete.
            Please refer to the instructions provided earlier and respond with only the complete JSON output.
            You have confirmation on your plans. Escape any special characters within JSON strings including apostrophes. 
            The last portion of your previous message is included.
            """
            # Preserve original input parameters
            corrected_args = args_dict.copy()
            if "Poe" in model:
                corrected_args['input'] = correction_prompt + "\n\nOriginal query: " + args_dict.get('input', '') + "\n\n End of last output - start from here: " + args_dict.get('output', '')[-400:]
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

def run_llm_chain(chains, chain_name, args_dict, max_retries, model = None, debug = None):
    chain = chains[chain_name]
    output = run_chain_with_retries(chain, max_retries=max_retries, args_dict=args_dict, is_correction=False, model=model, debug=debug)
    
    if output is None:
        st.error(f"Failed to get valid JSON response after {max_retries} attempts.")
        return None
    
    return output

def process_essence_and_facets(chains, input, max_retries, debug=False, style_axes = None, creativity_spectrum = None, model = None):
    parsed_output = run_llm_chain(chains, 'essence_and_facets', {"input": input}, max_retries, model, debug)
    if parsed_output is None:
        return None
    
    if "essence_and_facets" in parsed_output:
        st.session_state.essence_and_facets_output = parsed_output
        if creativity_spectrum == None:
            creativity_spectrum = parsed_output["essence_and_facets"]["creativity_spectrum"]
            st.session_state.creativity_spectrum = creativity_spectrum
        if style_axes == None:
            style_axes = parsed_output["essence_and_facets"]["style_axes"]
            st.session_state.style_axes = style_axes
    else:
        st.error(f"Failed to process essence and facets: Unexpected output structure")
        return None
    return parsed_output, style_axes, creativity_spectrum

def process_concepts(chains, input, essence, facets, max_retries, debug=False, style_axes = None, creativity_spectrum = {'wild':33.3, 'creative':33.3, 'grounded':33.3}, model = None):
    parsed_output = run_llm_chain(chains, 'concepts', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_transformative": creativity_spectrum['transformative'],
        "creativity_spectrum_inventive": creativity_spectrum['inventive'],
        "creativity_spectrum_literal": creativity_spectrum['literal'],
    }, max_retries, model, debug)
    if parsed_output is None:
        st.error(f"Failed to process concepts")
        return None
    return parsed_output


def process_artist_and_refined_concepts(chains, input, essence, facets, concepts, max_retries, debug=False, style_axes = None, creativity_spectrum = {'wild':33.3, 'creative':33.3, 'grounded':33.3}, model = None):
    parsed_output = run_llm_chain(chains, 'artist_and_refined_concepts', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_transformative": creativity_spectrum['transformative'],
        "creativity_spectrum_inventive": creativity_spectrum['inventive'],
        "creativity_spectrum_literal": creativity_spectrum['literal'],
        "concepts": [x['concept'] for x in concepts['concepts']]
    }, max_retries, model, debug)
    if parsed_output is None:
        st.error(f"Failed to process artist and refined concepts")
        return None
    return parsed_output

def process_mediums(chains, input, essence, facets, refined_concepts, max_retries, debug=False, style_axes = None, creativity_spectrum = {'wild':33.3, 'creative':33.3, 'grounded':33.3}, model = None):
    parsed_output = run_llm_chain(chains, 'medium', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "refined_concepts": [x['refined_concept'] for x in refined_concepts['refined_concepts']],
        "creativity_spectrum_transformative": creativity_spectrum['transformative'],
        "creativity_spectrum_inventive": creativity_spectrum['inventive'],
        "creativity_spectrum_literal": creativity_spectrum['literal'],
    }, max_retries, model, debug)
    if parsed_output is None:
        st.error(f"Failed to process mediums")
        return None
    return parsed_output

def process_refined_mediums(chains, input, essence, facets, mediums, artists, refined_concepts, max_retries, debug=False, style_axes = None, creativity_spectrum = {'wild':33.3, 'creative':33.3, 'grounded':33.3}, model = None):
    parsed_output = run_llm_chain(chains, 'refine_medium', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_transformative": creativity_spectrum['transformative'],
        "creativity_spectrum_inventive": creativity_spectrum['inventive'],
        "creativity_spectrum_literal": creativity_spectrum['literal'],
        "mediums": [x['medium'] for x in mediums['mediums']],
        "artists": artists,
        "refined_concepts": [x['refined_concept'] for x in refined_concepts['refined_concepts']]
    }, max_retries, model, debug)
    if parsed_output is None:
        st.error(f"Failed to process refined mediums")
        return None
    return parsed_output

def process_shuffled_review(chains, input, essence, facets, mediums, artists, refined_concepts, max_retries, debug=False, style_axes = None, creativity_spectrum = {'wild':33.3, 'creative':33.3, 'grounded':33.3}, model = None):
    review_artists = np.random.permutation(artists).tolist()
    parsed_output = run_llm_chain(chains, 'shuffled_review', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "creativity_spectrum_transformative": creativity_spectrum['transformative'],
        "creativity_spectrum_inventive": creativity_spectrum['inventive'],
        "creativity_spectrum_literal": creativity_spectrum['literal'],
        "mediums": [x['medium'] for x in mediums['mediums']],
        "artists": review_artists,
        "refined_concepts": [x['refined_concept'] for x in refined_concepts['refined_concepts']]
    }, max_retries, model, debug)
    if parsed_output is None:
        st.error(f"Failed to process shuffled review")
        return None
    return parsed_output

def process_facets(chains, input, concept, medium, max_retries, debug=False, style_axes = None, model = None):
    parsed_output = run_llm_chain(chains, 'facets', {"input": input, "concept": concept, "medium": medium, "style_axes":style_axes}, max_retries, model, debug)
    if parsed_output is None:
        st.error(f"Failed to process facets")
        return None
    return parsed_output

def process_artistic_guides(chains, input, concept, medium, facets, max_retries, debug=False, style_axes = None, model = None):
    parsed_output = run_llm_chain(chains, 'aspects_traits', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
    }, max_retries, model, debug)
    if parsed_output is None:
        st.error(f"Failed to process artistic guides")
        return None
    return parsed_output

def process_midjourney_prompts(chains, input, concept, medium, facets, artistic_guides, max_retries, debug=False, style_axes = None, model = None):
    parsed_output = run_llm_chain(chains, 'midjourney', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "artistic_guides": [x['artistic_guide'] for x in artistic_guides['artistic_guides']]
    }, max_retries, model, debug)
    if parsed_output is None:
        st.error(f"Failed to process midjourney prompts")
        return None
    if parsed_output.get('image_gen_prompts'):
        send_to_discord([prompt['image_gen_prompt'] for prompt in parsed_output['image_gen_prompts']], premessage=f'Generated Prompts for {concept} in {medium}:')
    return parsed_output

def process_artist_refined_prompts(chains, input, concept, medium, facets, image_gen_prompts, max_retries, debug=False, style_axes = None, model = None):
    parsed_output = run_llm_chain(chains, 'artist_refined', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "image_gen_prompts": [x['image_gen_prompt'] for x in image_gen_prompts['image_gen_prompts']]
    }, max_retries, model, debug)
    if parsed_output is None:
        st.error(f"Failed to process artist refined prompts")
        return None
    if parsed_output.get('artist_refined_prompts'):
        send_to_discord([prompt['artist_refined_prompt'] for prompt in parsed_output['artist_refined_prompts']], premessage=f'Artist-Refined Prompts for {concept} in {medium}:')
    return parsed_output

def process_revised_synthesized_prompts(chains, input, concept, medium, facets, artist_refined_prompts, max_retries, debug=False, style_axes = None, model = None):
    parsed_output = run_llm_chain(chains, 'revision_synthesis', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "style_axes": style_axes,
        "artist_refined_prompts": [x['artist_refined_prompt'] for x in artist_refined_prompts['artist_refined_prompts']]
    }, max_retries, model, debug)
    if parsed_output is None:
        st.error(f"Failed to process revised synthesized prompts")
        return None
    if parsed_output.get('revised_prompts'):
        send_to_discord([prompt['revised_prompt'] for prompt in parsed_output['revised_prompts']], premessage=f'Revised Prompts for {concept} in {medium}:')
    if parsed_output.get('synthesized_prompts'):
        send_to_discord([prompt['synthesized_prompt'] for prompt in parsed_output['synthesized_prompts']], premessage=f'Synthesized Prompts for {concept} in {medium}:')
    return parsed_output

@st.cache_data(persist=True)
def generate_concept_mediums(input, max_retries, temperature, model="gpt-3.5-turbo-16k", verbose=False, debug=False, aesthetics=aesthetics, style_axes=None, creativity_spectrum=None):
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API)
        selected_aesthetics = random.sample(aesthetics, 100)
        if "Poe" in model:
            selected_aesthetics = selected_aesthetics[:24]

        # o series doesn't use system prompts
        if model[0] == "o":
            chains = {
                'essence_and_facets': (
                    ChatPromptTemplate.from_messages([
                        ("human", essence_prompt)
                    ])
                    | llm
                ),
                'concepts': (
                    ChatPromptTemplate.from_messages([
                        ("human", concepts_prompt)
                    ])
                    | llm
                ),
                'artist_and_refined_concepts': (
                    ChatPromptTemplate.from_messages([
                        ("human", artist_and_critique_prompt)
                    ])
                    | llm
                ),
                'medium': (
                    ChatPromptTemplate.from_messages([
                        ("human", medium_prompt)
                    ])
                    | llm
                ),
                'refine_medium': (
                    ChatPromptTemplate.from_messages([
                        ("human", refine_medium_prompt)
                    ])
                    | llm
                ),
                'shuffled_review': (
                    ChatPromptTemplate.from_messages([
                        ("human", refine_medium_prompt)
                    ])
                    | llm
                )
            }
        else:
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
            essence_and_facets, style_axes, creativity_spectrum = process_essence_and_facets(chains, input, max_retries, debug, style_axes, creativity_spectrum, model)
            if essence_and_facets:
                if st.session_state.creativity_spectrum == None:
                    display_creativity_spectrum(essence_and_facets["essence_and_facets"]["creativity_spectrum"])
                else:
                    display_creativity_spectrum(st.session_state.creativity_spectrum)
                display_facets(essence_and_facets["essence_and_facets"]["facets"])
                display_style_axes(essence_and_facets["essence_and_facets"]["style_axes"])
            
            # Step 2: Concepts
            status.write("Generating Concepts...")
            concepts = process_concepts(
                chains, 
                input, 
                essence_and_facets["essence_and_facets"]["essence"], 
                essence_and_facets["essence_and_facets"]["facets"], 
                max_retries, 
                debug,
                style_axes, 
                creativity_spectrum,
                model)
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
                concepts,
                max_retries, 
                debug,
                style_axes, 
                creativity_spectrum,
                model)
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
                artist_and_refined_concepts, 
                max_retries, 
                debug,
                style_axes, 
                creativity_spectrum,
                model)
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
                mediums, 
                [x['artist'] for x in artist_and_refined_concepts['artists']], 
                artist_and_refined_concepts, 
                max_retries, 
                debug,
                style_axes, 
                creativity_spectrum,
                model)
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
                mediums, 
                [x['artist'] for x in artist_and_refined_concepts['artists']], 
                refined_mediums, 
                max_retries, 
                debug,
                style_axes, 
                creativity_spectrum,
                model)

            status.update(label="Generation Complete!", state="complete")

        refined_concepts = [x['refined_concept'] for x in shuffled_review['refined_concepts']]
        refined_mediums = [x['refined_medium'] for x in shuffled_review['refined_mediums']]
        concept_mediums = [{'concept': concept, 'medium': medium} for concept, medium in zip(refined_concepts, refined_mediums)]
        
        send_to_discord(concept_mediums, content_type='concepts')
        return concept_mediums, style_axes, creativity_spectrum
    except Exception as e:
        raise LofnError(f"Error in concept generation: {str(e)}") 

@st.cache_data(persist=True)
def generate_prompts(input, concept, medium, max_retries, temperature, model="gpt-3.5-turbo-16k", debug=False, aesthetics=aesthetics, style_axes=None, creativity_spectrum=None):
    try:
        llm = get_llm(model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API)
        selected_aesthetics = random.sample(aesthetics, 100)
        if "Poe" in model:
            selected_aesthetics = selected_aesthetics[:24]

        # o series doesn't use system prompts
        if model[0] == "o":
            chains = {
                'facets': (
                    ChatPromptTemplate.from_messages([("human", facets_prompt)])
                    | llm
                ),
                'aspects_traits': (
                    ChatPromptTemplate.from_messages([ ("human", aspects_traits_prompt)])
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
            status.write("Generating Facets...")
            facets = process_facets(chains, input, concept, medium, max_retries, debug, style_axes, model)

            display_facets(facets['facets'])
            
            status.write("Creating Artistic Guides...")
            artistic_guides = process_artistic_guides(chains, input, concept, medium, facets, max_retries, debug, style_axes, model)
            if debug:
                st.write("Artistic Guides:")
                for i, guide in enumerate(artistic_guides['artistic_guides'], 1):
                    st.write(f"{i}. {guide['artistic_guide']}")
            
            status.write("Generating Image Prompts...")
            midjourney_prompts = process_midjourney_prompts(chains, input, concept, medium, facets, artistic_guides, max_retries, debug, style_axes, model)
            if debug:
                st.write("Image Prompts:")
                for i, prompt in enumerate(midjourney_prompts['image_gen_prompts'], 1):
                    st.write(f"{i}. {prompt['image_gen_prompt']}")
            
            status.write("Refining Prompts...")
            artist_refined_prompts = process_artist_refined_prompts(chains, input, concept, medium, facets, midjourney_prompts, max_retries, debug, style_axes, model)
            if debug:
                st.write("Artist Refined Prompts:")
                for i, prompt in enumerate(artist_refined_prompts['artist_refined_prompts'], 1):
                    st.write(f"{i}. {prompt['artist_refined_prompt']}")
            
            status.write("Synthesizing Final Prompts...")
            revised_synthesized_prompts = process_revised_synthesized_prompts(chains, input, concept, medium, facets, artist_refined_prompts, max_retries, debug, style_axes, model)

            status.update(label="Prompt Generation Complete!", state="complete")

        df_prompts = pd.DataFrame({
            'Revised Prompts': [prompt['revised_prompt'] for prompt in revised_synthesized_prompts['revised_prompts']],
            'Synthesized Prompts': [prompt['synthesized_prompt'] for prompt in revised_synthesized_prompts['synthesized_prompts']]
        })    

        return df_prompts
    except Exception as e:
        raise LofnError(f"Error in prompt generation: {str(e)}")    

def generate_runway_prompt(input, concept, medium, image, prompt, style_axes, creativity_spectrum, max_retries, temperature, model, debug=False):
    # O1 takes too, long, if they want o1, use gpt-4o.
    if model[0] in ['o', 'P']:
        inner_model = 'gpt-4o-mini'
    else: 
        inner_model = model

    llm = get_llm(inner_model, temperature, Config.OPENAI_API, Config.ANTHROPIC_API)

    runway_prompt_template = """
        [Context: Runway Gen-3 Alpha Capabilities and Tips:
        High-Fidelity Video Generation:
        Produces photorealistic videos with exceptional detail
        Captures complex actions and natural motion
        Temporal Consistency:
        Maintains coherency of characters and objects across frames
        Reduces flickering and distortion for seamless viewing
        Advanced Control Features:
        Allows fine-tuning of style, atmosphere, lighting, and camera angles
        Supports detailed text prompts for precise control
        Prompt Structure:
        Use detailed, descriptive prompts for best results
        Include subject, action, setting, camera movement, and style keywords
        Example structure: "[Subject] [Action] in [Setting]. [Camera movement] reveals [Additional details]. [Style keywords]."
        Tips for Effective Prompting:
        Be as specific as possible in descriptions
        Experiment with prompt order and structure
        Include cinematographic elements (e.g., "low-angle tracking shot", "aerial pan")
        Specify visual style, color palette, and lighting
        Incorporate motion and transformation details
        Consider temporal flow (e.g., slow-motion, time-lapse)
        Use style keywords like "cinematic" or "IMAX"
        Character Consistency:
        Use very specific descriptors for characters (e.g., hair color, clothing)
        Maintain consistent prompts for characters across generations
        Consider using Custom Model feature for better consistency
        Practical Tips:
        Start with 5-second generations to conserve credits
        Reuse successful seeds for stylistic consistency
        Plan sequences in advance for better coherence
        Generate multiple options and curate in post-production
        Remember that Gen-3 Alpha is still in development, and its capabilities are continually evolving. Experiment with different approaches and learn from the community for best results.]
        [Goal: Generate a detailed prompt for Runway's Gen-3 Alpha video generation model based on the following input:
        User's Idea: {input}
        Concept: {concept}
        Medium: {medium}
        Style Axes: {style_axes}
        Creativity Spectrum: {creativity_spectrum}
        Image: {image}
        Prompt: {prompt}
        Runway Gen-3 Alpha Video Prompt Generation Guide
        Objective: Create a vivid, narrative-style prompt that leverages Gen-3 Alpha's strengths in fluid motion, scene evolution, and long-form coherence.
        I. Core Elements (40-50 words)
        Subject: Clearly describe the main focus
        Action: Detail what the subject is doing
        Setting: Specify the environment or backdrop
        Mood/Atmosphere: Convey the overall feeling or tone
        Style Keywords: Include terms like "cinematic", "IMAX", or specific genre references
        II. Cinematography (30-40 words)
        Camera Movement: Describe specific movements (e.g., "sweeping dolly shot", "intimate handheld")
        Shot Types: Specify angles and framing (e.g., "extreme close-up", "bird's-eye view")
        Transitions: Detail how scenes evolve or transform (e.g., "seamless morph", "match cut")
        III. Visual Aesthetics (30-40 words)
        Lighting: Describe quality, direction, and changes in lighting
        Color Palette: Specify key colors and their emotional significance
        Texture and Detail: Mention visual textures or intricate details to focus on
        IV. Motion and Transformation (30-40 words)
        Subject Movement: Describe how main elements move or change
        Background Activity: Detail secondary motion or events
        Pacing: Specify speed of action (e.g., "slow-motion", "time-lapse")
        V. Audio-Visual Synergy (20-30 words)
        Sound Cues: Suggest audio elements that could be visually represented
        Rhythm: Describe visual patterns or repetitions implying rhythm
        Emotional Crescendo: Detail how visuals build to enhance emotional impact
        VI. Stylistic Flourishes (20-30 words)
        Special Effects: Describe any CGI or stylized elements
        Artistic Influences: Reference specific directors or visual artists if relevant
        Unique Visuals: Detail any standout or unusual visual elements
        Prompt Structure:
        "[Core Elements]: [Cinematography description]. [Visual Aesthetics details] create a [mood] atmosphere. [Motion and Transformation explanation], while [Audio-Visual Synergy elements] enhance the scene. [Stylistic Flourishes] add depth and intrigue. Style keywords: [list relevant style terms]."
        **Instructions:**
        1. Analyze the given inputs carefully.
        2. Create a video prompt following the guide above, ensuring a cohesive narrative flow.
        3. Aim for a prompt length of 150-200 words.
        4. Use descriptive language that aligns with Gen-3 Alpha's capabilities.
        5. Focus on fluid transitions and evolving scenes.
        6. Experiment with the order of prompt elements for best results.
        7. Output the prompt in the following JSON format, escaping any special characters:
        json
        {{
            "runway_prompt": "Your generated prompt here"
        }}

        """

    # runway_prompt_template = """
    # Generate a detailed prompt for Runway's Gen-3 Alpha video generation model based on the following input:

    # User's Idea: {input}
    # Concept: {concept}
    # Medium: {medium}
    # Style Axes: {style_axes}
    # Creativity Spectrum: {creativity_spectrum}
    # Image: {image}
    # Prompt: {prompt}

    # # Award-Winning Video Prompt Generation Guide for Runway's Gen-3 Alpha

    # Objective: Create visually stunning, emotionally resonant, and conceptually innovative video prompts that push the boundaries of AI-generated cinematography.

    # I. Conceptualization (20-30 words)
    # 1. Core Concept: Describe a central, thought-provoking idea
    # 2. Emotional Resonance: Specify the primary emotion to evoke
    # 3. Visual Metaphor: Introduce a powerful visual metaphor related to the concept

    # II. Visual Language (30-40 words)
    # 1. Cinematographic Style: Choose a distinctive visual approach (e.g., "hyper-real long takes", "fragmented montage", "surreal forced perspective")
    # 2. Color Palette: Describe a unique color scheme and its emotional significance
    # 3. Lighting Dynamics: Detail innovative lighting that evolves throughout the sequence

    # III. Motion and Transformation (30-40 words)
    # 1. Camera Movement: Specify fluid, complex camera motions (e.g., "spiraling ascent transitioning to a cosmic zoom-out")
    # 2. Subject Transformation: Describe metamorphoses of key elements
    # 3. Transitional Fluidity: Explain seamless scene-to-scene transitions

    # IV. Temporal and Spatial Manipulation (20-30 words)
    # 1. Time Distortion: Incorporate non-linear time elements (e.g., "cascading time loops", "parallel timelines merging")
    # 2. Spatial Warping: Describe reality-bending spatial effects
    # 3. Dimensional Shifts: Suggest transitions between different realities or dimensions

    # V. Sensory Layering (20-30 words)
    # 1. Visual Texture: Specify unique textural elements that add depth
    # 2. Auditory Cues: Suggest synesthetic visual representations of sound
    # 3. Haptic Visuals: Describe visuals that evoke tactile sensations

    # VI. Narrative Complexity (20-30 words)
    # 1. Nested Stories: Introduce multi-layered narrative elements
    # 2. Perspective Shifts: Describe changes in point-of-view
    # 3. Symbolic Progression: Explain the evolution of visual symbols throughout the sequence

    # VII. Technical Innovation (20-30 words)
    # 1. AI-Enhanced Realism: Specify hyper-detailed elements beyond human perception
    # 2. Generative Patterns: Describe complex, evolving patterns that emerge and transform
    # 3. Style Fusion: Suggest blending of distinct artistic styles in novel ways

    # VIII. Emotional Journey (20-30 words)
    # 1. Emotional Arc: Map out an emotional progression
    # 2. Contrast and Resonance: Describe emotional contrasts and harmonies
    # 3. Culmination: Specify an emotionally impactful conclusion

    # IX. Cultural and Philosophical Depth (20-30 words)
    # 1. Cultural References: Incorporate diverse cultural elements
    # 2. Philosophical Undertones: Suggest underlying philosophical themes
    # 3. Zeitgeist Reflection: Reference contemporary issues or futuristic concepts

    # X. Runway-Specific Optimization
    # 1. Emphasize fluid motion and seamless transitions
    # 2. Focus on maintaining style consistency across the entire sequence
    # 3. Leverage Gen-3 Alpha's strength in complex scene evolution and long-form coherence

    # Prompt Structure Template Fill in each bracketed section for a single paragraph form prompt:
    # "[Conceptualization sentence]: [Visual Language sentence]. [Motion and Transformation sentence], [leading to Temporal and Spatial Manipulation sentence]. [Sensory Layering mixed with Narrative Complexity sentence]. [Technical Innovation and Emotional Journey sentence]. [Cultural and Philosophical Depth sentence]."

    # Before finalizing:
    # 1. Ensure the prompt is 200-250 words long
    # 2. Review for clarity, creativity, and emotional impact
    # 3. Verify that each section contributes to a cohesive whole
    # 4. Confirm the prompt leverages Gen-3 Alpha's unique capabilities
    # 5. Assess the prompt's potential for generating award-worthy visuals

    # # Instructions

    # Your task is to take an image prompt and transform it into a detailed video prompt suitable for Runway ML's Gen-3 Alpha text-to-video AI. 
    # 1. You will analyze the given image prompt carefully, noting all visual elements, style, mood, and composition.
    # 2. Create a video prompt following the Award-Winning Video Prompt Generation Guide for Runway's Gen-3 Alpha:  

    # Remember: The goal is to create a prompt that not only guides the AI but also inspires it to generate truly revolutionary visual narratives that captivate, challenge, and move the audience in unprecedented ways.
    # Include specific keywords for camera styles, lighting, movement speeds, movement types, and overall aesthetic.
    # Make sure the prompt is descriptive, clear, and aligns with Runway's Gen-3 Alpha capabilities.

    # Output the prompt in the following JSON format, ensuring to escape any special characters within JSON strings including apostrophes and quotation marks:

    # ```json
    # {{
    #     "runway_prompt": "Your generated prompt here"
    # }}
    # ```
    # """

    chain = (
        ChatPromptTemplate.from_messages([("human", runway_prompt_template)])
        | llm
    )
    
    output = run_chain_with_retries(chain, args_dict={
        "input": input,
        "concept": concept,
        "medium": medium,
        "style_axes": style_axes,
        "creativity_spectrum": creativity_spectrum,
        "image": image,
        "prompt": prompt
    }, max_retries=max_retries, model=model, debug=debug)

    if debug:
        st.write("Raw output from Runway prompt generation:")
        st.write(output)

    # Parse the output
    parsed_output, error = parse_output(str(output), debug)
    
    if parsed_output is not None and 'runway_prompt' in parsed_output:
        return parsed_output
    else:
        st.error(f"Failed to generate or parse Runway prompt: {error}")
        return {"runway_prompt": "Failed to generate Runway prompt"}

def generate_all_prompts(input, concept_mediums, max_retries, temperature, model, debug, aesthetics=aesthetics, image_model = "None"):
    results = []
    total_pairs = len(concept_mediums)
    
    for i, pair in enumerate(concept_mediums):
        st.write(f"Generating prompts for pair {i+1}/{total_pairs}: {pair['concept']} in {pair['medium']}")
        result = generate_prompts(input, pair['concept'], pair['medium'], max_retries, temperature, model, debug, aesthetics, image_model)
        results.append(result)
        st.markdown("---")  # Add a separator between each pair's results
        
    return results