# lofn_ui.py

import streamlit as st
import os
import requests
from datetime import datetime
import json
import ast
from langchain.chains.structured_output.base import create_structured_output_runnable
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic.experimental import ChatAnthropicTools
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema import OutputParserException
import numpy as np
import os
import pandas as pd
from string import Template
import json
import re
import random
import openai
import asyncio
import functools

# Read environment variables
OPENAI_API = os.environ.get('OPENAI_API', '')
ANTHROPIC_API = os.environ.get('ANTHROPIC_API', '')
webhook_url = os.environ.get('WEBHOOK_URL', '')

# Ensure the OpenAI API key is set
openai.api_key = os.environ.get('OPENAI_API', '')


def generate_dalle_images(input, concept, medium, df_prompts, max_retries, temperature, model, debug):
    st.write("Generating DALL-E 3 images...")
    
    # Combine Revised and Synthesized prompts
    all_prompts = pd.concat([df_prompts['Revised Prompts'], df_prompts['Synthesized Prompts']])
    
    for index, prompt in enumerate(all_prompts):
        st.write(f"Generating image for prompt {index + 1}: {prompt}")
        
        image_url = generate_image_dalle3(prompt)
        
        if image_url:
            st.image(image_url, caption=f"Generated image for {concept} in {medium} - Prompt {index + 1}")
            
            # Generate a title for the image
            try:
                title = generate_image_title(input, concept, medium, image_url, max_retries, temperature, model, debug)
                st.write(f"Generated title: {title}")
            except Exception as e:
                st.error(f"Error generating title: {str(e)}")
                title = "Untitled"
            
            # Save the image locally
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_type = "Revised" if index < len(df_prompts) else "Synthesized"
            filename = f"{timestamp}_{concept[0:10]}_{medium[0:10]}_{prompt_type}_{index + 1}.png"
            save_image_locally(image_url, filename)
            
            # Save metadata
            metadata = {
                "timestamp": timestamp,
                "concept": concept,
                "medium": medium,
                "prompt_type": prompt_type,
                "prompt_index": index + 1,
                "prompt": prompt,
                "title": title,
                "image_url": image_url,
                "filename": filename
            }
            save_metadata(metadata)
        else:
            st.write(f"Failed to generate image for prompt {index + 1}.")

    st.write("DALL-E 3 image generation complete.")

def save_metadata(metadata):
    # Ensure the metadata directory exists
    os.makedirs('/metadata', exist_ok=True)
    
    # Create a filename for the metadata
    metadata_filename = f"/metadata/{metadata['timestamp']}_{metadata['concept'][0:10]}_{metadata['medium'][0:10]}_{metadata['prompt_type']}_{metadata['prompt_index']}.json"
    
    # Save the metadata as a JSON file
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    st.write(f"Metadata saved as {metadata_filename}")

def generate_image_dalle3(prompt):
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            quality="hd",
            size="1024x1024"
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        st.write(f"An error occurred while generating the image with DALL-E 3: {str(e)}")
        return None

def save_image_locally(image_url, filename, directory='images'):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            # Ensure the directory exists
            os.makedirs(f'/{directory}', exist_ok=True)
            with open(f'/{directory}/{filename}', 'wb') as f:
                f.write(response.content)
            st.write(f"Image saved as /{directory}/{filename}")
        else:
            st.write(f"Failed to download image: {response.status_code}")
    except Exception as e:
        st.write(f"An error occurred while saving the image: {str(e)}")

@st.cache_data(persist=True)
def generate_image_title(input, concept, medium, image, max_retries, temperature, model, debug=False):
    llm = get_llm(model, temperature, OPENAI_API, ANTHROPIC_API)

    chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("human", image_title_prompt)]))
    
    output = run_chain_with_retries(chain, args_dict={
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": st.session_state.essence_and_facets_output['essence_and_facets']['facets'],
        "image": image
    }, max_retries=max_retries)

    title_output, error = parse_output(output, debug)
    
    if error:
        st.error(f"Error generating image title: {error}")
        return "Untitled"
    
    if title_output and "title" in title_output:
        return title_output["title"]
    else:
        st.error("Failed to generate image title")
        return "Untitled"

def read_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()

def extract_json_from_text(output):
    text_pattern = r"'text':\s*'.*?({.*}).*?'}"
    text_match = re.search(text_pattern, output, re.DOTALL)
    if text_match:
        return text_match.group(1)
    return None

def clean_json_string(json_string):
    json_string = json_string.replace('\\n', '').replace('\\\\', '\\')
    json_string = json_string.strip()
    json_string = re.sub(r"(?<!\\)'", '"', json_string)
    json_string = json_string.replace('\\"', '"').replace("'","").replace("\\","'")
    return json_string

def parse_json_string(json_string, parser):
    try:
        parsed_output = json.loads(json_string)
        return parser.parse(json.dumps(parsed_output))
    except (OutputParserException, json.JSONDecodeError) as e:
        st.write(f"An error occurred while parsing the output trying fixes: {e}")
        json_string = json_string + "}"
        parsed_output = json.loads(json_string)
        return parser.parse(json.dumps(parsed_output))

def parse_output(output, debug=False):
    try:
        if debug:
            st.write("Original output:")
            st.write(output)

        json_string = extract_json_from_text(output)
        if json_string is None:
            return None, "No JSON-like structure found in the output."

        json_string = clean_json_string(json_string)

        if debug:
            st.write("Extracted and cleaned JSON string:")
            st.write(json_string)

        return json.loads(json_string), None

    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {str(e)}"

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
        'complete_all_steps_clicked': False
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

def get_llm(model, temperature, openai_api_key=OPENAI_API, anthropic_api_key=ANTHROPIC_API):
    if model.startswith("claude"):
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=4096, anthropic_api_key=anthropic_api_key)
    else:
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=4096, openai_api_key=openai_api_key)

def run_chain_with_retries(_lang_chain, max_retries, args_dict=None, is_correction=False):
    output = None
    retry_count = 0
    while retry_count < max_retries:
        try:
            if is_correction:
                correction_prompt = """
                The previous response was not in the correct JSON format or was incomplete.
                Please refer to the instructions provided earlier and respond with only the complete JSON output.
                Ensure that all required fields are included and properly formatted according to the instructions.
                """
                output = _lang_chain.invoke({"correction_prompt": correction_prompt})
            else:
                output = _lang_chain.invoke(args_dict)
            break
        except Exception as e:
            st.write(f"An error occurred, retrying: {e}")
            retry_count += 1
    if retry_count >= max_retries:
        st.write("Max retries reached. Exiting.")
    return str(output)

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
    for attempt in range(max_retries):
        output = run_chain_with_retries(chain, max_retries=1, args_dict=args_dict, is_correction=(attempt > 0))
        parsed_output, error = parse_output(output)
        
        if parsed_output is not None:
            return output  # Return the raw output
        
        if attempt == max_retries - 1:
            st.error(f"Failed to get valid JSON response after {max_retries} attempts: {error}")
            return None
        
        st.write(f"Attempt {attempt + 1} failed: {error}. Retrying...")
    return None

def process_essence_and_facets(chains, input, max_retries, debug=False):
    raw_output = run_llm_chain(chains, 'essence_and_facets', {"input": input}, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is not None:
        st.session_state.essence_and_facets_output = output
        st.session_state.creativity_spectrum = output["essence_and_facets"]["creativity_spectrum"]
    else:
        st.error(f"Failed to process essence and facets: {error}")
    return output

def process_concepts(chains, input, essence, facets, creativity_spectrum, max_retries, debug=False):
    raw_output = run_llm_chain(chains, 'concepts', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "creativity_spectrum_wild": creativity_spectrum['wild'],
        "creativity_spectrum_creative": creativity_spectrum['creative'],
        "creativity_spectrum_grounded": creativity_spectrum['grounded'],
    }, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is None:
        st.error(f"Failed to process concepts: {error}")
    return output

def process_artist_and_refined_concepts(chains, input, essence, facets, concepts, max_retries, debug=False):
    raw_output = run_llm_chain(chains, 'artist_and_refined_concepts', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "concepts": [x['concept'] for x in concepts['concepts']]
    }, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is None:
        st.error(f"Failed to process artist and refined concepts: {error}")
    return output

def process_mediums(chains, input, essence, facets, refined_concepts, creativity_spectrum, max_retries, debug=False):
    raw_output = run_llm_chain(chains, 'medium', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "refined_concepts": [x['refined_concept'] for x in refined_concepts['refined_concepts']],
        "creativity_spectrum_wild": creativity_spectrum['wild'],
        "creativity_spectrum_creative": creativity_spectrum['creative'],
        "creativity_spectrum_grounded": creativity_spectrum['grounded'],
    }, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is None:
        st.error(f"Failed to process mediums: {error}")
    return output

def process_refined_mediums(chains, input, essence, facets, mediums, artists, refined_concepts, max_retries, debug=False):
    raw_output = run_llm_chain(chains, 'refine_medium', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "mediums": [x['medium'] for x in mediums['mediums']],
        "artists": artists,
        "refined_concepts": [x['refined_concept'] for x in refined_concepts['refined_concepts']]
    }, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is None:
        st.error(f"Failed to process refined mediums: {error}")
    return output

def process_shuffled_review(chains, input, essence, facets, mediums, artists, refined_concepts, max_retries, debug=False):
    review_artists = np.random.permutation(artists).tolist()
    raw_output = run_llm_chain(chains, 'shuffled_review', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "mediums": [x['medium'] for x in mediums['mediums']],
        "artists": review_artists,
        "refined_concepts": [x['refined_concept'] for x in refined_concepts['refined_concepts']]
    }, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is None:
        st.error(f"Failed to process shuffled review: {error}")
    return output

def process_facets(chains, input, concept, medium, max_retries, debug=False):
    raw_output = run_llm_chain(chains, 'facets', {"input": input, "concept": concept, "medium": medium}, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is None:
        st.error(f"Failed to process facets: {error}")
    return output

def process_artistic_guides(chains, input, concept, medium, facets, max_retries, debug=False):
    raw_output = run_llm_chain(chains, 'aspects_traits', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets']
    }, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is None:
        st.error(f"Failed to process artistic guides: {error}")
    return output

def process_midjourney_prompts(chains, input, concept, medium, facets, artistic_guides, max_retries, debug=False):
    raw_output = run_llm_chain(chains, 'midjourney', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "artistic_guides": [x['artistic_guide'] for x in artistic_guides['artistic_guides']]
    }, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is None:
        st.error(f"Failed to process midjourney prompts: {error}")
    else:
        send_to_discord([prompt['image_gen_prompt'] for prompt in output['image_gen_prompts']], premessage=f'Generated Prompts for {concept} in {medium}:')
    return output

def process_artist_refined_prompts(chains, input, concept, medium, facets, image_gen_prompts, max_retries, debug=False):
    raw_output = run_llm_chain(chains, 'artist_refined', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "image_gen_prompts": [x['image_gen_prompt'] for x in image_gen_prompts['image_gen_prompts']]
    }, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is None:
        st.error(f"Failed to process artist refined prompts: {error}")
    else:
        send_to_discord([prompt['artist_refined_prompt'] for prompt in output['artist_refined_prompts']], premessage=f'Artist-Refined Prompts for {concept} in {medium}:')
    return output

def process_revised_synthesized_prompts(chains, input, concept, medium, facets, artist_refined_prompts, max_retries, debug=False):
    raw_output = run_llm_chain(chains, 'revision_synthesis', {
        "input": input,
        "concept": concept,
        "medium": medium,
        "facets": facets['facets'],
        "artist_refined_prompts": [x['artist_refined_prompt'] for x in artist_refined_prompts['artist_refined_prompts']]
    }, max_retries)
    if raw_output is None:
        return None
    
    output, error = parse_output(raw_output, debug)
    if output is None:
        st.error(f"Failed to process revised synthesized prompts: {error}")
    else:
        send_to_discord([prompt['revised_prompt'] for prompt in output['revised_prompts']], premessage=f'Revised Prompts for {concept} in {medium}:')
        send_to_discord([prompt['synthesized_prompt'] for prompt in output['synthesized_prompts']], premessage=f'Synthesized Prompts for {concept} in {medium}:')
    return output

@st.cache_data(persist=True)
def generate_concept_mediums(input, max_retries, temperature, model="gpt-3.5-turbo-16k", verbose=False, debug=False, aesthetics=aesthetics):
    llm = get_llm(model, temperature, OPENAI_API, ANTHROPIC_API)
    selected_aesthetics = random.sample(aesthetics, 100)

    chains = {
        'essence_and_facets': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", concept_system), ("human", essence_prompt)])),
        'concepts': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", concept_system), ("human", concepts_prompt)])),
        'artist_and_refined_concepts': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", concept_system), ("human", artist_and_critique_prompt)])),
        'medium': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", concept_system), ("human", medium_prompt)])),
        'refine_medium': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", concept_system), ("human", refine_medium_prompt)])),
        'shuffled_review': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", concept_system), ("human", refine_medium_prompt)]))
    }

    essence_and_facets = process_essence_and_facets(chains, input, max_retries, debug)
    concepts = process_concepts(chains, input, essence_and_facets["essence_and_facets"]["essence"], essence_and_facets["essence_and_facets"]["facets"], st.session_state.creativity_spectrum, max_retries, debug)
    artist_and_refined_concepts = process_artist_and_refined_concepts(chains, input, essence_and_facets["essence_and_facets"]["essence"], essence_and_facets["essence_and_facets"]["facets"], concepts, max_retries, debug)
    mediums = process_mediums(chains, input, essence_and_facets["essence_and_facets"]["essence"], essence_and_facets["essence_and_facets"]["facets"], artist_and_refined_concepts, st.session_state.creativity_spectrum, max_retries, debug)
    refined_mediums = process_refined_mediums(chains, input, essence_and_facets["essence_and_facets"]["essence"], essence_and_facets["essence_and_facets"]["facets"], mediums, [x['artist'] for x in artist_and_refined_concepts['artists']], artist_and_refined_concepts, max_retries, debug)
    shuffled_review = process_shuffled_review(chains, input, essence_and_facets["essence_and_facets"]["essence"], essence_and_facets["essence_and_facets"]["facets"], mediums, [x['artist'] for x in artist_and_refined_concepts['artists']], artist_and_refined_concepts, max_retries, debug)

    refined_concepts = [x['refined_concept'] for x in artist_and_refined_concepts['refined_concepts']]
    refined_mediums = [x['refined_medium'] for x in shuffled_review['refined_mediums']]
    concept_mediums = [{'concept': concept, 'medium': medium} for concept, medium in zip(refined_concepts, refined_mediums)]
    send_to_discord(concept_mediums, content_type='concepts')
    return concept_mediums

@st.cache_data(persist=True)
def generate_prompts(input, concept, medium, max_retries, temperature, model="gpt-3.5-turbo-16k", debug=False, aesthetics=aesthetics):
    llm = get_llm(model, temperature, OPENAI_API, ANTHROPIC_API)
    selected_aesthetics = random.sample(aesthetics, 100)

    chains = {
        'facets': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", concept_system), ("human", facets_prompt)])),
        'aspects_traits': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", prompt_system), ("human", aspects_traits_prompt)])),
        'midjourney': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", prompt_system), ("human", midjourney_prompt)])),
        'artist_refined': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", prompt_system), ("human", artist_refined_prompt)])),
        'revision_synthesis': LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("system", prompt_system), ("human", revision_synthesis_prompt)]))
    }

    facets = process_facets(chains, input, concept, medium, max_retries, debug)
    artistic_guides = process_artistic_guides(chains, input, concept, medium, facets, max_retries, debug)
    midjourney_prompts = process_midjourney_prompts(chains, input, concept, medium, facets, artistic_guides, max_retries, debug)
    artist_refined_prompts = process_artist_refined_prompts(chains, input, concept, medium, facets, midjourney_prompts, max_retries, debug)
    revised_synthesized_prompts = process_revised_synthesized_prompts(chains, input, concept, medium, facets, artist_refined_prompts, max_retries, debug)

    df_prompts = pd.DataFrame({
        'Revised Prompts': [prompt['revised_prompt'] for prompt in revised_synthesized_prompts['revised_prompts']],
        'Synthesized Prompts': [prompt['synthesized_prompt'] for prompt in revised_synthesized_prompts['synthesized_prompts']]
    })

    if st.session_state.get('use_dalle3', False):
        generate_dalle_images(input, concept, medium, df_prompts, max_retries, temperature, model, debug)

    return df_prompts

def async_wrap(func):
    @functools.wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run

async_generate_prompts = async_wrap(generate_prompts)

async def generate_all_prompts_async(input, concept_mediums, max_retries, temperature, model, debug):
    tasks = []
    for pair in concept_mediums:
        task = async_generate_prompts(input, pair['concept'], pair['medium'], max_retries, temperature, model, debug)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

def generate_all_prompts(input, concept_mediums, max_retries, temperature, model, debug):
    return asyncio.run(generate_all_prompts_async(input, concept_mediums, max_retries, temperature, model, debug))


st.set_page_config(page_title="Lofn - The AI Artist", page_icon=":art:", layout="wide")
with open("/lofn/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

initialize_session_state()

st.title("LOFN - The AI Artist")

# Sidebar settings
st.sidebar.header('Patron Input Features')
model = st.sidebar.selectbox("Select model", ["claude-3-5-sonnet-20240620", "gpt-4o", "claude-3-opus-20240229", "gpt-4-turbo", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "gpt-3.5-turbo", "gpt-4"])
st.session_state['use_dalle3'] = st.sidebar.checkbox("Use DALL-E 3 (increased cost)", value=False)
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

# Main UI
with st.container():
    st.session_state.input = st.text_area("Describe your idea", "I want to capture the essence of a mysterious and powerful witch's familiar.")

    if st.button("Generate Concepts"):
        st.session_state.button_clicked = True

    if st.session_state.button_clicked:
        st.write("Generating")
        st.session_state['concept_mediums'] = generate_concept_mediums(st.session_state.input, max_retries=max_retries, temperature=temperature, model=model, debug=debug)

with st.container():    
    if st.session_state['concept_mediums'] is not None:
        st.write("Concepts Complete")
        df_concepts = pd.DataFrame(st.session_state['concept_mediums'])
        st.dataframe(df_concepts)
        
        st.session_state['pairs_to_try'] = st.multiselect("Select Pairs to Try", list(range(0, df_concepts.shape[0])), st.session_state['pairs_to_try'])
        
        if st.button("Generate Image Prompts"):
            for pair_i in st.session_state['pairs_to_try']:
                df_prompts = generate_prompts(st.session_state.input, st.session_state['concept_mediums'][pair_i]['concept'], st.session_state['concept_mediums'][pair_i]['medium'], model=model, debug=debug, max_retries=max_retries, temperature=temperature)
                st.write(f"Prompts for Concept: {st.session_state['concept_mediums'][pair_i]['concept']}, Medium: {st.session_state['concept_mediums'][pair_i]['medium']}")
                st.dataframe(df_prompts)
                dalle_prompt = dalle3_gen_prompt if enable_diversity else dalle3_gen_nodiv_prompt
                st.code(dalle_prompt.format(
                    concept=st.session_state['concept_mediums'][pair_i]['concept'], 
                    medium=st.session_state['concept_mediums'][pair_i]['medium'], 
                    input=st.session_state.input,
                    input_prompts = df_prompts['Revised Prompts'].tolist() + df_prompts['Synthesized Prompts'].tolist()
                ))
        else:
            st.write("Ready to generate Image Prompts")

    if st.session_state['concept_mediums'] is not None and model != "gpt-4":
        if st.button("Generate All"):
            for pair_i in range(len(st.session_state['concept_mediums'])):
                df_prompts = generate_prompts(st.session_state.input, st.session_state['concept_mediums'][pair_i]['concept'], st.session_state['concept_mediums'][pair_i]['medium'], model=model, debug=debug, max_retries=max_retries, temperature=temperature)
                st.write(f"Prompts for Concept: {st.session_state['concept_mediums'][pair_i]['concept']}, Medium: {st.session_state['concept_mediums'][pair_i]['medium']}")
                st.dataframe(df_prompts)   
                st.code(dalle3_gen_prompt.format(
                    concept=st.session_state['concept_mediums'][pair_i]['concept'], 
                    medium=st.session_state['concept_mediums'][pair_i]['medium'], 
                    input=st.session_state.input,
                    input_prompts = df_prompts['Revised Prompts'].tolist() + df_prompts['Synthesized Prompts'].tolist()
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