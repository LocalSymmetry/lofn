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
import plotly.graph_objects as go
import math
import json_repair

# Read environment variables
OPENAI_API = os.environ.get('OPENAI_API', '')
ANTHROPIC_API = os.environ.get('ANTHROPIC_API', '')
webhook_url = os.environ.get('WEBHOOK_URL', '')

# Ensure the OpenAI API key is set
openai.api_key = os.environ.get('OPENAI_API', '')

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

def generate_dalle_images(input, concept, medium, df_prompts, max_retries, temperature, model, debug):
    st.write("Generating DALL-E 3 images...")
    
    # Combine Revised and Synthesized prompts
    all_prompts = pd.concat([df_prompts['Revised Prompts'], df_prompts['Synthesized Prompts']])

    for index, prompt in enumerate(all_prompts):
        if debug:
            st.write(f"Generating image for prompt {index + 1}: {prompt}")
        
        image_url = generate_image_dalle3(prompt)
        
        if image_url:
            st.image(image_url, caption=f"Generated image for {concept} in {medium} - Prompt {index + 1}")
            
            # Generate a title for the image
            try:
                title_data_json = generate_image_title(input, concept, medium, image_url, max_retries, temperature, model, debug)
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_type = "Revised" if index < len(df_prompts) else "Synthesized"
            filename = f"{timestamp}_{model}_{concept[0:10]}_{medium[0:10]}_{prompt_type}_{index + 1}.png"
            save_image_locally(image_url, filename)

            # Save metadata
            metadata = {
                "timestamp": timestamp,
                "concept": concept,
                "medium": medium,
                "prompt_type": prompt_type,
                "prompt_index": index + 1,
                "prompt": prompt,
                "title": title_data['title'],
                "instagram_post": title_data['instagram_post'],
                "seo_keywords": title_data['seo_keywords'],
                "image_url": image_url,
                "model": model,
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
    
    # Ensure all data is JSON serializable
    def json_serializable(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    # Save the metadata as a JSON file
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2, default=json_serializable)
    
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

def parse_json_string(json_string, parser):
    try:
        parsed_output = json.loads(json_string)
        return parser.parse(json.dumps(parsed_output))
    except (OutputParserException, json.JSONDecodeError) as e:
        st.write(f"An error occurred while parsing the output trying fixes: {e}")
        json_string = json_string + "}"
        parsed_output = json.loads(json_string)
        return parser.parse(json.dumps(parsed_output))

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
    # Replace single quotes with double quotes, but not those escaped
    json_string = re.sub(r"(?<!\\)'", '"', json_string)
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

        json_string = clean_json_string(json_string)

        if debug:
            st.write("Extracted and cleaned JSON string:")
            st.write(json_string)

        # Attempt to parse the cleaned JSON
        parsed_json = json.loads(json_string)

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
    radar_fig = create_style_axes_chart(style_axes)
    
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
        title_text="Style Axes Values",
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
        st.plotly_chart(radar_fig, use_container_width=True)
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

def run_chain_with_retries(_lang_chain, max_retries, args_dict=None, is_correction=False, debug=False):
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
                if debug:
                    st.write(f"Attempt {retry_count + 1}: Using correction prompt")
                output = _lang_chain.invoke({"correction_prompt": correction_prompt})
            else:
                if debug:
                    st.write(f"Attempt {retry_count + 1}: Using original prompt")
                output = _lang_chain.invoke(args_dict)
            
            if debug:
                st.write(f"Raw output from LLM:\n{output}")
            
            # Parse the output
            parsed_output, error = parse_output(str(output), debug)
            if parsed_output is not None:
                if debug:
                    st.write("Successfully parsed JSON output")
                return parsed_output  # Return the parsed Python object
            else:
                st.write(f"Failed to parse JSON: {error}")
                raise ValueError(f"Invalid JSON: {error}")
        except Exception as e:
            st.write(f"An error occurred in attempt {retry_count + 1}: {e}")
            retry_count += 1
            is_correction = True  # Use correction prompt in next iteration
    if retry_count >= max_retries:
        st.write("Max retries reached. Exiting.")
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


def process_artist_and_refined_concepts(chains, input, essence, facets, style_axes, concepts, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'artist_and_refined_concepts', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "concepts": [x['concept'] for x in concepts['concepts']]
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process artist and refined concepts")
        return None
    return parsed_output

def process_mediums(chains, input, essence, facets, style_axes, refined_concepts, creativity_spectrum, max_retries, debug=False):
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

def process_refined_mediums(chains, input, essence, facets, style_axes, mediums, artists, refined_concepts, max_retries, debug=False):
    parsed_output = run_llm_chain(chains, 'refine_medium', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
        "mediums": [x['medium'] for x in mediums['mediums']],
        "artists": artists,
        "refined_concepts": [x['refined_concept'] for x in refined_concepts['refined_concepts']]
    }, max_retries)
    if parsed_output is None:
        st.error(f"Failed to process refined mediums")
        return None
    return parsed_output

def process_shuffled_review(chains, input, essence, facets, style_axes, mediums, artists, refined_concepts, max_retries, debug=False):
    review_artists = np.random.permutation(artists).tolist()
    parsed_output = run_llm_chain(chains, 'shuffled_review', {
        "input": input,
        "essence": essence,
        "facets": facets,
        "style_axes": style_axes,
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
            "Generating Midjourney Prompts",
            "Refining Prompts",
            "Synthesizing Final Prompts"
        ]
    }
    return steps[process_name][step]


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
        artist_and_refined_concepts = process_artist_and_refined_concepts(chains, input, essence_and_facets["essence_and_facets"]["essence"], essence_and_facets["essence_and_facets"]["facets"],  st.session_state.style_axes, concepts, max_retries, debug)
        if debug:
            st.write("Refined Concepts:")
            for i, concept in enumerate(artist_and_refined_concepts['refined_concepts'], 1):
                st.write(f"{i}. {concept['refined_concept']}")
        
        # Step 4: Generating Mediums
        status.write("Generating Mediums...")
        mediums = process_mediums(chains, input, essence_and_facets["essence_and_facets"]["essence"], essence_and_facets["essence_and_facets"]["facets"],  st.session_state.style_axes, artist_and_refined_concepts, st.session_state.creativity_spectrum, max_retries, debug)
        if debug:
            st.write("Initial Mediums:")
            for i, medium in enumerate(mediums['mediums'], 1):
                st.write(f"{i}. {medium['medium']}")
        
        # Step 5: Refining Mediums
        status.write("Refining Mediums...")
        refined_mediums = process_refined_mediums(chains, input, essence_and_facets["essence_and_facets"]["essence"], essence_and_facets["essence_and_facets"]["facets"],  st.session_state.style_axes, mediums, [x['artist'] for x in artist_and_refined_concepts['artists']], artist_and_refined_concepts, max_retries, debug)
        if debug:
            st.write("Refined Concepts:")
            for i, concept in enumerate(refined_mediums['refined_concepts'], 1):
                st.write(f"{i}. {concept['refined_concept']}")
            st.write("Refined Mediums:")
            for i, medium in enumerate(refined_mediums['refined_mediums'], 1):
                st.write(f"{i}. {medium['refined_medium']}")
        
        # Step 6: Shuffling and Reviewing
        status.write("Shuffling and Reviewing...")
        shuffled_review = process_shuffled_review(chains, input, essence_and_facets["essence_and_facets"]["essence"], essence_and_facets["essence_and_facets"]["facets"],  st.session_state.style_axes, mediums, [x['artist'] for x in artist_and_refined_concepts['artists']], refined_mediums, max_retries, debug)

        status.update(label="Generation Complete!", state="complete")

    refined_concepts = [x['refined_concept'] for x in shuffled_review['refined_concepts']]
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
        
        status.write("Generating Midjourney Prompts...")
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

    if st.session_state.get('use_dalle3', False):
        generate_dalle_images(input, concept, medium, df_prompts, max_retries, temperature, model, debug)

    return revised_synthesized_prompts

def generate_all_prompts(input, concept_mediums, max_retries, temperature, model, debug):
    results = []
    total_pairs = len(concept_mediums)
    
    for i, pair in enumerate(concept_mediums):
        st.write(f"Generating prompts for pair {i+1}/{total_pairs}: {pair['concept']} in {pair['medium']}")
        result = generate_prompts(input, pair['concept'], pair['medium'], max_retries, temperature, model, debug)
        results.append(result)
        st.markdown("---")  # Add a separator between each pair's results
        
    return results

st.set_page_config(page_title="Lofn - The AI Artist", page_icon=":art:", layout="wide")
with open("/lofn/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

initialize_session_state()

st.title("LOFN - The AI Artist")


st.sidebar.header('Style Personalization')
auto_style = st.sidebar.checkbox("Automatic Style", value=True)

if not auto_style:
    st.sidebar.subheader("Adjust Style Axes")
    style_axes = {
        "Abstraction vs. Realism": st.sidebar.slider("Abstraction vs. Realism", 0, 100, 50),
        "Emotional Valence": st.sidebar.slider("Emotional Valence", 0, 100, 50),
        "Color Intensity": st.sidebar.slider("Color Intensity", 0, 100, 50),
        "Symbolic Density": st.sidebar.slider("Symbolic Density", 0, 100, 50),
        "Compositional Complexity": st.sidebar.slider("Compositional Complexity", 0, 100, 50),
        "Textural Richness": st.sidebar.slider("Textural Richness", 0, 100, 50),
        "Symmetry vs. Asymmetry": st.sidebar.slider("Symmetry vs. Asymmetry", 0, 100, 50),
        "Novelty": st.sidebar.slider("Novelty", 0, 100, 50),
        "Figure-Ground Relationship": st.sidebar.slider("Figure-Ground Relationship", 0, 100, 50),
        "Dynamic vs. Static": st.sidebar.slider("Dynamic vs. Static", 0, 100, 50)
    }
else:
    style_axes = None

# Add style_axes to session state
st.session_state['style_axes'] = style_axes

# Sidebar settings
st.sidebar.header('Patron Input Features')
model = st.sidebar.selectbox("Select model", ["gpt-4o", "claude-3-5-sonnet-20240620",  "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-opus-20240229", "gpt-4-turbo", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "gpt-4"])
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

# Modify the main UI part
with st.container():
    st.session_state.input = st.text_area("Describe your idea", "I want to capture the essence of a mysterious and powerful witch's familiar.")

    if st.button("Generate Concepts"):
        st.session_state.button_clicked = True

    if st.session_state.button_clicked:
        st.write("Generating")
        st.session_state['concept_mediums'] = generate_concept_mediums(st.session_state.input, max_retries=max_retries, temperature=temperature, model=model, debug=debug)

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
                prompts = generate_prompts(st.session_state.input, pair['concept'], pair['medium'], model=model, debug=debug, max_retries=max_retries, temperature=temperature)
                
                dalle_prompt = dalle3_gen_prompt if enable_diversity else dalle3_gen_nodiv_prompt
                st.code(dalle_prompt.format(
                    concept=pair['concept'], 
                    medium=pair['medium'], 
                    input=st.session_state.input,
                    input_prompts=[p['revised_prompt'] for p in prompts['revised_prompts']] + 
                                  [p['synthesized_prompt'] for p in prompts['synthesized_prompts']]
                ))
        else:
            st.write("Ready to generate Image Prompts")

    if st.session_state['concept_mediums'] is not None and model != "gpt-4":
        if st.button("Generate All"):
            for i, pair in enumerate(st.session_state['concept_mediums']):
                st.write(f"Generating prompts for Pair {i + 1}:")
                st.markdown(f"*Concept:* {pair['concept']}")
                st.markdown(f"*Medium:* {pair['medium']}")
                prompts = generate_prompts(st.session_state.input, pair['concept'], pair['medium'], model=model, debug=debug, max_retries=max_retries, temperature=temperature)
                
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