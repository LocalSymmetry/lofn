# helpers.py

import re
import json
import streamlit as st
import requests
import random
import json_repair
from typing import Union, List
from datetime import datetime
import os
from config import Config
import plotly.graph_objects as go

def read_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()


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

def repair_json(json_string):
    try:
        return json_repair.repair_json(json_string)
    except Exception as e:
        st.write(f"Failed to repair JSON: {e}")
        return json_string

def clean_json_string(json_string):
    if json_string is None:
        return None
    # Remove newlines and extra backslashes
    json_string = json_string.replace('\\n', '').replace('\\\\', '\\')
    # Remove leading/trailing whitespace
    json_string = json_string.strip()
    # Replace remaining escaped quotes and clean up any leftover single quotes or backslashes
    json_string = json_string.replace('\\"', '"').replace("'", "").replace("\\", "'")
    json_pattern = r'({.*})'
    json_match = re.search(json_pattern, json_string, re.DOTALL)
    if json_match:
        return json_match.group(1)
    else:
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
            st.write("Error decoding JSON. Attempting automated repairs first.")
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

def initialize_session_state():
    default_values = {
        'concept_mediums': None,
        'pairs_to_try': [0],
        'button_clicked': False,
        'webhook_url': Config.webhook_url,
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

def filter_models_by_context_length(models, min_total_tokens=25000, min_response_tokens=15000):
    filtered_models = []
    for model in models:
        context_length = model.get('context_length', 0)
        if context_length >= min_total_tokens:
            # Compute max_tokens for the model
            max_tokens = context_length - 10000  # Subtract 10k for input prompts and retries
            if max_tokens >= min_response_tokens:
                filtered_models.append({
                    'id': model['id'],
                    'name': model['name'],
                    'context_length': context_length,
                    'max_tokens': max_tokens,
                    'description': model.get('description', ''),
                    'pricing': model.get('pricing', {}),
                })
    return filtered_models

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

def display_creativity_spectrum(creativity_spectrum):
    labels = ['literal', 'inventive', 'transformative']
    values = [creativity_spectrum['literal'], creativity_spectrum['inventive'], creativity_spectrum['transformative']]
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
    
    if style_axes != None:
        # Create a radial bar chart
        axes = list(style_axes.keys())
        values = list(style_axes.values())
    else:
        axes = ['None']
        values = [100]
    
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

def create_style_axes_chart(style_axes):
    if style_axes != None:
        categories = list(style_axes.keys())
        values = list(style_axes.values())
    else:
        categories = ['None']
        values = [100]

    
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