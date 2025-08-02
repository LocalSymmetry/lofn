# helpers.py

import re
import json
import streamlit as st
import requests
import random
import json_repair
import csv
from typing import Union, List
from datetime import datetime
import os
from config import Config
import plotly.graph_objects as go

def set_style_axes(auto_style: bool, style_axes=None):
    """Update st.session_state['style_axes'] respecting automatic mode."""
    if auto_style:
        if 'style_axes' not in st.session_state:
            st.session_state['style_axes'] = None
    else:
        st.session_state['style_axes'] = style_axes
    return st.session_state['style_axes']

def read_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()


def sample_artistic_frames(min_count: int = 40, max_count: int = 50) -> str:
    """Return a newline-separated list of randomly selected artistic frames."""
    with open('/lofn/prompts/frames.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    count = random.randint(min_count, max_count)
    sampled = random.sample(rows, count)
    frames = [
        f"{row['Category']}{row['Technique']}{row['Description']}"
        for row in sampled
    ]
    return "\n".join(frames)


def sample_video_frames(min_count: int = 40, max_count: int = 50) -> str:
    """Return a newline-separated list of randomly selected video frames."""
    path = os.path.join(os.path.dirname(__file__), 'prompts', 'video_frames.csv')
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(row for row in csvfile if row.strip())
        rows = list(reader)

    count = random.randint(min_count, max_count)
    sampled = random.sample(rows, count)
    frames = [
        f"{row['Category']}{row['Technique']}{row['Description']}"
        for row in sampled
    ]
    return "\n".join(frames)


def sample_music_genres(min_count: int = 40, max_count: int = 50) -> str:
    """Return a newline-separated list of randomly selected music genres."""
    path = os.path.join(os.path.dirname(__file__), 'prompts', 'genres.txt')
    with open(path, 'r') as file:
        genres = [line.strip() for line in file.readlines() if line.strip()]

    count = random.randint(min_count, max_count)
    sampled = random.sample(genres, count)
    return "\n".join(sampled)


def sample_music_frames(min_count: int = 40, max_count: int = 50) -> str:
    """Return a newline-separated list of randomly selected music frames."""
    path = os.path.join(os.path.dirname(__file__), 'prompts', 'music_frames.csv')
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(row for row in csvfile if row.strip())
        rows = list(reader)

    count = random.randint(min_count, max_count)
    sampled = random.sample(rows, count)
    frames = [
        f"{row['Category']}{row['Technique']}{row['Description']}"
        for row in sampled
    ]
    return "\n".join(frames)


def extract_json_from_text(output: str) -> Union[str, None]:
    """Extract a JSON object from a language model response."""

    if output is None:
        return None

    output = output.strip()

    # Look for a fenced code block first
    block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", output, re.IGNORECASE)
    if block:
        return block.group(1)

    # Next try to pull JSON from a 'text' field that contains a JSON string
    text_field = re.search(r"['\"]text['\"]:\s*['\"](\{[\s\S]*?\})['\"]", output)
    if text_field:
        return text_field.group(1)

    # Fallback to the first JSON-looking snippet
    match = re.search(r"\{[\s\S]*\}", output)
    return match.group(0) if match else None

def repair_json(json_string):
    try:
        return json_repair.repair_json(json_string)
    except Exception as e:
        st.write(f"Failed to repair JSON: {e}")
        return json_string

def normalize_quotes(s: str) -> str:
    """Replace curly quotes with straight quotes for valid JSON."""
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "«": '"',
        "»": '"',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s

def remove_escapes_outside_quotes(s: str) -> str:
    """Remove newline and tab escape sequences that appear outside quoted strings."""
    result = []
    in_quote = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '"' and (i == 0 or s[i-1] != '\\'):
            in_quote = not in_quote
            result.append(ch)
            i += 1
            continue
        if not in_quote and ch == '\\' and i + 1 < len(s) and s[i+1] in 'nrt':
            i += 2
            continue
        if not in_quote and ch in ('\n', '\r', '\t'):
            i += 1
            continue
        result.append(ch)
        i += 1
    return ''.join(result)


def clean_json_string(json_string):
    """Lightly clean a JSON string extracted from a response."""

    if json_string is None:
        return None

    json_string = json_string.strip()

    # Normalize fancy quotes before further cleaning
    json_string = normalize_quotes(json_string)

    # Remove escape sequences and stray newlines outside of quotes
    json_string = remove_escapes_outside_quotes(json_string)

    # Remove Markdown fences if they slipped through
    json_string = re.sub(r"^```(?:json)?", "", json_string, flags=re.IGNORECASE)
    json_string = re.sub(r"```$", "", json_string)

    # Keep only the content between the first opening and last closing brace
    start = json_string.find("{")
    end = json_string.rfind("}")
    if start != -1 and end != -1 and start < end:
        json_string = json_string[start:end + 1]

    return json_string

def parse_output(output, expected_schema, debug=False):
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
        except json.JSONDecodeError:
            st.write("Error decoding JSON. Attempting automated repairs first.")
            repaired_json_string = repair_json(json_string)
            parsed_json = json.loads(repaired_json_string)

        if debug:
            st.write("Successfully parsed JSON:")
            st.write(parsed_json)

        # Validate the parsed JSON against the expected schema
        if not validate_schema(parsed_json, expected_schema):
            if debug:
                st.write("Parsed JSON does not match the expected schema.")
                st.write(f"Expected Schema: {expected_schema}")
                st.write(f"Parsed JSON: {parsed_json}")
            return None, "Parsed JSON does not match the expected schema."

        return parsed_json, None

    except Exception as e:
        error_message = f"JSON parsing error: {str(e)}"
        st.write(error_message)
        return None, error_message

def validate_schema(data, schema):
    """
    Recursively validate JSON data against the expected schema.
    """
    if isinstance(schema, dict):
        if not isinstance(data, dict):
            return False
        for key, subschema in schema.items():
            if key not in data:
                return False
            if not validate_schema(data[key], subschema):
                return False
    elif isinstance(schema, list):
        if not isinstance(data, list):
            return False
        subschema = schema[0] if schema else None
        for item in data:
            if not validate_schema(item, subschema):
                return False
    elif isinstance(schema, tuple):
        if not isinstance(data, schema):
            return False
    else:
        if not isinstance(data, schema):
            return False
    return True

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

def create_creativity_spectrum_chart(creativity_spectrum):
    """Return a Plotly figure visualising the creativity spectrum."""
    labels = ['literal', 'inventive', 'transformative']
    values = [creativity_spectrum['literal'], creativity_spectrum['inventive'], creativity_spectrum['transformative']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker_colors=colors)])
    fig.update_layout(
        title_text="Creativity Spectrum",
        height=300
    )
    return fig

def display_creativity_spectrum(creativity_spectrum):
    """Display only the creativity spectrum chart."""
    fig = create_creativity_spectrum_chart(creativity_spectrum)
    st.plotly_chart(fig, use_container_width=True)

def display_style_axes(style_axes):
    """Display only the radar plot of style axes."""
    radar_fig = create_style_axes_chart(style_axes)
    radar_fig.update_layout(title_text="Style Axes (Used Values)", height=500)
    st.plotly_chart(radar_fig, use_container_width=True)

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

def display_creativity_and_style_axes(creativity_spectrum, style_axes):
    """Display creativity spectrum and style axes charts side by side."""
    col1, col2 = st.columns(2)
    with col1:
        fig_creativity = create_creativity_spectrum_chart(creativity_spectrum)
        st.plotly_chart(fig_creativity, use_container_width=True)
    with col2:
        fig_style = create_style_axes_chart(style_axes)
        fig_style.update_layout(title_text="Style Axes (Used Values)", height=500)
        st.plotly_chart(fig_style, use_container_width=True)

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

def create_mini_dashboard(pairs, key_prefix="pair"):
    """Display concept/medium pairs in a grid of columns with selection checkboxes.

    Parameters
    ----------
    pairs : list
        List of concept/medium dictionaries to display.
    key_prefix : str, optional
        Prefix for the Streamlit widget keys so that different dashboards can
        coexist without key collisions.
    """

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
                        concept_key = 'concept' if 'concept' in pair else 'hook'
                        medium_key = 'medium' if 'medium' in pair else 'arrangement'
                        st.markdown(f"**Concept:** {pair.get(concept_key)}")
                        st.markdown(f"**Medium:** {pair.get(medium_key)}")
                    checkbox_key = f"{key_prefix}_{i+j}"
                    if st.checkbox(f"Select Pair {i+j+1}", key=checkbox_key):
                        selected_pairs.append(i+j)
                    st.markdown("---")
    return selected_pairs

def display_temporary_results(title, data, is_dataframe=False, expanded=True):
    """Display short-lived results in an expander.

    Parameters
    ----------
    title : str
        Title of the expander.
    data : Any
        Data to render inside the expander. Can be list, dict, or dataframe.
    is_dataframe : bool, optional
        If True, render the data as a dataframe.
    expanded : bool, optional
        Whether the expander should be expanded initially.
    """
    with st.expander(title, expanded=expanded):
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

def display_temporary_results_no_expander(title, data, is_dataframe=False):
    """Display short-lived results in an expander.

    Parameters
    ----------
    title : str
        Title of the expander.
    data : Any
        Data to render inside the expander. Can be list, dict, or dataframe.
    is_dataframe : bool, optional
        If True, render the data as a dataframe.
    """
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

def get_input_settings() -> dict:
    """Return a JSON-serializable copy of ``st.session_state``."""
    allowed_types = (int, float, str, bool, list, dict, type(None))
    settings = {}
    for key, value in st.session_state.items():
        if isinstance(value, allowed_types):
            try:
                json.dumps(value)
            except TypeError:
                continue
            settings[key] = value
    return settings

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