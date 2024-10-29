# video_integration.py

import os
import logging
import requests
import json
import streamlit as st

from config import Config
from helpers import send_to_discord, parse_output
from llm_integration import (
    generate_video_concepts,
    generate_video_prompts,
    get_llm,
    call_language_model
)

logger = logging.getLogger(__name__)

def generate_runway_video(prompt, api_key, duration, resolution):
    """
    Generates a video based on the given prompt using the RunwayML API.

    Parameters:
        prompt (str): The video prompt text.
        api_key (str): The API key for authenticating with the RunwayML API.
        duration (int): The duration of the video in seconds.
        resolution (str): The resolution of the video (e.g., '720p').

    Returns:
        str or None: The URL of the generated video, or None if an error occurred.
    """
    # Build the request payload
    payload = {
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        # Include other parameters as required by the API
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post("https://api.runwayml.com/v1/videos/generate", json=payload, headers=headers)
        if response.status_code == 200:
            video_url = response.json().get("video_url")
            return video_url
        else:
            logger.error(f"RunwayML API error: {response.text}")
            return None
    except Exception as e:
        logger.exception("Error generating video via API: %s", e)
        return None

def generate_video_concepts_ui(input_text, max_retries, temperature, model, debug=False):
    """
    Generates video concepts and mediums and displays them in the UI.
    """
    try:
        concepts, style_axes, creativity_spectrum = generate_video_concepts(
            input_text,
            max_retries,
            temperature,
            model,
            debug,
        )
        st.session_state['video_concept_mediums'] = concepts
        st.session_state['style_axes'] = style_axes
        st.session_state['creativity_spectrum'] = creativity_spectrum
        st.success("Video concepts generated successfully!")
    except Exception as e:
        st.error("An error occurred while generating video concepts.")
        logger.exception("Error generating video concepts: %s", e)

def display_video_concepts():
    st.subheader("Generated Video Concepts and Mediums")
    # Assuming create_mini_dashboard is defined in helpers
    selected_pairs = create_mini_dashboard(st.session_state['video_concept_mediums'])
    st.session_state['selected_video_pairs'] = selected_pairs

    # Proceed to generate prompts for selected concepts
    if st.button("Generate Video Prompts for Selected Concepts"):
        if selected_pairs:
            for idx in selected_pairs:
                pair = st.session_state['video_concept_mediums'][idx]
                generate_video_prompts_for_pair(pair)
        else:
            st.warning("Please select at least one concept to generate prompts.")

def generate_video_prompts_for_pair(pair):
    st.subheader(f"Generating Video Prompts for '{pair['concept']}' in '{pair['medium']}'")
    try:
        with st.spinner(f"Generating video prompts for '{pair['concept']}'..."):
            prompts_df = generate_video_prompts(
                st.session_state['input'],
                pair['concept'],
                pair['medium'],
                max_retries=st.session_state.get('max_retries', 3),
                temperature=st.session_state.get('temperature', 0.7),
                model=st.session_state.get('model', 'default-model'),
                debug=st.session_state.get('debug', False),
                style_axes=st.session_state.get('style_axes'),
                creativity_spectrum=st.session_state.get('creativity_spectrum'),
            )
        st.session_state['video_prompts_df'] = prompts_df
        st.success(f"Video prompts generated for '{pair['concept']}'")
        display_video_prompts(prompts_df, pair)
    except Exception as e:
        st.error(f"An error occurred while generating video prompts for '{pair['concept']}'.")
        logger.exception("Error generating video prompts: %s", e)

def display_video_prompts(prompts_df, pair):
    st.subheader(f"Video Prompts for '{pair['concept']}' in '{pair['medium']}'")
    for idx, row in prompts_df.iterrows():
        st.markdown(f"**Prompt {idx+1}:** {row['Synthesized Prompts']}")
        st.markdown("---")
    generate_videos(prompts_df, pair)

def generate_videos(prompts_df, pair):
    st.subheader(f"Generating Videos for '{pair['concept']}'")
    try:
        with st.spinner(f"Generating videos for '{pair['concept']}'..."):
            for idx, row in prompts_df.iterrows():
                video_prompt = row['Synthesized Prompts']
                video_url = generate_runway_video(
                    video_prompt,
                    Config.RUNWAYML_API_KEY,
                    st.session_state.get('video_duration', 5),
                    st.session_state.get('video_resolution', '720p')
                )
                if video_url:
                    st.markdown(f"**Video {idx+1}:**")
                    st.video(video_url)
                else:
                    st.warning(f"Video {idx+1} could not be generated.")
        st.success(f"Videos generated for '{pair['concept']}'")
    except Exception as e:
        st.error(f"An error occurred while generating videos for '{pair['concept']}'.")
        logger.exception("Error generating videos: %s", e)
