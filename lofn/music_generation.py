# music_generation.py

import logging
import streamlit as st
from llm_integration import generate_music_prompts
from config import Config

logger = logging.getLogger(__name__)

def generate_music_prompts_ui():
    try:
        with st.spinner("Generating music prompts..."):
            music_prompt, lyrics_prompt = generate_music_prompts(
                st.session_state['input'],
                st.session_state.get('run_time', 3.0),
                max_retries=st.session_state.get('max_retries', 3),
                temperature=st.session_state.get('temperature', 0.7),
                model=st.session_state.get('model', 'default-model'),
                debug=st.session_state.get('debug', False),
            )
        st.session_state['music_prompt'] = music_prompt
        st.session_state['lyrics_prompt'] = lyrics_prompt
        st.success("Music prompts generated successfully!")
        display_music_prompts()
    except Exception as e:
        st.error("An error occurred while generating music prompts.")
        logger.exception("Error generating music prompts: %s", e)

def display_music_prompts():
    st.subheader("Generated Music Prompt")
    st.code(st.session_state['music_prompt'], language='text')

    st.subheader("Generated Lyrics Prompt")
    st.code(st.session_state['lyrics_prompt'], language='text')

    st.info("Copy the above prompts and paste them into Udio to generate your music.")

def generate_music():
    st.header("Generate Your Music Concept")

    # User input section
    st.subheader("Describe Your Song Idea")
    user_input = st.text_area(
        "",
        placeholder="Describe the themes, emotions, and specific elements you want in your song.",
        help="Provide a detailed description of your song idea.",
    )
    st.session_state['input'] = user_input

    # Provide an example or guidance
    if not user_input:
        st.info("Tip: Include themes, emotions, specific elements, and desired run time length. For example, 'A heartfelt ballad about overcoming challenges, evoking emotions of resilience and hope, with a desired length of 4 minutes.'")

    # Process Flow Control
    if st.button("Generate Music Prompts"):
        if user_input.strip() == "":
            st.warning("Please provide a description of your song idea.")
        else:
            generate_music_prompts_ui()
