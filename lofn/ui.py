import os
import logging
import random
import pandas as pd
import streamlit as st
import yaml

from image_generation import (
    render_image_controls,
    get_model_params,
    generate_dalle_images,
    save_music_metadata,
    save_video_metadata,
)
from datetime import datetime
from config import Config
from helpers import *
from llm_integration import *

logger = logging.getLogger(__name__)

with open('/lofn/prompts/panels.yaml', 'r') as f:
    PANEL_OPTIONS = yaml.safe_load(f)

PANEL_OPTIONS = [{'name': 'LLM Generated', 'prompt': ''}] + PANEL_OPTIONS

with open('/lofn/prompts/personalities.yaml', 'r') as f:
    PERSONALITY_OPTIONS = yaml.safe_load(f)

PERSONALITY_OPTIONS = [{'name': 'LLM Generated', 'prompt': ''}] + PERSONALITY_OPTIONS

class LofnError(Exception):
    """Custom exception class for Lofn-specific errors."""
    pass

class LofnApp:
    def __init__(self):
        self.model = None
        self.prompt_model = None
        self.image_model = None
        self.temperature = 0.7
        self.max_retries = 3
        self.debug:bool = False

        # Build the model lists
        self.available_models = self.get_available_models()
        self.available_image_models = self.get_available_image_models()
        self.initialize_session_state()

    def get_available_models(self):
        models = []

        # Add OpenAI-based models if OPENAI_API is available
        if Config.OPENAI_API:
            models.extend([
                "gpt-4.1", "o4-mini", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o1", "o3-mini", "gpt-4.5-preview", "gpt-4o-mini", "gpt-4o", "o3-mini-2025-01-31",
                "o1-2024-12-17", "o1-preview", "o1-mini",  
                "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "chatgpt-4o-latest",
                "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"
            ])
        # Add Anthropic models if ANTHROPIC_API is available
        if Config.ANTHROPIC_API:
            models.extend([
                "claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514", "claude-opus-4-20250514",
                "claude-3-5-sonnet-latest", "claude-3-5-haiku-20241022",
                "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620",
                "claude-3-opus-20240229", "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ])
        # Add Google models if GOOGLE_API is available
        if Config.GOOGLE_API:
            models.extend([
                "gemini-2.5-pro-preview-05-06", "gemini-2.5-pro-preview-06-05", "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-exp-03-25", "gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-exp", "gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp", "gemini-1.5-flash-002", "gemini-1.5-pro-002",
                "gemini-exp-1206", "gemini-exp-1121", "gemini-exp-1114",
                "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro",
                "gemini-1.5-pro-exp-0827", "gemini-1.5-pro-exp-0801"
            ])
        # Add Poe models if POE_API is available
        if Config.POE_API:
            models.extend([
                "Poe-o1", "Poe-GPT-4.5", "Poe-o3-mini", "Poe-o3-mini-high", "Poe-o1-preview-128k", "Poe-o1-mini-128k", "Poe-Gemini-1.5-Pro-128k",
                "Poe-Llama-3.1-405B-FW-128k", "Poe-Gemini-1.5-Flash-128k",
                "Poe-GPT-4o-Mini-128k", "Poe-GPT-4o-128k", "Poe-Claude-3.5-Sonnet-200k",
                "Poe-Mistral-Large-2-128k", "Poe-Llama-3.2-11B-FW-131k", 
                "Poe-Llama-3.2-90B-FW-131k", "Poe-Llama-3.1-8B-T-128k",
                "Poe-Llama-3.1-70B-FW-128k", "Poe-Llama-3.1-70B-T-128k",
                "Poe-Llama-3.1-8B-FW-128k", "Poe-GPT-4-Turbo-128k",
                "Poe-Claude-3-Opus-200k", "Poe-Claude-3-Sonnet-200k",
                "Poe-Claude-3-Haiku-200k", "Poe-Mixtral8x22b-Inst-FW",
                "Poe-Command-R", "Poe-Gemma-2-9b-T", "Poe-Mistral-Large-2",
                "Poe-Mistral-Medium", "Poe-Snowflake-Arctic-T", "Poe-RekaCore",
                "Poe-RekaFlash", "Poe-Command-R-Plus", "Poe-GPT-3.5-Turbo",
                "Poe-Mixtral-8x7B-Chat", "Poe-DeepSeek-Coder-33B-T",
                "Poe-CodeLlama-70B-T", "Poe-Qwen2-72B-Chat", "Poe-Qwen-72B-T",
                "Poe-Claude-2", "Poe-Google-PaLM", "Poe-Llama-3-8b-Groq",
                "Poe-Llama-3-8B-T", "Poe-Gemma-2-27b-T", "Poe-Assistant",
                "Poe-Claude-3.5-Sonnet", "Poe-GPT-4o-Mini", "Poe-GPT-4o",
                "Poe-Llama-3.1-405B-T", "Poe-Gemini-1.5-Flash",
                "Poe-Gemini-1.5-Pro", "Poe-Claude-3-Sonnet", 
                "Poe-Claude-3-Haiku", "Poe-Claude-3-Opus", "Poe-Gemini-1.0-Pro",
                "Poe-Llama-3-70B-T", "Poe-Llama-3-70b-Inst-FW",
                "Poe-Llama-3.2-90B-FW-131k", "Poe-Llama-3.2-11B-FW-131k"
            ])

        # Optionally fetch from OpenRouter if OPEN_ROUTER_API_KEY is present
        if Config.OPEN_ROUTER_API_KEY:
            try:
                or_models = fetch_openrouter_models()
                if or_models:
                    filtered_or_models = filter_models_by_context_length(
                        or_models, min_total_tokens=25000, min_response_tokens=15000
                    )
                    # Extract model IDs
                    models.extend(['OR-'+m['id'] for m in filtered_or_models])
                else:
                    print("No OpenRouter API key found. Skipping OpenRouter models.")
            except Exception as e:
                st.error("An error occurred while getting OpenRouter models.")
                logger.exception("Error getting OpenRouter models: %s", e)

        # Prioritize the most powerful models if available
        priority_order = [
            "gemini-2.5-pro-preview-06-05",
            "claude-opus-4-20250514",
            "o3",
            "claude-sonnet-4-20250514",
        ]
        ordered = [m for m in priority_order if m in models]
        ordered.extend([m for m in models if m not in ordered])

        return ordered

    def get_available_image_models(self):
        models = ["None"]
        if Config.FAL_API_KEY:
            models.extend([
                "fal-ai/flux-pro/v1.1-ultra", "fal-ai/flux-pro/v1.1", "fal-ai/recraft-v3",
                "fal-ai/omnigen-v1", "fal-ai/stable-diffusion-v35-large",
                "fal-ai/stable-diffusion-v35-medium", "fal-ai/flux-pro",
                "fal-ai/flux-realism", "fal-ai/flux-dev", "fal-ai/flux/schnell"
            ])
        if Config.IDEOGRAM_API_KEY:
            models.append("Ideogram")
        if Config.GOOGLE_PROJECT_ID:
            models.append("Google Imagen 3")
        if Config.OPENAI_API:
            models.append("DALL-E 3")
        if Config.POE_API:
            models.extend([
                "Poe-FLUX-pro-1.1-ultra", "Poe-FLUX-pro-1.1", "Poe-Imagen3", "Poe-StableDiffusion3.5-L",
                "Poe-FLUX-pro", "Poe-DALL-E-3", "Poe-Ideogram-v2",
                "Poe-Playground-v2.5","Poe-Playground-v3", "Poe-Ideogram",
                "Poe-FLUX-dev","Poe-FLUX-schnell","Poe-LivePortrait","Poe-StableDiffusion3",
                "Poe-SD3-Turbo", "Poe-StableDiffusionXL","Poe-StableDiffusion3-2B",
                "Poe-SD3-Medium","Poe-RealVisXL"
            ])
        return models

    def get_available_video_models(self):
        models = []
        if Config.RUNWAYML_API_KEY:
            models.append("RunwayML Gen-2")
        else:
            st.warning("No RunwayML API key found. Video generation models will not be available.")
        return models

    def run(self):
        # Create tabs within the app
        tabs = ["Image Generation", "Video Generation", "Music Generation"]
        selected_tab = st.sidebar.selectbox("Select Mode", tabs)

        self.render_sidebar()

        if selected_tab == "Image Generation":
            self.render_image_generation()
        elif selected_tab == "Video Generation":
            self.render_video_generation()
        elif selected_tab == "Music Generation":
            self.render_music_generation()

    def render_sidebar(self):
        st.sidebar.header('Settings')
        st.session_state['competition_mode'] = st.sidebar.checkbox(
            'Competition Mode', value=st.session_state.get('competition_mode', False)
        )
        if st.session_state['competition_mode']:
            # Use the main input text as the competition text
            st.session_state['competition_text'] = st.session_state.get('input', '')
            st.sidebar.info('Competition mode uses the main input text.')
            panel_names = [p['name'] for p in PANEL_OPTIONS] + ['Custom']
            st.session_state['selected_panel'] = st.sidebar.selectbox(
                'Panel Prompt', panel_names, index=panel_names.index(st.session_state.get('selected_panel', panel_names[0]))
            )
            if st.session_state['selected_panel'] == 'Custom':
                st.session_state['custom_panel'] = st.sidebar.text_area(
                    'Custom Panel Prompt', value=st.session_state.get('custom_panel', ''), height=150
                )
            elif st.session_state['selected_panel'] == 'LLM Generated':
                st.sidebar.info('Panel will be generated automatically.')
            else:
                st.session_state['custom_panel'] = next(p['prompt'] for p in PANEL_OPTIONS if p['name'] == st.session_state['selected_panel'])

            personality_names = [p['name'] for p in PERSONALITY_OPTIONS] + ['Custom']
            st.session_state['selected_personality'] = st.sidebar.selectbox(
                'Personality', personality_names, index=personality_names.index(st.session_state.get('selected_personality', personality_names[0]))
            )
            if st.session_state['selected_personality'] == 'Custom':
                st.session_state['custom_personality'] = st.sidebar.text_area(
                    'Custom Personality', value=st.session_state.get('custom_personality', ''), height=150
                )
            elif st.session_state['selected_personality'] == 'LLM Generated':
                st.sidebar.info('Personality will be generated automatically.')
            else:
                st.session_state['custom_personality'] = next(p['prompt'] for p in PERSONALITY_OPTIONS if p['name'] == st.session_state['selected_personality'])
            st.session_state['num_best_pairs'] = st.sidebar.number_input(
                'Top Pairs', min_value=1, max_value=10,
                value=st.session_state.get('num_best_pairs', 3), step=1
            )

        # Language Model Settings
        with st.sidebar.expander("Language Model Settings", expanded=True):
            self.model = st.selectbox(
                "Select Concept/Medium Model",
                self.available_models,
                index=self.available_models.index(
                    st.session_state.get('model', self.available_models[0])
                ) if self.available_models else 0,
                help="Model used generating concepts and mediums."
            )
            st.session_state['model'] = self.model
            self.prompt_model = st.selectbox(
                "Prompt Generation Model",
                self.available_models,
                index=self.available_models.index(
                    st.session_state.get('prompt_model', self.available_models[0])
                ) if self.available_models else 0,
                help="Model used for generating image prompts."
            )
            st.session_state['prompt_model'] = self.prompt_model

            if self.model.startswith('o1') or self.model.startswith('o3') or self.model.startswith('o4'):
                # Reasoning Level for o1
                # (For all models it’s accessible, but we only really use it if model is 'o1' or 'o1-mini')
                st.session_state['reasoning_level'] = st.selectbox(
                    "Reasoning Level (for o1 models)",
                    ["low", "medium", "high"],
                    index=1
                )
            else:   
                self.temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Controls the randomness of the model's output."
                )

            self.max_retries = st.slider(
                "Maximum Retries",
                min_value=1,
                max_value=10,
                value=3,
                help="Set the maximum number of retries for LLM responses."
            )

            self.debug = st.checkbox("Debug Mode", value=False, help="Enable debug mode for detailed logs.")

        with st.sidebar.expander("Image Generation Settings", expanded=True):
            self.available_image_models = self.get_available_image_models()

            self.image_model = st.selectbox(
                "Select Image Model",
                self.available_image_models,
                help="Choose the image generation model."
            )
            render_image_controls(self.image_model)

        with st.sidebar.expander("Style Personalization", expanded=False):
            st.session_state['auto_style'] = st.checkbox("Automatic Style", value=True, help="Enable automatic style determination.")
            if not st.session_state['auto_style']:
                st.subheader("Adjust Style Axes")
                selected_medium = st.session_state.get('selected_tab', 'Image Generation')
                if selected_medium == 'Image Generation':
                    style_axes = {
                       "Abstraction vs. Realism": st.slider("Abstraction vs. Realism (0: Abstract, 100: Realism)", 0, 100, 50),
                       "Emotional Valence": st.slider("Emotional Valence (0: Negative, 100: Positive)", 0, 100, 50),
                       "Color Intensity": st.slider("Color Intensity (0: Muted, 100: Vibrant)", 0, 100, 50),
                       "Symbolic Density": st.slider("Symbolic Density (0: Literal, 100: Symbolic)", 0, 100, 50),
                       "Compositional Complexity": st.slider("Compositional Complexity (0: Simple, 100: Complex)", 0, 100, 50),
                       "Textural Richness": st.slider("Textural Richness (0: Smooth, 100: Textured)", 0, 100, 50),
                       "Symmetry vs. Asymmetry": st.slider("Symmetry vs. Asymmetry (0: Asymmetrical, 100: Symmetrical)", 0, 100, 50),
                       "Novelty": st.slider("Novelty (0: Traditional, 100: Innovative)", 0, 100, 50),
                       "Figure-Ground Relationship": st.slider("Figure-Ground Relationship (0: Distinct, 100: Blended)", 0, 100, 50),
                       "Dynamic vs. Static": st.slider("Dynamic vs. Static (0: Static, 100: Dynamic)", 0, 100, 50)
                    }
                elif selected_medium == 'Video Generation':
                    style_axes = {
                       "Narrative Complexity": st.slider("Narrative Complexity (0: Simple, 100: Complex)", 0, 100, 50),
                       "Emotional Intensity": st.slider("Emotional Intensity (0: Subtle, 100: Intense)", 0, 100, 50),
                       "Visual Style": st.slider("Visual Style (0: Plain, 100: Stylized)", 0, 100, 50),
                       "Symbolism": st.slider("Symbolism (0: Literal, 100: Symbolic)", 0, 100, 50),
                       "Pacing": st.slider("Pacing (0: Slow, 100: Fast)", 0, 100, 50),
                       "Character Emphasis": st.slider("Character Emphasis (0: Background, 100: Foreground)", 0, 100, 50),
                       "Color Palette": st.slider("Color Palette (0: Monochrome, 100: Vibrant)", 0, 100, 50),
                       "Cinematography": st.slider("Cinematography (0: Static Shots, 100: Dynamic Shots)", 0, 100, 50),
                       "Surrealism vs. Realism": st.slider("Surrealism vs. Realism (0: Surreal, 100: Realistic)", 0, 100, 50),
                       "Dynamic vs. Static": st.slider("Dynamic vs. Static (0: Static, 100: Dynamic)", 0, 100, 50)
                    }
                elif selected_medium == 'Music Generation':
                    style_axes = {
                       "Tempo": st.slider("Tempo (0: Slow, 100: Fast)", 0, 100, 50),
                       "Mood": st.slider("Mood (0: Negative, 100: Positive)", 0, 100, 50),
                       "Instrumentation Complexity": st.slider("Instrumentation Complexity (0: Simple, 100: Complex)", 0, 100, 50),
                       "Lyrical Depth": st.slider("Lyrical Depth (0: Simple, 100: Profound)", 0, 100, 50),
                       "Genre Fusion": st.slider("Genre Fusion (0: Pure Genre, 100: Fusion)", 0, 100, 50),
                       "Vocal Style": st.slider("Vocal Style (0: Soft, 100: Powerful)", 0, 100, 50),
                       "Rhythmic Complexity": st.slider("Rhythmic Complexity (0: Simple, 100: Complex)", 0, 100, 50),
                       "Melodic Emphasis": st.slider("Melodic Emphasis (0: Background, 100: Foreground)", 0, 100, 50),
                       "Harmonic Richness": st.slider("Harmonic Richness (0: Simple, 100: Rich)", 0, 100, 50),
                       "Production Style": st.slider("Production Style (0: Raw, 100: Polished)", 0, 100, 50)
                    }
                else:
                    style_axes = {}
                st.session_state['style_axes'] = style_axes
            else:
                if 'style_axes' not in st.session_state:
                    st.session_state['style_axes'] = None
                st.session_state['auto_creativity_spectrum'] = st.checkbox("Automatic Creativity Spectrum", value=True, help="Enable automatic creativity spectrum determination.")
                if not st.session_state['auto_creativity_spectrum']:
                    st.subheader("Adjust Creativity Spectrum")
                    creativity_spectrum = {
                        "literal": st.slider("Literal", 0, 100, 33),
                        "inventive": st.slider("Inventive", 0, 100, 33),
                        "transformative": st.slider("Transformative", 0, 100, 34),
                    }
                    st.session_state['creativity_spectrum'] = creativity_spectrum
                elif 'creativity_spectrum' not in st.session_state:
                    st.session_state['creativity_spectrum'] = None

        with st.sidebar.expander("Discord Settings", expanded=False):
            st.session_state['send_to_discord'] = st.checkbox(
                "Send to Discord",
                value=True,
                help="Enable or disable sending prompts to a Discord channel.",
            )
            if st.session_state['send_to_discord']:
                st.session_state['use_default_webhook'] = st.checkbox(
                    "Use Environment Webhook",
                    value=True,
                    help="Enable manual webhook entry.",
                )
                if st.session_state['use_default_webhook']:
                    st.text("Webhook URL: **********")
                    webhook_url = Config.WEBHOOK_URL
                else:
                    webhook_url = st.text_input(
                        "Discord Webhook URL",
                        value=Config.WEBHOOK_URL,
                        placeholder="Enter your Discord webhook URL here",
                        help="Provide the webhook URL for your Discord channel.",
                    )
                st.session_state['webhook_url'] = webhook_url

    def render_image_generation(self):
        self.render_image_main_container()


    def render_image_main_container(self):
        st.header("Generate Your Art Concept")


        # The user’s idea. Tying it to session_state so it does not vanish
        st.subheader("Describe Your Idea")
        st.text_area(
            label="",
            label_visibility="collapsed", # hides it visually if desired
            key="input",  # This ensures the text stays in st.session_state['input']
            placeholder="Describe the essence of the art you wish to generate.",
            help="Provide a detailed description of your idea to get the best results.",
            height=200
        )

        # If there's no user input, provide a tip
        if not st.session_state['input']:
            st.info("Tip: You can describe anything from abstract concepts to detailed scenes. For example, 'I want to capture the essence of a mysterious and powerful witch's familiar.'")

        # Manual Input Option
        st.subheader("Manual Concept and Medium Input (Optional)")
        st.session_state['manual_input'] = st.checkbox("Manually input Concept and Medium")
        if st.session_state['manual_input']:
            manual_concept = st.text_input("Enter your Concept")
            manual_medium = st.text_input("Enter your Medium")
            if st.button("Generate Image Prompts for Manual Input"):
                if manual_concept.strip() == "" or manual_medium.strip() == "":
                    st.warning("Please provide both a concept and a medium.")
                else:
                    self.generate_prompts_for_manual_input(manual_concept, manual_medium)

        # Process Flow Control
        st.markdown("<div class='sticky-action-bar'>", unsafe_allow_html=True)
        if st.button("Generate Concepts"):
            if not st.session_state['input'].strip():
                st.warning("Please provide a description of your idea.")
            else:
                style_axes, creativity_spectrum = self.generate_concepts()

        if st.session_state.get('competition_mode') and st.button("Run Competition", key="run_competition"):
            if not st.session_state['input'].strip():
                st.warning("Please provide a description of your idea.")
            else:
                self.run_competition()
        st.markdown("</div>", unsafe_allow_html=True)

        # If we have concept_mediums, display them
        if 'concept_mediums' in st.session_state and st.session_state['concept_mediums']:
            self.display_concepts()

    def generate_concepts(self):
        try:
            st.session_state['concept_mediums'] = []
            st.session_state['progress_step'] = 0
            input_text = st.session_state['input']
            if st.session_state.get('competition_mode'):
                personality_text = st.session_state.get('custom_personality', '')
                if st.session_state.get('selected_personality') == 'LLM Generated':
                    if not personality_text:
                        personality_text = generate_personality_prompt(
                            st.session_state.get('input', ''),
                            self.max_retries,
                            self.temperature,
                            self.model,
                            self.debug,
                            st.session_state.get('reasoning_level', 'medium')
                        )
                        st.session_state['custom_personality'] = personality_text
                    display_temporary_results("Personality", personality_text, expanded=False)

                meta_prompt, frames_list = generate_meta_prompt(
                    st.session_state.get('input', ''),
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    reasoning_level=st.session_state.get('reasoning_level', 'medium'),
                    personality_prompt=personality_text,
                )
                st.session_state['meta_prompt'] = meta_prompt['meta_prompt']
                panel_text = st.session_state.get('custom_panel', '')
                if st.session_state.get('selected_panel') == 'LLM Generated':
                    if not panel_text:
                        panel_text = generate_panel_prompt(
                            meta_prompt['meta_prompt'],
                            self.max_retries,
                            self.temperature,
                            self.model,
                            self.debug,
                            st.session_state.get('reasoning_level', 'medium'),
                            personality_prompt=personality_text,
                        )
                        st.session_state['custom_panel'] = panel_text
                    display_temporary_results("Panel Prompt", panel_text, expanded=False)
                template = read_prompt('/lofn/prompts/overall_prompt_template.txt')
                input_text = (
                    template.replace('{Meta-Prompt}', meta_prompt['meta_prompt'])
                    .replace('{Panel-prompt}', panel_text)
                    .replace('{Personality-prompt}', personality_text)
                    .replace('{frames_list}', frames_list)
                    .replace('{input}', st.session_state.get('input', ''))
                )
                display_temporary_results("Meta Prompt", meta_prompt['meta_prompt'], expanded=False)
            st.session_state['prompt_input'] = input_text
            with st.spinner("Generating concepts..."):
                concepts, style_axes, creativity_spectrum = generate_concept_mediums(
                    input_text,
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state.get('style_axes', None),
                    creativity_spectrum=st.session_state.get('creativity_spectrum', None),
                    reasoning_level=st.session_state.get('reasoning_level', 'medium'),  # For o1
                )
            st.session_state['concept_mediums'] = concepts
            st.success("Concepts generated successfully!")
            st.session_state['progress_step'] = 1
            # Store the updated axes back in session state
            st.session_state['style_axes'] = style_axes
            st.session_state['creativity_spectrum'] = creativity_spectrum

            return style_axes, creativity_spectrum
        except Exception as e:
            st.error("An error occurred while generating concepts.")
            logger.exception("Error generating concepts: %s", e)

    def display_concepts(self):
        st.subheader("Generated Concepts and Mediums")

        # Display concepts in a mini-dashboard
        selected_pairs = create_mini_dashboard(st.session_state['concept_mediums'])
        st.session_state['selected_pairs'] = selected_pairs

        # Buttons to proceed
        if st.button("Generate Image Prompts for Selected Concepts"):
            if selected_pairs:
                for idx in selected_pairs:
                    pair = st.session_state['concept_mediums'][idx]
                    self.generate_prompts_for_pair(pair)
            else:
                st.warning("Please select at least one concept to generate prompts.")

        if st.button("Generate Image Prompts for All Concepts"):
            for pair in st.session_state['concept_mediums']:
                self.generate_prompts_for_pair(pair)

    def generate_prompts_for_pair(self, pair):
        st.subheader(f"Generating Prompts for '{pair['concept']}' in '{pair['medium']}'")
        try:
            with st.spinner(f"Generating prompts for '{pair['concept']}'..."):
                prompts_df = generate_image_prompts(
                    st.session_state['prompt_input'],
                    pair['concept'],
                    pair['medium'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.prompt_model,
                    debug=self.debug,
                    style_axes=st.session_state['style_axes'],
                    creativity_spectrum=st.session_state['creativity_spectrum'],
                    reasoning_level=st.session_state.get('reasoning_level','medium'),
                )
            st.session_state['prompts_df'] = prompts_df
            st.success(f"Prompts generated for '{pair['concept']}'")
            st.session_state['progress_step'] = 2
            self.display_prompts(prompts_df, pair)
        except Exception as e:
            st.error(f"An error occurred while generating prompts for '{pair['concept']}'.")
            logger.exception("Error generating prompts: %s", e)

    def generate_prompts_for_manual_input(self, concept, medium):
        st.subheader(f"Generating Prompts for '{concept}' in '{medium}'")
        try:
            with st.spinner(f"Generating prompts for '{concept}'..."):
                prompts_df = generate_image_prompts(
                    st.session_state['prompt_input'],
                    concept,
                    medium,
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.prompt_model,
                    debug=self.debug,
                    style_axes=st.session_state['style_axes'],
                    creativity_spectrum=st.session_state['creativity_spectrum'],
                    reasoning_level=st.session_state.get('reasoning_level','medium'),
                )
            st.session_state['prompts_df'] = prompts_df
            st.success(f"Prompts generated for '{concept}'")
            st.session_state['progress_step'] = 2
            self.display_prompts(prompts_df, {'concept': concept, 'medium': medium})
        except Exception as e:
            st.error(f"An error occurred while generating prompts for '{concept}'.")
            logger.exception("Error generating prompts: %s", e)

    def display_prompts(self, prompts_df, pair):
        st.subheader(f"Prompts for '{pair['concept']}' in '{pair['medium']}'")
        for idx, row in prompts_df.iterrows():
            st.markdown(f"**Prompt {idx+1}:**")
            st.code(row['Revised Prompts'], language='')
            st.markdown(f"**Synthesized Prompt {idx+1}:**")
            st.code(row['Synthesized Prompts'], language='')
            st.markdown("---")

        self.generate_images(prompts_df, pair)

    def run_competition(self):
        self.generate_concepts()
        pairs = st.session_state.get('concept_mediums', [])
        if not pairs:
            return
        try:
            with st.spinner("Panel voting on best pairs..."):
                best_pairs = select_best_pairs(
                    st.session_state['prompt_input'],
                    pairs,
                    st.session_state.get('num_best_pairs', 3),
                    self.max_retries,
                    self.temperature,
                    self.model,
                    self.debug,
                    st.session_state.get('reasoning_level', 'medium'),
                )
            st.session_state['best_pairs'] = best_pairs
            st.success("Best pairs selected by panel")
        except Exception as e:
            st.error("An error occurred while selecting best pairs.")
            logger.exception("Error selecting best pairs: %s", e)

        top_n = st.session_state.get('num_best_pairs', 3)
        gen_pairs = st.session_state.get('best_pairs', pairs)
        for pair in gen_pairs[:top_n]:
            self.generate_prompts_for_pair(pair)

    def run_video_competition(self):
        self.generate_ui_video_concepts()
        pairs = st.session_state.get('video_concept_mediums', [])
        if not pairs:
            return
        try:
            with st.spinner("Panel voting on best pairs..."):
                best_pairs = select_best_pairs(
                    st.session_state['input'],
                    pairs,
                    st.session_state.get('num_best_pairs', 3),
                    self.max_retries,
                    self.temperature,
                    self.model,
                    self.debug,
                    st.session_state.get('reasoning_level', 'medium'),
                )
            st.session_state['video_best_pairs'] = best_pairs
            st.success("Best video pairs selected by panel")
        except Exception as e:
            st.error("An error occurred while selecting best video pairs.")
            logger.exception("Error selecting video pairs: %s", e)

        top_n = st.session_state.get('num_best_pairs', 3)
        gen_pairs = st.session_state.get('video_best_pairs', pairs)
        for pair in gen_pairs[:top_n]:
            self.generate_video_prompts_for_pair(pair)

    def run_music_competition(self):
        try:
            personality_text = st.session_state.get('custom_personality', '')
            if st.session_state.get('selected_personality') == 'LLM Generated':
                if not personality_text:
                    personality_text = generate_personality_prompt(
                        st.session_state.get('input', ''),
                        self.max_retries,
                        self.temperature,
                        self.model,
                        self.debug,
                        st.session_state.get('reasoning_level', 'medium')
                    )
                    st.session_state['custom_personality'] = personality_text
                display_temporary_results("Personality", personality_text, expanded=False)
            meta_prompt, frames_list, genres_list = generate_meta_prompt(
                st.session_state.get('input', ''),
                max_retries=self.max_retries,
                temperature=self.temperature,
                model=self.model,
                debug=self.debug,
                reasoning_level=st.session_state.get('reasoning_level', 'medium'),
                medium="music",
                personality_prompt=personality_text,
            )
            st.session_state['meta_prompt'] = meta_prompt['meta_prompt']
            panel_text = st.session_state.get('custom_panel', '')
            if st.session_state.get('selected_panel') == 'LLM Generated':
                if not panel_text:
                    panel_text = generate_panel_prompt(
                        meta_prompt['meta_prompt'],
                        self.max_retries,
                        self.temperature,
                        self.model,
                        self.debug,
                        st.session_state.get('reasoning_level', 'medium'),
                        personality_prompt=personality_text,
                    )
                    st.session_state['custom_panel'] = panel_text
                display_temporary_results("Panel Prompt", panel_text, expanded=False)
            template = read_prompt('/lofn/prompts/music_overall_prompt_template.txt')
            input_text = (
                template.replace('{Meta-Prompt}', meta_prompt['meta_prompt'])
                .replace('{Panel-prompt}', panel_text)
                .replace('{Personality-prompt}', personality_text)
                .replace('{genres_list}', genres_list)
                .replace('{frames_list}', frames_list)
                .replace('{input}', st.session_state.get('input', ''))
            )
            display_temporary_results("Meta Prompt", meta_prompt['meta_prompt'], expanded=False)
            with st.spinner("Generating music prompts..."):
                music_prompt, lyrics_prompt, music_title = generate_music_prompts(
                    input_text,
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                )
            st.session_state['music_prompt'] = music_prompt
            st.session_state['lyrics_prompt'] = lyrics_prompt
            st.session_state['music_title'] = music_title

            metadata = {
                'timestamp': datetime.now(),
                'title': music_title,
                'music_prompt': music_prompt,
                'lyrics_prompt': lyrics_prompt,
                'input_text': st.session_state.get('input', ''),
                'competition': True,
                'model': self.model,
            }
            save_music_metadata(metadata)
            st.success("Music prompts generated successfully!")
            self.display_music_prompts()
        except Exception as e:
            st.error("An error occurred during music competition mode.")
            logger.exception("Error in music competition: %s", e)

    def generate_images(self, prompts_df, pair):
        st.subheader(f"Generating Images for '{pair['concept']}'")
        if self.image_model == "None":
            st.info("Image generation skipped.")
            return
        try:
            with st.spinner(f"Generating images for '{pair['concept']}'..."):
                generate_dalle_images(
                    st.session_state['prompt_input'],
                    pair['concept'],
                    pair['medium'],
                    prompts_df,
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.prompt_model,
                    debug=self.debug,
                    image_model=self.image_model,
                    style_axes=st.session_state['style_axes'],
                    creativity_spectrum=st.session_state['creativity_spectrum'],
                    OPENAI_API=Config.OPENAI_API,
                    reasoning_level=st.session_state['reasoning_level']
                )
            st.success(f"Images generated for '{pair['concept']}'")
            st.session_state['progress_step'] = 3
        except Exception as e:
            st.error(f"An error occurred while generating images for '{pair['concept']}'.")
            logger.exception("Error generating images: %s", e)

    def render_video_generation(self):
        st.header("Generate Your Art Video Concept")
        st.subheader("Describe Your Idea")
        st.text_area(
            label="",
            label_visibility="collapsed", # hides it visually if desired
            key="input",
            placeholder="Describe the essence of the art video you wish to generate.",
            help="Provide a detailed description of your idea to get the best results.",
            height=200
        )
        if not st.session_state['input']:
            st.info("Tip: Describe a scene or narrative you'd like to see in motion.")

        if st.button("Generate Video Concepts"):
            if not st.session_state['input'].strip():
                st.warning("Please provide a description of your idea.")
            else:
                self.generate_ui_video_concepts()

        if st.session_state.get('competition_mode') and st.button("Run Competition", key="run_video_competition"):
            if not st.session_state['input'].strip():
                st.warning("Please provide a description of your idea.")
            else:
                self.run_video_competition()

        if 'video_concept_mediums' in st.session_state and st.session_state['video_concept_mediums']:
            self.display_video_concepts()

    def render_video_sidebar(self):
        st.sidebar.header('Video Generation Settings')
        available_video_models = self.get_available_video_models()
        video_model = st.sidebar.selectbox("Select Video Model", available_video_models)
        video_duration = st.sidebar.slider("Video Duration (seconds)", 5, 60, 10)
        video_resolution = st.sidebar.selectbox("Video Resolution", ["480p","720p","1080p"])
        st.session_state['video_model'] = video_model
        st.session_state['video_duration'] = video_duration
        st.session_state['video_resolution'] = video_resolution

    def generate_ui_video_concepts(self):
        try:
            st.session_state['video_concept_mediums'] = []
            input_text = st.session_state['input']
            if st.session_state.get('competition_mode'):
                personality_text = st.session_state.get('custom_personality', '')
                if st.session_state.get('selected_personality') == 'LLM Generated':
                    if not personality_text:
                        personality_text = generate_personality_prompt(
                            st.session_state.get('input', ''),
                            self.max_retries,
                            self.temperature,
                            self.model,
                            self.debug,
                            st.session_state.get('reasoning_level','medium')
                        )
                        st.session_state['custom_personality'] = personality_text
                    display_temporary_results("Personality", personality_text, expanded=False)
                meta_prompt, frames_list = generate_meta_prompt(
                    st.session_state.get('input', ''),
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    reasoning_level=st.session_state.get('reasoning_level','medium'),
                    personality_prompt=personality_text,
                )
                st.session_state['meta_prompt'] = meta_prompt['meta_prompt']
                panel_text = st.session_state.get('custom_panel', '')
                if st.session_state.get('selected_panel') == 'LLM Generated':
                    if not panel_text:
                        panel_text = generate_panel_prompt(
                            meta_prompt['meta_prompt'],
                            self.max_retries,
                            self.temperature,
                            self.model,
                            self.debug,
                            st.session_state.get('reasoning_level','medium'),
                            personality_prompt=personality_text,
                        )
                        st.session_state['custom_panel'] = panel_text
                    display_temporary_results("Panel Prompt", panel_text, expanded=False)
                template = read_prompt('/lofn/prompts/video_overall_prompt_template.txt')
                input_text = (
                    template.replace('{Meta-Prompt}', meta_prompt['meta_prompt'])
                    .replace('{Panel-prompt}', panel_text)
                    .replace('{Personality-prompt}', personality_text)
                    .replace('{frames_list}', frames_list)
                    .replace('{input}', st.session_state.get('input', ''))
                )
                display_temporary_results("Meta Prompt", meta_prompt['meta_prompt'], expanded=False)
            st.session_state['prompt_input'] = input_text
            with st.spinner("Generating video concepts..."):
                concepts, style_axes, creativity_spectrum = generate_video_concept_mediums(
                    input_text,
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state.get('style_axes', None),
                    creativity_spectrum=st.session_state.get('creativity_spectrum', None),
                    reasoning_level=st.session_state.get('reasoning_level','medium'),
                )
            st.session_state['video_concept_mediums'] = concepts
            st.success("Video concepts generated successfully!")
            st.session_state['style_axes'] = style_axes
            st.session_state['creativity_spectrum'] = creativity_spectrum
        except Exception as e:
            st.error("An error occurred while generating video concepts.")
            logger.exception("Error generating video concepts: %s", e)

    def display_video_concepts(self):
        st.subheader("Generated Video Concepts and Mediums")
        selected_pairs = create_mini_dashboard(st.session_state['video_concept_mediums'])
        st.session_state['selected_video_pairs'] = selected_pairs

        if st.button("Generate Video Prompts for Selected Concepts"):
            if selected_pairs:
                for idx in selected_pairs:
                    pair = st.session_state['video_concept_mediums'][idx]
                    self.generate_video_prompts_for_pair(pair)
            else:
                st.warning("Please select at least one concept to generate prompts.")

        if st.button("Generate Video Prompts for All Concepts"):
            for pair in st.session_state['video_concept_mediums']:
                self.generate_video_prompts_for_pair(pair)

    def generate_video_prompts_for_pair(self, pair):
        st.subheader(f"Generating Video Prompts for '{pair['concept']}' in '{pair['medium']}'")
        try:
            with st.spinner(f"Generating video prompts for '{pair['concept']}'..."):
                prompts_df = generate_video_prompts(
                    st.session_state['input'],
                    pair['concept'],
                    pair['medium'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state['style_axes'],
                    creativity_spectrum=st.session_state['creativity_spectrum'],
                    reasoning_level=st.session_state.get('reasoning_level','medium'),
                )
            st.session_state['video_prompts_df'] = prompts_df
            st.success(f"Video prompts generated for '{pair['concept']}'")
            metadata = {
                'timestamp': datetime.now(),
                'concept': pair['concept'],
                'medium': pair['medium'],
                'prompts': prompts_df.to_dict(orient='list'),
                'input_text': st.session_state.get('input', ''),
                'competition': st.session_state.get('competition_mode', False),
                'model': self.model,
            }
            save_video_metadata(metadata)
            self.display_video_prompts(prompts_df, pair)
        except Exception as e:
            st.error(f"An error occurred while generating video prompts for '{pair['concept']}'.")
            logger.exception("Error generating video prompts: %s", e)

    def display_video_prompts(self, prompts_df, pair):
        st.subheader(f"Video Prompts for '{pair['concept']}' in '{pair['medium']}'")
        for idx, row in prompts_df.iterrows():
            st.markdown(f"**Prompt {idx+1}:**")
            st.code(row['Revised Prompts'], language='')
            st.markdown(f"**Synthesized Prompt {idx+1}:**")
            st.code(row['Synthesized Prompts'], language='')
            st.markdown("---")

    def render_music_generation(self):
        # self.render_music_sidebar()
        st.header("Generate Your Music Concept")

        st.subheader("Describe Your Song Idea")
        st.text_area(
            label="",
            label_visibility="collapsed", # hides it visually if desired
            key="input",
            placeholder="Describe the themes, emotions, and specific elements you want in your song.",
            help="Provide a detailed description of your song idea."
        )
        if not st.session_state['input']:
            st.info("Tip: Include themes, emotions, specific elements, and desired run time length.")

        if st.button("Generate Music Prompts"):
            if not st.session_state['input'].strip():
                st.warning("Please provide a description of your song idea.")
            else:
                self.generate_music_prompts_ui()

        if st.session_state.get('competition_mode') and st.button("Run Competition", key="run_music_competition"):
            if not st.session_state['input'].strip():
                st.warning("Please provide a description of your song idea.")
            else:
                self.run_music_competition()

        if 'music_prompt' in st.session_state and 'lyrics_prompt' in st.session_state:
            self.display_music_prompts()

    # def render_music_sidebar(self):
    #     st.sidebar.header('Music Generation Settings')

    #     run_time = st.sidebar.number_input(
    #         "Desired Song Length (minutes)",
    #         min_value=1.0,
    #         max_value=10.0,
    #         value=2.0,
    #         step=0.5,
    #         help="Set the desired length of the song."
    #     )
    #     st.session_state['run_time'] = run_time

    def generate_music_prompts_ui(self):
        try:
            with st.spinner("Generating music prompts..."):
                hooks, style_axes, creativity = generate_music_hook_arrangements(
                    st.session_state['input'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=None,
                    creativity_spectrum=None,
                    reasoning_level=st.session_state.get('reasoning_level','medium')
                )
                if hooks:
                    first = hooks[0]
                    music_prompt, lyrics_prompt, music_title = generate_music_prompts(
                        st.session_state['input'],
                        first['hook'],
                        first['arrangement'],
                        max_retries=self.max_retries,
                        temperature=self.temperature,
                        model=self.model,
                        debug=self.debug,
                        style_axes=style_axes,
                        creativity_spectrum=creativity,
                        reasoning_level=st.session_state.get('reasoning_level','medium')
                    )
                else:
                    music_prompt = lyrics_prompt = music_title = ""
            st.session_state['music_prompt'] = music_prompt
            st.session_state['lyrics_prompt'] = lyrics_prompt
            st.session_state['music_title'] = music_title
            metadata = {
                'timestamp': datetime.now(),
                'title': music_title,
                'music_prompt': music_prompt,
                'lyrics_prompt': lyrics_prompt,
                'input_text': st.session_state['input'],
                'competition': False,
                'model': self.model,
            }
            save_music_metadata(metadata)
            st.success("Music prompts generated successfully!")
            self.display_music_prompts()
        except Exception as e:
            st.error("An error occurred while generating music prompts.")
            logger.exception("Error generating music prompts: %s", e)

    def display_music_prompts(self):
        st.subheader("Generated Song Title")
        st.code(st.session_state['music_title'], language='text')

        st.subheader("Generated Music Prompt")
        st.code(st.session_state['music_prompt'], language='text')

        st.subheader("Generated Lyrics Prompt")
        st.code(st.session_state['lyrics_prompt'], language='text')
        st.info("Copy the above prompts and paste them into Udio to generate your music.")

    def initialize_session_state(self):
        default_values = {
            'selected_tab': 'Image Generation',
            'video_concept_mediums': None,
            'video_prompts_df': None,
            'music_title': None,
            'music_prompt': None,
            'lyrics_prompt': None,
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
            'image_model': 'None',
            'model': self.available_models[0],
            'prompt_model': 'gpt-4.1',
            'competition_mode': True,
            'competition_text': '',
            'selected_panel': PANEL_OPTIONS[0]['name'],
            'custom_panel': PANEL_OPTIONS[0]['prompt'],
            'selected_personality': PERSONALITY_OPTIONS[0]['name'],
            'custom_personality': PERSONALITY_OPTIONS[0]['prompt'],
            'num_best_pairs': 3,
            'prompt_input': '',
            'creativity_spectrum': None,
            'style_axes': None,
            'input': '',
            'progress_step': 0,
            'reasoning_level': 'medium',  # default reasoning if user doesn't set
        }

        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value

def main():
    st.set_page_config(
        page_title="Lofn - The AI Artist",
        page_icon=":art:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.title("LOFN - The AI Artist")

    if 'app' not in st.session_state:
        st.session_state['app'] = LofnApp()

    app = st.session_state['app']
    app.run()

if __name__ == "__main__":
    main()
