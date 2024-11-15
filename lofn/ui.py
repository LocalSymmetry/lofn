# ui.py

import os
import logging
import random
import pandas as pd
import streamlit as st
from image_generation import (
    render_image_controls,
    get_model_params,
    generate_dalle_images,
)
from config import Config
from helpers import *
from llm_integration import *

logger = logging.getLogger(__name__)

class LofnError(Exception):
    """Custom exception class for Lofn-specific errors."""
    pass

class LofnApp:
    def __init__(self):
        self.model = None
        self.image_model = None
        self.temperature = 0.7
        self.max_retries = 3
        self.debug = False

        # Build the model list dynamically
        self.available_models = self.get_available_models()
        self.available_image_models = self.get_available_image_models()
        self.initialize_session_state()

    def get_available_models(self):
        models = []

        # Add OpenAI models if OPENAI_API is available
        if Config.OPENAI_API:
            models.extend([
                "gpt-4o-mini", "gpt-4o", "o1-preview", "o1-mini",
                "gpt-4o-2024-08-06", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"
            ])
        # Add Anthropic models if ANTHROPIC_API is available
        if Config.ANTHROPIC_API:
            models.extend([
                "claude-3-5-sonnet-latest", "claude-3-5-haiku-20241022",
                "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
                "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
            ])
        # Add Google models if GOOGLE_API is available
        if Config.GOOGLE_API:
            models.extend([
                "gemini-1.5-flash-002", "gemini-1.5-pro-002", 
                "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro",
                "gemini-1.5-pro-exp-0827", "gemini-1.5-pro-exp-0801"
            ])
        # Add Poe models if POE_API is available
        if Config.POE_API:
            models.extend([
                "Poe-o1-preview-128k", "Poe-o1-mini-128k", "Poe-Gemini-1.5-Pro-128k", 
                "Poe-Llama-3.1-405B-FW-128k", "Poe-Gemini-1.5-Flash-128k", 
                "Poe-GPT-4o-Mini-128k", "Poe-GPT-4o-128k", "Poe-Claude-3.5-Sonnet-200k",
                "Poe-Mistral-Large-2-128k", "Poe-Llama-3.2-11B-FW-131k", "Poe-Llama-3.2-90B-FW-131k",
                "Poe-Llama-3.1-8B-T-128k", "Poe-Llama-3.1-70B-FW-128k", "Poe-Llama-3.1-70B-T-128k",
                "Poe-Llama-3.1-8B-FW-128k", "Poe-GPT-4-Turbo-128k", "Poe-Claude-3-Opus-200k",
                "Poe-Claude-3-Sonnet-200k", "Poe-Claude-3-Haiku-200k",
                "Poe-Mixtral8x22b-Inst-FW", "Poe-Command-R", "Poe-Gemma-2-9b-T", 
                "Poe-Mistral-Large-2", "Poe-Mistral-Medium",
                "Poe-Snowflake-Arctic-T", "Poe-RekaCore", "Poe-RekaFlash", "Poe-Command-R-Plus",
                "Poe-GPT-3.5-Turbo", "Poe-Mixtral-8x7B-Chat", "Poe-DeepSeek-Coder-33B-T",
                "Poe-CodeLlama-70B-T", "Poe-Qwen2-72B-Chat", "Poe-Qwen-72B-T", "Poe-Claude-2",
                "Poe-Google-PaLM", "Poe-Llama-3-8b-Groq", "Poe-Llama-3-8B-T", "Poe-Gemma-2-27b-T",
                "Poe-Assistant", "Poe-Claude-3.5-Sonnet", "Poe-GPT-4o-Mini", "Poe-GPT-4o",
                "Poe-Llama-3.1-405B-T", "Poe-Gemini-1.5-Flash", "Poe-Gemini-1.5-Pro",
                "Poe-Claude-3-Sonnet", "Poe-Claude-3-Haiku", "Poe-Claude-3-Opus",
                "Poe-Gemini-1.0-Pro", "Poe-Llama-3-70B-T", "Poe-Llama-3-70b-Inst-FW", "Poe-Llama-3.2-90B-FW-131k",
                "Poe-Llama-3.2-11B-FW-131k"
            ])

        # Fetch models from OpenRouter API if API key is available
        if Config.OPEN_ROUTER_API_KEY:
            try:
                or_models = fetch_openrouter_models()
                if or_models:
                    # Filter models based on context length and response tokens
                    filtered_or_models = filter_models_by_context_length(or_models, min_total_tokens=25000, min_response_tokens=15000)
                    # Extract model IDs for selection
                    models.extend(['OR-'+model['id'] for model in filtered_or_models])
                else:
                    print("No OpenRouter API key found. Skipping OpenRouter models.")
            except Exception as e:
                st.error("An error occurred while getting OpenRouter models.")
                logger.exception("Error getting OpenRouter models: %s", e)

        return models

    def get_available_image_models(self):
        models = []
        if Config.FAL_API_KEY:
            models.extend([
                "fal-ai/flux-pro/v1.1-ultra", "fal-ai/flux-pro/v1.1", "fal-ai/recraft-v3", "fal-ai/omnigen-v1", "fal-ai/stable-diffusion-v35-large", "fal-ai/stable-diffusion-v35-medium", "fal-ai/flux-pro", "fal-ai/flux-realism", "fal-ai/flux-dev", "fal-ai/flux/schnell",
                # Add other FAL models
            ])
        if Config.IDEOGRAM_API_KEY:
            models.extend([
                "Ideogram",
                # Add other Ideogram models
            ])
        if Config.GOOGLE_PROJECT_ID:
            models.append("Google Imagen 3")
        if Config.OPENAI_API:
            models.extend([
                "DALL-E 3",
                # Add other models
            ])
        if Config.POE_API:
            models.extend([
                "Poe-FLUX-pro-1.1", "Poe-Imagen3", "Poe-StableDiffusion3.5-L",
                "Poe-FLUX-pro", "Poe-DALL-E-3", "Poe-Ideogram-v2", "Poe-Playground-v2.5", 
                "Poe-Playground-v3", "Poe-Ideogram", "Poe-FLUX-dev",
                "Poe-FLUX-schnell", "Poe-LivePortrait", "Poe-StableDiffusion3",
                "Poe-SD3-Turbo", "Poe-StableDiffusionXL", "Poe-StableDiffusion3-2B",
                "Poe-SD3-Medium", "Poe-RealVisXL"
            ])
        return models

    def get_available_video_models(self):
        models = []
        if Config.RUNWAYML_API_KEY:
            models.extend([
                "RunwayML Gen-2",
                # Add other video models if available
            ])
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

        # Group settings under expandable sections
        with st.sidebar.expander("Language Model Settings", expanded=True):
            self.model = st.selectbox(
                "Select Language Model",
                self.available_models,
                help="Choose the language model for generating concepts and prompts.",
            )

            self.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Controls the randomness of the model's output.",
            )

            self.max_retries = st.slider(
                "Maximum Retries",
                min_value=1,
                max_value=10,
                value=3,
                help="Set the maximum number of retries for LLM responses.",
            )

            self.debug = st.checkbox("Debug Mode", value=False, help="Enable debug mode for detailed logs.")

        with st.sidebar.expander("Image Generation Settings", expanded=True):
            self.available_image_models = self.get_available_image_models()

            self.image_model = st.selectbox(
                "Select Image Model",
                self.available_image_models,
                help="Choose the image generation model.",
            )

            render_image_controls(self.image_model)


        with st.sidebar.expander("Style Personalization", expanded=False):
            st.session_state['auto_style'] = st.checkbox("Automatic Style", value=True, help="Enable automatic style determination.")
            if not st.session_state['auto_style']:
                st.subheader("Adjust Style Axes")
                selected_medium = st.session_state.get('selected_tab', 'Image Generation')
                if selected_medium == 'Image Generation':
                    # Image Style Axes
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
                    # Video Style Axes
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
                    # Music Style Axes
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
                else:
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
        # Main content area
        st.header("Generate Your Art Concept")

        # User input section
        st.subheader("Describe Your Idea")
        user_input = st.text_area(
            "",
            placeholder="Describe the essence of the art you wish to generate.",
            help="Provide a detailed description of your idea to get the best results.",
        )
        st.session_state['input'] = user_input

        # Provide an example or guidance
        if not user_input:
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
        if st.button("Generate Concepts"):
            if user_input.strip() == "":
                st.warning("Please provide a description of your idea.")
            else:
                style_axes, creativity_spectrum = self.generate_concepts()

        # Display concepts and proceed to prompt generation
        if 'concept_mediums' in st.session_state and st.session_state['concept_mediums']:
            # self.display_style_axes_and_creativity_spectrum()
            self.display_concepts()

    def generate_concepts(self):
        try:
            st.session_state['concept_mediums'] = []
            with st.spinner("Generating concepts..."):
                concepts, style_axes, creativity_spectrum = generate_concept_mediums(
                    st.session_state['input'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state.get('style_axes', None),
                    creativity_spectrum=st.session_state.get('creativity_spectrum', None),
                )
            st.session_state['concept_mediums'] = concepts
            st.success("Concepts generated successfully!")
            # Store the updated axes back into session state
            st.session_state['style_axes'] = style_axes
            st.session_state['creativity_spectrum'] = creativity_spectrum

            return style_axes, creativity_spectrum
        except Exception as e:
            st.error("An error occurred while generating concepts.")
            logger.exception("Error generating concepts: %s", e)

    def display_style_axes_and_creativity_spectrum(self):
        if 'style_axes' in st.session_state and st.session_state['style_axes']:
            display_style_axes(st.session_state['style_axes'])
        if 'creativity_spectrum' in st.session_state and st.session_state['creativity_spectrum']:
            display_creativity_spectrum(st.session_state['creativity_spectrum'])

    def display_concepts(self):
        st.subheader("Generated Concepts and Mediums")

        # Display concepts in a mini-dashboard
        selected_pairs = create_mini_dashboard(st.session_state['concept_mediums'])

        # Store selected pairs in session state
        st.session_state['selected_pairs'] = selected_pairs

        # Proceed to generate prompts for selected concepts
        if st.button("Generate Image Prompts for Selected Concepts"):
            if selected_pairs:
                for idx in selected_pairs:
                    pair = st.session_state['concept_mediums'][idx]
                    self.generate_prompts_for_pair(pair)
            else:
                st.warning("Please select at least one concept to generate prompts.")

        # Option to generate prompts for all concepts
        if st.button("Generate Image Prompts for All Concepts"):
            for pair in st.session_state['concept_mediums']:
                self.generate_prompts_for_pair(pair)

    def generate_prompts_for_pair(self, pair):
        st.subheader(f"Generating Prompts for '{pair['concept']}' in '{pair['medium']}'")
        try:
            with st.spinner(f"Generating prompts for '{pair['concept']}'..."):
                prompts_df = generate_image_prompts(
                    st.session_state['input'],
                    pair['concept'],
                    pair['medium'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state['style_axes'],
                    creativity_spectrum=st.session_state['creativity_spectrum'],
                )
            st.session_state['prompts_df'] = prompts_df
            st.success(f"Prompts generated for '{pair['concept']}'")
            self.display_prompts(prompts_df, pair)
        except Exception as e:
            st.error(f"An error occurred while generating prompts for '{pair['concept']}'.")
            logger.exception("Error generating prompts: %s", e)

    def generate_prompts_for_manual_input(self, concept, medium):
        st.subheader(f"Generating Prompts for '{concept}' in '{medium}'")
        try:
            with st.spinner(f"Generating prompts for '{concept}'..."):
                prompts_df = generate_image_prompts(
                    st.session_state['input'],
                    concept,
                    medium,
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state['style_axes'],
                    creativity_spectrum=st.session_state['creativity_spectrum'],
                )
            st.session_state['prompts_df'] = prompts_df
            st.success(f"Prompts generated for '{concept}'")
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

    def generate_images(self, prompts_df, pair):
        st.subheader(f"Generating Images for '{pair['concept']}'")
        try:
            # Placeholder for model parameters
            params = {}  # get_model_params(self.image_model)
            with st.spinner(f"Generating images for '{pair['concept']}'..."):
                generate_dalle_images(
                    st.session_state['input'],
                    pair['concept'],
                    pair['medium'],
                    prompts_df,
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    image_model=self.image_model,
                    style_axes=st.session_state['style_axes'],
                    creativity_spectrum=st.session_state['creativity_spectrum'],
                    OPENAI_API=Config.OPENAI_API
                )
            st.success(f"Images generated for '{pair['concept']}'")
        except Exception as e:
            st.error(f"An error occurred while generating images for '{pair['concept']}'.")
            logger.exception("Error generating images: %s", e)

    def render_video_generation(self):
        st.header("Generate Your Art Video Concept")

        # User input section
        st.subheader("Describe Your Idea")
        user_input = st.text_area(
            "",
            placeholder="Describe the essence of the art video you wish to generate.",
            help="Provide a detailed description of your idea to get the best results.",
        )
        st.session_state['input'] = user_input

        # Guidance for the user
        if not user_input:
            st.info("Tip: Describe a scene or narrative you'd like to see in motion. For example, 'A powerful and mysterious witch's familiar completing a dangerous and important quest.'")

        # Process Flow Control
        if st.button("Generate Video Concepts"):
            if user_input.strip() == "":
                st.warning("Please provide a description of your idea.")
            else:
                self.generate_ui_video_concepts()

        # Display concepts and proceed to prompt generation
        if 'video_concept_mediums' in st.session_state and st.session_state['video_concept_mediums']:
        #     self.display_style_axes_and_creativity_spectrum()
            self.display_video_concepts()

    def render_video_sidebar(self):
        st.sidebar.header('Video Generation Settings')

        # Video Model Selection
        available_video_models = self.get_available_video_models()
        video_model = st.sidebar.selectbox(
            "Select Video Model",
            available_video_models,
            help="Choose the video generation model.",
        )

        # Video Duration
        video_duration = st.sidebar.slider(
            "Video Duration (seconds)",
            min_value=5,
            max_value=60,
            value=10,
            help="Set the duration of the generated video.",
        )

        # Video Resolution
        video_resolution = st.sidebar.selectbox(
            "Video Resolution",
            ["480p", "720p", "1080p"],
            help="Select the resolution of the video.",
        )

        # Save settings to session state
        st.session_state['video_model'] = video_model
        st.session_state['video_duration'] = video_duration
        st.session_state['video_resolution'] = video_resolution

    def render_video_main_container(self):
        st.header("Generate Your Art Video Concept")

        # User input section
        st.subheader("Describe Your Idea")
        user_input = st.text_area(
            "",
            placeholder="Describe the essence of the art video you wish to generate.",
            help="Provide a detailed description of your idea to get the best results.",
        )
        st.session_state['input'] = user_input

        # Provide an example or guidance
        if not user_input:
            st.info("Tip: Describe a scene or narrative you'd like to see in motion. For example, 'A powerful and mysterious witch's familiar completing a dangerous and important quest.'")

        # Process Flow Control
        if st.button("Generate Video Concepts"):
            if user_input.strip() == "":
                st.warning("Please provide a description of your idea.")
            else:
                style_axes, creativity_spectrum = self.generate_ui_video_concepts()

        # # Display concepts and proceed to prompt generation
        # if 'video_concept_mediums' in st.session_state and st.session_state['video_concept_mediums']:
        #     self.display_style_axes_and_creativity_spectrum()
        #     self.display_video_concepts()

    def generate_ui_video_concepts(self):
        try:
            st.session_state['video_concept_mediums'] = []
            with st.spinner("Generating video concepts..."):
                concepts, style_axes, creativity_spectrum = generate_video_concept_mediums(
                    st.session_state['input'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state.get('style_axes', None),
                    creativity_spectrum=st.session_state.get('creativity_spectrum', None),
                )
            st.session_state['video_concept_mediums'] = concepts
            st.success("Video concepts generated successfully!")
            st.session_state['style_axes'] = style_axes
            st.session_state['creativity_spectrum'] = creativity_spectrum
        except Exception as e:
            st.error("An error occurred while generating video concepts.")
            logging.exception("Error generating video concepts: %s", e)

    def display_video_concepts(self):
        st.subheader("Generated Video Concepts and Mediums")

        # Display concepts in a mini-dashboard
        selected_pairs = create_mini_dashboard(st.session_state['video_concept_mediums'])

        # Store selected pairs in session state
        st.session_state['selected_video_pairs'] = selected_pairs

        # Proceed to generate prompts for selected concepts
        if st.button("Generate Video Prompts for Selected Concepts"):
            if selected_pairs:
                for idx in selected_pairs:
                    pair = st.session_state['video_concept_mediums'][idx]
                    self.generate_video_prompts_for_pair(pair)
            else:
                st.warning("Please select at least one concept to generate prompts.")

        # Option to generate prompts for all concepts
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
                )
            st.session_state['video_prompts_df'] = prompts_df
            st.success(f"Video prompts generated for '{pair['concept']}'")
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
        # Proceed to generate videos if applicable
        # self.generate_videos(prompts_df, pair)

    def generate_videos(self, prompts_df, pair):
        st.subheader(f"Generating Videos for '{pair['concept']}'")
        try:
            with st.spinner(f"Generating videos for '{pair['concept']}'..."):
                for idx, row in prompts_df.iterrows():
                    video_prompt = row['Video Prompts']
                    # video_url = generate_runway_video(
                    #     video_prompt,
                    #     Config.RUNWAYML_API_KEY,
                    #     st.session_state['video_duration'],
                    #     st.session_state['video_resolution']
                    # )
                    # if video_url:
                    #     st.markdown(f"**Video {idx+1}:**")
                    #     st.video(video_url)
                    # else:
                    #     st.warning(f"Video {idx+1} could not be generated.")
            st.success(f"Videos generated for '{pair['concept']}'")
        except Exception as e:
            st.error(f"An error occurred while generating videos for '{pair['concept']}'.")
            logger.exception("Error generating videos: %s", e)

    def render_music_generation(self):
        self.render_music_sidebar()
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
            st.info("Tip: Include themes, emotions, specific elements, and desired run time length. For example, 'I want a folksy metal song about not being able to find your tire pressure gauge.'")

        # Process Flow Control
        if st.button("Generate Music Prompts"):
            if user_input.strip() == "":
                st.warning("Please provide a description of your song idea.")
            else:
                self.generate_music_prompts_ui()

        # Display generated prompts
        if 'music_prompt' != None and 'lyrics_prompt' != None:
            self.display_music_prompts()

    def render_music_sidebar(self):
        st.sidebar.header('Music Generation Settings')

        # Desired Run Time Length
        run_time = st.sidebar.number_input(
            "Desired Song Length (minutes)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Set the desired length of the song.",
        )
        st.session_state['run_time'] = run_time

    def render_music_main_container(self):
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
                self.generate_music_prompts_ui()

        # Display generated prompts
        if 'music_prompt' in st.session_state and 'lyrics_prompt' in st.session_state:
            self.display_music_prompts()

    def generate_music_prompts_ui(self):
        try:
            with st.spinner("Generating music prompts..."):
                music_prompt, lyrics_prompt = generate_music_prompts(
                    st.session_state['input'],
                    st.session_state['run_time'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                )
            st.session_state['music_prompt'] = music_prompt
            st.session_state['lyrics_prompt'] = lyrics_prompt
            st.success("Music prompts generated successfully!")
            self.display_music_prompts()
        except Exception as e:
            st.error("An error occurred while generating music prompts.")
            logging.exception("Error generating music prompts: %s", e)

    def display_music_prompts(self):
        st.subheader("Generated Music Prompt")
        st.code(st.session_state['music_prompt'], language='text')

        st.subheader("Generated Lyrics Prompt")
        st.code(st.session_state['lyrics_prompt'], language='text')

        st.info("Copy the above prompts and paste them into Udio to generate your music.")

    def initialize_session_state(self):
        default_values = {
            'run_time': '3.0',
            'selected_tab': 'Image Generation',
            'video_concept_mediums': None,
            'video_prompts_df': None,
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
            'image_model':'Poe-FLUX-pro',
            'creativity_spectrum': None,
            'style_axes': None,
            'input': '',
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

    # Load custom CSS if available
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
