# ui.py

import streamlit as st
from config import Config
from helpers import *
from llm_integration import (
    read_prompt,
    get_llm,
    generate_concept_mediums,
    generate_prompts,
    fetch_openrouter_models
)
from image_generation import (
    render_image_controls,
    get_model_params,
    generate_dalle_images,
)
import logging

logger = logging.getLogger(__name__)
import os
import random
import pandas as pd


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
        self.llm_client = None
        self.image_generator = None

        # Build the model list dynamically
        self.available_models = self.get_available_models()

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
                "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
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
                "Poe-Google-PaLM", "Poe-Llama-3-8b-Groq", "Poe-Llama-3-8B-T", "Poe-Gemma-Instruct-7B-T",
                "Poe-MythoMax-L2-13B", "Poe-Code-Llama-34b", "Poe-Code-Llama-13b", "Poe-Solar-Mini",
                "Poe-GPT-3.5-Turbo-Instruct", "Poe-GPT-3.5-Turbo-Raw", "Poe-Claude-instant",
                "Poe-Mixtral-8x7b-Groq", "Poe-Mistral-7B-v0.3-T", "Poe-Llama-3-70b-Groq", "Poe-Gemma-2-27b-T",
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
                st.error("An error occurred while get OpenRouter models.")
                logger.exception("Error getting OpenRouter models: %s", e)

        # Add OpenRouter models if OPEN_ROUTER_API_KEY is available
        # if Config.OPEN_ROUTER_API_KEY:
        #     # For simplicity, let's add a few example models; you can extend this list
        #     or_models  = [
        #         "OR-meta-llama/llama-3.2-3b-instruct",  "OR-meta-llama/llama-3.2-1b-instruct", 
        #          "OR-meta-llama/llama-3.2-90b-vision-instruct",  "OR-meta-llama/llama-3.2-11b-vision-instruct",  
        #          "OR-meta-llama/llama-3.1-405b",  "OR-meta-llama/llama-3.1-405b-instruct",  
        #          "OR-meta-llama/llama-3.1-70b-instruct",  "OR-meta-llama/llama-3.1-8b-instruct",  
        #          "OR-meta-llama/llama-guard-2-8b",  "OR-meta-llama/llama-3-70b-instruct",  
        #          "OR-meta-llama/llama-3-8b-instruct",  "OR-openai/o1-mini-2024-09-12",  
        #          "OR-openai/o1-mini",  "OR-openai/o1-preview-2024-09-12",  "OR-openai/o1-preview",  
        #          "OR-openai/gpt-4o-mini-2024-07-18",  "OR-openai/gpt-4o-mini",  "OR-openai/gpt-4o-2024-08-06",  
        #          "OR-openai/chatgpt-4o-latest",  "OR-openai/gpt-4",  "OR-openai/gpt-4-32k",  "OR-openai/gpt-4-turbo",  
        #          "OR-openai/gpt-3.5-turbo",  "OR-openai/gpt-3.5-turbo-16k",  "OR-anthropic/claude-3.5-sonnet",  
        #          "OR-anthropic/claude-3.5-sonnet",  "OR-anthropic/claude-3-haiku",  
        #          "OR-anthropic/claude-3-sonnet", "OR-anthropic/claude-3-opus", "OR-anthropic/claude-instant-1.1",
        #           "OR-anthropic/claude-1.2",  "OR-anthropic/claude-1",  "OR-google/gemini-pro-1.5-exp",  
        #           "OR-google/gemini-flash-8b-1.5-exp",  "OR-google/gemini-flash-1.5-exp",  
        #           "OR-google/gemini-pro-1.5",  "OR-google/gemini-pro-1.5-exp",  "OR-google/gemini-pro-vision",  
        #           "OR-google/gemini-pro",  "OR-google/gemini-flash-1.5",  "OR-google/gemma-2-27b-it",  
        #           "OR-google/gemma-2-9b-it",  "OR-qwen/qwen-2.5-72b-instruct",  "OR-qwen/qwen-2-vl-72b-instruct",  
        #           "OR-qwen/qwen-2-vl-7b-instruct",  "OR-qwen/qwen-2-7b-instruct",  "OR-qwen/qwen-72b-chat",  
        #           "OR-qwen/qwen-110b-chat",  "OR-qwen/qwen-2-72b-instruct",  "OR-qwen/qwen-2-7b-instruct",  
        #           "OR-cohere/command-r-03-2024",  "OR-cohere/command-r-plus-04-2024",  "OR-cohere/command-r-plus-08-2024", 
        #           "OR-cohere/command-r-08-2024",  "OR-cohere/command-r-plus",  "OR-cohere/command-r",  "OR-cohere/command",  
        #           "OR-microsoft/phi-3.5-mini-128k-instruct",  "OR-microsoft/phi-3-mini-128k-instruct",  
        #           "OR-microsoft/phi-3-mini-128k-instruct",  "OR-microsoft/phi-3-medium-128k-instruct",  
        #           "OR-microsoft/phi-3-medium-128k-instruct",  "OR-microsoft/wizardlm-2-7b",  "OR-microsoft/wizardlm-2-8x22b",  
        #           "OR-ai21/jamba-1-5-large",  "OR-ai21/jamba-1-5-mini",  "OR-ai21/jamba-instruct",  
        #           "OR-perplexity/llama-3.1-sonar-huge-128k-online",  "OR-perplexity/llama-3.1-sonar-large-128k-online",  
        #           "OR-perplexity/llama-3.1-sonar-large-128k-chat",  "OR-perplexity/llama-3.1-sonar-small-128k-online",  
        #           "OR-perplexity/llama-3.1-sonar-small-128k-chat",  "OR-perplexity/llama-3-sonar-large-32k-online",  
        #           "OR-perplexity/llama-3-sonar-large-32k-chat",  "OR-perplexity/llama-3-sonar-small-32k-online",  
        #           "OR-perplexity/llama-3-sonar-small-32k-chat",  "OR-mistralai/pixtral-12b",  
        #           "OR-mistralai/mistral-7b-instruct-v0.3",  "OR-mistralai/mistral-7b-instruct-v0.2",  
        #           "OR-mistralai/mistral-7b-instruct", 
        #           "OR-mistralai/mixtral-8x22b-instruct",  "OR-mistralai/mixtral-8x7b-instruct",  
        #           "OR-mistralai/mixtral-8x7b",  "OR-mistralai/mistral-nemo", 
        #           "OR-mistralai/mistral-large",  "OR-mistralai/mistral-medium",  
        #           "OR-mistralai/mistral-small",  "OR-mistralai/mistral-tiny"]
        #     models.extend(or_models)
        return models

    def get_available_image_models(self):
        models = []
        if Config.FAL_API_KEY:
            models.extend([
                "fal-ai/flux-pro/v1.1", "fal-ai/flux-pro", "fal-ai/flux-realism", "fal-ai/flux-dev", "fal-ai/flux/schnell",
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
                "Poe-FLUX-pro-1.1", "Poe-FLUX-pro", "Poe-DALL-E-3", "Poe-Ideogram-v2", "Poe-Playground-v2.5", 
                "Poe-Playground-v3", "Poe-Ideogram", "Poe-FLUX-dev",
                "Poe-FLUX-schnell", "Poe-LivePortrait", "Poe-StableDiffusion3",
                "Poe-SD3-Turbo", "Poe-StableDiffusionXL", "Poe-StableDiffusion3-2B",
                "Poe-SD3-Medium", "Poe-RealVisXL"
            ])
        return models

    def run(self):
        # Initialize session state
        initialize_session_state()

        st.title("LOFN - The AI Artist")

        # Sidebar and UI elements
        self.render_sidebar()

        # Main container logic
        self.render_main_container()

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
                style_axes = {
                    "Abstraction vs. Realism": st.slider("Abstraction vs. Realism (0: Abstract)", 0, 100, 50),
                    "Emotional Valence": st.slider("Emotional Valence (0: Negative)", 0, 100, 50),
                    "Color Intensity": st.slider("Color Intensity (0: Muted)", 0, 100, 50),
                    "Symbolic Density": st.slider("Symbolic Density (0: Literal)", 0, 100, 50),
                    "Compositional Complexity": st.slider("Compositional Complexity (0: Simple)", 0, 100, 50),
                    "Textural Richness": st.slider("Textural Richness (0: Smooth)", 0, 100, 50),
                    "Symmetry vs. Asymmetry": st.slider("Symmetry vs. Asymmetry (0: Asymmetrical)", 0, 100, 50),
                    "Novelty": st.slider("Novelty (0: Traditional)", 0, 100, 50),
                    "Figure-Ground Relationship": st.slider("Figure-Ground Relationship (0: Distinct)", 0, 100, 50),
                    "Dynamic vs. Static": st.slider("Dynamic vs. Static (0: Static)", 0, 100, 50)
                }
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
                        placeholder=webhook_url_placeholder,
                        help="Provide the webhook URL for your Discord channel.",
                    )
                st.session_state['webhook_url'] = webhook_url
           

    def render_main_container(self):
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
                self.generate_concepts()
                


        # Display concepts and proceed to prompt generation
        if 'concept_mediums' in st.session_state and st.session_state['concept_mediums']:
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
            # Store the updated axes back into session state if needed
            st.session_state['style_axes'] = style_axes
            st.session_state['creativity_spectrum'] = creativity_spectrum
        except Exception as e:
            st.error("An error occurred while generating concepts.")
            logger.exception("Error generating concepts: %s", e)

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
                prompts_df = generate_prompts(
                    st.session_state['input'],
                    pair['concept'],
                    pair['medium'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state.get('style_axes', None),
                    creativity_spectrum=st.session_state.get('creativity_spectrum', None),
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
                prompts_df = generate_prompts(
                    st.session_state['input'],
                    pair['concept'],
                    pair['medium'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state.get('style_axes', None),
                    creativity_spectrum=st.session_state.get('creativity_spectrum', None),
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
            st.markdown(f"**Prompt {idx+1}:** {row['Revised Prompts']}")
            st.markdown(f"**Synthesized Prompt {idx+1}:** {row['Synthesized Prompts']}")
            st.markdown("---")

        self.generate_images(prompts_df, pair)

    def generate_images(self, prompts_df, pair):
        st.subheader(f"Generating Images for '{pair['concept']}'")
        try:
            params = get_model_params(self.image_model)
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
                    style_axes=st.session_state.get('style_axes', None),
                    creativity_spectrum=st.session_state.get('creativity_spectrum', None),
                )
            st.success(f"Images generated for '{pair['concept']}'")
        except Exception as e:
            st.error(f"An error occurred while generating images for '{pair['concept']}'.")
            logger.exception("Error generating images: %s", e)

    # Additional methods can be added here for further functionality

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
    else:
        logger.warning("style.css not found. Proceeding without custom styles.")

    if 'app' not in st.session_state:
        st.session_state['app'] = LofnApp()

    app = st.session_state['app']
    app.run()

if __name__ == "__main__":
    main()
