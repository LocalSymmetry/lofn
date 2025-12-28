import os
import logging
import random
import json
import pandas as pd
import streamlit as st
import yaml
import base64

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
from langchain.schema import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

PANEL_OPTIONS = yaml.safe_load(read_prompt('/lofn/prompts/panels.yaml'))

PANEL_OPTIONS = [{'name': 'LLM Generated', 'prompt': ''}] + PANEL_OPTIONS

PERSONALITY_OPTIONS = yaml.safe_load(read_prompt('/lofn/prompts/personalities.yaml'))

# Merge in user-defined personalities if the optional file exists
custom_personality_path = '/lofn/prompts/custom_personalities.yaml'
# Check if file exists using the same logic as read_prompt indirectly or just try/except
try:
    custom_content = read_prompt(custom_personality_path)
    custom_personalities = yaml.safe_load(custom_content) or []
    PERSONALITY_OPTIONS[:0] = custom_personalities
except FileNotFoundError:
    pass

PERSONALITY_OPTIONS = [{'name': 'LLM Generated', 'prompt': ''}] + PERSONALITY_OPTIONS


def image_context_to_string(images):
    """Utility to inline prepared images as data URLs separated by newlines."""

    return "\n".join(prepare_image_strings(images)) if images else ""

DEFAULT_MODEL_CONFIG_PATH = '/lofn/model_defaults.yaml'

# Limit the number of prompt files processed by the explorer
MAX_PROMPT_FILES = 2000


def load_model_defaults():
    """Load default model priority lists from YAML."""
    try:
        content = read_prompt(DEFAULT_MODEL_CONFIG_PATH)
        return yaml.safe_load(content) or {}
    except Exception as e:
        logger.warning("Could not load model defaults: %s", e)
        return {}


def build_prompt_index(base_path: str):
    """Load all prompt metadata files and build a searchable index.

    For directories with thousands of files, repeatedly parsing every JSON
    file is slow.  To speed up lookups we maintain an on-disk cache
    (``.index_cache.json``) that stores the parsed metadata along with each
    file's modification time.  Subsequent calls only read files that are new
    or have changed since the last cache build. Only the most recent
    ``MAX_PROMPT_FILES`` entries are scanned to keep lookups responsive.
    """

    index = {}
    if not os.path.exists(base_path):
        return index

    cache_path = os.path.join(base_path, ".index_cache.json")
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
    except Exception:
        cache = {}

    # Scan the directory for current JSON files, prioritizing most recent ones
    files = [
        e for e in os.scandir(base_path)
        if e.is_file() and e.name.endswith(".json")
    ]
    files.sort(key=lambda e: e.stat().st_mtime, reverse=True)
    entries = {e.name: e for e in files[:MAX_PROMPT_FILES]}

    updated = False

    # Add or refresh cache entries when files are new or modified
    for name, entry in entries.items():
        mtime = entry.stat().st_mtime
        cached = cache.get(name)
        if not cached or cached.get("mtime") != mtime:
            try:
                with open(entry.path, "r") as meta_file:
                    meta = json.load(meta_file)
            except Exception:
                continue
            haystack = " ".join(
                [
                    name,
                    str(meta.get("title", "")),
                    str(meta.get("prompt", "")),
                    str(meta.get("concept", "")),
                    str(meta.get("medium", "")),
                ]
            ).lower()
            cache[name] = {"meta": meta, "haystack": haystack, "mtime": mtime}
            updated = True

    # Remove cache entries for files that no longer exist
    missing = [name for name in list(cache.keys()) if name not in entries]
    for name in missing:
        del cache[name]
        updated = True

    if updated:
        try:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
        except Exception:
            pass

    # Return the index without the modification time field
    for name, data in cache.items():
        index[name] = {"meta": data["meta"], "haystack": data["haystack"]}

    return index

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
        self.model_priority = load_model_defaults()
        self.prioritize_models_for_mode('Image Generation')
        self.initialize_session_state()

    def get_available_models(self):
        models = []

        # Add OpenAI-based models if OPENAI_API is available
        if Config.OPENAI_API:
            models.extend([
                "gpt-5", "gpt-5-mini", "gpt-5-nano",
                "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
                "o3", "o3-pro", "o4-mini",
            ])
        # Add Anthropic models if ANTHROPIC_API is available
        if Config.ANTHROPIC_API:
            models.extend([
                "claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514", "claude-opus-4-20250514",
                "claude-3-5-sonnet-latest", "claude-3-5-haiku-20241022",
                "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620",
                "claude-3-opus-20240229", "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ])
        # Add Google models if GOOGLE_API_KEY is available
        if Config.GOOGLE_API_KEY:
            models.extend([
                "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash-lite-preview",
                "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash-preview",
                "gemini-1.5-pro", "gemini-1.5-flash",
            ])
        # Add Poe models if POE_API is available
        if Config.POE_API:
            models.extend([
                "Poe-Assistant", "Poe-App-Creator",
                "Poe-GPT-5", "Poe-GPT-5-mini", "Poe-GPT-5-nano",
                "Poe-GPT-4o", "Poe-GPT-4.1", "Poe-GPT-4.1-mini", "Poe-GPT-4.1-nano",
                "Poe-o3", "Poe-o3-pro", "Poe-o4-mini",
                "Poe-Claude-Opus-4.1", "Poe-Claude-Sonnet-4",
                "Poe-Gemini-2.5-Pro", "Poe-Gemini-2.5-Flash", "Poe-Gemini-2.5-Flash-Lite", "Poe-Gemini-2.5-Flash-Lite-Preview",
                "Poe-Gemini-2.0-Flash", "Poe-Gemini-2.0-Flash-Lite", "Poe-Gemini-2.0-Flash-Preview",
                "Poe-Gemini-1.5-Pro", "Poe-Gemini-1.5-Flash",
                "Poe-Grok-4", "Poe-Grok-3", "Poe-Grok-3-Mini",
                "Poe-GPT-OSS-120B-T",
                "Poe-DeepSeek-V3", "Poe-Deepseek-V3-FW", "Poe-Deepseek-R1",
                "Poe-Qwen2-72B-Chat", "Poe-Qwen2.5-VL-72B-T", "Poe-Qwen2.5-Coder-32B"
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
                    print("No OpenRouter models returned. Skipping OpenRouter models.")
            except Exception as e:
                st.error("An error occurred while getting OpenRouter models.")
                logger.exception("Error getting OpenRouter models: %s", e)

        # Prioritize the most powerful models if available
        priority_order = [
            "gemini-2.5-pro",
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
        if Config.GOOGLE_API_KEY:
            models.append("Gemini 2.5 Flash Image")
        if Config.GCP_PROJECT_ID:
            models.append("Google Imagen 3")
        if Config.OPENAI_API:
            models.append("DALL-E 3")
        if Config.POE_API:
            models.extend([
                "Poe-GPT-Image-1",
                "Poe-Imagen-4-Ultra-Exp", "Poe-Imagen-4", "Poe-Imagen-4-Fast",
                "Poe-Flux-Kontext-Max", "Poe-Flux-Kontext-Pro",
                "Poe-Seedream-3.0", "Poe-Phoenix-1.0",
                "Poe-FLUX-pro-1.1-ultra", "Poe-FLUX-pro-1.1",
                "Poe-Imagen-3", "Poe-Imagen-3-Fast",
                "Poe-Gemini-2.0-Flash-Preview",
                "Poe-Ideogram-v3", "Poe-Ideogram-v2a", "Poe-StableDiffusion3.5-L",
                "Poe-FLUX-pro", "Poe-DALL-E-3", "Poe-Ideogram-v2",
                "Poe-Playground-v2.5", "Poe-Playground-v3", "Poe-Ideogram",
                "Poe-FLUX-dev", "Poe-FLUX-schnell", "Poe-LivePortrait", "Poe-StableDiffusion3",
                "Poe-SD3-Turbo", "Poe-StableDiffusionXL", "Poe-StableDiffusion3-2B",
                "Poe-SD3-Medium", "Poe-RealVisXL"
            ])
        return models

    def get_available_video_models(self):
        models = []
        if Config.RUNWAYML_API_KEY:
            models.append("RunwayML Gen-2")
        else:
            st.warning("No RunwayML API key found. Video generation models will not be available.")
        return models

    def prioritize_models_for_mode(self, mode_name: str):
        """Place default models for the mode at the start of available_models."""
        mapping = {
            'Image Generation': 'art',
            'Video Generation': 'video',
            'Music Generation': 'music',
            'Personality Chat': 'art',
            'Image to Video Chat': 'video',
        }
        mode_key = mapping.get(mode_name, 'art')
        defaults = self.model_priority.get(mode_key, {})
        priority = defaults.get('concept_medium', []) + defaults.get('prompt', [])
        ordered = []
        for m in priority:
            if m in self.available_models and m not in ordered:
                ordered.append(m)
        self.available_models = ordered + [m for m in self.available_models if m not in ordered]

    def get_defaults_for_mode(self, mode_name: str):
        """Return default concept/medium and prompt models for a mode."""
        mapping = {
            'Image Generation': 'art',
            'Video Generation': 'video',
            'Music Generation': 'music',
            'Personality Chat': 'art',
            'Image to Video Chat': 'video',
        }
        mode_key = mapping.get(mode_name, 'art')
        defaults = self.model_priority.get(mode_key, {})
        cm_list = defaults.get('concept_medium', [])
        prompt_list = defaults.get('prompt', [])
        model = next((m for m in cm_list if m in self.available_models), None)
        prompt_model = next((m for m in prompt_list if m in self.available_models), None)
        if model is None:
            model = self.available_models[0] if self.available_models else None
        if prompt_model is None:
            if 'gpt-5' in self.available_models:
                prompt_model = 'gpt-5'
            elif 'gpt-4.1' in self.available_models:
                prompt_model = 'gpt-4.1'
            else:
                prompt_model = self.available_models[0] if self.available_models else None
        return model, prompt_model
    def run(self):
        # Create tabs within the app
        tabs = [
            "Image Generation",
            "Video Generation",
            "Music Generation",
            "Chat",
            "Prompt Explorer",
        ]
        current_tab = st.session_state.get('selected_tab', tabs[0])
        selected_tab = st.sidebar.selectbox("Select Mode", tabs, index=tabs.index(current_tab))
        if selected_tab != current_tab:
            self.prioritize_models_for_mode(selected_tab)
            model, prompt_model = self.get_defaults_for_mode(selected_tab)
            st.session_state['model'] = model
            st.session_state['prompt_model'] = prompt_model
        st.session_state['selected_tab'] = selected_tab

        self.render_sidebar()

        if selected_tab == "Image Generation":
            self.render_image_generation()
        elif selected_tab == "Video Generation":
            self.render_video_generation()
        elif selected_tab == "Music Generation":
            self.render_music_generation()
        elif selected_tab == "Chat":
            self.render_chat()
        elif selected_tab == "Prompt Explorer":
            self.render_prompt_explorer()

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
                'Personality',
                personality_names,
                index=personality_names.index(
                    st.session_state.get('selected_personality', personality_names[0])
                ),
                key='sidebar_personality_select',
            )
            if st.session_state['selected_personality'] == 'Custom':
                st.session_state['custom_personality'] = st.sidebar.text_area(
                    'Custom Personality',
                    value=st.session_state.get('custom_personality', ''),
                    height=150,
                    key='sidebar_custom_personality',
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
            tiktok_mode = st.checkbox(
                "TikTok Mode",
                value=st.session_state.get('tiktok_mode', False),
                help="Use FLUX Pro v1.1 Ultra at 9:16 and rotate output for Veo 3.",
            )
            st.session_state['tiktok_mode'] = tiktok_mode
            if tiktok_mode:
                self.image_model = 'fal-ai/flux-pro/v1.1-ultra'
                st.session_state['fal-ai/flux-pro/v1.1-ultra_image_size'] = '9:16'
                st.session_state['fal-ai/flux-pro/v1.1-ultra_num_images'] = 1
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
                       "Symbolism": st.slider("Symbolism (0: Literal, 100: Symbolic)", 0, 100, 50),
                       "Pacing (Energy Level)": st.slider("Pacing (Energy Level) (0: Slow, 100: Fast)", 0, 100, 50),
                       "Hook Intensity": st.slider("Hook Intensity (0: Gentle, 100: Immediate)", 0, 100, 50),
                       "Aesthetic Stylization": st.slider("Aesthetic Stylization (0: Realistic, 100: Stylized)", 0, 100, 50),
                       "Lighting Mood": st.slider("Lighting Mood (0: Bright, 100: Dark)", 0, 100, 50),
                       "Perspective & Lensing": st.slider("Perspective & Lensing (0: Wide/Deep, 100: POV/Shallow)", 0, 100, 50),
                       "Motion Quality": st.slider("Motion Quality (0: Smooth, 100: Chaotic)", 0, 100, 50),
                       "Surrealism vs. Realism (Physics)": st.slider("Surrealism vs. Realism (0: Realistic, 100: Surreal)", 0, 100, 50)
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
            label="Art Idea",
            label_visibility="collapsed",  # hides it visually if desired
            key="input",  # This ensures the text stays in st.session_state['input']
            placeholder="Describe the essence of the art you wish to generate.",
            help="Provide a detailed description of your idea to get the best results.",
            height=200
        )

        st.subheader("Reference Images (Optional)")
        uploaded_files = st.file_uploader(
            "Upload up to 5 images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="uploaded_images"
        )
        images = []
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.warning("Only the first 5 images will be used.")
            for file in uploaded_files[:5]:
                images.append(file)
        st.session_state['input_images'] = images

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
        if st.button("Generate Concepts", type="primary", icon=":material/lightbulb:"):
            if not st.session_state['input'].strip():
                st.warning("Please provide a description of your idea.")
            else:
                style_axes, creativity_spectrum = self.generate_concepts()

        if st.session_state.get('competition_mode') and st.button("Run Competition", key="run_competition", type="primary", icon=":material/trophy:"):
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
            images = st.session_state.get('input_images')
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
                            st.session_state.get('reasoning_level', 'medium'),
                            input_images=images,
                        )
                        st.session_state['custom_personality'] = personality_text
                    display_temporary_results("Personality", personality_text, expanded=False)

                meta_prompt, frames_list, art_styles_list = generate_meta_prompt(
                    st.session_state.get('input', ''),
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    reasoning_level=st.session_state.get('reasoning_level', 'medium'),
                    personality_prompt=personality_text,
                    input_images=images,
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
                            input_images=images,
                        )
                        st.session_state['custom_panel'] = panel_text
                    display_temporary_results("Panel Prompt", panel_text, expanded=False)
                template = read_prompt('/lofn/prompts/overall_prompt_template.txt')
                input_text = (
                    template.replace('{Meta-Prompt}', meta_prompt['meta_prompt'])
                    .replace('{Panel-prompt}', panel_text)
                    .replace('{Personality-prompt}', personality_text)
                    .replace('{frames_list}', frames_list)
                    .replace('{art_styles_list}', art_styles_list)
                    .replace('{input}', st.session_state.get('input', ''))
                    .replace('{image_context}', image_context_to_string(images))
                )
                display_temporary_results("Meta Prompt", meta_prompt['meta_prompt'], expanded=False)
            else:
                if images:
                    input_text = f"{input_text}\n\nReference Images:\n" + image_context_to_string(images)
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
                    input_images=images,
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
        selected_pairs = create_mini_dashboard(
            st.session_state['concept_mediums'], key_prefix="image_pair"
        )
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
                    input_images=st.session_state.get('input_images'),
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
                    input_images=st.session_state.get('input_images'),
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
                    st.session_state['prompt_input'],
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
            images = st.session_state.get('input_images')
            personality_text = st.session_state.get('custom_personality', '')
            if st.session_state.get('selected_personality') == 'LLM Generated':
                if not personality_text:
                    personality_text = generate_personality_prompt(
                        st.session_state.get('input', ''),
                        self.max_retries,
                        self.temperature,
                        self.model,
                        self.debug,
                        st.session_state.get('reasoning_level', 'medium'),
                        input_images=images,
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
                input_images=images,
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
                        input_images=images,
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
                .replace('{image_context}', image_context_to_string(images))
            )
            display_temporary_results("Meta Prompt", meta_prompt['meta_prompt'], expanded=False)
            st.session_state['prompt_input'] = input_text

            with st.spinner("Generating music concepts..."):
                concept_mediums, style_axes, creativity = generate_music_concept_mediums(
                    input_text,
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=None,
                    creativity_spectrum=None,
                    reasoning_level=st.session_state.get('reasoning_level', 'medium'),
                    input_images=images
                )

            st.session_state['style_axes'] = style_axes
            st.session_state['creativity_spectrum'] = creativity
            st.session_state['music_concept_mediums'] = concept_mediums

            pairs = concept_mediums
            if pairs:
                try:
                    with st.spinner("Panel voting on best pairs..."):
                        best_pairs = select_best_pairs(
                            input_text,
                            pairs,
                            st.session_state.get('num_best_pairs', 3),
                            self.max_retries,
                            self.temperature,
                            self.model,
                            self.debug,
                            st.session_state.get('reasoning_level', 'medium'),
                        )
                    st.session_state['music_best_pairs'] = best_pairs
                    st.success("Best music pairs selected by panel")
                except Exception as e:
                    st.error("An error occurred while selecting best music pairs.")
                    logger.exception("Error selecting music pairs: %s", e)
                    best_pairs = pairs
            else:
                best_pairs = []

            st.session_state['song_prompts'] = {'revised_prompts': [], 'synthesized_prompts': []}
            top_n = st.session_state.get('num_best_pairs', 3)
            gen_pairs = best_pairs if best_pairs else pairs
            for pair in gen_pairs[:top_n]:
                self.generate_music_prompts_for_pair(pair)

            st.success("Music prompts generated successfully!")
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
                    reasoning_level=st.session_state['reasoning_level'],
                    tiktok_mode=st.session_state.get('tiktok_mode', False),
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
            label="Video Idea",
            label_visibility="collapsed",  # hides it visually if desired
            key="input",
            placeholder="Describe the essence of the art video you wish to generate.",
            help="Provide a detailed description of your idea to get the best results.",
            height=200
        )
        st.subheader("Reference Images (Optional)")
        uploaded_files = st.file_uploader(
            "Upload up to 5 images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="uploaded_video_images"
        )
        images = []
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.warning("Only the first 5 images will be used.")
            for file in uploaded_files[:5]:
                images.append(file)
        st.session_state['input_images'] = images
        if not st.session_state['input']:
            st.info("Tip: Describe a scene or narrative you'd like to see in motion.")

        if st.button("Generate Video Concepts", type="primary", icon=":material/movie:"):
            if not st.session_state['input'].strip():
                st.warning("Please provide a description of your idea.")
            else:
                self.generate_ui_video_concepts()

        if st.session_state.get('competition_mode') and st.button("Run Competition", key="run_video_competition", type="primary", icon=":material/trophy:"):
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
            images = st.session_state.get('input_images')
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
                            st.session_state.get('reasoning_level','medium'),
                            input_images=images,
                        )
                        st.session_state['custom_personality'] = personality_text
                    display_temporary_results("Personality", personality_text, expanded=False)
                meta_prompt, frames_list, film_styles_list = generate_meta_prompt(
                    st.session_state.get('input', ''),
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    reasoning_level=st.session_state.get('reasoning_level','medium'),
                    medium="video",
                    personality_prompt=personality_text,
                    input_images=images,
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
                            input_images=images,
                        )
                        st.session_state['custom_panel'] = panel_text
                    display_temporary_results("Panel Prompt", panel_text, expanded=False)
                template = read_prompt('/lofn/prompts/video_overall_prompt_template.txt')
                input_text = (
                    template.replace('{Meta-Prompt}', meta_prompt['meta_prompt'])
                    .replace('{Panel-prompt}', panel_text)
                    .replace('{Personality-prompt}', personality_text)
                    .replace('{frames_list}', frames_list)
                    .replace('{film_styles_list}', film_styles_list)
                    .replace('{input}', st.session_state.get('input', ''))
                    .replace('{image_context}', image_context_to_string(images))
                )
                display_temporary_results("Meta Prompt", meta_prompt['meta_prompt'], expanded=False)
            else:
                if images:
                    input_text = f"{input_text}\n\nReference Images:\n" + image_context_to_string(images)
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
                    input_images=images,
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
        selected_pairs = create_mini_dashboard(
            st.session_state['video_concept_mediums'], key_prefix="video_pair"
        )
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
                    st.session_state['prompt_input'],
                    pair['concept'],
                    pair['medium'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state['style_axes'],
                    creativity_spectrum=st.session_state['creativity_spectrum'],
                    reasoning_level=st.session_state.get('reasoning_level','medium'),
                    input_images=st.session_state.get('input_images'),
                )
            st.session_state['video_prompts_df'] = prompts_df
            st.success(f"Video prompts generated for '{pair['concept']}'")
            metadata = {
                'timestamp': datetime.now(),
                'concept': pair['concept'],
                'medium': pair['medium'],
                'prompts': prompts_df.to_dict(orient='list'),
                'input_text': st.session_state.get('prompt_input', ''),
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

    def display_music_concepts(self):
        st.subheader("Generated Music Concepts and Mediums")
        selected_pairs = create_mini_dashboard(
            st.session_state['music_concept_mediums'], key_prefix="music_pair"
        )
        st.session_state['selected_music_pairs'] = selected_pairs

        if st.button("Generate Music Prompts for Selected Concepts"):
            if selected_pairs:
                for idx in selected_pairs:
                    pair = st.session_state['music_concept_mediums'][idx]
                    self.generate_music_prompts_for_pair(pair)
            else:
                st.warning("Please select at least one concept to generate prompts.")

        if st.button("Generate Music Prompts for All Concepts"):
            for pair in st.session_state['music_concept_mediums']:
                self.generate_music_prompts_for_pair(pair)

    def generate_music_prompts_for_pair(self, pair):
        st.subheader(f"Generating Music Prompts for '{pair['concept']}' in '{pair['medium']}'")
        try:
            with st.spinner(f"Generating music prompts for '{pair['concept']}'..."):
                song_prompts = generate_music_prompts(
                    st.session_state['prompt_input'],
                    pair['concept'],
                    pair['medium'],
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=st.session_state['style_axes'],
                    creativity_spectrum=st.session_state['creativity_spectrum'],
                    reasoning_level=st.session_state.get('reasoning_level','medium'),
                    input_images=st.session_state.get('input_images')
                )
            if not song_prompts:
                st.error("Failed to generate music prompts.")
                return

            st.success(f"Music prompts generated for '{pair['concept']}'")

            if 'song_prompts' not in st.session_state or st.session_state['song_prompts'] is None:
                st.session_state['song_prompts'] = {'revised_prompts': [], 'synthesized_prompts': []}

            st.session_state['song_prompts']['revised_prompts'].extend(song_prompts.get('revised_prompts', []))
            st.session_state['song_prompts']['synthesized_prompts'].extend(song_prompts.get('synthesized_prompts', []))

            for prompt in song_prompts.get('revised_prompts', []) + song_prompts.get('synthesized_prompts', []):
                metadata = {
                    'timestamp': datetime.now(),
                    'title': prompt['title'],
                    'music_prompt': prompt['music_prompt'],
                    'lyrics_prompt': prompt['lyrics_prompt'],
                    'input_text': st.session_state.get('prompt_input', ''),
                    'competition': st.session_state.get('competition_mode', False),
                    'model': self.model,
                }
                save_music_metadata(metadata)

            self.display_music_prompts_for_pair(song_prompts, pair)
        except Exception as e:
            st.error(f"An error occurred while generating music prompts for '{pair['concept']}'.")
            logger.exception("Error generating music prompts: %s", e)

    def display_music_prompts_for_pair(self, song_prompts, pair):
        st.subheader(f"Music Prompts for '{pair['concept']}' in '{pair['medium']}'")
        for prompt in song_prompts.get('revised_prompts', []):
            st.markdown("**Title**")
            st.code(prompt['title'], language='text')
            st.markdown("**Music Prompt**")
            st.code(prompt['music_prompt'], language='text')
            st.markdown("**Lyrics Prompt**")
            st.code(prompt['lyrics_prompt'], language='text')
            st.markdown('---')

        for prompt in song_prompts.get('synthesized_prompts', []):
            st.markdown("**Title**")
            st.code(prompt['title'], language='text')
            st.markdown("**Music Prompt**")
            st.code(prompt['music_prompt'], language='text')
            st.markdown("**Lyrics Prompt**")
            st.code(prompt['lyrics_prompt'], language='text')
            st.markdown('---')

    def render_music_generation(self):
        # self.render_music_sidebar()
        st.header("Generate Your Music Concept")

        st.subheader("Describe Your Song Idea")
        st.text_area(
            label="Song Idea",
            label_visibility="collapsed",  # hides it visually if desired
            key="input",
            placeholder="Describe the themes, emotions, and specific elements you want in your song.",
            help="Provide a detailed description of your song idea."
        )
        st.subheader("Reference Images (Optional)")
        uploaded_files = st.file_uploader(
            "Upload up to 5 images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="uploaded_music_images"
        )
        images = []
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.warning("Only the first 5 images will be used.")
            for file in uploaded_files[:5]:
                images.append(file)
        st.session_state['input_images'] = images
        if not st.session_state['input']:
            st.info("Tip: Include themes, emotions, specific elements, and desired run time length.")

        if st.button("Generate Music Prompts", type="primary", icon=":material/music_note:"):
            if not st.session_state['input'].strip():
                st.warning("Please provide a description of your song idea.")
            else:
                self.generate_music_prompts_ui()

        if st.session_state.get('competition_mode') and st.button("Run Competition", key="run_music_competition", type="primary", icon=":material/trophy:"):
            if not st.session_state['input'].strip():
                st.warning("Please provide a description of your song idea.")
            else:
                self.run_music_competition()

        if st.session_state.get('music_concept_mediums'):
            self.display_music_concepts()

        if st.session_state.get('song_prompts'):
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
            input_text = st.session_state['input']
            images = st.session_state.get('input_images')
            if images:
                input_text = f"{input_text}\n\nReference Images:\n" + image_context_to_string(images)
            st.session_state['prompt_input'] = input_text
            with st.spinner("Generating music prompts..."):
                concept_mediums, style_axes, creativity = generate_music_concept_mediums(
                    input_text,
                    max_retries=self.max_retries,
                    temperature=self.temperature,
                    model=self.model,
                    debug=self.debug,
                    style_axes=None,
                    creativity_spectrum=None,
                    reasoning_level=st.session_state.get('reasoning_level','medium'),
                    input_images=images
                )

                st.session_state['music_concept_mediums'] = concept_mediums
                st.session_state['style_axes'] = style_axes
                st.session_state['creativity_spectrum'] = creativity

                if concept_mediums:
                    first = concept_mediums[0]
                    song_prompts = generate_music_prompts(
                        input_text,
                        first['concept'],
                        first['medium'],
                        max_retries=self.max_retries,
                        temperature=self.temperature,
                        model=self.model,
                        debug=self.debug,
                        style_axes=style_axes,
                        creativity_spectrum=creativity,
                        reasoning_level=st.session_state.get('reasoning_level','medium'),
                        input_images=images
                    )
                else:
                    song_prompts = {'revised_prompts': [], 'synthesized_prompts': []}
            st.session_state['song_prompts'] = song_prompts
            for prompt in song_prompts.get('revised_prompts', []) + song_prompts.get('synthesized_prompts', []):
                metadata = {
                    'timestamp': datetime.now(),
                    'title': prompt['title'],
                    'music_prompt': prompt['music_prompt'],
                    'lyrics_prompt': prompt['lyrics_prompt'],
                    'input_text': input_text,
                    'competition': False,
                    'model': self.model,
                }
                save_music_metadata(metadata)
            st.success("Music prompts generated successfully!")
        except Exception as e:
            st.error("An error occurred while generating music prompts.")
            logger.exception("Error generating music prompts: %s", e)

    def display_music_prompts(self):
        song_prompts = st.session_state.get('song_prompts')
        if not song_prompts:
            return

        st.subheader("Revised Prompts")
        for prompt in song_prompts.get('revised_prompts', []):
            st.markdown("**Title**")
            st.code(prompt['title'], language='text')
            st.markdown("**Music Prompt**")
            st.code(prompt['music_prompt'], language='text')
            st.markdown("**Lyrics Prompt**")
            st.code(prompt['lyrics_prompt'], language='text')
            st.markdown('---')

        st.subheader("Synthesized Prompts")
        for prompt in song_prompts.get('synthesized_prompts', []):
            st.markdown("**Title**")
            st.code(prompt['title'], language='text')
            st.markdown("**Music Prompt**")
            st.code(prompt['music_prompt'], language='text')
            st.markdown("**Lyrics Prompt**")
            st.code(prompt['lyrics_prompt'], language='text')
            st.markdown('---')

        st.info("Copy any of the above prompts and paste them into Udio to generate your music.")

    def render_prompt_explorer(self):
        """Allow browsing of saved metadata, video, and music prompts."""
        st.header("Prompt Explorer")
        content_type = st.radio(
            "Select Content Type", ["Images", "Videos", "Music"], key="prompt_explorer_type"
        )
        base_paths = {
            "Images": "/metadata",
            "Videos": "/videos",
            "Music": "/music",
        }
        base_path = base_paths[content_type]
        index = build_prompt_index(base_path)
        if not index:
            st.info("No files found for this content type.")
            return

        file_names = list(index.keys())
        search_query = st.text_input("Search", key="prompt_explorer_search").strip().lower()
        if search_query:
            file_names = [name for name, rec in index.items() if search_query in rec["haystack"]]
            if not file_names:
                st.info("No files match the search query.")
                return
        else:
            files = list(index.keys())

        selected_file = st.selectbox("Select File", file_names, key="prompt_explorer_file")
        if not selected_file:
            return
        data = index[selected_file]["meta"]

        if content_type == "Images":
            image_path = None
            if data.get("image_filename"):
                image_path = data.get("image_filename")
                if not os.path.isabs(image_path):
                    image_path = os.path.join("/images", image_path)
            if image_path and os.path.exists(image_path):
                st.image(image_path, caption=data.get("title", ""))
            elif data.get("image_url"):
                st.image(data.get("image_url"), caption=data.get("title", ""))
            st.subheader("Image Prompt")
            st.code(data.get("prompt", ""), language="text")
        elif content_type == "Videos":
            if data.get("video_filename"):
                video_path = os.path.join("/videos", data.get("video_filename"))
                if os.path.exists(video_path):
                    st.video(video_path)
            elif data.get("video_url"):
                st.video(data.get("video_url"))
            st.subheader("Video Prompts")
            prompts = data.get("prompts")
            if isinstance(prompts, dict):
                prompts_df = pd.DataFrame(prompts)
                for idx, row in prompts_df.iterrows():
                    st.markdown(f"**Prompt {idx+1}:**")
                    if "Revised Prompts" in prompts_df.columns:
                        st.markdown("**Revised Prompt**")
                        st.code(row.get("Revised Prompts", ""), language="text")
                    if "Synthesized Prompts" in prompts_df.columns:
                        st.markdown("**Synthesized Prompt**")
                        st.code(row.get("Synthesized Prompts", ""), language="text")
                    st.markdown('---')
        else:  # Music
            if data.get("music_filename"):
                music_path = os.path.join("/music", data.get("music_filename"))
                if os.path.exists(music_path):
                    with open(music_path, "rb") as audio_file:
                        st.audio(audio_file.read())
            elif data.get("music_url"):
                st.audio(data.get("music_url"))
            if data.get("title"):
                st.subheader("Title")
                st.code(data.get("title", ""), language="text")
            st.subheader("Music Prompt")
            st.code(data.get("music_prompt", ""), language="text")
            st.subheader("Lyrics Prompt")
            st.code(data.get("lyrics_prompt", ""), language="text")

        st.subheader("Settings & Statistics")
        basic_keys = [
            "timestamp",
            "model",
            "title",
            "concept",
            "medium",
            "prompt_type",
            "prompt_index",
            "image_index",
        ]
        info = {k: data.get(k) for k in basic_keys if k in data}
        if info:
            st.json(info)
        if data.get("creativity_spectrum") or data.get("style_axes"):
            display_creativity_and_style_axes(
                data.get("creativity_spectrum", {}),
                data.get("style_axes", {}),
            )
        if data.get("input_settings"):
            with st.expander("Input Settings"):
                st.json(data.get("input_settings"))

    def render_chat(self):
        mode = st.selectbox(
            "Chat Mode", ["Personality Chat", "Image to Video Chat"], key="chat_mode"
        )
        self.render_chat_section(mode)

    def render_chat_section(self, mode):
        image_mode = mode == "Image to Video Chat"
        key_prefix = "image2video_" if image_mode else ""
        st.subheader(mode)

        personality_names = [p['name'] for p in PERSONALITY_OPTIONS] + ['Custom']
        select_key = f"{key_prefix}selected_personality" if key_prefix else "selected_personality"
        st.session_state[select_key] = st.selectbox(
            'Personality',
            personality_names,
            index=personality_names.index(
                st.session_state.get(select_key, personality_names[0])
            ),
            key=f"{key_prefix}personality_select",
        )
        custom_key = f"{key_prefix}custom_personality" if key_prefix else "custom_personality"
        if st.session_state[select_key] == 'Custom':
            st.session_state[custom_key] = st.text_area(
                'Custom Personality',
                value=st.session_state.get(custom_key, ''),
                height=150,
                key=f"{key_prefix}custom_personality_text",
            )
        else:
            st.session_state[custom_key] = next(
                p['prompt']
                for p in PERSONALITY_OPTIONS
                if p['name'] == st.session_state[select_key]
            )
        personality_text = st.session_state.get(custom_key, '')

        clear_key = "clear_image2video_chat_images" if image_mode else "clear_personality_chat_images"
        if st.session_state.get(clear_key):
            images_key = f"{key_prefix}chat_images" if image_mode else "personality_chat_images"
            st.session_state.pop(images_key, None)
            st.session_state[clear_key] = False

        st.subheader("Reference Media (Optional)")
        images_key = f"{key_prefix}chat_images" if image_mode else "personality_chat_images"
        uploaded_files = st.file_uploader(
            "Upload up to 5 images or videos",
            type=["png", "jpg", "jpeg", "mp4", "mov", "webm"],
            accept_multiple_files=True,
            key=images_key,
        )
        chat_media = []
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.warning("Only the first 5 files will be used.")
            for file in uploaded_files[:5]:
                chat_media.append(file)
        input_images_key = f"{key_prefix}chat_input_images" if image_mode else "chat_input_images"
        st.session_state[input_images_key] = chat_media

        history_key = f"{key_prefix}chat_history" if image_mode else "chat_history"
        if history_key not in st.session_state:
            st.session_state[history_key] = []

        for msg in st.session_state[history_key]:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            with st.chat_message(role):
                if isinstance(msg.content, list):
                    for part in msg.content:
                        if part.get("type") == "text":
                            st.markdown(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            url = part.get("image_url", "")
                            if isinstance(url, dict):
                                url = url.get("url", "")
                            if url.startswith("data:"):
                                st.image(base64.b64decode(url.split(",")[1]))
                            else:
                                st.image(url)
                        elif part.get("type") == "file":
                            file_data = part.get("file", {})
                            mime = file_data.get("mime_type", "")
                            b64 = file_data.get("b64_json", "")
                            if mime.startswith("image/"):
                                st.image(base64.b64decode(b64))
                            elif mime.startswith("video/"):
                                st.video(base64.b64decode(b64))
                else:
                    st.markdown(msg.content)

        input_key = "image2video_chat_input" if image_mode else "personality_chat_input"
        user_input = st.chat_input("Send a message", key=input_key)
        if user_input:
            history = st.session_state[history_key][:]
            images = st.session_state.get(input_images_key, [])
            prepared = prepare_image_messages(images)
            user_message = HumanMessage(
                content=[{"type": "text", "text": user_input}, *[m.content[0] for m in prepared]]
            )
            st.session_state[history_key].append(user_message)
            with st.chat_message("user"):
                st.markdown(user_input)
                for media in prepared:
                    part = media.content[0]
                    if part["type"] == "image_url":
                        url = part["image_url"]
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        st.image(base64.b64decode(url.split(",")[1]))
                    elif part["type"] == "file":
                        file_data = part["file"]
                        mime = file_data.get("mime_type", "")
                        b64 = file_data.get("b64_json", "")
                        if mime.startswith("image/"):
                            st.image(base64.b64decode(b64))
                        elif mime.startswith("video/"):
                            st.video(base64.b64decode(b64))
            logger.debug(f"{mode} user input: %s", user_input)
            if mode == "Image to Video Chat":
                response_text = run_personality_image2video_chat(
                    personality_text,
                    history,
                    user_input,
                    model=self.model,
                    temperature=self.temperature,
                    reasoning_level=st.session_state.get('reasoning_level', 'medium'),
                    debug=self.debug,
                    input_media=[m.content[0] for m in prepared],
                )
            else:
                response_text = run_personality_chat(
                    personality_text,
                    history,
                    user_input,
                    model=self.model,
                    temperature=self.temperature,
                    reasoning_level=st.session_state.get('reasoning_level', 'medium'),
                    debug=self.debug,
                    input_media=[m.content[0] for m in prepared],
                )
            logger.debug(f"{mode} response: %s", response_text)
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state[history_key].append(AIMessage(content=response_text))
            st.session_state[input_images_key] = []
            st.session_state[clear_key] = True

    def initialize_session_state(self):
        cm_model, prompt_model = self.get_defaults_for_mode('Image Generation')
        default_values = {
            'selected_tab': 'Image Generation',
            'video_concept_mediums': None,
            'video_prompts_df': None,
            'music_concept_mediums': None,
            'music_best_pairs': None,
            'song_prompts': None,
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
            'model': cm_model,
            'prompt_model': prompt_model,
            'competition_mode': True,
            'competition_text': '',
            'selected_panel': PANEL_OPTIONS[0]['name'],
            'custom_panel': PANEL_OPTIONS[0]['prompt'],
            'selected_personality': PERSONALITY_OPTIONS[0]['name'],
            'custom_personality': PERSONALITY_OPTIONS[0]['prompt'],
            'image2video_selected_personality': PERSONALITY_OPTIONS[0]['name'],
            'image2video_custom_personality': PERSONALITY_OPTIONS[0]['prompt'],
            'num_best_pairs': 3,
            'prompt_input': '',
            'creativity_spectrum': None,
            'style_axes': None,
            'input': '',
            'progress_step': 0,
            'reasoning_level': 'medium',  # default reasoning if user doesn't set
            'input_images': [],
            'chat_history': [],
            'chat_input_images': [],
            'clear_personality_chat_images': False,
            'image2video_chat_history': [],
            'image2video_chat_input_images': [],
            'clear_image2video_chat_images': False,
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
