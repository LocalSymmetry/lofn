# Repo Overview

This repository contains **Lofn**, an AI art generator built with Streamlit.  The project orchestrates language models and image generation services to produce polished prompts and final artwork.  Key directories and files include:

```
/lofn
  ├─ config.py              # Reads API keys from environment variables
  ├─ config.yaml            # Example configuration with all API fields
  ├─ helpers.py             # Utility functions for parsing JSON, charts, and Discord integration
  ├─ llm_integration.py     # Core logic for interacting with LLMs
  ├─ image_generation.py    # Interfaces with image and video generation APIs
  ├─ o1_integration.py      # Custom ChatModel for OpenAI o1/o1-mini
  ├─ ui.py                  # Streamlit user interface
  ├─ prompts/               # Prompt templates used by the chains
  └─ style.css              # Styling for Streamlit
Dockerfile                  # Container build instructions
entrypoint.sh               # Loads config and launches Streamlit
```

The `examples/` folder contains sample images referenced in the README.

# Important Components

## Prompt Templates
All prompt text lives under `lofn/prompts/`. The folder contains system prompts and step templates that guide each phase of the generation workflow. Examples include:

- **`concept_system.txt`** – initial system message describing the roles (art director, critic, etc.).
- **`essence_prompt.txt`** – determines the creativity spectrum, style axes, and extracts the idea's essence.
- **`concepts_prompt.txt`** – proposes multiple raw concepts based on the extracted essence.
- **`artist_and_critique_prompt.txt`** – pairs each concept with an obscure artist and critiques it in that voice.
- **`medium_prompt.txt`** and **`refine_medium_prompt.txt`** – pick a medium for each concept and refine the selection.
- **`facets_prompt.txt`** and **`aspects_traits_prompts.txt`** – generate scoring facets and artistic guides.
- **`imagegen_prompt.txt`** – converts each guide into a concise image prompt.
- **`artist_refined_prompt.txt`** and **`revision_synthesis_prompt.txt`** – rewrite, rank, and synthesize prompts.
- **Video prompts** – files prefixed with `video_` mirror the above stages for video creation.
- **Music prompts** – `music_essence_prompt.txt` and `music_creation_prompt.txt` craft song ideas and lyrics.

The `panels.yaml` file defines expert panel presets used in competition mode.

## LLM Pipeline (`llm_integration.py`)
This module contains the majority of the algorithmic logic.  Highlights:

- **Model Abstraction** – `get_llm()` returns a LangChain model wrapper for OpenAI, Anthropic, Google Gemini, Poe, local servers, and OpenRouter.  Custom classes `OpenRouterLLM` and `GeminiLLM` implement the `LLM` interface.
- **Prompt Chains** – Chains are built from `ChatPromptTemplate` objects and executed via `run_any_chain` with automatic retries (`run_chain_with_retries`) and JSON validation using `parse_output` from `helpers.py`.
- **Generation Workflow** – Functions such as `generate_concept_mediums`, `generate_prompts`, and `generate_video_prompts` orchestrate multi‑step flows:
  1. extract essence and facets from user input
  2. create concepts and refine them with artist voices
  3. choose and refine mediums
  4. generate facets, artistic guides, and final prompts
  5. optionally produce video or music prompts
- **Error Handling** – Each step validates returned JSON against a schema and retries the LLM call when parsing fails.

## Image and Video Generation (`image_generation.py`)
This module interfaces with multiple services including DALL·E 3, Google Imagen 3, FAL models, Ideogram, and Poe-based generators.  It handles saving images, creating titles/captions, optional Pika video generation, and persisting metadata to `/metadata`.

## Streamlit Interface (`ui.py`)
`ui.py` powers the Streamlit web app. It manages session state, presents sidebar controls, and renders results. Three modes are available from the sidebar:

1. **Image Generation** – the default mode for creating art prompts and generating images with the selected model.
2. **Video Generation** – mirrors the image flow but uses the `video_*` prompts to craft prompts suitable for tools like Runway Gen‑3.
3. **Music Generation** – produces a music prompt and an annotated lyrics prompt ready for Udio.

Users may also enable a "competition" mode that employs `meta_prompt_generation.txt` and `panels.yaml` for panel‑based critique.

## Helpers
`helpers.py` contains routines to extract and repair JSON from model responses, visualize creativity spectrum and style axes with Plotly, and send prompts or results to Discord webhooks.

## Configuration and Deployment
- `config.py` reads environment variables for API keys.  `config.yaml` is a template for these values.
- `Dockerfile` installs required packages and uses `entrypoint.sh` to read `config.yaml` at runtime, disable Streamlit telemetry, and launch the app.

## Image Generation Run Structure
A typical run progresses through several LLM-driven steps:

1. **Essence & Facets** – the system extracts the core idea, sets the creativity spectrum, and defines five evaluation facets.
2. **Concepts & Artists** – multiple concepts are proposed and paired with obscure artists for critique.
3. **Medium Selection** – each concept receives a suitable medium, then the pairing is refined.
4. **Artistic Guides** – facets are expanded into short guides describing mood, style, lighting, and tools.
5. **Prompt Creation** – the guides become image prompts, which are then rewritten in the artists' voices and synthesized into final prompts.
6. **Image Generation** – prompts are sent to the configured model (DALL·E 3, Imagen 3, FAL, Ideogram, etc.) and the resulting images are saved under `/images`.

# Usage Notes
Run `docker build` and `docker run` as described in the README to start the web interface on port 8501.  Generated images are stored under `/images` and metadata under `/metadata` (mounted via Docker volumes when running the container).
