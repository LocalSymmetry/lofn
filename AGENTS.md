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
All prompt text lives under `lofn/prompts/`.  Files such as `concept_system.txt`, `essence_prompt.txt`, and `imagegen_prompt.txt` are combined to build long prompts for the LLM pipeline.  The `panels.yaml` file defines expert panel presets.

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
`ui.py` provides the user-facing application.  It manages session state, presents sidebar controls for selecting models and style parameters, and displays generated concepts, prompts, and images.  Users can also run a "competition" mode which utilizes `meta_prompt_generation.txt` and `panels.yaml` for panel-based critique.

## Helpers
`helpers.py` contains routines to extract and repair JSON from model responses, visualize creativity spectrum and style axes with Plotly, and send prompts or results to Discord webhooks.

## Configuration and Deployment
- `config.py` reads environment variables for API keys.  `config.yaml` is a template for these values.
- `Dockerfile` installs required packages and uses `entrypoint.sh` to read `config.yaml` at runtime, disable Streamlit telemetry, and launch the app.

# Usage Notes
Run `docker build` and `docker run` as described in the README to start the web interface on port 8501.  Generated images are stored under `/images` and metadata under `/metadata` (mounted via Docker volumes when running the container).
