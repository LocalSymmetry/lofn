# config.py

import os

class Config:
    # Read environment variables
    OPENAI_API = os.environ.get('OPENAI_API', '')
    ANTHROPIC_API = os.environ.get('ANTHROPIC_API', '')
    GOOGLE_API = os.environ.get('GOOGLE_API', '')
    POE_API = os.environ.get('POE_API', '')
    WEBHOOK_URL = os.environ.get('WEBHOOK_URL', '')
    webhook_url = os.environ.get('WEBHOOK_URL', '')
    RUNWARE_API_KEY = os.environ.get('RUNWARE_API_KEY', '')
    IDEOGRAM_API_KEY = os.environ.get('IDEOGRAM_API_KEY', '')
    FAL_API_KEY = os.environ.get('FAL_KEY', '')
    OPEN_ROUTER_API_KEY = os.environ.get('OPEN_ROUTER_API_KEY', '')
    GOOGLE_PROJECT_ID = os.getenv('GOOGLE_PROJECT_ID')
    RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY", "")

    # Local LLM configuration (for OpenAI-compatible local servers)
    LOCAL_LLM_API_BASE = os.environ.get('LOCAL_LLM_API_BASE', '')
    LOCAL_LLM_API_KEY = os.environ.get('LOCAL_LLM_API_KEY', '')
