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