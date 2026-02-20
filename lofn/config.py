# config.py

import os
import yaml

class Config:
    # Read environment variables
    OPENAI_API = os.environ.get('OPENAI_API', '')
    ANTHROPIC_API = os.environ.get('ANTHROPIC_API', '')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', os.environ.get('GOOGLE_API', ''))
    POE_API = os.environ.get('POE_API', '')
    WEBHOOK_URL = os.environ.get('WEBHOOK_URL', '')
    webhook_url = os.environ.get('WEBHOOK_URL', '')
    RUNWARE_API_KEY = os.environ.get('RUNWARE_API_KEY', '')
    IDEOGRAM_API_KEY = os.environ.get('IDEOGRAM_API_KEY', '')
    FAL_API_KEY = os.environ.get('FAL_KEY', '')
    OPEN_ROUTER_API_KEY = os.environ.get('OPEN_ROUTER_API_KEY', '')
    GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', os.getenv('GOOGLE_PROJECT_ID', ''))
    GCP_LOCATION = os.getenv('GCP_LOCATION', '')
    RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY", "")

    # Local LLM configuration (for OpenAI-compatible local servers)
    LOCAL_LLM_API_BASE = os.environ.get('LOCAL_LLM_API_BASE', '')
    LOCAL_LLM_API_KEY = os.environ.get('LOCAL_LLM_API_KEY', '')


# Load overrides from a user-provided YAML file if available
CUSTOM_CONFIG_PATH = 'lofn/custom_configs.yaml'
if os.path.exists(CUSTOM_CONFIG_PATH):
    try:
        with open(CUSTOM_CONFIG_PATH, 'r') as f:
            custom_cfg = yaml.safe_load(f) or {}
        for key, value in custom_cfg.items():
            attr = key
            if key == 'FAL_KEY':
                attr = 'FAL_API_KEY'
            if key == 'GOOGLE_API':
                attr = 'GOOGLE_API_KEY'
            if key == 'GOOGLE_PROJECT_ID':
                attr = 'GCP_PROJECT_ID'
            setattr(Config, attr, value)
            os.environ[attr] = value
            if key != attr:
                os.environ[key] = value
            if key == 'WEBHOOK_URL':
                setattr(Config, 'webhook_url', value)
    except Exception:
        pass
