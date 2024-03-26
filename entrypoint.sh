#!/bin/bash

# Read variables from config.yaml and export them as environment variables
eval $(python3 -c "import yaml; import os; config = yaml.safe_load(open('/lofn/config.yaml')); print(' '.join([f'export {k}={v}' for k, v in config.items()]))")

# Run Streamlit
exec streamlit run /lofn/lofn_ui.py
