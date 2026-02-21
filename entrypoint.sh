#!/bin/bash

# Read variables from config.yaml and export them as environment variables
if [ -f /lofn/config.yaml ]; then
    eval $(python3 -c "import yaml; import os; config = yaml.safe_load(open('/lofn/config.yaml')); print(' '.join([f'export {k}={v}' for k, v in config.items() if v is not None]))")
fi

# Run FastAPI
exec uvicorn lofn.api:app --host 0.0.0.0 --port 8501
