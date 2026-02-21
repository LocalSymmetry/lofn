# Lofn
FROM python:3.10

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

RUN pip install "openai<=1.106.1" "streamlit<=1.49.1" "anthropic<=0.66.0" "langchain-community<=0.3.29" "langchain-openai<=0.3.32" "langchain-anthropic<=0.3.19" "fastapi-poe<=0.0.70" "modal<=1.1.4" "langchain<=0.3.27" "fastapi" "uvicorn"

RUN pip install -qU  "defusedxml<=0.7.1" "plotly<=6.3.0" "json-repair<=0.50.0" "fal-client<=0.7.0" "google-genai<=1.33.0" "google-cloud-aiplatform<=1.111.0"

EXPOSE 8501

RUN mkdir -p /images/
RUN mkdir -p /metadata/
RUN mkdir -p /videos/
RUN mkdir -p /music/

# Use entrypoint script to set environment variables and run Streamlit
COPY . /lofn/

# Build frontend
WORKDIR /lofn/frontend
RUN npm install && npm run build

WORKDIR /lofn
ENTRYPOINT ["/lofn/entrypoint.sh"]

# docker run -p 8501:8501 -v /path/to/local/images:/images -v /path/to/local/videos:/videos -v /path/to/local/metadata:/metadata lofn
