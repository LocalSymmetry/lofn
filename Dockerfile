# Lofn
FROM python:3.10

RUN pip install "openai<=1.35.13" "langchain<=0.2.7" "streamlit<=1.36.0" "anthropic<=0.31.0" "langchain-community<=0.2.7" "langchain-openai<=0.1.14" "fastapi-poe" "modal"

RUN pip install -qU "langchain-anthropic<=0.1.19" "defusedxml<=0.7.1" "plotly<=5.22.0" "json-repair<=0.25.3" "fal-client<=0.4.1" "google-generativeai<=0.7.2" "runware<=0.2.7" "google-cloud-aiplatform<=1.68.0"

EXPOSE 8501

RUN mkdir -p /images/
RUN mkdir -p /metadata/

# Use entrypoint script to set environment variables and run Streamlit
COPY * /lofn/
COPY lofn/style.css /
ENTRYPOINT ["/lofn/entrypoint.sh"]

# docker run -p 8501:8501 -v /path/to/local/images:/images -v /path/to/local/videos:/videos -v /path/to/local/metadata:/metadata lofn
