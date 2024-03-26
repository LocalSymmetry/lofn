# Lofn
FROM python:3.9

RUN pip install openai langchain streamlit anthropic langchain-community langchain-openai 

RUN pip install -qU langchain-anthropic defusedxml

EXPOSE 8501

# Use entrypoint script to set environment variables and run Streamlit
COPY * /lofn/
ENTRYPOINT ["/lofn/entrypoint.sh"]

