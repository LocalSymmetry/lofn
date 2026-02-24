# Lofn Music n8n Workflow

This directory contains the n8n workflow for Lofn Music, replicating the logic of the Python application using LangChain agents in n8n.

## Getting Started

1.  **Start n8n**:
    Run `docker-compose up -d` in this directory.
    Access n8n at `http://localhost:5678`.

2.  **Import Workflow**:
    Import the `workflow.json` file into n8n.

3.  **Configure Credentials**:
    You will need to configure the following credentials in n8n:
    *   **Google Gemini(PaLM) Api**: For the AI Agents.
    *   **Google Sheets**: For reading Genres and Music Frames (optional but recommended if using the Sheets nodes). The nodes are currently configured with placeholder/public URLs.

## Workflow Structure

The workflow consists of 11 Phases, each implemented as an AI Agent:
1.  Essence & Facets
2.  Concept Generation
3.  Concept Refinement
4.  Production Style Selection
5.  Production Refinement
6.  Scoring Facets
7.  Artistic Guides
8.  Raw Song Prompt Generation
9.  Influence-Refined Prompts
10. Prompt Evaluation & Synthesis
11. Final Prompt Generation

Each agent uses the Gemini 1.5 Pro model (configured as `gemini-3.1-pro-preview` in the JSON, adjustable in the node).

## Google Sheets Integration

Two Google Sheets nodes are included at the start of the workflow to load reference data:
*   **Genres**: [Link](https://docs.google.com/spreadsheets/d/1BeychqbAx3S5WxwD2Eq0wxTZ9mW0ohOA6yIJp6kJXuI/edit?usp=sharing)
*   **Music Frames**: [Link](https://docs.google.com/spreadsheets/d/18hzygQSedz2gBV6T2dPa5-zrYrcRa02-y-zv3VpRotg/edit?gid=0#gid=0)

Ensure these nodes can access the sheets or replace them with your own data sources.
