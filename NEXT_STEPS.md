# Lofn: The Agentic AI Phase - Next Steps

This document outlines the roadmap for transitioning Lofn from a Streamlit application into a fully Agentic AI Creative Studio.

## Phase 1: Foundation (Completed)
- [x] **Story Mode:** Implemented a dedicated "Story Generation" tab utilizing the structured prompt methodology (Essence -> Facets -> Generation).
- [x] **Hidden Personalities:** Integrated `LOFN-Prime-FULL` and `Alexis Dreams` as selectable personalities, loaded from a git-ignored configuration file.
- [x] **Path Hardening:** Refactored file path handling to support various deployment environments.

## Phase 2: Agentic Architecture
**Goal:** Decompose the monolithic application into distinct, autonomous personality agents.

### 1. Agent Definition
Each personality (e.g., Lofn, Alexis Dreams, Retro Coder) will become a distinct agent with a specific focus:
- **Art Generation Agent:** Specialized in Neo-Baroque Luminism and prompt engineering for DALL-E/Midjourney.
- **Music Generation Agent:** Specialized in Suno/Udio prompting, genre fusion, and lyric writing.
- **Video Generation Agent:** Specialized in RunwayML/Pika prompting and camera motion descriptions.
- **Social Media Agent:** Focused on captioning, overlays, and influencer guidance.

### 2. The Lofn MCP (Model Context Protocol)
Create a centralized `Lofn MCP` that exposes core skills and services to all agents:
- **Skills:** `generate_prompt`, `critique_art`, `search_web`, `access_archive`, `manage_memory`.
- **Services:** API wrappers for OpenAI, Anthropic, Fal.ai, Google Gemini, and Poe.
- **Context:** Shared state management for "Tree of Thoughts" reasoning and historical context.

### 3. Web Search Empowerment
- Enable agents to autonomously browse the web to research current art trends, music charts, and social media viral formats.
- Integrate search results directly into the "Essence" generation step.

## Phase 3: Modern Creative Studio UI
**Goal:** Transition from Streamlit to a modern React-based development environment.

### 1. React Frontend
- Develop a "Lofn Creative Studio" dashboard.
- **Features:**
    - Real-time chat interface with specific Agents.
    - Drag-and-drop artifact management (images, audio, text).
    - Multi-pane layout for "Artist-Critic" workflows (viewing generation and critique side-by-side).
    - System settings and personality configuration.

### 2. Backend API
- Expose the current Python logic (logic in `llm_integration.py` and `image_generation.py`) as a FastAPI or Flask service to power the React frontend.

## Phase 4: Magic Preservation & Evolution
**Constraint:** Ensure the "Magic of Lofn" is not lost during the transition.
- **Structured Prompts:** Maintain the deep, multi-stage prompt chains (Essence -> Facets -> Creation).
- **Tree of Thoughts:** Enforce the step-by-step reasoning and "Artist-Critic" refinement loops within the agent logic.
- **Random Injection:** Continue to use random styles, genres, and framings to prevent creative stagnation.
- **Extreme Context:** Ensure agents have full access to the "Archive" (reference library) and "Personality" definitions at all times.

## Phase 5: Storywriter Expansion
- Further refine the Story Mode to support:
    - **Chapter-based generation:** Keeping context across long narratives.
    - **Character consistency checks:** Using the "Critic" to ensure character voices remain stable.
    - **World-building wikis:** Automatically generating and updating a bible for the story's universe.
