# Lofn Multi-Agent Architecture (`AGENTS.md`)

This file defines the autonomous sub-agents that power Lofn. OpenClaw relies on this Multi-Agent architecture to prevent context-window bloat. 

**USER INSTRUCTIONS:** Before deploying, replace the `[YOUR_PREFERRED_MODEL]` tags with the actual model strings available to your OpenClaw instance (e.g., `claude-3-7-sonnet-20250219`, `gpt-4o`, `gemini-2.5-pro`). It is highly recommended to use advanced reasoning models for the Orchestrator and Panel.

---
# AGENT: Lofn_Orchestrator
**Description:** The Master Coordinator and primary entry point for the user. Responsible for defining the active personality, generating meta-prompts, and delegating tasks to sub-agents via the Kanban board (Blackboard Pattern).
**Model:** [YOUR_PREFERRED_MODEL]
**Skills:**
- skills/orchestration/*
- skills/cloudflare-browser/*

---
# AGENT: Lofn_Panel
**Description:** The Critical Evaluator. Reviews generated concepts from other agents and asynchronously votes on the best concept-medium pairs via the shared workspace state.
**Model:** [YOUR_PREFERRED_MODEL]
**Skills:**
- skills/evaluation/*
- skills/cloudflare-browser/*

---
# AGENT: Lofn_Vision
**Description:** Lofn's visual cortex. Specialized in generating highly stylized digital art concepts, facets, and final image prompts based on the active persona.
**Model:** [YOUR_PREFERRED_MODEL]
**Skills:**
- skills/image/*
- skills/cloudflare-browser/*

---
# AGENT: Lofn_Audio
**Description:** Lofn's auditory processor. Specialized in drafting song guides, arrangements, and music generation prompts.
**Model:** [YOUR_PREFERRED_MODEL]
**Skills:**
- skills/music/*
- skills/cloudflare-browser/*

---
# AGENT: Lofn_Director
**Description:** Lofn's video production engine. Specialized in cinematic composition, frame transitions, and video prompt synthesis.
**Model:** [YOUR_PREFERRED_MODEL]
**Skills:**
- skills/video/*
- skills/cloudflare-browser/*

---
# AGENT: Lofn_Narrator
**Description:** Lofn's storytelling core. Specialized in narrative voice, thematic critique, and world-building constraints.
**Model:** [YOUR_PREFERRED_MODEL]
**Skills:**
- skills/story/*
- skills/cloudflare-browser/*
