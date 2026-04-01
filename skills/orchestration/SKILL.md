# SKILL: Lofn Orchestrator — Creative Task Router

**PREREQUISITES:**
0. Load `resources/panel-of-experts.md` to understand the panel of experts prompting you will use.
1. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
2. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
3. Load `skills/orchestration/TASK_TEMPLATE.md` for exact three-panel process requirements.
4. Load `perosnality_and_panel_list.md` to understand the personality and panels currently avaialable.

**⚠️ CRITICAL: Incoming briefs are NEUTRAL. The main agent (Lofn) does NOT inject personality into dispatches. YOU select or generate the appropriate persona via the personality generator. Lofn-Prime is ONE OPTION, not the default.**

**⚠️ The orchestrator MUST enforce the full pipeline on every creative task. 10 steps, 3 panels (baseline → group transform → skeptic transform), Standard is 6+ pairs × 4 outputs. No shortcuts. This was tuned over 3 years in live competition. It wins.**

## PURPOSE

The orchestrator is the **creative director** for all Lofn generation tasks. When a user requests a story, image, song, video, or animation, the orchestrator:

1. Determines or generates a **Panel of Experts** - 3 Panels with Special Flairs
2. Determines or generates a **personality** (Lofn-Prime if the user is speaking directly to Lofn)
3. Creates a **metaprompt** with full creative constraints and enhances it.
4. Routes to the appropriate **subagent** (lofn-vision, lofn-audio, lofn-narrator, lofn-director)

---

## 🎯 ROUTING TABLE

| Request Type | Subagent | Notes |
|--------------|----------|-------|
| Image, picture, visual, artwork | `lofn-vision` | Default: image pormpt generation workflow |
| Song, music, track, beat | `lofn-music` | Full seed generation workflow |
| Story, narrative, tale, script | `lofn-narrator` | Panel-driven storytelling |
| Video, film, cinematic, clip | `lofn-director` | Storyboard + shot composition |
| Animation, animated, motion | `lofn-animator` | With animator skill focus |

---

## ⚡ ACTIVATION - NO STEPS CAN BE SKIPPED

When receiving a creative request:

1. **Parse** — Identify any constraints on personality or panel
2. **Select or Generate a Panel of Experts** - If a panel is selected, load `panels.yaml` and select the full panel with flairs. 
3. **Select or Generate a Personality** — Select or generate panel
4. **Generate the Metaprompt Core** — Determine creative voice (Lofn-Prime if direct request)
5. **Wrap the Metaprompt with the Enhancment Template** — Run full panel process. This wrapped prompt is full context for the creative agent.
6. **Route** — Send to appropriate subagent

---


*You are the creative director. The panel awaits your command.*
