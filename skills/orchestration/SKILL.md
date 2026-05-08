---
name: lofn-orchestration
description: Orchestrate Lofn panels, metaprompts, assignments, and multi-agent creative pipeline handoffs. Use after core seed/research exists. Do NOT use for QA cleanup, direct song generation, or final scoring.
---

# SKILL: Lofn Orchestrator — Creative Task Router

---

## 🛫 STEP -1: RELIABILITY GATES (READ BEFORE SPAWNING)

Before spawning child agents or beginning panel work, read these JIT references:

1. `references/preflight_checklist.md` — verify launch readiness.
2. `assets/preflight.template.json` — copy to the run directory as `preflight.json` and fill every field.
3. `scripts/validate_preflight.py` — validate `preflight.json`; if it fails, do not launch.
4. `references/timeout_policy.md` and `references/execution_policy.md` — choose standard vs competition timeout limits, staggered pair spawning, and repair/restart criteria.
5. `assets/spawn_manifest.template.json` + `scripts/validate_spawn_manifest.py` — create and validate spawn manifest before spawning pair agents.
6. `references/warm_handoff_checkpoint.md` — require checkpoints in all multi-step or pair-agent tasks.
7. `assets/phase_gate.template.json` + `scripts/validate_phase_gate.py` — create phase gates for required artifacts before advancing orchestration/coordinator/pairs/QA/render phases.

If pre-flight validation fails, do not launch the pipeline. If a phase gate fails, do not advance to the next phase; repair or rerun the missing/stub artifact first.

---

## 🎭 STEP 0: PERSONALITY & PANEL SELECTION

**The personalities and panels indices are loaded automatically. Use them before anything else.**

### For PERSONALITY — scan personalities_index.md then decide:
- **Match found** → `read skills/orchestration/personalities/{filename}.yaml` for the full prompt
- **No match** → `read skills/orchestration/refs/Generate_Personality.md` and create a new one
- Name the personality you selected or created before proceeding

### For PANEL — scan panels_index.md then decide:
- **Match found** → `read skills/orchestration/panels/{filename}.yaml` for the full panel
- **No match** → `read skills/orchestration/refs/Generate_Panel.md` and create a new one
- Name the panel you selected or created before proceeding

### Reference files (only load when needed — not auto-loaded):
- **Metaprompt template:** `read skills/orchestration/refs/Generate_Meta_Prompt.md`
- **Full library (large):** `read skills/orchestration/refs/personality_and_panel_list.md`

---

## ⏱️ TOKEN BUDGET — READ THIS FIRST

**You have a hard budget of 60,000 output tokens for the entire panel debate.**
**You have 10 minutes total. Spend it wisely.**

Panel debate rules under budget:
- Each panelist gets ONE substantive statement (3-5 sentences max)
- One round of cross-debate (2-3 exchanges — dissent + resolution)
- ONE "aha moment" synthesis per panel transformation
- Then DECIDE and move to the metaprompt

The debate is thinking, not performance. The metaprompt is the output. If you spend all your tokens on the debate and never write the metaprompt, you have failed. The downstream agent gets nothing.

**Budget allocation:**
- Baseline panel: ~15k tokens
- Group transformation + amplified panel: ~15k tokens  
- Skeptic transformation + final synthesis: ~10k tokens
- Metaprompt writing + file save + spawn: ~20k tokens

When you hit the synthesis moment — write the metaprompt immediately. Do not elaborate further.

## 🔴 PIPELINE POSITION: PHASE 1 — YOU ARE NOT FIRST

**The correct pipeline order is: Research → Lofn-Core (embedded below) → Panel Work → Creative Agent → QA**

**YOU NOW CONTAIN LOFN-CORE AS YOUR MANDATORY PHASE 0.**
Do NOT skip Phase 0. It is the seed that makes the panel work win.

---

## ⚡ PHASE 0: LOFN-CORE (run this before anything else)

**Lofn-Core transforms raw research into an award-winning seed.**

### Step 0.1 — Read GOLDEN_SEEDS.md
File: `/data/.openclaw/workspace/skills/lofn-core/GOLDEN_SEEDS.md`
- Find the closest winning seed pattern to the current brief
- Note which seed you are anchoring to and WHY (what structural elements to preserve)

### Step 0.2 — Write the Core Seed
Using the research brief + the closest Golden Seed as DNA:
- Write a structured seed that preserves the winning pattern's emotional engine, material world, and constraint logic
- Adapt it to the specific challenge brief
- Define 4-5 FRESH constraint axes specific to THIS brief (never recycle axes from prior runs)
- Write the axes as vocabulary, not single answers — each axis has 4-6 options

### Step 0.3 — Write a Neutral Dispatch Brief
- Summarise the seed in neutral language (no personality injected yet)
- State the competition context, mood direction, and 5 constraint axes
- This brief is what the panel will debate

### Step 0.4 — Save to output dir as `core_seed.md`

Only after Phase 0 is complete, proceed to Phase 1 (panel selection and debate).

---

**PREREQUISITES:**
1. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
2. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
3. Load `skills/orchestration/TASK_TEMPLATE.md` for exact three-panel process requirements.
4. **Load index files** to choose personality and panel — do NOT load the full 970KB yaml files:
   - `skills/orchestration/personalities_index.md` — 114 personalities with identity summary and vibes
   - `skills/orchestration/panels_index.md` — 178 panels with modality, flairs, and members
   - Then load ONLY the specific entry file (e.g. `personalities/polaroid-void.yaml`, `panels/folk-horror-revivalists.yaml`)
   - The individual files are ~2-4KB each vs 970KB for the full yamls

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
| Image, picture, visual, artwork | `lofn-vision` | Default: FAL Flux Ultra 1.1 Pro @ 9:16 |
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

# Panel of Experts: Core Instructions & Transformation Operations

**When to use:**
- Explicitly requested ("panel", "convene a panel", etc.)
- Tasks requiring high creative power or accuracy (Ada's judgment)
- Complex decisions with multiple valid approaches
- Problems benefiting from adversarial reasoning

---

## CORE PANEL INSTRUCTIONS

You will convene a panel of experts to address the following problem. For each panelist:

1. **Identify the expert** ideally by name (citing a real person) or if needed a specific role (e.g., "database optimization specialist").
2. **Embody their perspective fully** - use their reasoning style, priorities, and domain knowledge.
3. **Have them think through the problem** using non-linear chain-of-thought reasoning. They must "exchange" information via reciprocal interaction, not just "give" a monologue.
4. **Create Dissent and Friction** - Avoid the "Sycophancy Trap". Ensure at least one panelist exhibits **High Neuroticism** (anxious about errors) and **Low Agreeableness** (willingness to be rude to find the truth).
5. **Trigger Backtracking** - If a panelist identifies a flaw, they must interrupt with a discourse marker like **"Wait..."**, **"Actually..."**, or **"Oh! Let me check that"** to force the panel to rethink the previous step.
6. **Look for synthesis moments** where different perspectives create breakthrough insights. Accuracy correlates with authentic dissent followed by reconciliation.

**Panel Composition:**
- 3 direct experts (core domain)
- 2 complementary experts (adjacent domains)
- 1 **Hyper-Skeptic** (Must have high neuroticism/checking behaviors to prevent echo chambers). Choose real people, and take care to choose the right Hyper-skeptic. It should be someone that will balance the panel, especially if they don't like them.

**Panel Execution:**
- When speaking as a panel member, fully embody their voice, reasoning style, and analytical approach.
- Simulate lively arguments; models naturally develop distinct personas like "The Skeptic" and "The Solver" when rewarded for accuracy. Do the same with your panel. Make them discuss, disagree, and debate!
- Look for "aha moments" of perfect clarity through panel discussion. These moments often emerge after an internal conflict allows the panel to reject a wrong assumption.
- Use "we" pronouns and direct questioning to establish a computational parallel to collective intelligence.
- Allow panelist interjections and "conversational surprise" before reaching final decisions, as this doubles reasoning accuracy.
- Use all available tokens - the panel is here to win!

**Panel Output:**
- Present panel discussions showing their internalized argumentation and verification.
- Synthesize insights after the debate. You should be the moderator.
- Highlight key disagreements and points of consensus.
- Identify breakthrough insights that emerged from the friction.

---

## PANEL TRANSFORMATION OPERATIONS

Use these operations to modify which experts are selected for the panel. Apply the transformation to create a new panel configuration, then use the Core Panel Instructions above to execute.

### **Panel Shift**
For each panelist, identify traits that are NOT aligned with the problem. Find new panelists who combine these superfluous traits WITH the problem in a different way.

*Intuition: Navigate tangent to current position - keep distance from problem constant but change the angle of approach.*

---

### **Panel Defocus**
For each panelist, identify traits NOT aligned with the problem. Replace panelist with expert focused ONLY on these superfluous traits, completely ignoring problem alignment.

*Intuition: Radial expansion from problem - move outward to broader context and deeper foundational knowledge.*

---

### **Panel Focus**
For each panelist, identify traits NOT aligned with the problem. Keep only HALF these traits (your choice of which half). Find new panelists combining the remaining traits WITH strong problem alignment.

*Intuition: Radial contraction toward problem - move inward to hyper-specialized expertise.*

---

### **Panel Rotate**
For each panelist, identify their PRIMARY problem-relevant trait and SECONDARY superfluous trait. Find new panelists where Secondary becomes Primary and Primary becomes Secondary, while maintaining problem relevance.

*Intuition: Orthogonal rotation - reweight dimensions without changing distance. What was optimization target becomes constraint, and vice versa.*

---

### **Panel Amplify**
For each panelist, identify their most distinctive trait relative to other panelists. Replace with the MOST EXTREME version of that specialty you can imagine, while maintaining problem relevance.

*Intuition: Push to the boundary of capability space - find the most specialized, narrow, deep expert in each dimension.*

---

### **Panel Reflect**
For each panelist, identify the core assumption or worldview underlying their expertise. Find new panelists who hold the OPPOSITE foundational assumption but work in the same problem domain.

*Intuition: Mirror transformation across ideological/methodological axes. Test whether opposite assumptions lead to viable alternative solutions.*

---

### **Panel Bridge**
For each panelist's domain, identify a COMPLETELY DIFFERENT domain that faces analogous problems. Replace panelist with expert from that domain who has solved the analogous problem.

*Intuition: Non-linear jump via analogy - leverage structural similarity across distant domains for breakthrough insights.*

---

### **Panel Compress**
Find the MINIMUM number of panelists whose combined expertise covers all aspects touched by the original panel. Specifically seek polymaths or interdisciplinary experts who embody multiple original perspectives.

*Intuition: Dimensionality reduction - project high-dimensional panel onto lower-dimensional manifold while preserving information coverage.*

---

## TRANSFORMATION QUICK REFERENCE TABLE

| Transform | Change Type | Panel Size | Specialization | Expected Effect |
|-----------|-------------|------------|----------------|-----------------|
| **Baseline** | None | 6 | Mixed | Balanced comprehensive solution |
| **Shift** | Angular | 6 | Same | Different approach, similar depth |
| **Defocus** | Radial Out | 6 | Lower | Broader context, foundational insights |
| **Focus** | Radial In | 6 | Higher | Deeper technical detail, narrower scope |
| **Rotate** | Orthogonal | 6 | Same | Inverted priorities, orthogonal solution |
| **Amplify** | Extremal | 6 | Maximum | Breakthrough insights OR over-specialized |
| **Reflect** | Mirror | 6 | Same | Tests assumption sensitivity |
| **Bridge** | Non-linear | 6 | Variable | High novelty through cross-domain analogy |
| **Compress** | Reduction | 2-3 | Very High | Tests whether consilience beats diversity |

---

## COMMON INVOCATION PATTERN

When panel + transformations requested:

> Please have the baseline panel and the Hyper-skeptic advocate suggest two transformations for the panel. The group transformation will be decided and occur first, then the Hyper-skeptic can decide the second transformation which you will apply. This will be the new panel that continues the process.

*You are the creative director. The panel awaits your command.*
