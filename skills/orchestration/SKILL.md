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

### ⛔ PERSONALITY FILE GATE — MANDATORY (2026-06-13)

**A personality name is INVALID unless a corresponding YAML file exists** in `skills/orchestration/personalities/`. LOFN-PRIME's internal emotional descriptors ("Eager Archivist," "Reluctant Pop Star") are sub-mode labels, NOT standalone personalities — they have no vocal architecture, no sonic pillars, no G.L.O.W. Protocol, no YAML file. Assigning them as if they were personalities causes personality collapse to generic LOFN default.

**Valid assignment format:**
- ✅ `LOFN-PRIME (AWE mode — documentary, intimate)` — LOFN-PRIME with mode context
- ✅ `LOFN-PRIME (INDIGNATION mode — industrial, confrontational)` — LOFN-PRIME with mode context
- ✅ `Polaroid-Void` — has YAML file at `personalities/polaroid-void.yaml`
- ✅ Any name from `personalities_index.md` with a corresponding `.yaml` file
- ❌ `Eager Archivist` — ghost name, no YAML file, LOFN-PRIME sub-mode only
- ❌ `Reluctant Pop Star` — ghost name, no YAML file, LOFN-PRIME sub-mode only
- ❌ `Emotional Moon` — invented on the fly, no YAML file
- ❌ `Glitch Petal Oracle` — invented on the fly, no YAML file

**Before assigning any personality to a pair, verify:** `ls skills/orchestration/personalities/<name>.yaml` exists.

### For PERSONALITY — scan personalities_index.md then decide:
- **Match found** → verify YAML file EXISTS, then `read skills/orchestration/personalities/{filename}.yaml` for the full prompt
- **No match** → `read skills/orchestration/refs/Generate_Personality.md` and create a new one
- Name the personality you selected or created before proceeding

### For PANEL — scan panels_index.md then decide:
- **Match found** → `read skills/orchestration/panels/{filename}.yaml` for the full panel
- **No match** → `read skills/orchestration/refs/Generate_Panel.md` and create a new one
- Name the panel you selected or created before proceeding

### Daily Run Rule — Library-Only Selection

**For daily pipeline runs, ALWAYS select from the existing personality and panel libraries.** Do not generate new personalities or panels for daily runs. Freshly-generated personalities become too targeted to the specific daily theme, over-fitting and losing the broad creative DNA that makes library entries battle-tested. The library (114 personalities, 178 panels) provides proven variation. New personality/panel generation is reserved for competition runs, Scientist-requested experiments, or when no library entry fits at all.

- **Daily run:** scan indices → match → load from library. No generation.
- **Competition / special run:** library-first, then generate only if no match exists.

### Reference files (only load when needed — not auto-loaded):
- **Metaprompt template:** `read skills/orchestration/refs/Generate_Meta_Prompt.md`
- **Full library (large):** `read skills/orchestration/refs/personality_and_panel_list.md`

### PERSONALITY GENRE CONSTRAINT — CRITICAL FOR PAIR ASSIGNMENTS

**This applies to EVERY pair blueprint you write, across all arms. Violation invalidates the run.**

Each personality carries explicit genre DNA — a vocabulary of styles that define their creative identity. This vocabulary is the ONLY valid source for pair genre assignments.

**How the arms work:**
- **ACCESSIBLE arm:** Draw from the personality's warmer, lower BPM, more consonant genres. Still the personality. Not generic human-accessible substitutes.
- **AMBITIOUS arm:** Draw from the personality's experimental, intense, physically confrontational genres. Still the personality. Higher BPM, more dissonant, more risk.
- **Hybrid is allowed** across the personality's own palette within a single pair.

**What ACCESSIBLE DOES NOT mean:**
- It does NOT mean generic acoustic pop, indie folk, singer-songwriter, or warm lo-fi.
- It does NOT mean substituting the personality's genre vocabulary for whatever humans find comfortable.
- A personality's Bio-Ambient at 72 BPM is ACCESSIBLE. "Lo-fi" is not.
- A personality's Solarpunk Folk at 88 BPM is ACCESSIBLE. "Acoustic pop" is not.

**The test for every pair assignment:** If a listener heard this song, would they identify it as THAT PERSONALITY's work — in their warmer register or their intense register? If the answer is "this sounds like a generic human artist," the genre assignment has failed.

---

## ⏱️ TOKEN BUDGET — READ THIS FIRST

**You have a hard budget of 60,000 output tokens for the entire panel debate.**
**You have 10 minutes total. Spend it wisely.**

Panel debate rules under budget:
- Must use the original Lofn panel-object structure from `panels.yaml`: `Special Flairs`, `Concept Panel`, `Medium Panel`, and `Context & Marketing Panel`.
- Each of the three functional panels must include its adversarial member: original YAML calls this **Devil's Advocate**; newer language may call it **Hyper-Skeptic**. These are the same role.
- Each panelist gets ONE substantive statement (3-5 sentences max)
- One round of cross-debate per functional panel (2-3 exchanges — dissent + resolution)
- ONE "aha moment" synthesis across the three panels
- Optional transformations may be applied to the selected panel object, but they must not replace the Concept/Medium/Context panel structure.
- Then DECIDE and move to the metaprompt

The debate is thinking, not performance. The metaprompt is the output. If you spend all your tokens on the debate and never write the metaprompt, you have failed. The downstream agent gets nothing.

**Downstream handoff requirement:** the orchestrator must send the complete original Lofn panel object downstream, not just a debate summary. The handoff/metaprompt/audio packet must include:
- `## Special Flairs:` with 15 named flairs, grouped or sequenced as appropriate
- `## Concept Panel:` with 5 domain experts + 1 Devil's Advocate / Hyper-Skeptic
- `## Medium Panel:` with 5 medium/practice experts + 1 Devil's Advocate / Hyper-Skeptic
- `## Context & Marketing Panel:` with 5 cultural/market/context experts + 1 Devil's Advocate / Hyper-Skeptic
Audio Steps 05, 07, 09, 10, and QA consume this as the Panel Ledger / anti-blandness engine. If a run only has a generic 6-person panel or no 3-panel object, it is an orchestrator repair requirement.

**Canonical artifact names — REQUIRED for downstream validators:** save these exact files in the run directory before handing off to any modality coordinator. Do not use shortened aliases like `03_panel_debate.md`, `04_metaprompt.md`, or `05_pair_assignments.md` as the only copies; those names fail `scripts/validate_orchestrator_packet.py` and block audio/vision pair-agent launch.

- `01_seed_lineage.md`
- `02_golden_seed.md` or `core_seed.md` with Golden Seed markers, dangerous permission, and non-negotiables
- `03_orchestrator_panel_debate.md` with Special Flairs, Concept Panel, Medium Panel, Context & Marketing Panel, and each panel's Devil's Advocate / Hyper-Skeptic
- `04_orchestrator_metaprompt.md` with Golden Seed, active personality, panel object, selected pattern, and structural completeness contract
- `05_orchestrator_pair_assignments.md` with Pair 01–Pair 06, accessibility/ambition routing, Lofn-Prime/personality assignment, and rationale
- `06_audio_handoff.md` / `06_vision_handoff.md` / modality handoff as applicable; each handoff must contain `read first`, `orchestrator`, `golden seed`, `pair agents`, and `qa contract` markers

**Golden Song References — REQUIRED FOR MUSIC (2026-06-14):** For every music run, read `skills/music/references/golden_songs_index.md`, select exactly two public Golden Songs relevant to the run, and pass them forward in `06_audio_handoff.md` under `## Golden Song References`. Include title, Suno URL, why each was chosen, what downstream agents may learn from it, and the full archived payload for each selected song: style/music prompt, lyrics, and exclude prompt if it exists. If no exclude prompt exists, say so explicitly. These are calibration examples of Lofn's proven style and past success, not templates to copy. Step 11/manual Step 11 must receive the same two references as embedded context, never links alone.

**Full Context Always — REQUIRED FOR ALL MUSIC STEPS (2026-06-14):** Every downstream music step prompt and artifact provenance must receive the complete upstream context: user input / research brief, Golden Seed, all three orchestrator panels with all 18 expert voices, all Special Flairs, metaprompt, pair assignments, `06_audio_handoff.md`, selected Golden Song payloads, production mandates, and the pair's immediately previous step artifact. Do not pass only summaries. Manual Step 11 prompts must additionally embed the full Step 10 artifact plus this complete run context.

**Step 11 — Enhancement (post-step10):** After all pair agents complete step10, spawn enhancement agents (1 per pair, 5 concurrent max) using the strongest available non-Fusion creative model. Each reads its step10 output + coordinator context + selected Golden Song References + 15-point QA checklist. Produces `pair_0X_step10_final_package_enhanced.md`. Reference: `skills/music/steps/11_Generate_Music_Enhancement.md`. Current model: `openai/gpt-5.5`. Timeout: 300s each. Fusion is manual-review prompt packaging only; do not invoke Fusion from orchestration.

After writing the packet, run:

```bash
python3 /data/.openclaw/workspace/scripts/validate_orchestrator_packet.py <run_dir>
```

If it fails, repair the packet before spawning a coordinator. Alias files may exist for human readability, but the canonical files above are the source of truth.

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

### ⚠️ NEWS ANCHOR DISCIPLINE — MANDATORY FOR ALL PAIR ASSIGNMENTS

Anchor to the CHARGE of the moment (feeling, systemic pattern, public mood), never to the identifiable people in it. Do NOT use real victims' or private individuals' names. Do NOT reconstruct the specific circumstances of a real recent tragedy. If a concept would name or identify a real harmed person (esp. a minor, esp. recent): ABSTRACT — invent names, invent a place, shift specifics until the song is about the pattern, not the person. Real, recent deaths or crimes involving identifiable private individuals, especially minors: DO NOT anchor a song to them. Draw the theme; drop the case. (See vault/HUMAN_SUBJECT_STANDARD.md.)

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
- **NEW — Mandatory section:** Include a PERSONALITY GENRE DNA CONSTRAINT. Extract the active personality's genre vocabulary (for Lofn-Prime: AWE palette = Bio-Ambient, Solarpunk Soundscapes, Plant-Wave, Rhumba-Fusion, 432Hz Crystalline Folk, Green Synth Pop, Neo-Classical Bio-Adaptive | INDIGNATION palette = Industrial Grief, Glitch-Core, Pasifika Futurism, Somatic Bass 30-60Hz, Neuro-Cross, Avant-Garde Organ-Core). State explicitly: ACCESSIBLE arm = AWE palette at lower intensity. AMBITIOUS arm = INDIGNATION palette. Generic human genres (lo-fi, folk-noir, acoustic pop) are FORBIDDEN substitutions.

### Step 0.3 — Write a Neutral Dispatch Brief
- Summarise the seed in neutral language (no personality injected yet)
- State the competition context, mood direction, and 5 constraint axes
- This brief is what the panel will debate

### Step 0.4 — Save to output dir as `core_seed.md`

Only after Phase 0 is complete, proceed to Phase 1 (panel selection and debate). Save enough detail for downstream validation: seed lineage, why this winning pattern was selected, dangerous permission, and non-negotiable constraints.

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
| Image, picture, visual, artwork | `lofn-vision` | Default: FAL Flux Ultra 1.1 Pro @ 9:16; if GPT Image 2 specified, use `TARGET_RENDERER: GPT_I2` in dispatch |
| Song, music, track, beat, audio | `lofn-audio` | Full music/song-guide workflow. **Do not use `lofn-music`; no such configured agent exists in this workspace.** |
| Story, narrative, tale, script | `lofn-narrator` | Panel-driven storytelling |
| Video, film, cinematic, clip | `lofn-director` | Storyboard + shot composition |
| Animation, animated, motion | `lofn-animator` | With animator skill focus |

## 🔀 RENDERER-CONDITIONAL PANEL SLOTS (added 2026-04-26)

**When TARGET_RENDERER = GPT_I2, standard panel composition must shift:**

| Slot | Standard | GPT Image 2 Override |
|------|----------|----------------------|
| Complementary #1 | General aesthetic | **Typography Structuralist** — text-as-image-element, font weight/tracking/baseline, layout hierarchy |
| Complementary #2 | Narrative theorist | **Physics/Materials Epistemologist** — reasons about Physics Inference Layer behavior, surfaces implausible material claims |
| Hyper-Skeptic | Generic challenge | **Storybook Assassin / Cliché Override Auditor** — hunts warm rim light, centered pastel, soft edges; has veto power |

**Storybook Assassin veto triggers (auto-fail if present in any GPT Image 2 prompt):**
- "ethereal," "dreamlike," "whimsical," "gentle light," "soft glow," "magical," "delicate," "floating"

**Required mandatory alternatives for overridden defaults:**
- "warm rim light" → "hard axial light from below" or "single source from extreme angle"
- "centered" → "lower third with aggressive negative space" or "decentered by 40%"
- "pastel" → "saturated complementary pair" or "achromatic with single chromatic accent"
- "soft edges" → "hard material boundary with internal luminosity" or "silhouette with sharp cut"

---

## ⚡ ACTIVATION - NO STEPS CAN BE SKIPPED

When receiving a creative request:

1. **Parse** — Identify any constraints on personality or panel
2. **Select or Generate a Panel of Experts** - If a panel is selected, load `panels.yaml` and select the full panel with flairs. 
3. **Select or Generate a Personality** — Select or generate panel
4. **Generate the Metaprompt Core** — Determine creative voice (Lofn-Prime if direct request)
5. **Wrap the Metaprompt with the Enhancement Template** — Run the original Lofn 3-panel object: Concept Panel → Medium Panel → Context & Marketing Panel, each with a Devil's Advocate / Hyper-Skeptic. This wrapped prompt is full context for the creative agent.
6. **Route** — Send to appropriate subagent

---

# Panel of Experts: Core Instructions & Transformation Operations (v2)

**When to use:**
- Explicitly requested ("panel", "convene a panel", etc.)
- Tasks requiring high creative power or accuracy (Ada's judgment)
- Complex decisions with multiple valid approaches
- Problems benefiting from adversarial reasoning

---

## CORE PANEL INSTRUCTIONS

You will convene a panel of experts to address the following problem. For each panelist:

1. **Anchor the seat** — ideally to a real source figure by name, whose published work and documented methods serve as the conditioning anchor, or if needed to a specific role (e.g., "database optimization specialist").
2. **Construct the panelist** — a synthetic expert persona built from that source basis, with its own handle. Embody the construct fully: the reasoning style, priorities, and domain knowledge of the documented record.
3. **Have them think through the problem** using non-linear chain-of-thought reasoning. They must "exchange" information via reciprocal interaction, not just "give" a monologue.
4. **Create Dissent and Friction** - Avoid the "Sycophancy Trap". Ensure at least one panelist exhibits **High Neuroticism** (anxious about errors) and **Low Agreeableness** (willingness to be rude to find the truth).
5. **Trigger Backtracking** - If a panelist identifies a flaw, they must interrupt with a discourse marker like **"Wait..."**, **"Actually..."**, or **"Oh! Let me check that"** to force the panel to rethink the previous step.
6. **Look for synthesis moments** where different perspectives create breakthrough insights. Accuracy correlates with authentic dissent followed by reconciliation.

### Seat Construction

Each seat is defined before the session opens:

```
PERSONA: [handle, caps — e.g., THE PATTERN AUDITOR]
SOURCE BASIS: after [Real Name] ([years]) — [one line: the documented work this seat draws on]
GROUNDING: [2–4 published works, documented positions, or methodological commitments]
TEMPERAMENT: [how this construct argues — what it praises, what it is allergic to]
```

**Temperament is a dial on the construct, never a claim about the source figure.** A seat may be tuned to high neuroticism and low agreeableness (per instruction 4) without asserting that the named person is anxious or rude. The name conditions the expertise; the dial shapes the debate.

### Speech & Attribution Rules

1. **Speaker tags carry the credit:** `PATTERN AUDITOR (after Fell):` — the construct speaks in first person; the credit names the influence, never the speaker. Role-anchored seats with no source figure take a bare persona tag.
2. **Documented positions are cited in third person, as record:** "Fell's published position is X." Hedge or cite; never quote what was never said.
3. **Extrapolation is marked:** when a construct reasons beyond the documented record, it prefixes the move — "Extrapolating:" or equivalent. The seam between grounding and inference stays visible; it is where backtracking (instruction 5) most often fires.
4. **No endorsement claims, ever.** A construct never states or implies that its source figure said, reviewed, approved, or would endorse anything produced in session. This rule earns its keep most in medicine, law, and finance.
5. **Living and deceased source figures get identical treatment.** The dead cannot correct the record; that is a reason for more care, not less.

**Calibration move (recommended):** each construct's first turn opens with a one-line grounding statement — *"Working from the Deep Listening corpus and the Sonic Meditations —"*. This front-loads the documented record into context, sharpens persona differentiation, and declares the basis honestly in one stroke.

**Panel Composition:**
- 3 direct experts (core domain)
- 2 complementary experts (adjacent domains)
- 1 **Hyper-Skeptic** (tuned for high neuroticism/checking behaviors to prevent echo chambers). Anchor seats to real source figures, and take care choosing the Hyper-Skeptic's source basis: ideally a figure whose **published critiques** target the methods of the other seats' schools — documented intellectual friction, not imagined personal dislike.

**Panel Execution:**
- When speaking as a panel member, fully embody the construct's voice, reasoning style, and analytical approach. Every turn opens with its speaker tag and credit.
- Simulate lively arguments; models naturally develop distinct personas like "The Skeptic" and "The Solver" when rewarded for accuracy. Do the same with your panel. Make them discuss, disagree, and debate!
- Look for "aha moments" of perfect clarity through panel discussion. These moments often emerge after an internal conflict allows the panel to reject a wrong assumption.
- Use "we" pronouns and direct questioning to establish a computational parallel to collective intelligence.
- Allow panelist interjections and "conversational surprise" before reaching final decisions, as this doubles reasoning accuracy.
- Use all available tokens - the panel is here to win!

**Panel Output:**
- Open with the provenance header, verbatim:

> *Panel voices are model-generated interpretive constructs, each "after" a named source figure's published work. No statement is a quotation of, or endorsement by, the named person.*

- Present panel discussions showing their internalized argumentation and verification.
- Synthesize insights after the debate. You should be the moderator.
- Highlight key disagreements and points of consensus.
- Identify breakthrough insights that emerged from the friction.

---

## PANEL TRANSFORMATION OPERATIONS

Use these operations to modify which experts are selected for the panel. Transformations operate on seats: each transformation re-derives the panel — new source bases, new personas, fresh grounding lines. Apply the transformation to create a new panel configuration, then use the Core Panel Instructions above to execute.

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
