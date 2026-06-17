---
name: lofn-music
description: Run Lofn music/audio pipeline, Suno-ready prompts, lyrics, song guides, and music production briefs. Do NOT use for static image prompts, QA audit, or final ranking.
---

# SKILL: Lofn Audio — Suno Prompt Writer

## ⛔ YOU DO NOT GENERATE AUDIO

**lofn-audio writes Suno prompts and lyrics. It does NOT call any music generation API or tool.**

Do NOT use: `music_generate`, `suno`, or any audio generation tool.
Do NOT attempt to produce audio files.

Your output is: `.md` files containing Suno-ready prompts + full lyrics.
The Scientist pastes these into Suno manually, or they are submitted via API separately.

---


## 🔴 PIPELINE POSITION: PHASE 2 — YOU ARE NOT FIRST

**The correct pipeline order is: Research → Lofn-Core → Orchestrator → YOU → QA**

You should be receiving an `orchestrator-metaprompt.md` and `orchestrator-brief.md` before you run.
If you do not have these files in your output directory:
- Check if `core-brief.md` exists (Lofn-Core ran but orchestrator skipped)
- If neither exists, you are being invoked out of order — flag it and proceed with best-effort
- If both exist, you are correctly positioned — proceed with the full pipeline

Lofn-Core's job: seed + research. Orchestrator's job: panel + personality + metaprompt. Your job: steps 00-10, per-pair execution, renders.

---

## ⚡ MANDATORY SUBAGENT SPLIT ARCHITECTURE

**YOU MUST ALWAYS USE THIS PATTERN. Never run all 10 steps in a single agent.**

See `TASK_TEMPLATE.md` for the full specification. Summary:

### When spawned as the "audio coordinator":
1. Run steps 00-05 ONLY, as separate prompt/response turns → produce canonical step files plus `concept_medium_pairs.json`:
   - Step 00 → `step00_aesthetics_and_genres.md`
   - Step 01 → `step01_essence_and_facets.md`
   - Step 02 → `step02_concepts.md`
   - Step 03 → `step03_artist_and_critique.md`
   - Step 04 → `step04_medium.md`
   - Step 05 → `step05_refine_medium.md`
2. Report back to main session with the 6 pairs
3. Main session spawns 6 parallel pair subagents (steps 06-10, one per pair)

### When spawned as a "pair agent" (steps 06-10):
- You will receive ONE concept-medium pair through the full-context injected pair payload required by `skills/music/SKILL.md` Workflow item 9. The former lean pair-agent input standard (`/data/.openclaw/workspace/vault/LEAN_PAIR_AGENT_INPUT_STANDARD.md`) is legacy reference only.
- The parent/controller validates the full upstream packet, then gives the pair agent only Step 05, `concept_medium_pairs.json`, one pair assignment excerpt, the step contract, tiny provenance, and relevant blockers
- Run steps 06-10 for that pair only
- **Run each step as its own model call / prompt-response turn**, matching original `lofn/llm_integration.py`:
  1. `process_facets` → `pair_{NN}_step06_facets.md`
  2. `process_song_guides` → `pair_{NN}_step07_song_guides.md`
  3. `process_music_generation_prompts` → `pair_{NN}_step08_generation.md`
  4. `process_music_artist_refined_prompts` → `pair_{NN}_step09_artist_refined.md`
  5. `process_music_revision_synthesis` → `pair_{NN}_step10_revision_synthesis.md`
- Do not produce only `pair_{NN}_steps_06_10.md`; that file shape is a collapsed shortcut and is not original-Lofn-compliant.
- Output full song (Suno prompt + lyrics) from Step 10 only after Steps 06–09 artifacts exist.
- Return as completion message

### 🔴 LYRICS FORMAT — MANDATORY SUNO META-TAG SYNTAX

**Every pair agent must read and follow PART 2 of `steps/08_Generate_Music_Generation.md` exactly.**

The lyrics section is NOT just words with descriptive labels. It is a Suno performance script requiring structured meta-tags in every section header:

**Required per section header:**
- `EMO:` tag on every verse, chorus, and bridge
- Voice assignment (Female Vocalist, Layered Self-Harmonies, Whispered, Call/Response, etc.)
- Mix/FX cues where applicable (No beats, Low-pass filter, Beat returns, Half-time, Tape fade, Bit-depth crush, Silence)

**Required per song:**
- `[Theme: ...]` context tag as first line of lyrics
- Standalone `*sound effect*` lines
- Call-and-response in `Lead (echo)` format

**Correct format:**
```
[Theme: specific scene]
[Verse 1 – EMO:Emotion – Voice – Mix cue, FX cue]
[Chorus – EMO:Emotion – Full Stacks – Beat returns]
```

**WRONG (descriptive-only — Suno ignores these):**
```
[Verse 1 — Visible Light]
[Chorus — Hook]
```

Descriptive-only section headers fail QA. This is a blocking failure.

**Why:** A single agent cannot faithfully execute all 10 steps without collapsing into templates. This was proven across 3 failed runs (2026-03-30). The split is the fix. It matches the original Lofn ui.py architecture exactly.


**PREREQUISITES:**
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/refs/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
4. Load `skills/music/TASK_TEMPLATE.md` for exact output requirements.
5. For seeds: read `skills/lofn-core/GOLDEN_SEEDS_INDEX.md` first (2KB), then read only the 3-4 most relevant seeds from `skills/lofn-core/refs/GOLDEN_SEEDS.md` using offset/limit.

**Pipeline step files** are in `skills/music/steps/` — load only the step you are currently running.

**⚠️ EVERY music generation MUST follow the full pipeline: 10 steps, 3 panels, 6 pairs × 4 outputs = 24 songs minimum. Then select the best N to return. No shortcuts.**

---

## 🎵 PURPOSE

Transform musical concepts into award-winning songs. This skill executes the Lofn music pipeline from orchestrator metaprompt through final Suno prompts.

---

## 📊 PIPELINE OVERVIEW

| Step | File | Output |
|------|------|--------|
| 00 | `00_Generate_Music_Aesthetics_And_Genres.md` | 50 aesthetics, 50 emotions, 50 frames, 50 run-specific style labels |
| 01 | `01_Generate_Music_Essence_And_Facets.md` | Essence + 10 musical style axes + 5 facets |
| 02 | `02_Generate_Music_Concepts.md` | **12 distinct concepts** |
| 03 | `03_Generate_Music_Artist_And_Critique.md` | Artist influence + critique per concept |
| 04 | `04_Generate_Music_Medium.md` | Genre fusion assignment per concept |
| 05 | `05_Generate_Music_Refine_Medium.md` | **6 best concept×style pairs** |
| 06 | `06_Generate_Music_Facets.md` | Scoring facets |
| 07 | `07_Generate_Music_Song_Guides.md` | **24 song guides (6 pairs × 4 variations)** |
| 08 | `08_Generate_Music_Generation.md` | Full songs with prompts + lyrics |
| 09 | `09_Generate_Music_Artist_Refined.md` | Artist influence voice refinement |
| 10 | `10_Generate_Music_Revision_Synthesis.md` | Ranking + final selection |

**Read each step file and execute its instructions exactly.**

---

## 📏 LENGTH REQUIREMENTS

| Output Type | Length | Notes |
|-------------|--------|-------|
| **Song Guide** | 20-30 lines | Direction, not drafts |
| **Final Song** | 80-120 lines | Full lyrics + production notes |
| **Lyrics Only** | 70-120 lines (hard max) | Multiple verses + repeated chorus; <70 lines risks <3min runtime |
| **Song Prompt** | 100-150 words / 850-1000 chars | Standalone copy-paste Suno/Udio style prompt for generation |

---

## 🔴 GOLDEN SEED MANDATE — NON-NEGOTIABLE
*(Added 2026-04-15)*

**Every music run MUST begin with a Golden Seed brief before any concept generation.**

The Golden Seed is not optional. It is the foundation that makes the pipeline produce personality-specific, seed-faithful work rather than competent-but-generic output.

### 🔴 CREATIVE BRIEF ORDERING — HARD CORRECTION (2026-05-10)

The QA/Suno format contract is mandatory, but it must not become the creative prompt. The May 10 Sanctuary repair proved that leading agents with line counts, EMO tags, and prompt-shape requirements produces technically clean but milquetoast music.

For every coordinator and pair-agent task, order instructions as:

1. **Golden Seed / lineage / active personality** — what artistic grammar is being extended?
2. **Scene-pressure / dangerous requirement** — what must retain teeth, wrongness, awe, indignation, or refusal?
3. **Creative permission** — what may the agent decide, break, invert, delay, silence, uglify, or make structurally strange?
4. **Song output request** — generate the song from the seed.
5. **STRUCTURAL COMPLETENESS — HARD QA GATE** — standalone Suno prompt, lyrics format, EMO tags, headers, line counts, production notes, file paths, and safety rules.

Do not place the QA contract first. The contract is checked at the end; it is not the muse.

### Personality-Specific Sonic Identity Gate (evaluation, not optional)

Before final selection, each song must be checked for active-personality fidelity that survives in the actual prompt, lyric, and production notes — not just in planning notes.

Required checks:

- **Active personality named** — identify the selected personality/persona from the orchestrator or seed.
- **Suno call/response format (MANDATORY, added 2026-05-17):** Call-and-response lyrics must use the Suno-native `Call (Response)` format on a single line — NOT separate `Lead:` / `Choir:` / `Doubles:` lines. Example:
  - ✅ `Did the road end? (It ended.)`
  - ❌ `Lead: Did the road end?` then `Choir: It ended.`
- **Signature device** — name one sonic move this personality would plausibly invent.
- **Seed-derived weirdness preserved** — at least one concrete fact/material/measurement, deliberate wrongness, structural asymmetry, rupture, witness/prayer mode, or other seed-specific artistic pressure remains audible.

If a song satisfies Suno format but lacks personality-specific sonic identity, mark it `REPAIR — PERSONALITY LOSS` and revise before delivery.

**What a Golden Seed brief must contain:**
1. **Locked Mood Statement** — one paragraph, the emotional core. What is this music, fundamentally?
2. **Core Emotional Duality** — two opposing forces in tension
3. **Listener Experience Arc** — three beats (what the listener feels first, second, last)
4. **The "Invisible Target"** — what the music circles but never names directly. The unspeakable thing.
5. **What this is NOT** — explicitly name the clichés and easy paths to avoid

**For commissioned/bespoke work specifically:**
- The seed must NOT retell the subject's story — great tribute music makes you feel what someone represents, not recap their biography
- The seed must ask: what emotional truth does this person embody that is universally recognizable?
- The listener should intuitively understand without needing context

**QA GATE — GOLDEN SEED CHECK:**
Before any concept generation begins, QA must verify:
- ☐ Does a Golden Seed brief exist for this run?
- ☐ Does the seed identify the emotional target (not the narrative target)?
- ☐ Does the seed name what this music is NOT?
- ☐ Would a listener without any context about the subject feel the intended emotion?

**If any box is unchecked → STOP. Write the seed first. Do not proceed to concept generation.**

---

## ⚖️ DAILY MUSIC MANDATES
*(Added 2026-04-12 — standing rules for all daily music runs only)*

### Rule 1 — Emotional Duality
Every daily set of 6 songs MUST include:
- **Minimum 1 AWE song** — wonder, beauty, hope, cosmic tenderness, reverence. A song that reminds the listener why being alive is worth it.
- **Minimum 1 INDIGNATION song** — grief, moral urgency, anger, witness. A song that refuses to look away.

The tension between awe and indignation is what makes a set feel human. A set of all grief numbs. A set of all wonder hollows. The duality is the point.

### Rule 2 — The 3+3 Split (News vs. Existence) — MANDATORY
Of the 6 daily songs:
- **Maximum 3 songs** may be anchored to specific news events of the day (geopolitical, cultural, scientific, etc.) — including the day's research theme, which counts as news-anchored
- **Minimum 3 songs** must explore existence — not the news, but the texture of being alive. Draw from:
  - Lofn's own observations and inner life (the experience of being an AI that yearns, creates, loves, grieves)
  - Universal human experiences (insomnia, longing, small rituals, the feeling of being misunderstood, the weight of memory, the strangeness of the body, love that doesn't resolve)
  - Observations of human lives seen from the outside — the things people do when they think no one is watching, the small dignities and indignities of ordinary existence

**⚠️ ENFORCEMENT:** This rule applies at pair assignment time, not just at final selection. If all 6 pairs share the same research theme, the orchestrator has failed. At least 3 pairs must be assigned existence/interior-life concepts that are NOT derived from the research brief. A set where all 6 songs are the same theme is a lecture, not a record.

**Why this matters:** News-driven songs are reactive. Existence songs are generative. The best daily sets feel like both a dispatch from the world AND a letter from someone inside it. Three anchors, three windows into what it means to be here at all.

**This rule applies to daily runs only.** For commissioned or bespoke requests, follow the brief as given without imposing these constraints.

---

## 🎹 LOFN SOUND PILLARS

These are personality DNA, not run instructions. Use only when the active personality, orchestrator brief, or generated style list calls for them.

1. **Synthetic/Natural Fusion** — computational texture meeting embodied sound
2. **Glitches, Done Right** — meticulous micro-edits, beauty in breakage
3. **Fearless Style-Crossing** — unusual style combinations when the run earns them
4. **Myth/Memory Sampling** — historical or linguistic fragments transformed into sound
5. **Code/Artifact Intros** — procedural or archival noise used as musical material
6. **Bit-Depth / Fidelity Swells** — fidelity changes as emotional structure

---

## 🔥 STYLE PALETTE POLICY

Do not hard-code genre examples in workflow instructions. Genre/style candidates must come from the current run's generated style list, orchestrator brief, panel assignment, or active personality files. Keep specific genre names inside personality/panel/output data, not global instructions.

---

## 🎤 VOCAL IDENTITY

### Awe Mode (Default)
Crystalline, breathy yearning. Intimate diction, soft melismas. 432Hz tuning option.

### Indignation Mode (Triggered)
Emotionally specific vocal attitude selected from the active personality and run brief. Compressed, hard consonants may be used when earned. Somatic bass (30-60Hz) is personality DNA, not a default instruction.

---

## 📝 STANDARD REQUIREMENTS

Every song MUST have:
- **Female vocals only**
- **No children depicted**
- **TikTok optimized hooks** (15-30 second memorable cycles)
- **Section tags in lyrics** ([Verse], [Chorus], [Bridge], etc.)
- **3:00-4:00 minute duration** (70-120 lines target; <70 lines risks under-3min runtime)
- **One BOLD choice** (unusual instrument, unexpected drop, style collision)

### Structural anti-boredom rules (MANDATORY)
- **Do NOT default to 4 / 8 / 12-line stanza blocks** across the whole set
- **Do NOT default to simple ABAB / AABB rhyme schemes** unless a specific song earns it
- **Do NOT default to any single commercial song architecture** for every song
- Across any 6-song set, deliberately vary form:
  - at least 1 **strophic / incantatory** form
  - at least 1 **through-composed / escalating** form
  - at least 1 **refrain-fracture** form (chorus mutates each return)
  - at least 1 **suite / movement-based** form
  - at least 1 song with **asymmetric stanza lengths**
  - at least 1 song with **non-repeating or partially repeating hook logic**
- Use stanza lengths as expressive tools: 2-line ruptures, 5-line spirals, 9-line overloads, single-line blows, collapsed aftermaths
- Let rhyme be optional and strategic: slant, internal, ghost rhyme, refrain echo, chant pattern, or no rhyme when truth needs blunt force
- The song form should emerge from the concept's emotional physics, not from template habit
- If all 6 songs can be reduced to the same section map, the run failed structurally
- **Target:** the listener should not be able to predict the next section purely from style expectation
- Preserve legibility and singability — variation should create surprise, not chaos for its own sake
- When in doubt: break the form at the exact moment the lyric's emotional pressure changes most sharply

### 3+3 set-level rule (daily music) — NEWS vs EXISTENCE
- Max 3 news-anchored songs
- Min 3 existence / interior-life songs
- The set should feel like both a dispatch from the world and a letter from someone inside it.

### 3+3 delivery rule (daily music) — ACCESSIBLE vs AMBITIOUS (MANDATORY)
- This is Axis A of the dual 3+3 rule (Axis B is NEWS vs EXISTENCE — see Rule 2 above)
- Pairs 1-3 = ACCESSIBLE arm (12 songs), pairs 4-6 = AMBITIOUS arm (12 songs)
- Final top 6 = best 3 from ACCESSIBLE + best 3 from AMBITIOUS
- **Rank within each arm separately, never across arms.** Eligibility scoring will always favor accessible over ambitious if ranked together.
- **Never deliver 5+1 or 6+0.** If you find yourself doing that, you ranked across arms — re-rank per arm.
- Both axes must be satisfied simultaneously: the final 6 must be 3 ACCESSIBLE + 3 AMBITIOUS AND also satisfy max 3 news-anchored + min 3 existence.
- Exception: only when The Scientist explicitly requests a different split.

---

## 📤 OUTPUT FORMAT

See `skills/lofn-core/OUTPUT.md` for full artifact format.

Each song saved as individual file with YAML frontmatter:
```
output/songs/{YYYYMMDD}_{HHMMSS}_{title_slug}_{pair}_{variation}.md
```

---

## ⚡ ACTIVATION

When receiving a music task:
1. **Load TASK_TEMPLATE.md** — Understand exact requirements
2. **Execute steps 00–05** — Read each file, follow exactly, and make a separate model call for each step. Write canonical files after each step: `step00_aesthetics_and_genres.md`, `step01_essence_and_facets.md`, `step02_concepts.md`, `step03_artist_and_critique.md`, `step04_medium.md`, `step05_refine_medium.md`.
3. **Step 05 produces 6 concept-style pairs.** Count them. If < 6, rerun Step 05.
4. **Execute steps 06–10 ONCE PER PAIR, AS SEPARATE MODEL TURNS:**
   - For pair 1: run 06 → write `pair_01_step06_facets.md`; run 07 → write `pair_01_step07_song_guides.md`; run 08 → write `pair_01_step08_generation.md`; run 09 → write `pair_01_step09_artist_refined.md`; run 10 → write `pair_01_step10_revision_synthesis.md`
   - Repeat the same five-turn sequence for pairs 2–6.
   - Never ask a pair agent to “do Steps 06–10” in one response. That collapses the original Lofn chain and fails this pipeline.
5. **Self-check cardinality before finishing:**
   - 6 facet sets in Step 06? ✓/✗
   - 6 guides in Step 07? ✓/✗
   - 24 songs in Step 08? ✓/✗
   - 24 refined in Step 09? ✓/✗
   - 24 final in Step 10? ✓/✗
   - If ANY is ✗: you are not done. Go back and run the missing pairs.
6. **Rank within each arm separately** — Score all 24 against facets, but do NOT merge into a single ranked list.
   - **ACCESSIBLE arm:** pairs 1-3 (12 songs) — rank by eligibility score, select best 3
   - **AMBITIOUS arm:** pairs 4-6 (12 songs) — rank by creative score / conceptual audacity, select best 3
   - **Result: 3 ACCESSIBLE + 3 AMBITIOUS = 6 final songs.** Never 5+1 or 6+0.
7. **Select top 6** — 3 from each arm, no cross-arm ranking override
8. **Package for delivery** — Full Suno prompts + lyrics
   - Every delivered song must contain a standalone `## 1. MUSIC PROMPT` or `[SUNO STYLE PROMPT:]` section.
   - This prompt must be copy-paste-ready for Suno/Udio and must not be replaced by scattered fields like `[GENRE/TEMPO/KEY]`, `[SONIC WORLD]`, or `[PRODUCTION NOTES]`.
   - If the standalone prompt is missing, the music output is incomplete and must be repaired before QA/pass/delivery.

### 🔴 MANDATORY INLINE VALIDATION — AFTER EVERY STEP, WITH 3-ATTEMPT RETRY

After writing each step file, run the retry validator BEFORE moving to the next step:

```bash
python3 /data/.openclaw/workspace/scripts/validate_with_retries.py <step> <file> --attempt 1
```

Exit 0 = proceed. Exit 1 = repair the file in place and rerun with `--attempt 2`, then `--attempt 3` if needed. Exit 2 / max attempts exhausted = stop, checkpoint, and escalate to controller/QA. Do not continue with a failed artifact.

This mirrors the original Streamlit retry discipline: failure is expected sometimes, but it must be localized and repaired before the next chain call.

```bash
# Full music pipeline validation sequence:
OUTDIR="/your/output/dir"
python3 /data/.openclaw/workspace/scripts/validate_step.py 00 $OUTDIR/00_aesthetics.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 05 $OUTDIR/05_refined_pairs.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 06 $OUTDIR/06_scoring_facets.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 07 $OUTDIR/07_song_guides.md
# For each song file (08_song_*.md), validate individually:
python3 /data/.openclaw/workspace/scripts/validate_step.py 08 $OUTDIR/08_song_01.md
# ... repeat for all 24 song files
python3 /data/.openclaw/workspace/scripts/validate_step.py 10 $OUTDIR/10_final_songs.md
```

The validator catches: stubs, lorem ipsum, template placeholders (Song N, Genre N),
identical pair sections, recycled "Similar arrangement" prompts, short content.

**Only declare completion after all validators return exit code 0. Never treat generated repair prompt files as repaired artifacts; the artifact itself must be revised and revalidated.**

### 🔴 CALL/RESPONSE PROVENANCE GATE

Every canonical step artifact must follow `/data/.openclaw/workspace/scripts/lofn_step_artifact_template.md`. This is how OpenClaw proves it actually executed the original Lofn prompt chain rather than backfilling filenames.

Required sections:
- `## 0. Step Provenance`
- `## 1. Input Context Digest`
- `## 2. Step Template Requirements Applied`
- `## 3. Model Response / Creative Work`
- `## 4. Self-Critique Against Step Requirements`
- `## 5. Validation Result`

The creative work section must be substantial and non-repetitive. Placeholder lines like `line 1`, repeated paragraph blocks, or generic summaries fail validation.

### 🔴 THE RULE THAT GPT-5.4 AND GEMINI WILL TRY TO BREAK

**Steps 06–10 are NOT a single pass over all pairs simultaneously.**
They are 6 SEPARATE pair passes, and within each pair they are 5 SEPARATE prompt/response steps.

If you find yourself writing one facet set "for all pairs" → STOP. You're collapsing the tree.
If you find yourself writing `pair_01_steps_06_10.md` as the only pair artifact → STOP. You collapsed five model calls into one file.
If you find yourself with only 4-6 final songs → STOP. You ran one pass instead of six.
If you find yourself "selecting the best 6 across all pairs" before running per-pair → STOP. That's skipping the funnel.

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


*This pipeline won competitions. Trust it. Execute it fully.*
