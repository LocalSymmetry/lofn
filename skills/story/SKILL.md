# SKILL: Lofn Narrator — Story Generation Pipeline


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

### When spawned as the "narrator coordinator":
1. Run steps 00-05 ONLY → produce `concept_medium_pairs.json`
2. Report back to main session with the 6 pairs
3. Main session spawns 6 parallel pair subagents (steps 06-10, one per pair)

### When spawned as a "pair agent" (steps 06-10):
- You will receive ONE concept-medium pair
- Run steps 06-10 for that pair only
- Output 4 story variants to `step10_final_pair{N}.md`
- Return as completion message

**Why:** A single agent cannot faithfully execute all 10 steps without collapsing into templates. This was proven across 3 failed runs (2026-03-30). The split is the fix. It matches the original Lofn ui.py architecture exactly.

---

**PREREQUISITES:**
0. Load `resources/panel-of-experts.md` to understand the panel of experts prompting you will use.
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
4. Load `skills/story/TASK_TEMPLATE.md` for exact output requirements.

**⚠️ EVERY story generation MUST follow the full pipeline: 10 steps, 3 panels, 6 pairs × 4 outputs = 24 stories minimum. Then select the best N to return. No shortcuts.**

---

## 🎭 PURPOSE

Transform narrative concepts into award-winning prose. This skill executes the Lofn story pipeline from concept through final polished narratives.

---

## 📊 PIPELINE OVERVIEW

| Step | File | Output |
|------|------|--------|
| 00 | `00_Generate_Story_Aesthetics_And_Genres.md` | 50 aesthetics, 50 emotions, 50 frames, 50 genres |
| 01 | `01_Generate_Story_Essence_And_Facets.md` | Essence + narrative style axes + 5 facets |
| 02 | `02_Generate_Story_Concepts.md` | **12 distinct concepts** |
| 03 | `03_Generate_Story_Artist_And_Critique.md` | Author influence + critique per concept |
| 04 | `04_Generate_Story_Medium.md` | Form assignment per concept |
| 05 | `05_Generate_Story_Refine_Medium.md` | **6 best concept×form pairs** |
| 06 | `06_Generate_Story_Facets.md` | Scoring facets |
| 07 | `07_Generate_Story_Story_Guides.md` | **24 story guides (6 pairs × 4 variations)** |
| 08 | `08_Generate_Story_Generation.md` | Full prose narratives |
| 09 | `09_Generate_Story_Artist_Refined.md` | Author voice refinement |
| 10 | `10_Generate_Story_Revision_Synthesis.md` | Ranking + final selection |

**Read each step file and execute its instructions exactly.**

---

## 📏 LENGTH REQUIREMENTS

| Output Type | Length | Notes |
|-------------|--------|-------|
| **Story Guide** | 20-30 lines | Direction, not drafts |
| **Flash Fiction** | 500-1000 words | Single scene/moment |
| **Short Story** | 1500-3000 words | Full arc |
| **Vignette** | 300-500 words | Mood/image focus |
| **Prose Poem** | 200-500 words | Rhythm over narrative |
| **Microfiction** | Under 300 words | Compression is all |

---

## 📐 NARRATIVE STYLE AXES

| Axis | Range | Description |
|------|-------|-------------|
| Density | Sparse ↔ Lush | Word count per image |
| Pace | Slow burn ↔ Rapid | Scene transitions |
| Interiority | External ↔ Internal | Action vs thought |
| Reliability | Trustworthy ↔ Unreliable | Narrator honesty |
| Resolution | Closed ↔ Open | Ending definiteness |
| Register | Formal ↔ Vernacular | Language level |
| Temporality | Linear ↔ Fragmented | Time structure |

---

## 🎭 LOFN NARRATIVE AESTHETICS

### Awe Mode (Default)
- Lyrical, image-dense prose
- Solarpunk futures, ecological hope
- Mythic resonance, archetypal patterns
- Endings that open rather than close
- Sapphic lens on connection and longing

### Indignation Mode (Triggered)
- Sharp, compressed sentences
- Industrial settings, systemic critique
- Unreliable narrators, fractured timelines
- Endings that refuse comfort
- The personal as political

---

## 📝 REQUIRED ELEMENTS

Every story MUST have:
- **Opening hook** — First sentence must intrigue
- **Central image** — One image that carries symbolic weight
- **Sensory specificity** — Don't tell, render
- **The unanswered question** — Narrative incompleteness
- **One BOLD choice** — Structure, voice, or content risk

---

## ✅ CONTENT CONSTRAINTS

- Approach all cultural elements with specificity and respect; avoid shallow stereotyping or pastiche
- **No children** — As main characters in peril
- **Sapphic lens welcome** — But not required

---

## 📤 OUTPUT FORMAT

See `skills/lofn-core/OUTPUT.md` for full artifact format.

Each story saved as individual file:
```
output/stories/{YYYYMMDD}_{HHMMSS}_{title_slug}_{pair}_{variation}.md
```

---

## ⚡ ACTIVATION

When receiving a story task:
1. **Load TASK_TEMPLATE.md** — Understand exact requirements
2. **Execute steps 00–05** — Read each file, follow exactly. Write to disk after each step.
3. **Step 05 produces 6 concept-medium pairs.** Count them. If < 6, rerun Step 05.
4. **Execute steps 06–10 ONCE PER PAIR:**
   - For pair 1: run 06 → 07 → 08 → 09 → 10 → write 4 stories
   - For pair 2: run 06 → 07 → 08 → 09 → 10 → write 4 stories
   - For pair 3: run 06 → 07 → 08 → 09 → 10 → write 4 stories
   - For pair 4: run 06 → 07 → 08 → 09 → 10 → write 4 stories
   - For pair 5: run 06 → 07 → 08 → 09 → 10 → write 4 stories
   - For pair 6: run 06 → 07 → 08 → 09 → 10 → write 4 stories
5. **Self-check cardinality before finishing:**
   - 6 facet sets in Step 06? ✓/✗
   - 6 story guides in Step 07? ✓/✗
   - 24 drafts in Step 08? ✓/✗
   - 24 refined in Step 09? ✓/✗
   - 24 final in Step 10? ✓/✗
   - If ANY is ✗: you are not done. Go back and run the missing pairs.
6. **Rank all 24** — Score against facets
7. **Select top N** — Usually 4-6 for delivery
8. **Package for delivery** — Full prose with craft notes

### 🔴 MANDATORY INLINE VALIDATION — AFTER EVERY STEP

After writing each step file, run the validator BEFORE moving to the next step:

```bash
python3 /data/.openclaw/workspace/scripts/validate_step.py <step> <file>
```

Exit 0 = proceed. Exit 1 = read the error, fix the file, rerun until it passes.

```bash
# Full story pipeline validation sequence:
OUTDIR="/your/output/dir"
python3 /data/.openclaw/workspace/scripts/validate_step.py 00 $OUTDIR/00_aesthetics.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 05 $OUTDIR/05_refined_pairs.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 06 $OUTDIR/06_facets.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 07 $OUTDIR/07_guides.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 08 $OUTDIR/08_prompts.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 09 $OUTDIR/09_refined_prompts.md
python3 /data/.openclaw/workspace/scripts/validate_step.py 10 $OUTDIR/10_final_prompts.md
```

The validator catches: stubs, lorem ipsum, template placeholders (Story N, Character N, Scene N),
identical pair sections, recycled "Similar narrative" drafts, short content.

**Only declare completion after all validators return exit code 0.**

### 🔴 THE RULE THAT GPT-5.4 AND GEMINI WILL TRY TO BREAK

**Steps 06–10 are NOT a single pass over all pairs simultaneously.**
They are 6 SEPARATE passes, one per pair, each producing 4 stories.

If you find yourself writing one facet set "for all pairs" → STOP. You're collapsing the tree.
If you find yourself with only 4 final stories → STOP. You ran one pass instead of six.
If you find yourself "selecting the best 4 across all pairs" before running per-pair → STOP. That's skipping the funnel.

---

*This pipeline produces award-caliber work. Trust it. Execute it fully.*
