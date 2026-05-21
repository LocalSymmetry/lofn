# Lofn Video/Director — Subagent Architecture Pattern

## THE CORRECT SPLIT (from original Lofn ui.py)

The original Lofn app ran this exact pattern:
1. `generate_concept_mediums()` — steps 00-05 — ONE call, returns all concept-medium pairs
2. `select_best_pairs()` — panel votes on top N pairs
3. `generate_prompts_for_pair(pair)` — steps 06-10 — called ONCE PER PAIR, in parallel

**This is the mandatory architecture for all future runs.**

---

## PRE-CREATIVE ORCHESTRATOR GATE — REQUIRED

Before coordinator Step 00, validate that the run has a real Lofn-Core + orchestrator packet:

```bash
python3 /data/.openclaw/workspace/scripts/validate_orchestrator_packet.py <run_dir>
```

The packet must include substantial seed lineage, Golden Seed/core seed, the original Lofn panel object (`Special Flairs`, `Concept Panel`, `Medium Panel`, `Context & Marketing Panel`) with Devil's Advocate / Hyper-Skeptic roles, metaprompt, pair assignments with rationale, and modality handoff. If this fails, do not proceed. Launch or request `lofn-orchestrator` work instead.

---

## CANONICAL STEP ARTIFACT PROVENANCE — REQUIRED

Every canonical step file must follow `/data/.openclaw/workspace/scripts/lofn_step_artifact_template.md` and include:

1. `## 0. Step Provenance` — exact step file loaded, input artifacts used, model call mode, validation command.
2. `## 1. Input Context Digest` — concrete digest of prior artifacts; must name actual concepts/facets/media, not generic task text.
3. `## 2. Step Template Requirements Applied` — specific requirements from the loaded step prompt.
4. `## 3. Model Response / Creative Work` — the actual creative output.
5. `## 4. Self-Critique Against Step Requirements` — adversarial check.
6. `## 5. Validation Result` — paste validator output after pass.

A file with the right filename but without these sections is a backfilled artifact and fails. Do not write placeholders, repeated paragraphs, or self-check claims contradicted by the artifact body.

---

## SUBAGENT 1: Steps 00-05 (Concept-Medium Generation)

Receives: orchestrator output (metaprompt, personality, panel, constraint axes)

Executes as **six separate LLM turns**, matching original `generate_concept_mediums()` in `lofn/llm_integration.py`:
- Step 00: read `steps/00_Generate_Video_Aesthetics_And_Genres.md`, call the model, write `step00_aesthetics_and_genres.md` using the canonical provenance template. Provenance must cite the validated orchestrator packet
- Step 01: read `steps/01_Generate_Video_Essence_And_Facets.md`, call the model using Step 00 output, write `step01_essence_and_facets.md` using the canonical provenance template
- Step 02: read `steps/02_Generate_Video_Concepts.md`, call the model using Step 01 output, write `step02_concepts.md` using the canonical provenance template
- Step 03: read `steps/03_Generate_Video_Artist_And_Critique.md`, call the model using Step 02 output, write `step03_artist_and_critique.md` using the canonical provenance template
- Step 04: read `steps/04_Generate_Video_Medium.md`, call the model using Step 03 output, write `step04_medium.md` using the canonical provenance template
- Step 05: read `steps/05_Generate_Video_Refine_Medium.md`, call the model using Step 04 output, write `step05_refine_medium.md` using the canonical provenance template and `concept_medium_pairs.json` (6 pairs)

**Do not combine Steps 00–05 into one prompt, one response, or renamed summary files.** Summary files are not canonical original-Lofn step outputs.

### Mandatory validation + retry loop for Steps 00–05
After writing each step artifact, run:

```bash
python3 /data/.openclaw/workspace/scripts/validate_with_retries.py <STEP> <FILE> --attempt 1
```

If validation fails, repair in place and rerun attempts 2 and 3. After 3 failed attempts, stop and escalate. Do **not** continue with a failed artifact.

**STOP HERE. Do not proceed to step 06.**

---

## SUBAGENTS 2-7: Steps 06-10 (One Per Pair)

Each receives:
- The orchestrator metaprompt
- ONE specific concept-medium pair (concept, camera style, cinematic influence)
- The constraint axes
- The panel composition

Each executes (for its ONE pair only) as **five separate LLM turns**, matching original `generate_prompts_for_pair()` / prompt chain in `lofn/llm_integration.py`:
- Step 06: read `steps/06_Generate_Video_Facets.md`, call the model, write `pair_{NN}_step06_facets.md` using the canonical provenance template. Provenance must cite the Golden Seed, orchestrator metaprompt, pair assignment, Step 05 concept-medium pair, and prior coordinator outputs
- Step 07: read `steps/07_Generate_Video_Aspects_Traits.md`, call the model using Step 06 output, write `pair_{NN}_step07_aspects_traits.md` using the canonical provenance template
- Step 08: read `steps/08_Generate_Video_Generation.md`, call the model using Step 07 output, write `pair_{NN}_step08_generation.md` using the canonical provenance template
- Step 09: read `steps/09_Generate_Video_Artist_Refined.md`, call the model using Step 08 output, write `pair_{NN}_step09_artist_refined.md` using the canonical provenance template
- Step 10: read `steps/10_Generate_Video_Revision_Synthesis.md`, call the model using Step 09 output, write `pair_{NN}_step10_revision_synthesis.md` using the canonical provenance template

**Do not combine Steps 06–10 into one prompt, one response, or one omnibus file.** A rollup may be created afterward, but it is not canonical.

### Mandatory validation + retry loop for Steps 06–10
After writing each pair step artifact, run:

```bash
python3 /data/.openclaw/workspace/scripts/validate_with_retries.py <STEP> <FILE> --attempt 1
```

If validation fails, repair in place and rerun attempts 2 and 3. After 3 failed attempts, stop that pair and return the exact validator failure. Do not claim the pair is complete if a prior step failed.

Outputs to disk: five separate step files for its pair number
Returns: Step 10 final video prompts as output text

---

## ORCHESTRATION FLOW

```
Main session
  └── spawns Subagent 1 (steps 00-05)
         └── writes concept_medium_pairs.json
  └── reads concept_medium_pairs.json
  └── spawns Subagents 2-7 in parallel (one per pair)
         └── each writes full shot prompt package
  └── collects all 6 outputs
  └── QA gate
  └── Render via Veo3/Runway or deliver to Scientist
```

---

## concept_medium_pairs.json format
```json
[
  {
    "pair_num": 1,
    "concept": "Full refined shot/scene concept",
    "medium": "Camera style, format, rendering technique",
    "artist_influence": "Named director/cinematographer"
  }
]
```

## OUTPUT FORMAT FOR PAIR SUBAGENTS

Each pair subagent must return in step10:
- 4 video prompt variants, each complete and distinct
- Each variant: scene setup, camera movement, lighting, duration, audio/music notes
- Format compatible with Veo3 (see VEO3_PROMPT_FRAMEWORK.md) or Runway

Written to: `step10_final_pair{N}.md`
