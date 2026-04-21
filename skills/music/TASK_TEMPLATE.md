# Lofn Music — Subagent Architecture Pattern

## THE CORRECT SPLIT (from original Lofn ui.py)

The original Lofn app ran this exact pattern:
1. `generate_concept_mediums()` — steps 00-05 — ONE call, returns all concept-medium pairs
2. `select_best_pairs()` — panel votes on top N pairs
3. `generate_prompts_for_pair(pair)` — steps 06-10 — called ONCE PER PAIR, in parallel

**This is the mandatory architecture for all future runs.**

---

## SUBAGENT 1: Steps 00-05 (Concept-Medium Generation)

Receives: orchestrator output (metaprompt, personality, panel, constraint axes)

Executes:
- Step 00: Generate 50 aesthetics, emotions, compositions, genres
- Step 01: Extract essence, define style axes, creativity spectrum
- Step 02: Generate 12 concepts
- Step 03: Pair each concept with artist influence + critique
- Step 04: Assign medium/production style to each concept
- Step 05: Critique and refine → select 6 best concept-medium pairs

Outputs to disk: step00 through step05 files + `concept_medium_pairs.json` (6 pairs)

**STOP HERE. Do not proceed to step 06.**

---

## SUBAGENTS 2-7: Steps 06-10 (One Per Pair)

Each receives:
- The orchestrator metaprompt
- ONE specific concept-medium pair (name, concept text, medium/production style)
- The constraint axes
- The panel composition

Each executes (for its ONE pair only):
- Step 06: Generate facets for scoring
- Step 07: Write detailed song guide (mood, instrumentation, structure, BPM, key)
- Step 08: Generate Suno/Udio style prompt
- Step 09: Rewrite prompt in artist's voice
- Step 10: Critique, rank, synthesize → 4 final outputs (music prompt + full lyrics)

Outputs to disk: step06 through step10 files for its pair number
Returns: 4 final song prompts + lyrics as output text

---

## ORCHESTRATION FLOW

```
Main session
  └── spawns Subagent 1 (steps 00-05)
         └── writes concept_medium_pairs.json
  └── reads concept_medium_pairs.json
  └── spawns Subagents 2-7 in parallel (one per pair)
         └── each writes full song (prompt + lyrics)
  └── collects all 6 songs
  └── QA gate
  └── Deliver to Telegram
```

---

## concept_medium_pairs.json format
```json
[
  {
    "pair_num": 1,
    "concept": "Full refined concept text",
    "medium": "Full production style text",
    "artist_influence": "Named artist"
  }
]
```

## OUTPUT FORMAT FOR PAIR SUBAGENTS

Each pair subagent must return in step10 as a **single self-contained markdown file** with ALL of the following sections in this exact order:

```markdown
# [Song Title]

**Form:** [Declared song form — e.g. Rondo (A-B-A-C-A), Through-Composed, Strophic, etc.]
**Pair:** [N]
**Genre/Style:** [brief genre label]

---

## Suno Music Prompt
[The music_prompt — ≤1000 chars, no artist names, includes: genre tags, mood/EMO tags, instrumentation, vocal description (female, age range, tone, quirks), BPM, progression notes]

## Suno Lyrics Prompt
[The full lyrics — ≥50 lines, with [Section] tags, EMO tags on each section, varied stanza lengths, declared rhyme scheme, NO uniform 4-line stanzas, NO V1/C/V2/C/B/C without justification]
```

**Both prompts MUST be in the same file.** Do not split music prompts into a separate `09_suno_prompts.md`. Each song is one file, self-contained and ready to copy-paste directly into Suno.

Written to: `step10_final_pair{N}.md`

**CHECKLIST before saving:**
- [ ] Title present
- [ ] Song form declared
- [ ] Suno Music Prompt section present (≤1000 chars)
- [ ] Suno Lyrics Prompt section present (≥50 lines)
- [ ] Female vocals specified in music prompt
- [ ] EMO tag(s) in music prompt
- [ ] No artist names in music prompt
- [ ] Stanza lengths vary (not all 4-line)
- [ ] Not default V1/C/V2/C/B/C structure (or justified)
