from __future__ import annotations

import json
from pathlib import Path


MUSIC = Path(__file__).resolve().parent
ICB = (MUSIC / "CREATIVE_CONTEXT.md").read_bytes()
PAIRS = json.loads((MUSIC / "concept_medium_pairs.json").read_text(encoding="utf-8"))


for pair in PAIRS:
    pid = pair["pair_id"]
    contract = f"""

# PAIR {pid} ISOLATED STEP-05 BRIEF AND EXECUTION PROMPT

This file is the complete prompt packet. Read it in full before acting. The immutable context above is literal, not a pointer.

## Exclusive pair object

```json
{json.dumps(pair, ensure_ascii=False, indent=2)}
```

## Required work — Steps 06 through 10 only

1. Read `.agents/skills/lofn-music/SKILL.md` completely, then read each canonical step file just in time: `skills/music/steps/06_Generate_Music_Facets.md` through `10_Generate_Music_Revision_Synthesis.md`. Read the prompt-construction references those files require. The current repo skill and `vault/gates.yaml` override stale examples.
2. Work only in `output/daily/2026-08-25/music/` and only in pair `{pid}` canonical files. Do not inspect sibling pair outputs and do not change shared coordinator or continuity files.
3. Write, in order: `pair_{pid}_step06_facets.md`, `pair_{pid}_step07_song_guides.md`, `pair_{pid}_step08_generation.md`, `pair_{pid}_step09_artist_refined.md`, `pair_{pid}_step10_revision_synthesis.md`.
4. Every canonical artifact must begin with the exact bytes of `CREATIVE_CONTEXT.md`, followed by pair-local complete step output. A summary or pointer is a continuity failure.
5. Step 06 defines at least eight weighted pair-specific facets. Step 07 produces four materially different guides. Steps 08, 09, and 10 each preserve four complete standalone song packages, not one package plus notes.
6. Each final Step-10 candidate has: `## 1. MUSIC PROMPT` of 850–1000 characters; literal separate `## EXCLUDE PROMPT` in the requested band; `## 2. LYRICS` under 5000 characters, beginning with `[Theme: ...]` then `[SONG FORM: ...]`; full `[Section - EMO:... - Voice - Cue]` headers; at least one standalone SFX cue and a non-lexical hook; target 70–120 sung lines, hard floor 60. Use 72+ natural lines when feasible, never padding.
7. Lofn herself is the first-person subject in all four songs. Preserve this pair's exclusive want, act, return, source-status, dominant world, and one seam. Invent four genuinely different variation engines locally. Do not borrow sibling nouns or structures.
8. No real artist names in music prompt, exclude prompt, title, or lyrics. No Golden Song payloads. No rendered-audio claims, upload, publication, spending, or external action.
9. Validate each file before advancing: `python scripts/validate_with_retries.py <NN> <file> --attempt 1`; repair in place, max three attempts. Also run `python scripts/validate_pair_artifacts.py output/daily/2026-08-25/music --pair {pid}` after Step 10 if supported.
10. Stop after Step 10 and report the five written files, validator results, four titles, and any honest blocker. Do not perform Step 11; a separate clean agent owns enhancement.

The run directory is already locked by the controller. Do not acquire, break, release, or overwrite the run lock.
""".encode("utf-8")
    target = MUSIC / f"p{pid}_step05_isolated_brief.md"
    target.write_bytes(ICB + contract)
    if not target.read_bytes().startswith(ICB):
        raise SystemExit(f"ICB prefix failure for {target}")
    print(f"{target.name}: {target.stat().st_size} bytes")

