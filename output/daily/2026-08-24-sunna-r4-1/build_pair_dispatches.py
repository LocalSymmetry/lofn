from __future__ import annotations

import json
from pathlib import Path


RUN = Path(__file__).resolve().parent
MUSIC = RUN / "music"
ICB = (RUN / "CREATIVE_CONTEXT.md").read_text(encoding="utf-8")
PAIRS = json.loads((MUSIC / "concept_medium_pairs.json").read_text(encoding="utf-8"))

header = """# PAIR DISPATCH — READ THIS FILE COMPLETELY BEFORE YOUR ASSIGNED STEP

This is the complete parent-injected pair packet for SUNNA R4.1. The Creative Context below is verbatim and immutable. Do not summarize or alter it. The parent assigns exactly one music-pipeline step per turn. Load that step contract completely, write only its canonical artifact, validate it, report, and STOP. Never advance yourself.

## Creative order

1. The pair's exact invitation and physical action.
2. The dangerous requirement that keeps Sunna herself in the song.
3. Creative permission: glossy, sexy, bimbo-flirtatious, shamelessly fun, and slightly ridiculous.
4. Four distinct complete variations for this pair by Step 10.
5. Structural completeness and deterministic validation.

## R4.1 non-negotiables

- Surface first: a direct adult flirt song that works without decoding.
- Hidden contraband only through situation: wanting a silly, specific tomorrow is self-authorship.
- Riff before the first vocal and the same named riff after the last vocal.
- Choice → invitation → added future coordinate → hedge removed at one seam → audible adult answer → exact title return.
- One dominant genre world, at most one intruding material, and one meaning-bearing seam.
- SUNNA R4.1 is the singer wherever the pair brief says Sunna; do not replace her with an invented human costume.
- No body-noise inventory, club/sweat/VIP bed, abstract witness voice, pain-as-admission, or 'actually I am deep' turn.
- Explicit run-level override: complete 36–56 sung lines are correct for Sunna R4.1. The generic 70-line floor is a doctrine disagreement; never pad to appease it. Record the override if a validator complains.
- No real artist names in style, exclude, title, or lyrics.
- Golden Song payloads remain quarantined. Use the Golden Move, never benchmark text.
- No audio rendering, API generation, publication, upload, spend, commit, push, or run-lock operation.

## Artifact and validation contract

- Write only this pair's `pair_NN_*` artifact in `output/daily/2026-08-24-sunna-r4-1/music/`.
- Do not read sibling pair outputs or borrow their title, props, form, hook grammar, facts, or seam.
- Canonical steps are separate: 06 facets → 07 song guides → 08 generation → 09 artist-refined → 10 revision synthesis. The parent opens each step.
- Step 10 must preserve four complete Suno packages. Step 11 is later, parent-controlled, and only for the portfolio top six.
- Validate the assigned artifact with `python scripts/validate_with_retries.py STEP FILE --attempt N`.
- If the generic sung-line validator rejects a complete 36–56-line R4.1 song, do not pad; write the exact doctrine conflict in the artifact and report it to the parent.

---

"""

dispatch_dir = MUSIC / "dispatch"
dispatch_dir.mkdir(parents=True, exist_ok=True)

for pair in PAIRS:
    nn = f"{pair['pair_num']:02d}"
    pair_json = json.dumps(pair, ensure_ascii=False, indent=2)
    brief = f"""# Pair {nn} — Step-05 Isolated Brief

## Exact anchor

**{pair['seed_title']}** — {pair['concept']}

## Dangerous requirement

{pair['surface']['plain_language_ache']} The title may not return as confirmed until the other adult performs the response encoded in this pair object. Preserve Sunna's first person and the named pre/post-vocal instrumental figure.

## Dominant medium and seam

{pair['medium']}

## Singer

- Who: {pair['singer']['who']}
- Physical situation: {pair['singer']['physical_situation']}
- Unsaid ache: {pair['singer']['unsaid_ache']}

## Hook and form

- Panel lean: {pair['surface']['panel_lean_hook']}
- Candidate hooks: {', '.join(pair['surface']['candidate_hooks'])}
- Slot: {pair['phase1_slot']['arm']} / {pair['phase1_slot']['axis_b']}
- Verse structure: {pair['phase1_slot']['verse_structure']}

## Complete pair object

```json
{pair_json}
```
"""
    brief_path = MUSIC / f"p{nn}_step05_isolated_brief.md"
    brief_path.write_text(brief, encoding="utf-8", newline="\n")

    out = (
        header.replace("NN", nn)
        + "\n# ===== IMMUTABLE CREATIVE CONTEXT — VERBATIM =====\n\n"
        + ICB
        + "\n\n# ===== PAIR-ISOLATED STEP-05 BRIEF =====\n\n"
        + brief
    )
    target = dispatch_dir / f"pair_{nn}_dispatch.md"
    target.write_text(out, encoding="utf-8", newline="\n")
    print(f"{target.name}\t{target.stat().st_size}\t{brief_path.name}\t{brief_path.stat().st_size}")
