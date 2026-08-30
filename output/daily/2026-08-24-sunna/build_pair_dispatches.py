from __future__ import annotations

import json
from pathlib import Path


RUN = Path(__file__).resolve().parent
MUSIC = RUN / "music"
ICB = (RUN / "CREATIVE_CONTEXT.md").read_text(encoding="utf-8")
PAIRS = json.loads((MUSIC / "concept_medium_pairs.json").read_text(encoding="utf-8"))

header = """# PAIR DISPATCH — READ THIS FILE COMPLETELY BEFORE STEP 06

This is the complete parent-injected pair packet. The Creative Context below is verbatim and immutable. Do not summarize it. After this packet, load and execute each step contract only when its turn arrives: 06 → 07 → 08 → 09 → 10 → 11. Each step is a separate reasoning/writing turn and a separate canonical artifact. Validate after every step and stop after three failed attempts.

## Creative order

1. Golden Seed / lineage / SUNNA R4.
2. Pair's dangerous requirement.
3. Creative permission.
4. Four complete songs from this pair.
5. Structural completeness and validators.

## Hard boundaries

- Write only this pair's `pair_NN_*` artifacts in the run `music` directory.
- Do not read sibling pair outputs or borrow their titles, section maps, hook grammar, props, facts, or seam blocks.
- Four complete variations must survive Steps 08, 09, 10, and 11.
- Active personality is SUNNA R4 only. No LOFN-PRIME voice.
- No real artist names in style, exclude, lyrics, or titles.
- No audio rendering, API generation, publication, upload, spend, commit, push, or run-lock operation.
- Step 11 is the dedicated non-Fusion enhancement contract. Golden Song payloads remain quarantined; calibrate only against the Golden Move.
- Final Step-11 file name: `pair_NN_step10_final_package_enhanced.md`, containing all four packages.

## Required validation commands

Run from the repository root after each artifact:

`python scripts/validate_with_retries.py STEP FILE --attempt N`

After Step 10/11:

`python scripts/validate_pair_artifacts.py output/daily/2026-08-24-sunna/music NN --attempt N`

The parent runs cross-pair gates. Do not claim completion without the pair-level PASS.

---

"""

dispatch_dir = MUSIC / "dispatch"
dispatch_dir.mkdir(parents=True, exist_ok=True)

for pair in PAIRS:
    nn = f"{pair['pair_num']:02d}"
    brief = (MUSIC / f"p{nn}_step05_isolated_brief.md").read_text(encoding="utf-8")
    pair_json = json.dumps(pair, ensure_ascii=False, indent=2)
    out = (
        header.replace("NN", nn)
        + "\n# ===== IMMUTABLE CREATIVE CONTEXT — VERBATIM =====\n\n"
        + ICB
        + "\n\n# ===== PAIR-ISOLATED STEP-05 BRIEF =====\n\n"
        + brief
        + "\n\n# ===== MACHINE-READABLE PAIR OBJECT =====\n\n```json\n"
        + pair_json
        + "\n```\n"
    )
    target = dispatch_dir / f"pair_{nn}_dispatch.md"
    target.write_text(out, encoding="utf-8", newline="\n")
    print(f"{target.name}\t{target.stat().st_size}")
