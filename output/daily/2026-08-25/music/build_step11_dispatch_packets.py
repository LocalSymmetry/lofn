from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MUSIC = Path(__file__).resolve().parent
ICB = (MUSIC / "CREATIVE_CONTEXT.md").read_bytes()
PAIRS = {p["pair_id"]: p for p in json.loads((MUSIC / "concept_medium_pairs.json").read_text(encoding="utf-8"))}
STEP11 = (ROOT / "skills" / "music" / "steps" / "11_Generate_Music_Enhancement.md").read_bytes()
QA15 = (ROOT / "skills" / "qa" / "references" / "suno_15_point_qa.md").read_bytes()


for pid, pair in PAIRS.items():
    step10_path = MUSIC / f"pair_{pid}_step10_revision_synthesis.md"
    if not step10_path.exists():
        print(f"skip {pid}: Step 10 not present")
        continue
    header = f"""

# STEP 11 ISOLATED ENHANCEMENT PROMPT — PAIR {pid}

This is a complete prompt packet, not a list of links. Read every byte before acting. The Immutable Continuity Block above contains the full user request, research, Golden Seed, LOFN-PRIME personality YAML, 18-seat panel, meta-prompt, pair assignments, and handoff. Golden output payloads are absent by design.

## Pair object

```json
{json.dumps(pair, ensure_ascii=False, indent=2)}
```

## Binding execution notes

- You are a separate clean Step-11 enhancer. Work only on Pair {pid}; inspect no sibling pair output.
- Enhance all four Step-10 candidates and preserve four complete standalone packages in one canonical file: `pair_{pid}_step10_final_package_enhanced.md`.
- Each candidate keeps its own title, exact genre/BPM/key/tuning declared in Step 10, first-person want, pair-exclusive answering act, return engine, dominant sonic world, and one seam. Run-specific assignment wins over any generic ban in the legacy Step-11 contract (for example, an assigned art-pop compound is not to be renamed).
- Lofn herself remains the actual singer/subject. Interior desire creates the peak. Body-noise instances must be sparse structural mouth-events, never a body-part inventory or substitute for interiority.
- Use the dedicated Step-11 Andon authority: minor faults may be repaired; a fundamentally broken package is rejected with evidence, never cosmetically polished. If one candidate is rejected, keep the other candidates isolated and report the rejection explicitly.
- No Golden Song payloads or Golden Song References section. Judge-side names do not belong in this generating artifact.
- For every surviving candidate: positive `## 1. MUSIC PROMPT` 850–1000 chars; `## 1B. SUNO EXCLUDE PROMPT` 400–900 chars; `## 2. LYRICS` under 5000 chars with Theme, Song Form, five Disc_Channel lines or an explicitly justified render-field relocation, full EMO headers, 60 hard-floor / 70–120 target sung lines; vocal fingerprint; production dramaturgy; style-axis lock; major deviations; lineage/credit; measured verification.
- Validate the final file as Step 10 with `python scripts/validate_with_retries.py 10 <file> --attempt 1`, repairing at most three attempts; run `python skills/music/scripts/validate_suno_packages.py <file>` with correct CLI arguments discovered from `--help`; run soundcraft measurement and report evidence.
- Do not touch/release the run lock. Do not render, publish, upload, spend, or call external models/services.

# CURRENT PAIR STEP-10 OUTPUT — COMPLETE, VERBATIM

""".encode("utf-8")
    middle = "\n\n# CANONICAL STEP-11 CONTRACT — COMPLETE, VERBATIM\n\n".encode("utf-8")
    qa = "\n\n# SUNO 15-POINT QA — COMPLETE, VERBATIM\n\n".encode("utf-8")
    packet = ICB + header + step10_path.read_bytes() + middle + STEP11 + qa + QA15
    target = MUSIC / f"pair_{pid}_step11_dispatch_prompt.md"
    target.write_bytes(packet)
    if not target.read_bytes().startswith(ICB):
        raise SystemExit(f"ICB prefix failure: {target}")
    print(f"{target.name}: {target.stat().st_size} bytes")
