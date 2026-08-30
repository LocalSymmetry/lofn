from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MUSIC = Path(__file__).resolve().parent
ICB = (MUSIC / "CREATIVE_CONTEXT.md").read_bytes()
PAIRS = {p["pair_id"]: p for p in json.loads((MUSIC / "concept_medium_pairs.json").read_text(encoding="utf-8"))}
SELECTIONS = json.loads((MUSIC / "selected_candidates.json").read_text(encoding="utf-8"))
STEP11 = (ROOT / "skills" / "music" / "steps" / "11_Generate_Music_Enhancement.md").read_bytes()
QA15 = (ROOT / "skills" / "qa" / "references" / "suno_15_point_qa.md").read_bytes()


def validator_module():
    path = ROOT / "skills" / "music" / "scripts" / "validate_suno_packages.py"
    spec = importlib.util.spec_from_file_location("suno_validator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = validator_module()

for selected in SELECTIONS:
    pid = selected["pair_id"]
    pair = PAIRS[pid]
    source = MUSIC / selected["source_file"]
    text = source.read_text(encoding="utf-8", errors="replace")
    segment = validator.current_step11_segment(text)
    packages = validator.package_blocks(segment)
    variation = int(selected["variation"])
    if len(packages) != 4:
        raise RuntimeError(f"{source.name}: expected four packages, found {len(packages)}")
    package = packages[variation - 1].strip()
    if selected["title"].lower() not in package.lower():
        raise RuntimeError(f"{pid}: title mismatch in variation {variation}")

    header = f"""

# STEP 11 ISOLATED ENHANCEMENT PROMPT — {pid} SELECTED V{variation}

Read every byte before acting. The immutable context above contains Sunna's full personality, research, Golden Seed, panels, pair assignment, and modernization contract. Work on this selected package only and inspect no sibling output.

## Pair object

```json
{json.dumps(pair, ensure_ascii=False, indent=2)}
```

## Selection record

```json
{json.dumps(selected, ensure_ascii=False, indent=2)}
```

## Binding execution notes

- Enhance this one selected candidate into `pair_{int(pid[1:]):02d}_step10_selected_enhanced.md`; do not invent replacement candidates.
- Preserve the title unless an Andon-level defect makes it structurally impossible. Preserve the first-person want, answering action, return engine, ancestral invariant, century replacements, now-marker, swagger gap, tribute-act falsifier, and one seam.
- Sunna's personality wins any conflict with centre-panel defaults. Direct adult address, frivolity, flirtation, hope, and polish are allowed. No complete period ensemble or retro vocal posture may return during refinement.
- Positive `## 1. MUSIC PROMPT` 850–1000 chars; `## 1B. SUNO EXCLUDE PROMPT` 400–900; `## 2. LYRICS` under 5000 chars, 70–120 sung-line target / 60 hard floor; include title and lineage sidecars outside generator fields.
- Run Step-10 validation with at most three attempts, Suno package validation, and soundcraft measurement. Preserve the exact ICB byte prefix in the enhanced canonical artifact.
- Do not touch the run lock. Do not render, upload, publish, spend, or call an external model/service.

# SELECTED STEP-10 PACKAGE — COMPLETE, VERBATIM

{package}

# CANONICAL STEP-11 CONTRACT — COMPLETE, VERBATIM

""".encode("utf-8")
    tail = "\n\n# SUNO 15-POINT QA — COMPLETE, VERBATIM\n\n".encode("utf-8") + QA15
    packet = ICB + header + STEP11 + tail
    target = MUSIC / f"pair_{int(pid[1:]):02d}_step11_dispatch_prompt.md"
    target.write_bytes(packet)
    if not target.read_bytes().startswith(ICB):
        raise RuntimeError(f"ICB prefix failure: {target.name}")
    print(f"{target.name}: {target.stat().st_size} bytes; {selected['title']}")
