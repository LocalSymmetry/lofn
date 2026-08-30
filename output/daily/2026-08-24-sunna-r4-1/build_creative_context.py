from __future__ import annotations

import hashlib
from pathlib import Path


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]

PARTS = [
    ("RESEARCH BRIEF / USER REQUEST", RUN / "00_research_brief.md"),
    ("SEED LINEAGE", RUN / "01_seed_lineage.md"),
    ("GOLDEN SEED", RUN / "02_golden_seed.md"),
    ("ORCHESTRATOR PANEL LEDGER", RUN / "03_orchestrator_panel_debate.md"),
    ("ORCHESTRATOR METAPROMPT", RUN / "04_orchestrator_metaprompt.md"),
    ("PAIR ASSIGNMENTS", RUN / "05_orchestrator_pair_assignments.md"),
    ("AUDIO HANDOFF / CONFLICT RESOLUTION", RUN / "06_audio_handoff.md"),
    ("ACTIVE PERSONALITY — FULL CANONICAL SUNNA R4.1 YAML", ROOT / "skills/orchestration/personalities/sunna.yaml"),
]

chunks = [
    "# CREATIVE CONTEXT — IMMUTABLE CONTINUITY BLOCK\n\n"
    "**Run:** `2026-08-24-sunna-r4-1`  \n"
    "**Injection rule:** pass this file verbatim to every music step 00–11. Do not summarize it.  \n"
    "**Quarantine:** no archived Golden Song payload is present.  \n"
    "**Active personality only:** SUNNA R4.1. Lofn-Prime is not a creative assignment.\n"
]

for label, path in PARTS:
    if not path.exists():
        raise SystemExit(f"missing source: {path}")
    chunks.append(f"\n\n---\n\n# ===== {label} =====\n\n")
    chunks.append(path.read_text(encoding="utf-8"))

body = "".join(chunks).rstrip() + "\n"
source_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
body += f"\n<!-- ICB_SOURCE_SHA256 {source_digest} -->\n"

targets = [RUN / "CREATIVE_CONTEXT.md", RUN / "music/CREATIVE_CONTEXT.md"]
for target in targets:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8", newline="\n")

final_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
after_count = body.count("(after ")
print(f"ICB_BYTES={len(body.encode('utf-8'))}")
print(f"ICB_SHA256={final_digest}")
print(f"PERSONALITY_YAML_BYTES={len((ROOT / 'skills/orchestration/personalities/sunna.yaml').read_bytes())}")
print(f"AFTER_SPEAKER_COUNT={after_count}")
for target in targets:
    print(target)

if after_count != 18:
    raise SystemExit(f"expected exactly 18 '(after ' speaker tags, found {after_count}")
