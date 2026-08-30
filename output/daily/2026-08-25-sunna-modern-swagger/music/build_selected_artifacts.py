from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MUSIC = Path(__file__).resolve().parent
SELECTED_DIR = MUSIC / "selected"
SELECTED_DIR.mkdir(exist_ok=True)
SELECTIONS = {x["pair_id"]: x for x in json.loads((MUSIC / "selected_candidates.json").read_text(encoding="utf-8"))}
DEDICATIONS = (MUSIC / "DEDICATIONS.md").read_text(encoding="utf-8")


def validator_module():
    path = ROOT / "skills" / "music" / "scripts" / "validate_suno_packages.py"
    spec = importlib.util.spec_from_file_location("suno_validator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def dedication(pair_id: str, title: str) -> tuple[str, int]:
    match = re.search(
        rf"^## {re.escape(pair_id)} — {re.escape(title)} — (\d+) characters\s*\n\s*\n(.+?)(?=\n\n## |\Z)",
        DEDICATIONS,
        re.M | re.S,
    )
    if not match:
        raise RuntimeError(f"missing dedication for {pair_id} {title}")
    declared = int(match.group(1))
    body = match.group(2).strip()
    if len(body) != declared or declared > 500:
        raise RuntimeError(f"dedication count mismatch for {pair_id}: declared {declared}, actual {len(body)}")
    return body, declared


validator = validator_module()


def current_step10_text(text: str) -> str:
    matches = list(re.finditer(r"^#\s+.*\bStep\s*10\b.*$", text, re.I | re.M))
    return text[matches[-1].start():] if matches else text

# Declaration first, then desire/reciprocity, then latency/awe, then owned future.
sequence = ["P06", "P02", "P03", "P04", "P01", "P05"]
index = [
    "# Top 6 Final — KEEP THE BODY; REPLACE THE CENTURY", "",
    "**Status:** selected and independently Step-11 enhanced text packages. No audio rendered; no permission to render, publish, upload, or spend.", "",
    "The order is an emotional sequence, not a cross-pair score.", "",
]

for number, pid in enumerate(sequence, 1):
    selected = SELECTIONS[pid]
    pair_num = int(pid[1:])
    source = MUSIC / f"pair_{pair_num:02d}_step10_selected_enhanced.md"
    text = source.read_text(encoding="utf-8", errors="replace")
    packages = validator.package_blocks(current_step10_text(text))
    if len(packages) != 1:
        raise RuntimeError(f"{source.name}: expected one enhanced package, found {len(packages)}")
    package = packages[0].strip()
    if selected["title"].lower() not in package.lower():
        raise RuntimeError(f"{pid}: enhanced title mismatch")
    ded, ded_chars = dedication(pid, selected["title"])
    filename = f"20260825_{number:02d}_{slug(selected['title'])}_{pid}.md"
    target = SELECTED_DIR / filename
    frontmatter = f'''---
type: song
title: "{selected['title']}"
created: 2026-08-25
pipeline_run: "2026-08-25-sunna-modern-swagger"
pair: {pair_num}
source_variation: {selected['variation']}
selected: true
sequence: {number}
arm: "{selected['arm']}"
axis: "{selected['axis']}"
personality: "SUNNA"
target_platform: "Suno"
render_status: "text-only; not rendered"
qa_verdict: "SHIP — TEXT PACKAGES ONLY"
---

'''
    tail = f'''

## Final Dedication — {ded_chars} characters

{ded}

## Credits

Full source, scene, authorship, and no-sample credits: [CREDITS.md](../CREDITS.md).

#lofn/song #lofn/sunna #lofn/selected #lofn/pipeline/2026-08-25-sunna-modern-swagger
'''
    target.write_text(frontmatter + package + tail, encoding="utf-8", newline="\n")
    index.append(f"{number}. [{selected['title']}](selected/{filename}) — {selected['arm']} · {selected['axis']} · {pid} V{selected['variation']}")
    print(f"wrote {target.name}: {target.stat().st_size} bytes")

(MUSIC / "TOP_6_FINAL.md").write_text("\n".join(index) + "\n", encoding="utf-8", newline="\n")
print("wrote TOP_6_FINAL.md")
