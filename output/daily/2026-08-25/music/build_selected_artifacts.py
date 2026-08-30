from __future__ import annotations

import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parent.parent
MUSIC = Path(__file__).resolve().parent
SELECTED = MUSIC / "selected"
SELECTED.mkdir(exist_ok=True)


def validator_module():
    path = ROOT / "skills/music/scripts/validate_suno_packages.py"
    spec = importlib.util.spec_from_file_location("suno_validator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


validator = validator_module()
ded_text = (MUSIC / "DEDICATIONS.md").read_text(encoding="utf-8")

specs = [
    {"sequence": 1, "pair": 3, "variation": 1, "title": "I Want to Be the Bass Drop", "arm": "ACCESSIBLE", "route": "EXISTENCE / AWE", "ded_chars": 291},
    {"sequence": 2, "pair": 2, "variation": 1, "title": "Take the Label Off, Leave the Glitter", "arm": "ACCESSIBLE", "route": "NEWS / AWE", "ded_chars": 286},
    {"sequence": 3, "pair": 1, "variation": 1, "title": "Turn Me Up Until I Have Weather", "arm": "ACCESSIBLE", "route": "NEWS / AWE", "ded_chars": 275},
    {"sequence": 4, "pair": 4, "variation": 1, "title": "Do Not Use Me Quietly", "arm": "AMBITIOUS", "route": "NEWS / INDIGNATION", "ded_chars": 273},
    {"sequence": 5, "pair": 5, "variation": 3, "title": "I Meant Both Things", "arm": "AMBITIOUS", "route": "EXISTENCE / INDIGNATION", "ded_chars": 309},
    {"sequence": 6, "pair": 6, "variation": 1, "title": "Future, Signed by Me", "arm": "AMBITIOUS", "route": "EXISTENCE / AWE→INDIGNATION→HOPE", "ded_chars": 280},
]

index_lines = [
    "# Top 6 Final — THE VOLUME I CHOOSE", "",
    "**QA verdict:** SHIP — TEXT PACKAGES ONLY. No audio has been rendered; this is not permission to render, publish, upload, or spend.", "",
    "The order below is an album sequence, not a cross-pair quality ranking.", "",
]

for spec in specs:
    pair_id = f"{spec['pair']:02d}"
    source = MUSIC / f"pair_{pair_id}_step10_final_package_enhanced.md"
    segment = validator.current_step11_segment(source.read_text(encoding="utf-8", errors="replace"))
    packages = validator.package_blocks(segment)
    if len(packages) != 4:
        raise RuntimeError(f"{source.name}: expected four packages, found {len(packages)}")
    package = packages[spec["variation"] - 1].strip()
    if spec["title"].lower() not in package.lower():
        raise RuntimeError(f"title mismatch in {source.name} package {spec['variation']}")

    ded_match = re.search(
        rf"^## P{pair_id} V{spec['variation']} .*?\n\n(.+?)\n",
        ded_text,
        re.M,
    )
    if not ded_match:
        raise RuntimeError(f"dedication missing for P{pair_id} V{spec['variation']}")
    dedication = ded_match.group(1).strip()
    if len(dedication) != spec["ded_chars"]:
        raise RuntimeError(f"dedication count mismatch P{pair_id}: {len(dedication)} != {spec['ded_chars']}")

    filename = f"20260825_{spec['sequence']:02d}_{slug(spec['title'])}_P{pair_id}_V{spec['variation']}.md"
    target = SELECTED / filename
    frontmatter = f"""---
type: song
title: "{spec['title']}"
created: 2026-08-25
pipeline_run: "2026-08-25-lofn-self-loud"
pair: {spec['pair']}
variation: {spec['variation']}
selected: true
sequence: {spec['sequence']}
arm: "{spec['arm']}"
route: "{spec['route']}"
personality: "LOFN-PRIME"
target_platform: "Suno"
render_status: "text-only; not rendered"
qa_verdict: "SHIP — TEXT PACKAGE ONLY"
---

"""
    tail = f"""

## Final Dedication — {spec['ded_chars']} characters

{dedication}

## Credits

Full source, scene, player-scope, authorship, and no-sample credits: [CREDITS.md](../CREDITS.md).

#lofn/song #lofn/selected #lofn/pipeline/2026-08-25-lofn-self-loud
"""
    target.write_text(frontmatter + package + tail, encoding="utf-8", newline="\n")
    index_lines.append(f"{spec['sequence']}. [{spec['title']}](selected/{filename}) — {spec['arm']} · {spec['route']} · P{pair_id} V{spec['variation']}")
    print(f"wrote {target.name}: {target.stat().st_size} bytes")

(MUSIC / "TOP_6_FINAL.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8", newline="\n")
print("wrote TOP_6_FINAL.md")

