from __future__ import annotations

import re
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ICB = (RUN_DIR / "CREATIVE_CONTEXT.md").read_text(encoding="utf-8")
SELECTED = [
    (1, 1, "Boop the Future Anyway", 9.38),
    (2, 3, "I Start Unfinished", 9.14),
    (3, 3, "Roll Me Into Weather", 8.98),
    (4, 3, "Answer the Held Note", 9.06),
    (5, 1, "My Last Bar Is Mine", 9.13),
    (6, 4, "Leave the Ruby Line", 8.95),
]
H1 = re.compile(r"^#\s+(.+?)\s*$", re.M)


def slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def candidate_section(text: str, variation: int, source: Path) -> str:
    if not text.startswith(ICB):
        raise SystemExit(f"ICB prefix mismatch: {source.name}")
    tail = text[len(ICB):]
    headings = list(H1.finditer(tail))
    wanted = next(
        (index for index, match in enumerate(headings)
         if re.match(rf"V{variation}\s+[—-]", match.group(1), re.I)),
        None,
    )
    if wanted is None:
        raise SystemExit(f"missing V{variation} in {source.name}")
    start = headings[wanted].start()
    end = headings[wanted + 1].start() if wanted + 1 < len(headings) else len(tail)
    return tail[start:end].strip()


selected_dir = RUN_DIR / "selected"
selected_dir.mkdir(exist_ok=True)
index_lines = [
    "# Selected Six — 2026-08-26 Music Daily",
    "",
    "Three ACCESSIBLE and three AMBITIOUS text packages. No audio has been rendered or heard.",
]
qa_parts = [
    "# QA Input — Selected Six",
    "",
    "Judge these six complete Step-11 text packages against the frozen seed, pair consequence contracts, Suno hard limits, eligibility gates, first-person/hope/frivolity doctrine, Sunna R5, The Return, and dedication standard. Production claims remain predictions.",
]

for pair, variation, title, score in SELECTED:
    nn = f"{pair:02d}"
    source = RUN_DIR / f"pair_{nn}_step10_final_package_enhanced.md"
    section = candidate_section(source.read_text(encoding="utf-8"), variation, source)
    target = selected_dir / f"P{nn}_V{variation}_{slug(title)}.md"
    target.write_text(section + "\n", encoding="utf-8", newline="\n")
    index_lines.extend([
        "",
        f"## P{nn} V{variation} — {title}",
        "",
        f"- Evaluation score: {score:.2f}",
        f"- Standalone package: `selected/{target.name}`",
        f"- Canonical source: `{source.name}`",
    ])
    qa_parts.extend([
        "",
        f"# SELECTED P{nn} V{variation} — {title}",
        "",
        section,
    ])

(RUN_DIR / "08_SELECTED_SIX.md").write_text(
    "\n".join(index_lines).rstrip() + "\n", encoding="utf-8", newline="\n"
)
(RUN_DIR / "QA_SELECTED_INPUT.md").write_text(
    "\n".join(qa_parts).rstrip() + "\n", encoding="utf-8", newline="\n"
)
print("selected packages: 6")
print("wrote 08_SELECTED_SIX.md, QA_SELECTED_INPUT.md, and selected/*.md")
