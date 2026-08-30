#!/usr/bin/env python3
"""Build CREDITS.md — the companion file that makes "full credits below" TRUE.

WHY: on 2026-08-15, `"full credits below"` shipped in SIX of twenty-four dedications
**pointing at nothing** — no companion file existed. The promise is first in the trim
order precisely because a dangling promise is worse than no promise. Pair 01's tier hit
the same choice today and OMITTED the line rather than write one it could not honour;
Pair 03 kept it, and this file is why that was allowed.

⭐ It imports `dedication_lib` rather than re-implementing the measurement — the rule
this run earned the hard way, four times, in this exact field.

Usage: python3 output/daily/2026-08-21/build_credits.py
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import dedication_lib as D  # noqa: E402

RUN = pathlib.Path(__file__).parent
VAR_RE = re.compile(r"^#{1,4}[ \t]*(?:[^\w\s#]+[ \t]*)?"
                    r"(?:(?:PACKAGE|VARIATION|SONG|TRACK)[ \t]+V?(\d)|V(\d))\b[^\n]*$",
                    re.M | re.I)
TITLE_RE = re.compile(r"^#{1,4}\s*3\.\s*TITLE\s*$(.*?)(?=^#{1,4}\s)", re.S | re.M)

rows, problems = [], []
for f in sorted(RUN.glob("pair_*_step10_final_package_enhanced.md")):
    pair = f.name.split("_")[1]
    text = f.read_text(encoding="utf-8", errors="replace")
    marks = list(VAR_RE.finditer(text))
    if len(marks) != 4:
        problems.append(f"{f.name}: {len(marks)} variation headings, expected 4 "
                        f"-- EXTRACTION ERROR, refusing to report this pair")
        continue
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        chunk = text[m.start():end]
        vid = m.group(1) or m.group(2)
        tm = TITLE_RE.search(chunk)
        title = ""
        if tm:
            body = tm.group(1)
            fenced = re.search(r"```[a-z]*\n(.*?)```", body, re.S)
            title = (fenced.group(1) if fenced else body).strip().splitlines()[0].strip() if body.strip() else ""
        title = title.strip("*_` ").strip()
        ded, n, verdict = D.check(chunk)
        if verdict in ("OVER_CAP", "EMPTY_AGAINST_HEADING"):
            problems.append(f"P{pair} V{vid}: dedication {verdict} ({n} chars)")
        rows.append((pair, vid, title, ded, n, verdict))

if not rows:
    sys.exit("FATAL: no packages extracted -- refusing to write an empty CREDITS.md.")

out = [
    "# CREDITS — `daily-music-2026-08-21`",
    "## The companion file every dedication saying “full credits below” points at",
    "",
    "> On 2026-08-15 that phrase shipped in **six of twenty-four dedications pointing at",
    "> nothing** — no companion file existed. It is first in the dedication trim order",
    "> because a dangling promise is worse than no promise. **This file is the promise kept.**",
    "",
    "> ⛔ **Nothing here is a sample.** Every track is generated. Where a scene or a figure",
    "> of sound was asked for **by name in a style field**, it is named here — *we asked for it",
    "> by name, so we name them*. Where nothing was asked for, nothing is claimed, and the",
    "> dedication says so rather than staying silent: an over-credit is visible and",
    "> embarrassing, but **an under-credit reads as a completed check.**",
    "",
    "> ⭐ **No player, group, label or recording named on this page appears in any",
    "> generator-facing field.** *A name in the prompt is an instruction to imitate; a name",
    "> in the dedication is a debt paid.*",
    "",
    "---",
    "",
    "## THE SHIPPED DEDICATIONS",
    "",
    "| pair | var | title | chars | dedication |",
    "|---|---|---|---:|---|",
]
for pair, vid, title, ded, n, verdict in rows:
    flat = " ".join(ded.split())
    mark = "" if verdict == "ok" else f" **[{verdict}]**"
    out.append(f"| P{pair} | V{vid} | {title or '_(untitled)_'} | {n}{mark} | {flat} |")

out += [
    "",
    "---",
    "",
    "## THE PANEL",
    "",
    "> *Panel voices are model-generated interpretive constructs, each “after” a named source",
    "> figure’s published work. No statement is a quotation of, or endorsement by, the named",
    "> person.* Full debate: `03_panel_debate.md`.",
    "",
    "**Concept** — after Hito Steyerl · Rosa Menkman · Tacita Dean · William Basinski · "
    "W. G. Sebald · Wendy Hui Kyong Chun *(Hyper-Skeptic)*.",
    "",
    "**Medium** — after Boards of Canada · Leyland Kirby · Burial · Pye Corner Audio · "
    "Broadcast · Philip Glass *(Hyper-Skeptic)*.",
    "",
    "**Context & Marketing** — after danah boyd · Robin Dunbar · David Hesmondhalgh · "
    "Olivia Rodrigo · Tony Hawk · Philip N. Cohen *(Hyper-Skeptic)*.",
    "",
    "**Transform seats** — after Marie Kondo · Brewster Kahle · Herbert A. Simon · "
    "Viktor Mayer-Schönberger · Sianne Ngai, and six role-anchored practitioners "
    "(histopathology · dendrochronology · genebanking · reed-making · manuscript recovery · "
    "destructive testing).",
    "",
    "## THE DAY’S SOURCES",
    "",
    "**Source 2, quoted verbatim and credited:** Bandcamp Daily, *“Gasba Forever: The Next",
    "Phase of Algeria’s Millenia-Old Folk Tradition”* (2026-08-21) and *“The Ongoing",
    "Transformation of Cabo Verdean Music”* (2026-08-20). **Source 3:** NASA APOD,",
    "*“Time-Lapse of the Star S301 Orbiting the Black Hole in the Center of the Galaxy”*",
    "(2026-08-21). Full ledger of all twenty-five sources: `00_research_brief.md`.",
    "",
    "## AUTHORSHIP",
    "",
    "Written by **Lofn** (Claude) with **Dr. Local Symmetry**. ⛔ Nothing in this run was",
    "rendered, published, or spent. 💜",
    "",
]
(RUN / "CREDITS.md").write_text("\n".join(out) + "\n", encoding="utf-8")

print(f"CREDITS.md written: {len(rows)} dedications")
lens = [n for *_, n, v in rows if v == "ok"]
print(f"  measured via dedication_lib (the ONE implementation): "
      f"min {min(lens)} / max {max(lens)} / cap {D.CAP}")
if problems:
    print(f"  !! {len(problems)} problem(s):")
    for p in problems:
        print("     -", p)
    sys.exit(1)
print("  all dedications within cap; no empty extraction against a present heading.")
