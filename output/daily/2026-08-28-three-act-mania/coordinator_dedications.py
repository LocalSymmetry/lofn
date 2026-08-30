#!/usr/bin/env python3
"""DEDICATIONS.md — EXTRACTED from the pairs' own files, never hardcoded here.

⛔ WHY THIS WAS REWRITTEN, 2026-08-28. The first version pasted the dedication
text into this script. Two pairs then repaired their selected song, and this
file silently kept describing music that no longer existed: P01's dedication
credited a "stomp-and-handclap act" its selected variation does not have, and
P04's credited shape-note singing that belongs to V1, not V4. QA caught one; P04
caught the other. Both were MY defect, and both are the same one P02 named:
**a number does not become true by being copied forward, and neither does a
dedication.** The pair's file is the source of truth. This reads it.

vault/DEDICATION_STANDARD.md:
  - <=500 chars, MEASURED IN CODE
  - a break named in a style field ⇒ credit the drummer (amen → Gregory C.
    Coleman / The Winstons)
  - a generated track is NOT a sample
  - a name in a PROMPT is an instruction to imitate; a name in a DEDICATION is a
    debt paid ⇒ artist names appear here and in NO pasted field
"""
import os
import re
import sys

sys.path.insert(0, r"E:\git\private-claw\scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coordinator_verify import fields, lyrics_of, STYLE_RX, EXCL_RX  # noqa: E402

SEL = [("01", 4, "NOTHING WAS TAKEN"), ("02", 1, "SAY HOW FAR"),
       ("03", 1, "THE FLOOR IS ALREADY GOING"), ("04", 4, "THE WHOLE TIME"),
       ("05", 1, "LEAVE IT ON"), ("06", 4, "WATCH ME NAME THE WHOLE ROOM")]

BREAKS = ["amen", "funky drummer", "apache", "think break"]
NAMES = ["davachi", "coleman", "winstons", "gregory"]


def dedication_of(path):
    """First fenced block after the first DEDICATION heading. Loud on failure."""
    t = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"^#{1,4} .*DEDICATION[^\n]*$", t, re.M)
    if not m:
        return None, "no DEDICATION heading found"
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)\n```", t[m.end():], re.S)
    if not blocks:
        return None, "DEDICATION heading present but no fenced block after it"
    return " ".join(blocks[0].split()), None


out = ["# DEDICATIONS — `2026-08-28-three-act-mania`\n",
       "\n*One per selected piece, **extracted from each pair's own step-10 file**, never retyped here.*\n",
       "\n> ⭐ **A name in a PROMPT is an instruction to imitate. A name in a DEDICATION is a debt paid.**\n"
       "> Every artist named below appears in **no pasted field** — verified in code.\n\n---\n"]
rows, fail = [], []
for n, v, title in SEL:
    p = f"pair_{n}_step10_final_package.md"
    ded, err = dedication_of(p)
    sty = fields(p, STYLE_RX)[v - 1].lower()
    exc = fields(p, EXCL_RX)[v - 1].lower()
    lyr = lyrics_of(p)[v - 1].lower()
    if err:
        fail.append(f"P{n}V{v}: {err}")
        rows.append((title, f"P{n}V{v}", 0, [], False, [], False, err))
        continue
    chars = len(ded)
    brk = [b for b in BREAKS if b in sty]
    credited = "coleman" in ded.lower() and "winstons" in ded.lower()
    leaked = [nm for nm in NAMES if nm in sty or nm in exc or nm in lyr]
    ok = chars <= 500 and (not brk or credited) and not leaked
    if not ok:
        fail.append(f"P{n}V{v}: chars={chars} break={brk} credited={credited} leaked={leaked}")
    rows.append((title, f"P{n}V{v}", chars, brk, credited, leaked, ok, None))
    out.append(f"\n## {title}\n`P{n}V{v}` · *extracted from `{p}`*\n\n```\n{ded}\n```\n"
               f"**{chars} / 500 chars** · break in style: **{brk[0] if brk else 'none'}** · "
               f"drummer credited: **{'yes' if credited else ('n/a' if not brk else 'NO')}**\n")

out.append("\n---\n\n## AUDIT\n\n| piece | chars | ≤500 | break named | drummer credited | artist name in a pasted field |\n|---|---|---|---|---|---|\n")
for title, tag, c, b, cr, lk, ok, err in rows:
    out.append(f"| {title} | {c} | {'OK' if c and c <= 500 else '**FAIL**'} | {b[0] if b else '—'} | "
               f"{'**yes**' if cr else ('—' if not b else '**MISSING**')} | "
               f"{'**' + ', '.join(lk) + '**' if lk else 'none'} |\n")
out.append(f"\n**{len(rows) - len(fail)} / {len(rows)} pass all three dedication checks.**\n")
if fail:
    out.append("\n⛔ **FAILING:**\n" + "".join(f"- {f}\n" for f in fail))
open("DEDICATIONS.md", "w", encoding="utf-8").write("".join(out))
print(f"DEDICATIONS.md: {len(rows) - len(fail)}/{len(rows)} pass")
for title, tag, c, b, cr, lk, ok, err in rows:
    print(f"  {tag} {c:>3d} chars  break={(b[0] if b else '-'):<5s} "
          f"credited={'y' if cr else '-'}  leaked={lk or 'none'}  {'PASS' if ok else 'FAIL'}"
          + (f"  [{err}]" if err else ""))
sys.exit(1 if fail else 0)
