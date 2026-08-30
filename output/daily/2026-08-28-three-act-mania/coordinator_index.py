#!/usr/bin/env python3
"""Emit the INDEX's measured table FROM THE SHIPPED FILE. No hand-typed numbers.

The run's own finding, applied to the run's own report: never print a count you
did not derive from the thing you measured.
"""
import os
import re
import sys

sys.path.insert(0, r"E:\git\private-claw\scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_soundcraft as ms  # noqa: E402
from coordinator_verify import fields, lyrics_of, STYLE_RX, EXCL_RX, HARD, MARQUE, scan  # noqa: E402

TITLES = ["Nothing Was Taken", "Say How Far", "The Floor Is Already Going",
          "The Whole Time", "Leave It On", "Watch Me Name The Whole Room"]
SRC = [("01", 4), ("02", 1), ("03", 1), ("04", 4), ("05", 1), ("06", 4)]
ACTS = (r"\bACT\s*(?:ONE|I|1)\b(?!\w)", r"\bACT\s*(?:TWO|II|2)\b(?!\w)",
        r"\bACT\s*(?:THREE|III|3)\b(?!\w)")

sty = fields("SUNO_PACKAGES.md", STYLE_RX)
exc = fields("SUNO_PACKAGES.md", EXCL_RX)
lyr = lyrics_of("SUNO_PACKAGES.md")
assert len(sty) == len(exc) == len(lyr) == len(TITLES), \
    f"field-count mismatch: {len(sty)}/{len(exc)}/{len(lyr)} vs {len(TITLES)} titles"

rows, allpass = [], True
print("| # | title | style | excl | lyrics | sung lines | rhyme | line-ret | w/line | acts | SFX | hw | marque |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, (t, s, e, b) in enumerate(zip(TITLES, sty, exc, lyr), 1):
    sung = ms.sung(b)
    pr = ms.profile(sung)
    hdr = "\n".join(l for l in b.splitlines() if l.strip().startswith("["))
    acts = sum(1 for a in ACTS if re.search(a, hdr, re.I))
    sfx = len(re.findall(r"^\s*\*?SFX", b, re.M | re.I))
    hw = scan("\n".join(sung), HARD)
    mq = scan("\n".join(sung) + "\n" + s, MARQUE, polarity_aware=True)
    ok = (850 <= len(s) <= 1000 and len(b) <= 5000 and 70 <= pr["lines"] <= 120
          and pr["end_rhyme"] >= 0.30 and pr["line_return"] >= 0.20
          and pr["words_per_line"] <= 7.5 and acts == 3 and sfx >= 1
          and not hw and not mq)
    allpass &= ok
    print(f"| {i} | **{t}** | {len(s)} | {len(e)} | {len(b)} | {pr['lines']} | "
          f"{pr['end_rhyme']:.3f} | {pr['line_return']:.3f} | {pr['words_per_line']:.2f} | "
          f"{acts}/3 | {sfx} | {len(hw)} | {len(mq)} |")
print(f"\nDERIVED: {len(TITLES)} songs measured from SUNO_PACKAGES.md · "
      f"{'ALL PASS' if allpass else 'SEE ROWS'}")
print(f"style range {min(len(x) for x in sty)}-{max(len(x) for x in sty)} · "
      f"lyrics {min(len(x) for x in lyr)}-{max(len(x) for x in lyr)} · "
      f"sung lines {min(ms.profile(ms.sung(x))['lines'] for x in lyr)}-"
      f"{max(ms.profile(ms.sung(x))['lines'] for x in lyr)}")
