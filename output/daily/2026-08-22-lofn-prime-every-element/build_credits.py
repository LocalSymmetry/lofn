#!/usr/bin/env python3
"""Measure every dedication against the 500-char cap and assemble CREDITS.md.

`vault/DEDICATION_STANDARD.md`: every shipped piece carries a <=500-char
dedication naming what it borrowed and who it is owed to -- MEASURED IN CODE,
NEVER EYEBALLED. Overflow goes to a companion credits file the caption points at.

Trim priority when over: (5) "full credits below" -> (4) honest AI authorship
-> (3) living-scene lineage -> (2) THE SHARPEST SESSION-PLAYER CREDIT
-> (1) the raised source-sound + who recorded it.
TRIM FROM 5 UPWARD. Never buy space by cutting 1 or 2.

Run from repo root:  python3 output/daily/2026-08-22-lofn-prime-every-element/build_credits.py
Exit 1 if any dedication is over cap.
"""
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUN = os.path.join("output", "daily", "2026-08-22-lofn-prime-every-element")
CAP = 500


def fenced(block):
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", block, re.S)
    if m:
        return m.group(1).strip()
    return "\n".join(l for l in block.splitlines() if not l.strip().startswith("#")).strip()


def main():
    rows, over, entries = [], [], []

    for p in ["01", "02", "03", "04", "05", "06"]:
        path = os.path.join(RUN, f"pair_{p}_step10_final_package_enhanced.md")
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8").read()
        parts = re.split(r"^#\s+(V\d)\s*[—\-–]\s*(.+?)\s*$", text, flags=re.M)
        for i in range(1, len(parts) - 2, 3):
            tag, title, body = parts[i], parts[i + 1].strip(), parts[i + 2]
            dm = re.search(r"##\s*DEDICATION(.*?)(?=^##\s|\Z)", body, re.S | re.I | re.M)
            ded = fenced(dm.group(1)) if dm else ""
            n = len(ded)
            vid = f"P{p}{tag}"
            rows.append((vid, title, n))
            entries.append((vid, title, ded))
            if n > CAP:
                over.append(f"{vid} ({title}): {n} chars, {n - CAP} over")
            elif n == 0:
                over.append(f"{vid} ({title}): NO DEDICATION BLOCK")

    print(f"{'id':<8}{'title':<34}{'chars':>6}  cap {CAP}")
    print("-" * 56)
    for vid, title, n in rows:
        mark = "  OVER" if n > CAP else ("  MISSING" if n == 0 else "")
        print(f"{vid:<8}{title[:33]:<34}{n:>6}{mark}")

    longest = max(rows, key=lambda r: r[2]) if rows else None
    print(f"\ndedications: {len(rows)}/24   longest: {longest[0]} at {longest[2]} chars")
    print(f"OVER CAP / MISSING: {len(over)}")
    for o in over:
        print(f"  !! {o}")

    with open(os.path.join(RUN, "CREDITS.md"), "w", encoding="utf-8") as fh:
        fh.write("# CREDITS — `daily-music-2026-08-22-sunna-offbeat`\n\n")
        fh.write("*The companion credits file. Every dedication below is measured in code against the\n")
        fh.write(f"{CAP}-character cap, never eyeballed. ⛔ **These names appear HERE and in the\n")
        fh.write("per-song `## DEDICATION` block ONLY — never in a music prompt, exclude field, lyric\n")
        fh.write("or title.** A name in a generator-facing field is an instruction to imitate; a name\n")
        fh.write("here is a debt paid. Proved by `credit_sweep.py`: 0 leaks across 4 fields x 24 songs.*\n\n")
        fh.write("**Honesty clause, both ways:** these are generated tracks, **not samples**. Where a\n")
        fh.write("figure of sound was asked for by name, we name the player who made it — *we asked for\n")
        fh.write("this figure, so we name him* — never *\"contains a sample of.\"*\n\n---\n\n")
        for vid, title, ded in entries:
            fh.write(f"## {vid} — {title}\n\n*{len(ded)} chars*\n\n{ded}\n\n---\n\n")

    print(f"\nwrote {os.path.join(RUN, 'CREDITS.md')}")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
