#!/usr/bin/env python3
"""Prove RULING C3: credited players appear ONLY in DEDICATION blocks.

THE RULE (`vault/DEDICATION_STANDARD.md`, CLAUDE.md, and this run's RULING C3):
  - A name in a GENERATOR-FACING field is an instruction to imitate.
  - A name in a DEDICATION is a debt paid.
  - Player credit is REQUIRED, not optional. "If a prompt names a break,
    credit the drummer."

So the ban is not on naming people. It is on naming them where a generator
reads. Generator-facing = `## 1. MUSIC PROMPT`, `## 1B. SUNO EXCLUDE PROMPT`,
`## 2. LYRICS`, `## 3. TITLE`. Non-generator = `## DEDICATION`, `## MAJOR
DEVIATIONS`, and any prose outside a variation.

This script extracts every capitalised multi-word name from the DEDICATION
blocks, then asserts none of them appears in any generator-facing field.

Run from repo root:  python3 output/daily/2026-08-22/credit_sweep.py
Exit 1 if any credited name leaks into a generator-facing field.
"""
import os
import re
import sys

RUN = os.path.join("output", "daily", "2026-08-22")
GENERATOR_FACING = ["1. MUSIC PROMPT", "1B. SUNO EXCLUDE PROMPT", "2. LYRICS", "3. TITLE"]

# Two capitalised words in a row, allowing internal particles/initials.
NAME_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+(?:[A-Z]\.|[A-Z][a-z]+|de|van|der|the|of|and)){1,3})\b"
)
# Words that look like names but are not people.
STOPWORDS = {
    "The Scientist", "The Winstons", "Suno Style", "Song Form", "Major Deviations",
    "Music Prompt", "Suno Exclude", "Theme", "Song", "Verse", "Chorus", "Refrain",
}


def sections(body):
    """{heading: text} for one variation body."""
    out, parts = {}, re.split(r"^##\s+(.+?)\s*$", body, flags=re.M)
    for i in range(1, len(parts) - 1, 2):
        out[parts[i].strip()] = parts[i + 1]
    return out


def main():
    leaks, credited, per_pair = [], set(), {}

    for p in ["01", "02", "03", "04", "05", "06"]:
        path = os.path.join(RUN, f"pair_{p}_step10_final_package_enhanced.md")
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8").read()
        parts = re.split(r"^#\s+(V\d)\s*[—\-–]\s*(.+?)\s*$", text, flags=re.M)
        chunks = [(parts[i], parts[i + 2]) for i in range(1, len(parts) - 2, 3)]
        pair_names = set()

        for tag, body in chunks:
            secs = sections(body)
            ded = secs.get("DEDICATION", "")
            names = {
                n.strip() for n in NAME_RE.findall(ded)
                if n.strip() not in STOPWORDS and len(n.strip()) > 4
            }
            pair_names |= names
            credited |= names

            for heading in GENERATOR_FACING:
                field = secs.get(heading, "")
                if not field:
                    continue
                for n in names:
                    # Match the FULL name, and also the surname alone.
                    surname = n.split()[-1]
                    for probe in (n, surname):
                        if len(probe) < 5:
                            continue
                        if re.search(rf"\b{re.escape(probe)}\b", field):
                            leaks.append(f"P{p}{tag}: {probe!r} appears in '{heading}'")

        per_pair[p] = sorted(pair_names)

    print("CREDITED NAMES BY PAIR (from ## DEDICATION blocks only)")
    print("-" * 60)
    for p, names in per_pair.items():
        print(f"  P{p}: {', '.join(names) if names else '(none)'}")

    print(f"\ndistinct credited names: {len(credited)}")
    print(f"generator-facing fields swept: {len(GENERATOR_FACING)} x 24 variations")
    print(f"\nLEAKS: {len(leaks)}")
    for l in leaks:
        print(f"  LEAK  {l}")
    if not leaks:
        print("  none - every credited name is confined to its DEDICATION block.")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
