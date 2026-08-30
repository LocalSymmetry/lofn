#!/usr/bin/env python3
"""Restore the ``` fence around every `## 2. LYRICS` body that is missing one.

WHY: `validate_step.py`'s step-10 music check used to count a markdown code
fence as lyric content, so a correctly-fenced package hard-failed on
"[Theme:] must be first". Four of six pair agents adapted by DELETING the
fence -- i.e. the broken gate did not merely fail to catch defects, it
DEFORMED the artifacts to satisfy itself. The gate was repaired mid-run
(see INSTRUMENT_LOG.md §2); this restores the house format it had displaced.

The LYRICS field is a paste-ready Suno field. Unfenced, its boundary is
ambiguous and a markdown renderer eats the [bracket] headers and *cue* lines.

Mechanical and content-preserving: only fence lines are ADDED. The script
asserts that the non-fence content is byte-identical before and after, and
refuses to write if it is not.

Run from repo root:  python3 output/daily/2026-08-22/restore_lyric_fences.py
"""
import os
import re
import sys

RUN = os.path.join("output", "daily", "2026-08-22")
SECTION = re.compile(r"(^##\s*2\.\s*LYRICS[^\n]*\n)(.*?)(?=^##\s)", re.M | re.S)


def strip_fences(s):
    """Comparable form: non-blank, non-fence lines only.

    The first version kept blank lines, so re-wrapping a body whose trailing
    blank lines differed tripped the safety assert on P05 even though not one
    character of text had changed. The invariant that actually matters is
    TEXTUAL CONTENT, not vertical whitespace -- so blank lines are discounted
    alongside fences. The assert stays; it is just measuring the right thing.
    """
    return "\n".join(
        ln.rstrip() for ln in s.splitlines()
        if ln.strip() and ln.strip() != "```"
    ).strip()


def main():
    total_added, touched = 0, []

    for p in ["01", "02", "03", "04", "05", "06"]:
        path = os.path.join(RUN, f"pair_{p}_step10_final_package_enhanced.md")
        if not os.path.isfile(path):
            print(f"P{p}: MISSING, skipped")
            continue
        original = open(path, encoding="utf-8").read()
        added = 0

        def repl(m):
            nonlocal added
            head, body = m.group(1), m.group(2)
            if "```" in body:
                return m.group(0)          # already fenced, leave alone
            added += 1
            return f"{head}\n```\n{body.strip()}\n```\n\n"

        updated = SECTION.sub(repl, original)

        if added == 0:
            print(f"P{p}: already fenced ({original.count('## 2. LYRICS')} sections) - untouched")
            continue

        # SAFETY: content must be identical once fences are discounted.
        if strip_fences(original) != strip_fences(updated):
            print(f"P{p}: ABORT - content would change beyond fences. NOT WRITTEN.")
            return 2

        open(path, "w", encoding="utf-8").write(updated)
        total_added += added
        touched.append(f"P{p} ({added})")
        print(f"P{p}: fenced {added} lyrics section(s)")

    print(f"\nfences restored: {total_added} across {len(touched)} pair(s): {', '.join(touched) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
