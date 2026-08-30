#!/usr/bin/env python3
"""Measure EVERY variation in EVERY step-10 package. Counts only, no judgment.

WHY THIS EXISTS: `scripts/validate_step.py` anchors its music-prompt band check on
the FIRST `## 1. MUSIC PROMPT` in a file. A step-10 package holds FOUR variations,
so 18 of this run's 24 variations were never measured by it. A gate that reads
one quarter of the artifact and reports PASS is not measuring the artifact.

Run from repo root:  python3 output/daily/2026-08-22/measure_variations.py
Exit 1 if any HARD gate fails.
"""
import os
import re
import sys

RUN = os.path.join("output", "daily", "2026-08-22")

PROMPT_LO, PROMPT_HI = 850, 1000
PROMPT_TGT_LO, PROMPT_TGT_HI = 870, 960
PROMPT_HUG = 985
LYRICS_MAX, LYRICS_TGT = 5000, 4800
SUNG_LO, SUNG_HI = 70, 120
SUNG_TGT_LO, SUNG_TGT_HI = 78, 110
SUNG_FLOOR_HUG = 72

HOUSE_LEXICON = [
    "more sub and more sky", "small astonished laugh", "triple arch",
    "triple-arch panorama", "small enough to understand",
]
# Sung numerals must be WORDS. Digits are an instruction to a renderer to guess.
DIGIT_RE = re.compile(r"(?<![A-Za-z])\d+(?![A-Za-z])")


def fenced(block):
    """Content of the first ``` fence in a chunk, fences removed.

    FALLS BACK to the bare body when there is no fence. The first version of
    this script returned "" instead, and reported "0 sung lines" for 16 of 24
    variations -- a false positive on my own instrument, not a defect in the
    artifacts. Pairs 01 and 04 fence their lyrics; 02, 03, 05 and 06 do not.
    That divergence is real and is reported separately as UNFENCED; it must
    not be reported as EMPTY.
    """
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", block, re.S)
    if m:
        return m.group(1).strip()
    return "\n".join(
        ln for ln in block.splitlines() if not ln.strip().startswith("#")
    ).strip()


def is_fenced(block):
    return bool(re.search(r"```[a-zA-Z]*\n(.*?)```", block, re.S))


def sung_lines(lyrics):
    """Lines actually SUNG: not blank, not a [bracket] header, not an *SFX* cue."""
    out = []
    for ln in lyrics.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            continue
        if s.startswith("*") and s.endswith("*"):
            continue
        out.append(s)
    return out


def main():
    rows, hard_fails, flags = [], [], []

    for p in ["01", "02", "03", "04", "05", "06"]:
        path = os.path.join(RUN, f"pair_{p}_step10_final_package_enhanced.md")
        if not os.path.isfile(path):
            hard_fails.append(f"P{p}: step-10 package MISSING")
            continue
        text = open(path, encoding="utf-8").read()

        # Split on variation headings: "# V1 — Title"
        parts = re.split(r"^#\s+(V\d)\s*[—\-–]\s*(.+?)\s*$", text, flags=re.M)
        # parts = [preamble, tag, title, body, tag, title, body, ...]
        chunks = [(parts[i], parts[i + 1], parts[i + 2])
                  for i in range(1, len(parts) - 2, 3)]

        if len(chunks) != 4:
            hard_fails.append(f"P{p}: found {len(chunks)} variations, expected 4")

        for tag, title, body in chunks:
            pm = re.search(r"##\s*1\.\s*MUSIC PROMPT(.*?)(?=##\s)", body, re.S | re.I)
            lm = re.search(r"##\s*2\.\s*LYRICS(.*?)(?=##\s)", body, re.S | re.I)
            prompt = fenced(pm.group(1)) if pm else ""
            lyrics = fenced(lm.group(1)) if lm else ""
            sung = sung_lines(lyrics)
            n_prompt, n_lyr, n_sung = len(prompt), len(lyrics), len(sung)
            vid = f"P{p}{tag}"

            if not (PROMPT_LO <= n_prompt <= PROMPT_HI):
                hard_fails.append(f"{vid}: prompt {n_prompt} outside {PROMPT_LO}-{PROMPT_HI}")
            elif not (PROMPT_TGT_LO <= n_prompt <= PROMPT_TGT_HI):
                flags.append(f"{vid}: prompt {n_prompt} outside target {PROMPT_TGT_LO}-{PROMPT_TGT_HI}")
            if n_prompt >= PROMPT_HUG:
                flags.append(f"{vid}: prompt {n_prompt} BOUNDARY-HUGGING (>={PROMPT_HUG})")
            if prompt and prompt[-1] not in ".!?":
                hard_fails.append(f"{vid}: prompt lacks terminal punctuation (ends {prompt[-12:]!r})")

            if n_lyr > LYRICS_MAX:
                hard_fails.append(f"{vid}: lyrics field {n_lyr} over hard cap {LYRICS_MAX}")
            elif n_lyr > LYRICS_TGT:
                flags.append(f"{vid}: lyrics field {n_lyr} over target {LYRICS_TGT}")

            if not (SUNG_LO <= n_sung <= SUNG_HI):
                hard_fails.append(f"{vid}: {n_sung} sung lines outside {SUNG_LO}-{SUNG_HI}")
            elif not (SUNG_TGT_LO <= n_sung <= SUNG_TGT_HI):
                flags.append(f"{vid}: {n_sung} sung lines outside target {SUNG_TGT_LO}-{SUNG_TGT_HI}")
            if n_sung <= SUNG_FLOOR_HUG:
                flags.append(f"{vid}: {n_sung} sung lines FLOOR-HUGGING (<={SUNG_FLOOR_HUG})")

            # A sung numeral written as digits is a coin toss at render time.
            digit_hits = [s for s in sung if DIGIT_RE.search(s)]
            if digit_hits:
                hard_fails.append(f"{vid}: DIGITS in {len(digit_hits)} sung line(s): {digit_hits[0][:48]!r}")

            low = (prompt + "\n" + lyrics).lower()
            for phrase in HOUSE_LEXICON:
                if phrase in low:
                    flags.append(f"{vid}: house-lexicon phrase {phrase!r}")

            rows.append((vid, title.strip(), n_prompt, n_lyr, n_sung))

    print(f"{'id':<8}{'title':<34}{'prompt':>7}{'lyrics':>8}{'sung':>6}")
    print("-" * 63)
    for vid, title, a, b, c in rows:
        print(f"{vid:<8}{title[:33]:<34}{a:>7}{b:>8}{c:>6}")

    print(f"\nvariations measured: {len(rows)}/24")
    print(f"HARD FAILS: {len(hard_fails)}")
    for f in hard_fails:
        print(f"  FAIL  {f}")
    print(f"FLAGS: {len(flags)}")
    for f in flags:
        print(f"  flag  {f}")
    return 1 if hard_fails else 0


if __name__ == "__main__":
    sys.exit(main())
