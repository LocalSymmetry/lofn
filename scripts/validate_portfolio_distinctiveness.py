#!/usr/bin/env python3
"""Validate cross-pair distinctiveness for Lofn final Step 10 files.

Catches the failure mode where each pair is structurally valid in isolation but
shares the same prompt/lyric skeleton across the portfolio.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path


def norm(s: str) -> str:
    s = re.sub(r"\[[^\]]+\]", "", s)  # remove headers
    s = re.sub(r"\*[^*]+\*", "", s)  # remove sfx
    s = re.sub(r"\b(pair|variant)\s+\w+\b", "", s, flags=re.I)
    s = re.sub(r"[^a-z0-9' ]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_variant(text: str) -> str:
    starts = [m.start() for m in re.finditer(r"^#{2,3}\s+VARIANT\b", text, re.I | re.M)]
    if not starts:
        return text
    start = starts[0]
    later = [x for x in starts if x > start]
    end = later[0] if later else len(text)
    return text[start:end]


def lyrics_only(text: str) -> str:
    m = re.search(r"##\s*2\.\s*LYRICS\s*(.*?)(?=\nGATE CHECK|\n##\s*3\.|\n#{2,3}\s+VARIANT|\Z)", text, re.I | re.S)
    return m.group(1) if m else text


def prompt_only(text: str) -> str:
    m = re.search(r"##\s*1\.\s*MUSIC PROMPT\s*(.*?)(?=\n##\s*2\.\s*LYRICS)", text, re.I | re.S)
    return m.group(1) if m else ""


def ngrams(s: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = s.split()
    return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("--max-lyric-sim", type=float, default=0.42)
    ap.add_argument("--max-prompt-sim", type=float, default=0.58)
    ap.add_argument("--max-ngram-jaccard", type=float, default=0.18)
    args = ap.parse_args()

    root = Path(args.audio_dir)
    files = sorted(p for p in root.glob("pair_*_step10_revision_synthesis.md") if ".repair_attempt_" not in p.name)
    if len(files) < 2:
        print("FAIL: need at least two pair Step 10 files")
        return 1

    entries = []
    for p in files:
        v = first_variant(p.read_text(errors="replace"))
        entries.append((p.name, norm(prompt_only(v)), norm(lyrics_only(v))))

    failures = []
    for i in range(len(entries)):
        for j in range(i+1, len(entries)):
            a, aprompt, alyr = entries[i]
            b, bprompt, blyr = entries[j]
            psim = difflib.SequenceMatcher(None, aprompt, bprompt).ratio()
            lsim = difflib.SequenceMatcher(None, alyr, blyr).ratio()
            ag, bg = ngrams(alyr), ngrams(blyr)
            jac = len(ag & bg) / max(1, len(ag | bg))
            if psim > args.max_prompt_sim:
                failures.append(f"prompt similarity too high {a} vs {b}: {psim:.3f}")
            if lsim > args.max_lyric_sim:
                failures.append(f"lyric similarity too high {a} vs {b}: {lsim:.3f}")
            if jac > args.max_ngram_jaccard:
                failures.append(f"5-gram overlap too high {a} vs {b}: {jac:.3f}")

    if failures:
        print("PORTFOLIO DISTINCTIVENESS FAIL")
        for f in failures:
            print("- " + f)
        return 1
    print("PORTFOLIO DISTINCTIVENESS PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
