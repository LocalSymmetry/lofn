#!/usr/bin/env python3
"""Validate Lofn Suno-ready markdown packages.

Usage:
  python skills/music/scripts/validate_suno_packages.py <file-or-dir> [...]

Checks final delivery structure, not creative quality:
- ## 1. MUSIC PROMPT exists and is 850-1000 chars by default
- ## 2. LYRICS exists
- ## 3. TITLE exists
- [SONG FORM:] exists in lyrics
- EMO: section headers exist and no bare architectural EMO labels are used
- at least one short *sfx cue* exists
- no obvious real-artist prompt contamination from a small common list
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ARTIST_BLOCKLIST = {
    "taylor swift", "billie eilish", "lana del rey", "beyonce", "beyoncé",
    "radiohead", "nine inch nails", "trent reznor", "bjork", "björk", "arca",
    "charli xcx", "fka twigs", "bon iver", "mitski", "sufjan stevens",
}

PROMPT_RE = re.compile(r"^## 1\. MUSIC PROMPT\s*\n(?P<body>.*?)(?=^## 2\. LYRICS\s*$)", re.M | re.S)
LYRICS_RE = re.compile(r"^## 2\. LYRICS\s*\n(?P<body>.*?)(?=^## 3\. TITLE\s*$)", re.M | re.S)
TITLE_RE = re.compile(r"^## 3\. TITLE\s*\n(?P<body>.*?)(?=^## |\Z)", re.M | re.S)


def iter_files(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            out.extend(sorted(p.glob("*.md")))
        elif p.is_file():
            out.append(p)
    return out


def strip_md(s: str) -> str:
    return re.sub(r"```.*?```", "", s, flags=re.S).strip()


def validate(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errs: list[str] = []

    pm = PROMPT_RE.search(text)
    lm = LYRICS_RE.search(text)
    tm = TITLE_RE.search(text)
    if not pm:
        errs.append("missing ## 1. MUSIC PROMPT before ## 2. LYRICS")
    else:
        prompt = strip_md(pm.group("body"))
        n = len(prompt)
        if n < 850 or n > 1000:
            errs.append(f"music prompt length {n}, expected 850-1000")
        low = prompt.lower()
        hits = sorted(a for a in ARTIST_BLOCKLIST if a in low)
        if hits:
            errs.append("artist names in prompt: " + ", ".join(hits))
        if "avoid" not in low and "blacklist" not in low and "no " not in low:
            errs.append("music prompt lacks explicit avoid/blacklist language")

    if not lm:
        errs.append("missing ## 2. LYRICS before ## 3. TITLE")
    else:
        lyrics = lm.group("body")
        if "[SONG FORM:" not in lyrics:
            errs.append("missing [SONG FORM:] in lyrics")
        if "EMO:" not in lyrics:
            errs.append("missing EMO: headers in lyrics")
        if re.search(r"EMO:\s*(AWE|INDIGNATION|SYNTHESIS)\s*(?:[\]–\-]|$)", lyrics, re.I):
            errs.append("bare architectural EMO label used")
        if not re.search(r"^\*[^*\n]{1,40}\*\s*$", lyrics, re.M):
            errs.append("missing standalone short *SFX cue*")
        sung_lines = [ln for ln in lyrics.splitlines() if ln.strip() and not ln.strip().startswith("[") and not ln.strip().startswith("*") and not ln.strip().startswith("#")]
        if len(sung_lines) < 60:
            errs.append(f"only {len(sung_lines)} probable sung lines, expected >=60")

    if not tm or not strip_md(tm.group("body")):
        errs.append("missing ## 3. TITLE content")

    return errs


def main() -> int:
    files = iter_files(sys.argv[1:])
    if not files:
        print("No markdown files found", file=sys.stderr)
        return 2
    failed = 0
    for f in files:
        errs = validate(f)
        if errs:
            failed += 1
            print(f"FAIL {f}")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"PASS {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
