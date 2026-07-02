#!/usr/bin/env python3
"""Validate Lofn Suno-ready markdown packages.

Usage:
  python skills/music/scripts/validate_suno_packages.py <file-or-dir> [...]

Checks final delivery structure, not creative quality:
- ## 1. MUSIC PROMPT exists and is 850-1000 chars by default
- ## 1B. SUNO EXCLUDE PROMPT exists and is <=1000 chars
- ## 2. LYRICS exists
- lyrics field is <5000 chars (Suno render hard limit; target <=4800)
- ## 3. TITLE exists
- [SONG FORM:] exists in lyrics
- full EMO performance-script section headers exist and no bare architectural EMO labels are used
- at least one short *sfx cue* exists
- no prompt/procedure/QA debris in sung lines
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
PROMPT_WITH_EXCLUDE_RE = re.compile(r"^## 1\. MUSIC PROMPT\s*\n(?P<body>.*?)(?=^## 1B\. SUNO EXCLUDE PROMPT\s*$)", re.M | re.S)
EXCLUDE_RE = re.compile(r"^## 1B\. SUNO EXCLUDE PROMPT\s*\n(?P<body>.*?)(?=^## 2\. LYRICS\s*$)", re.M | re.S)
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

    pm = PROMPT_WITH_EXCLUDE_RE.search(text) or PROMPT_RE.search(text)
    em = EXCLUDE_RE.search(text)
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
        if any(term in low for term in ("avoid:", "avoid ", "do not ", "no male", "no child", "blacklist")):
            errs.append("style prompt appears to contain avoid/exclude language; use ## 1B. SUNO EXCLUDE PROMPT")

    if not em:
        errs.append("missing ## 1B. SUNO EXCLUDE PROMPT before ## 2. LYRICS")
    else:
        exclude = strip_md(em.group("body"))
        n = len(exclude)
        if n == 0:
            errs.append("empty Suno exclude prompt")
        if n > 1000:
            errs.append(f"Suno exclude prompt length {n}, expected <=1000")
        if re.search(r"\b(avoid|do not|don't|must not|please|the song should not)\b", exclude, re.I):
            errs.append("exclude prompt contains prose negation; use concrete comma-separated terms")

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
        if re.search(r"\b(prompt|QA gate|taxonomy|production manual|this song is about)\b", lyrics, re.I):
            errs.append("possible prompt/procedure/QA debris in lyrics")
        sung_lines = [ln for ln in lyrics.splitlines() if ln.strip() and not ln.strip().startswith("[") and not ln.strip().startswith("*") and not ln.strip().startswith("#")]
        if len(sung_lines) < 60:
            errs.append(f"only {len(sung_lines)} probable sung lines, expected >=60")
        # Suno lyrics-field HARD CAP: everything pasted into Suno's lyrics box
        # (Theme + SONG FORM + Disc_Channel + all headers + SFX + sung lines)
        # must be under 5000 chars or Suno will not render. Target <=4800 for margin.
        fence = re.search(r"```[a-zA-Z]*\n(.*?)```", lyrics, re.S)
        field = (fence.group(1) if fence else lyrics).strip()
        field_len = len(field)
        if field_len >= 5000:
            errs.append(f"lyrics field {field_len} chars exceeds Suno hard limit (must be <5000, target <=4800); trim lines/headers or move Disc_Channel to a production sidecar - line-count target yields to this cap")

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
