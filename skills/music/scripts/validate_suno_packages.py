#!/usr/bin/env python3
"""Validate Lofn Suno-ready markdown packages.

Usage:
  python skills/music/scripts/validate_suno_packages.py <file-or-dir> [...] [--min-sung-lines N] [--max-sung-lines N]

Checks every detected package in the newest Step-11 segment of a cumulative
artifact (or the whole file for a standalone package), not creative quality:
- a legacy (`## 1. MUSIC PROMPT`) or Step-11 canonical
  (`## SUNO STYLE PROMPT`) style block exists and is 850-1000 chars
- the matching legacy or canonical Suno exclude block exists and is <=1000 chars
- the matching legacy or canonical lyrics block exists
- lyrics field is <5000 chars (Suno render hard limit; target <=4800)
- ## 3. TITLE exists
- [SONG FORM:] exists in lyrics
- full EMO performance-script section headers exist and no bare architectural EMO labels are used
- at least one short *sfx cue* exists
- no prompt/procedure/QA debris in sung lines
- no obvious real-artist prompt contamination from a small common list

The default sung-line floor remains 60. Personality-specific runs may pass an
explicit lower floor without weakening canonical Lofn validation.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ARTIST_BLOCKLIST = {
    "taylor swift", "billie eilish", "lana del rey", "beyonce", "beyoncé",
    "radiohead", "nine inch nails", "trent reznor", "bjork", "björk", "arca",
    "charli xcx", "fka twigs", "bon iver", "mitski", "sufjan stevens",
}

PROMPT_HEADING = r"(?:1\. MUSIC PROMPT|SUNO STYLE PROMPT)"
EXCLUDE_HEADING = r"(?:1B\. SUNO EXCLUDE PROMPT|SUNO EXCLUDE PROMPT)"
LYRICS_HEADING = r"(?:2\. LYRICS|SUNO ENHANCED LYRICS)"
PROMPT_RE = re.compile(
    rf"^## {PROMPT_HEADING}\s*\n(?P<body>.*?)(?=^## {LYRICS_HEADING}\s*$)", re.M | re.S)
PROMPT_WITH_EXCLUDE_RE = re.compile(
    rf"^## {PROMPT_HEADING}\s*\n(?P<body>.*?)(?=^## {EXCLUDE_HEADING}\s*$)", re.M | re.S)
EXCLUDE_RE = re.compile(
    rf"^## {EXCLUDE_HEADING}\s*\n(?P<body>.*?)(?=^## {LYRICS_HEADING}\s*$)", re.M | re.S)
LYRICS_RE = re.compile(
    rf"^## {LYRICS_HEADING}\s*\n(?P<body>.*?)(?=^## 3\. TITLE\s*$)", re.M | re.S)
TITLE_RE = re.compile(r"^## 3\. TITLE\s*\n(?P<body>.*?)(?=^## |\Z)", re.M | re.S)
STEP11_HEADING_RE = re.compile(r"^#\s+.*\bstep\s*11\b.*$", re.I | re.M)
# Accepts BOTH heading conventions in use across the pipeline: the step-11
# "PACKAGE n" form and the step-10 pair contract's "VARIATION n" form.
# 2026-08-05: three pair agents independently found that a `### VARIATION n`
# file matched ZERO headings here, collapsed to a single block, and reported
# PASS after inspecting 1 of 4 packages -- a false CLEAN over 75% of the work.
PACKAGE_HEADING_RE = re.compile(
    # [2026-08-21] Accept the `# V1 — TITLE` v-form and an optional leading glyph run
    # / ordinal prefix. The 2026-08-21 handoff PINNED `# V<n> — TITLE`, and this reader
    # could not see it: it raised SPLITTER DEFECT rather than scoring V1 four times,
    # which is the correct fail-closed behaviour, but the run's own pinned convention
    # should never trip its own validator. Also accepts `VARIATION V1`.
    r"^#{1,4}[ \t]*(?:[^\w\s#]+[ \t]*)?(?:\d+[A-Za-z]?[.)][ \t]*)?"
    r"(?:(?:enhanced\s+)?(?:package|variation)\s+V?\d+|V0?[1-9])\b.*$", re.I | re.M)
# Cardinality anchor: every real package has exactly one MUSIC PROMPT heading.
PROMPT_HEADING_RE = re.compile(rf"^## {PROMPT_HEADING}\s*$", re.M)


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
    # Remove fence marker lines, not the payload inside them.  The old regex
    # deleted an entire fenced prompt and then reported a misleading length 0.
    return re.sub(r"^```[^\n]*$", "", s, flags=re.M).strip()


def current_step11_segment(text: str) -> str:
    """Isolate the newest Step-11 H1 from a cumulative pair artifact."""
    matches = list(STEP11_HEADING_RE.finditer(text))
    return text[matches[-1].start():] if matches else text


def package_blocks(text: str) -> list[str]:
    """Return every package in the current Step-11 segment, in order."""
    segment = current_step11_segment(text)
    starts = [m.start() for m in PACKAGE_HEADING_RE.finditer(segment)]
    expected = len(PROMPT_HEADING_RE.findall(segment))
    if not starts:
        if expected > 1:
            # The splitter matched no heading but the file plainly holds several
            # packages. Returning [segment] here is how a validator inspects one
            # package and reports PASS for all of them. Fail LOUD instead.
            raise ValueError(
                f"SPLITTER DEFECT: found 0 package/variation headings but "
                f"{expected} '## 1. MUSIC PROMPT' sections. The heading "
                f"convention in this file is not one this validator recognises "
                f"-- fix the splitter, do not trust the verdict.")
        return [segment]
    blocks = [segment[start:(starts[i + 1] if i + 1 < len(starts) else len(segment))]
              for i, start in enumerate(starts)]
    # A file may label one package twice (e.g. "## Package 2" immediately
    # followed by "### VARIATION 2"). A heading with no MUSIC PROMPT beneath it
    # is a LABEL, not a package boundary -- carry it into the block that follows.
    merged: list[str] = []
    carry = ""
    for blk in blocks:
        if not PROMPT_HEADING_RE.search(blk):
            carry += blk          # label-only heading; hold it for the real block
            continue
        merged.append(carry + blk)
        carry = ""
    if carry and merged:          # trailing label-only text belongs to the last block
        merged[-1] += carry
    blocks = merged
    if expected and len(blocks) != expected:
        raise ValueError(
            f"SPLITTER DEFECT: split into {len(blocks)} blocks but found "
            f"{expected} '## 1. MUSIC PROMPT' sections -- extraction cardinality "
            f"does not match. Fix the splitter, do not trust the verdict.")
    return blocks


def validate_package(
    text: str,
    min_sung_lines: int = 60,
    max_sung_lines: int | None = None,
) -> list[str]:
    errs: list[str] = []

    pm = PROMPT_WITH_EXCLUDE_RE.search(text) or PROMPT_RE.search(text)
    em = EXCLUDE_RE.search(text)
    lm = LYRICS_RE.search(text)
    tm = TITLE_RE.search(text)
    if not pm:
        errs.append("missing Suno style prompt block before lyrics")
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
        errs.append("missing Suno exclude prompt block before lyrics")
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
        errs.append("missing Suno lyrics block before ## 3. TITLE")
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
        if len(sung_lines) < min_sung_lines:
            errs.append(
                f"only {len(sung_lines)} probable sung lines, expected >={min_sung_lines}"
            )
        if max_sung_lines is not None and len(sung_lines) > max_sung_lines:
            errs.append(
                f"{len(sung_lines)} probable sung lines, expected <={max_sung_lines}"
            )
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


def validate(
    path: Path,
    min_sung_lines: int = 60,
    max_sung_lines: int | None = None,
) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks = package_blocks(text)
    errs: list[str] = []
    for index, block in enumerate(blocks, start=1):
        errs.extend(
            f"package {index}: {err}"
            for err in validate_package(
                block,
                min_sung_lines=min_sung_lines,
                max_sung_lines=max_sung_lines,
            )
        )
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--min-sung-lines", type=int, default=60,
                    help="run/personality-specific hard floor; default preserves canonical Lofn behavior")
    ap.add_argument("--max-sung-lines", type=int, default=None,
                    help="optional run/personality-specific hard ceiling")
    args = ap.parse_args()
    if args.min_sung_lines < 1:
        ap.error("--min-sung-lines must be positive")
    if args.max_sung_lines is not None and args.max_sung_lines < args.min_sung_lines:
        ap.error("--max-sung-lines must be >= --min-sung-lines")
    files = iter_files(args.paths)
    if not files:
        print("No markdown files found", file=sys.stderr)
        return 2
    failed = 0
    for f in files:
        errs = validate(
            f,
            min_sung_lines=args.min_sung_lines,
            max_sung_lines=args.max_sung_lines,
        )
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
