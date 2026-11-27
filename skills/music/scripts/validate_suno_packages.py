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
- lyrics field is <5000 chars (Suno render hard limit; v6 target 2600-3600, soft ceiling 3800)
- ## 3. TITLE exists
- [SONG FORM:] exists in lyrics
- full EMO performance-script section headers exist and no bare architectural EMO labels are used
- at least one standalone *SFX cue* exists and fits `sfx_cue_max_chars`
  (vault/gates.yaml; FAIL-OPEN to a built-in 60 if that file/key is unreadable)
- no prompt/procedure/QA debris in sung lines
- no obvious real-artist prompt contamination from a small common list

LENGTH (V6-19, 2026-11-25): the default length gate is SUNG WORDS, read from
`vault/gates.yaml` (`sung_words_hard_floor` / `sung_words_hard_ceiling`, FAIL-OPEN to
300 / 520; whitespace tokens on the sung lines, the same unit as scripts/validate_step.py).
The line count is no longer checked by default: measured on 98 of our v6 lyrics, lines
add nothing to predicted length once words are known
(output/analysis/2026-11-25_v6-330-length-budget/RESULT.md). A run may still pass
`--min-sung-lines` / `--max-sung-lines`, or `--min-sung-words` / `--max-sung-words`,
for a declared form. A gates file with `length_gate: lines` restores the line floor.
"""
from __future__ import annotations

import argparse
import functools
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# SFX cue cap -- the ONE threshold in this file that is single-sourced.
#
# [2026-09-11] The cap was hardcoded `{1,40}` here, `{1,50}` at
# scripts/validate_step.py:743 and `{0,80}` at scripts/clean_lyrics.py:48 --
# three numbers for one concept, with run-local copies inheriting whichever one
# their author happened to read. Measured over 1600 cue lines in 912 package
# blocks, 40 sits exactly on p95: the gate was rejecting the house's own top 5%
# of cues, and it failed two finished songs on cues of 43 and 47 chars. The
# value now lives in vault/gates.yaml (`sfx_cue_max_chars`, measured basis in
# the comment there).
#
# FAIL-OPEN, per gates.yaml:12-14 -- "A broken gates.yaml must NEVER hard-fail
# an otherwise-valid run." A missing file, an unreadable one, a missing key, or
# a non-positive-integer value all fall back to DEFAULT_SFX_CUE_MAX_CHARS. This
# script therefore reads gates.yaml but never DEPENDS on it.
# --------------------------------------------------------------------------
DEFAULT_SFX_CUE_MAX_CHARS = 60
DEFAULT_MIN_SUNG_LINES = 45      # read only under `length_gate: lines`
DEFAULT_MIN_SUNG_WORDS = 300
DEFAULT_MAX_SUNG_WORDS = 520


def _repo_root() -> Path:
    # skills/music/scripts/validate_suno_packages.py -> repo root is 3 levels up.
    return Path(__file__).resolve().parent.parent.parent.parent


def load_min_sung_lines(gates_path: Path | None = None) -> int:
    """Return sung_lines_hard_floor from vault/gates.yaml; FAIL-OPEN to 45."""
    try:
        root = str(_repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        from scripts.validate_step import load_gates  # type: ignore

        value = load_gates(gates_path).get("sung_lines_hard_floor")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return DEFAULT_MIN_SUNG_LINES
        return value
    except Exception:  # noqa: BLE001 -- fail open on ANYTHING
        return DEFAULT_MIN_SUNG_LINES


@functools.lru_cache(maxsize=1)
def _cached_min_sung_lines() -> int:
    return load_min_sung_lines()


def load_length_gate(gates_path: Path | None = None) -> dict:
    """Return {'mode', 'min_words', 'max_words'} from vault/gates.yaml.

    FAIL-OPEN to the built-in word gate (300-520) on any problem. `mode` is
    'words' (house default since 2026-11-25) or 'lines' (older gates files).
    """
    out = {"mode": "words", "min_words": DEFAULT_MIN_SUNG_WORDS, "max_words": DEFAULT_MAX_SUNG_WORDS}
    try:
        root = str(_repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        from scripts.validate_step import load_gates  # type: ignore

        gates = load_gates(gates_path)
        mode = str(gates.get("length_gate", "words")).strip().lower()
        out["mode"] = "lines" if mode == "lines" else "words"
        for key, name in (("sung_words_hard_floor", "min_words"), ("sung_words_hard_ceiling", "max_words")):
            value = gates.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
                out[name] = value
    except Exception:  # noqa: BLE001 -- fail open on ANYTHING
        pass
    return out


@functools.lru_cache(maxsize=1)
def _cached_length_gate() -> tuple:
    g = load_length_gate()
    return (g["mode"], g["min_words"], g["max_words"])


def load_sfx_cue_max_chars(gates_path: Path | None = None) -> int:
    """Return the SFX cue char cap from vault/gates.yaml; FAIL-OPEN to 60.

    Reuses scripts/validate_step.py's ``load_gates`` -- the house already owns
    the gates reader (including its PyYAML-optional fallback parser), and a
    second implementation is a second thing to drift.
    """
    try:
        root = str(_repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        from scripts.validate_step import load_gates  # type: ignore

        value = load_gates(gates_path).get("sfx_cue_max_chars")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            if value is not None:
                print(
                    f"WARN: sfx_cue_max_chars={value!r} is not a positive int; "
                    f"using built-in default {DEFAULT_SFX_CUE_MAX_CHARS}",
                    file=sys.stderr,
                )
            return DEFAULT_SFX_CUE_MAX_CHARS
        return value
    except Exception as exc:  # noqa: BLE001 -- fail open on ANYTHING
        print(
            f"WARN: could not read sfx_cue_max_chars ({exc}); using built-in "
            f"default {DEFAULT_SFX_CUE_MAX_CHARS}",
            file=sys.stderr,
        )
        return DEFAULT_SFX_CUE_MAX_CHARS


@functools.lru_cache(maxsize=1)
def _cached_sfx_cue_max_chars() -> int:
    """Default-path loader: read gates.yaml once per process, warn once."""
    return load_sfx_cue_max_chars()


# Any standalone single-asterisk line, unbounded. The LENGTH is judged
# afterwards so a present-but-long cue is reported as long, never as missing.
SFX_CUE_RE = re.compile(r"^\*([^*\n]+)\*\s*$", re.M)

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
    # [2026-11-25] The v-form also matched the `## V6 SETTINGS` heading (the model
    # version, not a package number) that the 2026-11-25 step-11 contract pins, so a
    # one-package enhanced file split into 0 blocks and raised SPLITTER DEFECT. A
    # V<n> heading followed by SETTINGS is a settings block, never a package.
    # PROVENANCE: this lookahead was added at 21:19Z on 2026-11-25 by a step-11 pair
    # subagent in the middle of its own wave, outside its remit (a pair agent does not
    # edit the reader that scores it). The coordinator reviewed the diff the same hour,
    # found it to be the whole change and correct, kept it, and added the regression
    # test `test_v6_settings_heading_is_not_a_package_heading`. The cause was the
    # coordinator's: the packet pinned the heading without first running the shape
    # through this reader.
    r"^#{1,4}[ \t]*(?:[^\w\s#]+[ \t]*)?(?:\d+[A-Za-z]?[.)][ \t]*)?"
    r"(?:(?:enhanced\s+)?(?:package|variation)\s+V?\d+|V0?[1-9](?![ \t]+SETTINGS\b)|(?:terminal\s+)?P\d{2}-[RS]\d+)\b.*$", re.I | re.M)
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
    # A single fenced block is the pasted field; adjacent production-wrap
    # headings/notes are not part of it. Keep multiple blocks visible to the
    # validator rather than arbitrarily choosing one ambiguous field.
    fields = re.findall(r"^```[^\n]*\n(.*?)^```\s*$", s, re.M | re.S)
    if len(fields) == 1:
        return fields[0].strip()
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
    min_sung_lines: int | None = None,
    max_sung_lines: int | None = None,
    sfx_cue_max_chars: int | None = None,
    min_sung_words: int | None = None,
    max_sung_words: int | None = None,
) -> list[str]:
    errs: list[str] = []
    mode, gate_min_words, gate_max_words = _cached_length_gate()
    # An explicit line or word bound from the caller is a declared form: only
    # the bounds the caller gave are checked. With none given, the house gate
    # applies: sung words (V6-19), or the line floor under `length_gate: lines`.
    caller_declared = any(v is not None for v in (min_sung_lines, max_sung_lines, min_sung_words, max_sung_words))
    if not caller_declared:
        if mode == "lines":
            min_sung_lines = _cached_min_sung_lines()
        else:
            min_sung_words, max_sung_words = gate_min_words, gate_max_words
    if sfx_cue_max_chars is None:
        sfx_cue_max_chars = _cached_sfx_cue_max_chars()

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
        hits = sorted(a for a in ARTIST_BLOCKLIST
                      if re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", low))
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
        lyrics = strip_md(lm.group("body"))
        if "[SONG FORM:" not in lyrics:
            errs.append("missing [SONG FORM:] in lyrics")
        if "EMO:" not in lyrics:
            errs.append("missing EMO: headers in lyrics")
        if re.search(r"EMO:\s*(AWE|INDIGNATION|SYNTHESIS)\s*(?:[\]–\-]|$)", lyrics, re.I):
            errs.append("bare architectural EMO label used")
        # A block passes if ANY ONE standalone cue is within the cap. Report the
        # two failure states apart: "missing" when there is no cue at all, and
        # "too long" when one is present but over the cap. The old message said
        # `missing standalone short *SFX cue*` for BOTH, which cost this run
        # time on two shipped songs whose cues were present at 43 and 47 chars.
        cues = [m.group(1) for m in SFX_CUE_RE.finditer(lyrics)]
        if not cues:
            errs.append("missing standalone *SFX cue* in lyrics")
        elif not any(len(c) <= sfx_cue_max_chars for c in cues):
            worst = min(cues, key=len)          # the one closest to passing
            shown = worst if len(worst) <= 60 else worst[:57] + "..."
            errs.append(
                f"*SFX cue* too long ({len(worst)} chars, max {sfx_cue_max_chars}): "
                f"*{shown}*"
                + (f" [+{len(cues) - 1} other cue(s), all over cap]" if len(cues) > 1 else "")
            )
        if re.search(r"\b(prompt|QA gate|taxonomy|production manual|this song is about)\b", lyrics, re.I):
            errs.append("possible prompt/procedure/QA debris in lyrics")
        sung_lines = [ln for ln in lyrics.splitlines() if ln.strip() and not ln.strip().startswith("[") and not ln.strip().startswith("*") and not ln.strip().startswith("#")]
        if min_sung_lines is not None and len(sung_lines) < min_sung_lines:
            errs.append(
                f"only {len(sung_lines)} probable sung lines, expected >={min_sung_lines}"
            )
        if max_sung_lines is not None and len(sung_lines) > max_sung_lines:
            errs.append(
                f"{len(sung_lines)} probable sung lines, expected <={max_sung_lines}"
            )
        sung_words = sum(len(ln.split()) for ln in sung_lines)
        if min_sung_words is not None and sung_words < min_sung_words:
            errs.append(
                f"only {sung_words} sung words in {len(sung_lines)} lines, expected >={min_sung_words} "
                f"(target about 400 for a usual 3:30)"
            )
        if max_sung_words is not None and sung_words > max_sung_words:
            errs.append(
                f"{sung_words} sung words in {len(sung_lines)} lines, expected <={max_sung_words} "
                f"(target about 400 for a usual 3:30; declare the form in RUN_GATES.yaml if it is meant)"
            )
        # Suno lyrics-field HARD CAP: everything pasted into Suno's lyrics box
        # (Theme + SONG FORM + all headers + SFX + sung lines) must be under
        # 5000 chars or Suno will not render. V6 target 2600-3600 (soft ceiling 3800).
        fence = re.search(r"```[a-zA-Z]*\n(.*?)```", lyrics, re.S)
        field = (fence.group(1) if fence else lyrics).strip()
        field_len = len(field)
        if field_len >= 5000:
            errs.append(f"lyrics field {field_len} chars exceeds Suno hard limit (must be <5000, v6 target 2600-3600, soft ceiling 3800); trim lines/headers and keep Disc_Channel in ## Production Sidecar")

    if not tm or not strip_md(tm.group("body")):
        errs.append("missing ## 3. TITLE content")

    return errs


def validate(
    path: Path,
    min_sung_lines: int | None = None,
    max_sung_lines: int | None = None,
    sfx_cue_max_chars: int | None = None,
    min_sung_words: int | None = None,
    max_sung_words: int | None = None,
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
                sfx_cue_max_chars=sfx_cue_max_chars,
                min_sung_words=min_sung_words,
                max_sung_words=max_sung_words,
            )
        )
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--min-sung-lines", type=int, default=None,
                    help="optional declared-form line floor (no line check by default since 2026-11-25)")
    ap.add_argument("--max-sung-lines", type=int, default=None,
                    help="optional declared-form line ceiling")
    ap.add_argument("--min-sung-words", type=int, default=None,
                    help="declared-form word floor; default reads vault/gates.yaml sung_words_hard_floor (300)")
    ap.add_argument("--max-sung-words", type=int, default=None,
                    help="declared-form word ceiling; default reads vault/gates.yaml sung_words_hard_ceiling (520)")
    args = ap.parse_args()
    for name in ("min_sung_lines", "max_sung_lines", "min_sung_words", "max_sung_words"):
        value = getattr(args, name)
        if value is not None and value < 1:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.min_sung_lines is not None and args.max_sung_lines is not None and args.max_sung_lines < args.min_sung_lines:
        ap.error("--max-sung-lines must be >= --min-sung-lines")
    if args.min_sung_words is not None and args.max_sung_words is not None and args.max_sung_words < args.min_sung_words:
        ap.error("--max-sung-words must be >= --min-sung-words")
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
            min_sung_words=args.min_sung_words,
            max_sung_words=args.max_sung_words,
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
