"""Build a genuinely blind comparison set for lofn-qa procedure 4.5.

Lesson L10 (vault/COMPETITION_LEARNINGS.md, 2026-07-01): a blind set whose
members self-identify, or which contains an empty stub, VOIDS the calibration
and must be rebuilt, not ranked around. This script therefore fails CLOSED:
any extraction problem is exit 1, never a silent skip.

What it does:
  - package mode extracts, per candidate variation, ONLY the music prompt,
    EXCLUDE field, and lyrics field;
  - lyrics mode extracts ONLY sung words for the sealed Absolute Force read:
    no title, prompt, Theme/Form/EMO headers, rationale, score, or identity;
  - step-11 multi-variation files are split correctly in both modes;
  - strips every provenance line (pair ids, run ids, verdicts, self-checks,
    Major Deviations, byte counts, "measured" annotations);
  - adds one or more golden controls + one decoy the same way (title lines removed; internal
    lyric text stays — a lyric that names its own title is content, not
    metadata);
  - shuffles with an explicit seed, writes package_A..Z.md, one member per
    file, RE-STATS each (mode floor or exit 1), and writes the
    coordinator-only key OUTSIDE the blind directory.

Usage:
  python scripts/build_blind_set.py --mode lyrics --out <blind_dir> \
      --key <key_file> --seed <int> --golden <file> [--golden <file> ...] \
      --decoy <file> <candidate_files...>
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

MIN_BYTES_BY_MODE = {"package": 1500, "lyrics": 600}

# lines that leak provenance — dropped everywhere
_LEAK = re.compile(
    r"(?i)(pair[ _]?0?\d|run:|source:|golden|staff pick|era-mate|suno url|"
    r"publication status|verdict|self-check|major deviation|measured|"
    r"gate_report|provenance|continuity payload|special flairs|icb|"
    r"selected_for_top_six|step ?1[01]|variation angle|bytes)"
)

# Lyrics may legitimately sing words such as "golden", "source", or
# "measured". Only provenance-shaped field lines are leaks in lyrics mode.
_LYRIC_META = re.compile(
    r"(?i)^(?:pair(?:[ _]?id)?\s*:|run(?:[ _]?id)?\s*:|source\s*:|"
    r"publication status\s*:|verdict\s*:|self-check\s*:|major deviation\s*:|"
    r"gate_report\s*:|provenance\s*:|selected_for_top_six\s*:|"
    r"step ?1[01]\s*:|variation angle\s*:|bytes\s*:|staff pick\s*:)"
)


def _clean(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not _LEAK.search(l)).strip()


def _split_variations(text: str) -> list[str]:
    """Split a step-11 package into per-variation payload chunks.

    A variation starts at a heading that contains 'V<N>' or 'VARIATION <N>'
    and runs to the next such heading (or EOF). Falls back to the whole file
    as one chunk when no variation headings exist.
    """
    # boundary = the FIRST heading mentioning each variation number, in order;
    # later sub-headings that re-mention V<N> never split.
    firsts: dict[int, int] = {}
    for m in re.finditer(r"^#+ .*(?:\bV([1-9])\b|VARIATION ([1-9]))", text, re.M):
        n = int(m.group(1) or m.group(2))
        if n not in firsts:
            firsts[n] = m.start()
    marks = sorted(firsts.values())
    if not marks:
        return [text]
    marks.append(len(text))
    return [text[a:b] for a, b in zip(marks, marks[1:])]


def _package_payload(chunk: str) -> str:
    """Keep only prompt/exclude/lyrics sections of a chunk, cleaned."""
    keep: list[str] = []
    for sec in re.split(r"(?=^#+ )", chunk, flags=re.M):
        head = sec.splitlines()[0] if sec.splitlines() else ""
        if re.search(r"(?i)(music prompt|exclude|lyrics|theme)", head) or \
                sec.lstrip().startswith("["):
            keep.append(sec)
    body = _clean("\n".join(keep) if keep else chunk)
    return body


def _drop_frontmatter(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[i + 1:])
    return text


def _heading_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) sections; preamble is intentionally omitted."""
    marks = list(re.finditer(r"^#{1,6}\s+(.+?)\s*$", text, re.M))
    return [
        (m.group(1), text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)])
        for i, m in enumerate(marks)
    ]


def _plain_sung_words(text: str) -> str:
    """Strip judge-biasing package notation while preserving the lyric itself."""
    kept: list[str] = []
    for raw in _drop_frontmatter(text).splitlines():
        line = raw.strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if line.startswith(("#", ">", "```")) or line == "---":
            continue
        if re.fullmatch(r"\[(?:Theme|Song Form)\s*:.*\]", line, re.I):
            continue
        if re.fullmatch(r"\[[^\]]+\]", line):  # section / EMO / production header
            continue
        if re.fullmatch(r"\*[^*]+\*", line):   # non-lexical or SFX cue
            continue
        if line.startswith("|") or re.fullmatch(r"[-:| ]+", line):
            continue
        line = re.sub(r"[*_]{1,2}", "", line).strip()
        if line and not _LYRIC_META.search(line):
            kept.append(line)
    while kept and kept[-1] == "":
        kept.pop()
    return "\n".join(kept)


def _lyrics_payload(chunk: str) -> str:
    """Extract the largest lyrics-labelled section, or clean a plain-lyric file."""
    lyric_sections = [body for head, body in _heading_sections(chunk)
                      if re.search(r"(?i)\blyrics?\b", head)]
    candidates = [_plain_sung_words(body) for body in lyric_sections]
    candidates = [body for body in candidates if body]
    if candidates:
        return max(candidates, key=lambda body: len(body.encode("utf-8")))
    return _plain_sung_words(chunk)


def _payload(chunk: str, mode: str) -> str:
    return _lyrics_payload(chunk) if mode == "lyrics" else _package_payload(chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=sorted(MIN_BYTES_BY_MODE), default="package")
    ap.add_argument("--golden", action="append", required=True,
                    help="Golden/control file; repeat for an absolute bracket")
    ap.add_argument("--decoy", required=True)
    ap.add_argument("candidates", nargs="+")
    a = ap.parse_args()
    min_bytes = MIN_BYTES_BY_MODE[a.mode]

    out = Path(a.out)
    key_path = Path(a.key)
    if key_path.resolve().is_relative_to(out.resolve()):
        print("FATAL: key file must live OUTSIDE the blind directory", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, str]] = []
    for path in a.candidates:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        chunks = _split_variations(text)
        stem = Path(path).stem
        kept = 0
        for i, ch in enumerate(chunks, 1):
            body = _payload(ch, a.mode)
            if len(body.encode()) >= min_bytes:
                entries.append((f"CAND:{stem}:V{i}", body))
                kept += 1
        print(f"  {stem}: {len(chunks)} chunk(s) -> {kept} member(s)")
    controls = [(f"GOLDEN:{Path(path).stem}", path) for path in a.golden]
    controls.append(("DECOY", a.decoy))
    for tag, path in controls:
        body = _payload(Path(path).read_text(encoding="utf-8", errors="replace"), a.mode)
        if len(body.encode()) < min_bytes:
            print(f"FATAL: {tag} payload under {min_bytes}B after cleaning", file=sys.stderr)
            return 1
        entries.append((tag, body))

    n_cands = len(entries) - len(controls)
    if n_cands < 1:
        print("FATAL: zero candidate variations extracted", file=sys.stderr)
        return 1

    random.seed(a.seed)
    random.shuffle(entries)

    key_lines = []
    for i, (tag, body) in enumerate(entries):
        label = (chr(ord("A") + i) if i < 26
                 else "A" + chr(ord("A") + i - 26))  # AA, AB, ... past Z
        p = out / f"package_{label}.md"
        p.write_text(body, encoding="utf-8")
        size = p.stat().st_size
        if size < min_bytes:
            print(f"FATAL: {p.name} re-stats at {size}B (< {min_bytes}) — stub; set VOID", file=sys.stderr)
            return 1
        leak_pattern = _LYRIC_META if a.mode == "lyrics" else _LEAK
        leak = [l for l in body.splitlines() if leak_pattern.search(l)]
        if leak:
            print(f"FATAL: {p.name} still carries provenance: {leak[0][:60]}", file=sys.stderr)
            return 1
        key_lines.append(f"{label} = {tag}")
    key_path.write_text("\n".join(key_lines), encoding="utf-8")
    print(f"OK: mode={a.mode}; {len(entries)} members "
          f"({n_cands} candidates + {len(a.golden)} golden + decoy), "
          f"all >= {min_bytes}B, key at {key_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
