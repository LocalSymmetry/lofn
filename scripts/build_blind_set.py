"""Build a genuinely blind golden+decoy set for lofn-qa procedure 4.5.

Lesson L10 (vault/COMPETITION_LEARNINGS.md, 2026-07-01): a blind set whose
members self-identify, or which contains an empty stub, VOIDS the calibration
and must be rebuilt, not ranked around. This script therefore fails CLOSED:
any extraction problem is exit 1, never a silent skip.

What it does:
  - extracts, per candidate variation, ONLY the bare payload — the
    `## 1. MUSIC PROMPT` paragraph, the EXCLUDE field, and the lyrics field —
    from step-11 packages (multi-variation files are split correctly);
  - strips every provenance line (pair ids, run ids, verdicts, self-checks,
    Major Deviations, byte counts, "measured" annotations);
  - adds golden + decoy payloads the same way (title lines removed; internal
    lyric text stays — a lyric that names its own title is content, not
    metadata);
  - shuffles with an explicit seed, writes package_A..Z.md, one member per
    file, RE-STATS each (>= 1500 bytes or exit 1), and writes the
    coordinator-only key OUTSIDE the blind directory.

Usage:
  python scripts/build_blind_set.py --out <blind_dir> --key <key_file> \
      --seed <int> --golden <file> --decoy <file> <candidate_step11_files...>
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

MIN_BYTES = 1500

# lines that leak provenance — dropped everywhere
_LEAK = re.compile(
    r"(?i)(pair[ _]?0?\d|run:|source:|golden|staff pick|era-mate|suno url|"
    r"publication status|verdict|self-check|major deviation|measured|"
    r"gate_report|provenance|continuity payload|special flairs|icb|"
    r"selected_for_top_six|step ?1[01]|variation angle|bytes)"
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


def _payload(chunk: str) -> str:
    """Keep only prompt/exclude/lyrics sections of a chunk, cleaned."""
    keep: list[str] = []
    for sec in re.split(r"(?=^#+ )", chunk, flags=re.M):
        head = sec.splitlines()[0] if sec.splitlines() else ""
        if re.search(r"(?i)(music prompt|exclude|lyrics|theme)", head) or \
                sec.lstrip().startswith("["):
            keep.append(sec)
    body = _clean("\n".join(keep) if keep else chunk)
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--decoy", required=True)
    ap.add_argument("candidates", nargs="+")
    a = ap.parse_args()

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
            body = _payload(ch)
            if len(body.encode()) >= MIN_BYTES:
                entries.append((f"CAND:{stem}:V{i}", body))
                kept += 1
        print(f"  {stem}: {len(chunks)} chunk(s) -> {kept} member(s)")
    for tag, path in (("GOLDEN", a.golden), ("DECOY", a.decoy)):
        body = _payload(Path(path).read_text(encoding="utf-8", errors="replace"))
        if len(body.encode()) < MIN_BYTES:
            print(f"FATAL: {tag} payload under {MIN_BYTES}B after cleaning", file=sys.stderr)
            return 1
        entries.append((tag, body))

    n_cands = len(entries) - 2
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
        if size < MIN_BYTES:
            print(f"FATAL: {p.name} re-stats at {size}B (< {MIN_BYTES}) — stub; set VOID", file=sys.stderr)
            return 1
        leak = [l for l in body.splitlines() if _LEAK.search(l)]
        if leak:
            print(f"FATAL: {p.name} still carries provenance: {leak[0][:60]}", file=sys.stderr)
            return 1
        key_lines.append(f"{label} = {tag}")
    key_path.write_text("\n".join(key_lines), encoding="utf-8")
    print(f"OK: {len(entries)} members ({n_cands} candidates + golden + decoy), "
          f"all >= {MIN_BYTES}B, key at {key_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
