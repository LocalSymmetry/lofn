#!/usr/bin/env python3
"""Validate cross-pair distinctiveness for all Lofn final packages.

Catches the failure mode where each pair is structurally valid in isolation but
shares the same prompt/lyric skeleton across the portfolio.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

try:
    # scripts/ is sys.path[0] when run directly; single-source the ceilings from
    # vault/gates.yaml. Fail-open: a broken import falls back to the CLI defaults.
    from validate_step import load_gates
except Exception:  # noqa: BLE001
    def load_gates(path=None):  # type: ignore
        return {}


def norm(s: str) -> str:
    # Preserve Suno v5.5 structured prompt fields (`[genre: ...]`) while still
    # flattening ordinary lyric section headers for comparison.
    s = re.sub(r"\[([a-z_ ]{2,32})\s*:\s*([^\]]+)\]", r"\1 \2", s, flags=re.I)
    s = re.sub(r"\[[^\]]+\]", "", s)  # remove remaining performance headers
    s = re.sub(r"\*[^*]+\*", "", s)  # remove sfx
    s = re.sub(r"\b(pair|variant)\s+\w+\b", "", s, flags=re.I)
    s = re.sub(r"[^a-z0-9' ]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stage_segment(text: str, step: int) -> str:
    """Return only the newest requested stage from a cumulative artifact.

    Pair chains legitimately use H1, H2, or H3 for their current stage.  The
    old H1-only reader retained the shared ~97 KB ICB in several canonical
    files, turning the pairwise comparison into a prohibitively slow audit of
    context that is supposed to be excluded.
    """
    marker = "# ===== END CREATIVE CONTEXT ====="
    cut = text.rfind(marker)
    current = text[cut + len(marker):] if cut >= 0 else text
    matches = list(re.finditer(rf"^#{{1,3}}\s+.*\bstep\s*{step}\b.*$", current, re.I | re.M))
    if not matches:
        return current
    # The first requested-stage heading owns the artifact. Later same-step
    # headings (for example ``## Terminal Step 10 Object``) are subsections,
    # not replacement stage boundaries. The ICB marker, when present, has
    # already removed historical context.
    return current[matches[0].start():]


def step10_segment(text: str) -> str:
    """Backward-compatible Step-10 stage selector."""
    return stage_segment(text, 10)


def first_variant(text: str) -> str:
    starts = [m.start() for m in re.finditer(r"^#{1,3}\s+VARIA(?:NT|TION)\b", text, re.I | re.M)]
    if not starts:
        return text
    start = starts[0]
    later = [x for x in starts if x > start]
    end = later[0] if later else len(text)
    return text[start:end]


def variants(text: str) -> list[str]:
    """Return every Step-10 finalist block from the current stage in order."""
    starts = [m.start() for m in re.finditer(
        r"^#{1,3}\s+(?:VARIA(?:NT|TION)\b|(?:REVISED|SYNTHESIZED)\s+PROMPT\s+\d+\b)",
        text,
        re.I | re.M,
    )]
    if not starts:
        return [text]
    return [text[start:(starts[i + 1] if i + 1 < len(starts) else len(text))]
            for i, start in enumerate(starts)]


def packages(text: str) -> list[str]:
    """Return every Step-11 Package/Enhanced Package block in order."""
    starts = [m.start() for m in re.finditer(
        r"^#{1,3}\s+(?:ENHANCED\s+)?PACKAGE\s+\d+\b", text, re.I | re.M
    )]
    if not starts:
        return [text]
    return [text[start:(starts[i + 1] if i + 1 < len(starts) else len(text))]
            for i, start in enumerate(starts)]


def prompt_blocks(text: str) -> list[str]:
    """Split current-stage output on actual Suno positive-prompt field headers.

    Pair writers legitimately use several finalist-heading dialects.  The
    renderer field itself is the stable boundary shared by every dialect.
    """
    starts = [m.start() for m in re.finditer(
        r"^##\s*(?:1\.\s*MUSIC PROMPT|SUNO STYLE PROMPT)\s*$", text, re.I | re.M
    )]
    if not starts:
        return [text]
    return [text[start:(starts[i + 1] if i + 1 < len(starts) else len(text))]
            for i, start in enumerate(starts)]


def lyrics_only(text: str) -> str:
    m = re.search(
        r"##\s*(?:2\.\s*LYRICS|SUNO ENHANCED LYRICS)\s*(.*?)(?=\n##\s+|\nGATE CHECK|\n#{2,3}\s+VARIANT|\Z)",
        text,
        re.I | re.S,
    )
    return m.group(1) if m else text


def prompt_only(text: str) -> str:
    m = re.search(
        r"##\s*(?:1\.\s*MUSIC PROMPT|SUNO STYLE PROMPT)\s*(.*?)"
        r"(?=\n##\s*(?:(?:1B\.\s*SUNO\s+)?EXCLUDE PROMPT|SUNO EXCLUDE PROMPT|"
        r"2\.\s*LYRICS|SUNO ENHANCED LYRICS))",
        text,
        re.I | re.S,
    )
    return m.group(1) if m else ""


def ngrams(s: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = s.split()
    return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))


def main() -> int:
    g = load_gates()
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("--step", type=int, choices=(10, 11), default=10)
    ap.add_argument("--expected-total", type=int, default=None)
    ap.add_argument("--max-lyric-sim", type=float, default=None)
    ap.add_argument("--max-prompt-sim", type=float, default=None)
    ap.add_argument("--max-ngram-jaccard", type=float, default=None)
    args = ap.parse_args()
    max_lyric_sim = args.max_lyric_sim if args.max_lyric_sim is not None else float(g.get("portfolio_max_lyric_similarity", 0.42))
    max_prompt_sim = args.max_prompt_sim if args.max_prompt_sim is not None else float(g.get("portfolio_max_prompt_similarity", 0.58))
    max_ngram_jaccard = args.max_ngram_jaccard if args.max_ngram_jaccard is not None else float(g.get("portfolio_max_ngram_jaccard", 0.18))

    root = Path(args.audio_dir)
    if args.step == 10:
        patterns = ("pair_*_step10_revision_synthesis.md",)
    else:
        # Both names are canonical in live runs: the latter is the current
        # TASK_TEMPLATE handoff filename even though its H1 correctly says
        # Step 11.  A reader that knows only the historical filename silently
        # loses the entire final portfolio.
        patterns = (
            "pair_*_step11_final_package_enhanced.md",
            "pair_*_step10_final_package_enhanced.md",
        )
    files = sorted({
        p
        for pattern in patterns
        for p in root.glob(pattern)
        if ".repair_attempt_" not in p.name
    })
    # Final render-facing Step-11 selections use stable ordinal filenames such
    # as `01_title.md` rather than cumulative pair-artifact names. Accept that
    # canonical delivery layout only when the historical pattern finds none.
    if args.step == 11 and not files:
        files = sorted(p for p in root.glob("*.md") if re.match(r"^\d{2}_.+\.md$", p.name, re.I))
    if len(files) < 2:
        print(f"FAIL: need at least two pair Step {args.step} files")
        return 1

    entries = []
    for p in files:
        # Cumulative artifacts preserve prior stages as exact byte prefixes.
        # Isolate the current stage before locating every finalist.
        current = stage_segment(p.read_text(errors="replace"), args.step)
        blocks = prompt_blocks(current)
        for index, block in enumerate(blocks, start=1):
            entries.append((p.name, f"{p.name}#V{index:02d}",
                            norm(prompt_only(block)), norm(lyrics_only(block))))

    if args.expected_total is not None and len(entries) != args.expected_total:
        print(f"FAIL: expected {args.expected_total} finalist packages, found {len(entries)}")
        return 1

    # An empty extraction is an INSTRUMENT failure, never a similarity score.
    # Two empty strings compare as 1.000 (2026-08-04: 30 phantom collisions at
    # exactly 1.000 between pairs with different genres, keys and tempos).
    empty = [label for _f, label, pr, ly in entries if not pr.strip() or not ly.strip()]
    if empty:
        print("ERROR: extraction returned empty text — the parser did not match this "
              "run's heading convention. This is a validator defect, NOT a distinctiveness result.")
        for label in empty:
            print(f"- empty extraction: {label}")
        return 2

    print(f"extracted {len(entries)} finalist packages from {len(files)} pair files")
    for _f, label, pr, ly in entries:
        print(f"  {label}: prompt {len(pr)} chars, lyrics {len(ly)} chars")

    failures = []
    for i in range(len(entries)):
        for j in range(i+1, len(entries)):
            afile, a, aprompt, alyr = entries[i]
            bfile, b, bprompt, blyr = entries[j]
            if afile == bfile:
                continue
            # autojunk=False is MANDATORY: the default discards frequently-occurring
            # characters in sequences over 200 elements, which silently UNDER-reports
            # similarity on long near-identical prompts (2026-08-05: four template
            # prompts reported as 94% distinct).
            psim = difflib.SequenceMatcher(None, aprompt, bprompt, autojunk=False).ratio()
            lsim = difflib.SequenceMatcher(None, alyr, blyr, autojunk=False).ratio()
            ag, bg = ngrams(alyr), ngrams(blyr)
            jac = len(ag & bg) / max(1, len(ag | bg))
            if psim > max_prompt_sim:
                failures.append(f"prompt similarity too high {a} vs {b}: {psim:.3f}")
            if lsim > max_lyric_sim:
                failures.append(f"lyric similarity too high {a} vs {b}: {lsim:.3f}")
            if jac > max_ngram_jaccard:
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
