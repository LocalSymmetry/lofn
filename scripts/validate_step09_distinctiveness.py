#!/usr/bin/env python3
"""Validate cross-pair distinctiveness for Lofn Step 09 artist-refinement files.

Sibling of ``validate_step06_distinctiveness.py`` (step 06) and
``validate_portfolio_distinctiveness.py`` (step 10). Honors EXECUTION.md §4:
"Distinctiveness across pairs is checked at 06, 09, and 10." Catches the case
where each pair's artist-refine pass reused one critique skeleton — the
mid-chain convergence that a step-10-only check would miss until the end.

Thresholds are single-sourced from ``vault/gates.yaml`` (FAIL-OPEN to built-in
defaults that mirror it). A breach is a REPAIR TRIGGER (bounded max-3 loop ->
quarantine -> human surface), NOT a run kill; a deliberate shared motif that
survives repair is a human waive, never a silent ship.

Usage:
  validate_step09_distinctiveness.py <run_dir> [--max-sim F]
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

try:
    from validate_step import load_gates
except Exception:  # noqa: BLE001 — a broken import must never block a run
    def load_gates(path=None):  # type: ignore
        return {}


# Generic step-09 phrasing that, repeated across pairs, signals a cloned critique
# skeleton rather than a pair-specific artist refinement.
BOILERPLATE_PHRASES = [
    "refine toward sharper medium identity",
    "remove generic cinematic swelling",
    "artist critique demands active verbs",
    "concrete nouns",
    "hook with bodily pressure",
    "four final variants should diverge",
]


def norm(s: str) -> str:
    s = re.sub(r"\[[^\]]+\]", "", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def body(text: str) -> str:
    # Step artifacts are cumulative: Step 09 begins with the exact bytes of
    # Step 08, which itself contains older "Complete Step Output" headings.
    # Auditing the first match therefore inspects historical work and can report
    # a false missing-critique failure.  The newest (last) delimited block is the
    # Step-09 body that this validator is responsible for.
    matches = list(re.finditer(
        r"## 4\. Complete Step Output(.*?)(?=\n## 5\.|\n## 6\.|\Z)",
        text,
        re.I | re.S,
    ))
    if matches:
        return matches[-1].group(1)
    matches = list(re.finditer(
        r"## 3\. Model Response / Creative Work(.*?)(?=\n## 4\.|\n## 5\.|\Z)",
        text,
        re.I | re.S,
    ))
    return matches[-1].group(1) if matches else text


def step09_segment(text: str) -> str:
    """Return only the newest Step-09 section from a cumulative artifact."""
    # Pair chains are allowed to label the current stage with H1, H2, or H3.
    # Restricting this to H1 silently retained the cumulative ICB for chains
    # that used ``## Pair NN · Step 09`` and made SequenceMatcher compare
    # ~100 KB of shared context instead of the newest refinement body.
    marker = "# ===== END CREATIVE CONTEXT ====="
    cut = text.rfind(marker)
    current = text[cut + len(marker):] if cut >= 0 else text
    matches = list(re.finditer(r"^#{1,3}\s+.*\bstep\s*0?9\b.*$", current, re.I | re.M))
    # The first Step-09 heading owns the current artifact; later same-step
    # headings are internal subsections such as a supplied-panel refinement.
    # The ICB terminator, when present, has already removed historical context.
    if not matches:
        return current
    return current[matches[0].start():]


def main() -> int:
    g = load_gates()
    ap = argparse.ArgumentParser(description="Step 09 cross-pair distinctiveness (repair-trigger gate).")
    ap.add_argument("run_dir", help="run directory containing pair_*_step09_artist_refined.md files")
    ap.add_argument("--max-sim", type=float, default=None,
                    help="max pairwise similarity (default: gates.yaml step09_max_pair_similarity)")
    args = ap.parse_args()
    max_sim = args.max_sim if args.max_sim is not None else float(g.get("step09_max_pair_similarity", 0.62))

    root = Path(args.run_dir)
    files = sorted(p for p in root.glob("pair_*_step09_artist_refined.md") if ".repair_attempt_" not in p.name)
    if len(files) < 2:
        print("FAIL: need at least two Step 09 files")
        return 1

    entries = []
    failures = []
    for p in files:
        segment = step09_segment(p.read_text(errors="replace"))
        # Adversarial reasoning often lives immediately before the renderable
        # Complete Step Output.  Check the current Step-09 segment for that
        # evidence, while comparing only its output body for convergence.
        low = segment.lower()
        repeated = [ph for ph in BOILERPLATE_PHRASES if ph in low]
        if len(repeated) >= 3:
            failures.append(f"{p.name} contains cloned Step 09 boilerplate: {', '.join(repeated[:3])}")
        # Require an actual named adversarial perspective, not generic artist critique.
        if not re.search(r"devil|hyper-skeptic|critic|objection|resolution", low):
            failures.append(f"{p.name} lacks adversarial critique/resolution language")
        entries.append((p.name, norm(body(segment))))

    # An empty extraction is an INSTRUMENT failure, never a similarity score
    # (two empty strings compare as 1.000).
    blank = [name for name, txt in entries if not txt.strip()]
    if blank:
        print("ERROR: extraction returned empty text — validator defect, not a result.")
        for name in blank:
            print(f"- empty extraction: {name}")
        return 2
    print(f"extracted {len(entries)} pair files: " +
          ", ".join(f"{n} ({len(t)} chars)" for n, t in entries))

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            # autojunk=False is MANDATORY — the default silently under-reports
            # similarity on sequences over 200 elements (2026-08-05 ledger).
            r = difflib.SequenceMatcher(None, entries[i][1], entries[j][1], autojunk=False).ratio()
            if r > max_sim:
                failures.append(
                    f"Step 09 similarity too high {entries[i][0]} vs {entries[j][0]}: {r:.3f} (max {max_sim:.2f})"
                )

    if failures:
        print("STEP 09 DISTINCTIVENESS FAIL (repair trigger - not a run kill)")
        for f in failures:
            print("- " + f)
        return 1
    print("STEP 09 DISTINCTIVENESS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
