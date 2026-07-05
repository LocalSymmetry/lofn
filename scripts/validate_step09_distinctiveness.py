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
    m = re.search(r"## 4\. Complete Step Output(.*?)(?=\n## 5\.|\n## 6\.|\Z)", text, re.I | re.S)
    if m:
        return m.group(1)
    m = re.search(r"## 3\. Model Response / Creative Work(.*?)(?=\n## 4\.|\n## 5\.|\Z)", text, re.I | re.S)
    return m.group(1) if m else text


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
        txt = body(p.read_text(errors="replace"))
        low = txt.lower()
        repeated = [ph for ph in BOILERPLATE_PHRASES if ph in low]
        if len(repeated) >= 3:
            failures.append(f"{p.name} contains cloned Step 09 boilerplate: {', '.join(repeated[:3])}")
        # Require an actual named adversarial perspective, not generic artist critique.
        if not re.search(r"devil|hyper-skeptic|critic|objection|resolution", low):
            failures.append(f"{p.name} lacks adversarial critique/resolution language")
        entries.append((p.name, norm(txt)))

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            r = difflib.SequenceMatcher(None, entries[i][1], entries[j][1]).ratio()
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
