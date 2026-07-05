#!/usr/bin/env python3
"""Validate Step 06 facets are pair-specific and complete (cross-pair distinctiveness).

The earliest point cross-pair convergence shows up: if the 6 pairs' facet sets
collapse toward one template at step 06, five more steps compound the clone
before the portfolio-level check (step 10) ever runs. Sibling of
``validate_step09_distinctiveness.py`` (step 09) and
``validate_portfolio_distinctiveness.py`` (step 10); together they honor
EXECUTION.md §4: "Distinctiveness across pairs is checked at 06, 09, and 10."

Thresholds are single-sourced from ``vault/gates.yaml`` (FAIL-OPEN: a missing or
unparseable gates.yaml falls back to the built-in defaults that mirror it, and
the ``load_gates`` import failing never blocks a run). A breach is a REPAIR
TRIGGER — it routes into the bounded max-3-attempt repair loop, then quarantine +
human surface; it is NOT a run kill, and a DELIBERATE shared motif that survives
repair is a human waive, never a silent ship.

Usage:
  validate_step06_distinctiveness.py <run_dir> [--max-sim F] [--min-facets N]
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

try:
    # scripts/ is sys.path[0] when this file is run directly, so the sibling
    # single-source loader imports cleanly. Fail-open if it ever doesn't.
    from validate_step import load_gates
except Exception:  # noqa: BLE001 — a broken import must never block a run
    def load_gates(path=None):  # type: ignore
        return {}


# Generic step-06 phrasing that, repeated across pairs, signals template padding
# rather than pair-specific facets.
GENERIC = [
    "preserve concept",
    "preserve medium",
    "make bodily evidence audible",
    "avoid generic proof metaphors",
    "maintain singable hook",
    "signature rupture differs from all other pairs",
    "hook must name what abstraction cannot touch",
]


def body(text: str) -> str:
    m = re.search(r"## 4\. Complete Step Output(.*?)(?=\n## 5\.|\n## 6\.|\Z)", text, re.I | re.S)
    if m:
        return m.group(1)
    m = re.search(r"## 3\. Model Response / Creative Work(.*?)(?=\n## 4\.|\n## 5\.|\Z)", text, re.I | re.S)
    return m.group(1) if m else text


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9' ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    g = load_gates()
    ap = argparse.ArgumentParser(description="Step 06 cross-pair distinctiveness (repair-trigger gate).")
    ap.add_argument("run_dir", help="run directory containing pair_*_step06_facets.md files")
    ap.add_argument("--max-sim", type=float, default=None,
                    help="max pairwise facet similarity (default: gates.yaml step06_max_pair_similarity)")
    ap.add_argument("--min-facets", type=int, default=None,
                    help="min substantive facets per pair (default: gates.yaml step06_min_facets)")
    args = ap.parse_args()
    max_sim = args.max_sim if args.max_sim is not None else float(g.get("step06_max_pair_similarity", 0.50))
    min_facets = args.min_facets if args.min_facets is not None else int(g.get("step06_min_facets", 8))

    root = Path(args.run_dir)
    files = sorted(p for p in root.glob("pair_*_step06_facets.md") if ".repair_attempt_" not in p.name)
    if len(files) < 2:
        print("FAIL: need at least two Step 06 files")
        return 1

    failures = []
    entries = []
    for p in files:
        txt = body(p.read_text(errors="replace"))
        low = txt.lower()
        # Count substantive facets: markdown headings / numbered facets OR JSON-style objects.
        facet_hits = len(re.findall(r"(?im)^\s*(?:[-*]\s*)?(?:facet\s+\d+|facet\s+[a-z]|\d+\.|#{3,}\s*)", txt))
        json_name_hits = len(re.findall(r'"name"\s*:', txt, re.I))
        facet_hits = max(facet_hits, json_name_hits)
        if facet_hits < min_facets:
            failures.append(f"{p.name} has too few substantive facets ({facet_hits}; need >={min_facets})")
        if "failure mode" not in low and "failure_mode" not in low and "fails if" not in low:
            failures.append(f"{p.name} lacks failure modes for facets")
        if "why it matters" not in low and "why_it_matters" not in low and '"why"' not in low and "purpose" not in low:
            failures.append(f"{p.name} lacks why-it-matters/purpose explanations")
        repeated = [phrase for phrase in GENERIC if phrase in low]
        if len(repeated) >= 3:
            failures.append(f"{p.name} relies on generic Step 06 boilerplate: {', '.join(repeated[:3])}")
        entries.append((p.name, norm(txt)))

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            r = difflib.SequenceMatcher(None, entries[i][1], entries[j][1]).ratio()
            if r > max_sim:
                failures.append(
                    f"Step 06 similarity too high {entries[i][0]} vs {entries[j][0]}: {r:.3f} (max {max_sim:.2f})"
                )

    if failures:
        print("STEP 06 DISTINCTIVENESS FAIL (repair trigger - not a run kill)")
        for f in failures:
            print("- " + f)
        return 1
    print("STEP 06 DISTINCTIVENESS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
