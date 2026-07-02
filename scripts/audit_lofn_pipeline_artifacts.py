#!/usr/bin/env python3
"""Audit Lofn pipeline artifact granularity.

This checks whether a run directory preserves the original Lofn execution shape:
- coordinator steps 00-05 as separate artifacts
- pair-agent steps 06-10 as separate artifacts per pair

It intentionally flags collapsed files like pair_01_steps_06_10.md because those can
hide a single-prompt shortcut instead of one prompt/response per step.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

COORDINATOR_STEPS = {
    "00": "step00_aesthetics_and_genres.md",
    "01": "step01_essence_and_facets.md",
    "02": "step02_concepts.md",
    "03": "step03_artist_and_critique.md",
    "04": "step04_medium.md",
    "05": "step05_refine_medium.md",
}
PAIR_STEPS = [f"{i:02d}" for i in range(6, 11)]


def has_step_file(root: Path, step: str, pair: int | None = None) -> list[Path]:
    files = [p for p in root.glob("*.md") if p.is_file() and ".repair_attempt_" not in p.name]
    matches: list[Path] = []
    if pair is None:
        patterns = [
            re.compile(rf"^(step)?{step}[_-].*\.md$", re.I),
            re.compile(rf"^.*[_-]step{step}[_-].*\.md$", re.I),
        ]
    else:
        p2 = f"{pair:02d}"
        patterns = [
            re.compile(rf"^pair[_-]?{p2}[_-]step{step}([_-].*)?\.md$", re.I),
            re.compile(rf"^step{step}[_-]pair[_-]?{p2}([_-].*)?\.md$", re.I),
            re.compile(rf"^pair[_-]?{p2}[_-]{step}([_-].*)?\.md$", re.I),
        ]
    for path in files:
        if any(pat.match(path.name) for pat in patterns):
            matches.append(path)
    return matches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Run directory or audio subdirectory")
    ap.add_argument("--pairs", type=int, default=6)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    audio_dir = run_dir / "audio" if (run_dir / "audio").is_dir() else run_dir
    if not audio_dir.is_dir():
        print(f"FAIL: not a directory: {audio_dir}")
        return 2

    failures: list[str] = []
    warnings: list[str] = []

    print(f"Auditing Lofn artifacts in: {audio_dir}")

    for step, filename in COORDINATOR_STEPS.items():
        expected = audio_dir / filename
        if expected.exists():
            print(f"PASS coordinator step {step}: {expected.name}")
        else:
            loose = has_step_file(audio_dir, step)
            if loose:
                warnings.append(
                    f"non-canonical coordinator step {step} artifact(s): "
                    + ", ".join(p.name for p in loose)
                    + f"; expected {filename}"
                )
            failures.append(f"missing canonical coordinator step {step} artifact: {filename}")

    if not (audio_dir / "concept_medium_pairs.json").exists():
        failures.append("missing concept_medium_pairs.json")
    else:
        print("PASS concept_medium_pairs.json")

    collapsed = sorted(audio_dir.glob("pair_*_steps_06_10.md"))
    if collapsed:
        warnings.append(
            "collapsed pair files present: " + ", ".join(p.name for p in collapsed)
        )

    for pair in range(1, args.pairs + 1):
        for step in PAIR_STEPS:
            matches = has_step_file(audio_dir, step, pair=pair)
            if matches:
                print(
                    f"PASS pair {pair:02d} step {step}: "
                    + ", ".join(p.name for p in matches)
                )
            else:
                failures.append(f"missing separate pair {pair:02d} step {step} artifact")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"- {w}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"- {f}")
        return 1

    print("\nPASS: artifact granularity matches original Lofn step-by-step pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
