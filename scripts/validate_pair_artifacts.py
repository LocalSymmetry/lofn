#!/usr/bin/env python3
"""Validate one Lofn pair-agent's canonical Steps 06-10 artifacts.

This is a deterministic controller/agent gate. It does not repair creative content by
itself; it runs the existing per-step validator and writes repair prompts through
validate_with_retries.py. Agents must repair in place and rerun until pass or 3
attempts are exhausted.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STEP_SUFFIX = {
    "06": "facets",
    "07": "song_guides",
    "08": "generation",
    "09": "artist_refined",
    "10": "revision_synthesis",
}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir", help="Audio output directory")
    ap.add_argument("pair", help="Pair number, e.g. 01")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    pair = str(args.pair).zfill(2)
    validator = Path(__file__).with_name("validate_with_retries.py")
    failures: list[str] = []

    if not audio_dir.is_dir():
        print(f"FAIL: not a directory: {audio_dir}")
        return 2

    print(f"PAIR VALIDATION START pair_{pair} in {audio_dir}")
    for step, suffix in STEP_SUFFIX.items():
        path = audio_dir / f"pair_{pair}_step{step}_{suffix}.md"
        if not path.exists():
            msg = f"missing pair_{pair}_step{step}_{suffix}.md"
            print(f"FAIL: {msg}")
            failures.append(msg)
            continue
        proc = run([
            sys.executable,
            str(validator),
            step,
            str(path),
            "--attempt",
            str(args.attempt),
            "--max-attempts",
            str(args.max_attempts),
        ])
        out = (proc.stdout + proc.stderr).strip()
        if out:
            print(out)
        if proc.returncode != 0:
            failures.append(f"step {step} failed validation: {path.name}")

    if failures:
        print("\nPAIR VALIDATION FAIL")
        for f in failures:
            print("- " + f)
        if args.attempt >= args.max_attempts:
            print("MAX_ATTEMPTS_EXHAUSTED: stop pair, write pair_VALIDATION_BLOCKED, escalate.")
            return 2
        print("Repair the failing artifacts in place, then rerun with --attempt", args.attempt + 1)
        return 1

    print("\nPAIR VALIDATION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
