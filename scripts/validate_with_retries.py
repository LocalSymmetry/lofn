#!/usr/bin/env python3
"""Validate a Lofn step artifact with retry guidance.

This script does not call an LLM. It gives agents/controllers a deterministic
3-attempt loop contract:
  1. validate current artifact
  2. if invalid, write a repair prompt/checklist file
  3. agent repairs the artifact
  4. rerun until pass or attempts exhausted

Usage:
  python3 scripts/validate_with_retries.py <step> <file> [--attempt N] [--max-attempts 3]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("file")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    path = Path(args.file)
    name = path.name
    canonical = (
        name.startswith("step")
        or (name.startswith("pair_") and "_step" in name)
    )
    if not canonical:
        print(f"VALIDATION SKIP: {path} is a pre-pipeline/research/orchestration artifact, not a canonical Lofn step file")
        return 0

    validator = Path(__file__).with_name("validate_step.py")
    proc = subprocess.run(
        [sys.executable, str(validator), str(args.step), str(path)],
        text=True,
        capture_output=True,
    )

    if proc.returncode == 0:
        print(proc.stdout.strip())
        print(f"VALIDATION PASS on attempt {args.attempt}/{args.max_attempts}")
        return 0

    err = (proc.stdout + proc.stderr).strip()
    print(err)
    print(f"VALIDATION FAIL on attempt {args.attempt}/{args.max_attempts}")

    repair_path = path.with_suffix(path.suffix + f".repair_attempt_{args.attempt}.md")
    repair_path.write_text(
        "# Lofn Step Repair Required\n\n"
        f"Artifact: `{path}`\n"
        f"Step: `{str(args.step).zfill(2)}`\n"
        f"Attempt: {args.attempt}/{args.max_attempts}\n\n"
        "## Validator failure\n\n"
        "```\n" + err + "\n```\n\n"
        "## Repair instructions\n\n"
        "Revise the artifact in place. Preserve valid creative content. Fix only the failing structural/content requirements. "
        "Then rerun validate_with_retries.py with the next attempt number.\n\n"
        "If this is Step 08 or Step 10 music, verify: standalone `## 1. MUSIC PROMPT`, `## 2. LYRICS`, "
        "bracketed `[SONG FORM: ...]`, full `[Section - EMO:... - Voice - Cue]` headers, SFX cue, non-lexical hook, "
        "and no bare `[EMO:...]`, `EMO HEADER:`, or plain `SONG FORM:` lines.\n",
        encoding="utf-8",
    )
    print(f"Wrote repair prompt: {repair_path}")

    if args.attempt >= args.max_attempts:
        print("MAX_ATTEMPTS_EXHAUSTED: stop, checkpoint, and escalate to controller/QA.")
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
