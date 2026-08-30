#!/usr/bin/env python3
"""Validate that a Lofn run has a real core/orchestrator packet.

This is intentionally lightweight but catches the common shortcut where a creative
agent writes a single summary called 'orchestrator' instead of running the modern
Lofn-Core + 3-panel orchestrator process.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def require(path: Path, min_bytes: int, needles: list[str]) -> list[str]:
    errs = []
    if not path.exists():
        return [f"missing {path.name}"]
    text = path.read_text(errors="replace")
    if len(text.encode()) < min_bytes:
        errs.append(f"{path.name} too short ({len(text.encode())} bytes < {min_bytes})")
    low = text.lower()
    for n in needles:
        if n.lower() not in low:
            errs.append(f"{path.name} missing marker: {n}")
    return errs


def resolve(root: Path, *names: str) -> Path:
    """Return the first existing alias, else the first name (for the error message).

    The repo carries TWO documented filename dialects for the Phase-1 packet:
    `lofn/SKILL.md` Phase 1 step 6 saves `03_panel_debate.md` / `04_metaprompt.md` /
    `05_pair_assignments.md` / `06_<modality>_handoff.md`, while this validator was
    written against the older `03_orchestrator_*` / `06_audio_handoff.md` names.
    Accepting both is the instrument fix; deforming a correct run's filenames to
    satisfy a stale reader is how a broken gate damages the artifact
    (2026-08-26, run `2026-08-26-opus-masterpiece`).
    """
    for n in names:
        if (root / n).exists():
            return root / n
    return root / names[0]


def main() -> int:
    if len(sys.argv) != 2:
        return fail("usage: validate_orchestrator_packet.py <run_dir>")
    root = Path(sys.argv[1])
    errs: list[str] = []

    errs += require(root / "01_seed_lineage.md", 1500, ["seed", "lineage", "why"])
    errs += require(root / "02_golden_seed.md", 1800, ["golden seed", "lineage", "non-negotiable", "permission"])
    errs += require(resolve(root, "03_orchestrator_panel_debate.md", "03_panel_debate.md"), 5000, [
        "special flairs",
        "concept panel",
        "medium panel",
        "context & marketing panel",
        "synthesis",
    ])
    _panel = resolve(root, "03_orchestrator_panel_debate.md", "03_panel_debate.md")
    panel_text = _panel.read_text(errors="replace").lower() if _panel.exists() else ""
    for panel_name in ["concept panel", "medium panel", "context & marketing panel"]:
        idx = panel_text.find(panel_name)
        if idx >= 0:
            chunk = panel_text[idx: idx + 2000]
            if "devil" not in chunk and "hyper-skeptic" not in chunk and "hyperskeptic" not in chunk:
                errs.append(f"{_panel.name} {panel_name} missing Devil's Advocate / Hyper-Skeptic role")
    errs += require(resolve(root, "04_orchestrator_metaprompt.md", "04_metaprompt.md"), 2500, [
        "golden seed",
        "active personality",
        "panel",
        "pattern",
        "structural completeness",
    ])
    errs += require(resolve(root, "05_orchestrator_pair_assignments.md", "05_pair_assignments.md"), 2500, [
        "pair 01",
        "pair 06",
        "accessible",
        "ambitious",
        "lofn-prime",
        "rationale",
    ])
    errs += require(resolve(root, "06_audio_handoff.md", "06_music_handoff.md"), 1800, [
        "read first",
        "orchestrator",
        "golden seed",
        "pair agents",
        "qa contract",
    ])

    # Optional preferred artifact names from the orchestrator skill.
    if not (root / "core_seed.md").exists():
        print("WARN: core_seed.md not present; accepting 01/02 seed artifacts if otherwise valid")

    if errs:
        print("ORCHESTRATOR PACKET FAIL")
        for e in errs:
            print("- " + e)
        return 1

    print("ORCHESTRATOR PACKET PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
