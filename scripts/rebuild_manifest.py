#!/usr/bin/env python3
"""Rebuild the Lofn RUN_STATE manifest by stat-ing an output/<run>/ directory.

Upgrade-plan item 1.1 (optional helper). DISK IS AUTHORITY: this prints a
manifest that is a *rebuildable cache derived by stat-ing files*, never a
hand-asserted second truth. The coordinator may run this to re-derive run state
on resume; if the printed manifest and disk ever disagree, disk wins (re-run me).

It computes, per artifact found on disk:
  {step, pair, canonical_path, exists, byte_size, sha, status}
plus the ICB sha when a CREATIVE_CONTEXT.md / ICB file is present, and a terse
summary footer (artifacts found, total bytes).

FAIL-OPEN: a missing/odd directory prints an empty-but-valid manifest and exits
0 with a warning. This helper never hard-fails a run; it only observes.

Usage:
  rebuild_manifest.py <output_run_dir>            # print manifest JSON to stdout
  rebuild_manifest.py <output_run_dir> --out PATH # also write it to PATH
  rebuild_manifest.py --help
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


_STEP_RE = re.compile(r"step[_-]?(\d{2})", re.I)
_PAIR_RE = re.compile(r"pair[_-]?(\d+)", re.I)
_ICB_NAMES = {"creative_context.md", "icb.md", "creative_context_block.md"}


def _sha(p: Path) -> str:
    try:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _classify(p: Path):
    step_m = _STEP_RE.search(p.name)
    pair_m = _PAIR_RE.search(p.name)
    return (step_m.group(1) if step_m else None,
            pair_m.group(1).zfill(2) if pair_m else None)


def build_manifest(run_dir: Path) -> dict:
    """Stat-walk run_dir and return the RUN_STATE manifest dict. Never raises."""
    artifacts: list[dict] = []
    icb_sha = None
    total_bytes = 0
    try:
        if run_dir.exists() and run_dir.is_dir():
            for p in sorted(run_dir.rglob("*")):
                if not p.is_file():
                    continue
                try:
                    size = p.stat().st_size
                except Exception:
                    size = -1
                total_bytes += max(size, 0)
                step, pair = _classify(p)
                rec = {
                    "step": step,
                    "pair": pair,
                    "canonical_path": str(p.relative_to(run_dir)).replace("\\", "/"),
                    "exists": True,
                    "byte_size": size,
                    "sha": _sha(p),
                    # status is disk-observable only: a non-trivial file is "done",
                    # a stub is "pending". Gate verdict / quarantine are coordinator-owned.
                    "status": "done" if size >= 200 else "pending",
                }
                artifacts.append(rec)
                if p.name.lower() in _ICB_NAMES:
                    icb_sha = rec["sha"]
        else:
            print(f"WARN: run dir not found or not a directory: {run_dir}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 — fail open; emit what we have
        print(f"WARN: manifest rebuild hit a snag ({exc}); manifest is partial", file=sys.stderr)
    return {
        "run_dir": str(run_dir).replace("\\", "/"),
        "icb_sha": icb_sha,
        "artifact_count": len(artifacts),
        "total_bytes": total_bytes,
        "artifacts": artifacts,
        "_note": "Derived by stat-ing disk. Disk is authority; re-run on disagreement.",
    }


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    out_path = None
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 < len(argv):
            out_path = Path(argv[i + 1])
        argv = [a for j, a in enumerate(argv) if j not in (i, i + 1)]
    if not argv:
        print("usage: rebuild_manifest.py <output_run_dir> [--out PATH]", file=sys.stderr)
        return 0  # fail-open: bad invocation still exits 0
    manifest = build_manifest(Path(argv[0]))
    blob = json.dumps(manifest, indent=2)
    print(blob)
    if out_path is not None:
        try:
            out_path.write_text(blob, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: could not write manifest to {out_path} ({exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
