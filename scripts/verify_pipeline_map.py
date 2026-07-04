#!/usr/bin/env python3
"""Verify tools/explorer/pipeline_map.yaml against the real repo tree.

The Lofn Prompt Explorer renders entirely from the manifest, so the manifest must
never silently disagree with disk (design plan gate 6: "manifest honesty"). The
backend runs this at startup and shows a banner on failure; CI runs it too.

Checks, both directions:
  FORWARD  — every path the manifest names (nodes, editions, variants, routers,
             templates, gate validators, validators, gates_file, libraries +
             their derived files, reference dirs, execution layer) EXISTS.
  REVERSE  — every skills/<rail>/steps/*.md and skills/orchestration/steps/*.md
             on disk is represented as a manifest node (catches NEW step files).

Exit 0 = clean; 1 = mismatch (details printed). FAIL-LOUD, never fail-open: a
rotted map is a bug to surface, not to paper over.

Usage:
  verify_pipeline_map.py [--root DIR] [--map PATH] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_yaml(path: Path):
    """Load YAML with whichever parser is available (ruamel preferred)."""
    text = path.read_text(encoding="utf-8")
    try:
        from ruamel.yaml import YAML

        return YAML(typ="safe").load(text)
    except Exception:
        pass
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception as exc:  # pragma: no cover
        print(
            "ERROR: need ruamel.yaml or PyYAML to parse the manifest "
            f"({type(exc).__name__}: {exc}). Run via the explorer venv.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def find_repo_root(start: Path, markers: list[str]) -> Path | None:
    for d in [start, *start.parents]:
        if all((d / m).exists() for m in markers):
            return d
    return None


def collect_declared_paths(m: dict) -> list[tuple[str, str]]:
    """Return (path, context) pairs the manifest asserts must exist."""
    out: list[tuple[str, str]] = []

    def add(p, ctx):
        if p:
            out.append((p, ctx))

    for rail in m.get("rails", []):
        rid = rail.get("id", "?")
        for key, val in (rail.get("router") or {}).items():
            add(val, f"rail {rid} router.{key}")
        for node in rail.get("nodes", []):
            nid = node.get("id", "?")
            add(node.get("path"), f"node {nid}")
            add(node.get("edition_compressed"), f"node {nid} edition_compressed")
            for var in node.get("variants", []) or []:
                add(var.get("path"), f"node {nid} variant {var.get('id','?')}")

    for g in m.get("gates", []):
        add(g.get("validator"), f"gate {g.get('id','?')} validator")

    for v in m.get("validators", []):
        add(v.get("path"), f"validator {v.get('id','?')}")

    add(m.get("gates_file"), "gates_file")

    for lib in m.get("libraries", []):
        lid = lib.get("id", "?")
        add(lib.get("dir"), f"library {lid} dir")
        add(lib.get("path"), f"library {lid} path")
        for key, val in (lib.get("derived") or {}).items():
            add(val, f"library {lid} derived.{key}")

    for ref in m.get("reference_dirs", []):
        add(ref.get("dir"), f"reference_dir {ref.get('id','?')}")

    for ex in m.get("execution_layer", []):
        add(ex.get("skill"), f"execution {ex.get('id','?')} skill")
        add(ex.get("execution"), f"execution {ex.get('id','?')} execution")

    return out


def reverse_check(root: Path, m: dict) -> list[str]:
    """Step files on disk that no manifest node points at."""
    declared = set()
    for rail in m.get("rails", []):
        for node in rail.get("nodes", []):
            if node.get("path"):
                declared.add((root / node["path"]).resolve())
            for var in node.get("variants", []) or []:
                if var.get("path"):
                    declared.add((root / var["path"]).resolve())

    disk: list[Path] = []
    for pat in (
        "skills/orchestration/steps/*.md",
        "skills/music/steps/*.md",
        "skills/image/steps/*.md",
        "skills/video/steps/*.md",
        "skills/story/steps/*.md",
    ):
        disk.extend(root.glob(pat))

    return [
        str(p.relative_to(root)).replace("\\", "/")
        for p in sorted(disk)
        if p.resolve() not in declared
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--map", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    default_map = here.parent / "tools" / "explorer" / "pipeline_map.yaml"
    map_path = (args.map or default_map).resolve()
    if not map_path.exists():
        print(f"ERROR: manifest not found: {map_path}", file=sys.stderr)
        return 2

    m = _load_yaml(map_path)
    markers = m.get("repo_root_markers", ["SOUL.md", "vault/gates.yaml"])
    root = (args.root or find_repo_root(map_path.parent, markers) or here.parent).resolve()

    missing = [
        (p, ctx)
        for p, ctx in collect_declared_paths(m)
        if not (root / p).exists()
    ]
    orphans = reverse_check(root, m)

    result = {
        "root": str(root),
        "map": str(map_path),
        "missing": [{"path": p, "context": ctx} for p, ctx in missing],
        "orphan_step_files": orphans,
        "ok": not missing and not orphans,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"pipeline_map.yaml verify — root={root}")
        if missing:
            print(f"\n{len(missing)} DECLARED PATH(S) MISSING ON DISK:")
            for p, ctx in missing:
                print(f"  - {p}   ({ctx})")
        if orphans:
            print(f"\n{len(orphans)} STEP FILE(S) ON DISK NOT IN THE MANIFEST:")
            for p in orphans:
                print(f"  - {p}")
        if result["ok"]:
            print("OK — manifest and tree agree.")
        else:
            print("\nMISMATCH — manifest and tree disagree (see above).")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
