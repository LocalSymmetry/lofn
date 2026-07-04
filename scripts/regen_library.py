#!/usr/bin/env python3
"""Regenerate the DERIVED personality/panel files from the canonical per-item YAMLs.

Ground truth (docs/explorer-ground-truth.md §1): `skills/orchestration/<coll>/*.yaml`
are the source of truth; the monolith `<coll>.yaml` and `<coll>_index.{json,md}` are
derived and had NO regenerator. This is it. Stdlib-only so it runs under any Python.

What it does, per collection (personalities | panels):
  * monolith  — faithful BYTE concatenation of the canonical per-item files (each
                file is already a 1-element `- name:` list entry), sorted by slug.
                Maximally byte-preserving; even raw-mode files pass through verbatim.
  * index_json — membership sync: keep existing rows (PRESERVING curated fields like
                identity/vibes/modality/flairs/members), add minimal rows for new
                files, drop rows whose file vanished, renumber `id`. Output format
                (compact vs pretty) is detected from the existing file to limit churn.
  * index_md  — regenerated from the synced rows.

`-1.yaml` variant twins are excluded (orphaned, un-indexed). Nothing is deleted from
the canonical dir; this only writes the 3 derived files.

Usage:
  regen_library.py personalities|panels|all [--root DIR] [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MARKERS = ["SOUL.md", "vault/gates.yaml", "skills/orchestration"]
_NAME_RE = re.compile(r"^\s*-?\s*name:\s*(.+?)\s*$", re.MULTILINE)

SPECS = {
    "personalities": {
        "dir": "skills/orchestration/personalities",
        "monolith": "skills/orchestration/personalities.yaml",
        "index_json": "skills/orchestration/personalities_index.json",
        "index_md": "skills/orchestration/personalities_index.md",
        "md_columns": ["#", "Name", "Identity", "Vibes", "File"],
        "md_fields": [None, "name", "identity", "vibes", "file"],
    },
    "panels": {
        "dir": "skills/orchestration/panels",
        "monolith": "skills/orchestration/panels.yaml",
        "index_json": "skills/orchestration/panels_index.json",
        "index_md": "skills/orchestration/panels_index.md",
        "md_columns": ["#", "Name", "Modality", "Key Flairs", "Notable Members", "File"],
        "md_fields": [None, "name", "modality", "flairs", "members", "file"],
    },
}


def find_root(start: Path) -> Path:
    for d in [start, *start.parents]:
        if all((d / m).exists() for m in MARKERS):
            return d
    raise SystemExit(f"repo root not found from {start}")


def strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def canonical_files(root: Path, spec: dict) -> list[Path]:
    d = root / spec["dir"]
    return sorted(p for p in d.glob("*.yaml") if not p.stem.endswith("-1"))


def name_of(p: Path) -> str:
    m = _NAME_RE.search(p.read_text(encoding="utf-8", errors="replace"))
    return strip_quotes(m.group(1)) if m else p.stem


def build_monolith(files: list[Path]) -> bytes:
    """Concatenate per-item file bytes, guaranteeing a newline between entries."""
    chunks: list[bytes] = []
    for p in files:
        raw = p.read_bytes()
        if not raw.endswith(b"\n"):
            raw += b"\r\n" if b"\r\n" in raw else b"\n"
        chunks.append(raw)
    return b"".join(chunks)


def sync_index(files: list[Path], spec: dict, existing_text: str | None) -> tuple[list[dict], bool]:
    existing = []
    pretty = True
    if existing_text:
        try:
            existing = json.loads(existing_text)
            pretty = existing_text.lstrip().startswith("[\n") or "\n  " in existing_text
        except Exception:
            existing = []
    by_slug = {}
    for row in existing:
        f = row.get("file") or ""
        slug = Path(f).stem if f else str(row.get("slug", ""))
        if slug:
            by_slug[slug] = row
    coll_dir = Path(spec["dir"]).name
    rows = []
    for i, p in enumerate(files, 1):
        slug = p.stem
        row = dict(by_slug.get(slug, {}))
        row["id"] = i
        row["name"] = row.get("name") or name_of(p)
        row["slug"] = slug
        row["file"] = f"{coll_dir}/{slug}.yaml"
        rows.append(row)
    return rows, pretty


def render_index_md(rows: list[dict], spec: dict) -> str:
    cols = spec["md_columns"]
    fields = spec["md_fields"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for i, row in enumerate(rows, 1):
        cells = []
        for col, field in zip(cols, fields):
            if field is None:
                cells.append(str(i))
            elif field == "file":
                cells.append(f"`{row.get('file','')}`")
            else:
                v = row.get(field, "")
                if isinstance(v, list):
                    sep = ", " if field == "members" else "; "
                    v = sep.join(str(x) for x in v)
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def regen_collection(root: Path, coll: str, check: bool) -> dict:
    spec = SPECS[coll]
    files = canonical_files(root, spec)
    changed = []

    def write_if_changed(rel: str, data: bytes):
        p = root / rel
        old = p.read_bytes() if p.exists() else None
        if old == data:
            return
        changed.append(rel)
        if not check:
            tmp = p.with_suffix(p.suffix + ".tmp~")
            tmp.write_bytes(data)
            import os
            os.replace(tmp, p)

    monolith = build_monolith(files)
    write_if_changed(spec["monolith"], monolith)

    idx_path = root / spec["index_json"]
    existing_text = idx_path.read_text(encoding="utf-8") if idx_path.exists() else None
    rows, pretty = sync_index(files, spec, existing_text)
    if pretty:
        idx_json = json.dumps(rows, ensure_ascii=True, indent=2) + "\n"
    else:
        idx_json = json.dumps(rows, ensure_ascii=True, separators=(",", ": ")) + "\n"
    write_if_changed(spec["index_json"], idx_json.encode("utf-8"))
    write_if_changed(spec["index_md"], render_index_md(rows, spec).encode("utf-8"))

    return {"collection": coll, "canonical_items": len(files), "changed": changed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("collection", choices=["personalities", "panels", "all"])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="report changes without writing")
    args = ap.parse_args()

    root = (args.root or find_root(Path(__file__).resolve().parent)).resolve()
    colls = ["personalities", "panels"] if args.collection == "all" else [args.collection]
    results = [regen_collection(root, c, args.check) for c in colls]
    for r in results:
        verb = "would change" if args.check else "wrote"
        if r["changed"]:
            print(f"{r['collection']}: {r['canonical_items']} items; {verb}: {', '.join(r['changed'])}")
        else:
            print(f"{r['collection']}: {r['canonical_items']} items; up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
