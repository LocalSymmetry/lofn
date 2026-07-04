"""Supplemental libraries: collections, search, card projections, sync, regen.

Editing principle (plan §3.5): the RAW file is always the write surface (byte-safe
via FileService). Cards, list splits, and CSV tables are read-only PROJECTIONS
built here for display/search. The only structured writes in the app are
gates.yaml table-mode and library regen.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from .files import FileService
from .manifest import Manifest
from .yamlio import load

_NAME_RE = re.compile(r"^\s*-?\s*name:\s*(.+?)\s*$", re.MULTILINE)
_H_RE = re.compile(r"^\s{0,6}#{1,4}\s+(.*\S)\s*$", re.MULTILINE)
_SECTION_HDR_RE = re.compile(r"^\s{0,6}#{1,4}\s+(.*\S)\s*$")
_NUM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)\s*$")


class LibraryService:
    def __init__(self, repo_root: Path, manifest: Manifest, files: FileService):
        self.root = repo_root.resolve()
        self.manifest = manifest
        self.files = files

    # -- collection listing ----------------------------------------------
    def collections(self) -> list[dict]:
        out = []
        for lib in self.manifest.libraries:
            out.append({
                "id": lib["id"],
                "title": lib.get("title", lib["id"]),
                "kind": lib["kind"],
                "card": lib.get("card"),
            })
        return out

    def _lib(self, lib_id: str) -> dict:
        lib = self.manifest.library(lib_id)
        if not lib:
            raise KeyError(lib_id)
        return lib

    def list_items(self, lib_id: str, q: str | None = None) -> dict:
        lib = self._lib(lib_id)
        kind = lib["kind"]
        if kind == "yaml-collection":
            return self._list_yaml_items(lib, q)
        if kind in ("line-list", "comma-list"):
            return self._list_terms(lib, q)
        if kind == "csv":
            return self._list_csv(lib, q)
        raise ValueError(f"unknown library kind: {kind}")

    def _list_yaml_items(self, lib: dict, q: str | None) -> dict:
        d = self.root / lib["dir"]
        suffix = lib.get("variant_suffix")
        items = []
        variants = 0
        ql = q.lower() if q else None
        for p in sorted(d.glob(lib.get("glob", "*.yaml"))):
            stem = p.stem
            is_variant = bool(suffix) and stem.endswith(suffix)
            text = p.read_text(encoding="utf-8", errors="replace")
            m = _NAME_RE.search(text)
            name = _strip_quotes(m.group(1)) if m else stem
            if ql and ql not in name.lower() and ql not in text.lower():
                continue
            if is_variant:
                variants += 1
            items.append({
                "name": name,
                "file": self.files.rel_of(p),
                "slug": stem,
                "is_variant": is_variant,
            })
        canonical = [i for i in items if not i["is_variant"]]
        return {
            "id": lib["id"],
            "kind": "yaml-collection",
            "title": lib.get("title", lib["id"]),
            "count": len(canonical),
            "variant_count": variants,
            "items": items,
        }

    def _list_terms(self, lib: dict, q: str | None) -> dict:
        view = self.files.read(lib["path"])
        if lib["kind"] == "comma-list":
            terms = [t.strip() for t in view.text.split(",")]
        else:
            terms = [t for t in view.text.split("\n")]
        terms = [t for t in terms if t.strip()]
        if q:
            ql = q.lower()
            terms = [t for t in terms if ql in t.lower()]
        return {
            "id": lib["id"],
            "kind": lib["kind"],
            "title": lib.get("title", lib["id"]),
            "path": view.path,
            "count": len(terms),
            "items": terms,
            "sha": view.sha,
            "writable": view.writable,
        }

    def _list_csv(self, lib: dict, q: str | None) -> dict:
        view = self.files.read(lib["path"])
        rows = list(csv.reader(io.StringIO(view.text)))
        header, data = (rows[0], rows[1:]) if rows else ([], [])
        # music_frames.csv has a leading blank line -> header may be empty; find it.
        if header == [] or all(not c for c in header):
            for i, r in enumerate(rows):
                if any(c.strip() for c in r):
                    header, data = r, rows[i + 1:]
                    break
        if q:
            ql = q.lower()
            data = [r for r in data if any(ql in (c or "").lower() for c in r)]
        return {
            "id": lib["id"],
            "kind": "csv",
            "title": lib.get("title", lib["id"]),
            "path": view.path,
            "columns": lib.get("columns") or header,
            "count": len(data),
            "rows": data[:5000],
            "truncated": len(data) > 5000,
            "sha": view.sha,
            "writable": view.writable,
        }

    # -- single item ------------------------------------------------------
    def get_item(self, lib_id: str, file_rel: str) -> dict:
        lib = self._lib(lib_id)
        view = self.files.read(file_rel)
        card = None
        if lib["kind"] == "yaml-collection" and view.roundtrip:
            try:
                data, _ = load(self.files.read_bytes(file_rel))
                item = data[0] if isinstance(data, list) and data else data
                name = item.get("name")
                prompt = item.get("prompt", "")
                card = self._card(lib, str(name), str(prompt))
            except Exception:
                card = None
        return {"view": view.__dict__, "card": card}

    def put_item(self, lib_id: str, file_rel: str, text: str, base_sha: str | None) -> dict:
        self._lib(lib_id)  # validate id
        view = self.files.write_text(file_rel, text, base_sha=base_sha)
        return view.__dict__

    def _card(self, lib: dict, name: str, prompt: str) -> dict:
        if lib.get("card") == "panel":
            return {"name": name, "type": "panel", "sections": _panel_sections(prompt)}
        # personality (default): heading outline + preview
        headings = [h.strip() for h in _H_RE.findall(prompt)][:40]
        return {
            "name": name,
            "type": "personality",
            "headings": headings,
            "preview": prompt.strip()[:400],
        }

    # -- sync / drift -----------------------------------------------------
    def sync_status(self, lib_id: str) -> dict:
        lib = self._lib(lib_id)
        if lib["kind"] != "yaml-collection":
            return {"id": lib_id, "applicable": False}
        d = self.root / lib["dir"]
        suffix = lib.get("variant_suffix")
        canonical = {
            p.stem for p in d.glob(lib.get("glob", "*.yaml"))
            if not (suffix and p.stem.endswith(suffix))
        }
        derived = lib.get("derived", {})
        index_slugs = self._index_slugs(derived.get("index_json"))
        monolith_names = self._monolith_names(derived.get("monolith"))
        return {
            "id": lib_id,
            "applicable": True,
            "canonical_count": len(canonical),
            "index_count": len(index_slugs) if index_slugs is not None else None,
            "monolith_count": len(monolith_names) if monolith_names is not None else None,
            "missing_from_index": sorted(canonical - index_slugs) if index_slugs is not None else [],
            "stale_in_index": sorted(index_slugs - canonical) if index_slugs is not None else [],
            "regen": lib.get("regen"),
        }

    def _index_slugs(self, index_json_rel: str | None) -> set | None:
        if not index_json_rel:
            return None
        p = self.root / index_json_rel
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            slugs = set()
            for row in data:
                f = row.get("file") or ""
                slugs.add(Path(f).stem if f else str(row.get("slug", "")))
            return {s for s in slugs if s}
        except Exception:
            return None

    def _monolith_names(self, monolith_rel: str | None) -> set | None:
        if not monolith_rel:
            return None
        p = self.root / monolith_rel
        if not p.exists():
            return None
        try:
            return {_strip_quotes(m).strip() for m in _NAME_RE.findall(p.read_text(encoding="utf-8", errors="replace"))}
        except Exception:
            return None


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _panel_sections(prompt: str) -> list[dict]:
    """Split a panel prompt into its `##` sections with numbered members."""
    sections: list[dict] = []
    current = None
    for line in prompt.splitlines():
        mh = _SECTION_HDR_RE.match(line)
        if mh:
            current = {"title": mh.group(1).strip().rstrip(":"), "items": []}
            sections.append(current)
            continue
        if current is not None:
            mi = _NUM_RE.match(line)
            if mi:
                current["items"].append(mi.group(1).strip())
    return sections
