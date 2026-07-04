"""gates.yaml model + the prose/YAML drift (meta-check) reader.

Numbers are read live from vault/gates.yaml (never restated in app code). Drift
rows come verbatim from `validate_step.py --meta-check` — the same scan the
pipeline uses — so the UI can't disagree with the CLI.
"""
from __future__ import annotations

import re
from pathlib import Path

from .files import FileService
from .validators import ValidatorRunner
from .yamlio import load

_SECTION_RE = re.compile(r"^#\s*-{2,}\s*(.+?)\s*-*$")
_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")
_DRIFT_RE = re.compile(
    r"^WARN:\s+(?P<file>.+?)\s+restates\s+'(?P<label>[^']+)'\s+as\s+'(?P<value>[^']*)'\s+which disagrees"
)


class GatesService:
    def __init__(self, repo_root: Path, files: FileService, runner: ValidatorRunner,
                 gates_file: str = "vault/gates.yaml"):
        self.root = repo_root.resolve()
        self.files = files
        self.runner = runner
        self.gates_file = gates_file

    def read(self) -> dict:
        """Parsed gates.yaml grouped by its own comment sections, plus raw text."""
        view = self.files.read(self.gates_file)
        raw = self.files.read_bytes(self.gates_file)
        parsed, _meta = load(raw)
        parsed = dict(parsed) if parsed else {}
        groups = self._grouped(view.text, parsed)
        return {
            "path": view.path,
            "sha": view.sha,
            "text": view.text,
            "eol": view.eol,
            "writable": view.writable,
            "roundtrip": view.roundtrip,
            "groups": groups,
            "flat": {k: _jsonable(v) for k, v in parsed.items()},
        }

    def _grouped(self, text: str, parsed: dict) -> list[dict]:
        section = "General"
        seen: dict[str, list] = {}
        order: list[str] = []
        for line in text.splitlines():
            mh = _SECTION_RE.match(line.strip())
            if mh:
                section = mh.group(1).strip(" -")
                continue
            if line[:1] not in (" ", "\t", "#", "", "-"):
                mk = _KEY_RE.match(line)
                if mk and mk.group(1) in parsed:
                    if section not in seen:
                        seen[section] = []
                        order.append(section)
                    seen[section].append(
                        {"key": mk.group(1), "value": _jsonable(parsed[mk.group(1)])}
                    )
        return [{"section": s, "entries": seen[s]} for s in order]

    def meta_check(self) -> dict:
        res = self.runner.run_by_id("validate_step", ["--meta-check", "--root", "."])
        rows = []
        for line in (res["stderr"] or "").splitlines():
            m = _DRIFT_RE.match(line.strip())
            if m:
                rows.append({
                    "file": m.group("file").replace("\\", "/"),
                    "label": m.group("label"),
                    "value": m.group("value"),
                })
        summary = ""
        for line in (res["stdout"] or "").splitlines():
            if line.startswith("META-CHECK:"):
                summary = line
        return {
            "summary": summary,
            "rows": rows,
            "count": len(rows),
            "raw_stdout": res["stdout"],
            "raw_stderr": res["stderr"],
            "duration_ms": res["duration_ms"],
        }


def _jsonable(v):
    """Coerce ruamel scalars/containers to plain JSON-serialisable values."""
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    return str(v)
