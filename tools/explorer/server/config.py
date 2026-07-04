"""Settings, repo-root discovery, and the write-path jail.

The server owns exactly one machine and one repo. Everything it can write is
constrained by the manifest's `write_allowlist`; this module is where that jail
is enforced (design plan gate 2). No app-side state beyond these values.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MARKERS = ["SOUL.md", "vault/gates.yaml", "skills/orchestration"]


def find_repo_root(start: Path, markers: list[str] = DEFAULT_MARKERS) -> Path:
    """Walk up from `start` until a directory contains all `markers`."""
    start = start.resolve()
    for d in [start, *start.parents]:
        if all((d / m).exists() for m in markers):
            return d
    raise RuntimeError(
        f"repo root not found from {start} (markers: {markers})"
    )


def path_is_write_allowed(rel_posix: str, patterns: list[str]) -> bool:
    """Is a repo-relative POSIX path writable per the allowlist?

    Supports trailing `/**` prefix globs (`skills/**`), fnmatch globs, and exact
    paths (`tools/explorer/pipeline_map.yaml`). Anything else is read-only.
    """
    rel_posix = rel_posix.replace("\\", "/").lstrip("/")
    if ".." in Path(rel_posix).parts:
        return False
    for pat in patterns:
        pat = pat.replace("\\", "/")
        if pat.endswith("/**"):
            prefix = pat[:-3]
            if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
                return True
        elif any(ch in pat for ch in "*?[") and fnmatch.fnmatch(rel_posix, pat):
            return True
        elif rel_posix == pat:
            return True
    return False


@dataclass
class Settings:
    repo_root: Path
    manifest_path: Path
    python_exe: str
    host: str = "127.0.0.1"
    port: int = 8765

    @property
    def web_dist(self) -> Path:
        return self.manifest_path.parent / "web" / "dist"


def load_settings() -> Settings:
    """Assemble settings from the source layout (tools/explorer/server/…)."""
    here = Path(__file__).resolve()
    explorer_dir = here.parents[1]          # tools/explorer
    guess_root = here.parents[3]            # repo root
    try:
        repo_root = find_repo_root(guess_root)
    except RuntimeError:
        repo_root = find_repo_root(Path.cwd())

    import sys

    return Settings(
        repo_root=repo_root,
        manifest_path=explorer_dir / "pipeline_map.yaml",
        python_exe=os.environ.get("LOFN_EXPLORER_PYTHON", sys.executable),
        host=os.environ.get("LOFN_EXPLORER_HOST", "127.0.0.1"),
        port=int(os.environ.get("LOFN_EXPLORER_PORT", "8765")),
    )
