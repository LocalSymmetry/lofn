"""Git status / diff / revert — the app shows dirty state and diff vs HEAD, and
can restore a file. No commit/push UI in v1 (git stays in the user's hands).
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class GitService:
    def __init__(self, repo_root: Path):
        self.root = repo_root.resolve()

    def _git(self, *args: str, timeout: int = 15) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

    def status_map(self) -> dict[str, str]:
        """repo-relative POSIX path -> one of modified|added|untracked|deleted|renamed."""
        cp = self._git("status", "--porcelain")
        out: dict[str, str] = {}
        for line in cp.stdout.splitlines():
            if not line or len(line) < 4:
                continue
            code, rest = line[:2], line[3:]
            path = rest.split(" -> ")[-1].strip().strip('"')
            path = path.replace("\\", "/")
            if "?" in code:
                out[path] = "untracked"
            elif "D" in code:
                out[path] = "deleted"
            elif "R" in code:
                out[path] = "renamed"
            elif "A" in code:
                out[path] = "added"
            else:
                out[path] = "modified"
        return out

    def status_of(self, rel: str) -> str:
        return self.status_map().get(rel.replace("\\", "/"), "clean")

    def diff(self, rel: str) -> str:
        """Unified diff of the working tree vs HEAD for one file."""
        rel = rel.replace("\\", "/")
        cp = self._git("diff", "HEAD", "--", rel)
        if cp.stdout.strip():
            return cp.stdout
        # Untracked file: show it as an all-added diff against /dev/null.
        cp2 = self._git("diff", "--no-index", "--", "/dev/null", rel)
        return cp2.stdout

    def revert(self, rel: str) -> dict:
        """Restore a tracked file to its HEAD content. Untracked files have no
        HEAD version, so this reports that rather than deleting user work."""
        rel = rel.replace("\\", "/")
        if self.status_of(rel) == "untracked":
            return {"reverted": False, "reason": "untracked — nothing in HEAD to revert to"}
        cp = self._git("checkout", "HEAD", "--", rel)
        if cp.returncode != 0:
            return {"reverted": False, "reason": cp.stderr.strip() or "git checkout failed"}
        return {"reverted": True}
