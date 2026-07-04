"""Run the repo's own validator scripts as subprocesses.

Zero domain logic lives here: thresholds, checks, and drift all come from the
existing scripts (design plan gate 3, "no reimplementation"). Only scripts named
in the manifest's `validators` list may be run (no arbitrary execution).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .manifest import Manifest


class ValidatorRunner:
    def __init__(self, repo_root: Path, python_exe: str, manifest: Manifest):
        self.root = repo_root.resolve()
        self.python = python_exe
        self.allowed = {v["id"]: v["path"] for v in manifest.validators}
        self.allowed_paths = set(self.allowed.values())

    def run_script(self, script_rel: str, args: list[str], timeout: int = 120) -> dict:
        script_rel = script_rel.replace("\\", "/")
        if script_rel not in self.allowed_paths:
            raise PermissionError(f"script not in manifest validator allowlist: {script_rel}")
        cmd = [self.python, str(self.root / script_rel), *args]
        t0 = time.perf_counter()
        try:
            cp = subprocess.run(
                cmd,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            rc, out, err = cp.returncode, cp.stdout, cp.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = 124, e.stdout or "", (e.stderr or "") + f"\n(timed out after {timeout}s)"
        ms = round((time.perf_counter() - t0) * 1000)
        return {
            "cmd": " ".join(str(c) for c in [Path(self.python).name, script_rel, *args]),
            "returncode": rc,
            "stdout": out,
            "stderr": err,
            "duration_ms": ms,
        }

    def run_by_id(self, validator_id: str, args: list[str], timeout: int = 120) -> dict:
        if validator_id not in self.allowed:
            raise PermissionError(f"unknown validator id: {validator_id}")
        return self.run_script(self.allowed[validator_id], args, timeout=timeout)
