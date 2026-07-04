"""Path-jailed file access with byte-preserving reads and atomic writes.

The editor works in `\n`-normalized text (Monaco's native form). To keep a no-op
save byte-identical (gate 1) even for RAW text saves, writes re-apply the file's
own EOL and BOM, derived from the current on-disk bytes at write time. That same
read gives the conflict sha, so a concurrent pipeline write can't be clobbered.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .config import path_is_write_allowed
from .yamlio import can_roundtrip, detect_eol, _BOM


class PathJailError(PermissionError):
    """Raised when a path escapes the repo or the write allowlist."""


class ConflictError(RuntimeError):
    """Raised when the on-disk sha no longer matches the caller's base sha."""

    def __init__(self, current_sha: str):
        super().__init__("file changed on disk since it was read")
        self.current_sha = current_sha


@dataclass
class FileView:
    path: str            # repo-relative POSIX
    text: str            # LF-normalized
    eol: str             # "crlf" | "lf"
    bom: bool
    sha: str
    mtime: float
    size: int
    writable: bool
    roundtrip: bool      # can this file offer structured (ruamel) editing?
    is_yaml: bool
    encoding_ok: bool    # strict UTF-8? if not, forced read-only (can't round-trip)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FileService:
    def __init__(self, repo_root: Path, write_allowlist: list[str]):
        self.root = repo_root.resolve()
        self.allowlist = write_allowlist

    # -- path jail --------------------------------------------------------
    def resolve(self, rel: str) -> Path:
        """Resolve a repo-relative path, refusing anything outside the repo."""
        rel = rel.replace("\\", "/").lstrip("/")
        p = (self.root / rel).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise PathJailError(f"path escapes repo root: {rel}")
        return p

    def rel_of(self, p: Path) -> str:
        return str(p.resolve().relative_to(self.root)).replace("\\", "/")

    def is_writable(self, rel: str) -> bool:
        return path_is_write_allowed(rel.replace("\\", "/"), self.allowlist)

    # -- read -------------------------------------------------------------
    def read(self, rel: str) -> FileView:
        p = self.resolve(rel)
        if not p.is_file():
            raise FileNotFoundError(rel)
        raw = p.read_bytes()
        bom = raw.startswith(_BOM)
        body = raw[len(_BOM):] if bom else raw
        eol = detect_eol(body)
        # Strict decode: a file that is NOT clean UTF-8 must never be re-encoded
        # from a lossy `replace` decode (that would corrupt it), so we force it
        # read-only and out of structured mode.
        try:
            text = body.decode("utf-8").replace("\r\n", "\n")
            encoding_ok = True
        except UnicodeDecodeError:
            text = body.decode("utf-8", errors="replace").replace("\r\n", "\n")
            encoding_ok = False
        rel_posix = self.rel_of(p)
        is_yaml = p.suffix.lower() in (".yaml", ".yml")
        return FileView(
            path=rel_posix,
            text=text,
            eol="crlf" if eol == "\r\n" else "lf",
            bom=bom,
            sha=_sha(raw),
            mtime=p.stat().st_mtime,
            size=len(raw),
            writable=self.is_writable(rel_posix) and encoding_ok,
            roundtrip=is_yaml and encoding_ok and can_roundtrip(raw),
            is_yaml=is_yaml,
            encoding_ok=encoding_ok,
        )

    def read_bytes(self, rel: str) -> bytes:
        return self.resolve(rel).read_bytes()

    # -- write ------------------------------------------------------------
    def _encode(self, cur: bytes | None, text: str) -> bytes:
        """Bytes to write. A no-op (text unchanged) returns the ORIGINAL bytes
        verbatim — byte-identical even for mixed-EOL files. A real edit applies
        the file's dominant EOL/BOM (a new file defaults to CRLF)."""
        normalized = text.replace("\r\n", "\n")
        if cur is None:
            return normalized.replace("\n", "\r\n").encode("utf-8")
        bom = cur.startswith(_BOM)
        body = cur[len(_BOM):] if bom else cur
        if normalized == body.decode("utf-8").replace("\r\n", "\n"):
            return cur  # exact no-op — preserves mixed EOL, trailing bytes, everything
        out = normalized.replace("\n", detect_eol(body)).encode("utf-8")
        return (_BOM + out) if bom else out

    def compute_write(self, rel: str, text: str, base_sha: str | None = None) -> bytes:
        """The bytes a write would produce, WITHOUT writing (jail + conflict +
        UTF-8 guards enforced). Shared by write_text and the acceptance sweep."""
        p = self.resolve(rel)
        rel_posix = self.rel_of(p)
        if not self.is_writable(rel_posix):
            raise PathJailError(f"path not in write allowlist: {rel_posix}")
        cur = p.read_bytes() if p.exists() else None
        if cur is not None:
            if base_sha is not None and base_sha != _sha(cur):
                raise ConflictError(_sha(cur))
            body = cur[len(_BOM):] if cur.startswith(_BOM) else cur
            try:
                body.decode("utf-8")
            except UnicodeDecodeError:
                raise PathJailError(f"refusing to overwrite non-UTF-8 file: {rel_posix}")
        return self._encode(cur, text)

    def write_text(self, rel: str, text: str, base_sha: str | None = None) -> FileView:
        """Write `text` (LF-normalized) atomically. `base_sha` guards against
        clobbering a concurrent change: mismatch -> ConflictError."""
        out = self.compute_write(rel, text, base_sha)
        p = self.resolve(rel)
        self._atomic_write(p, out)
        return self.read(self.rel_of(p))

    def write_bytes(self, rel: str, data: bytes) -> FileView:
        """Write exact bytes (used by structured/regen writers that already
        produced final bytes via yamlio.dump)."""
        p = self.resolve(rel)
        rel_posix = self.rel_of(p)
        if not self.is_writable(rel_posix):
            raise PathJailError(f"path not in write allowlist: {rel_posix}")
        self._atomic_write(p, data)
        return self.read(rel_posix)

    @staticmethod
    def _atomic_write(p: Path, data: bytes) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp~")
        tmp.write_bytes(data)
        os.replace(tmp, p)  # atomic on same filesystem (Windows + POSIX)
