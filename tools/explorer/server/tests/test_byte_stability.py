"""Acceptance gate #1 (+ jail, conflict, EOL) — the non-waivable byte-stability sweep.

A no-op save must be byte-identical for EVERY editable file. This test reproduces
exactly what FileService.write_text produces for a no-op (raw text re-encoded with
the file's own EOL/BOM) and asserts it equals the original bytes, across the whole
write-allowlist surface. Non-UTF-8 files are asserted read-only (never corruptible).

Run: tools/explorer/server/.venv/Scripts/python -m pytest tools/explorer/server/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]  # repo root
sys.path.insert(0, str(ROOT))

from tools.explorer.server.config import load_settings  # noqa: E402
from tools.explorer.server.files import (  # noqa: E402
    ConflictError, FileService, PathJailError,
)
from tools.explorer.server.manifest import load_manifest  # noqa: E402
from tools.explorer.server.yamlio import _BOM  # noqa: E402

SETTINGS = load_settings()
MANIFEST = load_manifest(SETTINGS.manifest_path)
FS = FileService(SETTINGS.repo_root, MANIFEST.write_allowlist)

EXTS = {".md", ".yaml", ".yml", ".txt", ".csv", ".json"}
ROOTS = ["skills", "vault", ".claude/skills"]


def _editable_files() -> list[Path]:
    out: list[Path] = []
    for r in ROOTS:
        for p in (SETTINGS.repo_root / r).rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS and ".venv" not in p.parts:
                out.append(p)
    return out


ALL = _editable_files()


def test_editable_surface_nonempty():
    assert len(ALL) > 200, f"expected a large editable surface, found {len(ALL)}"


@pytest.mark.parametrize("path", ALL, ids=lambda p: str(p.relative_to(SETTINGS.repo_root)))
def test_noop_save_is_byte_identical(path: Path):
    raw = path.read_bytes()
    rel = str(path.relative_to(SETTINGS.repo_root)).replace("\\", "/")
    view = FS.read(rel)
    if not view.encoding_ok:
        # non-UTF-8 -> must be read-only (never re-encoded)
        assert not view.writable, f"{rel} is non-UTF-8 but writable"
        return
    # The real write path, computed without mutating disk. A no-op (view.text
    # unchanged) must reproduce the original bytes exactly — even for mixed EOL.
    assert FS.compute_write(rel, view.text, view.sha) == raw, f"no-op save changed bytes for {rel}"


def test_known_raw_mode_files():
    """The 4 files with irregular block scalars are raw-mode (no structured edit),
    yet still byte-safe (covered by the sweep above)."""
    for name in ["lofn-prime-mini", "emotion-architects", "holographic-heritage", "masters-of-the-dark-room"]:
        for coll in ("personalities", "panels"):
            p = SETTINGS.repo_root / "skills/orchestration" / coll / f"{name}.yaml"
            if p.exists():
                v = FS.read(str(p.relative_to(SETTINGS.repo_root)).replace("\\", "/"))
                assert v.roundtrip is False, f"{name} unexpectedly round-trips"


def test_eol_detected_per_file():
    crlf = FS.read("skills/orchestration/personalities/afrogroove.yaml")
    assert crlf.eol == "crlf"
    # eol is always one of the two, and a no-op save is byte-identical regardless
    # of mixed EOL (the real guarantee); don't hardcode a file as pure-LF since
    # several repo files are mixed.
    v = FS.read("vault/film_styles.txt")
    assert v.eol in ("crlf", "lf")
    assert FS.compute_write(v.path, v.text, v.sha) == (SETTINGS.repo_root / "vault/film_styles.txt").read_bytes()


def test_path_jail_blocks_traversal():
    with pytest.raises(PathJailError):
        FS.resolve("../outside.txt")


def test_write_outside_allowlist_blocked():
    with pytest.raises(PathJailError):
        FS.write_text("README.md", "x", base_sha=None)


def test_stale_base_sha_conflicts():
    rel = "skills/orchestration/personalities/afrogroove.yaml"
    v = FS.read(rel)
    with pytest.raises(ConflictError):
        FS.write_text(rel, v.text, base_sha="deadbeef")


def test_real_noop_write_roundtrips(tmp_path):
    """Exercise the ACTUAL write path (atomic replace) on a real file, no-op."""
    rel = "skills/orchestration/personalities/afrogroove.yaml"
    before = (SETTINGS.repo_root / rel).read_bytes()
    v = FS.read(rel)
    FS.write_text(rel, v.text, base_sha=v.sha)
    after = (SETTINGS.repo_root / rel).read_bytes()
    assert after == before, "real no-op write changed bytes"
