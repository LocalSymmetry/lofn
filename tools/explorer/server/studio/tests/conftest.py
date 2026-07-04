"""Shared test fixtures. All tests are MOCK-only and spend $0 (never hit a live
provider endpoint, never start a server)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``studio`` importable as a top-level package when pytest is run from the
# studio dir (mirrors how the v1 tests bootstrap the server package).
_SERVER_DIR = Path(__file__).resolve().parents[2]   # tools/explorer/server
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

# The tests dir itself, so sibling helper modules (``_fixtures_music``) import as
# top-level modules regardless of the pytest invocation directory.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# The repo root, so the v1 byte-safe layer reaches its package by fully-qualified
# name (``tools.explorer.server.files``) when studio is imported top-level here
# (promote.py's write-back path; api.py's FileService). Harmless if already present.
_REPO_ROOT = _SERVER_DIR.parents[2]  # tools/explorer/server -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def tmp_fixtures(tmp_path: Path) -> Path:
    d = tmp_path / "fixtures"
    d.mkdir()
    return d


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """A throwaway repo root for KeyResolver (no real .env.studio)."""
    return tmp_path
