"""WS promote tests — $0, no server. Never mutates real canon.

Every write goes to a TEMP repo root (a copy of vault/gates.yaml + a prose file)
with its own FileService, so the real vault/gates.yaml is never touched. Covers:
  * preview: gates.yaml value diff + simple prose-patch diff, no write,
  * apply: values land byte-safely via FileService; ruamel round-trip preserves
    comments/structure; unknown keys are skipped (never added to canon),
  * apply: a non-unique prose patch is REFUSED (never an ambiguous rewrite),
  * brief: a PROMOTION_BRIEF.md renders the structural hand-off.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from studio import promote
from studio.flowspec import find_repo_root

try:
    from tools.explorer.server.files import FileService
except ImportError:  # pragma: no cover
    from files import FileService  # type: ignore

_REPO_ROOT = find_repo_root()
_ALLOWLIST = ["vault/**", "skills/**"]


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """A throwaway repo root with a real copy of vault/gates.yaml + a prose file."""
    (tmp_path / "vault").mkdir()
    shutil.copy2(_REPO_ROOT / "vault" / "gates.yaml", tmp_path / "vault" / "gates.yaml")
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "note.md").write_text(
        "The music prompt band is 850-1000 characters (canon).\n",
        encoding="utf-8", newline="\n",
    )
    return tmp_path


def _fs(root: Path) -> FileService:
    return FileService(root, _ALLOWLIST)


def test_preview_gate_value_diff_no_write(temp_repo: Path):
    fs = _fs(temp_repo)
    before = (temp_repo / "vault" / "gates.yaml").read_bytes()
    res = promote.preview(fs, gate_values={"music_prompt_hug_ceiling": 990})
    assert res["gates"]["changed"] is True
    assert "music_prompt_hug_ceiling" in res["gates"]["applied"]
    assert "-music_prompt_hug_ceiling: 985" in res["gates"]["diff"]
    assert "+music_prompt_hug_ceiling: 990" in res["gates"]["diff"]
    # PREVIEW writes nothing.
    assert (temp_repo / "vault" / "gates.yaml").read_bytes() == before


def test_preview_skips_unknown_key(temp_repo: Path):
    fs = _fs(temp_repo)
    res = promote.preview(fs, gate_values={"not_a_real_gate": 5})
    assert res["gates"]["changed"] is False
    assert "not_a_real_gate" in res["skipped_keys"]


def test_apply_gate_value_bytes_safe(temp_repo: Path):
    fs = _fs(temp_repo)
    orig = (temp_repo / "vault" / "gates.yaml").read_text(encoding="utf-8")
    out = promote.apply(fs, temp_repo, gate_values={"music_prompt_hug_ceiling": 990})
    assert out["applied_gates"]["music_prompt_hug_ceiling"] == 990
    after = (temp_repo / "vault" / "gates.yaml").read_text(encoding="utf-8")
    # Only the one value line changed; comments/structure preserved (round-trip).
    assert "music_prompt_hug_ceiling: 990" in after
    assert "music_prompt_hug_ceiling: 985" not in after
    # A representative comment + an untouched key survive verbatim.
    assert "# vault/gates.yaml" in after
    assert orig.count("\n") == after.count("\n")  # no lines added/removed
    # meta-check ran (WARN-only, exits 0 on this isolated copy or reports cleanly).
    assert "meta_check" in out and "exit_code" in out["meta_check"]


def test_apply_band_value(temp_repo: Path):
    fs = _fs(temp_repo)
    out = promote.apply(fs, temp_repo, gate_values={"music_prompt_chars": [860, 1010]})
    after = (temp_repo / "vault" / "gates.yaml").read_text(encoding="utf-8")
    assert out["applied_gates"]["music_prompt_chars"] == [860, 1010]
    assert "music_prompt_chars: [860, 1010]" in after


def test_apply_simple_prose_patch(temp_repo: Path):
    fs = _fs(temp_repo)
    out = promote.apply(fs, temp_repo, patches=[{
        "file": "skills/note.md",
        "old": "850-1000 characters",
        "new": "860-1010 characters",
    }])
    applied = {p["file"]: p for p in out["applied_patches"]}
    assert applied["skills/note.md"]["applied"] is True
    txt = (temp_repo / "skills" / "note.md").read_text(encoding="utf-8")
    assert "860-1010 characters" in txt


def test_apply_refuses_non_unique_patch(temp_repo: Path):
    fs = _fs(temp_repo)
    (temp_repo / "skills" / "dup.md").write_text(
        "band X band X band X\n", encoding="utf-8", newline="\n")
    out = promote.apply(fs, temp_repo, patches=[{
        "file": "skills/dup.md", "old": "band X", "new": "band Y",
    }])
    applied = {p["file"]: p for p in out["applied_patches"]}
    assert applied["skills/dup.md"]["applied"] is False
    assert "not unique" in applied["skills/dup.md"]["detail"]
    # The file is unchanged (ambiguous patch refused).
    assert (temp_repo / "skills" / "dup.md").read_text(encoding="utf-8") == "band X band X band X\n"


def test_apply_refuses_out_of_jail_gate_write(tmp_path: Path):
    # A FileService whose allowlist does NOT include vault/** must refuse the write.
    (tmp_path / "vault").mkdir()
    shutil.copy2(_REPO_ROOT / "vault" / "gates.yaml", tmp_path / "vault" / "gates.yaml")
    fs = FileService(tmp_path, ["skills/**"])  # vault NOT writable
    out = promote.apply(fs, tmp_path, gate_values={"music_prompt_hug_ceiling": 990})
    assert out["applied_gates"] == {}
    assert any("refused" in e for e in out["errors"])


def test_brief_renders_structural_handoff():
    md = promote.brief(
        run_a="run-A", run_b="run-B", title="Wide portfolio",
        summary="Wide 10x2 beat classic 6x4 on the barbell.",
        structural_changes=[{"kind": "new gate", "where": "step 05",
                             "what": "pair_count_band widened to n_pairs"}],
        knob_delta=[{"knob": "n_pairs", "a": 6, "b": 10}],
    )
    assert "# PROMOTION BRIEF" in md
    assert "REQUIRES DELIBERATE CANON EDIT" in md
    assert "pair_count_band widened" in md
    assert "`n_pairs`" in md
