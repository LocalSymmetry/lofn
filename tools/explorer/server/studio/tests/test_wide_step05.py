"""Wide-10x2 step-05 validator test — $0, no server, no live provider.

Proves the wide portfolio (n_pairs=10) validates end-to-end at step 05:
  * the run-local gates materializer emits ``pair_count_band`` derived from n_pairs
    (so a 10-pair concept_medium_pairs.json is IN band), and
  * validate_step.py --gates reads that band (fail-open to [4,7] without it) and
    passes a 10-pair step-05 artifact.

Also asserts the classic-6x4 preset still materializes a band that admits 6 pairs
(no regression for the canon default), and that WITHOUT --gates the hardcoded-era
default still holds (10 pairs would fail; 6 passes).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from studio.flowspec import find_repo_root, flows_dir
from studio.knobs import load_base_gates, load_preset, materialize_gates

_REPO_ROOT = find_repo_root()
_VALIDATE = _REPO_ROOT / "scripts" / "validate_step.py"


def _step05_artifact() -> str:
    """A canonical, gate-passing step-05 music artifact (markers pair/concept/medium)."""
    body = "\n".join(
        f"Pair {i}: concept '{['tidewrack','emberfall','glasslimb','saltmoor','ashfen','coldbell','driftmark','longshoal','nettlewick','stonewash'][i-1]}' "
        f"paired with medium 'photographic realism at dawn variation {i}' — a distinct "
        f"concept x medium selection that carries its own emotional charge and craft note."
        for i in range(1, 11)
    )
    return (
        "# Step 05 — Concept x Medium Pairs\n\n"
        "This step selects the concept and medium pairs for the portfolio. Each "
        "pair binds one concept to one medium so the downstream song generation "
        "has a concrete brief.\n\n"
        "## Selected pairs\n\n" + body + "\n\n"
        "Markers present: pair, concept, medium.\n"
    )


def _pairs_json(n: int) -> list:
    return [
        {"pair_num": i, "concept": f"concept-{i}", "medium": f"medium-{i}"}
        for i in range(1, n + 1)
    ]


def _run_validate(step: str, artifact: Path, gates: Path | None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(_VALIDATE)]
    if gates is not None:
        cmd += ["--gates", str(gates)]
    cmd += [step, str(artifact)]
    return subprocess.run(cmd, cwd=str(_REPO_ROOT), text=True, capture_output=True, timeout=120)


def _write_step05(run_dir: Path, n_pairs: int) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    art = run_dir / "pair_00_step05_pairs.md"  # name must start with pair_ + _step
    # Actually step-05 is a coordinator artifact; use a canonical step name.
    art = run_dir / "step05_concept_medium_pairs.md"
    art.write_text(_step05_artifact(), encoding="utf-8", newline="\n")
    (run_dir / "concept_medium_pairs.json").write_text(
        json.dumps(_pairs_json(n_pairs), indent=2), encoding="utf-8", newline="\n")
    return art


def test_materializer_emits_pair_count_band_from_n_pairs(tmp_path: Path):
    base = load_base_gates(_REPO_ROOT)
    ks = load_preset(flows_dir(_REPO_ROOT), "wide-10x2")
    gates_path = materialize_gates(tmp_path, base, ks)
    from studio.knobs import _yaml, _plainify
    data = _plainify(_yaml().load(gates_path.read_text(encoding="utf-8")))
    assert data["pair_count_band"] == [10, 10]
    # Canon vault/gates.yaml is UNCHANGED — the key is only in the run-local file.
    assert "pair_count_band" not in load_base_gates(_REPO_ROOT)


def test_wide_10x2_step05_passes_with_run_local_gates(tmp_path: Path):
    base = load_base_gates(_REPO_ROOT)
    ks = load_preset(flows_dir(_REPO_ROOT), "wide-10x2")
    run_dir = tmp_path / "run"
    gates_path = materialize_gates(run_dir, base, ks)  # writes run_dir/gates.yaml
    art = _write_step05(run_dir, n_pairs=10)

    proc = _run_validate("05", art, gates_path)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"


def test_classic_6x4_step05_passes_with_run_local_gates(tmp_path: Path):
    base = load_base_gates(_REPO_ROOT)
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    run_dir = tmp_path / "run"
    gates_path = materialize_gates(run_dir, base, ks)
    art = _write_step05(run_dir, n_pairs=6)

    proc = _run_validate("05", art, gates_path)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"


def test_no_gates_flag_keeps_canon_default_band(tmp_path: Path):
    """WITHOUT --gates, the canon default [4,7] holds: 6 pairs pass, 10 fail."""
    run6 = tmp_path / "r6"
    art6 = _write_step05(run6, n_pairs=6)
    assert _run_validate("05", art6, None).returncode == 0

    run10 = tmp_path / "r10"
    art10 = _write_step05(run10, n_pairs=10)
    proc = _run_validate("05", art10, None)
    assert proc.returncode != 0
    assert "4-7 pairs" in (proc.stdout + proc.stderr)
