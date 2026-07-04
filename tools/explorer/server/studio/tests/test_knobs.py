"""WS2 knobs: safe evaluator (property + adversarial), derivations, invariants,
preset load/save/duplicate, and the run-local gates materializer.

All $0 — pure Python, no provider, no server."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from studio.knobs import (
    SafeEvaluator, EvalError, KnobSet,
    load_preset, save_preset, duplicate_preset, list_presets,
    build_run_gates, materialize_gates, load_base_gates,
)
from studio.flowspec import find_repo_root, flows_dir

_REPO_ROOT = find_repo_root()


# ---------------------------------------------------------------------------
# Safe evaluator — property + adversarial.
# ---------------------------------------------------------------------------

def test_eval_integer_arithmetic():
    ev = SafeEvaluator({"a": 6, "b": 4})
    assert ev.eval("a * b") == 24
    assert ev.eval("a - 2") == 4
    assert ev.eval("a + b * 2") == 14
    assert ev.eval("a % 4") == 2
    assert ev.eval("a // 4") == 1


def test_eval_ceil_floor_min_max():
    ev = SafeEvaluator({"n": 7})
    assert ev.eval("ceil(n / 2)") == 4
    assert ev.eval("floor(n / 2)") == 3
    assert ev.eval("min(n, 3, 10)") == 3
    assert ev.eval("max(n, 3, 10)") == 10
    assert ev.eval("abs(0 - n)") == 7


@pytest.mark.parametrize("n", list(range(2, 13)))
def test_eval_barbell_property(n):
    """PROPERTY: ceil(n/2) + (n - ceil(n/2)) == n for every n in range."""
    ev = SafeEvaluator({"n_pairs": n})
    acc = ev.eval("ceil(n_pairs / 2)")
    amb = ev.eval("n_pairs - ceil(n_pairs / 2)")
    assert acc + amb == n
    assert acc == math.ceil(n / 2)


def test_eval_band_subscript():
    ev = SafeEvaluator({"sung_lines": [70, 120], "sung_lines_target": [78, 110]})
    assert ev.eval("sung_lines[0]") == 70
    assert ev.eval("sung_lines[1]") == 120
    assert ev.eval("sung_lines[0] < sung_lines_target[0]") is True


def test_eval_comparisons_and_bool():
    ev = SafeEvaluator({"a": 70, "b": 78, "c": 110})
    assert ev.eval("a < b") is True
    assert ev.eval("a < b and b < c") is True
    assert ev.eval("a > b or b < c") is True
    assert ev.eval("not (a > b)") is True


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('/etc/passwd')",
    "().__class__",
    "a.b",
    "[x for x in range(3)]",
    "lambda: 1",
    "1 if a else 2",
    "a; b",
    "print(a)",
    "eval('1')",
    "{'k': 1}",
    "f'{a}'",
])
def test_eval_rejects_dangerous(expr):
    """ADVERSARIAL: nothing outside the allowlist evaluates. No eval() reachable."""
    with pytest.raises(EvalError):
        SafeEvaluator({"a": 1, "b": 2}).eval(expr)


def test_eval_unknown_name_raises():
    with pytest.raises(EvalError):
        SafeEvaluator({}).eval("nope * 2")


def test_eval_division_by_zero_raises():
    with pytest.raises(EvalError):
        SafeEvaluator({"a": 1}).eval("a / 0")


def test_eval_disallowed_function_raises():
    with pytest.raises(EvalError):
        SafeEvaluator({"a": 1}).eval("sum(a)")


# ---------------------------------------------------------------------------
# Derivations + fixpoint.
# ---------------------------------------------------------------------------

def test_derived_chain_resolves_via_fixpoint():
    """barbell_ambitious depends on barbell_accessible depends on n_pairs."""
    ks = KnobSet.from_dict({"knobs": {
        "n_pairs": {"value": 6},
        "barbell_accessible": {"expr": "ceil(n_pairs / 2)", "override": None},
        "barbell_ambitious": {"expr": "n_pairs - barbell_accessible"},
        "total_prompts": {"expr": "n_pairs * 4"},
    }})
    assert ks.resolved["barbell_accessible"] == 3
    assert ks.resolved["barbell_ambitious"] == 3
    assert ks.resolved["total_prompts"] == 24


def test_override_pins_derived_knob():
    ks = KnobSet.from_dict({"knobs": {
        "n_pairs": {"value": 6},
        "barbell_accessible": {"expr": "ceil(n_pairs / 2)", "override": 5},
    }})
    # Override wins over the expr.
    assert ks.resolved["barbell_accessible"] == 5


def test_cycle_raises():
    with pytest.raises(EvalError):
        KnobSet.from_dict({"knobs": {
            "a": {"expr": "b + 1"},
            "b": {"expr": "a + 1"},
        }})


# ---------------------------------------------------------------------------
# Invariants — the canon preset holds; violations block Run.
# ---------------------------------------------------------------------------

def test_classic_preset_invariants_hold():
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    assert ks.ok
    assert ks.invariant_violations() == []


def test_n_concepts_below_2x_npairs_violates():
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    # 10 pairs needs >= 20 concepts; default n_concepts is 12 -> violation.
    ks10 = ks.with_value("n_pairs", 10)
    assert not ks10.ok
    viols = {v.expr for v in ks10.invariant_violations()}
    assert any("n_concepts" in e for e in viols)


def test_barbell_halves_sum_invariant():
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    # Any n_pairs keeps accessible+ambitious==n_pairs (derived), so that invariant
    # never breaks — assert it stays satisfied across the range.
    for n in range(2, 13):
        k = ks.with_value("n_pairs", n)
        r = {v.expr: v for v in k.check_invariants()}
        sum_inv = next(e for e in r if "barbell_accessible + barbell_ambitious" in e)
        assert r[sum_inv].ok


def test_sung_line_band_ordering_invariant():
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    # Break the ordering: push target-low below the floor.
    bad = ks.with_value("sung_lines_target", [60, 110])
    assert not bad.ok
    assert any("sung_lines[0] < sung_lines_target[0]" in v.expr
               for v in bad.invariant_violations())


def test_min_expr_bound_surfaced_as_invariant():
    ks = KnobSet.from_dict({"knobs": {
        "n_pairs": {"value": 6},
        "n_concepts": {"value": 4, "min_expr": "2 * n_pairs"},
    }})
    # 4 < 2*6 -> the implicit min_expr bound fails.
    assert not ks.ok
    viols = ks.invariant_violations()
    assert any("n_concepts >= 2 * n_pairs" in v.expr for v in viols)


# ---------------------------------------------------------------------------
# with_value is copy-on-write.
# ---------------------------------------------------------------------------

def test_with_value_does_not_mutate_original():
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    ks2 = ks.with_value("n_pairs", 8)
    assert ks.resolved["n_pairs"] == 6      # original untouched
    assert ks2.resolved["n_pairs"] == 8
    assert ks2.resolved["total_prompts"] == 32


def test_with_value_on_derived_raises():
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    with pytest.raises(ValueError):
        ks.with_value("barbell_accessible", 5)  # derived -> must use override


# ---------------------------------------------------------------------------
# Preset load / save / duplicate round-trip.
# ---------------------------------------------------------------------------

def test_preset_save_load_roundtrip(tmp_path: Path):
    flows = tmp_path / "flows"
    (flows / "presets").mkdir(parents=True)
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    save_preset(flows, ks, name="mine")
    back = load_preset(flows, "mine")
    assert back.resolved["n_pairs"] == ks.resolved["n_pairs"]
    assert back.resolved["total_prompts"] == ks.resolved["total_prompts"]
    assert back.gates_bindings() == ks.gates_bindings()


def test_preset_duplicate(tmp_path: Path):
    flows = tmp_path / "flows"
    (flows / "presets").mkdir(parents=True)
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    save_preset(flows, ks, name="orig")
    duplicate_preset(flows, "orig", "copy")
    names = list_presets(flows)
    assert "orig" in names and "copy" in names


def test_wide_preset_loads():
    """The wide-10x2 preset ships in the box; confirm it resolves."""
    ks = load_preset(flows_dir(_REPO_ROOT), "wide-10x2")
    assert ks.resolved["n_pairs"] == 10


# ---------------------------------------------------------------------------
# Run-local gates materializer.
# ---------------------------------------------------------------------------

def test_build_run_gates_binds_and_passes_through():
    base = load_base_gates(_REPO_ROOT)
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    merged = build_run_gates(base, ks)
    # gates_key-bound knob written.
    assert merged["total_prompts"] == 24
    assert merged["music_prompt_chars"] == [850, 1000]
    # Un-knobbed canon keys pass through unchanged.
    assert "house_lexicon" in merged
    assert "ban_words" in merged
    assert merged["house_lexicon"] == base["house_lexicon"]


def test_knob_change_moves_materialized_threshold():
    base = load_base_gates(_REPO_ROOT)
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4").with_value(
        "n_variations_per_pair", 2
    )
    merged = build_run_gates(base, ks)
    assert merged["total_prompts"] == 12    # 6 pairs * 2 variations


def test_materialize_gates_writes_file(tmp_path: Path):
    base = load_base_gates(_REPO_ROOT)
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    p = materialize_gates(tmp_path, base, ks)
    assert p.exists()
    assert p.name == "gates.yaml"
    # LF-normalized (no CRLF) — v1 byte discipline.
    assert b"\r\n" not in p.read_bytes()


def test_materialized_gates_read_by_validate_step():
    """DUMB BACKEND / gate parity: the canon validator's own loader reads our file
    and sees the knob-adjusted number."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_step", _REPO_ROOT / "scripts" / "validate_step.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import tempfile
    base = load_base_gates(_REPO_ROOT)
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4").with_value(
        "n_variations_per_pair", 2
    )
    d = Path(tempfile.mkdtemp())
    p = materialize_gates(d, base, ks)
    gates = mod.load_gates(p)
    assert gates["total_prompts"] == 12
    assert gates["music_prompt_chars"] == [850, 1000]
    assert len(gates["house_lexicon"]) == len(base["house_lexicon"])
