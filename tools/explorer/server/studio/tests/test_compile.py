"""WS2 /compile: resolved prompts + diffs + token/cost estimate, and S3
(Compile ≡ execute) at the function level. All $0 — mock/no provider."""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.knobs import load_preset
from studio.flowspec import find_repo_root, flows_dir, load_flowspec
from studio.pricing import PricingTable
from studio.compile import compile_flow, load_flow_and_knobs
from studio.resolver import resolve

_REPO_ROOT = find_repo_root()


def _music_flow():
    return load_flowspec(flows_dir(_REPO_ROOT) / "lofn-music@1.flow.yaml")


def _classic():
    return load_preset(flows_dir(_REPO_ROOT), "classic-6x4")


def _pricing():
    return PricingTable.default(_REPO_ROOT)


_CTX = {
    "input": "tide pools", "seed": "S", "meta_prompt": "M", "personality": "P",
    "concept_panel": "C", "medium_panel": "MD", "marketing_panel": "MK",
    "flairs": "F", "genres_list": "g", "frames_list": "f", "image_context": "",
}


def test_compile_single_step():
    res = compile_flow(
        _music_flow(), _classic(), _CTX, _pricing(), step="00", repo_root=_REPO_ROOT
    )
    assert len(res.steps) == 1
    sc = res.steps[0]
    assert sc.step == "00"
    assert sc.est_tokens_in > 0
    assert sc.priced          # opus-4-8 is in pricing.yaml
    assert sc.cost_estimate and sc.cost_estimate > 0


def test_compile_all_steps():
    res = compile_flow(_music_flow(), _classic(), _CTX, _pricing(), repo_root=_REPO_ROOT)
    ids = [s.step for s in res.steps]
    # 12 LLM steps (00..11); the portfolio validator is excluded.
    assert ids == ["00", "01", "02", "03", "04", "05",
                   "06", "07", "08", "09", "10", "11"]
    assert res.total_cost_estimate and res.total_cost_estimate > 0


def test_compile_fanout_cost_multiplied_by_pairs():
    """A fan-out step's cost is per-call * n_pairs; a coordinator step is *1."""
    res = compile_flow(_music_flow(), _classic(), _CTX, _pricing(), repo_root=_REPO_ROOT)
    coord = next(s for s in res.steps if s.step == "00")
    fan = next(s for s in res.steps if s.step == "08")
    # single-call token estimate for each; the fan-out total is *6 pairs.
    per_call_fan = resolve(_music_flow(), _classic(), _CTX, "08", pair=1,
                           repo_root=_REPO_ROOT).meta["est_tokens_in"]
    assert fan.est_tokens_in == per_call_fan * 6
    # coordinator not multiplied.
    per_call_coord = resolve(_music_flow(), _classic(), _CTX, "00",
                             repo_root=_REPO_ROOT).meta["est_tokens_in"]
    assert coord.est_tokens_in == per_call_coord


def test_compile_diff_reflects_knob_change():
    """The 'feel the knob' loop: n_pairs 6->8 produces a diff naming both."""
    current = _classic().with_value("n_pairs", 8).with_value("n_concepts", 16)
    res = compile_flow(
        _music_flow(), current, _CTX, _pricing(),
        step="00", baseline=_classic(), repo_root=_REPO_ROOT,
    )
    diff = res.steps[0].diff
    assert diff is not None
    assert any(l.startswith("-") and "exactly 6" in l for l in diff.splitlines())
    assert any(l.startswith("+") and "exactly 8" in l for l in diff.splitlines())


def test_compile_no_baseline_has_no_diff():
    res = compile_flow(
        _music_flow(), _classic(), _CTX, _pricing(), step="00", repo_root=_REPO_ROOT
    )
    assert res.steps[0].diff is None


def test_compile_allowed_on_broken_preset_but_run_blocked():
    """Compile is allowed on invariant violations; only Run is blocked (§3.2)."""
    broken = _classic().with_value("n_pairs", 10)   # n_concepts 12 < 20 -> violation
    res = compile_flow(
        _music_flow(), broken, _CTX, _pricing(), step="00", repo_root=_REPO_ROOT
    )
    d = res.as_dict()
    assert d["run_allowed"] is False
    assert d["invariant_violations"]         # reported, not fatal
    assert res.steps and res.steps[0].resolved.user   # still compiled


def test_compile_equals_resolve_S3():
    """S3 foundation: the bytes /compile surfaces are exactly resolve()'s bytes."""
    flow, ks = _music_flow(), _classic()
    res = compile_flow(flow, ks, _CTX, _pricing(), step="08", repo_root=_REPO_ROOT)
    direct = resolve(flow, ks, _CTX, "08", pair=1, repo_root=_REPO_ROOT)
    assert res.steps[0].resolved.user == direct.user
    assert res.steps[0].resolved.text_sha == direct.text_sha


# ---------------------------------------------------------------------------
# load_flow_and_knobs — inline vs preset vs default.
# ---------------------------------------------------------------------------

def test_load_knobs_preset_name():
    flow, ks = load_flow_and_knobs("lofn-music", 1, "classic-6x4", _REPO_ROOT)
    assert ks.resolved["n_pairs"] == 6


def test_load_knobs_inline_bare_map():
    inline = {
        "n_pairs": {"value": 8},
        "n_variations_per_pair": {"value": 4},
        "total_prompts": {"expr": "n_pairs * n_variations_per_pair"},
    }
    _, ks = load_flow_and_knobs("lofn-music", 1, inline, _REPO_ROOT)
    assert ks.resolved["n_pairs"] == 8
    assert ks.resolved["total_prompts"] == 32


def test_load_knobs_default_from_flow():
    _, ks = load_flow_and_knobs("lofn-music", 1, None, _REPO_ROOT)
    assert ks.resolved["n_pairs"] == 6   # flow's knobs_defaults -> classic-6x4


def test_load_flow_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_flow_and_knobs("lofn-nonexistent", 1, None, _REPO_ROOT)
