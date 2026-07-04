"""Pricing math vs hand-computed fixtures."""
from __future__ import annotations

from pathlib import Path

from studio.pricing import PricingTable, ModelPrice, budget_estimate

_REPO_ROOT = Path(__file__).resolve().parents[5]   # repo root (E:/git/lofn)


def _table_from_repo() -> PricingTable:
    return PricingTable.default(_REPO_ROOT)


def test_seeded_rates_load():
    t = _table_from_repo()
    assert t.has("claude-opus-4-8")
    assert t.has("claude-fable-5")


def test_opus_cost_hand_computed():
    t = _table_from_repo()
    # opus-4-8 = 5/25 per MTok. 1,000,000 in + 200,000 out.
    est = t.estimate("claude-opus-4-8", 1_000_000, 200_000)
    assert est.priced
    # 1.0 * 5.0  +  0.2 * 25.0  =  5.0 + 5.0 = 10.0
    assert abs(est.cost - 10.0) < 1e-9


def test_haiku_cost_hand_computed():
    t = _table_from_repo()
    # haiku-4-5 = 1/5. 500,000 in + 500,000 out => 0.5*1 + 0.5*5 = 3.0
    est = t.estimate("claude-haiku-4-5", 500_000, 500_000)
    assert abs(est.cost - 3.0) < 1e-9


def test_fable_cost_hand_computed():
    t = _table_from_repo()
    # fable-5 = 10/50. 100k in + 100k out => 0.1*10 + 0.1*50 = 6.0
    est = t.estimate("claude-fable-5", 100_000, 100_000)
    assert abs(est.cost - 6.0) < 1e-9


def test_unknown_model_is_unpriced_loudly():
    t = _table_from_repo()
    est = t.estimate("some-model-that-does-not-exist", 1000, 1000)
    assert est.priced is False
    assert est.cost is None
    assert "unpriced" in est.note


def test_poe_and_or_prefix_stripping():
    # A bare table with only claude-opus-4-8 resolves Poe-/OR- prefixed variants.
    t = PricingTable({"claude-opus-4-8": ModelPrice("claude-opus-4-8", 5.0, 25.0)})
    assert t.has("Poe-claude-opus-4-8")
    assert t.has("OR-claude-opus-4-8")
    est = t.estimate("OR-claude-opus-4-8", 1_000_000, 0)
    assert abs(est.cost - 5.0) < 1e-9


def test_budget_estimate_multiplies_calls():
    t = _table_from_repo()
    # 24 calls of opus at 10k in / 2k out each.
    est = budget_estimate(t, "claude-opus-4-8", 10_000, 2_000, n_calls=24)
    # per call: 0.01*5 + 0.002*25 = 0.05 + 0.05 = 0.10 ; *24 = 2.40
    assert abs(est.cost - 2.40) < 1e-9
    assert est.tokens_in == 240_000
    assert est.tokens_out == 48_000
