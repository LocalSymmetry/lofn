"""Pricing table + budget estimation (plan §7).

Loads ``tools/explorer/pricing.yaml`` (per-MTok in/out, user-editable) and turns a
model + token usage into a dollar cost. An UNKNOWN model is not silently free — it
resolves to ``unpriced`` and the caller counts tokens only, loudly.

DUMB BACKEND: no numbers are hard-coded here beyond the "per million" scale; every
rate comes from the YAML. OpenRouter returns a per-request cost in its response —
the engine PREFERS that actual (see ``ProviderResult.cost``); this table is the
estimate and the fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

_MTOK = 1_000_000


def _yaml_load(path: Path) -> dict:
    from ruamel.yaml import YAML

    y = YAML(typ="safe")
    with path.open("r", encoding="utf-8") as f:
        return y.load(f) or {}


@dataclass(frozen=True)
class ModelPrice:
    """Per-MTok input/output rate for one model."""

    model: str
    in_per_mtok: float
    out_per_mtok: float

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in / _MTOK * self.in_per_mtok
            + tokens_out / _MTOK * self.out_per_mtok
        )


@dataclass
class CostEstimate:
    """Result of a cost computation. ``priced`` is False for unknown models."""

    model: str
    tokens_in: int
    tokens_out: int
    cost: Optional[float]        # None when unpriced
    priced: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost": self.cost,
            "priced": self.priced,
            "note": self.note,
        }


class PricingTable:
    """Model → ``ModelPrice`` loaded from ``pricing.yaml``."""

    def __init__(self, prices: Dict[str, ModelPrice], currency: str = "USD"):
        self._prices = prices
        self.currency = currency

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "PricingTable":
        data = _yaml_load(path)
        currency = data.get("currency", "USD")
        out: Dict[str, ModelPrice] = {}
        for model, rate in (data.get("models") or {}).items():
            try:
                out[str(model)] = ModelPrice(
                    model=str(model),
                    in_per_mtok=float(rate["in"]),
                    out_per_mtok=float(rate["out"]),
                )
            except (KeyError, TypeError, ValueError):
                # Skip malformed entries rather than crash the server (fail-open,
                # matching the v1 gates loader). An unparseable row just means the
                # model is treated as unpriced.
                continue
        return cls(out, currency=currency)

    @classmethod
    def default(cls, repo_root: Path) -> "PricingTable":
        return cls.load(repo_root / "tools" / "explorer" / "pricing.yaml")

    # -- lookup -------------------------------------------------------------
    def _lookup(self, model: str) -> Optional[ModelPrice]:
        """Exact match, then a prefix-stripped fallback (``Poe-`` / ``OR-``)."""
        if model in self._prices:
            return self._prices[model]
        for prefix in ("Poe-", "OR-"):
            if model.startswith(prefix):
                bare = model[len(prefix):]
                if bare in self._prices:
                    return self._prices[bare]
        return None

    def has(self, model: str) -> bool:
        return self._lookup(model) is not None

    def estimate(self, model: str, tokens_in: int, tokens_out: int) -> CostEstimate:
        """Compute a cost estimate. Unknown model -> unpriced (token-count only)."""
        price = self._lookup(model)
        if price is None:
            return CostEstimate(
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=None,
                priced=False,
                note=f"unpriced: '{model}' not in pricing.yaml — counting tokens only",
            )
        return CostEstimate(
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=price.cost(tokens_in, tokens_out),
            priced=True,
        )


def budget_estimate(
    table: PricingTable,
    model: str,
    est_tokens_in: int,
    est_tokens_out: int,
    n_calls: int = 1,
) -> CostEstimate:
    """Pre-run helper: project the cost of ``n_calls`` identical calls.

    Used by the dry-run / launch form to show a projected spend before any API
    call is made. Multiplies per-call token estimates by ``n_calls``.
    """
    return table.estimate(
        model,
        est_tokens_in * max(n_calls, 0),
        est_tokens_out * max(n_calls, 0),
    )
