"""Compile — the zero-cost preview that /compile and the engine share (plan §6, §9).

``compile_flow`` resolves a flow's steps against a knob preset + context and
returns exactly what the engine will send, plus per-step diffs and a token/cost
estimate (via WS1 pricing). It is a THIN orchestration layer over the single
resolution path in :mod:`resolver` — it never assembles a prompt itself, so
acceptance gate **S3** (Compile ≡ execute) holds by construction.

Two diff modes (plan §8.2 "feel the knob"):
  * against a *baseline* knob preset — the signature interaction: turn a knob and
    watch the resolved prompt move. ``compile_flow(..., baseline=...)`` returns a
    per-step unified diff of the resolved user prompt vs the baseline resolution.
  * no baseline — diffs are empty; the response is just the resolved prompts +
    cost estimate.

Cost estimate: the input token estimate is the resolver's chars/4 heuristic
(labeled). Output tokens are unknown pre-run, so the estimate uses each tier's
``max_tokens`` as a worst-case output ceiling (clearly labeled ``worst_case``) —
the launch form / budget cap reads this to project a ceiling before any API call.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .flowspec import FlowSpec, find_repo_root, load_flowspec, flows_dir
from .knobs import KnobSet, load_preset
from .pricing import PricingTable
from .resolver import ResolvedPrompt, resolve, find_step, step_is_fanout


@dataclass
class StepCompile:
    """One step's compiled result."""

    step: str
    pair: Optional[int]
    resolved: ResolvedPrompt
    est_tokens_in: int
    est_tokens_out_ceiling: int
    cost_estimate: Optional[float]
    priced: bool
    cost_note: str = ""
    diff: Optional[str] = None          # unified diff vs baseline (None if no baseline)

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "pair": self.pair,
            "resolved": self.resolved.as_dict(),
            "est_tokens_in": self.est_tokens_in,
            "est_tokens_out_ceiling": self.est_tokens_out_ceiling,
            "cost_estimate": self.cost_estimate,
            "priced": self.priced,
            "cost_note": self.cost_note,
            "diff": self.diff,
        }


@dataclass
class CompileResult:
    """The /compile response payload."""

    flow: str
    flow_version: int
    knobs: dict
    steps: List[StepCompile] = field(default_factory=list)
    total_est_tokens_in: int = 0
    total_est_tokens_out_ceiling: int = 0
    total_cost_estimate: Optional[float] = None
    any_unpriced: bool = False
    invariant_violations: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "flow": self.flow,
            "flow_version": self.flow_version,
            "knobs": self.knobs,
            "steps": [s.as_dict() for s in self.steps],
            "totals": {
                "est_tokens_in": self.total_est_tokens_in,
                "est_tokens_out_ceiling": self.total_est_tokens_out_ceiling,
                "cost_estimate": self.total_cost_estimate,
                "cost_estimate_note": (
                    "worst-case: input=chars/4 heuristic, output=tier max_tokens; "
                    "some models unpriced" if self.any_unpriced
                    else "worst-case: input=chars/4 heuristic, output=tier max_tokens"
                ),
            },
            # Compile is allowed on a broken preset (plan §3.2) — we REPORT the
            # violations so the UI can block Run, but /compile still returns 200.
            "invariant_violations": self.invariant_violations,
            "run_allowed": not self.invariant_violations,
        }


def _tier_for_step(flow: FlowSpec, step_id: str) -> dict:
    stepdef = find_step(flow, step_id)
    tier = stepdef.get("tier")
    return (flow.data.get("model_tiers", {}) or {}).get(tier, {}) if tier else {}


def _fanout_count(knobs: KnobSet) -> int:
    """How many pairs the fan-out will iterate (for cost multiplication)."""
    n = knobs.resolved.get("n_pairs")
    try:
        return max(1, int(n))
    except (TypeError, ValueError):
        return 1


def compile_flow(
    flow: FlowSpec,
    knobs: KnobSet,
    context: Dict[str, str],
    pricing: PricingTable,
    *,
    step: Optional[str] = None,
    baseline: Optional[KnobSet] = None,
    repo_root: Optional[Path] = None,
) -> CompileResult:
    """Compile a flow (or one step) to resolved prompts + diffs + cost estimate.

    Args:
      flow / knobs / context: the run definition (same objects the engine uses).
      pricing: WS1 pricing table for the cost estimate.
      step: compile only this step id; None compiles every LLM step (coordinator
        steps once; each fan-out step once at pair 1 as the representative, with
        the per-pair cost multiplied by n_pairs).
      baseline: if given, each step's diff is the unified diff of the resolved
        user prompt vs the resolution under ``baseline`` knobs — the "feel the
        knob" preview.
    """
    repo_root = repo_root or find_repo_root()

    result = CompileResult(
        flow=flow.name,
        flow_version=flow.version,
        knobs=knobs.summary(),
        invariant_violations=[
            {"expr": r.expr, "detail": r.detail}
            for r in knobs.invariant_violations()
        ],
    )

    fanout_n = _fanout_count(knobs)

    step_ids = [step] if step else _llm_step_ids(flow)
    for sid in step_ids:
        is_fanout = step_is_fanout(flow, sid)
        pair = 1 if is_fanout else None

        resolved = resolve(flow, knobs, context, sid, pair=pair, repo_root=repo_root)
        est_in = int(resolved.meta.get("est_tokens_in", 0))

        tier = _tier_for_step(flow, sid)
        model = tier.get("model", "")
        out_ceiling = int(tier.get("max_tokens", 0) or 0)

        est = pricing.estimate(model, est_in, out_ceiling) if model else None
        per_call_cost = est.cost if est else None
        priced = bool(est and est.priced)
        note = est.note if est else "no model on tier"

        # Fan-out steps run once PER pair — multiply the cost for the total.
        multiplier = fanout_n if is_fanout else 1
        step_cost = (per_call_cost * multiplier) if per_call_cost is not None else None

        diff = None
        if baseline is not None:
            base_resolved = resolve(
                flow, baseline, context, sid, pair=pair, repo_root=repo_root
            )
            diff = _unified_diff(
                base_resolved.user, resolved.user,
                f"{sid} (baseline)", f"{sid} (current)",
            )

        sc = StepCompile(
            step=sid,
            pair=pair,
            resolved=resolved,
            est_tokens_in=est_in * multiplier,
            est_tokens_out_ceiling=out_ceiling * multiplier,
            cost_estimate=step_cost,
            priced=priced,
            cost_note=note,
            diff=diff,
        )
        result.steps.append(sc)

        result.total_est_tokens_in += sc.est_tokens_in
        result.total_est_tokens_out_ceiling += sc.est_tokens_out_ceiling
        if step_cost is None:
            result.any_unpriced = True
        else:
            result.total_cost_estimate = (result.total_cost_estimate or 0.0) + step_cost

    return result


def _llm_step_ids(flow: FlowSpec) -> List[str]:
    ids: List[str] = []
    for stage in flow.data["stages"]:
        for s in stage.get("steps", []):
            if s.get("kind") == "validator":
                continue
            ids.append(s["id"])
    return ids


def _unified_diff(a: str, b: str, a_label: str, b_label: str) -> str:
    diff = difflib.unified_diff(
        a.splitlines(keepends=True),
        b.splitlines(keepends=True),
        fromfile=a_label,
        tofile=b_label,
        n=3,
    )
    return "".join(diff)


# ---------------------------------------------------------------------------
# Loading helpers for the API (resolve flow@ver + knob preset from disk).
# ---------------------------------------------------------------------------

def _flow_path(name: str, version: int, root: Path) -> Path:
    from .flowspec import flow_filename

    # Prefer the exact @<version>.flow.yaml (no suffix); fall back to any match.
    exact = flows_dir(root) / flow_filename(name, version)
    if exact.exists():
        return exact
    matches = sorted(flows_dir(root).glob(f"{name}@{version}*.flow.yaml"))
    if not matches:
        raise FileNotFoundError(f"flow {name}@{version} not found")
    return matches[0]


def load_flow_and_knobs(
    flow_name: str,
    flow_version: int,
    knobs_spec,
    repo_root: Path,
) -> tuple[FlowSpec, KnobSet]:
    """Load a flow@ver and a KnobSet from a request.

    ``knobs_spec`` is either a preset name (str) or an inline knobs dict (the UI
    sends the live edited knobs). A None ``knobs_spec`` falls back to the flow's
    declared ``knobs_defaults`` preset.
    """
    flow = load_flowspec(_flow_path(flow_name, flow_version, repo_root))

    if knobs_spec is None:
        preset_ref = flow.data.get("knobs_defaults", "")
        preset_name = Path(preset_ref).name if preset_ref else "classic-6x4"
        knobs = load_preset(flows_dir(repo_root), preset_name)
    elif isinstance(knobs_spec, str):
        knobs = load_preset(flows_dir(repo_root), knobs_spec)
    elif isinstance(knobs_spec, dict):
        # Accept EITHER the full preset shape ({"knobs": {...}, "invariants": [...]})
        # or a bare knob map ({"n_pairs": {...}, ...}) the UI sends inline.
        if "knobs" in knobs_spec and isinstance(knobs_spec["knobs"], dict):
            knobs = KnobSet.from_dict(knobs_spec)
        else:
            knobs = KnobSet.from_dict({"knobs": knobs_spec})
    else:
        raise ValueError(f"unsupported knobs spec type {type(knobs_spec).__name__}")

    return flow, knobs
