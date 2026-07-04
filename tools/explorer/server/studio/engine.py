"""Engine — the async run state machine (plan §5, WS3).

Executes a FlowSpec: coordinator stages run sequentially, the ONE ``foreach: pair``
stage fans out behind a semaphore (default ``max_concurrency`` 6), then run-level
portfolio validators run. Every step goes through the same loop:

    resolve (WS2 resolver)  ->  provider call (WS1, streaming, usage captured)
    ->  write artifact under its CANONICAL name  ->  run the step's gates
        (validate_step.py --gates <run-local>  + check_human_subjects.py where
         the step carries g.human_subject)
    ->  route on the result.

``on_fail`` policies (plan §5):
  * ``block``           — hard-fail the run at this step.
  * ``flag``            — record an amber chip and CONTINUE (advisory gate).
  * ``retry``           — re-call with a repair prompt built from the validator's
                          stderr, honoring validate_with_retries.py semantics +
                          its ``.repair_attempt_N.md`` convention, max 3. Exhausted
                          => QUARANTINE the pair.
  * ``quarantine_pair`` — quarantine immediately on hard fail (the step-11 Andon).

QUARANTINE IS LOUD (S6): before ANY portfolio stage, if pairs broke open the
engine emits the doctrine line — ``"N of <n_pairs> pairs broke open at step X
(gate: <name>)"`` — into the runlog and the run record. Never a silent 5-shipped-
as-6.

HOLD_FOR_HUMAN (S7): ``check_human_subjects.py`` returns exit 2 (HOLD) on a
name+crime / identifying-tuple lyric — the engine pauses the pair with a blocking
event and waits for a human ``decision`` (proceed / kill). REAL GRIEF IS NOT RAW
MATERIAL is enforced in the lab too.

BUDGET HARD-STOP (S4): a pre-run estimate seeds the meter; the live meter
accumulates actual usage; crossing the cap CANCELS remaining calls loudly,
finishes in-flight artifacts, and leaves the run resumable.

RESUME (S5): disk is authority (rebuild_manifest doctrine). On resume the engine
re-scans the run dir, trusts valid artifacts, and restarts at the first missing/
failed step per lane — an interrupted mock run resumes to a byte-identical set.

DUMB BACKEND: the engine never reimplements a gate. It shells out to the repo's
own ``scripts/validate_*.py`` with the run-local ``--gates`` file.

The LLM-judge QA (lofn-qa) is available as an OPTIONAL experimental flow step,
OFF by default, behind ``experimental_qa`` (plan decision §13.2).
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .compile import _fanout_count
from .flowspec import FlowSpec, find_repo_root, sha256_text
from .knobs import KnobSet, load_base_gates, materialize_gates
from .pricing import PricingTable
from .providers import ProviderClient, ProviderRequest, ProviderError, ProviderResult
from .resolver import ResolvedPrompt, resolve, find_step, step_is_fanout
from .runstore import RunStore, new_step_result, resolve_artifact_name, pair_label


# ---------------------------------------------------------------------------
# Gate → validator routing (plan §5, pipeline_map.yaml `gates`).
# ---------------------------------------------------------------------------

# Gates enforced by validate_step.py (one shell-out per step enforces all of them).
_VALIDATE_STEP_GATES = {
    "g.step00", "g.music_prompt", "g.music_lyrics", "g.music_house_lexicon",
    "g.music_one_fact", "g.image_scene", "g.collapse",
}
# Gate enforced by check_human_subjects.py (exit 2 == HOLD_FOR_HUMAN).
_HUMAN_SUBJECT_GATE = "g.human_subject"
# Gates with validator: null — prose/advisory, recorded but not independently shelled.
_ADVISORY_GATES = {"g.tree_branches", "g.pairs_distinct", "g.andon"}
# Run-level portfolio gate.
_PORTFOLIO_GATE = "g.portfolio"


class RunCancelled(Exception):
    """Raised to unwind the run when cancelled or budget-aborted."""


@dataclass
class GateOutcome:
    """One gate's result (schema: stepResult.gate_results). ``ok`` None == FLAG."""

    gate: str
    ok: Optional[bool]
    detail: str = ""

    def as_dict(self) -> dict:
        return {"gate": self.gate, "pass": self.ok, "detail": self.detail[:1000]}


@dataclass
class StepValidation:
    """The combined outcome of running a step's gates."""

    gates: List[GateOutcome] = field(default_factory=list)
    hard_fail: bool = False            # a blocking validator returned non-zero
    hold_for_human: bool = False       # check_human_subjects returned HOLD (exit 2)
    stderr: str = ""                   # validator stderr, for the repair prompt

    @property
    def any_flag(self) -> bool:
        return any(g.ok is None for g in self.gates)


# ---------------------------------------------------------------------------
# The run controller.
# ---------------------------------------------------------------------------

@dataclass
class RunHandle:
    """Live, in-process control surface for one run (cancel/resume/decision).

    The API layer keeps a registry of these keyed by run id. HOLD decisions and
    cancellation flow through the asyncio primitives here.
    """

    run_id: str
    store: RunStore
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    # pair label -> (Event, decision_holder) for HOLD_FOR_HUMAN resolution.
    _holds: Dict[str, "asyncio.Event"] = field(default_factory=dict)
    _decisions: Dict[str, str] = field(default_factory=dict)
    task: Optional[asyncio.Task] = None

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def decide(self, pair: Optional[str], decision: str) -> bool:
        """Resolve a HOLD (proceed | kill). Returns True if a hold was waiting."""
        key = pair or "_run"
        self._decisions[key] = decision
        ev = self._holds.get(key)
        if ev is not None:
            ev.set()
            return True
        return False

    def _hold_event(self, pair: Optional[str]) -> "asyncio.Event":
        key = pair or "_run"
        ev = self._holds.get(key)
        if ev is None:
            ev = asyncio.Event()
            self._holds[key] = ev
        return ev


class Engine:
    """Executes one run to completion (or cancel / budget-abort / failure)."""

    def __init__(
        self,
        *,
        flow: FlowSpec,
        knobs: KnobSet,
        context: Dict[str, str],
        store: RunStore,
        handle: RunHandle,
        client_factory: Callable[[str], ProviderClient],
        pricing: PricingTable,
        repo_root: Path,
        base_gates: Optional[dict] = None,
        dry_run: bool = False,
        experimental_qa: bool = False,
        max_concurrency: Optional[int] = None,
        provider_override: Optional[str] = None,
    ):
        self.flow = flow
        self.knobs = knobs
        self.context = context
        self.store = store
        self.handle = handle
        self.client_factory = client_factory
        # When set (e.g. the $0 mock test suite), EVERY provider call routes to this
        # provider (the mock dialect) regardless of the flow's tier provider. The
        # model on the tier is preserved so fixtures/pricing key correctly.
        self.provider_override = provider_override
        self.pricing = pricing
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.experimental_qa = experimental_qa
        self.base_gates = base_gates if base_gates is not None else load_base_gates(repo_root)
        # live meter
        self._spend = 0.0
        self._tokens_in = 0
        self._tokens_out = 0
        self._t0 = 0.0
        self._quarantined: List[Tuple[str, str, str]] = []  # (pair, step, gate)
        self._budget_aborted = False
        # concurrency
        n = max_concurrency if max_concurrency is not None else knobs.resolved.get("max_concurrency", 6)
        try:
            self._max_conc = max(1, int(n))
        except (TypeError, ValueError):
            self._max_conc = 6
        self._n_pairs = _fanout_count(knobs)
        # run-local gates file (materialized once; validators read via --gates)
        self._gates_path: Optional[Path] = None

    # -- lifecycle ----------------------------------------------------------
    async def run(self) -> Dict[str, Any]:
        """Execute the run. Returns the final (redacted) run.json."""
        self._t0 = time.monotonic()
        self.store.set_status("running")
        self.store.log({"event": "run_start", "flow": self.flow.name,
                        "version": self.flow.version, "n_pairs": self._n_pairs,
                        "dry_run": self.dry_run, "max_concurrency": self._max_conc})
        # Materialize the run-local knob-adjusted gates.yaml the validators read.
        if not self.dry_run:
            self._gates_path = materialize_gates(self.store.run_dir, self.base_gates, self.knobs)
        try:
            await self._run_stages()
        except RunCancelled:
            status = "cancelled"
            self.store.log({"event": "run_cancelled",
                            "budget_aborted": self._budget_aborted})
            self._finalize(status)
            return self.store.run()
        except Exception as e:  # pragma: no cover - defensive
            self.store.log({"event": "run_error", "error": f"{type(e).__name__}: {e}"})
            self._finalize("failed")
            return self.store.run()

        self._finalize("complete")
        return self.store.run()

    def _finalize(self, status: str) -> None:
        self.store.set_totals(
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            cost_usd=round(self._spend, 6),
            duration_s=round(time.monotonic() - self._t0, 3),
            pairs_total=self._n_pairs,
            pairs_quarantined=len({p for (p, _, _) in self._quarantined}),
        )
        self.store.set_budget_spend(self._spend, aborted_on_cap=self._budget_aborted)
        self.store.set_status(status)
        self.store.log({"event": "run_end", "status": status,
                        "spend_usd": round(self._spend, 6),
                        "pairs_quarantined": len({p for (p, _, _) in self._quarantined})})

    # -- stage orchestration ------------------------------------------------
    async def _run_stages(self) -> None:
        for stage in self.flow.data["stages"]:
            self._check_cancelled()
            if stage.get("foreach") == "pair":
                await self._run_fanout_stage(stage)
                # LOUD quarantine banner BEFORE any portfolio stage (S6).
                self._emit_quarantine_banner()
            elif stage.get("id") == "portfolio":
                await self._run_portfolio_stage(stage)
            else:
                await self._run_sequential_stage(stage, pair=None)

    async def _run_sequential_stage(self, stage: dict, pair: Optional[int]) -> None:
        for step in stage.get("steps", []):
            self._check_cancelled()
            await self._execute_step(stage, step, pair=pair)

    async def _run_fanout_stage(self, stage: dict) -> None:
        sem = asyncio.Semaphore(self._max_conc)

        async def one_pair(p: int) -> None:
            async with sem:
                await self._run_pair_lane(stage, p)

        await asyncio.gather(*(one_pair(p) for p in range(1, self._n_pairs + 1)))

    async def _run_pair_lane(self, stage: dict, pair: int) -> None:
        """Run every step of the fan-out for one pair; stop this lane on quarantine."""
        lab = pair_label(pair)
        self.store.log({"event": "pair_start", "pair": lab})
        for step in stage.get("steps", []):
            self._check_cancelled()
            quarantined = await self._execute_step(stage, step, pair=pair)
            if quarantined:
                self.store.log({"event": "pair_quarantined", "pair": lab, "step": step["id"]})
                return
        self.store.log({"event": "pair_done", "pair": lab})

    # -- the per-step loop --------------------------------------------------
    async def _execute_step(self, stage: dict, step: dict, *, pair: Optional[int]) -> bool:
        """Run one step. Returns True if the pair was quarantined (stop the lane)."""
        if step.get("kind") == "validator":
            # Portfolio-style pure validator steps run in the portfolio stage; a
            # validator step inside a normal stage is a no-op here.
            return False

        step_id = step["id"]
        lab = pair_label(pair) if step_is_fanout(self.flow, step_id) else None
        stage_id = stage.get("id")
        tier = (self.flow.data.get("model_tiers", {}) or {}).get(step.get("tier"), {})
        provider = self.provider_override or tier.get("provider", "mock")
        model = tier.get("model", "")

        rec = new_step_result(stage=stage_id, step=step_id, pair=lab,
                              provider=provider, model=model)
        rec["status"] = "resolving"
        rec["attempts"] = 0
        self.store.upsert_step(rec)
        self._log_step(step_id, lab, "resolving")

        # RESUME: trust a valid on-disk artifact (disk is authority).
        artifact_name = resolve_artifact_name(step.get("emits"), pair)
        if artifact_name and not self.dry_run and self.store.artifact_exists(artifact_name):
            if self._artifact_is_valid(step, artifact_name, pair):
                rec["status"] = "done"
                rec["artifact"] = artifact_name
                rec["artifact_sha"] = self.store.artifact_sha(artifact_name)
                rec["attempts"] = rec.get("attempts", 0) or 1
                self.store.upsert_step(rec)
                self._log_step(step_id, lab, "done", note="resumed (valid on disk)")
                return False

        # Resolve the prompt (the ONE path — S3 byte-equality).
        resolved = resolve(self.flow, self.knobs, self.context, step_id,
                           pair=pair, repo_root=self.repo_root)

        if self.dry_run:
            # Compile-all only: record the resolution, make NO API call.
            rec["status"] = "done"
            rec["artifact"] = None
            self.store.upsert_step(rec)
            self.store.log({"event": "step_dry_run", "step": step_id, "pair": lab,
                            "text_sha": resolved.text_sha,
                            "est_tokens_in": resolved.meta.get("est_tokens_in")})
            return False

        # The retry loop (validate_with_retries semantics; max_attempts from on_fail).
        on_fail = step.get("on_fail") or {"policy": "block"}
        policy = on_fail.get("policy", "block")
        max_attempts = int(on_fail.get("max_attempts", 3)) if policy == "retry" else 1

        repair_note: Optional[str] = None
        attempt = 0
        while True:
            attempt += 1
            rec["attempts"] = attempt
            rec["status"] = "calling"
            self.store.upsert_step(rec)
            self._log_step(step_id, lab, "calling", attempt=attempt)

            # Budget hard-stop check BEFORE spending (pre-call estimate).
            self._preflight_budget(resolved, tier)

            result = await self._call_provider(
                provider, model, resolved, tier, step_id, lab, repair_note
            )
            self._meter(result, model)

            # Write the artifact under its canonical name.
            if artifact_name:
                sha = self.store.write_artifact(artifact_name, result.text)
                rec["artifact"] = artifact_name
                rec["artifact_sha"] = sha
            rec["status"] = "validating"
            self.store.upsert_step(rec)
            self._log_step(step_id, lab, "validating")

            # Gates.
            validation = self._run_gates(step, artifact_name, pair)
            rec["gate_results"] = [g.as_dict() for g in validation.gates]

            # HOLD_FOR_HUMAN takes precedence — pause and wait for a decision.
            if validation.hold_for_human:
                rec["status"] = "held_for_human"
                self.store.upsert_step(rec)
                decision = await self._await_human(step_id, lab, validation)
                if decision == "kill":
                    self._quarantine(lab, step_id, _HUMAN_SUBJECT_GATE, rec,
                                     detail="human decided KILL on HOLD_FOR_HUMAN")
                    return True
                # proceed: treat the hold as cleared, continue as pass.
                validation.hold_for_human = False

            if not validation.hard_fail:
                # pass (possibly with amber flags -> flagged, else done)
                rec["status"] = "flagged" if validation.any_flag else "done"
                self.store.upsert_step(rec)
                self._log_step(step_id, lab, rec["status"],
                               flags=[g.gate for g in validation.gates if g.ok is None])
                return False

            # HARD FAIL — route by policy.
            if policy == "flag":
                rec["status"] = "flagged"
                self.store.upsert_step(rec)
                self._log_step(step_id, lab, "flagged", note="hard fail downgraded by flag policy")
                return False

            if policy == "quarantine_pair":
                self._quarantine(lab, step_id, _first_failed_gate(validation), rec,
                                 detail=validation.stderr)
                return True

            if policy == "retry" and attempt < max_attempts:
                # Build the repair prompt from the validator's stderr and write the
                # .repair_attempt_N.md convention file, then re-call.
                repair_note = self._write_repair(step_id, artifact_name, attempt,
                                                 max_attempts, validation.stderr, pair)
                rec["status"] = "retrying"
                self.store.upsert_step(rec)
                self._log_step(step_id, lab, "retrying", attempt=attempt,
                               gate=_first_failed_gate(validation))
                continue

            # block, or retries exhausted -> quarantine the pair (or fail the run
            # for a coordinator step, which has no pair to quarantine).
            gate = _first_failed_gate(validation)
            if pair is not None:
                self._quarantine(lab, step_id, gate, rec, detail=validation.stderr)
                return True
            # Coordinator step hard-fail with no pair: fail the whole run loudly.
            rec["status"] = "failed"
            self.store.upsert_step(rec)
            self.store.log({"event": "coordinator_step_failed", "step": step_id,
                            "gate": gate, "detail": validation.stderr[:2000]})
            raise RunCancelled(f"coordinator step {step_id} failed at gate {gate}")

    # -- provider call ------------------------------------------------------
    async def _call_provider(
        self, provider: str, model: str, resolved: ResolvedPrompt, tier: dict,
        step_id: str, lab: Optional[str], repair_note: Optional[str],
    ) -> ProviderResult:
        client = self.client_factory(provider)
        user = resolved.user
        if repair_note:
            user = user + "\n\n" + repair_note
        req = ProviderRequest(
            model=model,
            system=resolved.system,
            messages=[{"role": "user", "content": user}],
            max_tokens=int(tier.get("max_tokens", 4096) or 4096),
            stream=True,
            params=dict(tier.get("params", {}) or {}),
        )
        final: Optional[ProviderResult] = None
        try:
            async for chunk in client.complete(req):
                if chunk.final is not None:
                    final = chunk.final
                elif chunk.text:
                    # stream deltas onto the runlog sparingly (usage is the event
                    # SSE cares about; a token-firehose would bloat the log).
                    pass
        except ProviderError as e:
            # A provider error is a transport/gate-neutral failure — surface it.
            raise RunCancelled(f"provider error on step {step_id}: {e}") from e
        if final is None:
            raise RunCancelled(f"provider returned no final result on step {step_id}")
        return final

    # -- metering + budget --------------------------------------------------
    def _meter(self, result: ProviderResult, model: str) -> None:
        self._tokens_in += result.usage.in_
        self._tokens_out += result.usage.out
        # Prefer the provider-reported actual cost (OpenRouter); else price it.
        if result.cost is not None:
            self._spend += float(result.cost)
        else:
            est = self.pricing.estimate(model, result.usage.in_, result.usage.out)
            if est.priced and est.cost is not None:
                self._spend += est.cost
            # unpriced: count tokens only, loudly (once per model would be ideal;
            # the runlog note here is enough for the meter).
        self.store.set_totals(tokens_in=self._tokens_in, tokens_out=self._tokens_out,
                              cost_usd=round(self._spend, 6))
        self.store.set_budget_spend(self._spend)
        # Live cap check AFTER accounting the just-finished call.
        cap = self.store.budget_cap()
        if cap is not None and self._spend > cap:
            self._budget_aborted = True
            self.store.log({"event": "budget_exceeded", "spend_usd": round(self._spend, 6),
                            "cap_usd": cap,
                            "message": f"budget cap ${cap} crossed at ${self._spend:.4f}; "
                                       "cancelling remaining calls loudly. Run is resumable."})
            raise RunCancelled("budget cap exceeded")

    def _preflight_budget(self, resolved: ResolvedPrompt, tier: dict) -> None:
        """Refuse to START a call that the estimate says would cross the cap.

        Uses the chars/4 input estimate + the tier's max_tokens output ceiling. A
        crossing cancels loudly BEFORE spending (finishes in-flight artifacts,
        leaves the run resumable — S4)."""
        cap = self.store.budget_cap()
        if cap is None:
            return
        model = tier.get("model", "")
        est_in = int(resolved.meta.get("est_tokens_in", 0))
        out_ceiling = int(tier.get("max_tokens", 0) or 0)
        est = self.pricing.estimate(model, est_in, out_ceiling)
        projected = self._spend + (est.cost or 0.0)
        if est.priced and projected > cap:
            self._budget_aborted = True
            self.store.log({"event": "budget_preflight_abort",
                            "spend_usd": round(self._spend, 6), "cap_usd": cap,
                            "projected_usd": round(projected, 6),
                            "message": f"next call projected to ${projected:.4f} would cross "
                                       f"the ${cap} cap; cancelling loudly before spending. "
                                       "Run is resumable."})
            raise RunCancelled("budget cap would be exceeded by next call")

    # -- gate execution (dumb backend: shell out to canon validators) -------
    def _run_gates(self, step: dict, artifact_name: Optional[str],
                   pair: Optional[int]) -> StepValidation:
        gates = list(step.get("gates", []))
        out = StepValidation()
        if not artifact_name:
            return out
        artifact_path = self.store.artifact_path(artifact_name)

        # 1) validate_step.py enforces every validate_step-family gate in one call.
        vs_gates = [g for g in gates if g in _VALIDATE_STEP_GATES]
        if vs_gates:
            rc, so, se = self._run_validate_step(step["id"], artifact_path)
            passed = rc == 0
            for g in vs_gates:
                out.gates.append(GateOutcome(gate=g, ok=passed,
                                             detail="" if passed else _tail(se or so)))
            if not passed:
                out.hard_fail = True
                out.stderr = (se or so).strip()

        # 2) check_human_subjects.py — exit 2 covers BOTH HOLD_FOR_HUMAN and the
        #    softer ONLINE_CHECK_REQUIRED. We parse the JSON `recommendation`:
        #      HOLD_FOR_HUMAN (minor-as-victim / identifying-tuple / name+crime)
        #        -> pause the pair, block until a human decides (S7).
        #      ONLINE_CHECK_REQUIRED (a bare person name, no harm tuple)
        #        -> amber FLAG, continue (the recent-news check is a human follow-up,
        #           not a lab-blocking event).
        if _HUMAN_SUBJECT_GATE in gates:
            rc, so, se = self._run_human_subjects(artifact_path)
            recommendation = _hs_recommendation(so, rc)
            if recommendation == "HOLD_FOR_HUMAN":
                out.hold_for_human = True
                out.gates.append(GateOutcome(gate=_HUMAN_SUBJECT_GATE, ok=False,
                                             detail="HOLD_FOR_HUMAN: " + _tail(so)))
            elif recommendation == "ONLINE_CHECK_REQUIRED":
                out.gates.append(GateOutcome(gate=_HUMAN_SUBJECT_GATE, ok=None,
                                             detail="ONLINE_CHECK_REQUIRED (person name present): "
                                                    + _tail(so)))
            elif rc == 0 or recommendation == "PASS_TO_NEXT_GATE":
                out.gates.append(GateOutcome(gate=_HUMAN_SUBJECT_GATE, ok=True))
            else:
                # fail-open: an unparseable/errored checker is a FLAG, not a hard block
                out.gates.append(GateOutcome(gate=_HUMAN_SUBJECT_GATE, ok=None,
                                             detail="checker error (fail-open): " + _tail(se or so)))

        # 3) advisory/prose gates — recorded as FLAG (amber), never independently fail.
        for g in gates:
            if g in _ADVISORY_GATES:
                out.gates.append(GateOutcome(gate=g, ok=None, detail="advisory/prose gate"))

        return out

    def _artifact_is_valid(self, step: dict, artifact_name: str, pair: Optional[int]) -> bool:
        """For resume: does the on-disk artifact pass its blocking gates?"""
        v = self._run_gates(step, artifact_name, pair)
        return not v.hard_fail and not v.hold_for_human

    def _run_validate_step(self, step_id: str, artifact_path: Path) -> Tuple[int, str, str]:
        cmd = [sys.executable, str(self.repo_root / "scripts" / "validate_step.py")]
        if self._gates_path is not None:
            cmd += ["--gates", str(self._gates_path)]
        cmd += [str(step_id), str(artifact_path)]
        return _run_subprocess(cmd, self.repo_root)

    def _run_human_subjects(self, artifact_path: Path) -> Tuple[int, str, str]:
        cmd = [sys.executable, str(self.repo_root / "scripts" / "check_human_subjects.py"),
               str(artifact_path)]
        return _run_subprocess(cmd, self.repo_root)

    # -- portfolio stage ----------------------------------------------------
    async def _run_portfolio_stage(self, stage: dict) -> None:
        if self.dry_run:
            self.store.log({"event": "portfolio_skipped", "reason": "dry_run"})
            return
        for step in stage.get("steps", []):
            run_script = step.get("run")
            gates = list(step.get("gates", []))
            if not run_script:
                continue
            cmd = [sys.executable, str(self.repo_root / run_script), str(self.store.run_dir)]
            rc, so, se = _run_subprocess(cmd, self.repo_root)
            ok = rc == 0
            self.store.log({"event": "portfolio_gate", "step": step.get("id"),
                            "gate": gates[0] if gates else _PORTFOLIO_GATE,
                            "pass": ok, "detail": _tail(se or so)})
            prec = new_step_result(stage="portfolio", step=step.get("id"), pair=None,
                                   status="done" if ok else "failed")
            prec["gate_results"] = [{"gate": (gates[0] if gates else _PORTFOLIO_GATE),
                                     "pass": ok, "detail": _tail(se or so)}]
            self.store.upsert_step(prec)

        # Optional experimental LLM-judge QA (lofn-qa) — off by default.
        if self.experimental_qa:
            await self._run_experimental_qa()

    async def _run_experimental_qa(self) -> None:
        """OPTIONAL experimental flow step (plan §13.2): run the lofn-qa prompts
        via the API as an LLM judge. Off by default; recorded as a verdict, never
        gating. Uses the coordinator tier's model for the judge call."""
        self.store.log({"event": "experimental_qa_start"})
        tier = (self.flow.data.get("model_tiers", {}) or {}).get("coordinator", {})
        provider = self.provider_override or tier.get("provider", "mock")
        model = tier.get("model", "")
        # Assemble a compact judge prompt over the run's final artifacts.
        artifacts = sorted(p.name for p in self.store.run_dir.glob("*step*.md"))
        judge_prompt = (
            "You are lofn-qa, the adversarial auditor. Given the run's final "
            "artifacts listed below, return a one-line SHIP / REPAIR / FAIL verdict "
            "with a one-sentence reason.\n\nArtifacts: " + ", ".join(artifacts)
        )
        req = ProviderRequest(model=model, messages=[{"role": "user", "content": judge_prompt}],
                              max_tokens=int(tier.get("max_tokens", 4096) or 4096), stream=True)
        try:
            client = self.client_factory(provider)
            final = None
            async for chunk in client.complete(req):
                if chunk.final is not None:
                    final = chunk.final
            verdict_text = final.text if final else ""
            self._meter(final, model) if final else None
        except (ProviderError, RunCancelled) as e:
            verdict_text = f"(experimental QA unavailable: {e})"
        self.store.append_verdict({"kind": "experimental_qa", "model": model,
                                   "verdict": verdict_text[:2000], "ts": _now()})
        self.store.log({"event": "experimental_qa_done"})

    # -- quarantine + human hold + repair ----------------------------------
    def _quarantine(self, lab: Optional[str], step_id: str, gate: str,
                    rec: dict, *, detail: str = "") -> None:
        self._quarantined.append((lab or "??", step_id, gate))
        rec["status"] = "quarantined"
        self.store.upsert_step(rec)
        self.store.log({"event": "quarantine", "pair": lab, "step": step_id,
                        "gate": gate, "detail": _tail(detail)})

    def _emit_quarantine_banner(self) -> None:
        """The LOUD doctrine line before any portfolio stage (S6)."""
        pairs = {p for (p, _, _) in self._quarantined}
        if not pairs:
            return
        # Group by the step/gate where each pair broke.
        by_pair = {}
        for (p, s, g) in self._quarantined:
            by_pair.setdefault(p, (s, g))
        # Emit one banner per distinct break point (usually one).
        parts = []
        for p, (s, g) in sorted(by_pair.items()):
            parts.append(f"pair {p} at step {s} (gate: {g})")
        n = len(pairs)
        banner = (f"{n} of {self._n_pairs} pairs broke open: " + "; ".join(parts))
        self.store.log({"event": "quarantine_banner", "message": banner,
                        "pairs_broken": n, "pairs_total": self._n_pairs})
        self.store.update_run(quarantine_banner=banner)

    async def _await_human(self, step_id: str, lab: Optional[str],
                           validation: StepValidation) -> str:
        """Block the pair until a human decides proceed/kill (S7)."""
        self.store.log({"event": "hold_for_human", "step": step_id, "pair": lab,
                        "gate": _HUMAN_SUBJECT_GATE,
                        "detail": _tail(next((g.detail for g in validation.gates
                                              if g.gate == _HUMAN_SUBJECT_GATE), "")),
                        "message": "HOLD_FOR_HUMAN: a human must click proceed or kill. "
                                   "REAL GRIEF IS NOT RAW MATERIAL."})
        ev = self.handle._hold_event(lab)
        ev.clear()
        # Wait for either a decision or a cancel.
        cancel_wait = asyncio.create_task(self.handle.cancel_event.wait())
        hold_wait = asyncio.create_task(ev.wait())
        done, pending = await asyncio.wait(
            {cancel_wait, hold_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
        if self.handle.cancel_event.is_set():
            raise RunCancelled("cancelled while holding for human")
        decision = self.handle._decisions.get(lab or "_run", "kill")
        self.store.log({"event": "hold_resolved", "step": step_id, "pair": lab,
                        "decision": decision})
        return decision

    def _write_repair(self, step_id: str, artifact_name: Optional[str], attempt: int,
                      max_attempts: int, stderr: str, pair: Optional[int]) -> str:
        """Write the ``.repair_attempt_N.md`` file (validate_with_retries convention)
        and return the repair note to append to the next call's prompt."""
        note = (
            "# Lofn Step Repair Required\n\n"
            f"Step: `{str(step_id).zfill(2)}`  Attempt: {attempt}/{max_attempts}\n\n"
            "## Validator failure\n\n```\n" + stderr.strip() + "\n```\n\n"
            "## Repair instructions\n\n"
            "Revise the artifact in place. Preserve valid creative content. Fix ONLY "
            "the failing structural/content requirements above. Return the COMPLETE "
            "corrected artifact.\n"
        )
        if artifact_name:
            repair_name = f"{artifact_name}.repair_attempt_{attempt}.md"
            try:
                self.store.write_artifact(repair_name, note)
            except ValueError:
                pass
        return note

    # -- helpers ------------------------------------------------------------
    def _check_cancelled(self) -> None:
        if self.handle.cancel_event.is_set():
            raise RunCancelled("cancelled by request")

    def _log_step(self, step_id: str, lab: Optional[str], status: str, **extra) -> None:
        ev = {"event": "step", "step": step_id, "pair": lab, "status": status}
        ev.update({k: v for k, v in extra.items() if v not in (None, [], "")})
        self.store.log(ev)


# ---------------------------------------------------------------------------
# Subprocess + small text helpers.
# ---------------------------------------------------------------------------

def _run_subprocess(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    """Run a validator; return (rc, stdout, stderr). Never raises on non-zero."""
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True,
                              timeout=120)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", "validator timed out"
    except Exception as e:  # pragma: no cover
        return 125, "", f"validator launch error: {type(e).__name__}: {e}"


def _tail(s: str, n: int = 1500) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else "…" + s[-n:]


def _hs_recommendation(stdout: str, rc: int) -> str:
    """Parse check_human_subjects.py JSON stdout for its ``recommendation``.

    Distinguishes the true HOLD (minor-as-victim / identifying-tuple / name+crime)
    from the softer ONLINE_CHECK_REQUIRED (a bare person name). Falls back to the
    exit code if the JSON can't be parsed (rc 2 -> treat as HOLD, fail-safe)."""
    try:
        data = json.loads(stdout)
        rec = data.get("recommendation")
        if rec in ("HOLD_FOR_HUMAN", "ONLINE_CHECK_REQUIRED", "PASS_TO_NEXT_GATE"):
            return rec
    except Exception:
        pass
    if rc == 2:
        return "HOLD_FOR_HUMAN"
    if rc == 0:
        return "PASS_TO_NEXT_GATE"
    return "ERROR"


def _first_failed_gate(v: StepValidation) -> str:
    for g in v.gates:
        if g.ok is False:
            return g.gate
    return "g.unknown"


def _now() -> str:
    from .runstore import utcnow_iso
    return utcnow_iso()
