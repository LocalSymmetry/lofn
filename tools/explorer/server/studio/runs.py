"""Run service — launch / list / get / cancel / resume / decide (plan §5, §9).

This is the layer between the FastAPI endpoints (:mod:`api`) and the async
:class:`Engine`. It:

  * builds the initial ``run.json`` provenance from a launch request,
  * creates the :class:`RunStore` under ``output/studio/<run-id>/``,
  * builds the provider ``client_factory`` (mock replays fixtures; a real provider
    resolves its key from :class:`KeyResolver`),
  * launches the engine as a background asyncio task and tracks a :class:`RunHandle`,
  * resumes a run from disk (authority) after a kill or a budget abort,
  * routes HOLD_FOR_HUMAN decisions.

CANON IMMUTABILITY (S1): every write lives under ``output/studio/**`` (engine-
owned) or the explicitly-saved ``flows/**``. KEYS (S2): only the resolved key
strings are handed to the store as ``secrets`` for exact-literal scrubbing; they
never enter run.json / runlog / SSE / any response.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compile import load_flow_and_knobs
from .engine import Engine, RunHandle
from .flowspec import FlowSpec, find_repo_root, sha256_text, run_dir_root, load_flowspec
from .keys import KeyResolver
from .knobs import KnobSet, load_base_gates
from .pricing import PricingTable
from .providers import ProviderClient
from .runstore import RunStore, build_initial_run_json, utcnow_iso


# ---------------------------------------------------------------------------
# Run id + git provenance.
# ---------------------------------------------------------------------------

def new_run_id() -> str:
    """A sortable-ish, collision-resistant run id (also the dir name)."""
    stamp = utcnow_iso().replace(":", "").replace("-", "").replace("T", "-").rstrip("Z")
    return f"run-{stamp}-{uuid.uuid4().hex[:8]}"


def _git_head_sha(root: Path) -> str:
    from .flowspec import _git_head_sha as head
    return head(root)


def _git_dirty(root: Path) -> bool:
    """Best-effort working-tree dirty check (never raises; assumes clean on error)."""
    import subprocess
    try:
        proc = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                              text=True, capture_output=True, timeout=15)
        return bool(proc.stdout.strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Context builder — the verbatim-injection ICB pieces + their provenance refs.
# ---------------------------------------------------------------------------

def _sha(text: str) -> str:
    return sha256_text(text)


def build_context_and_refs(
    body_context: Dict[str, Any],
) -> tuple[Dict[str, str], Dict[str, Any]]:
    """Split a launch request's context into (slot→value, provenance refs).

    The launch form sends the resolved ICB slot values directly (seed text,
    personality YAML, panel text, brief). We record a ref (sha + optional name)
    per known ICB slot for run.json.context, and pass the raw values to the
    resolver for verbatim injection.
    """
    context: Dict[str, str] = {}
    for k, v in (body_context or {}).items():
        context[k] = "" if v is None else str(v)

    refs: Dict[str, Any] = {}
    for slot in ("seed", "personality", "panel", "meta_prompt"):
        if slot in context and context[slot]:
            refs[slot] = {"sha": _sha(context[slot])}
    if "input" in context:
        refs["input"] = context["input"]
    return context, refs


# ---------------------------------------------------------------------------
# The service.
# ---------------------------------------------------------------------------

@dataclass
class RunService:
    """Owns the live run registry and the launch/resume machinery."""

    repo_root: Path
    _handles: Dict[str, RunHandle] = field(default_factory=dict)

    # -- factory helpers ----------------------------------------------------
    def _runs_root(self) -> Path:
        return run_dir_root(self.repo_root)

    def _client_factory(self, keys: KeyResolver, fixtures_dir: Path):
        def factory(provider: str) -> ProviderClient:
            return ProviderClient(provider, keys, fixtures_dir=fixtures_dir)
        return factory

    def _all_secrets(self, keys: KeyResolver) -> List[str]:
        secrets: List[str] = []
        for provider in ("anthropic", "openai", "openrouter", "poe", "gemini"):
            k = keys.resolve(provider)
            if k:
                secrets.append(k)
        return secrets

    # -- launch -------------------------------------------------------------
    async def launch(
        self,
        *,
        flow_name: str,
        flow_version: int,
        knobs_spec: Any,
        context: Dict[str, Any],
        title: str,
        budget_cap_usd: float,
        dry_run: bool = False,
        experimental_qa: bool = False,
        fixtures_dir: Optional[Path] = None,
        keys: Optional[KeyResolver] = None,
        wait: bool = False,
        provider_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create + start a run. Returns the initial (redacted) run.json.

        ``wait=True`` runs to completion inline (used by the $0 mock test suite);
        otherwise the engine runs as a background task and the caller polls /runs/{id}.
        ``provider_override`` forces every call to one provider (e.g. ``"mock"`` in
        the $0 test suite) while preserving the flow's model tiers.
        """
        flow, knobs = load_flow_and_knobs(flow_name, flow_version, knobs_spec, self.repo_root)

        keys = keys or KeyResolver.from_root(self.repo_root)
        secrets = self._all_secrets(keys)
        fixtures_dir = fixtures_dir or (Path(__file__).resolve().parent / "fixtures")

        run_id = new_run_id()
        ctx, refs = build_context_and_refs(context)

        run_json = build_initial_run_json(
            run_id=run_id,
            title=title or flow_name,
            flow_name=flow.name,
            flow_version=flow.version,
            flow_file_sha=flow.file_sha(),
            knobs_resolved=dict(knobs.resolved),
            overlays=self._overlays_for(flow),
            canon_sha=_git_head_sha(self.repo_root),
            canon_dirty=_git_dirty(self.repo_root),
            context=refs,
            model_tiers=dict(flow.data.get("model_tiers", {}) or {}),
            budget_cap_usd=float(budget_cap_usd),
        )
        store = RunStore.create(self._runs_root(), run_id, run_json, secrets=secrets)
        # Persist the FULL resolved ICB context to the run dir (engine-owned, under
        # output/studio/**, NOT in run.json which stays lean/ref-only). Resume reads
        # this so a re-resolved step is byte-identical (S5) — the slots are the same
        # verbatim ICB, not empty. Secret-scrubbed on write.
        _write_context_file(store, ctx)

        handle = RunHandle(run_id=run_id, store=store)
        self._handles[run_id] = handle

        engine = Engine(
            flow=flow, knobs=knobs, context=ctx, store=store, handle=handle,
            client_factory=self._client_factory(keys, fixtures_dir),
            pricing=PricingTable.default(self.repo_root),
            repo_root=self.repo_root,
            base_gates=load_base_gates(self.repo_root),
            dry_run=dry_run, experimental_qa=experimental_qa,
            provider_override=provider_override,
        )

        if wait:
            await engine.run()
        else:
            handle.task = asyncio.create_task(engine.run())
        return store.run()

    def _overlays_for(self, flow: FlowSpec) -> List[Dict[str, str]]:
        """Record any overlay files the flow declares (path + sha)."""
        out: List[Dict[str, str]] = []
        for stage in flow.data.get("stages", []):
            for step in stage.get("steps", []):
                ov = step.get("overlay")
                if ov:
                    p = (self.repo_root / ov)
                    if p.exists():
                        out.append({"path": ov, "sha": _sha(p.read_text(encoding="utf-8"))})
        return out

    # -- resume -------------------------------------------------------------
    async def resume(
        self,
        run_id: str,
        *,
        budget_cap_usd: Optional[float] = None,
        fixtures_dir: Optional[Path] = None,
        keys: Optional[KeyResolver] = None,
        wait: bool = False,
        provider_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resume a run from disk (authority). Re-scans the run dir and restarts at
        the first missing/failed step per lane (S5). Optionally raises the cap
        (S4 recovery)."""
        run_dir = self._runs_root() / run_id
        rj = run_dir / "run.json"
        if not rj.exists():
            raise FileNotFoundError(f"run {run_id} not found")

        import json
        data = json.loads(rj.read_text(encoding="utf-8"))
        flow_name = data["flow"]["name"]
        flow_version = int(data["flow"]["version"])
        knobs_resolved = data.get("knobs", {})

        flow = load_flowspec(_flow_path(self.repo_root, flow_name, flow_version))
        # Reconstruct a KnobSet from the resolved values (literal knobs suffice for
        # a resume: derived values are already resolved on disk).
        knobs = KnobSet.from_dict({"knobs": {k: {"value": v} for k, v in knobs_resolved.items()}})

        keys = keys or KeyResolver.from_root(self.repo_root)
        secrets = self._all_secrets(keys)
        fixtures_dir = fixtures_dir or (Path(__file__).resolve().parent / "fixtures")

        store = RunStore.open(run_dir, secrets=secrets)
        # Optionally raise the cap so a budget-aborted run can finish (S4).
        if budget_cap_usd is not None:
            b = dict(data.get("budget") or {})
            b["cap_usd"] = float(budget_cap_usd)
            b["aborted_on_cap"] = False
            store.update_run(budget=b)

        # Restore the FULL ICB context persisted at launch (disk is authority). If
        # the file is absent (an older run), fall back to the run.json input ref.
        ctx = _read_context_file(store)
        if not ctx:
            ctx, _ = build_context_and_refs(_context_from_run(data))
        handle = RunHandle(run_id=run_id, store=store)
        self._handles[run_id] = handle

        engine = Engine(
            flow=flow, knobs=knobs, context=ctx, store=store, handle=handle,
            client_factory=self._client_factory(keys, fixtures_dir),
            pricing=PricingTable.default(self.repo_root),
            repo_root=self.repo_root,
            base_gates=load_base_gates(self.repo_root),
            provider_override=provider_override,
        )
        # Re-seed the live meter from what was already spent on disk (so a resumed
        # run's cap accounting continues, not restarts).
        engine._spend = float((data.get("budget") or {}).get("spend_usd", 0.0) or 0.0)
        totals = data.get("totals") or {}
        engine._tokens_in = int(totals.get("tokens_in", 0) or 0)
        engine._tokens_out = int(totals.get("tokens_out", 0) or 0)

        if wait:
            await engine.run()
        else:
            handle.task = asyncio.create_task(engine.run())
        return store.run()

    # -- registry / control -------------------------------------------------
    def list_runs(self) -> List[Dict[str, Any]]:
        """Summaries of every run on disk (disk is authority; no in-memory index)."""
        import json
        root = self._runs_root()
        out: List[Dict[str, Any]] = []
        if not root.exists():
            return out
        for rj in sorted(root.glob("*/run.json")):
            try:
                data = json.loads(rj.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "id": data.get("id"),
                "title": data.get("title"),
                "created": data.get("created"),
                "flow": data.get("flow"),
                "status": data.get("status"),
                "totals": data.get("totals"),
                "budget": data.get("budget"),
                "quarantine_banner": data.get("quarantine_banner"),
            })
        out.sort(key=lambda r: r.get("created") or "", reverse=True)
        return out

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        import json
        rj = self._runs_root() / run_id / "run.json"
        if not rj.exists():
            return None
        from .keys import redact
        return redact(json.loads(rj.read_text(encoding="utf-8")))

    def get_store(self, run_id: str) -> Optional[RunStore]:
        run_dir = self._runs_root() / run_id
        if not (run_dir / "run.json").exists():
            return None
        h = self._handles.get(run_id)
        if h is not None:
            return h.store
        return RunStore.open(run_dir)

    def cancel(self, run_id: str) -> bool:
        h = self._handles.get(run_id)
        if h is None:
            return False
        h.request_cancel()
        return True

    def decide(self, run_id: str, pair: Optional[str], decision: str) -> bool:
        h = self._handles.get(run_id)
        if h is None:
            return False
        if decision not in ("proceed", "kill"):
            raise ValueError("decision must be 'proceed' or 'kill'")
        return h.decide(pair, decision)

    def handle(self, run_id: str) -> Optional[RunHandle]:
        return self._handles.get(run_id)


# ---------------------------------------------------------------------------
# module helpers.
# ---------------------------------------------------------------------------

_CONTEXT_FILE = "_context.json"


def _write_context_file(store: RunStore, ctx: Dict[str, str]) -> None:
    """Persist the resolved ICB context under the run dir (engine-owned).

    Scrubbed for secrets (S2). This is NOT run.json — it is the resume aid so a
    re-resolved step reproduces the exact bytes (S5). It never leaves the machine
    and no endpoint serves it."""
    import json
    from .keys import redact, redact_with_values
    data = redact(dict(ctx))
    blob = json.dumps(data, ensure_ascii=False, indent=2)
    if store.secrets:
        blob = redact_with_values(blob, store.secrets)
    (store.run_dir / _CONTEXT_FILE).write_text(blob + "\n", encoding="utf-8", newline="\n")


def _read_context_file(store: RunStore) -> Dict[str, str]:
    import json
    p = store.run_dir / _CONTEXT_FILE
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: ("" if v is None else str(v)) for k, v in data.items()}
    except Exception:
        return {}


def _flow_path(root: Path, name: str, version: int) -> Path:
    from .compile import _flow_path as fp
    return fp(name, version, root)


def _context_from_run(data: dict) -> Dict[str, Any]:
    """Rebuild the slot→value context for a resume.

    run.json stores refs (shas), not raw ICB text (to keep it lean). On resume the
    coordinator artifacts already on disk carry the resolved context downstream, so
    a resumed run only needs the `input` brief that was preserved verbatim in the
    ref. Missing slots fill empty (the resolver's legacy behavior); the resume path
    trusts on-disk artifacts rather than re-resolving completed steps.
    """
    refs = data.get("context") or {}
    ctx: Dict[str, Any] = {}
    if isinstance(refs.get("input"), str):
        ctx["input"] = refs["input"]
    return ctx
