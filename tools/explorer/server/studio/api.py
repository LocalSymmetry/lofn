"""Studio API routes (WS1 slice): providers status + test ping.

Registered onto the existing v1 FastAPI app in ``main.py`` via ``register_studio``.
Later workstreams append their routers here too. Endpoints live under
``/api/studio/*`` (plan §9).

KEY HYGIENE (acceptance gate S2): ``GET /providers`` reports ``configured: bool``
and an optional model list only — never key material. ``POST /providers/{p}/test``
does a 1-token ping for a real provider; the mock provider returns instantly, and
the $0 test suite exercises the mock path, never a live endpoint.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .keys import KeyResolver, PROVIDER_KEY_NAMES, KEYLESS_PROVIDERS
from .pricing import PricingTable
from .providers import ProviderClient, ProviderRequest, ProviderError
from .compile import compile_flow, load_flow_and_knobs, _flow_path
from .knobs import (
    load_base_gates, build_run_gates, list_presets, load_preset, preset_path,
)
from .resolver import OverlayError
from .flowspec import (
    find_repo_root, run_dir_root, flows_dir, load_flowspec, list_flow_versions,
    flow_has_run, copy_on_write, validate_flowspec, load_yaml, next_version,
)
from .runs import RunService
from . import judge as judge_mod
from . import promote as promote_mod
try:  # subpackage (server.studio) vs top-level (tests) import layouts
    from ..files import FileService
    from ..manifest import load_manifest
except ImportError:  # pragma: no cover - top-level layout (tests)
    from tools.explorer.server.files import FileService  # type: ignore
    from tools.explorer.server.manifest import load_manifest  # type: ignore

# A small curated model list per provider, surfaced by /providers so the UI has
# something to populate a picker with. NOT authoritative pricing — that's
# pricing.yaml. Mirrors the legacy model_defaults.yaml families.
_MODEL_LIST = {
    "anthropic": [
        "claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5", "claude-fable-5",
    ],
    "openai": ["gpt-5.4", "gpt-5.3", "gpt-5-mini", "o4-mini"],
    "openrouter": ["anthropic/claude-sonnet-4.5", "x-ai/grok-4"],
    "gemini": ["gemini-3.1-pro-preview", "gemini-2.5-flash"],
    "poe": ["Poe-Claude-Opus-4.6", "Poe-GPT-5.4", "Poe-Gemini-3.1-Pro"],
    "mock": [],
}


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def build_studio_router(repo_root: Path, *, runs_root: Optional[Path] = None) -> APIRouter:
    """Build the /api/studio router. ``repo_root`` locates .env.studio + pricing.

    ``runs_root`` overrides where runs are written (default: output/studio under
    the repo). Tests pass a tmp dir so an in-process launch never touches the repo.
    """
    router = APIRouter(prefix="/api/studio", tags=["studio"])

    # ONE run service per app: owns the live RunHandle registry (cancel/resume/
    # decision route through it). Disk remains the authority for run state.
    run_service = RunService(repo_root=repo_root)
    if runs_root is not None:
        run_service._runs_root = lambda: runs_root  # type: ignore[method-assign]

    def _runs_root_dir() -> Path:
        return runs_root if runs_root is not None else run_dir_root(repo_root)

    def _run_dir(run_id: str) -> Path:
        d = (_runs_root_dir() / run_id).resolve()
        # jail: the run dir must live under the runs root (no traversal via id).
        if _runs_root_dir().resolve() not in d.parents and d != _runs_root_dir().resolve():
            raise HTTPException(status_code=400, detail=f"invalid run id: {run_id!r}")
        if not (d / "run.json").exists():
            raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
        return d

    def _file_service() -> FileService:
        # The v1 byte-safe write path, jailed by the manifest's write_allowlist.
        manifest = load_manifest(repo_root / "tools" / "explorer" / "pipeline_map.yaml")
        return FileService(repo_root, manifest.write_allowlist)

    def _resolver() -> KeyResolver:
        # Rebuild per request so a freshly-edited .env.studio is picked up without
        # a server restart (keys never cached in memory long-term).
        return KeyResolver.from_root(repo_root)

    @router.get("/providers")
    def list_providers():
        resolver = _resolver()
        providers = list(PROVIDER_KEY_NAMES.keys()) + list(KEYLESS_PROVIDERS)
        out = []
        for p in providers:
            entry = {
                "provider": p,
                "configured": resolver.is_configured(p),
            }
            if _MODEL_LIST.get(p):
                entry["model_list"] = _MODEL_LIST[p]
            out.append(entry)
        return out

    @router.post("/providers/{provider}/test")
    async def test_provider(provider: str):
        if provider not in PROVIDER_KEY_NAMES and provider not in KEYLESS_PROVIDERS:
            raise HTTPException(status_code=404, detail=f"unknown provider '{provider}'")

        resolver = _resolver()

        if provider == "mock":
            # Instant OK — no fixture required for a liveness check.
            return {"provider": "mock", "ok": True, "note": "mock provider always ready"}

        if not resolver.is_configured(provider):
            return {
                "provider": provider,
                "ok": False,
                "note": "not configured (no API key found in env or .env.studio)",
            }

        client = ProviderClient(provider, resolver)
        req = ProviderRequest(
            model=_ping_model(provider),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            stream=True,
        )
        try:
            got_final = False
            async for chunk in client.complete(req):
                if chunk.final is not None:
                    got_final = True
            return {"provider": provider, "ok": got_final}
        except ProviderError as e:
            # ProviderError messages are constructed without key material.
            return {"provider": provider, "ok": False, "note": str(e)}

    def _root() -> Path:
        return find_repo_root() if _repo_root_is_placeholder() else repo_root

    # -- /flows (Flow Lab): list families + get one version -----------------
    @router.get("/flows")
    def list_flows():
        """List every flow version on disk (one entry per @N file), with an
        ``immutable`` flag (a flow that has ever produced a run is read-only —
        plan §4). The Flow Lab picker + version list read this."""
        root = _root()
        d = flows_dir(root)
        out = []
        if d.exists():
            for fp in sorted(d.glob("*.flow.yaml")):
                try:
                    spec = load_flowspec(fp, validate=False)
                except Exception:
                    continue
                out.append({
                    "flow": spec.name,
                    "version": spec.version,
                    "modality": spec.data.get("modality"),
                    "title": spec.name,
                    "file": fp.name,
                    "file_sha": spec.file_sha(),
                    "immutable": flow_has_run(spec.name, spec.version, root),
                    "versions": list_flow_versions(spec.name, root),
                })
        return out

    @router.get("/flows/{name}")
    def get_flow(name: str, version: int = 1):
        """Return one flow@version — the full FlowSpec the pipeline rail renders
        from, plus provenance (``file_sha``) and the ``immutable`` gate."""
        root = _root()
        try:
            spec = load_flowspec(_flow_path(name, version, root), validate=False)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        data = dict(spec.data)
        data["file"] = spec.path.name
        data["file_sha"] = spec.file_sha()
        data["immutable"] = flow_has_run(name, version, root)
        data["versions"] = list_flow_versions(name, root)
        return data

    @router.get("/flows/{name}/raw")
    def get_flow_raw(name: str, version: int = 1):
        """The verbatim on-disk YAML text for the raw-YAML Monaco tab."""
        root = _root()
        try:
            fp = _flow_path(name, version, root)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {
            "flow": name,
            "version": version,
            "file": fp.name,
            "yaml": fp.read_text(encoding="utf-8"),
            "immutable": flow_has_run(name, version, root),
        }

    @router.post("/flows/{name}/version")
    def fork_flow(name: str, body: ForkBody):
        """Copy-on-write a flow to @N+1 ("New version from this"). If ``yaml`` is
        given (the raw-YAML editor's edited buffer) it is schema-validated and
        written as the new version; otherwise the source is duplicated verbatim.
        The source is never overwritten — this is how an immutable (already-run)
        flow is edited (plan §4)."""
        root = _root()
        try:
            src = load_flowspec(_flow_path(name, body.version, root), validate=False)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        if body.yaml is not None:
            import io
            from .flowspec import _yaml, dump_yaml_str
            try:
                data = _yaml().load(io.StringIO(body.yaml))
                plain = json.loads(json.dumps(data, default=lambda o: (
                    dict(o) if hasattr(o, "items") else
                    list(o) if isinstance(o, (list, tuple)) else str(o)
                )))
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"YAML did not parse: {e}")
            if str(plain.get("flow")) != name:
                raise HTTPException(status_code=400,
                                    detail=f"raw YAML flow name {plain.get('flow')!r} must equal {name!r}")
            new_ver = next_version(name, root)
            plain["version"] = new_ver
            plain["base"] = f"{name}@{src.version}"
            errors = validate_flowspec(plain)
            if errors:
                raise HTTPException(status_code=422,
                                    detail="FlowSpec failed schema validation:\n  - " + "\n  - ".join(errors))
            from .flowspec import flow_filename
            out = flows_dir(root) / flow_filename(name, new_ver)
            out.write_text(dump_yaml_str(plain), encoding="utf-8", newline="\n")
        else:
            out = copy_on_write(src, root=root)

        spec = load_flowspec(out, validate=False)
        return {
            "flow": spec.name,
            "version": spec.version,
            "file": out.name,
            "file_sha": spec.file_sha(),
        }

    # -- /presets (Flow Lab Knobs panel): list / get / save / duplicate -----
    @router.get("/presets")
    def presets_list(flow: str | None = None):
        """List knob presets (``<name>.knobs.yaml`` under flows/presets/). ``flow``
        is accepted for API symmetry; presets are flow-agnostic today."""
        root = _root()
        out = []
        for name in list_presets(flows_dir(root)):
            try:
                ks = load_preset(flows_dir(root), name)
                out.append({"name": name, "desc": ks.desc,
                            "file": f"presets/{name}.knobs.yaml"})
            except Exception:
                out.append({"name": name, "file": f"presets/{name}.knobs.yaml"})
        return out

    @router.get("/presets/{name}")
    def preset_get(name: str):
        """Return one preset's definitions (the KnobSetSpec the Knobs panel edits)."""
        root = _root()
        p = preset_path(flows_dir(root), name)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"preset '{name}' not found")
        try:
            return load_preset(flows_dir(root), name).to_dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/presets")
    def preset_save(body: PresetSaveBody):
        """Save / duplicate a preset (Knobs panel save + duplicate ops). Writes
        ONLY under flows/presets/** (byte-safe, LF). ``knobs`` is the full
        KnobSetSpec; on a name collision it overwrites that preset."""
        from .knobs import KnobSet, save_preset
        root = _root()
        if not body.name or not body.name.strip():
            raise HTTPException(status_code=400, detail="a preset name is required")
        spec = dict(body.knobs or {})
        spec.setdefault("name", body.name)
        try:
            ks = KnobSet.from_dict(spec)
            out = save_preset(flows_dir(root), ks, name=body.name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"name": body.name, "file": f"presets/{out.name}"}

    # -- /compile (WS2): resolved prompts + diffs + token/cost estimate -----
    @router.post("/compile")
    def compile_endpoint(body: CompileBody):
        """Resolve a flow (or one step) → the exact bytes the engine will send,
        plus per-step diffs vs a baseline preset and a token/cost estimate.

        S3 (Compile ≡ execute): this uses the SAME resolver.resolve() the engine
        calls, so what this returns is byte-identical to what will be sent.
        Compile is allowed on a broken preset (invariant violations are reported,
        not fatal); only Run is blocked.
        """
        root = find_repo_root() if _repo_root_is_placeholder() else repo_root
        try:
            flow, knobs = load_flow_and_knobs(
                body.flow, body.version, body.knobs, root
            )
            baseline = None
            if body.baseline is not None:
                _, baseline = load_flow_and_knobs(
                    body.flow, body.version, body.baseline, root
                )
            pricing = PricingTable.default(root)
            result = compile_flow(
                flow, knobs, body.context or {}, pricing,
                step=body.step, baseline=baseline, repo_root=root,
            )
            return result.as_dict()
        except (FileNotFoundError, KeyError) as e:
            raise HTTPException(status_code=404, detail=str(e))
        except OverlayError as e:
            raise HTTPException(status_code=422, detail=f"overlay did not apply: {e}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # -- /gates/preview (WS2): the run-local gates.yaml a run would materialize --
    @router.post("/gates/preview")
    def gates_preview(body: GatesPreviewBody):
        """Preview the run-local gates mapping (base vault/gates.yaml + knob
        overrides) that validate_step.py --gates would read. No file is written;
        this is the zero-cost preview of the materialized thresholds."""
        root = find_repo_root() if _repo_root_is_placeholder() else repo_root
        try:
            _, knobs = load_flow_and_knobs(
                body.flow or "lofn-music", body.version, body.knobs, root
            )
            base = load_base_gates(root)
            return {
                "gates": build_run_gates(base, knobs),
                "bindings": knobs.gates_bindings(),
            }
        except (FileNotFoundError, KeyError) as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    def _repo_root_is_placeholder() -> bool:
        # main.py passes the real repo root; guard against a bare/relative one.
        return not (repo_root / "vault" / "gates.yaml").exists()

    # -- /runs (WS3): launch / list / get / events / cancel / resume / decision --
    @router.post("/runs")
    async def launch_run(body: LaunchBody):
        """Launch a run (plan §9). ``dry_run: true`` compiles every step (records
        resolutions) WITHOUT any API call. No run starts without an explicit budget
        cap (plan §13.4) — a missing cap is a 400."""
        if body.budget_cap_usd is None or body.budget_cap_usd < 0:
            raise HTTPException(status_code=400,
                                detail="a non-negative budget_cap_usd is required (no run starts without a cap)")
        try:
            result = await run_service.launch(
                flow_name=body.flow,
                flow_version=body.version,
                knobs_spec=body.knobs,
                context=body.context or {},
                title=body.title or body.flow,
                budget_cap_usd=body.budget_cap_usd,
                dry_run=bool(body.dry_run),
                experimental_qa=bool(body.experimental_qa),
                # A dry run makes NO API calls (compile-all only) so we run it inline
                # and return the finished record. A real run streams in the
                # background; the client tails /runs/{id}/events for progress.
                wait=bool(body.dry_run),
            )
            return result
        except (FileNotFoundError, KeyError) as e:
            raise HTTPException(status_code=404, detail=str(e))
        except OverlayError as e:
            raise HTTPException(status_code=422, detail=f"overlay did not apply: {e}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/runs")
    def list_runs():
        return run_service.list_runs()

    @router.get("/runs/{run_id}")
    def get_run(run_id: str):
        data = run_service.get_run(run_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
        return data

    @router.get("/runs/{run_id}/events")
    async def run_events(run_id: str):
        """SSE tail of runlog.jsonl (plan §5: one event stream, two consumers).

        Replays the existing log then streams new events live. A finished run's
        stream ends after a ``run_end`` event; a live run streams until the client
        disconnects. Redaction happened at write time — every frame is already
        key-safe (S2)."""
        store = run_service.get_store(run_id)
        if store is None:
            raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        # Seed with the existing log (replay), then subscribe live.
        for ev in store.read_runlog():
            queue.put_nowait(ev)

        def _listener(ev: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ev)

        store.add_listener(_listener)

        async def gen():
            try:
                # If the run is already finished, drain the replay then stop.
                finished = run_service.get_run(run_id) or {}
                already_done = finished.get("status") in ("complete", "failed", "cancelled")
                while True:
                    try:
                        ev = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if already_done and queue.empty():
                            break
                        # keep-alive comment frame
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if ev.get("event") == "run_end":
                        break
            finally:
                store.remove_listener(_listener)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @router.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        ok = run_service.cancel(run_id)
        if not ok:
            raise HTTPException(status_code=409,
                                detail=f"run '{run_id}' has no live task to cancel (already finished or not in this process)")
        return {"run_id": run_id, "cancelling": True}

    @router.post("/runs/{run_id}/resume")
    async def resume_run(run_id: str, body: ResumeBody | None = None):
        try:
            result = await run_service.resume(
                run_id,
                budget_cap_usd=(body.budget_cap_usd if body else None),
            )
            return result
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/runs/{run_id}/decision")
    def run_decision(run_id: str, body: DecisionBody):
        """Resolve a HOLD_FOR_HUMAN (proceed | kill) for a paused pair (S7)."""
        try:
            ok = run_service.decide(run_id, body.pair, body.decision)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not ok:
            raise HTTPException(status_code=409,
                                detail=f"run '{run_id}' has no pending hold for pair {body.pair!r}")
        return {"run_id": run_id, "pair": body.pair, "decision": body.decision}

    # -- /judge (WS6): compare + fail-closed blind set + verdict -------------
    @router.post("/judge")
    def judge_endpoint(body: JudgeBody):
        """Build a fail-closed blind set from two finished runs (plan §8.4).

        Also returns the OPEN comparison (aligned diffs + gate/cost/knob deltas)
        so the Compare & Judge surface has both views. The blind ``key`` (which
        run each side came from) is coordinator-only and returned separately from
        the provenance-free cards."""
        da, db = _run_dir(body.run_a), _run_dir(body.run_b)
        return {
            "compare": judge_mod.compare(da, db),
            "blind": judge_mod.blind_judge(da, db, seed=body.seed),
        }

    @router.post("/judge/verdict")
    def judge_verdict(body: VerdictBody):
        """Append a verdict to BOTH runs' run.json (byte-safe, redacted)."""
        da, db = _run_dir(body.run_a), _run_dir(body.run_b)
        rec = judge_mod.append_verdict(da, db, body.verdict or {})
        return {"run_a": body.run_a, "run_b": body.run_b, "verdict": rec}

    # -- /promote (WS): value/patch write-back + brief ----------------------
    @router.post("/promote/preview")
    def promote_preview(body: PromoteBody):
        """The diffs a promotion WOULD produce (no write)."""
        try:
            return promote_mod.preview(
                _file_service(),
                gate_values=body.gate_values,
                patches=body.patches,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.post("/promote/apply")
    def promote_apply(body: PromoteBody):
        """AUTO-APPLY value + simple-patch changes via FileService, run
        --meta-check after, return the new git status (values/patches only)."""
        try:
            return promote_mod.apply(
                _file_service(), repo_root,
                gate_values=body.gate_values,
                patches=body.patches,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.post("/promote/brief")
    def promote_brief(body: BriefBody):
        """Generate the PROMOTION_BRIEF.md text for structural changes."""
        return {"brief": promote_mod.brief(
            run_a=body.run_a, run_b=body.run_b,
            title=body.title or "Structural promotion",
            summary=body.summary or "",
            structural_changes=body.structural_changes,
            knob_delta=body.knob_delta,
        )}

    return router


class CompileBody(BaseModel):
    """POST /api/studio/compile request (plan §9)."""

    flow: str
    version: int = 1
    # knobs: a preset NAME (str), an inline knobs dict, or null (flow default).
    knobs: object | None = None
    context: dict | None = None
    step: str | None = None
    # baseline: preset name / inline dict to diff against ("feel the knob").
    baseline: object | None = None


class GatesPreviewBody(BaseModel):
    flow: str | None = None
    version: int = 1
    knobs: object | None = None


class ForkBody(BaseModel):
    """POST /api/studio/flows/{name}/version — copy-on-write to @N+1."""

    version: int = 1               # source version to fork
    yaml: str | None = None        # optional edited raw YAML to write as the new version


class PresetSaveBody(BaseModel):
    """POST /api/studio/presets — save / duplicate a knob preset."""

    name: str
    knobs: dict | None = None      # full KnobSetSpec ({name?, desc?, knobs, invariants?})


class LaunchBody(BaseModel):
    """POST /api/studio/runs request (plan §9)."""

    flow: str
    version: int = 1
    knobs: object | None = None          # preset name, inline dict, or null (flow default)
    context: dict | None = None          # slot -> verbatim ICB value (seed/personality/panel/input)
    title: str | None = None
    budget_cap_usd: float | None = None  # REQUIRED (no run without a cap)
    dry_run: bool = False                # compile-all only, no API calls
    experimental_qa: bool = False        # optional lofn-qa LLM-judge step (off by default)


class ResumeBody(BaseModel):
    budget_cap_usd: float | None = None  # raise the cap so a budget-aborted run can finish


class DecisionBody(BaseModel):
    pair: str | None = None              # the held pair label ("03"); null for a run-level hold
    decision: str                        # "proceed" | "kill"


class JudgeBody(BaseModel):
    """POST /api/studio/judge request (plan §9)."""

    run_a: str
    run_b: str
    seed: int = 0                        # randomizes A/B sides deterministically


class VerdictBody(BaseModel):
    """POST /api/studio/judge/verdict — appends to BOTH runs."""

    run_a: str
    run_b: str
    verdict: dict | None = None          # {matchup_id, winner, note, ...}


class PromoteBody(BaseModel):
    """POST /api/studio/promote/preview | /apply request (plan §9)."""

    # gates.yaml key -> proposed value (EXISTING keys only; a promotion changes a
    # threshold, never introduces one).
    gate_values: dict | None = None
    # simple prose patches: [{"file","old","new"}] exact, unique, non-regex swaps.
    patches: list[dict] | None = None


class BriefBody(BaseModel):
    """POST /api/studio/promote/brief — the structural-change hand-off."""

    run_a: str | None = None
    run_b: str | None = None
    title: str | None = None
    summary: str | None = None
    structural_changes: list[dict] | None = None
    knob_delta: list[dict] | None = None


def _ping_model(provider: str) -> str:
    return {
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5-mini",
        "openrouter": "anthropic/claude-sonnet-4.5",
        "gemini": "gemini-2.5-flash",
        "poe": "Poe-Claude-Haiku-4.5",
    }.get(provider, "unknown")


def register_studio(app, repo_root: Path, *, runs_root: Optional[Path] = None) -> None:
    """Attach the studio router to the existing FastAPI app (called from main)."""
    app.include_router(build_studio_router(repo_root, runs_root=runs_root))
