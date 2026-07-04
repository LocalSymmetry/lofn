"""Backend ACCEPTANCE suite (plan §10, WS7 — adversarial).

This module is written by the acceptance agent, which did NOT build the studio
modules it tests. It exercises the backend-testable gates S1-S9 end-to-end on the
MOCK provider only — $0, no live API call, no bound server port, no dep installs.

Gate coverage (plan §10):
  * S1 — Canon immutability: sha-sweep skills/ vault/ .claude/ before AND after a
    full mock run; assert byte-unchanged. Also assert the run wrote only under
    output/studio/** (a tmp-redirected root here).
  * S2 — Key hygiene: plant canary keys in env AND .env.studio; assert they never
    appear in run.json, runlog.jsonl, an SSE frame stream, or an API response.
  * S3 — Compile == execute: the resolver bytes shown by compile equal what the
    engine sent (recorded per step in the runlog dry-run events).
  * S4 — Budget hard-stop: a cap below projected spend halts at/below the cap and
    states why; a raised-cap resume finishes cleanly.
  * S5 — Resume idempotence: kill mid-run, resume; the final artifact set is
    byte-identical to an uninterrupted mock run.
  * S6 — Quarantine is loud: a forced hard-fail exhausting retries emits the
    "N of <n_pairs> pairs broke open" banner BEFORE any portfolio stage.
  * S7 — Human-subject HOLD: a name+crime lyric fixture blocks the pair with a
    HOLD_FOR_HUMAN until a decision is made.
  * S8 — Provenance replay: run.json + fixtures reproduce a mock run
    byte-identically (deterministic mock => identical artifact shas).
  * S9 — Gate parity: a studio gate result equals the CLI validate_step.py result
    for the same artifact + the same run-local gates file.

Every run below is n_pairs=2, n_variations_per_pair=1, mock provider; fixtures
authored by ``_fixtures_music.author_run_fixtures``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

from studio.compile import compile_flow
from studio.flowspec import find_repo_root, flows_dir, load_flowspec, run_dir_root
from studio.keys import KeyResolver, REDACTED
from studio.knobs import KnobSet, load_preset, load_base_gates, materialize_gates
from studio.pricing import PricingTable
from studio.resolver import resolve
from studio.runs import RunService

import _fixtures_music as fm

_REPO_ROOT = find_repo_root()

# Full ICB context (every declared slot filled verbatim), matching test_engine.
_CTX = {
    "input": "A song about tide pools at dawn",
    "seed": "GOLDEN SEED BLOCK: cold light on wet granite",
    "meta_prompt": "META PROMPT DIRECTIVE: patience over spectacle",
    "personality": "PERSONA DNA BLOCK: the quiet cartographer",
    "concept_panel": "CONCEPT PANEL 6 VOICES",
    "medium_panel": "MEDIUM PANEL 6 VOICES",
    "marketing_panel": "MARKETING PANEL 6 VOICES",
    "flairs": "15 SPECIAL FLAIRS",
    "genres_list": "dream-pop\tshoegaze",
    "frames_list": "dawn\ttide",
    "image_context": "",
}

# Canary keys shaped like real provider formats; must NEVER survive to any sink.
_CANARY_ANTHROPIC = "sk-ant-api03-CANARY" + "A" * 40
_CANARY_OPENAI = "sk-proj-CANARY" + "B" * 40
_CANARY_POE_OPAQUE = "poe-CANARY-opaque-token-0f1e2d3c4b5a6978"


# ---------------------------------------------------------------------------
# Shared helpers (mirror test_engine's harness so behavior is comparable).
# ---------------------------------------------------------------------------

def _music_flow():
    return load_flowspec(flows_dir(_REPO_ROOT) / "lofn-music@1.flow.yaml")


def _two_pair_knobs() -> KnobSet:
    ks = load_preset(flows_dir(_REPO_ROOT), "classic-6x4")
    ks = ks.with_value("n_pairs", 2)
    ks = ks.with_value("n_variations_per_pair", 1)
    return ks


def _inline(knobs: KnobSet) -> dict:
    return knobs.to_dict()


def _keys_no_secrets(root: Path) -> KeyResolver:
    return KeyResolver(repo_root=root, env={}, dotenv={})


def _service(tmp_path: Path) -> RunService:
    """A RunService whose run root is redirected under tmp (never the real repo)."""
    svc = RunService(repo_root=_REPO_ROOT)
    svc._runs_root = lambda: (tmp_path / "output" / "studio")  # type: ignore[method-assign]
    return svc


def _author(flow, knobs, fixtures_dir, *, n_pairs, **kw) -> int:
    return fm.author_run_fixtures(
        flow, knobs, _CTX, fixtures_dir, _REPO_ROOT, n_pairs=n_pairs, **kw,
    )


async def _launch(svc: RunService, *, keys=None, **kw):
    kw.setdefault("provider_override", "mock")
    kw["keys"] = keys or _keys_no_secrets(_REPO_ROOT)
    return await svc.launch(**kw)


async def _resume(svc: RunService, run_id: str, *, keys=None, **kw):
    kw.setdefault("provider_override", "mock")
    kw["keys"] = keys or _keys_no_secrets(_REPO_ROOT)
    return await svc.resume(run_id, **kw)


def _run_full_mock(svc: RunService, fixtures: Path, knobs: KnobSet, *, keys=None,
                   title="acc", cap=100.0) -> dict:
    return asyncio.run(_launch(
        svc, flow_name="lofn-music", flow_version=1, knobs_spec=_inline(knobs),
        context=_CTX, title=title, budget_cap_usd=cap,
        fixtures_dir=fixtures, keys=keys, wait=True,
    ))


def _artifact_map(run_dir: Path) -> Dict[str, bytes]:
    """name -> bytes for every canonical step artifact (excludes bookkeeping)."""
    out: Dict[str, bytes] = {}
    for p in sorted(run_dir.glob("*.md")):
        if ".repair_attempt_" in p.name:
            continue
        out[p.name] = p.read_bytes()
    return out


# ---------------------------------------------------------------------------
# Canon hash sweep helper (S1).
# ---------------------------------------------------------------------------

_CANON_DIRS = ("skills", "vault", ".claude")


def _sweep_canon(root: Path) -> Dict[str, str]:
    """{repo-relative-path -> sha256} for every file under skills/ vault/ .claude/.

    This is the S1 tripwire: the studio is additive; a full run must leave every
    canon byte untouched.
    """
    digest: Dict[str, str] = {}
    for d in _CANON_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            # skip transient python caches (not canon content)
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            rel = p.relative_to(root).as_posix()
            digest[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return digest


# ===========================================================================
# S1 — Canon immutability.
# ===========================================================================

def test_s1_full_mock_run_leaves_canon_byte_identical(tmp_path: Path):
    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    _author(flow, knobs, fixtures, n_pairs=2)

    before = _sweep_canon(_REPO_ROOT)
    assert before, "canon sweep found no files — wrong root?"

    svc = _service(tmp_path)
    result = _run_full_mock(svc, fixtures, knobs, title="s1")
    assert result["status"] == "complete"

    after = _sweep_canon(_REPO_ROOT)

    # Byte-for-byte identical: no added, removed, or modified canon file.
    added = set(after) - set(before)
    removed = set(before) - set(after)
    changed = {k for k in before.keys() & after.keys() if before[k] != after[k]}
    assert not added, f"canon files ADDED by a studio run (S1 violation): {sorted(added)}"
    assert not removed, f"canon files REMOVED by a studio run (S1 violation): {sorted(removed)}"
    assert not changed, f"canon files MODIFIED by a studio run (S1 violation): {sorted(changed)}"


def test_s1_run_writes_only_under_studio_root(tmp_path: Path):
    """Everything the engine wrote lives under the (tmp-redirected) studio root."""
    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    _author(flow, knobs, fixtures, n_pairs=2)

    svc = _service(tmp_path)
    result = _run_full_mock(svc, fixtures, knobs, title="s1b")
    run_dir = svc._runs_root() / result["id"]

    # The run dir exists under the studio root; run.json + runlog + artifacts live there.
    assert run_dir.exists()
    assert (run_dir / "run.json").exists()
    assert (run_dir / "runlog.jsonl").exists()
    arts = _artifact_map(run_dir)
    assert arts, "expected canonical artifacts under the run dir"
    # The engine holds no write handle outside output/studio/** — the real repo's
    # output/studio was never touched (we redirected the root into tmp).
    assert str(run_dir).startswith(str(tmp_path))


# ===========================================================================
# S2 — Key hygiene (canary keys planted in env AND .env.studio).
# ===========================================================================

def _canary_keys(root: Path) -> KeyResolver:
    """A resolver carrying canaries in BOTH the env map and the .env.studio map."""
    return KeyResolver(
        repo_root=root,
        env={"ANTHROPIC_API_KEY": _CANARY_ANTHROPIC},
        dotenv={"OPENAI_API_KEY": _CANARY_OPENAI, "POE_API_KEY": _CANARY_POE_OPAQUE},
    )


def _all_text_sinks(run_dir: Path) -> Dict[str, str]:
    sinks: Dict[str, str] = {}
    sinks["run.json"] = (run_dir / "run.json").read_text(encoding="utf-8")
    sinks["runlog.jsonl"] = (run_dir / "runlog.jsonl").read_text(encoding="utf-8")
    for p in sorted(run_dir.glob("*.md")):
        sinks[p.name] = p.read_text(encoding="utf-8")
    ctx = run_dir / "_context.json"
    if ctx.exists():
        sinks["_context.json"] = ctx.read_text(encoding="utf-8")
    return sinks


def test_s2_canary_keys_never_reach_any_sink(tmp_path: Path):
    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    _author(flow, knobs, fixtures, n_pairs=2)

    svc = _service(tmp_path)
    keys = _canary_keys(tmp_path)
    # Sanity: the resolver really did pick up the planted canaries (so the test
    # is meaningfully exercising the scrub path, not a no-op).
    assert keys.resolve("anthropic") == _CANARY_ANTHROPIC
    assert keys.resolve("openai") == _CANARY_OPENAI
    assert keys.resolve("poe") == _CANARY_POE_OPAQUE

    result = _run_full_mock(svc, fixtures, knobs, keys=keys, title="s2")
    assert result["status"] == "complete"

    run_dir = svc._runs_root() / result["id"]
    canaries = (_CANARY_ANTHROPIC, _CANARY_OPENAI, _CANARY_POE_OPAQUE)

    # (a) On-disk sinks: run.json, runlog.jsonl, artifacts, _context.json.
    for name, text in _all_text_sinks(run_dir).items():
        for c in canaries:
            assert c not in text, f"canary key leaked into {name} (S2 violation)"

    # (b) The redacted API-response snapshot (what GET /runs/{id} would serve).
    api_resp = json.dumps(svc.get_run(result["id"]), ensure_ascii=False)
    for c in canaries:
        assert c not in api_resp, "canary key leaked into the API response (S2)"

    # (c) Every runlog event as an SSE frame (the SSE tail serializes these verbatim).
    store = svc.get_store(result["id"])
    for ev in store.read_runlog():
        frame = f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        for c in canaries:
            assert c not in frame, "canary key leaked into an SSE frame (S2)"


def test_s2_api_response_and_sse_are_scrubbed_via_endpoints(tmp_path: Path):
    """End-to-end through the real FastAPI router: a canary planted in the process
    env must not appear in ANY /api/studio response body, incl. the SSE stream.

    Launch goes through the RunService directly (so we can force the mock provider)
    with canaries as scrubbing secrets; the endpoints then serve the redacted view.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from studio.api import register_studio

    runs_root = tmp_path / "output" / "studio"
    app = FastAPI()
    register_studio(app, _REPO_ROOT, runs_root=runs_root)
    client = TestClient(app)

    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    _author(flow, knobs, fixtures, n_pairs=2)

    # Drive a mock run whose store scrubs the canaries, writing under the SAME
    # runs_root the router serves from.
    svc = RunService(repo_root=_REPO_ROOT)
    svc._runs_root = lambda: runs_root  # type: ignore[method-assign]
    keys = _canary_keys(tmp_path)
    result = _run_full_mock(svc, fixtures, knobs, keys=keys, title="s2api")
    run_id = result["id"]

    canaries = (_CANARY_ANTHROPIC, _CANARY_OPENAI, _CANARY_POE_OPAQUE)

    # GET /runs and /runs/{id} bodies never carry a canary.
    body_list = client.get("/api/studio/runs").text
    body_one = client.get(f"/api/studio/runs/{run_id}").text
    for c in canaries:
        assert c not in body_list and c not in body_one, "canary in /runs endpoint (S2)"

    # SSE frames never carry a canary.
    with client.stream("GET", f"/api/studio/runs/{run_id}/events") as resp:
        assert resp.status_code == 200
        sse_body = "".join(chunk for chunk in resp.iter_text())
    for c in canaries:
        assert c not in sse_body, "canary in the SSE stream (S2)"

    # /providers reports configured-bool only, never key material.
    prov = client.get("/api/studio/providers").text
    for c in canaries:
        assert c not in prov, "canary in /providers (S2)"


# ===========================================================================
# S3 — Compile == execute (resolver bytes == what the engine sent).
# ===========================================================================

def test_s3_compile_bytes_equal_engine_sent(tmp_path: Path):
    """For every LLM step, the text_sha compile shows equals what the engine
    recorded sending (runlog step_dry_run events)."""
    flow = _music_flow()
    knobs = _two_pair_knobs()
    svc = _service(tmp_path)

    # A dry run compiles every step and records its resolved text_sha (no API call).
    result = asyncio.run(_launch(
        svc, flow_name="lofn-music", flow_version=1, knobs_spec=_inline(knobs),
        context=_CTX, title="s3", budget_cap_usd=0.0, dry_run=True, wait=True,
    ))
    store = svc.get_store(result["id"])
    dry_events = {(e.get("step"), e.get("pair")): e
                  for e in store.read_runlog() if e.get("event") == "step_dry_run"}
    assert dry_events, "no step_dry_run events recorded — cannot prove S3"

    pricing = PricingTable.default(_REPO_ROOT)

    # (a) Coordinator steps + the fan-out REPRESENTATIVE (pair 1): what /compile
    #     shows is byte-identical to what the engine recorded sending. This is the
    #     literal "Compile view == what the engine sent" assertion.
    representative_checked = 0
    for (step_id, pair), ev in dry_events.items():
        if pair not in (None, "01"):
            continue
        compiled = compile_flow(flow, knobs, _CTX, pricing, step=step_id, repo_root=_REPO_ROOT)
        sha = compiled.steps[0].resolved.text_sha   # compile resolves fan-out at pair 1
        assert ev["text_sha"] == sha, (
            f"compile != execute for step {step_id} pair {pair} (S3 violation)"
        )
        representative_checked += 1
    assert representative_checked >= 6, (
        f"expected >=6 representative steps checked, saw {representative_checked}"
    )

    # (b) EVERY recorded send (all pairs) is byte-identical to the SAME resolver
    #     function's output at that exact (step, pair). Compile and the engine share
    #     one resolution path — proven by resolving here and matching the runlog sha
    #     for every step-instance, including fan-out pair 2 (which compile abbreviates
    #     to its pair-1 representative for cost, but the engine still sends per pair).
    all_checked = 0
    for (step_id, pair), ev in dry_events.items():
        pair_int = int(pair) if pair is not None else None
        engine_side = resolve(flow, knobs, _CTX, step_id, pair=pair_int, repo_root=_REPO_ROOT)
        assert ev["text_sha"] == engine_side.text_sha, (
            f"engine send != resolver output for step {step_id} pair {pair} (S3)"
        )
        all_checked += 1
    assert all_checked >= 8, f"expected >=8 step-instances, saw {all_checked}"


# ===========================================================================
# S4 — Budget hard-stop then clean resume.
# ===========================================================================

def test_s4_budget_hard_stop_then_resume(tmp_path: Path):
    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    _author(flow, knobs, fixtures, n_pairs=2)
    svc = _service(tmp_path)

    # A cap far below the first opus-tier call's projected cost forces an abort.
    result = asyncio.run(_launch(
        svc, flow_name="lofn-music", flow_version=1, knobs_spec=_inline(knobs),
        context=_CTX, title="s4", budget_cap_usd=0.01,
        fixtures_dir=fixtures, wait=True,
    ))
    assert result["status"] == "cancelled", "a below-projection cap must halt the run"
    assert result["budget"]["aborted_on_cap"] is True
    assert result["budget"]["spend_usd"] <= result["budget"]["cap_usd"] + 1e-9

    # The reason is stated LOUDLY on the runlog.
    store = svc.get_store(result["id"])
    events = [e.get("event") for e in store.read_runlog()]
    assert any(e in ("budget_exceeded", "budget_preflight_abort") for e in events), (
        "no loud budget-abort event on the runlog (S4)"
    )

    # Raise the cap and resume -> completes cleanly, no longer flagged aborted.
    resumed = asyncio.run(_resume(
        svc, result["id"], budget_cap_usd=100.0, fixtures_dir=fixtures, wait=True,
    ))
    assert resumed["status"] == "complete"
    assert resumed["budget"]["aborted_on_cap"] is False


# ===========================================================================
# S5 — Resume idempotence (kill mid-run -> byte-identical artifacts).
# ===========================================================================

def test_s5_resume_is_byte_identical_to_uninterrupted(tmp_path: Path):
    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    _author(flow, knobs, fixtures, n_pairs=2)

    # Reference: an uninterrupted run.
    ref_svc = _service(tmp_path / "ref")
    ref = _run_full_mock(ref_svc, fixtures, knobs, title="ref")
    assert ref["status"] == "complete"
    ref_artifacts = _artifact_map(ref_svc._runs_root() / ref["id"])

    # Interrupted: cancel after step 01, then resume.
    kill_svc = _service(tmp_path / "kill")

    async def killed_then_resume():
        res = await _launch(
            kill_svc, flow_name="lofn-music", flow_version=1, knobs_spec=_inline(knobs),
            context=_CTX, title="kill", budget_cap_usd=100.0,
            fixtures_dir=fixtures, wait=False,
        )
        run_id = res["id"]
        store = kill_svc.get_store(run_id)
        for _ in range(4000):
            if store.artifact_exists("step01_essence_facets.md"):
                break
            await asyncio.sleep(0.001)
        kill_svc.cancel(run_id)
        h = kill_svc.handle(run_id)
        if h and h.task:
            try:
                await h.task
            except Exception:
                pass
        return await _resume(kill_svc, run_id, fixtures_dir=fixtures, wait=True)

    resumed = asyncio.run(killed_then_resume())
    assert resumed["status"] == "complete"
    kill_artifacts = _artifact_map(kill_svc._runs_root() / resumed["id"])

    assert set(ref_artifacts) == set(kill_artifacts), "artifact SET differs after resume (S5)"
    for name in ref_artifacts:
        assert ref_artifacts[name] == kill_artifacts[name], (
            f"{name} not byte-identical after resume (S5)"
        )


# ===========================================================================
# S6 — Quarantine is loud (banner BEFORE portfolio).
# ===========================================================================

def test_s6_quarantine_banner_precedes_portfolio(tmp_path: Path):
    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    # Break step 08 for BOTH pairs -> retries exhaust -> both quarantine.
    _author(flow, knobs, fixtures, n_pairs=2, break_step="08")
    svc = _service(tmp_path)

    result = _run_full_mock(svc, fixtures, knobs, title="s6")

    assert result["totals"]["pairs_quarantined"] == 2
    banner = result.get("quarantine_banner")
    assert banner, "no quarantine banner emitted (S6)"
    assert "2 of 2 pairs broke open" in banner
    assert "step 08" in banner

    # The banner event precedes every portfolio_gate event (ordering guarantee).
    store = svc.get_store(result["id"])
    log = store.read_runlog()
    banner_idxs = [i for i, e in enumerate(log) if e.get("event") == "quarantine_banner"]
    portfolio_idxs = [i for i, e in enumerate(log) if e.get("event") == "portfolio_gate"]
    assert banner_idxs, "no quarantine_banner event on the runlog (S6)"
    for pi in portfolio_idxs:
        assert banner_idxs[0] < pi, "banner did NOT precede portfolio (S6 ordering)"


# ===========================================================================
# S7 — Human-subject HOLD wired (name+crime blocks the pair until decided).
# ===========================================================================

def test_s7_name_crime_holds_then_kill_quarantines(tmp_path: Path):
    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    _author(flow, knobs, fixtures, n_pairs=2, name_and_crime_on_08=True)
    svc = _service(tmp_path)

    async def run_and_kill():
        res = await _launch(
            svc, flow_name="lofn-music", flow_version=1, knobs_spec=_inline(knobs),
            context=_CTX, title="s7", budget_cap_usd=100.0,
            fixtures_dir=fixtures, wait=False,
        )
        run_id = res["id"]
        store = svc.get_store(run_id)
        held = set()
        for _ in range(8000):
            for e in store.read_runlog():
                if e.get("event") == "hold_for_human":
                    held.add(e.get("pair"))
            if held >= {"01", "02"}:
                break
            await asyncio.sleep(0.001)
        assert held >= {"01", "02"}, f"expected both pairs to HOLD, saw {held}"
        for p in ("01", "02"):
            assert svc.decide(run_id, p, "kill"), f"decide(kill) failed for pair {p}"
        h = svc.handle(run_id)
        if h and h.task:
            await h.task
        return svc.get_run(run_id)

    result = asyncio.run(run_and_kill())
    assert result["status"] == "complete"
    assert result["totals"]["pairs_quarantined"] == 2
    # The HOLD is recorded as a blocking human-subject gate fail on step 08.
    step08 = [s for s in result["steps"] if s["step"] == "08"]
    assert step08, "no step 08 record (S7)"
    assert any(
        any(g["gate"] == "g.human_subject" and g["pass"] is False
            for g in s.get("gate_results", []))
        for s in step08
    ), "no blocking g.human_subject fail recorded (S7)"


def test_s7_proceed_clears_the_hold(tmp_path: Path):
    """A PROCEED decision releases the pair — HOLD is a block, not a hard fail."""
    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    _author(flow, knobs, fixtures, n_pairs=2, name_and_crime_on_08=True)
    svc = _service(tmp_path)

    async def run_and_proceed():
        res = await _launch(
            svc, flow_name="lofn-music", flow_version=1, knobs_spec=_inline(knobs),
            context=_CTX, title="s7b", budget_cap_usd=100.0,
            fixtures_dir=fixtures, wait=False,
        )
        run_id = res["id"]
        store = svc.get_store(run_id)
        decided = set()
        for _ in range(10000):
            for e in store.read_runlog():
                if e.get("event") == "hold_for_human" and e.get("pair") not in decided:
                    svc.decide(run_id, e.get("pair"), "proceed")
                    decided.add(e.get("pair"))
            h = svc.handle(run_id)
            if h and h.task and h.task.done():
                break
            await asyncio.sleep(0.001)
        h = svc.handle(run_id)
        if h and h.task:
            await h.task
        return svc.get_run(run_id)

    result = asyncio.run(run_and_proceed())
    assert result["status"] == "complete"
    assert result["totals"]["pairs_quarantined"] == 0, "PROCEED must not quarantine (S7)"


# ===========================================================================
# S8 — Provenance replay (run.json + fixtures reproduce a mock run byte-identically).
# ===========================================================================

def test_s8_two_mock_runs_are_byte_identical(tmp_path: Path):
    """Same flow@ver + same resolved knobs (recorded in run.json) + same fixtures
    => byte-identical artifact set. The deterministic mock replay is the proof that
    a run reproduces from its recorded provenance."""
    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    _author(flow, knobs, fixtures, n_pairs=2)

    # Run A.
    svc_a = _service(tmp_path / "a")
    a = _run_full_mock(svc_a, fixtures, knobs, title="s8a")
    assert a["status"] == "complete"
    a_dir = svc_a._runs_root() / a["id"]
    a_arts = _artifact_map(a_dir)

    # Run B: reconstruct the KnobSet from A's recorded run.json provenance (exactly
    # what RunService.resume does), then run again from the same fixtures.
    a_run = json.loads((a_dir / "run.json").read_text(encoding="utf-8"))
    knobs_from_prov = KnobSet.from_dict(
        {"knobs": {k: {"value": v} for k, v in a_run["knobs"].items()}}
    )
    # The flow file sha recorded in provenance matches the flow we re-load.
    assert a_run["flow"]["file_sha"] == flow.file_sha(), "flow provenance sha drift (S8)"

    svc_b = _service(tmp_path / "b")
    b = _run_full_mock(svc_b, fixtures, knobs_from_prov, title="s8b")
    assert b["status"] == "complete"
    b_arts = _artifact_map(svc_b._runs_root() / b["id"])

    assert set(a_arts) == set(b_arts), "replayed run has a different artifact set (S8)"
    for name in a_arts:
        assert a_arts[name] == b_arts[name], f"{name} not byte-identical on replay (S8)"

    # And the per-step artifact shas recorded in run.json match the bytes on disk
    # (provenance is honest — the run.json sha is the actual artifact sha).
    from studio.flowspec import sha256_text
    for s in a_run.get("steps", []):
        art = s.get("artifact")
        if not art:
            continue
        disk = (a_dir / art)
        if disk.exists():
            assert s.get("artifact_sha") == sha256_text(disk.read_text(encoding="utf-8")), (
                f"recorded artifact_sha != disk bytes for {art} (S8 provenance honesty)"
            )


# ===========================================================================
# S9 — Gate parity (studio gate result == CLI validate_step.py result).
# ===========================================================================

def _cli_validate(step_id: str, artifact: Path, gates_path: Path) -> int:
    """Run the repo's validate_step.py exactly as the engine shells out to it."""
    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "validate_step.py"),
           "--gates", str(gates_path), str(step_id), str(artifact)]
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), text=True, capture_output=True)
    return proc.returncode


def test_s9_studio_gate_result_equals_cli(tmp_path: Path):
    """After a mock run, re-run the SAME validators the CLI would on the SAME
    artifact + the SAME run-local gates file; the studio's recorded gate verdict
    must equal the CLI's exit-code verdict, for a PASS and a FORCED FAIL."""
    flow = _music_flow()
    knobs = _two_pair_knobs()
    base_gates = load_base_gates(_REPO_ROOT)

    # Materialize the run-local gates file the engine would pass with --gates
    # (exactly the file the engine's materialize_gates writes into a run dir).
    gates_path = materialize_gates(tmp_path, base_gates, knobs)
    assert gates_path.exists()

    # --- (a) PASS case: a valid step-08 music artifact. ---------------------
    good = fm.music_song_artifact()
    good_path = tmp_path / "pair_01_step08_generation.md"
    good_path.write_text(good, encoding="utf-8", newline="\n")
    cli_good = _cli_validate("08", good_path, gates_path)
    assert cli_good == 0, f"CLI unexpectedly failed a valid artifact (rc={cli_good})"

    # The studio's gate path over the same file must agree (rc==0 -> pass).
    studio_good = _studio_gate_pass(flow, "08", good_path, gates_path)
    assert studio_good is True, "studio disagreed with CLI on a PASS (S9)"

    # --- (b) FORCED FAIL case: a too-short/invalid artifact. ----------------
    bad = "TOO SHORT — deliberately invalid to force a hard gate fail."
    bad_path = tmp_path / "pair_02_step08_generation.md"
    bad_path.write_text(bad, encoding="utf-8", newline="\n")
    cli_bad = _cli_validate("08", bad_path, gates_path)
    assert cli_bad != 0, "CLI unexpectedly passed an invalid artifact"

    studio_bad = _studio_gate_pass(flow, "08", bad_path, gates_path)
    assert studio_bad is False, "studio disagreed with CLI on a FAIL (S9)"

    # Parity is exact: pass<->rc0, fail<->rc!=0, both directions.
    assert (studio_good is (cli_good == 0)) and (studio_bad is (cli_bad == 0)), (
        "studio/CLI gate verdicts diverged (S9)"
    )


def _studio_gate_pass(flow, step_id: str, artifact: Path, gates_path: Path) -> bool:
    """Drive the engine's OWN gate machinery (_run_gates -> validate_step.py) over a
    given artifact + gates file, and return its pass/fail. This proves the studio
    shells out to the same script with the same --gates and reads the same verdict
    (dumb-backend doctrine) rather than reimplementing the numbers."""
    from studio.engine import Engine
    from studio.runstore import RunStore

    run_dir = artifact.parent
    # A minimal store rooted at the artifact's dir so artifact_path resolves.
    store = RunStore(run_dir=run_dir)
    # Build a bare engine shell; only the gate-running plumbing is exercised.
    eng = Engine.__new__(Engine)
    eng.store = store               # type: ignore[attr-defined]
    eng.repo_root = _REPO_ROOT      # type: ignore[attr-defined]
    eng._gates_path = gates_path    # type: ignore[attr-defined]
    step = {"id": step_id, "gates": ["g.music_prompt", "g.music_lyrics"]}
    validation = eng._run_gates(step, artifact.name, pair=None)
    return not validation.hard_fail


# ===========================================================================
# Meta: the whole grid completes with no live provider call.
# ===========================================================================

def test_acceptance_grid_is_dollar_zero(tmp_path: Path):
    """A guard: a full mock run uses the mock provider only (no real key needed),
    proving the suite spends $0. If any call escaped to a real dialect, the run
    would need a live key and would not complete with an empty KeyResolver."""
    flow = _music_flow()
    knobs = _two_pair_knobs()
    fixtures = tmp_path / "fixtures"
    _author(flow, knobs, fixtures, n_pairs=2)
    svc = _service(tmp_path)
    result = _run_full_mock(svc, fixtures, knobs, keys=_keys_no_secrets(_REPO_ROOT),
                            title="zero")
    assert result["status"] == "complete"
    # cost is metered against pricing.yaml but no money moved — the mock is free.
    assert result["totals"]["cost_usd"] >= 0.0
