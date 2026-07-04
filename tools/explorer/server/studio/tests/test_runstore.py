"""WS3 runstore unit tests — canonical naming, incremental run.json (schema-valid),
append-only runlog, and KEY HYGIENE (S2): a planted canary key never survives into
run.json, the runlog, or an artifact. $0, no provider, no server.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.flowspec import find_repo_root, validate_run
from studio.runstore import (
    RunStore, build_initial_run_json, new_step_result,
    resolve_artifact_name, pair_label,
)

_REPO_ROOT = find_repo_root()
_CANARY = "sk-ant-canarycanarycanarycanary123456789"


def _initial(cap=25.0):
    return build_initial_run_json(
        run_id="run-test-0001", title="t",
        flow_name="lofn-music", flow_version=1, flow_file_sha="deadbeef",
        knobs_resolved={"n_pairs": 2, "n_variations_per_pair": 1, "total_prompts": 2},
        overlays=[], canon_sha="abc123", canon_dirty=False,
        context={"input": "brief"}, model_tiers={"pair": {"model": "claude-opus-4-8"}},
        budget_cap_usd=cap,
    )


def test_canonical_artifact_naming():
    assert pair_label(3) == "03"
    assert pair_label(None) is None
    assert resolve_artifact_name("pair_{pair}_step08_generation.md", 3) == "pair_03_step08_generation.md"
    assert resolve_artifact_name("step02_concepts.md", None) == "step02_concepts.md"
    assert resolve_artifact_name(None, 1) is None


def test_initial_run_json_is_schema_valid():
    errors = validate_run(_initial())
    assert errors == [], errors


def test_incremental_run_json_stays_schema_valid(tmp_path: Path):
    store = RunStore.create(tmp_path, "run-test-0001", _initial())
    store.set_status("running")
    rec = new_step_result(stage="coordinator", step="00", pair=None,
                          provider="anthropic", model="claude-opus-4-8", status="done")
    rec["tokens_in"] = 100
    rec["tokens_out"] = 200
    rec["cost_usd"] = 0.0055
    rec["duration_s"] = 1.2
    rec["attempts"] = 1
    rec["artifact"] = "step00_aesthetics_genres.md"
    rec["artifact_sha"] = "cafe"
    rec["gate_results"] = [{"gate": "g.step00", "pass": True, "detail": ""}]
    store.upsert_step(rec)
    store.set_totals(tokens_in=100, tokens_out=200, cost_usd=0.0055)
    store.set_budget_spend(0.0055)

    on_disk = json.loads(store.run_json_path.read_text(encoding="utf-8"))
    assert validate_run(on_disk) == [], "run.json drifted from schema after updates"
    assert on_disk["status"] == "running"
    assert on_disk["steps"][0]["step"] == "00"


def test_upsert_replaces_not_duplicates(tmp_path: Path):
    store = RunStore.create(tmp_path, "r", _initial())
    a = new_step_result(stage="pairs", step="08", pair="01", status="retrying")
    store.upsert_step(a)
    b = new_step_result(stage="pairs", step="08", pair="01", status="done")
    store.upsert_step(b)
    steps = json.loads(store.run_json_path.read_text(encoding="utf-8"))["steps"]
    matching = [s for s in steps if s["step"] == "08" and s["pair"] == "01"]
    assert len(matching) == 1
    assert matching[0]["status"] == "done"


def test_runlog_is_append_only_and_replayable(tmp_path: Path):
    store = RunStore.create(tmp_path, "r", _initial())
    store.log({"event": "run_start"})
    store.log({"event": "step", "step": "00", "status": "done"})
    store.log({"event": "run_end", "status": "complete"})
    events = store.read_runlog()
    assert [e["event"] for e in events] == ["run_start", "step", "run_end"]
    # every event carries a ts
    assert all("ts" in e for e in events)


def test_canary_key_never_survives(tmp_path: Path):
    """S2: a planted key is scrubbed from run.json, runlog, AND artifacts."""
    store = RunStore.create(tmp_path, "r", _initial(), secrets=[_CANARY])
    # try to leak it via every write path
    store.update_run(title=f"leak {_CANARY} here")
    store.log({"event": "oops", "note": f"key={_CANARY}"})
    store.write_artifact("step00_aesthetics_genres.md",
                         f"the model echoed {_CANARY} back into the artifact")

    run_text = store.run_json_path.read_text(encoding="utf-8")
    log_text = store.runlog_path.read_text(encoding="utf-8")
    art_text = (tmp_path / "r" / "step00_aesthetics_genres.md").read_text(encoding="utf-8")

    assert _CANARY not in run_text
    assert _CANARY not in log_text
    assert _CANARY not in art_text
    assert "[REDACTED]" in run_text or "[REDACTED]" in log_text


def test_artifact_path_is_jailed(tmp_path: Path):
    store = RunStore.create(tmp_path, "r", _initial())
    with pytest.raises(ValueError):
        store.artifact_path("../escape.md")


def test_atomic_run_json_is_always_parseable(tmp_path: Path):
    """The run.json write is atomic (temp + replace) so a resume can always parse it."""
    store = RunStore.create(tmp_path, "r", _initial())
    for i in range(20):
        store.set_totals(tokens_in=i, tokens_out=i * 2)
        # each read mid-stream must parse
        json.loads(store.run_json_path.read_text(encoding="utf-8"))
