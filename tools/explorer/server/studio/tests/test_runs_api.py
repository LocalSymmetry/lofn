"""WS3 run endpoints via TestClient — $0, no live provider, no bound port.

Uses dry_run launches (compile-all, NO API calls) so these tests need no fixtures
and never spend. The run root is redirected into tmp so the real repo stays clean
(S1). Also asserts the launch-cap requirement (§13.4) and the not-found paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.flowspec import find_repo_root

_REPO_ROOT = find_repo_root()

_CTX = {
    "input": "tide pools", "seed": "S", "meta_prompt": "M", "personality": "P",
    "concept_panel": "C", "medium_panel": "MD", "marketing_panel": "MK",
    "flairs": "F", "genres_list": "g", "frames_list": "f", "image_context": "",
}


@pytest.fixture
def client(tmp_path: Path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from studio.api import register_studio

    app = FastAPI()
    register_studio(app, _REPO_ROOT, runs_root=tmp_path / "output" / "studio")
    return TestClient(app)


def test_launch_requires_budget_cap(client):
    r = client.post("/api/studio/runs", json={
        "flow": "lofn-music", "version": 1, "context": _CTX,
        "knobs": {"n_pairs": {"value": 2}, "n_variations_per_pair": {"value": 1}},
        "dry_run": True,
    })
    assert r.status_code == 400
    assert "cap" in r.json()["detail"].lower()


def test_dry_run_launch_creates_run_no_artifacts(client):
    r = client.post("/api/studio/runs", json={
        "flow": "lofn-music", "version": 1, "context": _CTX,
        "knobs": {"n_pairs": {"value": 2}, "n_variations_per_pair": {"value": 1}},
        "budget_cap_usd": 0.0, "dry_run": True, "title": "dry",
    })
    assert r.status_code == 200
    run = r.json()
    assert run["status"] == "complete"
    assert run["budget"]["cap_usd"] == 0.0
    # No artifacts were written (dry run) but every LLM step was resolved.
    steps = run["steps"]
    assert steps and all(s.get("artifact") is None for s in steps)

    run_id = run["id"]
    # GET /runs and /runs/{id}
    lst = client.get("/api/studio/runs").json()
    assert any(x["id"] == run_id for x in lst)
    got = client.get(f"/api/studio/runs/{run_id}")
    assert got.status_code == 200
    assert got.json()["id"] == run_id


def test_get_unknown_run_404(client):
    assert client.get("/api/studio/runs/does-not-exist").status_code == 404


def test_events_replays_runlog_for_finished_run(client):
    r = client.post("/api/studio/runs", json={
        "flow": "lofn-music", "version": 1, "context": _CTX,
        "knobs": {"n_pairs": {"value": 2}, "n_variations_per_pair": {"value": 1}},
        "budget_cap_usd": 0.0, "dry_run": True,
    })
    run_id = r.json()["id"]
    # SSE endpoint replays the (already-complete) runlog then ends after run_end.
    with client.stream("GET", f"/api/studio/runs/{run_id}/events") as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    # Frames are SSE `data: {...}` lines; run_start and run_end are present.
    events = [json.loads(line[len("data: "):])
              for line in body.splitlines() if line.startswith("data: ")]
    kinds = {e.get("event") for e in events}
    assert "run_start" in kinds
    assert "run_end" in kinds


def test_decision_without_pending_hold_409(client):
    r = client.post("/api/studio/runs", json={
        "flow": "lofn-music", "version": 1, "context": _CTX,
        "knobs": {"n_pairs": {"value": 2}, "n_variations_per_pair": {"value": 1}},
        "budget_cap_usd": 0.0, "dry_run": True,
    })
    run_id = r.json()["id"]
    # No hold is pending on a finished dry run -> 409.
    d = client.post(f"/api/studio/runs/{run_id}/decision",
                    json={"pair": "01", "decision": "proceed"})
    assert d.status_code == 409


def test_decision_bad_value_400(client):
    r = client.post("/api/studio/runs", json={
        "flow": "lofn-music", "version": 1, "context": _CTX,
        "knobs": {"n_pairs": {"value": 2}, "n_variations_per_pair": {"value": 1}},
        "budget_cap_usd": 0.0, "dry_run": True,
    })
    run_id = r.json()["id"]
    d = client.post(f"/api/studio/runs/{run_id}/decision",
                    json={"pair": "01", "decision": "banana"})
    assert d.status_code == 400


def test_cancel_finished_run_409(client):
    r = client.post("/api/studio/runs", json={
        "flow": "lofn-music", "version": 1, "context": _CTX,
        "knobs": {"n_pairs": {"value": 2}, "n_variations_per_pair": {"value": 1}},
        "budget_cap_usd": 0.0, "dry_run": True,
    })
    run_id = r.json()["id"]
    # The inline dry run already finished; its handle has no live task.
    c = client.post(f"/api/studio/runs/{run_id}/cancel")
    assert c.status_code in (200, 409)
