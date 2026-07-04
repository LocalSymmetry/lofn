"""Judge + promote API routes via TestClient — $0, no live provider, no port.

The run root is redirected into tmp so no real run is needed / mutated. Promote
preview/brief hit the real canon vault/gates.yaml READ-ONLY (preview writes
nothing); apply is exercised in test_promote.py against a temp copy, never here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.flowspec import find_repo_root

_REPO_ROOT = find_repo_root()

_CLEAN = "## 1. MUSIC PROMPT\n\n" + (
    "Dream-pop with shoegaze texture at a patient tempo over layered guitars "
    "washed in reverb and slow tremolo, a rounded analog bass breathing with the "
    "kick while brushed drums keep a soft tidal pulse under wide blooming pads. "
) * 8 + "\n"


@pytest.fixture
def client(tmp_path: Path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from studio.api import register_studio

    app = FastAPI()
    runs_root = tmp_path / "output" / "studio"
    register_studio(app, _REPO_ROOT, runs_root=runs_root)
    return TestClient(app), runs_root


def _seed_run(runs_root: Path, run_id: str) -> None:
    d = runs_root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "pair_01_step08_generation.md").write_text(_CLEAN, encoding="utf-8", newline="\n")
    (d / "run.json").write_text(json.dumps({
        "id": run_id, "title": run_id,
        "steps": [{"step": "08", "pair": "01",
                   "artifact": "pair_01_step08_generation.md",
                   "gate_results": [{"gate": "g.music_prompt", "pass": True}]}],
        "knobs": {"n_pairs": 6}, "totals": {"cost_usd": 1.0, "tokens_in": 10, "tokens_out": 5},
        "budget": {"cap_usd": 100.0, "spend_usd": 1.0}, "status": "complete", "verdicts": [],
    }, indent=2) + "\n", encoding="utf-8", newline="\n")


def test_judge_endpoint_returns_blind_and_compare(client):
    c, runs_root = client
    _seed_run(runs_root, "run-A")
    _seed_run(runs_root, "run-B")
    r = c.post("/api/studio/judge", json={"run_a": "run-A", "run_b": "run-B", "seed": 3})
    assert r.status_code == 200
    body = r.json()
    assert "compare" in body and "blind" in body
    assert body["blind"]["n_matchups"] == 1
    # cards carry no provenance
    for m in body["blind"]["matchups"]:
        assert set(m.keys()) == {"matchup_id", "A", "B"}


def test_judge_verdict_appends_both(client):
    c, runs_root = client
    _seed_run(runs_root, "run-A")
    _seed_run(runs_root, "run-B")
    r = c.post("/api/studio/judge/verdict", json={
        "run_a": "run-A", "run_b": "run-B",
        "verdict": {"matchup_id": "M01", "winner": "B"},
    })
    assert r.status_code == 200
    for rid in ("run-A", "run-B"):
        data = json.loads((runs_root / rid / "run.json").read_text(encoding="utf-8"))
        assert data["verdicts"] and data["verdicts"][0]["winner"] == "B"


def test_judge_unknown_run_404(client):
    c, _ = client
    r = c.post("/api/studio/judge", json={"run_a": "nope", "run_b": "also-nope"})
    assert r.status_code == 404


def test_promote_preview_reads_canon_readonly(client):
    c, _ = client
    before = (_REPO_ROOT / "vault" / "gates.yaml").read_bytes()
    r = c.post("/api/studio/promote/preview",
               json={"gate_values": {"music_prompt_hug_ceiling": 990}})
    assert r.status_code == 200
    assert r.json()["gates"]["changed"] is True
    # canon untouched by a preview
    assert (_REPO_ROOT / "vault" / "gates.yaml").read_bytes() == before


def test_promote_brief_endpoint(client):
    c, _ = client
    r = c.post("/api/studio/promote/brief", json={
        "run_a": "run-A", "run_b": "run-B", "title": "Wide portfolio",
        "structural_changes": [{"kind": "new gate", "where": "step 05", "what": "band widened"}],
        "knob_delta": [{"knob": "n_pairs", "a": 6, "b": 10}],
    })
    assert r.status_code == 200
    assert "PROMOTION BRIEF" in r.json()["brief"]
