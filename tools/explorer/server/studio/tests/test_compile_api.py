"""WS2 API: POST /api/studio/compile and /api/studio/gates/preview via TestClient.
$0 — no live provider, the server object is built in-process (never bound to a port)."""
from __future__ import annotations

import pytest

from studio.flowspec import find_repo_root

_REPO_ROOT = find_repo_root()

_CTX = {
    "input": "tide pools", "seed": "S", "meta_prompt": "M", "personality": "P",
    "concept_panel": "C", "medium_panel": "MD", "marketing_panel": "MK",
    "flairs": "F", "genres_list": "g", "frames_list": "f", "image_context": "",
}


@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from studio.api import register_studio

    app = FastAPI()
    register_studio(app, _REPO_ROOT)
    return TestClient(app)


def test_compile_endpoint_single_step(client):
    r = client.post("/api/studio/compile", json={
        "flow": "lofn-music", "version": 1, "step": "00", "context": _CTX,
    })
    assert r.status_code == 200
    j = r.json()
    assert j["run_allowed"] is True
    assert len(j["steps"]) == 1
    assert j["steps"][0]["resolved"]["user"]
    assert j["steps"][0]["est_tokens_in"] > 0


def test_compile_endpoint_diff(client):
    inline = {
        "n_pairs": {"value": 8}, "n_variations_per_pair": {"value": 4},
        "n_concepts": {"value": 16},
        "barbell_accessible": {"expr": "ceil(n_pairs / 2)", "override": None},
        "barbell_ambitious": {"expr": "n_pairs - barbell_accessible"},
        "total_prompts": {"expr": "n_pairs * n_variations_per_pair",
                          "gates_key": "total_prompts"},
    }
    r = client.post("/api/studio/compile", json={
        "flow": "lofn-music", "version": 1, "step": "00", "context": _CTX,
        "knobs": inline, "baseline": "classic-6x4",
    })
    assert r.status_code == 200
    diff = r.json()["steps"][0]["diff"]
    assert any(l.startswith("+") and "exactly 8" in l for l in diff.splitlines())
    assert any(l.startswith("-") and "exactly 6" in l for l in diff.splitlines())


def test_compile_endpoint_broken_preset_200_but_not_allowed(client):
    r = client.post("/api/studio/compile", json={
        "flow": "lofn-music", "version": 1, "step": "00", "context": _CTX,
        "knobs": {"n_pairs": {"value": 10}, "n_concepts": {"value": 12,
                                                           "min_expr": "2 * n_pairs"}},
    })
    assert r.status_code == 200
    assert r.json()["run_allowed"] is False


def test_compile_unknown_flow_404(client):
    r = client.post("/api/studio/compile", json={
        "flow": "lofn-nope", "version": 1, "step": "00", "context": _CTX,
    })
    assert r.status_code == 404


def test_gates_preview_reflects_knobs(client):
    r = client.post("/api/studio/gates/preview", json={
        "flow": "lofn-music", "version": 1,
        "knobs": {
            "n_pairs": {"value": 8}, "n_variations_per_pair": {"value": 4},
            "total_prompts": {"expr": "n_pairs * n_variations_per_pair",
                              "gates_key": "total_prompts"},
        },
    })
    assert r.status_code == 200
    j = r.json()
    assert j["gates"]["total_prompts"] == 32
    assert "house_lexicon" in j["gates"]     # canon passthrough
    assert j["bindings"]["total_prompts"] == 32


def test_gates_preview_default_preset(client):
    r = client.post("/api/studio/gates/preview", json={
        "flow": "lofn-music", "version": 1,
    })
    assert r.status_code == 200
    assert r.json()["gates"]["total_prompts"] == 24
