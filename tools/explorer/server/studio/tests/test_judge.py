"""WS6 judge tests — $0, no server, no live provider.

Covers:
  * blind_judge FAILS CLOSED on a dirty payload (a leaking artifact never emits a
    self-identifying card; a clean pair does),
  * extract_payload's fail-closed thresholds (too short / still leaking),
  * append_verdict lands the SAME verdict in BOTH runs' run.json,
  * compare returns aligned diffs + gate/cost/knob deltas.

Runs are hand-authored on disk (a minimal run.json + a couple of artifacts) so the
test needs no engine run and spends nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio import judge


# A payload big enough to clear MIN_BYTES (1500) with only payload-field headings.
_CLEAN_PROMPT = "## 1. MUSIC PROMPT\n\n" + (
    "Dream-pop with shoegaze texture at a patient tempo, an adult vocalist singing "
    "close and unhurried over layered guitars washed in reverb and slow tremolo. "
    "The low end is a rounded analog bass that breathes with the kick while brushed "
    "drums keep a soft, tidal pulse under wide stereo pads that bloom in the chorus. "
) * 6 + "\n\n## 2. LYRICS\n\n[Theme: tide pools at dawn]\n" + (
    "the tide pulls the morning thin and the light bends slow across the wet stone\n"
) * 20


# A payload that CANNOT be cleanly extracted: nearly every line is provenance, so
# after the leak-line strip the remaining body falls under MIN_BYTES -> fail closed
# (the build_blind_set L10 doctrine: a stub/too-thin card VOIDS the set).
_DIRTY = "## 1. MUSIC PROMPT\n\n" + "\n".join(
    f"pair 0{i%9+1} verdict SHIP self-check measured bytes provenance line {i}"
    for i in range(60)
) + "\n\nonly this one short tail line survives cleaning\n"


def _write_run(run_dir: Path, run_id: str, artifacts: dict[str, str],
               *, cost: float = 1.0, knobs: dict | None = None,
               gate_results: dict | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = []
    for name in artifacts:
        (run_dir / name).write_text(artifacts[name], encoding="utf-8", newline="\n")
        steps.append({
            "step": name.split("step")[-1][:2] if "step" in name else "00",
            "pair": None, "artifact": name,
            "gate_results": (gate_results or {}).get(name, [{"gate": "g.x", "pass": True}]),
        })
    run_json = {
        "id": run_id, "title": run_id, "steps": steps,
        "knobs": knobs or {"n_pairs": 6},
        "totals": {"cost_usd": cost, "tokens_in": 100, "tokens_out": 50},
        "budget": {"cap_usd": 100.0, "spend_usd": cost},
        "status": "complete", "verdicts": [],
    }
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")


def test_extract_payload_fail_closed(tmp_path: Path):
    # Clean payload extracts.
    assert judge.extract_payload(_CLEAN_PROMPT) is not None
    # Too-short payload fails closed.
    assert judge.extract_payload("## 1. MUSIC PROMPT\n\ntiny") is None
    # A payload that still carries a leak line fails closed (never a self-id card).
    assert judge.extract_payload(_DIRTY) is None


def test_blind_judge_drops_dirty_matchup(tmp_path: Path):
    a = tmp_path / "run-A"
    b = tmp_path / "run-B"
    # step08 clean in both; step10 clean in A but DIRTY in B -> that matchup drops.
    _write_run(a, "run-A", {
        "pair_01_step08_generation.md": _CLEAN_PROMPT,
        "pair_01_step10_generation.md": _CLEAN_PROMPT,
    })
    _write_run(b, "run-B", {
        "pair_01_step08_generation.md": _CLEAN_PROMPT,
        "pair_01_step10_generation.md": _DIRTY,
    })

    res = judge.blind_judge(a, b, seed=7)
    ids = {m["matchup_id"] for m in res["matchups"]}
    # The clean step08 pair emits a card; the dirty step10 pair is dropped.
    assert res["n_matchups"] == 1
    assert res["n_dropped"] == 1
    # No card carries any run id / pair id / verdict string.
    for m in res["matchups"]:
        assert set(m.keys()) == {"matchup_id", "A", "B"}
        blob = (m["A"] + m["B"]).lower()
        assert "run-a" not in blob and "run-b" not in blob
        assert "verdict" not in blob and "pair 0" not in blob
    # The de-blinding key is coordinator-only and covers exactly the emitted cards.
    assert set(res["key"].keys()) == ids
    for mid, k in res["key"].items():
        assert set(k["A"].split()) and set(k["B"].split())
        assert {k["A"], k["B"]} == {"run-A", "run-B"}


def test_blind_judge_randomizes_sides_by_seed(tmp_path: Path):
    a = tmp_path / "run-A"
    b = tmp_path / "run-B"
    art = {"pair_01_step08_generation.md": _CLEAN_PROMPT + "\nAAA distinct A side\n"}
    art_b = {"pair_01_step08_generation.md": _CLEAN_PROMPT + "\nBBB distinct B side\n"}
    _write_run(a, "run-A", art)
    _write_run(b, "run-B", art_b)
    # Different seeds can flip which run is on side A; the key always de-blinds it.
    r0 = judge.blind_judge(a, b, seed=0)
    r1 = judge.blind_judge(a, b, seed=1)
    for r in (r0, r1):
        m = r["matchups"][0]
        key = r["key"][m["matchup_id"]]
        # Side A's payload came from key["A"]'s run.
        a_side_is_run_a = "AAA distinct A side" in m["A"]
        assert (key["A"] == "run-A") == a_side_is_run_a


def test_append_verdict_lands_in_both_runs(tmp_path: Path):
    a = tmp_path / "run-A"
    b = tmp_path / "run-B"
    _write_run(a, "run-A", {"pair_01_step08_generation.md": _CLEAN_PROMPT})
    _write_run(b, "run-B", {"pair_01_step08_generation.md": _CLEAN_PROMPT})

    rec = judge.append_verdict(a, b, {"matchup_id": "M01", "winner": "A", "note": "warmer"})
    assert rec["ts"]
    for run_dir in (a, b):
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert len(data["verdicts"]) == 1
        v = data["verdicts"][0]
        assert v["matchup_id"] == "M01" and v["winner"] == "A"


def test_compare_aligned_diffs_and_deltas(tmp_path: Path):
    a = tmp_path / "run-A"
    b = tmp_path / "run-B"
    _write_run(a, "run-A", {"step02_concepts.md": "alpha\nbeta\ngamma\n"},
               cost=1.0, knobs={"n_pairs": 6})
    _write_run(b, "run-B", {"step02_concepts.md": "alpha\nBETA\ngamma\ndelta\n"},
               cost=2.5, knobs={"n_pairs": 10})

    res = judge.compare(a, b)
    assert res["run_a"] == "run-A" and res["run_b"] == "run-B"
    art = next(x for x in res["artifacts"] if x["artifact"] == "step02_concepts.md")
    assert art["in_a"] and art["in_b"]
    assert not art["diff"]["identical"]
    assert art["diff"]["added"] >= 1 and art["diff"]["removed"] >= 1
    # cost + knob deltas
    assert res["cost_delta"]["delta_cost_usd"] == pytest.approx(1.5)
    knob_delta = {d["knob"]: (d["a"], d["b"]) for d in res["knob_delta"]}
    assert knob_delta["n_pairs"] == (6, 10)
