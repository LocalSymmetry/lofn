"""Compare & Judge — aligned diffs, gate/cost/knob deltas, and a fail-closed
blind set (plan §8.4 / §9, WS6 backend half).

Two operations, both over two finished runs on disk (``output/studio/<id>/``):

  * :func:`compare` — the OPEN comparison: per-artifact aligned diffs (matched by
    canonical name across the two runs), the gate-report delta (per step: pass/
    fail/flag transitions), and the cost / token / knob deltas. Full provenance;
    this is the honest side-by-side.

  * :func:`blind_judge` — the BLIND comparison: payload-only cards that REUSE the
    extraction + FAIL-CLOSED doctrine of ``scripts/build_blind_set.py`` (Lesson
    L10). Every provenance string — pair ids, run ids, verdicts, self-checks,
    Major Deviations, byte counts — is stripped. If a payload cannot be cleanly
    extracted (too short after cleaning, or still carrying a provenance leak) the
    card FAILS CLOSED: it is NOT emitted (never a self-identifying card that would
    VOID the calibration). A/B sides are randomized with an explicit seed so the
    verdict can't lean on side.

:func:`append_verdict` writes a recorded verdict onto BOTH runs' ``run.json``
(``verdicts`` list) via the store's byte-safe append — the OEC is the operator's taste,
recorded honestly at n=1.

DUMB BACKEND: this module reads run artifacts + run.json; it never re-runs a gate
(the recorded ``gate_results`` on each step are the authority) and never writes
outside ``output/studio/**``.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .runstore import RunStore

# ---------------------------------------------------------------------------
# Blind-set extraction doctrine — ported from scripts/build_blind_set.py.
# The regex + cleaning + fail-closed thresholds are kept in lock-step with the
# script so a card built here obeys the exact same L10 rules.
# ---------------------------------------------------------------------------

MIN_BYTES = 1500

# lines that leak provenance — dropped everywhere (verbatim from build_blind_set).
_LEAK = re.compile(
    r"(?i)(pair[ _]?0?\d|run:|source:|golden|staff pick|era-mate|suno url|"
    r"publication status|verdict|self-check|major deviation|measured|"
    r"gate_report|provenance|continuity payload|special flairs|icb|"
    r"selected_for_top_six|step ?1[01]|variation angle|bytes)"
)


def _clean(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not _LEAK.search(l)).strip()


def _payload(chunk: str) -> str:
    """Keep only prompt/exclude/lyrics/theme sections of a chunk, cleaned.

    Mirrors ``build_blind_set._payload``: split on markdown headings, keep the
    sections whose heading names a payload field (or a bracketed block), then run
    the leak-line filter over what remains.
    """
    keep: List[str] = []
    for sec in re.split(r"(?=^#+ )", chunk, flags=re.M):
        head = sec.splitlines()[0] if sec.splitlines() else ""
        if re.search(r"(?i)(music prompt|exclude|lyrics|theme)", head) or \
                sec.lstrip().startswith("["):
            keep.append(sec)
    return _clean("\n".join(keep) if keep else chunk)


def extract_payload(text: str) -> Optional[str]:
    """Extract the bare, provenance-stripped payload from an artifact.

    FAILS CLOSED (returns None) when the cleaned payload is under ``MIN_BYTES`` or
    STILL carries a provenance leak after cleaning — exactly the two conditions
    ``build_blind_set.py`` treats as fatal. A None here means "do not emit a card
    for this artifact" (never a self-identifying stub that voids the set)."""
    body = _payload(text)
    if len(body.encode("utf-8")) < MIN_BYTES:
        return None
    if any(_LEAK.search(l) for l in body.splitlines()):
        return None
    return body


# ---------------------------------------------------------------------------
# Run loading helpers.
# ---------------------------------------------------------------------------

def _load_run_json(run_dir: Path) -> Dict[str, Any]:
    rj = run_dir / "run.json"
    if not rj.exists():
        raise FileNotFoundError(f"run.json not found under {run_dir}")
    return json.loads(rj.read_text(encoding="utf-8"))


def _final_artifacts(run_dir: Path) -> Dict[str, Path]:
    """Canonical artifact name -> path for a run's step outputs.

    Only the real step artifacts (``step*.md`` / ``pair_*_step*.md``), never the
    ``.repair_attempt_N.md`` scratch files or run.json / gates.yaml.
    """
    out: Dict[str, Path] = {}
    for p in sorted(run_dir.glob("*.md")):
        n = p.name
        if ".repair_attempt_" in n:
            continue
        if n.startswith("step") or (n.startswith("pair_") and "_step" in n):
            out[n] = p
    return out


def _steps_by_key(run: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map an artifact name -> its recorded step record (for the gate delta)."""
    out: Dict[str, Dict[str, Any]] = {}
    for s in run.get("steps", []):
        art = s.get("artifact")
        if art:
            out[art] = s
    return out


# ---------------------------------------------------------------------------
# OPEN comparison — aligned diffs + gate/cost/knob deltas.
# ---------------------------------------------------------------------------

def _line_diff(a: str, b: str) -> Dict[str, Any]:
    """A compact unified diff summary (added / removed line counts + the hunks).

    We keep it dependency-light: a difflib unified diff, plus counts. The UI can
    render the raw unified block in a DiffPanel (same as v1)."""
    import difflib
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    ud = list(difflib.unified_diff(a_lines, b_lines, lineterm="", n=3))
    added = sum(1 for l in ud if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in ud if l.startswith("-") and not l.startswith("---"))
    return {
        "added": added,
        "removed": removed,
        "identical": a == b,
        "unified": "\n".join(ud),
    }


def _gate_summary(step: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """pass/fail/flag counts from a step's recorded gate_results (None == flag)."""
    out = {"pass": 0, "fail": 0, "flag": 0}
    if not step:
        return out
    for g in step.get("gate_results", []) or []:
        v = g.get("pass")
        if v is True:
            out["pass"] += 1
        elif v is False:
            out["fail"] += 1
        else:
            out["flag"] += 1
    return out


def compare(run_a: Path, run_b: Path) -> Dict[str, Any]:
    """OPEN side-by-side of two runs.

    Returns aligned per-artifact diffs (matched by canonical name), a gate-report
    delta per matched step, and cost/token/knob deltas. Full provenance — this is
    the honest comparison, NOT the blind one.
    """
    run_a, run_b = Path(run_a), Path(run_b)
    ja, jb = _load_run_json(run_a), _load_run_json(run_b)
    arts_a, arts_b = _final_artifacts(run_a), _final_artifacts(run_b)
    steps_a, steps_b = _steps_by_key(ja), _steps_by_key(jb)

    names = sorted(set(arts_a) | set(arts_b))
    artifacts: List[Dict[str, Any]] = []
    for name in names:
        pa = arts_a.get(name)
        pb = arts_b.get(name)
        ta = pa.read_text(encoding="utf-8") if pa else ""
        tb = pb.read_text(encoding="utf-8") if pb else ""
        entry: Dict[str, Any] = {
            "artifact": name,
            "in_a": pa is not None,
            "in_b": pb is not None,
            "diff": _line_diff(ta, tb),
            "gate_delta": {
                "a": _gate_summary(steps_a.get(name)),
                "b": _gate_summary(steps_b.get(name)),
            },
        }
        artifacts.append(entry)

    ta_tot = ja.get("totals") or {}
    tb_tot = jb.get("totals") or {}

    def _num(d: Dict[str, Any], k: str) -> float:
        v = d.get(k)
        return float(v) if isinstance(v, (int, float)) else 0.0

    cost_delta = {
        "a_cost_usd": _num(ta_tot, "cost_usd"),
        "b_cost_usd": _num(tb_tot, "cost_usd"),
        "delta_cost_usd": round(_num(tb_tot, "cost_usd") - _num(ta_tot, "cost_usd"), 6),
        "a_tokens_in": int(_num(ta_tot, "tokens_in")),
        "b_tokens_in": int(_num(tb_tot, "tokens_in")),
        "a_tokens_out": int(_num(ta_tot, "tokens_out")),
        "b_tokens_out": int(_num(tb_tot, "tokens_out")),
        "delta_tokens_in": int(_num(tb_tot, "tokens_in") - _num(ta_tot, "tokens_in")),
        "delta_tokens_out": int(_num(tb_tot, "tokens_out") - _num(ta_tot, "tokens_out")),
    }

    ka = ja.get("knobs") or {}
    kb = jb.get("knobs") or {}
    knob_delta: List[Dict[str, Any]] = []
    for k in sorted(set(ka) | set(kb)):
        va, vb = ka.get(k), kb.get(k)
        if va != vb:
            knob_delta.append({"knob": k, "a": va, "b": vb})

    return {
        "run_a": ja.get("id"),
        "run_b": jb.get("id"),
        "artifacts": artifacts,
        "cost_delta": cost_delta,
        "knob_delta": knob_delta,
    }


# ---------------------------------------------------------------------------
# BLIND comparison — payload-only cards, fail-closed, randomized sides.
# ---------------------------------------------------------------------------

@dataclass
class BlindMatchup:
    matchup_id: str
    artifact: str                 # the canonical name the two payloads came from
    side_a: str                   # payload text shown as "A"
    side_b: str                   # payload text shown as "B"
    # The coordinator-only key: which run each side actually came from. This is
    # NOT part of the blind cards served to a judge — it is returned SEPARATELY so
    # a verdict can be de-blinded, mirroring build_blind_set's out-of-band key.
    _a_run: str = field(default="", repr=False)
    _b_run: str = field(default="", repr=False)


def blind_judge(run_a: Path, run_b: Path, *, seed: int = 0) -> Dict[str, Any]:
    """Build a fail-closed blind set from two runs' final artifacts.

    For every artifact present in BOTH runs, extract each side's provenance-
    stripped payload. A matchup is emitted ONLY when BOTH payloads extract cleanly
    (fail-closed: if either side can't be cleanly extracted, the whole matchup is
    dropped — no self-identifying card, never a stub). A/B sides are randomized
    per matchup with the explicit seed so a verdict can't key on side.

    Returns ``{"matchups": [...cards without run ids...], "key": {matchup_id ->
    {"A": run_id, "B": run_id, "artifact": name}}, "dropped": [...]}``. The
    ``key`` is the coordinator-only de-blinding map (kept OUT of the cards).
    """
    run_a, run_b = Path(run_a), Path(run_b)
    ja, jb = _load_run_json(run_a), _load_run_json(run_b)
    id_a = ja.get("id") or run_a.name
    id_b = jb.get("id") or run_b.name
    arts_a, arts_b = _final_artifacts(run_a), _final_artifacts(run_b)

    rng = random.Random(seed)
    common = sorted(set(arts_a) & set(arts_b))

    cards: List[Dict[str, Any]] = []
    key: Dict[str, Dict[str, str]] = {}
    dropped: List[Dict[str, str]] = []

    for i, name in enumerate(common, 1):
        pa = extract_payload(arts_a[name].read_text(encoding="utf-8"))
        pb = extract_payload(arts_b[name].read_text(encoding="utf-8"))
        # FAIL CLOSED: both sides must extract cleanly, else drop the matchup.
        if pa is None or pb is None:
            dropped.append({
                "artifact": name,
                "reason": "payload failed clean extraction (fail-closed); "
                          "matchup not emitted",
                "a_ok": str(pa is not None),
                "b_ok": str(pb is not None),
            })
            continue
        mid = f"M{i:02d}"
        # Randomize which run lands on side A.
        if rng.random() < 0.5:
            side_a, side_b, a_run, b_run = pa, pb, id_a, id_b
        else:
            side_a, side_b, a_run, b_run = pb, pa, id_b, id_a
        cards.append({
            "matchup_id": mid,
            # NB: the card carries NO run id / pair id / verdict — only the two
            # payloads and a neutral matchup label the judge picks a winner in.
            "A": side_a,
            "B": side_b,
        })
        key[mid] = {"A": a_run, "B": b_run, "artifact": name}

    return {
        "run_a": id_a,
        "run_b": id_b,
        "seed": seed,
        "matchups": cards,
        "key": key,
        "dropped": dropped,
        "n_matchups": len(cards),
        "n_dropped": len(dropped),
    }


# ---------------------------------------------------------------------------
# Verdict recording — append to BOTH runs' run.json.
# ---------------------------------------------------------------------------

def append_verdict(
    run_a: Path,
    run_b: Path,
    verdict: Dict[str, Any],
    *,
    secrets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Append ``verdict`` to BOTH runs' ``run.json`` ``verdicts`` list (byte-safe).

    Uses the run store's append path (redaction + atomic write) on each run so a
    key can never leak into a recorded verdict. The SAME verdict object is written
    to both (with an added ``ts``) so the two runs agree on the matchup outcome.
    Returns the stored verdict (with ``ts``).
    """
    from .runstore import utcnow_iso
    rec = dict(verdict)
    rec.setdefault("kind", "blind_judge")
    rec.setdefault("ts", utcnow_iso())
    for run_dir in (Path(run_a), Path(run_b)):
        if not (run_dir / "run.json").exists():
            raise FileNotFoundError(f"run.json not found under {run_dir}")
        store = RunStore.open(run_dir, secrets=list(secrets or []))
        store.append_verdict(dict(rec))
    return rec
