"""Run store — the on-disk home of a studio run (plan §5, WS3).

Every run lives under ``output/studio/<run-id>/`` and owns three kinds of file:

  * ``run.json``      — full provenance, written INCREMENTALLY (schema:
                        schemas/run.schema.json, the frozen WS0 contract). Flow
                        name+version+file-sha; fully-resolved knobs (incl.
                        derived); overlays+shas; canon git sha + dirty flag;
                        context refs+shas; per-step provider/model/tokens/cost/
                        duration/attempts/gate-results/artifact-sha; totals;
                        budget cap+spend; status; judge verdicts.
  * ``runlog.jsonl``  — APPEND-ONLY. One JSON line per state transition + usage
                        event. This is the single event stream with TWO consumers
                        (plan §5): SSE tails it live; the UI replays it for a
                        finished run.
  * artifacts         — the step outputs, written under CANONICAL names
                        (``pair_03_step08_generation.md``, ``step02_concepts.md``)
                        so every existing validator, ``rebuild_manifest.py`` and
                        even ``lofn-qa`` work on a studio run unchanged.
  * ``gates.yaml``    — the run-local, knob-adjusted gates the validators read via
                        ``--gates`` (materialized by :mod:`knobs`).

KEYS NEVER APPEAR (acceptance gate S2): every write path routes through the
redaction filter in :mod:`keys` before it touches ``run.json``, the runlog, or an
SSE frame. The runstore is handed the live key strings so it can scrub exact
literals too (belt-and-suspenders).

CANON IMMUTABILITY (S1): the engine OWNS ``output/studio/**`` and holds no write
handle anywhere else. Every path this module constructs lives under the run dir;
it never calls the v1 FileService and never escapes the run root.

DISK IS AUTHORITY (rebuild_manifest doctrine): on resume the store re-scans the
run dir and trusts valid artifacts — there is no separate index to drift.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .flowspec import sha256_bytes, sha256_text
from .keys import redact, redact_with_values


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp (schema: run.json.created)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Canonical artifact naming.
# ---------------------------------------------------------------------------

def pair_label(pair: Optional[int]) -> Optional[str]:
    """Two-digit zero-padded pair label (``3`` -> ``"03"``) or None.

    Canonical names use ``pair_03_...`` (rebuild_manifest's ``_PAIR_RE`` zfills to
    2; the plan example is ``pair_03_step08_generation.md``).
    """
    if pair is None:
        return None
    return str(int(pair)).zfill(2)


def resolve_artifact_name(emits: Optional[str], pair: Optional[int]) -> Optional[str]:
    """Fill the flow step's ``emits`` template with the pair label.

    A fan-out step declares ``emits: pair_{pair}_step06_facets.md``; a coordinator
    step declares a plain name (``step00_aesthetics_genres.md``). Returns None for
    a validator step (no ``emits``).
    """
    if not emits:
        return None
    lab = pair_label(pair)
    if "{pair}" in emits:
        return emits.replace("{pair}", lab if lab is not None else "00")
    return emits


# ---------------------------------------------------------------------------
# The store.
# ---------------------------------------------------------------------------

@dataclass
class RunStore:
    """Owns one run's directory: run.json, runlog.jsonl, artifacts, gates.yaml.

    Thread-safe for the fan-out (the engine writes from several pair coroutines
    concurrently); a single lock guards run.json mutation and the runlog append.
    ``secrets`` are the live resolved key strings — used ONLY to scrub exact
    literals out of every text write (never persisted).
    """

    run_dir: Path
    secrets: List[str] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _run: Dict[str, Any] = field(default_factory=dict, repr=False)
    # Listeners (SSE tails) get every appended runlog event live.
    _listeners: List[Callable[[dict], None]] = field(default_factory=list, repr=False)

    # -- construction -------------------------------------------------------
    @classmethod
    def create(
        cls,
        runs_root: Path,
        run_id: str,
        run_json: Dict[str, Any],
        *,
        secrets: Optional[List[str]] = None,
    ) -> "RunStore":
        """Create a fresh run dir + initial run.json + empty runlog."""
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        store = cls(run_dir=run_dir, secrets=list(secrets or []))
        store._run = dict(run_json)
        store._flush_run()
        # touch the runlog so an SSE tail can open it immediately
        store.runlog_path.write_text("", encoding="utf-8", newline="\n")
        return store

    @classmethod
    def open(cls, run_dir: Path, *, secrets: Optional[List[str]] = None) -> "RunStore":
        """Re-open an existing run (resume). Loads run.json from disk (authority)."""
        store = cls(run_dir=run_dir, secrets=list(secrets or []))
        rj = run_dir / "run.json"
        if rj.exists():
            store._run = json.loads(rj.read_text(encoding="utf-8"))
        return store

    # -- paths --------------------------------------------------------------
    @property
    def run_json_path(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def runlog_path(self) -> Path:
        return self.run_dir / "runlog.jsonl"

    @property
    def run_id(self) -> str:
        return self._run.get("id", self.run_dir.name)

    # -- run.json snapshot --------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """A redacted deep copy of the current run.json (safe to serve)."""
        with self._lock:
            return redact(json.loads(json.dumps(self._run)))

    def raw_run(self) -> Dict[str, Any]:
        """The in-memory run dict WITHOUT redaction (internal engine use only —
        never returned by an endpoint)."""
        with self._lock:
            return json.loads(json.dumps(self._run))

    # -- atomic writes ------------------------------------------------------
    def _atomic_write(self, path: Path, text: str) -> None:
        """Write a file atomically (temp + os.replace) so a killed engine never
        leaves a half-written run.json — the resume path can always parse it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _scrub(self, obj: Any) -> Any:
        """Redact by shape AND by exact secret literal (both passes, S2)."""
        red = redact(obj)
        if self.secrets:
            # exact-literal scrub over the serialized text, then re-parse
            blob = redact_with_values(json.dumps(red, ensure_ascii=False), self.secrets)
            return json.loads(blob)
        return red

    def _flush_run(self) -> None:
        scrubbed = self._scrub(self._run)
        self._atomic_write(self.run_json_path, json.dumps(scrubbed, indent=2, ensure_ascii=False) + "\n")

    # -- run.json mutation --------------------------------------------------
    def update_run(self, **fields) -> None:
        """Shallow-merge top-level fields into run.json and flush."""
        with self._lock:
            self._run.update(fields)
            self._flush_run()

    def set_status(self, status: str) -> None:
        self.update_run(status=status)
        self.log({"event": "run_status", "status": status})

    def set_totals(self, **totals) -> None:
        with self._lock:
            cur = dict(self._run.get("totals") or {})
            cur.update(totals)
            self._run["totals"] = cur
            self._flush_run()

    def set_budget_spend(self, spend_usd: float, *, aborted_on_cap: Optional[bool] = None) -> None:
        with self._lock:
            b = dict(self._run.get("budget") or {})
            b["spend_usd"] = round(spend_usd, 6)
            if aborted_on_cap is not None:
                b["aborted_on_cap"] = aborted_on_cap
            self._run["budget"] = b
            self._flush_run()

    def budget_cap(self) -> Optional[float]:
        with self._lock:
            cap = (self._run.get("budget") or {}).get("cap_usd")
            return float(cap) if cap is not None else None

    def upsert_step(self, step_result: Dict[str, Any]) -> None:
        """Insert or replace a per-step provenance record (keyed by step+pair).

        Steps are appended as the run progresses; re-running a step (retry/resume)
        REPLACES its record rather than duplicating it, so run.json stays a clean
        current-state view. Matching is on (stage?, step, pair).
        """
        with self._lock:
            steps: List[dict] = self._run.setdefault("steps", [])
            key = (step_result.get("step"), step_result.get("pair"))
            for i, s in enumerate(steps):
                if (s.get("step"), s.get("pair")) == key:
                    steps[i] = step_result
                    break
            else:
                steps.append(step_result)
            self._flush_run()

    def append_verdict(self, verdict: Dict[str, Any]) -> None:
        with self._lock:
            self._run.setdefault("verdicts", []).append(verdict)
            self._flush_run()

    # -- runlog (append-only) ----------------------------------------------
    def log(self, event: Dict[str, Any]) -> dict:
        """Append one event to runlog.jsonl (scrubbed) and notify SSE listeners.

        Every event carries a ``ts`` (ISO UTC). The append is line-atomic (open in
        append mode, one write). Returns the scrubbed event so a caller can also
        push it somewhere.
        """
        ev = dict(event)
        ev.setdefault("ts", utcnow_iso())
        scrubbed = self._scrub(ev)
        line = json.dumps(scrubbed, ensure_ascii=False)
        with self._lock:
            with self.runlog_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
            for cb in list(self._listeners):
                try:
                    cb(scrubbed)
                except Exception:
                    pass
        return scrubbed

    def add_listener(self, cb: Callable[[dict], None]) -> None:
        with self._lock:
            self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[dict], None]) -> None:
        with self._lock:
            if cb in self._listeners:
                self._listeners.remove(cb)

    def read_runlog(self) -> List[dict]:
        """All runlog events (for replay / tests)."""
        if not self.runlog_path.exists():
            return []
        out: List[dict] = []
        for ln in self.runlog_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out

    # -- artifacts ----------------------------------------------------------
    def artifact_path(self, name: str) -> Path:
        """Path under the run dir for a CANONICAL artifact name (path-jailed)."""
        p = (self.run_dir / name).resolve()
        if self.run_dir.resolve() not in p.parents:
            raise ValueError(f"artifact name escapes run dir: {name!r}")
        return p

    def write_artifact(self, name: str, text: str) -> str:
        """Write an artifact under its canonical name (byte-safe, LF). Returns sha.

        The artifact body is NOT redacted-by-shape (it is model creative output,
        not a log) but the exact-secret literal scrub still runs — a key can never
        end up in a saved artifact even if a model echoed one back.
        """
        if self.secrets:
            text = redact_with_values(text, self.secrets)
        path = self.artifact_path(name)
        self._atomic_write(path, text)
        return sha256_text(text)

    def artifact_exists(self, name: str) -> bool:
        try:
            return self.artifact_path(name).exists()
        except ValueError:
            return False

    def read_artifact(self, name: str) -> Optional[str]:
        try:
            p = self.artifact_path(name)
        except ValueError:
            return None
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8")

    def artifact_sha(self, name: str) -> Optional[str]:
        p = self.artifact_path(name)
        if not p.exists():
            return None
        return sha256_bytes(p.read_bytes())


# ---------------------------------------------------------------------------
# run.json builders — assemble the initial provenance record (plan §5).
# ---------------------------------------------------------------------------

def build_initial_run_json(
    *,
    run_id: str,
    title: str,
    flow_name: str,
    flow_version: int,
    flow_file_sha: str,
    knobs_resolved: Dict[str, Any],
    overlays: List[Dict[str, str]],
    canon_sha: str,
    canon_dirty: bool,
    context: Dict[str, Any],
    model_tiers: Dict[str, Any],
    budget_cap_usd: float,
) -> Dict[str, Any]:
    """Construct the initial run.json (status=pending). Matches run.schema.json.

    ``knobs_resolved`` is the fully-resolved value map (incl. derived). ``context``
    is the seed/personality/panel refs+shas (never the raw key material). Totals
    and per-step records are filled in incrementally by the engine.
    """
    return {
        "id": run_id,
        "title": title,
        "created": utcnow_iso(),
        "flow": {
            "name": flow_name,
            "version": int(flow_version),
            "file_sha": flow_file_sha,
        },
        "knobs": dict(knobs_resolved),
        "overlays": list(overlays),
        "canon_sha": canon_sha,
        "canon_dirty": bool(canon_dirty),
        "context": dict(context),
        "model_tiers": dict(model_tiers),
        "steps": [],
        "totals": {
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "duration_s": 0.0,
            "pairs_total": 0,
            "pairs_quarantined": 0,
        },
        "budget": {
            "cap_usd": float(budget_cap_usd),
            "spend_usd": 0.0,
            "aborted_on_cap": False,
        },
        "status": "pending",
        "verdicts": [],
    }


def new_step_result(
    *,
    stage: Optional[str],
    step: str,
    pair: Optional[str],
    status: str = "pending",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """A blank per-step provenance record (schema: stepResult)."""
    rec: Dict[str, Any] = {"step": step, "status": status}
    if stage is not None:
        rec["stage"] = stage
    rec["pair"] = pair
    if provider is not None:
        rec["provider"] = provider
    if model is not None:
        rec["model"] = model
    return rec
