"""Promotion — write winning knob/threshold VALUES back to canon (plan §8, §9,
WS backend half; decision §13 #3).

The studio proves a knob change in the lab; promotion carries the WINNING VALUES
back to canon through the v1 byte-safe :class:`FileService` write path:

  * :func:`preview` — the diff a promotion WOULD produce, WITHOUT writing:
      - ``gates`` value changes: a ruamel round-trip edit of ``vault/gates.yaml``
        that touches ONLY the changed scalar values (comments / structure / EOL /
        BOM byte-preserved); the preview returns the current vs proposed bytes as
        a unified diff.
      - ``patches`` simple prose-number changes: an explicit ``old -> new`` string
        replacement in an allowlisted prose file (exact, unique-match, no regex);
        previewed as a unified diff.
      - ``structural`` changes (flow shape, step order, new gates) are NOT applied
        here — they are described and routed to :func:`brief`.

  * :func:`apply` — AUTO-APPLY the value + simple-patch changes via
    :class:`FileService` (path-jailed, conflict-guarded, byte-safe), then run
    ``scripts/validate_step.py --meta-check`` and return its result + the new git
    status. NEVER rewrites a step's creative prose except an explicit approved
    patch, and always through the byte-safe path.

  * :func:`brief` — generate a ``PROMOTION_BRIEF.md`` for structural flow changes:
    a human-readable record of what the run changed and what a canon maintainer
    must deliberately apply to canon prose (the studio never edits flow-structure
    prose automatically).

CANON SAFETY: every write goes through ``FileService`` (the write allowlist +
path jail + UTF-8 guard + atomic write). A value change to ``vault/gates.yaml`` is
allowed (``vault/**`` is in the allowlist); anything outside is refused by the
jail, not by this module.
"""
from __future__ import annotations

import difflib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The v1 byte-safe write layer lives in the parent ``server`` package. studio is
# imported two ways: as ``server.studio`` (the app, via ``.studio.api``) where the
# relative import resolves, and as top-level ``studio`` (the $0 test suite, with
# tools/explorer/server on sys.path) where the parent must be reached by its fully
# qualified name. Try both.
try:  # subpackage layout (server.studio.*)
    from ..files import ConflictError, FileService, PathJailError
    from ..yamlio import dump as yaml_dump, load as yaml_load
except ImportError:  # pragma: no cover - top-level layout (tests)
    from tools.explorer.server.files import ConflictError, FileService, PathJailError  # type: ignore
    from tools.explorer.server.yamlio import dump as yaml_dump, load as yaml_load  # type: ignore

GATES_FILE = "vault/gates.yaml"


# ---------------------------------------------------------------------------
# Byte-safe gates.yaml value editing (ruamel round-trip; values only).
# ---------------------------------------------------------------------------

def _coerce_like(current: Any, proposed: Any) -> Any:
    """Coerce ``proposed`` to the shape of ``current`` so a round-trip write keeps
    the value's YAML type (a band stays a 2-list; an int stays an int).

    Bands: a [lo, hi] proposed onto a [lo, hi] current stays a list. Scalars keep
    int/float/bool typing where the current value implies it."""
    if isinstance(current, (list, tuple)) and isinstance(proposed, (list, tuple)):
        return [_coerce_like(c, p) for c, p in zip(current, proposed)] + \
               list(proposed[len(current):])
    if isinstance(current, bool):
        return bool(proposed)
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(proposed)
        except (TypeError, ValueError):
            return proposed
    if isinstance(current, float):
        try:
            return float(proposed)
        except (TypeError, ValueError):
            return proposed
    return proposed


def _apply_gate_values(raw: bytes, values: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """Return (new_bytes, applied) after setting each key's scalar value in the
    ruamel round-trip structure. Only keys that ALREADY exist are touched (a
    promotion changes an existing threshold; it never introduces a new key into
    canon gates.yaml). Unknown keys are ignored and reported back to the caller.
    """
    data, meta = yaml_load(raw)
    applied: Dict[str, Any] = {}
    for k, v in values.items():
        if k not in data:
            # Never silently add a key to canon gates — skip + report.
            continue
        cur = data[k]
        coerced = _coerce_like(cur, v)
        if _jsonable(cur) == _jsonable(coerced):
            continue  # no-op value; nothing to change
        # For a band (a ruamel sequence), mutate elements IN PLACE so the flow
        # style ([lo, hi]) and any comments are preserved byte-for-byte. Only a
        # scalar assignment replaces the node.
        if isinstance(cur, list) and isinstance(coerced, list) and len(cur) == len(coerced):
            for i, elem in enumerate(coerced):
                cur[i] = elem
        else:
            data[k] = coerced
        applied[k] = coerced
    return yaml_dump(data, meta), applied


def _jsonable(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    return str(v)


def _unified(before: str, after: str, path: str) -> str:
    return "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
    ))


# ---------------------------------------------------------------------------
# Simple prose-number patches (explicit, unique, non-regex old->new).
# ---------------------------------------------------------------------------

@dataclass
class PatchResult:
    file: str
    ok: bool
    detail: str
    diff: str = ""
    new_text: str = ""      # LF-normalized proposed text (apply uses this)
    base_sha: str = ""


def _plan_patch(files: FileService, file: str, old: str, new: str) -> PatchResult:
    """Plan an exact, unique ``old -> new`` replacement in a prose file.

    Refuses (ok=False) unless: the file is writable, the ``old`` string appears
    EXACTLY ONCE (a unique, unambiguous edit — no regex, no multi-hit), and ``new``
    differs from ``old``. This is deliberately conservative: promotion never
    rewrites creative prose, only an explicit approved number/word swap."""
    try:
        view = files.read(file)
    except FileNotFoundError:
        return PatchResult(file=file, ok=False, detail="file not found")
    if not view.writable:
        return PatchResult(file=file, ok=False, detail="file not in write allowlist / not UTF-8")
    if old == new:
        return PatchResult(file=file, ok=False, detail="old == new (no-op)")
    count = view.text.count(old)
    if count == 0:
        return PatchResult(file=file, ok=False, detail=f"pattern not found: {old!r}")
    if count > 1:
        return PatchResult(file=file, ok=False,
                           detail=f"pattern is not unique ({count} matches); refuse ambiguous patch")
    new_text = view.text.replace(old, new)
    return PatchResult(
        file=file, ok=True, detail="ready",
        diff=_unified(view.text, new_text, view.path),
        new_text=new_text, base_sha=view.sha,
    )


# ---------------------------------------------------------------------------
# preview / apply / brief.
# ---------------------------------------------------------------------------

def preview(
    files: FileService,
    *,
    gate_values: Optional[Dict[str, Any]] = None,
    patches: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """The diffs a promotion WOULD produce, WITHOUT writing anything.

    ``gate_values``: gates.yaml key -> proposed value (existing keys only).
    ``patches``: list of ``{"file", "old", "new"}`` exact prose swaps.
    """
    out: Dict[str, Any] = {"gates": None, "patches": [], "skipped_keys": []}

    if gate_values:
        raw = files.read_bytes(GATES_FILE)
        before = raw.decode("utf-8")
        new_bytes, applied = _apply_gate_values(raw, gate_values)
        after = new_bytes.decode("utf-8")
        skipped = [k for k in gate_values if k not in applied]
        out["gates"] = {
            "file": GATES_FILE,
            "applied": {k: _jsonable(v) for k, v in applied.items()},
            "diff": _unified(before, after, GATES_FILE),
            "changed": bool(applied),
        }
        out["skipped_keys"] = skipped

    for p in patches or []:
        pr = _plan_patch(files, p.get("file", ""), p.get("old", ""), p.get("new", ""))
        out["patches"].append({
            "file": pr.file, "ok": pr.ok, "detail": pr.detail, "diff": pr.diff,
        })

    return out


def apply(
    files: FileService,
    repo_root: Path,
    *,
    gate_values: Optional[Dict[str, Any]] = None,
    patches: Optional[List[Dict[str, str]]] = None,
    python_exe: Optional[str] = None,
) -> Dict[str, Any]:
    """AUTO-APPLY value + simple-patch changes via the byte-safe FileService, run
    ``--meta-check`` after, and return the new git status.

    Value changes to ``vault/gates.yaml`` are written with ruamel round-trip bytes
    through ``FileService.write_bytes`` (path-jailed). Prose patches are written
    with ``FileService.write_text`` under a ``base_sha`` conflict guard. A patch
    that can't be planned cleanly is REFUSED (never a best-effort creative rewrite).
    Structural changes are NOT applied here — use :func:`brief`.
    """
    applied_gates: Dict[str, Any] = {}
    applied_patches: List[Dict[str, Any]] = []
    errors: List[str] = []

    # 1) gates.yaml value write (byte-safe, values only).
    if gate_values:
        raw = files.read_bytes(GATES_FILE)
        new_bytes, applied = _apply_gate_values(raw, gate_values)
        if applied:
            try:
                files.write_bytes(GATES_FILE, new_bytes)
                applied_gates = {k: _jsonable(v) for k, v in applied.items()}
            except PathJailError as e:
                errors.append(f"gates write refused: {e}")
        skipped = [k for k in gate_values if k not in applied]
        if skipped:
            errors.append(f"gate keys not present in canon (skipped): {', '.join(skipped)}")

    # 2) prose patches (explicit, unique, byte-safe with conflict guard).
    for p in patches or []:
        pr = _plan_patch(files, p.get("file", ""), p.get("old", ""), p.get("new", ""))
        if not pr.ok:
            errors.append(f"patch refused for {pr.file}: {pr.detail}")
            applied_patches.append({"file": pr.file, "applied": False, "detail": pr.detail})
            continue
        try:
            files.write_text(pr.file, pr.new_text, base_sha=pr.base_sha)
            applied_patches.append({"file": pr.file, "applied": True, "detail": "written"})
        except ConflictError as e:
            errors.append(f"patch conflict for {pr.file}: file changed on disk")
            applied_patches.append({"file": pr.file, "applied": False,
                                    "detail": f"conflict (current sha {e.current_sha})"})
        except PathJailError as e:
            errors.append(f"patch refused for {pr.file}: {e}")
            applied_patches.append({"file": pr.file, "applied": False, "detail": str(e)})

    # 3) meta-check AFTER (prose/YAML disagreement scan — WARN only, exit 0).
    meta = _run_meta_check(repo_root, python_exe)

    # 4) new git status (what the maintainer will see / commit).
    status = _git_status(repo_root)

    return {
        "applied_gates": applied_gates,
        "applied_patches": applied_patches,
        "errors": errors,
        "meta_check": meta,
        "git_status": status,
    }


def brief(
    *,
    run_a: Optional[str] = None,
    run_b: Optional[str] = None,
    title: str = "Structural promotion",
    summary: str = "",
    structural_changes: Optional[List[Dict[str, Any]]] = None,
    knob_delta: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate the ``PROMOTION_BRIEF.md`` text for a structural flow change.

    Structural changes (flow shape, step order, new/removed gates) are NEVER
    auto-applied — this brief is the deliberate hand-off a canon maintainer reads
    before touching canon prose. Value + simple-patch promotions still auto-apply
    (decision §13 #3) but a brief accompanies them for the record.
    """
    lines: List[str] = []
    lines.append(f"# PROMOTION BRIEF — {title}")
    lines.append("")
    if run_a or run_b:
        lines.append(f"**Runs compared:** `{run_a or '?'}` vs `{run_b or '?'}`")
        lines.append("")
    if summary:
        lines.append(summary.strip())
        lines.append("")

    lines.append("## Auto-applied (values + simple patches)")
    lines.append("")
    if knob_delta:
        lines.append("| knob | A | B |")
        lines.append("| --- | --- | --- |")
        for d in knob_delta:
            lines.append(f"| `{d.get('knob')}` | {d.get('a')} | {d.get('b')} |")
    else:
        lines.append("_No knob-value deltas recorded for this promotion._")
    lines.append("")

    lines.append("## Structural changes — REQUIRES DELIBERATE CANON EDIT")
    lines.append("")
    lines.append("> The studio does NOT edit flow-structure prose automatically. "
                 "Apply each of the following to canon by hand, then re-run "
                 "`validate_step.py --meta-check`.")
    lines.append("")
    if structural_changes:
        for i, c in enumerate(structural_changes, 1):
            kind = c.get("kind", "change")
            where = c.get("where", "")
            what = c.get("what", "")
            lines.append(f"{i}. **{kind}** — {where}")
            if what:
                lines.append(f"   - {what}")
    else:
        lines.append("_No structural changes recorded._")
    lines.append("")

    lines.append("## Checklist")
    lines.append("")
    lines.append("- [ ] Reviewed the auto-applied gates.yaml value diff")
    lines.append("- [ ] Reviewed each simple prose patch diff")
    lines.append("- [ ] Applied structural changes to canon prose by hand")
    lines.append("- [ ] `validate_step.py --meta-check` exits 0 (no new prose/YAML drift)")
    lines.append("- [ ] Committed with a clear provenance message")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# subprocess helpers.
# ---------------------------------------------------------------------------

def _run_meta_check(repo_root: Path, python_exe: Optional[str]) -> Dict[str, Any]:
    exe = python_exe or sys.executable
    cmd = [exe, str(repo_root / "scripts" / "validate_step.py"),
           "--meta-check", "--root", "."]
    try:
        proc = subprocess.run(cmd, cwd=str(repo_root), text=True,
                              capture_output=True, timeout=120)
        return {
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
        }
    except Exception as e:  # pragma: no cover - defensive
        return {"exit_code": 125, "ok": False, "stdout": "",
                "stderr": f"meta-check launch error: {type(e).__name__}: {e}"}


def _git_status(repo_root: Path) -> str:
    try:
        proc = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_root),
                              text=True, capture_output=True, timeout=15)
        return (proc.stdout or "").strip()
    except Exception:
        return ""
