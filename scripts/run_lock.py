#!/usr/bin/env python3
"""run_lock.py — the filesystem interlock behind "one controller per date".

WHY THIS EXISTS (2026-07-24, RUN_LEDGER):
    At 23:18 a second daily controller wrote into `output/daily/2026-07-24/`
    while another run was live in that directory, and overwrote its ICB,
    research brief, steps 00-05, both judges' reports and all six step-11
    deliverables — about eleven hours of pipeline output. It was recovered
    only because an unrelated backup had been taken minutes earlier.

    The rule "one controller per date" already existed. It existed as PROSE in
    a skill file, and prose cannot interlock: nothing on disk announced that a
    run was live, so the second controller could not have detected the
    collision even if it had checked. This script is that announcement.

CONTRACT
    A run's FIRST action — before the research brief, before the ICB, before
    any artifact — is `acquire`. A run that has not acquired its lock has no
    business writing into a run directory.

    acquire    create `<run_dir>/.RUN_LOCK` atomically (O_EXCL), or resume an
               existing lock that belongs to the SAME run slug. A lock held by
               a DIFFERENT run slug is refused, always, live or stale.
    heartbeat  refresh the lock; run it at every wave boundary, in the same
               coordinator step as the RUN_STATE rebuild (EXECUTION.md 6).
    check      read-only report. Exit 0 free/own, 3 foreign.
    release    mark the run complete. The file STAYS, and the directory stays
               claimed: a FINISHED run's artifacts are at least as destroyable
               as a live one's, so a different run slug is still refused. Only
               the same run may re-open its own directory.
    break      human-only escape hatch. Never auto-steals: the old lock is
               archived inside the new one, and a reason is mandatory.

WHY STALENESS DOES NOT UNLOCK ANYTHING:
    The overwritten run was mid-flight through an ELEVEN HOUR pipeline with
    long quiet stretches. Any timeout short enough to be useful would have
    stolen that lock too. So staleness changes the MESSAGE and never the
    verdict: an old lock is reported as probably-abandoned, and only a human
    may act on that.

    python3 scripts/run_lock.py acquire output/daily/2026-07-24 \
        --run-slug <run-slug> --engine claude
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

LOCK_NAME = ".RUN_LOCK"
DEFAULT_STALE_HOURS = 4.0
GATES = Path(__file__).resolve().parents[1] / "vault" / "gates.yaml"

FREE, OWNED, FOREIGN, ERROR = 0, 0, 3, 2


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def age_hours(stamp: str) -> float | None:
    try:
        t = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None
    return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0


def stale_hours() -> float:
    """Threshold lives in gates.yaml with the other numbers (CLAUDE.md), but a
    missing or malformed file must never stop a run from acquiring its lock."""
    try:
        for line in GATES.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("run_lock_stale_hours:"):
                return float(line.split(":", 1)[1].split("#")[0].strip())
    except (OSError, ValueError):
        pass
    return DEFAULT_STALE_HOURS


def controller_id(explicit: str | None) -> str:
    """Stable within a session, distinct across sessions. Callers that know
    their own session id should pass it; the pid fallback is good enough to
    tell two concurrent controllers apart, which is all this needs to do."""
    if explicit:
        return explicit
    for var in ("CLAUDE_SESSION_ID", "CODEX_SESSION_ID", "LOFN_CONTROLLER_ID"):
        if os.environ.get(var):
            return f"{var.split('_')[0].lower()}:{os.environ[var]}"
    return f"pid:{os.getpid()}@{socket.gethostname()}"


def read_lock(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        # A corrupt lock is a FOREIGN lock. It is never nothing.
        return {"_unreadable": str(exc), "run_slug": "<unreadable>", "status": "unknown"}


def write_lock(path: Path, lock: dict, exclusive: bool = False) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def describe(lock: dict) -> str:
    age = age_hours(lock.get("heartbeat_utc", ""))
    age_txt = "unknown age" if age is None else f"last heartbeat {age:.1f} h ago"
    return (
        f"  run_slug   : {lock.get('run_slug')}\n"
        f"  controller : {lock.get('controller_id')} ({lock.get('engine')})\n"
        f"  status     : {lock.get('status')}\n"
        f"  started    : {lock.get('started_utc')}\n"
        f"  heartbeat  : {lock.get('heartbeat_utc')}  ({age_txt})\n"
        f"  phase      : {lock.get('phase')}"
    )


def new_lock(args, cid: str) -> dict:
    return {
        "run_slug": args.run_slug,
        "controller_id": cid,
        "engine": args.engine,
        "host": socket.gethostname(),
        "status": "live",
        "started_utc": now(),
        "heartbeat_utc": now(),
        "phase": args.phase or "acquired",
        "controllers": [{"controller_id": cid, "joined_utc": now()}],
    }


def cmd_acquire(args) -> int:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / LOCK_NAME
    cid = controller_id(args.controller_id)

    try:
        write_lock(path, new_lock(args, cid), exclusive=True)
        print(f"LOCK ACQUIRED  {path}\n  run_slug   : {args.run_slug}\n  controller : {cid}")
        return FREE
    except FileExistsError:
        pass

    lock = read_lock(path) or {}
    same_run = lock.get("run_slug") == args.run_slug and "_unreadable" not in lock

    if not same_run:
        age = age_hours(lock.get("heartbeat_utc", ""))
        if lock.get("status") == "complete":
            verdict = "FINISHED — its artifacts are still in this directory"
        elif age is not None and age < stale_hours():
            verdict = "LIVE"
        else:
            verdict = "possibly abandoned — only a human may decide that"
        print(
            f"⛔ REFUSED — {path} is held by a DIFFERENT run ({verdict}).\n"
            f"{describe(lock)}\n\n"
            f"You asked to write '{args.run_slug}' into this directory. Writing here would\n"
            f"overwrite that run's artifacts. This is the 2026-07-24 collision.\n\n"
            f"Do ONE of these — never a silent overwrite:\n"
            f"  * give this run its own directory (a distinct run slug in the path), or\n"
            f"  * if you meant to RESUME the run above, acquire with its exact run_slug, or\n"
            f"  * ask the human. Only they may run `break`, and only after seeing this.",
            file=sys.stderr,
        )
        return FOREIGN

    if lock.get("status") == "complete":
        print(f"LOCK REOPENED  {path}  (previous run marked complete)")
    age = age_hours(lock.get("heartbeat_utc", ""))
    fresh = age is not None and age < stale_hours() and lock.get("status") == "live"
    if fresh and lock.get("controller_id") != cid and not args.takeover:
        print(
            f"⛔ REFUSED — same run slug, but another controller looks LIVE on it.\n"
            f"{describe(lock)}\n\n"
            f"Two controllers on one run corrupt the same files as two controllers on\n"
            f"one directory. If the other controller is genuinely gone, re-run with\n"
            f"--takeover; if it is not, stop and let it finish.",
            file=sys.stderr,
        )
        return FOREIGN

    lock["controller_id"] = cid
    lock["engine"] = args.engine or lock.get("engine")
    lock["status"] = "live"
    lock["heartbeat_utc"] = now()
    lock["phase"] = args.phase or lock.get("phase")
    lock.setdefault("controllers", []).append({"controller_id": cid, "joined_utc": now(),
                                               "resumed": True})
    write_lock(path, lock)
    print(f"LOCK RESUMED   {path}\n{describe(lock)}")
    return OWNED


def cmd_heartbeat(args) -> int:
    path = Path(args.run_dir) / LOCK_NAME
    lock = read_lock(path)
    cid = controller_id(args.controller_id)
    if lock is None:
        print(
            f"⛔ NO LOCK at {path}. This run never acquired one, which means nothing on\n"
            f"disk was protecting the artifacts it has already written. Acquire now.",
            file=sys.stderr,
        )
        return ERROR
    if lock.get("controller_id") != cid:
        print(
            f"⛔ NOT YOUR LOCK — another controller owns this directory.\n{describe(lock)}\n\n"
            f"Stop writing here.",
            file=sys.stderr,
        )
        return FOREIGN
    lock["heartbeat_utc"] = now()
    if args.phase:
        lock["phase"] = args.phase
    write_lock(path, lock)
    print(f"HEARTBEAT      {path}  phase={lock.get('phase')}")
    return OWNED


def cmd_check(args) -> int:
    path = Path(args.run_dir) / LOCK_NAME
    lock = read_lock(path)
    if lock is None:
        print(f"FREE           no {LOCK_NAME} in {args.run_dir}")
        return FREE
    cid = controller_id(args.controller_id)
    mine = lock.get("controller_id") == cid
    same = args.run_slug is None or lock.get("run_slug") == args.run_slug
    state = "OWN" if mine else ("SAME-RUN" if same else "FOREIGN")
    print(f"{state:<14} {path}\n{describe(lock)}")
    # Deliberately mirrors `acquire`: a foreign run slug is FOREIGN whatever its
    # status. `check` must never report writable a directory `acquire` refuses.
    return FREE if same else FOREIGN


def cmd_release(args) -> int:
    path = Path(args.run_dir) / LOCK_NAME
    lock = read_lock(path)
    if lock is None:
        print(f"NO LOCK to release at {path}", file=sys.stderr)
        return ERROR
    cid = controller_id(args.controller_id)
    if lock.get("controller_id") != cid and not args.takeover:
        print(f"⛔ NOT YOUR LOCK.\n{describe(lock)}", file=sys.stderr)
        return FOREIGN
    lock["status"] = "complete"
    lock["heartbeat_utc"] = now()
    lock["completed_utc"] = now()
    lock["phase"] = args.phase or "complete"
    write_lock(path, lock)
    print(f"RELEASED       {path}  (file kept as the run's record)")
    return OWNED


def cmd_break(args) -> int:
    """Human-only. An agent must not run this on its own initiative — the whole
    point of the lock is that a second controller cannot decide for itself that
    the first one is finished."""
    path = Path(args.run_dir) / LOCK_NAME
    old = read_lock(path)
    if old is None:
        print(f"NO LOCK to break at {path}", file=sys.stderr)
        return ERROR
    cid = controller_id(args.controller_id)
    lock = new_lock(args, cid)
    lock["broken_locks"] = old.get("broken_locks", []) + [
        {"broken_utc": now(), "by": cid, "reason": args.reason, "previous": old}
    ]
    write_lock(path, lock)
    print(
        f"LOCK BROKEN    {path}\n  reason: {args.reason}\n"
        f"  previous run '{old.get('run_slug')}' archived inside the new lock.\n"
        f"  ⚠ Its artifacts are still on disk and are NOT yours. Do not overwrite\n"
        f"    them — move them aside or write into a new directory."
    )
    return OWNED


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, slug_required=False):
        sp.add_argument("run_dir")
        sp.add_argument("--run-slug", required=slug_required)
        sp.add_argument("--engine", default="claude")
        sp.add_argument("--controller-id")
        sp.add_argument("--phase")
        sp.add_argument("--takeover", action="store_true")

    common(sub.add_parser("acquire", help="first action of a run"), slug_required=True)
    common(sub.add_parser("heartbeat", help="refresh at every wave boundary"))
    common(sub.add_parser("check", help="read-only"))
    common(sub.add_parser("release", help="mark the run complete"))
    b = sub.add_parser("break", help="HUMAN ONLY — never auto-steal")
    common(b, slug_required=True)
    b.add_argument("--reason", required=True)

    args = p.parse_args()
    return {"acquire": cmd_acquire, "heartbeat": cmd_heartbeat, "check": cmd_check,
            "release": cmd_release, "break": cmd_break}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
