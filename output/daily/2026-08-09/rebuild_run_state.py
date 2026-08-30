# -*- coding: utf-8 -*-
"""Rebuild RUN_STATE.md BY STAT-ING DISK -- run 2026-08-09_daily_music_genz.

DISK IS THE ONLY AUTHORITY. This manifest is a CACHE the coordinator regenerates by
stat-ing files, never a hand-asserted second truth. If manifest and disk disagree,
disk wins and this script is how you find out.

A completion message not backed by a file on disk counts as INCOMPLETE -- a subagent
reply that says "let me write this now" is not done until the file exists.

Usage:  python3 output/daily/2026-08-09/rebuild_run_state.py
"""
from __future__ import print_function
import hashlib, io, os, sys, datetime

RUN  = os.path.dirname(os.path.abspath(__file__))
PAIRS = ["01", "02", "03", "04", "05", "06"]
STEPS = [
    ("06", "step06_facets"),
    ("07", "step07_song_guides"),
    ("08", "step08_generation"),
    ("09", "step09_artist_refined"),
    ("10", "step10_final_package"),
    ("11", "step11_final_package_enhanced"),
]
TRIVIAL = 400          # bytes; below this a "written" artifact is a stub, not a step
SPINE = {"07", "09", "10"}   # the NO-SKIP editorial spine

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

def main():
    icb = os.path.join(RUN, "CREATIVE_CONTEXT.md")
    icb_sha = sha(icb) if os.path.exists(icb) else "MISSING"

    rows, missing_spine, pending = [], [], []
    for p in PAIRS:
        for step, stem in STEPS:
            name = "pair_%s_%s.md" % (p, stem)
            path = os.path.join(RUN, name)
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            if not exists:
                status, verdict = "pending", "-"
                pending.append(name)
                if step in SPINE:
                    missing_spine.append(name)
            elif size < TRIVIAL:
                status, verdict = "pending", "TRIVIAL(%d B)" % size
                pending.append(name)
                if step in SPINE:
                    missing_spine.append(name + " (stub)")
            else:
                status, verdict = "done", "re-stat pending"
            rows.append({"step": step, "pair": p, "path": "output/daily/2026-08-09/" + name,
                         "exists": exists, "bytes": size,
                         "sha": sha(path) if exists else "-",
                         "status": status, "verdict": verdict})

    done = sum(1 for r in rows if r["status"] == "done")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    out = []
    out.append("# RUN_STATE — `2026-08-09_daily_music_genz`\n")
    out.append("*Rebuilt by stat-ing disk at %s. **Disk is authority.** "
               "This file is a cache, never a second truth.*\n" % now)
    out.append("**ICB sha (first 12):** `%s` — proves the frozen block has not changed.\n" % icb_sha)
    out.append("**Artifacts done: %d / %d**  ·  pending: %d\n" % (done, len(rows), len(rows) - done))
    out.append("\n| step | pair | artifact | exists | bytes | sha | status |")
    out.append("|---|---|---|---|---:|---|---|")
    for r in rows:
        out.append("| %s | %s | `%s` | %s | %d | `%s` | **%s** |" % (
            r["step"], r["pair"], os.path.basename(r["path"]),
            "yes" if r["exists"] else "**NO**", r["bytes"], r["sha"], r["status"]))

    out.append("\n## NO-SKIP spine check (steps 07 / 09 / 10)\n")
    if missing_spine:
        out.append("⛔ **NON-CANONICAL** — the editorial spine is incomplete. A run missing 07/09/10 for any "
                   "non-quarantined pair cannot receive a SHIP verdict and cannot be published under Lofn's "
                   "name. Missing:\n")
        for m in missing_spine:
            out.append("- `%s`" % m)
    else:
        out.append("✅ All six pairs have steps 07, 09 and 10 on disk at non-trivial size.")

    out.append("\n## Warm handoff\n")
    out.append("```\nstep_completed        : %s\nbuilding_toward       : %s\n"
               "rejected_alternatives : see the cut ledger in step05_refine_medium.md\n"
               "seed_fidelity         : THE THING WITH NO REPLAY BUTTON — two-layer stack, "
               "present tense, overheard not addressed\n```" % (
                   "pair fan-out 06-11" if pending else "all pair artifacts landed",
                   "coordinator re-stat -> selection -> lofn-qa -> INDEX"))
    io.open(os.path.join(RUN, "RUN_STATE.md"), "w", encoding="utf-8", newline="\n").write(
        "\n".join(out) + "\n")

    print("RUN_STATE.md rebuilt from disk: %d/%d done, %d pending" % (done, len(rows), len(rows) - done))
    if missing_spine:
        print("NO-SKIP spine incomplete: %d artifacts" % len(missing_spine))
    return 0

if __name__ == "__main__":
    sys.exit(main())
