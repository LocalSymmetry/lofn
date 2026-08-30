#!/usr/bin/env python3
"""Rebuild RUN_STATE.md for daily-music-2026-08-22-sunna-offbeat by STAT-ING DISK.

Disk is the only authority. This is a cache the coordinator rebuilds, never a
hand-asserted second truth. If manifest and disk disagree, re-derive from disk.

Run from repo root:  python3 output/daily/2026-08-22/rebuild_run_state.py
"""
import hashlib
import os
import sys

RUN = os.path.join("output", "daily", "2026-08-22")
PAIRS = ["01", "02", "03", "04", "05", "06"]
PAIR_STEPS = [
    ("06", "step06_facets"),
    ("07", "step07_song_guides"),
    ("08", "step08_generation"),
    ("09", "step09_artist_refined"),
    ("10", "step10_final_package_enhanced"),
]
RUN_LEVEL = [
    ("00", "00_research_brief.md"),
    ("S2", "SOURCE2.txt"),
    ("02", "02_golden_seed.md"),
    ("03", "03_panel_debate.md"),
    ("04", "04_metaprompt.md"),
    ("05", "05_pair_assignments.md"),
    ("ICB", "CREATIVE_CONTEXT.md"),
    ("QA", "QA_REPORT.md"),
    ("DED", "DEDICATIONS.md"),
    ("TOP6", "TOP6_PASTE_READY.md"),
    ("IDX", "INDEX.md"),
]


def stat(rel):
    path = os.path.join(RUN, rel)
    if not os.path.isfile(path):
        return {"exists": False, "bytes": 0, "sha": "-"}
    raw = open(path, "rb").read()
    return {
        "exists": True,
        "bytes": len(raw),
        "sha": hashlib.sha256(raw).hexdigest()[:12],
    }


def main():
    icb = stat("CREATIVE_CONTEXT.md")
    rows, done, pending = [], 0, 0

    for code, name in RUN_LEVEL:
        s = stat(name)
        status = "done" if s["exists"] else "pending"
        done += s["exists"]
        pending += not s["exists"]
        rows.append(
            f"| {code} | — | run | `{name}` | {'yes' if s['exists'] else 'NO'} "
            f"| {s['bytes']} | `{s['sha']}` | — | {status} |"
        )

    for p in PAIRS:
        for step, suffix in PAIR_STEPS:
            name = f"pair_{p}_{suffix}.md"
            s = stat(name)
            status = "done" if s["exists"] else "pending"
            done += s["exists"]
            pending += not s["exists"]
            rows.append(
                f"| {step} | P{p} | music | `{name}` | {'yes' if s['exists'] else 'NO'} "
                f"| {s['bytes']} | `{s['sha']}` | — | {status} |"
            )

    out = [
        "# RUN_STATE — `daily-music-2026-08-22-sunna-offbeat`",
        "",
        "> **Metadata only.** No creative payload, no ICB summary. Rebuilt by stat-ing disk",
        "> (`rebuild_run_state.py`), written as the LAST action of each wave, never ahead of the files.",
        "> If this file and disk disagree, **disk wins** and this gets rebuilt.",
        "",
        f"**ICB sha:** `{icb['sha']}` ({icb['bytes']} bytes) — the block every pair received verbatim.",
        f"**Artifacts on disk:** {done} · **pending:** {pending}",
        "",
        "| step | pair | modality | canonical_path | exists | bytes | sha | gate_verdict | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    out += rows
    out += [
        "",
        "**Resume rule:** rebuild from disk, then run only `pending`. ⛔ **Never regenerate a `done`",
        "artifact** and never re-run a passed gate. A reply that says *\"let me write this\"* counts as",
        "`pending` until the file is on disk.",
        "",
        "**Quarantine rule:** a pair failing the SAME gate on its 3rd repair attempt is terminal — mark",
        "`quarantined`, do NOT consume its artifact downstream, and surface \"N of 6 broke open at step X",
        "(gate: …)\" BEFORE QA. Progress is measured on the failed gate's own value, not on raw bytes.",
        "",
        "**Circuit breaker:** if ONE named gate fails across many pairs, HALT and name it for a human.",
        "Independent one-off failures retry and quarantine normally; only SAME-gate correlation trips it.",
        "",
    ]
    text = "\n".join(out) + "\n"
    open(os.path.join(RUN, "RUN_STATE.md"), "w", encoding="utf-8").write(text)
    print(f"RUN_STATE rebuilt: {done} done, {pending} pending, ICB sha {icb['sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
