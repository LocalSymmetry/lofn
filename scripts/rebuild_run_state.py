#!/usr/bin/env python3
"""Rebuild RUN_STATE.md for a Lofn run by STAT-ING THE FILES ON DISK.

EXECUTION.md §6: the manifest is never hand-asserted. Disk is authority. If the
manifest and disk disagree, disk wins. A completion message not backed by a file
counts as incomplete.

Usage:  python3 scripts/rebuild_run_state.py output/<run-slug> [--pairs 6]

Prints a one-line summary and writes <run-dir>/RUN_STATE.md.
"""
import argparse
import hashlib
import io
import os
import sys

# ⛔ Force UTF-8 on stdout. Without this the script dies with UnicodeEncodeError
# partway through printing on a default Windows console (cp1252) — AFTER emitting a
# plausible-looking partial report. A gate that prints half a clean result and then
# crashes is worse than one that never runs: the operator sees ticks and stops reading.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODALITY_LAYOUTS = {
    "music": {
        "coordinator": [
            "core_seed.md", "CONFLICT_RESOLUTION_01.md", "PERSONALITY_DNA_INJECT.md",
            "03_panel_debate.md", "04_metaprompt.md", "05_pair_assignments.md",
            "CREATIVE_CONTEXT.md", "06_music_handoff.md",
            "step00_aesthetics_and_genres.md", "step01_essence_and_facets.md",
            "step02_concepts.md", "step03_artist_and_critique.md",
            "step04_medium.md", "step05_refine_medium.md",
        ],
        "pair_steps": [
            ("06", "step06_facets"),
            ("07", "step07_song_guides"),
            ("08", "step08_generation"),
            ("09", "step09_artist_refined"),
            ("10", "step10_revision_synthesis"),
            ("11", "step10_final_package_enhanced"),
        ],
    },
    "story": {
        "coordinator": [
            "00_research_brief.md", "core_seed.md", "02_golden_seed.md",
            "03_orchestrator_panel_debate.md", "04_orchestrator_metaprompt.md",
            "05_orchestrator_pair_assignments.md",
            "CREATIVE_CONTEXT.md", "06_narrator_handoff.md",
            "step00_aesthetics_emotions_frames_genres.md",
            "step01_essence_facets_style_axes.md",
            "step02_twelve_campaign_concepts.md",
            "step03_artist_influence_critique.md",
            "step04_narrative_medium_styles.md",
            "step05_refined_concept_medium_pairs.md",
        ],
        "pair_steps": [
            ("06", "step06_facets"),
            ("07", "step07_story_guides"),
            ("08", "step08_generation"),
            ("09", "step09_artist_refined"),
            ("10", "step10_revision_synthesis"),
        ],
    },
}
# A file at or under this size did not really run (the "## Step 00" stub case).
TRIVIAL_BYTES = 400


def stat_one(run_dir, name):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return {"exists": False, "byte_size": 0, "byte_size_ondisk": 0,
                "sha": "", "status": "pending"}
    data = io.open(path, "rb").read()
    # Identity is keyed on LF-NORMALIZED content, never on the raw bytes.
    # Line endings are not content: on Windows a text-mode write turns every
    # \n into \r\n, so the same artifact hashes two different ways depending on
    # who measured it. That is the CRLF class from COMPETITION_LEARNINGS L25/L31
    # (two prior instances, 2026-08-03), and it false-fails a re-stat that
    # compares a writer's in-memory count against a reader's on-disk count.
    lf = data.replace(b"\r\n", b"\n")
    return {
        "exists": True,
        "byte_size": len(lf),          # canonical: LF-normalized
        "byte_size_ondisk": len(data),  # what `ls` reports on this platform
        "sha": hashlib.sha256(lf).hexdigest()[:12],
        # Non-trivial size is a necessary condition, never a sufficient one:
        # the coordinator re-stat (§3) still decides `done`.
        "status": "done" if len(data) > TRIVIAL_BYTES else "quarantined",
    }


def detect_modality(run_dir):
    story_markers = (
        "06_narrator_handoff.md",
        "step04_narrative_medium_styles.md",
        "pair_01_step07_story_guides.md",
    )
    if any(os.path.exists(os.path.join(run_dir, name)) for name in story_markers):
        return "story"
    return "music"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--modality", choices=("auto", *MODALITY_LAYOUTS), default="auto")
    args = ap.parse_args()
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        sys.exit(f"no such run directory: {run_dir}")

    modality = detect_modality(run_dir) if args.modality == "auto" else args.modality
    layout = MODALITY_LAYOUTS[modality]

    rows = []
    for name in layout["coordinator"]:
        rows.append(dict(step="coord", pair="-", canonical_path=name, **stat_one(run_dir, name)))
    for n in range(1, args.pairs + 1):
        for step, suffix in layout["pair_steps"]:
            name = f"pair_{n:02d}_{suffix}.md"
            rows.append(dict(step=step, pair=f"{n:02d}", canonical_path=name,
                             **stat_one(run_dir, name)))

    icb = stat_one(run_dir, "CREATIVE_CONTEXT.md")
    done = sum(1 for r in rows if r["status"] == "done")
    pending = sum(1 for r in rows if r["status"] == "pending")
    quarantined = sum(1 for r in rows if r["status"] == "quarantined")

    crlf_delta = icb["byte_size_ondisk"] - icb["byte_size"]
    out = [
        f"# RUN_STATE — `{os.path.basename(run_dir.rstrip('/\\'))}`",
        "",
        "> Disk-derived manifest (`EXECUTION.md` §6). Rebuilt by stat-ing files, never hand-asserted.",
        "> **Disk is authority.** A completion claim not backed by a file on disk counts as incomplete.",
        "",
        f"**modality:** `{modality}` · **icb_sha:** `{icb['sha']}` (LF-normalized content — the canonical identity) ·"
        f" **icb_bytes:** {icb['byte_size']:,} content / {icb['byte_size_ondisk']:,} on-disk (CRLF)"
        " — proves the frozen ICB has not changed.",
        "",
        "> ⚠️ **A re-stat must compare the LF-normalized numbers.** On this run the ICB's raw"
        f" on-disk size differs by {crlf_delta:,} bytes because of line endings. Comparing a"
        " writer's in-memory count against a reader's raw on-disk count can false-fail a"
        " correct artifact.",
        "",
        f"**done {done} · pending {pending} · quarantined {quarantined}**",
        "",
        "| step | pair | canonical_path | exists | byte_size | sha | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['step']} | {r['pair']} | `{r['canonical_path']}` | "
            f"{'yes' if r['exists'] else 'NO'} | {r['byte_size']:,} | "
            f"`{r['sha'] or '-'}` | {r['status']} |"
        )
    body = "\n".join(out) + "\n"
    io.open(os.path.join(run_dir, "RUN_STATE.md"), "w", encoding="utf-8").write(body)
    print(f"RUN_STATE.md rebuilt: modality={modality} done={done} pending={pending} quarantined={quarantined} "
          f"icb_sha={icb['sha']} icb_bytes={icb['byte_size']}")


if __name__ == "__main__":
    main()
