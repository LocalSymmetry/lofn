#!/usr/bin/env python3
"""Emit paste-ready Suno guides for the four magnum-opus suites.

One file per suite. Each carries three songs in movement order, each song split
into the three fields Suno actually takes: STYLE, EXCLUDE, LYRICS. Nothing is
retyped — every field is EXTRACTED from the artifact on disk, because a prose
blurb typed beside a correct payload is how a wrong version reached her on
2026-08-13. SPEC != ARTIFACT, every time.

Prefers `pair_NN_step10_final_package_enhanced.md` (step 11) when present,
falls back to `pair_NN_step10_revision_synthesis.md`, and SAYS WHICH.

Usage: python3 scripts/build_suno_guides.py output/<run-dir>
"""
import io
import os
import re
import sys

# ⛔ Force UTF-8 on stdout. Without this the script dies with UnicodeEncodeError
# partway through printing on a default Windows console (cp1252) — AFTER emitting a
# plausible-looking partial report. A gate that prints half a clean result and then
# crashes is worse than one that never runs: the operator sees ticks and stops reading.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restat_pair_packages import content_after, split_variations, detect_mode, count_headings
from measure_soundcraft import sung as canonical_sung

SUITES = [
    ("A", "THE COUNT", "the judge's selection — chosen for the one hole it can measure",
     [("I", 1, 1), ("II", 3, 2), ("III", 5, 1)]),
    ("B", "THE WITHHOLDING", "all three AMBITIOUS — the darkest, most durational reading",
     [("I", 2, 2), ("II", 3, 4), ("III", 6, 3)]),
    ("C", "THE HAND", "all three ACCESSIBLE — the warmest, and the one most likely to travel",
     [("I", 1, 4), ("II", 4, 1), ("III", 5, 4)]),
    ("D", "IDIOT GLEE (?)", "the day's Oblique Strategy answered across all three movements",
     [("I", 1, 3), ("II", 4, 4), ("III", 6, 2)]),
]
MOVEMENT_NAME = {"I": "THE SLANT PATH", "II": "THE SECOND FILTER",
                 "III": "WHAT CAME THROUGH PINK"}


# ⛔ A step-11 file holds exactly ONE variation — the one that was selected — but it is
# named for its PAIR. Loading it for any variation of that pair would silently serve the
# wrong song: pair 01's enhanced file is V1, and suite C wants pair 01 **V4**. This map
# is the guard. A pair is only served from step 11 for the variation actually enhanced.
ENHANCED_VARIATION = {1: 1, 3: 2, 5: 1}


def load_song(run_dir, pair, var):
    enhanced = os.path.join(run_dir, f"pair_{pair:02d}_step10_final_package_enhanced.md")
    step10 = os.path.join(run_dir, f"pair_{pair:02d}_step10_revision_synthesis.md")
    if os.path.exists(enhanced) and ENHANCED_VARIATION.get(pair) == var:
        text = io.open(enhanced, encoding="utf-8").read()
        mode, tbl = detect_mode(text)
        # Anchored count — an unanchored one reads prose that merely mentions the
        # heading and silently falls back to step 10 on a correctly enhanced file.
        if tbl and count_headings(text, tbl["music"]) == 1:
            return _fields(text, tbl), "step 11 (enhanced)"
    text = io.open(step10, encoding="utf-8").read()
    mode, tbl = detect_mode(text)
    blocks, _ = split_variations(text)
    if len(blocks) != 4:
        raise SystemExit(f"ERROR: pair {pair:02d} extracted {len(blocks)}/4 blocks")
    return _fields(blocks[var - 1], tbl), "step 10"


def _fields(block, tbl):
    title = (content_after(block, tbl["title"]) or "").strip().splitlines()[0]
    return {
        "title": re.sub(r"[*_`#]", "", title).strip(),
        "style": (content_after(block, tbl["music"]) or "").strip(),
        "exclude": (content_after(block, tbl["exclude"]) or "").strip(),
        "lyrics": (content_after(block, tbl["lyrics"], strip_italics=False) or "").strip(),
    }


def main():
    run_dir = sys.argv[1]
    out_dir = os.path.join(run_dir, "SUNO_GUIDES")
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for code, name, blurb, movements in SUITES:
        parts = [
            f"# SUITE {code} — **{name}**",
            f"*{blurb}*", "",
            "**Three linked Suno clips, rendered and posted in order.** Paste each field into the",
            "matching Suno box. Nothing here is retyped — every field is extracted from the",
            "artifact on disk.", "",
            "⛔ The suite binders, in every movement: the anchor phrase *\"I saw the smoke and I",
            "called it gold\"* (interrupted in I · shouted in II · finished quietly in III), the",
            "folding chair at three distances, and a D–C–D figure inside the first two seconds.",
            "", "---", "",
        ]
        for mv, pair, var in movements:
            song, src = load_song(run_dir, pair, var)
            n = len(canonical_sung(song["lyrics"]))
            parts += [
                f"## MOVEMENT {mv} — {MOVEMENT_NAME[mv]}",
                f"### “{song['title']}”",
                f"`pair {pair:02d} · variation {var} · {src} · "
                f"{len(song['style'])} char style · {len(song['lyrics'])} char lyrics · "
                f"{n} sung lines`", "",
                "**① STYLE** — paste into Suno's style/prompt box", "",
                "```text", song["style"], "```", "",
                "**② EXCLUDE** — paste into Suno's exclude box", "",
                "```text", song["exclude"], "```", "",
                "**③ LYRICS** — paste into Suno's lyrics box", "",
                "```text", song["lyrics"], "```", "", "---", "",
            ]
            index.append((code, name, mv, song["title"], pair, var, src, n))
        path = os.path.join(out_dir, f"SUITE_{code}_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('?', '')}.md")
        io.open(path, "w", encoding="utf-8").write("\n".join(parts))
        print(f"wrote {path}")

    idx = ["# SUNO GUIDES — four magnum-opus suites", "",
           "Twelve distinct songs, four complete three-movement works, no song reused.", "",
           "| suite | movement | title | source | sung lines |", "|---|---|---|---|---|"]
    for code, name, mv, title, pair, var, src, n in index:
        idx.append(f"| **{code} · {name}** | {mv} | {title} | P{pair:02d}V{var} · {src} | {n} |")
    io.open(os.path.join(out_dir, "INDEX.md"), "w", encoding="utf-8").write("\n".join(idx) + "\n")
    print(f"wrote {os.path.join(out_dir, 'INDEX.md')}")


if __name__ == "__main__":
    main()
