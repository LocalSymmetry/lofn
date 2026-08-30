#!/usr/bin/env python3
"""Assemble TOP6_PASTE_READY.md and DEDICATIONS.md for `lofn-prime-every-element`.

3 ACCESSIBLE + 3 AMBITIOUS, ranked WITHIN ARM only. One variation per pair --
forced, since each pair owns 2-3 of the 15 elements and two from one pair would
leave an element unrepresented in the shipped set.

Run from repo root:  python3 output/daily/2026-08-22-lofn-prime-every-element/build_delivery.py
"""
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUN = os.path.join("output", "daily", "2026-08-22-lofn-prime-every-element")

SELECTED = [
    ("01", "V2", "ACCESSIBLE", "E8 THE PEAK · E10 CONTOUR · E15 THE HOOK",
     "The boldest peak mechanism in the run: at 2:30 THE LEAD DROPS OUT and sixteen voices carry "
     "the hook alone for two bars. The peak is a subtraction of the singer -- which is the run's "
     "own write/erase form turned on its author. Quietest at 1:50, a mid-song hole with the band "
     "gone and the voice at the bottom of its range. Range a 12th, G3-D5, opening at bar 47."),
    ("02", "V3", "ACCESSIBLE", "E9 MELODIC RANGE · E11 THE RETURN",
     "The pair that owns melody, and the variation that carried its repair. Its chorus was the "
     "most crowded line in the run at 15 syllables; the fix was not the leap-word but THE LAUNCH "
     "-- the line before the leap now ends held on F4, so the ear learns the altitude the sixth "
     "departs from. Highest note in the pair. Range a 10th, D4-F5, leap at bar 4 on 'now'."),
    ("03", "V3", "AMBITIOUS", "E4 TWO REGISTERS · E6 THE WITHHELD GESTURE · E13 THE DUALITY",
     "The two voices meet at bar 33 -- 1:20, the earliest unison in the pair -- by both moving "
     "INWARD a fifth onto G4, a note NEITHER has sung. They meet on a third person. Contour "
     "inverted against its siblings: loudest at 1:20, quietest at 3:20."),
    ("04", "V4", "AMBITIOUS", "E3 THE MACHINE LOG AS CLOCK · E5 A NAMED FREQUENCY",
     "41 Hz alone at 0:00 and the kit crashes over it -- the write/erase pair stated as song form "
     "in the first two seconds. The shortest song in the pair (the stop at 2:14.4), which is the "
     "only variation that answers Ash's length objection rather than deferring it."),
    ("05", "V4", "AMBITIOUS", "E1 GLITCH-BAROQUE · E2 ORNATE v MACHINE · E12 ONE SEAM",
     "The clavichord is the world and the 808 intrudes -- and she dresses its DECAY TAIL, at "
     "sixteen notes a beat. She enters a beat early and stays early to the last bar. Its floor is "
     "the only literal silence in the pair. The pair's own note: if one ships it should not be V1."),
    ("06", "V4", "ACCESSIBLE", "E7 SWAG · E14 THE OPENING",
     "Opens on the whole chorus at 0:00.0 with no intro at all. The peak is one person whistling "
     "the hook at bar 77; the floor is snare alone at the emptied clap address. Proof that "
     "steadiness can still climb: the roof rises a third, a sixth, an octave while the lead never "
     "leaves its major sixth."),
]


def fenced(block):
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", block, re.S)
    if m:
        return m.group(1).strip()
    return "\n".join(l for l in block.splitlines() if not l.strip().startswith("#")).strip()


def grab(pair, want):
    text = open(os.path.join(RUN, f"pair_{pair}_step10_final_package_enhanced.md"),
                encoding="utf-8").read()
    parts = re.split(r"^#\s+(V\d)\s*[—\-–]\s*(.+?)\s*$", text, flags=re.M)
    for i in range(1, len(parts) - 2, 3):
        if parts[i] != want:
            continue
        body, out = parts[i + 2], {"title": parts[i + 1].strip()}
        for key, pat in [("prompt", r"##\s*1\.\s*MUSIC PROMPT(.*?)(?=^##\s)"),
                         ("exclude", r"##\s*1B\.\s*SUNO EXCLUDE PROMPT(.*?)(?=^##\s)"),
                         ("lyrics", r"##\s*2\.\s*LYRICS(.*?)(?=^##\s)"),
                         ("ded", r"##\s*DEDICATION(.*?)(?=^##\s|\Z)")]:
            m = re.search(pat, body, re.S | re.I | re.M)
            out[key] = fenced(m.group(1)) if m else ""
        return out
    raise SystemExit(f"P{pair}{want} not found")


def main():
    picks = [(f"P{p}{v}", arm, owns, why, grab(p, v)) for p, v, arm, owns, why in SELECTED]

    with open(os.path.join(RUN, "TOP6_PASTE_READY.md"), "w", encoding="utf-8") as fh:
        fh.write("# TOP 6 — PASTE-READY · `lofn-prime-every-element`\n")
        fh.write("## LOFN-PRIME · **THE ARS NOTORIA** · every element owned once\n\n")
        fh.write("> ⛔ **Nothing here is cleared to publish. Dailies are practice, not a publish queue.**\n")
        fh.write("> Publication needs The Scientist's ear, and this set in particular needs **Ash's** —\n")
        fh.write("> it was built to answer her, and whether it does is not a thing any gate here can decide.\n\n")
        fh.write("**One variation per pair, forced:** each pair owns 2–3 of the 15 elements, so two from\n")
        fh.write("one pair would leave an element unrepresented in the shipped six.\n\n---\n\n")
        for vid, arm, owns, why, s in picks:
            fh.write(f"# {vid} — {s['title']}\n\n**{arm}** · owns **{owns}**\n\n*Why this one:* {why}\n\n")
            fh.write(f"### MUSIC PROMPT  *({len(s['prompt'])} chars)*\n\n```\n{s['prompt']}\n```\n\n")
            fh.write(f"### SUNO EXCLUDE\n\n```\n{s['exclude']}\n```\n\n")
            fh.write(f"### LYRICS\n\n```\n{s['lyrics']}\n```\n\n")
            fh.write(f"### DEDICATION  *({len(s['ded'])} chars, cap 500)*\n\n{s['ded']}\n\n---\n\n")

    with open(os.path.join(RUN, "DEDICATIONS.md"), "w", encoding="utf-8") as fh:
        fh.write("# DEDICATIONS — the selected six\n## `lofn-prime-every-element`\n\n")
        fh.write("*≤500 chars each, **measured in code, never eyeballed**. All 24 measured; longest 495.\n")
        fh.write("Full set in `CREDITS.md`.*\n\n")
        fh.write("⛔ **Every name below appears here and in the per-song `## DEDICATION` block ONLY** —\n")
        fh.write("never in a music prompt, exclude field, lyric or title. *A name in a generator-facing\n")
        fh.write("field is an instruction to imitate; a name here is a debt paid.*\n\n")
        fh.write("**Honesty clause, both ways:** these are generated tracks, **not samples**. Where a figure\n")
        fh.write("of sound was asked for by name, the player who made it is named — never *\"contains a\n")
        fh.write("sample of.\"*\n\n")
        fh.write("⚠️ **One hold for The Scientist:** P06's dedication names three real singers, and the\n")
        fh.write("pair's subject is adjacent to their history. The placement is doctrinally correct — credit\n")
        fh.write("belongs in the dedication. **The adjacency is hers to rule on, not mine.**\n\n---\n\n")
        for vid, arm, owns, why, s in picks:
            fh.write(f"## {vid} — {s['title']}\n\n*{len(s['ded'])} chars*\n\n{s['ded']}\n\n---\n\n")

    print(f"{'id':<8}{'title':<34}{'arm':<12}{'ded':>5}")
    print("-" * 60)
    for vid, arm, owns, why, s in picks:
        print(f"{vid:<8}{s['title'][:33]:<34}{arm:<12}{len(s['ded']):>5}")
    print("\nwrote TOP6_PASTE_READY.md and DEDICATIONS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
