#!/usr/bin/env python3
"""Assemble TOP6_PASTE_READY.md and DEDICATIONS.md from the selected six.

Selection is 3 ACCESSIBLE + 3 AMBITIOUS, ranked WITHIN ARM ONLY (never 5+1 by
global score), one variation per pair -- forced, because each pair carries one
allocated fact and one allocated bed, so two from a pair would collide on the
six-different-beds rule.

Run from repo root:  python3 output/daily/2026-08-22/build_delivery.py
"""
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUN = os.path.join("output", "daily", "2026-08-22")

# (pair, variation, arm, bed, why-this-one) -- rationale from QA_REPORT.md §6.
SELECTED = [
    ("01", "V1", "ACCESSIBLE", "swing / jump blues · 102 BPM",
     "QA: 'the strongest set in the run - the only pair where the praise is consistently a person "
     "doing a specific thing rather than an inventory of nouns.' V4 carries the pair's one flat-test "
     "failure, so the pick avoids it."),
    ("02", "V4", "ACCESSIBLE", "two-tone ska / offbeat chop · 126 BPM",
     "QA: 'V4 is the standout of the pair - the singing coach, the woman who does the low bit wrong "
     "and loud, the two blokes who give her the seat. Appetite inside an INDIGNATION song, which is "
     "harder than either alone.'"),
    ("03", "V2", "ACCESSIBLE", "rocksteady / one-drop into dub · 108 BPM",
     "QA: the clearest appetite-as-verb in the pair - she spends money on heat she does not need, on "
     "purpose, and says so. V1 carries two evaluations ('it is lovely') that break Sunna's own rule."),
    ("04", "V1", "AMBITIOUS", "ska-punk x dub x boom-bap · 118 BPM",
     "QA: 'the idea is the best in the run' - a hole in a hook a hundred people fill unasked, and the "
     "singer standing in it deliberately. V3's 95 sung lines are ~a third two-syllable echoes."),
    ("05", "V2", "AMBITIOUS", "rap-rock hybrid · 100 BPM",
     "QA: uses no guitar swell at all - the drive-as-colour clause honoured. V3 is a counting song "
     "inside a one-number rule and spends its numeric budget before the hinge."),
    ("06", "V4", "AMBITIOUS", "art-pop / dry brass stabs · 112 BPM",
     "QA: 'the best-varied pair by every instrument I ran'; V4's last verse is 'the best writing in "
     "the run.'"),
]


def fenced(block):
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", block, re.S)
    if m:
        return m.group(1).strip()
    return "\n".join(l for l in block.splitlines() if not l.strip().startswith("#")).strip()


def grab(pair, want):
    path = os.path.join(RUN, f"pair_{pair}_step10_final_package_enhanced.md")
    text = open(path, encoding="utf-8").read()
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
    top, deds = [], []
    for pair, var, arm, bed, why in SELECTED:
        s = grab(pair, var)
        vid = f"P{pair}{var}"
        top.append((vid, arm, bed, why, s))
        deds.append((vid, s["title"], s["ded"]))

    with open(os.path.join(RUN, "TOP6_PASTE_READY.md"), "w", encoding="utf-8") as fh:
        fh.write("# TOP 6 — PASTE-READY · `daily-music-2026-08-22-sunna-offbeat`\n")
        fh.write("## SUNNA leads · the offbeat lineage · **THE ONE NOBODY PLAYS**\n\n")
        fh.write("> ⚠️ **PROVISIONAL — THE RUN-LEVEL CIRCUIT BREAKER IS TRIPPED.** These six are the best "
                 "of each arm and every one passes every hard gate, but four of six pairs carry one "
                 "shared defect class (see `HELD_FOR_HUMAN.md`). **Nothing here is cleared to publish; "
                 "dailies are practice, not a publish queue.**\n\n")
        fh.write("**Ranked WITHIN ARM only.** One variation per pair — forced, since each pair owns one "
                 "bed and one allocated fact.\n\n---\n\n")
        for vid, arm, bed, why, s in top:
            fh.write(f"# {vid} — {s['title']}\n\n**{arm}** · {bed}\n\n*Why this one:* {why}\n\n")
            fh.write(f"### MUSIC PROMPT  *({len(s['prompt'])} chars)*\n\n```\n{s['prompt']}\n```\n\n")
            fh.write(f"### SUNO EXCLUDE\n\n```\n{s['exclude']}\n```\n\n")
            fh.write(f"### LYRICS\n\n```\n{s['lyrics']}\n```\n\n")
            fh.write(f"### DEDICATION  *({len(s['ded'])} chars, cap 500)*\n\n{s['ded']}\n\n---\n\n")

    with open(os.path.join(RUN, "DEDICATIONS.md"), "w", encoding="utf-8") as fh:
        fh.write("# DEDICATIONS — the selected six\n")
        fh.write("## `daily-music-2026-08-22-sunna-offbeat`\n\n")
        fh.write("*`vault/DEDICATION_STANDARD.md`: every shipped piece carries a ≤500-char dedication "
                 "naming what it borrowed and who it is owed to — **measured in code, never eyeballed**. "
                 "All 24 measured; longest 494. Full set in `CREDITS.md`.*\n\n")
        fh.write("⛔ **These names appear here and in the per-song `## DEDICATION` block ONLY** — never in "
                 "a music prompt, exclude field, lyric or title. Proved by `credit_sweep.py`: **0 leaks "
                 "across 4 generator-facing fields × 24 songs.** A name in a generator-facing field is an "
                 "instruction to imitate; a name here is a debt paid.\n\n")
        fh.write("⭐ **The credit this system is built to forget, paid.** The run's seed is a *deleted "
                 "beat*. Both players who authored the deletions are named: **Carlton Barrett** for the "
                 "one drop — the pattern defined by the kick that does not land on beat one — and "
                 "**Lloyd Knibb** for the ska stroke, the upstroke that became an entire music. Living-"
                 "scene credit is easy and already gated; the session player who made the figure is "
                 "invisible, usually dead, and asks for nothing.\n\n")
        fh.write("**Honesty clause, both ways:** these are generated tracks, **not samples**. We asked "
                 "for these figures by name, so we name the players — never *\"contains a sample of.\"*\n\n---\n\n")
        for vid, title, ded in deds:
            fh.write(f"## {vid} — {title}\n\n*{len(ded)} chars*\n\n{ded}\n\n---\n\n")

    print(f"{'id':<8}{'title':<32}{'arm':<12}{'ded':>5}")
    print("-" * 58)
    for vid, arm, bed, why, s in top:
        print(f"{vid:<8}{s['title'][:31]:<32}{arm:<12}{len(s['ded']):>5}")
    print("\nwrote TOP6_PASTE_READY.md and DEDICATIONS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
