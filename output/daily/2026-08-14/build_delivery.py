#!/usr/bin/env python3
"""Build TOP6_PASTE_READY.md and DEDICATIONS.md by EXTRACTION from the shipped packages.

⭐ WHY THIS IS A SCRIPT AND NOT PROSE. On 2026-08-13 a delivered paste-ready file carried a
V1 prose blurb sitting above a correct V2 payload — the one coordinator error that reached
The Scientist. The verdict was:

    EVERYTHING GENERATED MECHANICALLY WAS RIGHT.
    EVERY ERROR CAME FROM PROSE TYPED FROM MEMORY BESIDE IT.

So every field a human will paste into Suno is EXTRACTED from the artifact on disk. The only
hand-written text in the output is labelled EDITORIAL.

Usage: python3 output/daily/2026-08-14/build_delivery.py
"""
import pathlib
import re
import sys

RUN = pathlib.Path("output/daily/2026-08-14")
PAIRS = ["01", "02", "03", "04", "05", "06"]
META = {  # slot facts from 05_pair_assignments.md — arm/personality/bed, NOT song content
    "01": ("ACCESSIBLE", "SUNNA", "ska / two-tone / rocksteady + horns", "AWE", "EXISTENCE"),
    "02": ("ACCESSIBLE", "LOFN-PRIME", "modern power-pop (girl-group intrusion)", "AWE", "EXISTENCE"),
    "03": ("ACCESSIBLE", "SUNNA", "dub", "AWE", "NEWS"),
    "04": ("AMBITIOUS", "LOFN-PRIME", "jungle / breakbeat", "INDIGNATION", "NEWS"),
    "05": ("AMBITIOUS", "SUNNA", "speed garage / bassline (gear-change)", "AWE", "EXISTENCE"),
    "06": ("AMBITIOUS", "LOFN-PRIME", "Jersey / Baltimore club (Glitch-Baroque)", "INDIGNATION", "NEWS"),
}


def block(text, *headings):
    for h in headings:
        m = re.search(r"^#{1,4}\s*" + h + r".*?$", text, re.M | re.I)
        if not m:
            continue
        rest = text[m.end():]
        nxt = re.search(r"^#{1,4}\s+", rest, re.M)
        body = rest[:nxt.start()] if nxt else rest
        fen = re.findall(r"```[a-z]*\n(.*?)```", body, re.S)
        return (fen[0] if fen else body).strip()
    return ""


def dedication(text):
    """The dedication body only — blockquote OR fenced OR first prose para.

    Three shipped conventions were measured on disk: a `>` blockquote (P01, P04), a
    ``` fenced block (P02, P03, P06), and a fenced block with a heading that already
    states the count. My first version handled only the blockquote and returned "" for
    three of six — and the caller printed that as `0` instead of raising.
    ⛔ AN EMPTY EXTRACTION IS A HARD ERROR, NEVER A VALUE. That is the rule this file
    exists to enforce, and this function broke it. Fixed in both places.
    Trailing commentary (trim-priority notes, char-count annotations) is ABOUT the
    dedication and is excluded — counting it once fabricated a cap breach.
    """
    # ⛔ ANCHOR TO A HEADING, not to any line containing the word. The loose version
    # matched a PROSE MENTION of "dedication" inside P06's repaired LINEAGE block and
    # swallowed 2,534 chars, reporting a cap breach that does not exist. The script
    # fail-closed and refused to write, which is the behaviour that caught it — but the
    # matcher was still wrong, and this is the third time in this run that a loose
    # pattern has produced a confident false number.
    m = re.search(r"^#{1,4}[^\n]*DEDICATION[^\n]*\n(.*?)(?=\n#{1,4}\s|\Z)", text, re.S | re.I | re.M)
    if not m:
        m = re.search(r"^\*\*[^\n]*DEDICATION[^\n]*\n(.*?)(?=\n#{1,4}\s|\n\*\(|\Z)", text, re.S | re.I | re.M)
    if not m:
        return ""
    body = m.group(1)
    fen = re.findall(r"```[a-z]*\n(.*?)```", body, re.S)
    if fen:
        return " ".join(fen[0].split()).strip()
    quoted = [ln.lstrip()[1:].strip() for ln in body.splitlines() if ln.lstrip().startswith(">")]
    if quoted:
        return " ".join(x for x in quoted if x).strip()
    para = [p.strip() for p in body.strip().split("\n\n") if p.strip() and not p.strip().startswith("*(")]
    return " ".join(para[0].split()).strip() if para else ""


def extract_tag(lyrics, tag):
    m = re.search(r"\[" + tag + r":\s*(.*?)\]", lyrics, re.S)
    return m.group(1).strip().replace("\n", " ") if m else ""


songs, problems = [], []
for p in PAIRS:
    fp = RUN / f"pair_{p}_step10_final_package_enhanced.md"
    if not fp.exists():
        problems.append(f"P{p}: enhanced package missing")
        continue
    t = fp.read_text(encoding="utf-8", errors="replace")
    s = dict(pair=p,
             title=block(t, r"3\.\s*TITLE", r"TITLE").splitlines()[0].strip().strip("*`# ") if block(t, r"3\.\s*TITLE", r"TITLE") else "",
             prompt=block(t, r"1\.\s*MUSIC PROMPT", r"MUSIC PROMPT"),
             exclude=block(t, r"1B\.\s*SUNO EXCLUDE PROMPT", r"EXCLUDE PROMPT"),
             lyrics=block(t, r"2\.\s*LYRICS", r"LYRICS"),
             ded=dedication(t))
    s["theme"] = extract_tag(s["lyrics"], "Theme")
    s["form"] = extract_tag(s["lyrics"], "SONG FORM")
    for k in ("title", "prompt", "exclude", "lyrics", "ded"):
        if not s[k]:
            problems.append(f"P{p}: EMPTY EXTRACTION for {k} — HARD ERROR, not a value")
    if len(s["ded"]) > 500:
        problems.append(f"P{p}: dedication {len(s['ded'])} chars OVER the 500 cap")
    songs.append(s)

if problems:
    print("EXTRACTION PROBLEMS — refusing to write a delivery file from a bad read:")
    for pr in problems:
        print("  -", pr)
    sys.exit(1)

# ---------------------------------------------------------------- TOP 6
out = ["# TOP 6 — PASTE-READY SUNO PACKAGES · daily 2026-08-14",
       "",
       "**Run:** `2026-08-14-daily-music` · **THE EMBARRASSING DETAIL, AMPLIFIED** · engine: **RETURN**",
       "**Six songs, one per pair — 3 ACCESSIBLE + 3 AMBITIOUS, ranked within arm, never 5+1.**",
       "",
       "> ⭐ **Every field below is EXTRACTED from the shipped artifact on disk by "
       "`build_delivery.py`, never retyped.** `[Theme:]` and `[SONG FORM:]` are pulled from each "
       "song's own lyrics field. The only hand-written lines in this file are marked EDITORIAL.",
       "",
       "⛔ **Nothing here is rendered.** These are text packages; the render/publish line is "
       "`vault/AUTONOMY.md` and it is The Scientist's.",
       "",
       "---",
       ""]
for s in songs:
    arm, who, bed, mode, axis = META[s["pair"]]
    out += [f"## {s['pair']}. **{s['title']}**",
            "",
            f"`{arm}` · `{who}` · `{mode}` · `{axis}` · bed: *{bed}*",
            "",
            f"**Theme (extracted):** {s['theme']}",
            "",
            f"**Song form (extracted):** {s['form']}",
            "",
            f"### Style prompt — paste into Suno's style box  ·  {len(s['prompt'])} chars",
            "```", s["prompt"], "```",
            "",
            f"### Exclude prompt — paste into Suno's exclude box  ·  {len(s['exclude'])} chars",
            "```", s["exclude"], "```",
            "",
            f"### Lyrics field — paste whole  ·  {len(s['lyrics'])} chars (hard cap 5000)",
            "```", s["lyrics"], "```",
            "",
            f"### Dedication  ·  {len(s['ded'])} chars (cap 500)",
            "", f"> {s['ded']}",
            "", "---", ""]
(RUN / "TOP6_PASTE_READY.md").write_text("\n".join(out), encoding="utf-8")

# ---------------------------------------------------------------- DEDICATIONS
d = ["# DEDICATIONS · daily 2026-08-14",
     "",
     "`vault/DEDICATION_STANDARD.md` — every shipped piece carries a **≤500-char** dedication naming what",
     "it borrowed and who it is owed to. **Measured in code, never eyeballed.**",
     "",
     "⛔ **Honesty clause, both directions:** Suno generates, it does not sample. These say *\"we asked for",
     "this figure by name, so we name them\"* — **never** *\"contains a sample of…\"*. A false credit would",
     "make the true ones worthless.",
     "",
     "| # | song | chars | cap |", "|---|---|---|---|"]
for s in songs:
    ok = "✅" if len(s["ded"]) <= 500 else "❌ OVER"
    d.append(f"| {s['pair']} | {s['title']} | **{len(s['ded'])}** | {ok} |")
d += ["", "---", ""]
for s in songs:
    d += [f"## {s['pair']}. {s['title']}  ·  {len(s['ded'])} chars", "", f"> {s['ded']}", ""]
(RUN / "DEDICATIONS.md").write_text("\n".join(d), encoding="utf-8")

print(f"{'pair':5s} {'title':32s} {'style':>6s} {'excl':>6s} {'lyrics':>7s} {'ded':>5s}")
for s in songs:
    print(f"{s['pair']:5s} {s['title'][:32]:32s} {len(s['prompt']):6d} {len(s['exclude']):6d} "
          f"{len(s['lyrics']):7d} {len(s['ded']):5d}")
print(f"\nwrote TOP6_PASTE_READY.md and DEDICATIONS.md — {len(songs)} songs, 0 extraction problems")
