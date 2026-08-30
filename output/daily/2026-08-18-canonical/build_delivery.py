#!/usr/bin/env python3
"""Build TOP6_PASTE_READY.md and DEDICATIONS.md by EXTRACTION from the shipped packages.

⭐ WHY THIS IS A SCRIPT AND NOT PROSE. On 2026-08-13 a delivered paste-ready file carried a
V1 prose blurb sitting above a correct V2 payload — the one coordinator error that reached
The Scientist. The verdict was:

    EVERYTHING GENERATED MECHANICALLY WAS RIGHT.
    EVERY ERROR CAME FROM PROSE TYPED FROM MEMORY BESIDE IT.

So every field a human will paste into Suno is EXTRACTED from the artifact on disk. The only
hand-written text in the output is labelled EDITORIAL.

Usage: python3 output/daily/2026-08-18-canonical/build_delivery.py
"""
import pathlib
import re
import sys

RUN = pathlib.Path("output/daily/2026-08-18-canonical")
PAIRS = ["01", "02", "03", "04", "05", "06"]
META = {  # slot facts from 05_pair_assignments.md — arm/personality/bed, NOT song content
    "01": ("ACCESSIBLE", "SUNNA", "modern power-pop (1963 girl-group intrusion)", "AWE", "EXISTENCE"),
    "02": ("ACCESSIBLE", "SUNNA", "street-band pop (1979 two-tone intrusion)", "AWE", "EXISTENCE"),
    "03": ("ACCESSIBLE", "SUNNA", "modern rocksteady-pop (1967 one-drop intrusion)", "AWE", "NEWS"),
    "04": ("AMBITIOUS", "SUNNA", "modern glam-pop (1973 double-drum stomp intrusion)", "AWE", "EXISTENCE"),
    "05": ("AMBITIOUS", "SUNNA", "modern nu-disco (1972 wah-clavinet intrusion)", "INDIGNATION", "NEWS"),
    "06": ("AMBITIOUS", "SUNNA", "contemporary funk-pop (Afrobeat interlock intrusion)", "AWE", "NEWS"),
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


# QA's ranked selection (output/daily/2026-08-18-canonical/QA_REPORT.md), best within arm.
# Editorial rationale is hand-written and labelled; every PAYLOAD below is extracted.
PICKS = {"01": ("V2", "ACCESSIBLE", 1), "02": ("V2", "ACCESSIBLE", 2), "03": ("V4", "ACCESSIBLE", 3),
         "05": ("V1", "AMBITIOUS", 1), "06": ("V3", "AMBITIOUS", 2), "04": ("V3", "AMBITIOUS", 3)}

VAR_RE = re.compile(r"^#{1,4}[ \t]*V([1-4])\b.*$", re.M)


def variation_chunk(text, vid):
    """Slice ONE variation out of a four-song enhanced package.

    Without this, block() returns the FIRST heading match in the whole file, i.e. V1
    every time, and the delivery silently ships V1 while claiming the ranked pick.
    Same defect class the P02 step-11 tier found in verify_enhanced.py, 2026-08-18.
    """
    hits = list(VAR_RE.finditer(text))
    for i, m in enumerate(hits):
        if m.group(1) == vid.lstrip("V"):
            end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            return text[m.start():end]
    return ""


songs, problems = [], []
for p in PAIRS:
    fp = RUN / f"pair_{p}_step10_final_package_enhanced.md"
    if not fp.exists():
        problems.append(f"P{p}: enhanced package missing")
        continue
    full = fp.read_text(encoding="utf-8", errors="replace")
    vid, arm, rank = PICKS[p]
    t = variation_chunk(full, vid)
    if not t:
        problems.append(f"P{p}: variation {vid} NOT FOUND — HARD ERROR")
        continue
    ded_src = full  # the dedication is pair-level, not per-variation
    s = dict(pair=p, vid=vid, arm=arm, rank=rank,
             title=(re.match(r"^#{1,4}[ 	]*V[1-4][ 	]*[—–-][ 	]*(.+?)[ 	]*$", t, re.M).group(1).strip().strip("*`# ")
                    if re.match(r"^#{1,4}[ 	]*V[1-4][ 	]*[—–-][ 	]*(.+?)[ 	]*$", t, re.M) else ""),
             prompt=block(t, r"SUNO STYLE PROMPT", r"1\.\s*MUSIC PROMPT", r"MUSIC PROMPT"),
             exclude=block(t, r"SUNO EXCLUDE", r"1B\.\s*SUNO EXCLUDE PROMPT", r"EXCLUDE PROMPT"),
             lyrics=block(t, r"SUNO LYRICS", r"2\.\s*LYRICS", r"LYRICS"),
             ded=dedication(ded_src))
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
out = ["# TOP 6 — PASTE-READY SUNO PACKAGES · daily 2026-08-18 (canonical)",
       "",
       "**Run:** `2026-08-18-canonical-sunna-r3` · **APPETITE WITH A TIMETABLE** · personality: **SUNNA R3 + R3.1**",
       "**Six songs, one per pair — 3 ACCESSIBLE + 3 AMBITIOUS, ranked within arm, never 5+1.**",
       "**Selection is QA's** (`QA_REPORT.md`), not the coordinator's. QA verdict: **REPAIR** — see the INDEX.",
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
d = ["# DEDICATIONS · daily 2026-08-18 (canonical)",
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
