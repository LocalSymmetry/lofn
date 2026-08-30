#!/usr/bin/env python3
"""Build the delivery: the six selected songs as individual artifacts, the paste-ready
sheet, and DEDICATIONS.md.

Imports `dedication_lib` for the character count — the ONE implementation. Four separate
readers over-counted this field in eight days; this file does not become the fifth.
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import dedication_lib as D  # noqa: E402

ROOT = HERE.parents[2]
SONGS = ROOT / "output" / "songs"
SONGS.mkdir(parents=True, exist_ok=True)
STAMP = "20260821"

# QA's within-arm ranking (ranked within arm only, never a global score).
SELECTED = [
    ("01", "4", "ACCESSIBLE", 1, "INDIGNATION", 92, "twenty twenty-two",
     "C-18 THE CLEAN CUT", "industrial gospel, quiet"),
    ("02", "2", "ACCESSIBLE", 2, "AWE", 76, "ten kilometres down",
     "C-20 SHELF COUNT", "rolling half-time bass music"),
    ("03", "3", "ACCESSIBLE", 3, "AWE + rupture", 84, "two years",
     "C-22 THE GUARD'S ROOM", "slinky almost-R&B with neo-baroque harmony"),
    ("04", "3", "AMBITIOUS", 1, "AWE", 68, "four hundred and fifty million years",
     "C-24 THE PART THAT NEVER SURVIVES", "glitch-baroque over cosmic synths"),
    ("05", "4", "AMBITIOUS", 2, "INDIGNATION", 104, "thirty femtoseconds",
     "C-27 BEFORE ANYONE LOOKED", "frenetic drum-machine pulse, dry and quiet"),
    ("06", "2", "AMBITIOUS", 3, "AWE", 88, "sixty-five percent",
     "C-28 CANE SEASON", "gasba seam over a modern world"),
]

VAR_RE = re.compile(r"^#{1,4}[ \t]*(?:[^\w\s#]+[ \t]*)?"
                    r"(?:(?:PACKAGE|VARIATION|SONG|TRACK)[ \t]+V?(\d)|V(\d))\b[^\n]*$",
                    re.M | re.I)


def block(chunk, heading):
    m = re.search(r"^#{1,4}[ \t]*(?:[^\w\s#]+[ \t]*)?(?:\d+[A-Za-z]?[.)][ \t]*)?"
                  + heading + r".*?$", chunk, re.M | re.I)
    if not m:
        return ""
    rest = chunk[m.end():]
    nxt = re.search(r"^#{1,4}\s", rest, re.M)
    body = rest[:nxt.start()] if nxt else rest
    fen = re.findall(r"```[a-zA-Z]*\n(.*?)```", body, re.S)
    return (fen[0] if fen else body).strip()


def chunks(pair):
    f = HERE / f"pair_{pair}_step10_final_package_enhanced.md"
    t = f.read_text(encoding="utf-8", errors="replace")
    marks = list(VAR_RE.finditer(t))
    if len(marks) != 4:
        sys.exit(f"FATAL: {f.name} -> {len(marks)} variations, expected 4. "
                 f"EXTRACTION ERROR, never a score.")
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(t)
        out[m.group(1) or m.group(2)] = t[m.start():end]
    return out


slug = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:48]

paste, deds, written = [], [], []
for pair, vid, arm, rank, mode, bpm, fact, concept, genre in SELECTED:
    ch = chunks(pair)[vid]
    prompt = block(ch, r"MUSIC PROMPT")
    excl = block(ch, r"(?:SUNO\s+)?EXCLUDE(?:\s+PROMPT)?")
    lyr = block(ch, r"LYRICS")
    title = block(ch, r"TITLE").splitlines()[0].strip().strip("*_` ") if block(ch, r"TITLE") else ""
    ded, dn, dv = D.check(ch)
    for name, val in (("MUSIC PROMPT", prompt), ("LYRICS", lyr), ("TITLE", title)):
        if not val:
            sys.exit(f"FATAL: P{pair}V{vid} empty extraction for {name} — HARD ERROR.")

    fn = SONGS / f"{STAMP}_{slug(title)}_P{pair}_V{vid}.md"
    fn.write_text(
        "---\n"
        "type: song\n"
        f'title: "{title}"\n'
        "created: 2026-08-21\n"
        'pipeline_run: "daily-music-2026-08-21"\n'
        f"pair: {int(pair)}\n"
        f"variation: {int(vid)}\n"
        f'pair_name: "{concept}"\n'
        "selected: true\n"
        f"rank: {rank}\n"
        f"arm: {arm}\n"
        f"mode: {mode}\n"
        'personality: "LOFN-PRIME"\n'
        f'seed_title: "THE ONE LEFT WHO CAN STILL HEAR IT"\n'
        f'seed_archetype: "THE LAST WITNESS (inverted)"\n'
        f'genre: "{genre}"\n'
        f"bpm: {bpm}\n"
        f'sung_fact: "{fact}"\n'
        'renderer: "Suno (text only — NOT rendered, NOT published, NOT spent)"\n'
        f"qa_verdict: SHIP\n"
        f"somatic_gate: 3/3 YES\n"
        f"music_prompt_chars: {len(prompt)}\n"
        f"exclude_chars: {len(excl)}\n"
        f"lyrics_field_chars: {len(lyr)}\n"
        f"dedication_chars: {dn}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"## 1. MUSIC PROMPT\n\n```text\n{prompt}\n```\n\n"
        f"## 1B. SUNO EXCLUDE PROMPT\n\n```text\n{excl}\n```\n\n"
        f"## 2. LYRICS\n\n```text\n{lyr}\n```\n\n"
        f"## DEDICATION ({dn} chars)\n\n{ded}\n",
        encoding="utf-8")
    written.append(fn.name)

    paste.append(
        f"\n---\n\n## {arm} #{rank} — **{title}**  ·  P{pair}V{vid}\n"
        f"*{concept} · {mode} · {bpm} BPM · {genre} · sings “{fact}”*\n\n"
        f"**STYLE PROMPT** ({len(prompt)} chars)\n```text\n{prompt}\n```\n\n"
        f"**EXCLUDE** ({len(excl)} chars)\n```text\n{excl}\n```\n\n"
        f"**LYRICS** ({len(lyr)} chars)\n```text\n{lyr}\n```\n")
    deds.append(f"| {arm} #{rank} | **{title}** | P{pair}V{vid} | {dn} | {' '.join(ded.split())} |")

(HERE / "TOP6_PASTE_READY.md").write_text(
    "# TOP 6 — PASTE-READY\n## `daily-music-2026-08-21` · QA verdict **SHIP** · Somatic Gate **3/3 on all six pairs**\n\n"
    "> ⛔ **Nothing here was rendered, published, or spent.** These are paste-ready text packages;\n"
    "> Suno is the destination and **the render is The Scientist's.** Dailies are practice, not a\n"
    "> publish queue — publication additionally needs the cross-model step-11 review and her ear,\n"
    "> with borderline defaulting to HOLD.\n\n"
    "**Ranked WITHIN each arm only** — never a global score. One variation per pair, which\n"
    "the sung-fact allocation forces: every variation inside a pair carries that pair's one\n"
    "allocated number, so two from one pair would collide on the six-different-numbers rule.\n"
    + "".join(paste), encoding="utf-8")

(HERE / "DEDICATIONS.md").write_text(
    "# DEDICATIONS — the selected six\n## `daily-music-2026-08-21`\n\n"
    "> Measured in code via `dedication_lib.measure_dedication()` — **the one implementation**.\n"
    "> Four separate readers over-counted this field in eight days and every one of them would\n"
    "> have bounced a compliant artifact. Cap **500**.\n\n"
    "> ⭐ **No name below appears in any generator-facing field.** *A name in the prompt is an\n"
    "> instruction to imitate; a name in the dedication is a debt paid.* Full companion file:\n"
    "> `CREDITS.md` (all 24).\n\n"
    "| arm | title | id | chars | dedication |\n|---|---|---|---:|---|\n"
    + "\n".join(deds) + "\n", encoding="utf-8")

print(f"wrote {len(written)} song artifacts to output/songs/")
for w in written:
    print("   ", w)
print("wrote TOP6_PASTE_READY.md and DEDICATIONS.md")
