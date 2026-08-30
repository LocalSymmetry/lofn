#!/usr/bin/env python3
"""Independent coordinator re-stat of the six STEP-11 ENHANCED packages.

Single-song files, so the variation-splitting logic in coordinator_restat.py does not
apply — this is a separate reader rather than a widened one, because widening a matcher
to cover a second convention is how a matcher starts failing in both directions (L31).

Same instrument discipline: print what was EXTRACTED before trusting what was CONCLUDED;
an empty extraction is a HARD ERROR, never a score; assert the cardinality; word
boundaries on every token sweep.

Usage: python3 output/daily/2026-08-14/verify_enhanced.py
"""
import pathlib
import re
import sys

RUN = pathlib.Path("output/daily/2026-08-14")
PAIRS = [f"{i:02d}" for i in range(1, 7)]
G = dict(prompt_lo=850, prompt_hi=1000, prompt_hug=985, excl_lo=400, excl_hi=1000,
         lyr_max=5000, lyr_target=4800, lines_lo=70, lines_hi=120, lines_hug=72, ded_max=500)
NAMED_FIGURES = {"amen": "Gregory C. Coleman / The Winstons",
                 "funky drummer": "Clyde Stubblefield",
                 "think break": "Lyn Collins",
                 "apache": "the Incredible Bongo Band"}
BANNED_TOKENS = ["snarl", "phonk", "supersaw", "airhorn"]
BANNED_OPENERS = ["begin in", "begin with", "use ", "build the track from",
                  "chronology:", "for an adult human singer"]


def wb(text, tok):
    return len(re.findall(r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])", text.lower()))


def block(text, *headings):
    for h in headings:
        m = re.search(r"^#{1,4}\s*" + h + r".*?$", text, re.M | re.I)
        if not m:
            continue
        rest = text[m.end():]
        nxt = re.search(r"^#{1,4}\s+", rest, re.M)
        body = rest[:nxt.start()] if nxt else rest
        fenced = re.findall(r"```[a-z]*\n(.*?)```", body, re.S)
        return (fenced[0] if fenced else body).strip()
    return ""


def sung(lyrics):
    out = []
    for ln in lyrics.splitlines():
        s = ln.strip()
        if not s or s.startswith("[") or s.startswith("#"):
            continue
        if re.match(r"^\*[^*]*\*$", s):          # *SFX* cue
            continue
        out.append(s)
    return out


found, problems = 0, []
print("STEP-11 ENHANCED PACKAGES — independent coordinator re-stat\n" + "=" * 62)

for p in PAIRS:
    fp = RUN / f"pair_{p}_step10_final_package_enhanced.md"
    if not fp.exists():
        print(f"\nPAIR {p}: NOT YET ON DISK")
        continue
    found += 1
    t = fp.read_text(encoding="utf-8", errors="replace")
    prompt = block(t, r"1\.\s*MUSIC PROMPT", r"MUSIC PROMPT")
    excl = block(t, r"1B\.\s*SUNO EXCLUDE PROMPT", r"EXCLUDE PROMPT")
    lyr = block(t, r"2\.\s*LYRICS", r"LYRICS")
    title = block(t, r"3\.\s*TITLE", r"TITLE")
    print(f"\nPAIR {p}  ({fp.stat().st_size:,} bytes)")
    print(f"  extracted: prompt {len(prompt)}c · exclude {len(excl)}c · "
          f"lyrics {len(lyr)}c · title {'yes' if title else 'NO'}")
    if not prompt or not lyr:
        problems.append(f"P{p}: EMPTY EXTRACTION (prompt={len(prompt)} lyrics={len(lyr)}) "
                        f"— HARD ERROR, not a score")
        print("  !! EMPTY EXTRACTION — refusing to score this file")
        continue

    sl = sung(lyr)
    flags = []
    if not (G["prompt_lo"] <= len(prompt) <= G["prompt_hi"]):
        flags.append(f"PROMPT_CHARS={len(prompt)}")
    elif len(prompt) >= G["prompt_hug"]:
        flags.append(f"hug(prompt {len(prompt)})")
    if prompt.rstrip()[-1] not in ".!?":
        flags.append("NO_TERMINAL_PUNCT")
    if not (G["excl_lo"] <= len(excl) <= G["excl_hi"]):
        flags.append(f"EXCLUDE_CHARS={len(excl)}")
    if len(lyr) >= G["lyr_max"]:
        flags.append(f"LYRICS_OVER_HARD_CAP={len(lyr)}")
    elif len(lyr) > G["lyr_target"]:
        flags.append(f"lyrics>{G['lyr_target']} ({len(lyr)})")
    if not (G["lines_lo"] <= len(sl) <= G["lines_hi"]):
        flags.append(f"SUNG_LINES={len(sl)}")
    elif len(sl) <= G["lines_hug"]:
        flags.append(f"hug(lines {len(sl)})")
    for op in BANNED_OPENERS:
        if prompt.lower().startswith(op):
            flags.append(f"BANNED_OPENER:{op.strip()}")
    for tok in BANNED_TOKENS:
        if wb(prompt, tok):
            flags.append(f"BANNED_TOKEN:{tok}")
    if not re.search(r"\[Theme:", lyr):
        flags.append("NO_THEME_TAG")
    if not re.search(r"\[SONG FORM:", lyr, re.I):
        flags.append("NO_SONGFORM_TAG")
    if not re.search(r"\[Disc_", lyr):
        flags.append("NO_DISC_CHANNEL_BLOCK")
    else:
        print(f"  Disc_ channels in field: {len(re.findall(r'\[Disc_', lyr))}")
    if not re.search(r"^\s*\*[^*]+\*\s*$", lyr, re.M):
        flags.append("NO_SFX_CUE")
    bare = len(re.findall(r"EMO:\s*(?:AWE|INDIGNATION)\s*[-\]]", lyr, re.I))
    if bare:
        flags.append(f"BARE_EMO={bare}")
    emo = len(re.findall(r"EMO:", lyr))
    digits = re.findall(r"(?<![\w#])\d{1,4}(?![\w%])", "\n".join(sl))
    if digits:
        flags.append(f"SUNG_DIGITS={digits[:5]}")

    # dedication: measured, never eyeballed
    ded = ""
    m = re.search(r"(?:^|\n)#{1,4}[^\n]*DEDICATION[^\n]*\n(.*?)(?=\n#{1,4}\s|\Z)", t,
                  re.S | re.I)
    if m:
        body = m.group(1)
        fen = re.findall(r"```[a-z]*\n(.*?)```", body, re.S)
        if fen:
            ded = fen[0].strip()
        else:
            # The dedication is the QUOTED/first prose block only. Everything after it —
            # a trim-priority note, a char-count annotation — is commentary ABOUT the
            # dedication, not the dedication. My first version counted the commentary and
            # reported P01 at 611 against a true 453, i.e. a fabricated cap breach.
            # This is the documented trap: a validator dying on a trailing annotation,
            # "a habit we encourage". Fifth run running, the AGENT was right and the
            # COORDINATOR's instrument was wrong.
            quoted = [ln[1:].strip() for ln in body.splitlines() if ln.lstrip().startswith(">")]
            if quoted:
                ded = " ".join(x for x in quoted if x).strip()
            else:
                ded = body.strip().split("\n\n")[0].strip()
    if ded:
        print(f"  dedication: {len(ded)} chars (cap {G['ded_max']})")
        if len(ded) > G["ded_max"]:
            flags.append(f"DEDICATION_OVER_CAP={len(ded)}")
    else:
        print("  dedication: not separately extractable (may be inside LINEAGE block)")

    owed = [f"{k} -> {v}" for k, v in NAMED_FIGURES.items() if wb(prompt, k)]
    generic = bool(re.search(r"(?<![a-z])(chopped |re)?break(beat)?s?(?![a-z])", prompt.lower()))
    if owed:
        print(f"  NAMED FIGURE in shipped STYLE -> credit owed: {owed}")
        for k, v in NAMED_FIGURES.items():
            if wb(prompt, k):
                who = v.split(" / ")[0].split()[-1]
                if who.lower() not in t.lower():
                    problems.append(f"P{p}: STYLE names '{k}' but '{v}' is not credited in the file")
    elif generic:
        print("  generic break language only — NO named figure, NO credit owed "
              "(a false credit makes the true ones worthless)")
    else:
        print("  no break language in shipped STYLE — no drummer credit owed")

    print(f"  sung lines {len(sl)} · EMO headers {emo} · "
          f"{'FLAGS: ' + '; '.join(flags) if flags else 'clean'}")
    for f in flags:
        if f.split("(")[0].split("=")[0].isupper():
            problems.append(f"P{p}: {f}")

print("\n" + "=" * 62)
print(f"enhanced packages on disk: {found}/6")
print(f"problems: {len(problems)}")
for pr in problems:
    print(f"  - {pr}")
if found < 6:
    print("\n(partial run — not all pairs have reported yet)")
