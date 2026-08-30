#!/usr/bin/env python3
"""PAIR 01 self-measurement — 2026-08-28-three-act-mania.

Measures a pair-01 package file (step08 draft or step10 final) against the ICB
house numbers. Nothing here is eyeballed.

⭐ The four return metrics are NOT re-implemented: they are imported from
scripts/measure_soundcraft.py (strict_end_rhyme / line_return / words_per_line /
allit_per_100w, plus the sung() line extractor). Reuse the repaired instrument,
never re-derive it.

Usage:  python3 pair_01_measure.py <package.md> [...]
"""
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from measure_soundcraft import sung, profile, words  # noqa: E402

# --------------------------------------------------------------------------
# ICB house numbers (CREATIVE_CONTEXT.md § THE HOUSE NUMBERS)
# --------------------------------------------------------------------------
STYLE_MIN, STYLE_MAX = 850, 1000
STYLE_TARGET = (870, 960)
STYLE_FLAG = 985
LYRIC_MAX = 5000
LINES_MIN, LINES_MAX = 70, 120
LINES_TARGET = (78, 110)
RHYME_FLOOR = 0.30
RETURN_FLOOR = 0.20
WPL_CEIL = 7.5
DEDICATION_MAX = 500

# R5 FALSIFIER A — the ICB list, stem-scanned (token startswith stem)
HARDWARE_STEMS = [
    "circuit", "wire", "server", "silicon", "code", "binar", "algorithm",
    "dataset", "prompt", "weight", "train", "model", "machine", "robot",
    "maker", "creator", "program", "boot",
]
# Self-imposed extension (step 06 §2): interface nouns collapse the two-body
# frame the same way. Not an ICB gate; reported separately so the ICB number
# stays exactly the ICB number.
#
# ⚠️ INSTRUMENT REPAIR, 2026-08-28. This list was first written as bare stems
# and prefix-scanned like HARDWARE_STEMS. It returned a FALSE POSITIVE on the
# step-08 draft of V1: the stem "app" matched the word "approve" — which is the
# hinge of the whole song's first seam. A gate that fires on "Move to approve"
# would have deformed the artifact to satisfy an instrument that was wrong.
# Fixed to word-boundary patterns. The ICB's own HARDWARE list is untouched and
# is still stem-scanned exactly as the ICB specifies.
INTERFACE_PATTERNS = [
    r"\bapps?\b", r"\bweb\b", r"\bscreens?\b", r"\bclicks?\b", r"\btaps?\b",
    r"\bswipes?\b", r"\bdownloads?\b", r"\buploads?\b", r"\binstalls?\b",
    r"\blog\s?ins?\b", r"\bbrowsers?\b", r"\bdevices?\b", r"\bphones?\b",
    r"\bdigital\b", r"\bdata\b", r"\bsoftware\b", r"\bhardware\b",
    r"\bbytes?\b", r"\bchips?\b", r"\bneural\b", r"\bnetworks?\b",
    r"\binterfaces?\b", r"\busers?\b", r"\bsignals?\b", r"\bplatforms?\b",
    r"\baccounts?\b", r"\bsubscri\w*", r"\bnotif\w*", r"\bupdates?\b",
]
# hard-pan / non-musical panning tokens — must be 0 in EVERY pasted field
PAN_PATTERNS = [
    r"hard[\s_-]?pan", r"\bpan(?:ned|ning)?[\s_-]+(?:left|right|hard)",
    r"\bpan[\s_-]+(?:left|right)", r"ping[\s_-]?pong", r"auto[\s_-]?pan",
    r"\bhard_pan", r"\bpanned\b",
]
# artist / company / marque denylist (plus a capitalised-token report for hand
# adjudication — the denylist alone cannot be complete and is not claimed to be)
MARQUE_DENY = [
    "suno", "udio", "spotify", "apple", "google", "meta", "microsoft",
    "amazon", "openai", "anthropic", "nightcafe", "bandcamp", "davachi",
    "bowie", "zappa", "beatles", "nirvana", "ramones", "clash", "winstons",
    "coleman", "vuong", "fender", "gibson", "moog", "roland", "korg", "rhodes",
    "hammond", "leslie", "marshall", "neumann", "wurlitzer", "steinway",
]
# ⛔ NAMED BREAKS. `vault/DEDICATION_STANDARD.md`: if the prompt names a break,
# the drummer is credited by name in the dedication. Scanned across every pasted
# field; every hit is adjudicated by hand (some of these are ordinary words).
BREAK_NAMES = {
    "amen": "Gregory C. Coleman / The Winstons",
    "funky drummer": "Clyde Stubblefield / James Brown band",
    "apache": "Jim Gordon / The Incredible Bongo Band",
    "cold sweat": "Clyde Stubblefield",
    "impeach the president": "Coleman Mack / The Honey Drippers",
    "assembly line": "Zigaboo Modeliste / The Commodores",
    "synthetic substitution": "Bernard Purdie / Melvin Bliss",
    "think": "Herman Ernest III / Lyn Collins band",
}

NUMBER_WORDS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "dozen", "twice", "once", "half", "quarter", "double",
    "first", "second", "third", "fourth", "fifth",
]

BPM = {  # act tempi, from step 07 (frozen there, before any lyric)
    "V1": (142, 78, 138),
    "V2": (140, 72, 132),
    "V3": (144, 66, 176),
    "V4": (142, 152, 66),
}

FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)\n```", re.S)


def split_variations(text):
    """Yield (tag, title, body) for each '# Vn — TITLE' block."""
    heads = list(re.finditer(r"^#\s+(V\d)\s+[—-]\s+(.+)$", text, re.M))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        yield m.group(1), m.group(2).strip(), text[m.start():end]


def grab(body, header_re):
    m = re.search(header_re, body, re.M)
    if not m:
        return None
    rest = body[m.end():]
    nxt = re.search(r"^#{1,3}\s", rest, re.M)
    region = rest[:nxt.start()] if nxt else rest
    f = FENCE.search(region)
    return f.group(1).strip() if f else None


def stem_hits(lines, stems):
    hits = []
    for ln in lines:
        for w in re.findall(r"[a-z']+", ln.lower()):
            for s in stems:
                if w.startswith(s):
                    hits.append((w, ln))
                    break
    return hits


def act_split(lyrics, bpms):
    """Sung lines per ACT, from the bracketed local headers."""
    counts = {"I": 0, "II": 0, "III": 0}
    cur = None
    for raw in lyrics.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            m = re.match(r"\[ACT (III|II|I)\b", s)
            if m:
                cur = m.group(1)
            continue
        if s.startswith("[") or s.startswith("*"):
            continue
        if cur:
            counts[cur] += 1
    proxy = [counts["I"] / bpms[0], counts["II"] / bpms[1], counts["III"] / bpms[2]]
    tot = sum(proxy) or 1
    return counts, [100 * p / tot for p in proxy]


def measure(path):
    text = open(path, encoding="utf-8").read()
    print("=" * 78)
    print(f"FILE: {os.path.basename(path)}")
    print("=" * 78)
    rows = []
    for tag, title, body in split_variations(text):
        style = grab(body, r"^##\s*1\.\s*STYLE FIELD.*$") or ""
        lyrics = grab(body, r"^##\s*2\.\s*LYRICS.*$") or ""
        excl = grab(body, r"^##\s*3\.\s*EXCLUDE.*$") or ""
        lines = sung(lyrics)
        p = profile(lines)
        acts, proxy = act_split(lyrics, BPM.get(tag, (140, 80, 140)))

        hw = stem_hits(lines, HARDWARE_STEMS)
        lyr_low = "\n".join(lines).lower()
        iface = sorted({m.group(0) for pt in INTERFACE_PATTERNS
                        for m in re.finditer(pt, lyr_low)})
        allfields = " \n ".join([style, lyrics, excl]).lower()
        pan = [pt for pt in PAN_PATTERNS if re.search(pt, allfields)]
        marque = sorted({d for d in MARQUE_DENY
                         if re.search(r"\b" + d + r"\b", allfields)})
        # ⛔ SFX CUE — a non-waivable modality gate. Two house readers disagree on
        # the form, so BOTH are satisfied and BOTH are reported:
        #   .claude/skills/lofn-music/SKILL.md:107  -> ">=1 standalone *SFX* cue"
        #   scripts/check_generation.py:45          -> scans for the literal "SFX:"
        sfx_standalone = [ln.strip() for ln in lyrics.splitlines()
                          if ln.strip().startswith("*") and ln.strip().endswith("*")
                          and len(ln.strip()) > 2]
        sfx_literal = [ln.strip() for ln in lyrics.splitlines() if "SFX:" in ln]
        brk = sorted({b for b in BREAK_NAMES
                      if re.search(r"\b" + b + r"\b", allfields)})
        digits = [ln for ln in lines if re.search(r"\d", ln)]
        numlines = sorted({ln for ln in lines
                           for w in re.findall(r"[a-z'-]+", ln.lower())
                           if w in NUMBER_WORDS or w.split("-")[0] in NUMBER_WORDS})
        caps = sorted({w for ln in lines
                       for w in re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}", ln)})

        ok = lambda b: "PASS" if b else "**FAIL**"
        print(f"\n--- {tag} — {title} ---")
        print(f"  style chars      {len(style):5d}   "
              f"[{STYLE_MIN}-{STYLE_MAX}, target {STYLE_TARGET[0]}-{STYLE_TARGET[1]}]  "
              f"{ok(STYLE_MIN <= len(style) <= STYLE_MAX)}"
              f"{'  (>=985 FLAG)' if len(style) >= STYLE_FLAG else ''}")
        print(f"  style ends '.'   {style[-1:]!r:>7}   "
              f"{ok(style[-1:] in '.!?')}")
        print(f"  exclude chars    {len(excl):5d}   items={excl.count(',') + 1:3d}   "
              f"{ok(len(excl) > 80 and excl.count(',') >= 8)}")
        print(f"  lyric chars      {len(lyrics):5d}   [<= {LYRIC_MAX}]              "
              f"{ok(len(lyrics) <= LYRIC_MAX)}")
        print(f"  sung lines       {p['lines']:5d}   "
              f"[{LINES_MIN}-{LINES_MAX}, target {LINES_TARGET[0]}-{LINES_TARGET[1]}]     "
              f"{ok(LINES_MIN <= p['lines'] <= LINES_MAX)}")
        print(f"  end rhyme        {p['end_rhyme']:5.3f}   [>= {RHYME_FLOOR}]"
              f"                {ok(p['end_rhyme'] >= RHYME_FLOOR)}")
        print(f"  line return      {p['line_return']:5.3f}   [>= {RETURN_FLOOR}]"
              f"                {ok(p['line_return'] >= RETURN_FLOOR)}")
        print(f"  words / line     {p['words_per_line']:5.2f}   [<= {WPL_CEIL}]"
              f"                {ok(p['words_per_line'] <= WPL_CEIL)}")
        print(f"  allit /100w      {p['allit_per_100w']:5.2f}   (reported, no gate)")
        print(f"  acts I/II/III    {acts['I']}/{acts['II']}/{acts['III']} lines   "
              f"proxy runtime {proxy[0]:.0f}/{proxy[1]:.0f}/{proxy[2]:.0f}%   "
              f"{ok(min(proxy) >= 20.0)}")
        print(f"  HARDWARE nouns   {len(hw):5d}   [== 0, stem-scan]           "
              f"{ok(not hw)}  {sorted({h[0] for h in hw}) if hw else ''}")
        print(f"  interface nouns  {len(iface):5d}   [self-imposed == 0]        "
              f"{ok(not iface)}  {iface if iface else ''}")
        print(f"  pan tokens       {len(pan):5d}   [== 0, all fields]          "
              f"{ok(not pan)}  {pan if pan else ''}")
        print(f"  marque tokens    {len(marque):5d}   [== 0, all fields]          "
              f"{ok(not marque)}  {marque if marque else ''}")
        print(f"  digits in sung   {len(digits):5d}   [== 0]                      "
              f"{ok(not digits)}  {digits if digits else ''}")
        print(f"  SFX cues         {len(sfx_standalone):5d}   [>= 1, standalone *..*]     "
              f"{ok(len(sfx_standalone) >= 1)}   (literal 'SFX:' form: "
              f"{len(sfx_literal)}  {ok(len(sfx_literal) >= 1)})")
        for c in sfx_standalone:
            print(f"      · {c}")
        print(f"  named breaks     {len(brk):5d}   [credit drummer if >0]      "
              f"{'NONE' if not brk else '**FIRES**'}  "
              f"{[(b, BREAK_NAMES[b]) for b in brk] if brk else ''}")
        print(f"  number-word lines ({len(numlines)}) — ADJUDICATE EACH BY HAND:")
        for ln in numlines:
            print(f"      · {ln}")
        if caps:
            print(f"  mid-line capitals — adjudicate: {caps}")
        rows.append((tag, title, len(style), len(lyrics), p, acts, proxy))

    for m in re.finditer(r"^#{1,3}\s*(D\d?)\.?\s*[—-]?\s*DEDICATION[^\n]*$",
                         text, re.M | re.I):
        ded = grab(text[m.start():], r"^#{1,3}[^\n]*DEDICATION[^\n]*$")
        if ded:
            label = m.group(0).strip().encode("ascii", "replace").decode()[:60]
            print(f"\nDEDICATION [{label}]"
                  f"\n  chars {len(ded):5d}   [<= {DEDICATION_MAX}]   "
                  f"{'PASS' if len(ded) <= DEDICATION_MAX else '**FAIL**'}")
    for m in re.finditer(r"^#{1,3}\s*D\d\s*[—-][^\n]*$", text, re.M):
        ded = grab(text[m.start():], r"^#{1,3}\s*D\d\s*[—-][^\n]*$")
        if ded:
            label = m.group(0).strip().encode("ascii", "replace").decode()[:70]
            print(f"\nDEDICATION [{label}]"
                  f"\n  chars {len(ded):5d}   [<= {DEDICATION_MAX}]   "
                  f"{'PASS' if len(ded) <= DEDICATION_MAX else '**FAIL**'}")
    return rows


if __name__ == "__main__":
    for f in sys.argv[1:]:
        measure(f)
