#!/usr/bin/env python3
"""PAIR 04 self-measurement — `2026-08-28-three-act-mania`.

Measures the four Suno packages against the ICB house numbers. Nothing is eyeballed.

Rhyme return and the sung-line definition are IMPORTED from the house instrument
(`scripts/measure_soundcraft.py`) so the numbers are the house numbers, not a
re-implementation of them. `A GATE WITH NO READER IS NOT A GATE` — and a reader
that re-derives the measure is a second, unaudited instrument.

Usage:  python pair_04_measure.py [path-to-package-md]
        default = pair_04_step10_final_package.md in this directory
"""
import os
import re
import sys

try:                                   # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import measure_soundcraft as MS  # noqa: E402  -- the house instrument

# ---------------------------------------------------------------- gates (ICB)
G = dict(
    style_min=850, style_max=1000, style_lo=870, style_hi=960, style_flag=985,
    exclude_min_chars=60,
    lyric_max=5000,
    lines_min=70, lines_max=120, lines_lo=78, lines_hi=110,
    rhyme_return_floor=0.30,
    line_return_floor=0.20,
    mean_words_ceiling=7.5,
    numeric_facts_max=1,
    dedication_max=500,
)

# ------------------------------------------------- R5 FALSIFIER A: stem list
# ICB list + this pair's own additions (see step06 §4). STEM scan, not token
# scan: a word-boundary scan cannot see an inflected form (2026-08-27, F5).
HARDWARE_STEMS = [
    # ICB, verbatim
    "circuit", "wire", "server", "silicon", "code", "binar", "algorithm",
    "dataset", "prompt", "weigh", "train", "model", "machine", "robot",
    "maker", "creator", "program", "boot",
    # P04's own additions (machine-noise is the failure, not the word list)
    "data", "neural", "digital", "hardware", "software", "compute",
    "processor", "chip", "token", "cyber", "android", "automat",
]
# `serv` would eat the ordinary verb "serve"; `server` is the noun we ban.
# `weigh` is the strict stem for the news item's own word, "open-weight".

PAN_TOKENS = [
    "hard pan", "hard-pan", "hardpan", "panned hard", "ping-pong", "ping pong",
    "extreme stereo", "wide stereo pan", "auto-pan", "autopan", "pan sweep",
]

# ------------------------------------------- artist / company / marque scan
# Explicit denylist of the tokens most likely to leak into a Suno style field,
# plus a capitalisation reporter for anything unexpected.
MARQUE_DENY = [
    "nine inch nails", "ministry", "skinny puppy", "front 242", "die krupps",
    "vnv nation", "combichrist", "rammstein", "kmfdm", "prodigy", "aphex",
    "goldie", "roni size", "pendulum", "noisia", "chase and status",
    "rj hampton", "dj rashad", "traxman", "spinn", "teklife", "aretha",
    "mahalia", "sam cooke", "staple singers", "kanye", "beyonce", "adele",
    "bowie", "zappa", "grateful dead", "davachi", "sunn o", "swans",
    "roland", "moog", "korg", "ableton", "serum", "kontakt", "waves",
    "melodyne", "autotune", "auto-tune", "antares", "fender", "gibson",
    "marshall", "neumann", "shure", "spotify", "suno", "udio", "openai",
    "anthropic", "google", "meta", "nvidia", "amazon", "apple", "microsoft",
    "winstons", "coleman",   # the drummer belongs in the DEDICATION, never a field
]
CAP_OK = {
    "A", "I", "The", "It", "EBM", "BPM", "EMO", "ACT", "Act", "No", "Two",
    "Come", "Then", "Open", "Filed", "Down", "Season", "Stopped", "Clear",
    "Run", "Not", "You", "Nobody", "Nothing", "Read", "Look", "Move", "Say",
    "Take", "Sit", "Find", "Line", "Every", "That", "Somebody", "Is", "Fine",
    "There", "All", "Both", "Here", "Grass", "But", "They", "While", "And",
    "Copy", "Put", "Name", "Walk", "Season.", "Stomp", "Everybody", "Give",
    "Filed.", "Down.", "季",
}

NUM_CARDINALS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "dozen",
}
# Reported SEPARATELY and deliberately NOT counted as numeric facts:
# sequence adverbs and frequency adverbs state no quantity about the world.
NUM_SEQUENCE = {"first", "second", "third", "last", "once", "twice"}


# ------------------------------------------------------------------ parsing
def parse(path):
    txt = open(path, encoding="utf-8").read()
    fields = {}
    pat = re.compile(
        r"<!--\s*MEASURE\s+(\w+)\s+(\w+)\s*-->\s*\n```[a-zA-Z]*\n(.*?)\n```",
        re.S,
    )
    for m in pat.finditer(txt):
        fields.setdefault(m.group(1), {})[m.group(2).lower()] = m.group(3).strip()
    titles = dict(re.findall(r"<!--\s*TITLE\s+(\w+)\s*:\s*(.+?)\s*-->", txt))
    return fields, titles


def stem_hits(text, stems):
    """TRUE stem scan: substring WITHIN a word, never across word boundaries.

    ⚠️ REPAIR 2026-08-28, step 09. The first version also scanned a
    letters-only string with the spaces removed, which manufactured three
    false positives across word boundaries:
        'no gri[d at a]ll'  ·  'stoppe[d at a] door'  ·  'change[d at a]ll'
    all reported as the banned stem `data`, with an EMPTY carrier list — the
    empty carrier list is the tell. Hyphenated forms are already caught by the
    word scan ('open-weight' -> 'open weight' -> the word 'weight' carries
    'weigh'), so the flattened scan bought nothing and cost three phantoms.
    ⛔ A detector written in response to a finding gets the same scrutiny as
    the finding.
    """
    words_ = re.sub(r"[^a-z ]", " ", text.lower()).split()
    out = []
    for s in stems:
        carriers = sorted({w for w in words_ if s in w})
        if carriers:
            out.append((s, carriers))
    return out


def deny_hits(text, deny):
    """⚠️ REPAIR 2026-08-28, step 09: single-token denies are matched on WORD
    BOUNDARIES. The first version substring-matched everything, so the company
    token `meta` fired on `metal percussion` in three variations. Multi-word
    denies stay as substrings (they cannot collide the same way)."""
    low = re.sub(r"[^a-z ]", " ", text.lower())
    low = re.sub(r"\s+", " ", low).strip()
    toks = set(low.split())
    out = []
    for d in deny:
        if " " in d:
            if d in low:
                out.append(d)
        elif d in toks:
            out.append(d)
    return out


def unexpected_caps(text):
    out = []
    for sent in re.split(r"(?<=[.;:!?])\s+|\n", text):
        toks = sent.strip().split()
        for i, t in enumerate(toks):
            bare = t.strip('.,;:!?()[]"\'-')
            if not bare or not bare[0].isupper():
                continue
            if i == 0 or bare in CAP_OK or bare.isupper() and len(bare) <= 4:
                continue
            out.append(bare)
    return sorted(set(out))


def rhyme_control(lines):
    """⭐ A CONTROL ON A RESULT THAT FLATTERS ME.

    `strict_end_rhyme` counts a line as a hit when another line within +/-4
    carries the same key — and an EXACTLY REPEATED line always does. This pair
    repeats its pole ~21 times per song, so the headline rhyme figure is partly
    the pole measuring itself. This re-measures end-rhyme over the NON-REPEATED
    lines only. It can only lower the number, never raise it. If this still
    clears 0.30, the rhyme is written, not returned.
    """
    def norm(s):
        s = re.sub(r"[^a-z0-9' ]", " ", s.lower())
        return re.sub(r"\s+", " ", s).strip()
    counts = {}
    for l in lines:
        k = norm(l)
        counts[k] = counts.get(k, 0) + 1
    uniq = [l for l in lines if counts[norm(l)] == 1]
    return MS.strict_end_rhyme(uniq), len(uniq)


def line_return(lines):
    """Exact repeated-line ratio: share of sung lines whose normalised text
    occurs more than once. Normalisation = lowercase, drop everything but
    letters/digits/apostrophe/space, collapse whitespace."""
    def norm(s):
        s = re.sub(r"[^a-z0-9' ]", " ", s.lower())
        return re.sub(r"\s+", " ", s).strip()
    keys = [norm(l) for l in lines]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    rep = sum(1 for k in keys if counts[k] > 1)
    return rep / max(1, len(keys)), counts


def numerals(lines):
    facts, seq = [], []
    for l in lines:
        for d in re.findall(r"\d+", l):
            facts.append(("DIGIT", d, l))
        for w in re.findall(r"[a-z']+", l.lower()):
            if w in NUM_CARDINALS:
                facts.append(("WORD", w, l))
            elif w in NUM_SEQUENCE:
                seq.append((w, l))
    return facts, seq


def acts(block):
    """Sung-line counts per ACT, from the bracketed section headers."""
    cur, out = None, {}
    for raw in block.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            m = re.match(r"\[ACT\s+(I+V?|IV|V?I*)\b", s)
            cur = m.group(1) if m else cur
            continue
        if s.startswith("*") and s.endswith("*"):
            continue
        if cur:
            out[cur] = out.get(cur, 0) + 1
    return out


def ok(flag):
    return "PASS" if flag else "**FAIL**"


# ------------------------------------------------------------------- report
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "pair_04_step10_final_package.md")
    fields, titles = parse(path)
    if not fields:
        print(f"no MEASURE blocks found in {path}")
        return 1

    print(f"# PAIR 04 SELF-MEASUREMENT\n\nsource: `{os.path.basename(path)}`")
    print(f"house instrument: `scripts/measure_soundcraft.py` "
          f"(end_rhyme, sung-line definition)\n")

    rows, fails = [], []
    for v in sorted(fields):
        f = fields[v]
        style = f.get("style", "")
        excl = f.get("exclude", "")
        lyr = f.get("lyrics", "")
        lines = MS.sung(lyr)
        n = len(lines)

        # ⛔ REPAIR 2026-08-28, step 10: THE UNIT WAS WRONG.
        # `measure_soundcraft` exposes TWO end-rhyme functions. `end_rhyme()`
        # is the diagnostic one (rime key = last vowel cluster to end);
        # `strict_end_rhyme()` is THE CANONICAL RETURN METRIC (key = last 3
        # characters) and it is what `profile()`/`profile_file()` uses -- which
        # is exactly what the ICB gate row names. The first version of this
        # script called the diagnostic one and reported it against the gate:
        # a PROXY reported as the PREDICATE. Now all four canonical numbers
        # come from `MS.profile()`, one implementation per measurement.
        prof = MS.profile(lines)
        er = prof["end_rhyme"]
        lr = prof["line_return"]
        wpl = prof["words_per_line"]
        _, counts = line_return(lines)
        facts, seq = numerals(lines)
        act_counts = acts(lyr)
        tot = sum(act_counts.values()) or 1
        pct = {k: round(100 * v2 / tot) for k, v2 in act_counts.items()}

        erc, n_uniq = rhyme_control(lines)

        hw_lyr = stem_hits(" ".join(lines), HARDWARE_STEMS)
        hw_sty = stem_hits(style + " " + excl, HARDWARE_STEMS)
        mq = deny_hits(style + " " + excl + " " + lyr, MARQUE_DENY)
        # ⚠️ REPAIR 2026-08-28, step 09: the panning gate bans panning as an
        # INSTRUCTION. It is scanned over the style field and the lyric only.
        # The first version also scanned the EXCLUDE field, where the same
        # tokens mean the opposite — it scored compliance as a violation in
        # all four variations. `pan_excluded` is the positive check.
        # ⚠️ 2026-08-28, step 10: the ICB's panning row says "0" without naming
        # a scope, while the row above it says "0 in every pasted field". Rather
        # than rule an ambiguity in my own favour, the exclude fields were
        # REWRITTEN to carry no flagged token under EITHER reading -- they now
        # say "percussion pushed to the edges of the stereo field, bouncing
        # left-right stereo effects". The positive check therefore looks for a
        # stereo-placement exclusion, not for the banned tokens themselves.
        pan = deny_hits(style + " " + lyr, PAN_TOKENS)
        pan_all = deny_hits(style + " " + excl + " " + lyr, PAN_TOKENS)
        pan_excluded = [w for w in ("stereo", "panning", "panned")
                        if w in excl.lower()]
        caps = unexpected_caps(style + "\n" + excl)

        checks = [
            (f"{v} style chars {len(style)}",
             G["style_min"] <= len(style) <= G["style_max"]),
            (f"{v} style ends as sentence",
             style.rstrip().endswith((".", "!", "?"))),
            (f"{v} exclude non-trivial ({len(excl)} chars)",
             len(excl) >= G["exclude_min_chars"]),
            (f"{v} lyric chars {len(lyr)}", len(lyr) <= G["lyric_max"]),
            (f"{v} sung lines {n}", G["lines_min"] <= n <= G["lines_max"]),
            (f"{v} rhyme return {er:.3f}", er >= G["rhyme_return_floor"]),
            (f"{v} rhyme CONTROL (repeats removed, n={n_uniq}) {erc:.3f}",
             erc >= G["rhyme_return_floor"]),
            (f"{v} line return {lr:.3f}", lr >= G["line_return_floor"]),
            (f"{v} panning excluded, not instructed", bool(pan_excluded)),
            (f"{v} mean words/line {wpl:.2f}", wpl <= G["mean_words_ceiling"]),
            (f"{v} hardware stems in lyrics {len(hw_lyr)}", not hw_lyr),
            (f"{v} hardware stems in style/exclude {len(hw_sty)}", not hw_sty),
            (f"{v} artist/company/marque {len(mq)}", not mq),
            (f"{v} pan tokens, instruction scope {len(pan)}", not pan),
            (f"{v} pan tokens, ALL fields incl. exclude {len(pan_all)}",
             not pan_all),
            (f"{v} sung numeric facts {len(facts)}",
             len(facts) <= G["numeric_facts_max"]),
            (f"{v} acts unequal, none <20% {pct}",
             len(pct) == 3 and min(pct.values()) >= 20
             and max(pct.values()) - min(pct.values()) >= 8),
        ]
        for label, good in checks:
            if not good:
                fails.append(label)

        rows.append(dict(
            v=v, title=titles.get(v, ""), style=len(style), excl=len(excl),
            lyr=len(lyr), n=n, er=er, erc=erc, lr=lr, wpl=wpl, facts=facts,
            seq=seq, pct=pct, hw_lyr=hw_lyr, hw_sty=hw_sty, mq=mq, pan=pan,
            pan_ex=pan_excluded, pan_all=pan_all, caps=caps,
            off_target=not (G["lines_lo"] <= n <= G["lines_hi"]),
            in_style_target=G["style_lo"] <= len(style) <= G["style_hi"],
            flagged=len(style) >= G["style_flag"],
            top=sorted(((c, k) for k, c in counts.items() if c > 1),
                       reverse=True)[:4],
        ))

    hdr = ("| var | title | style | excl | lyric | lines | rhyme≥.30 | "
           "rhyme CONTROL≥.30 | line≥.20 | w/line≤7.5 | hw | marque | pan | "
           "num | acts % |")
    print(hdr)
    print("|" + "---|" * 15)
    for r in rows:
        sflag = "⚠" if r["flagged"] else ("" if r["in_style_target"] else "~")
        print(f"| {r['v']} | {r['title']} | {r['style']}{sflag} "
              f"| {r['excl']} | {r['lyr']} | {r['n']}"
              f"{'⚠' if r['off_target'] else ''} | {r['er']:.3f} | "
              f"{r['erc']:.3f} | "
              f"{r['lr']:.3f} | {r['wpl']:.2f} | {len(r['hw_lyr'])+len(r['hw_sty'])} "
              f"| {len(r['mq'])} | {len(r['pan'])} | {len(r['facts'])} | "
              f"{r['pct']} |")
    print("\n`⚠` on style = at/over the 985 flag · `~` = inside 850–1000 but "
          "outside the 870–960 target · `⚠` on lines = inside 70–120 but "
          "outside the 78–110 target.")
    # ⚠️ NAME THE UNIT BEFORE YOU MEASURE (QA, step 10).
    print("\n⚠️ UNIT DECLARATION for `acts %`: this is each act's share of SUNG "
          "LINES, not of runtime. A-D2's ~20% floor is on RUNTIME. The two are "
          "the same unit ONLY IF the vocal delivery rate is constant across "
          "acts — otherwise a double-time act spends its lines in half the "
          "seconds and a 21% line share renders as ~12%. V4 (the SELECTED song) "
          "pins the rate in the style field and in every ACT II header "
          "(`VOCAL KEEPS THE 88 RATE`), which is what makes this column "
          "readable as runtime. V1–V3 do NOT carry that pin, so their `acts %` "
          "is a line share only and their fast act may render shorter.")

    print("\n## detail\n")
    for r in rows:
        print(f"**{r['v']} — {r['title']}**")
        print(f"- sung numeric facts: "
              f"{[(a, b) for a, b, _ in r['facts']] or 'none'}  "
              f"(⭐ target: exactly one, the word `two`, at the hinge)")
        print(f"- sequence/frequency words (NOT numeric facts, reported for "
              f"honesty): {sorted({w for w, _ in r['seq']}) or 'none'}")
        print(f"- hardware stems: lyrics {r['hw_lyr'] or 0} · "
              f"style/exclude {r['hw_sty'] or 0}")
        print(f"- marque hits: {r['mq'] or 'none'} · pan tokens, instruction "
              f"scope: {r['pan'] or 'none'} · pan tokens, ALL fields incl. "
              f"exclude: {r['pan_all'] or 'none'} · stereo-placement "
              f"exclusion present (positive check): "
              f"{r['pan_ex'] or 'NONE — add one'}")
        print(f"- unexpected capitalised tokens in style/exclude (review): "
              f"{r['caps'] or 'none'}")
        print(f"- most-repeated lines: "
              f"{[f'{c}x \"{k}\"' for c, k in r['top']]}")
        print()

    # ------------------------------------------------------------ dedication
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"<!--\s*MEASURE\s+DEDICATION\s*-->\s*\n```[a-zA-Z]*\n(.*?)\n```",
                  txt, re.S)
    if m:
        d = m.group(1).strip()
        good = len(d) <= G["dedication_max"]
        print(f"## dedication\n\n- chars: **{len(d)}** / {G['dedication_max']} "
              f"— {ok(good)}")
        if not good:
            fails.append(f"dedication {len(d)} chars")
    else:
        print("## dedication\n\n- **MISSING**")
        fails.append("dedication missing")

    print("\n## verdict\n")
    if fails:
        print(f"**{len(fails)} FAILING CHECK(S):**")
        for f_ in fails:
            print(f"- ⛔ {f_}")
    else:
        print("**ALL CHECKS PASS.**")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
