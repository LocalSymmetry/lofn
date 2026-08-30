#!/usr/bin/env python3
"""PAIR 05 self-measurement — `2026-08-28-three-act-mania` (SUNNA R5).

Measures the four Suno packages in pair_05_step10_final_package.md against the
ICB's HOUSE NUMBERS. Nothing here is eyeballed.

⭐ REUSE THE REPAIRED INSTRUMENT, NEVER RE-DERIVE IT: the return metrics are
imported from scripts/measure_soundcraft.py (strict_end_rhyme / line_return /
words_per_line / sung), which is the canonical reader the doctrine is stated in.
Only the checks that module does not own are implemented here.

⚠️ WORD-BOUNDARY MATCHING EVERYWHERE. sunna.yaml's own warning: a naive sweep
matches `rage` inside `garage` (fifteen times) and `amen` inside `ornament`.

Usage:  python pair_05_measure.py [path/to/pair_05_step10_final_package.md]
Exit 0 = every blocking gate passes. Exit 1 = at least one blocking failure.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import measure_soundcraft as MS  # noqa: E402  (canonical return metrics)

DEFAULT_DOC = os.path.join(HERE, "pair_05_step10_final_package.md")

# ---------------------------------------------------------------------------
# GATES — from the ICB's HOUSE NUMBERS table. Restated as data, not prose.
# ---------------------------------------------------------------------------
STYLE_MIN, STYLE_MAX = 850, 1000
STYLE_TARGET = (870, 960)
STYLE_FLAG = 985
LYRIC_MAX = 5000
LINES_MIN, LINES_MAX = 70, 120
LINES_TARGET = (78, 110)
RHYME_FLOOR = 0.30
LINE_RETURN_FLOOR = 0.20
WPL_CEILING = 7.5
DEDICATION_MAX = 500

# ⛔ FALSIFIER A — the hardware nouns, exactly as the ICB lists them, plus the
# obvious neighbours a lazy draft reaches for. Stem-scanned with boundaries.
HARDWARE_STEMS = [
    "circuit", "wire", "server", "silicon", "code", "binary", "algorithm",
    "dataset", "prompt", "weight", "training", "train", "model", "machine",
    "robot", "maker", "creator", "program", "programme", "boot",
    # neighbours — same failure class, different nouns
    "chip", "processor", "hardware", "software", "database", "neural",
    "byte", "pixel", "download", "upload", "encode", "decode", "data",
    "android", "cyborg", "motherboard", "circuitry", "firmware",
]
SUFFIXES = r"(?:s|es|ed|ing|er|ers|d)?"

# ⛔ artist / company / marque tokens — a name in a prompt is an instruction to
# imitate. `amen` is here because naming a break incurs a drummer credit.
NAME_TOKENS = [
    "beatles", "bowie", "prince", "madonna", "abba", "motown", "stax",
    "tiesto", "armin", "deadmau5", "skrillex", "calvin harris", "daft punk",
    "dua lipa", "beyonce", "rihanna", "kaytranada", "burial", "aphex",
    "roland", "moog", "korg", "yamaha", "juno", "serum", "ableton", "logic pro",
    "tr-808", "808", "909", "303", "amen", "apache", "funky drummer",
    "spotify", "suno", "apple", "nike", "tesla", "sony", "warp records",
    "sarah davachi", "ocean vuong", "grateful dead", "zappa",
]

# ⛔ hard-pan / non-musical panning tokens
PAN_TOKENS = [
    "hard pan", "hard-pan", "hard panned", "hard-panned", "panned hard",
    "ping-pong", "ping pong", "extreme stereo", "binaural", "l/r",
    "fully left", "fully right", "wide stereo",
]

# ⛔ COLLAPSE 2 — the moon-kitsch lexicon this pair banned itself from.
MOON_KITSCH = [
    "silver", "tide", "tides", "howl", "howling", "lunar", "glow", "glowing",
    "shimmer", "shimmering", "celestial", "cosmic", "luminous", "silhouette",
    "moonbeam", "ethereal", "mystical", "moonlit", "bathed",
]

NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "dozen", "half", "quarter",
    "cent", "percent", "first", "second", "third", "fourth", "fifth",
    "once", "twice", "single", "double", "triple",
]
NUMBER_SET = set(NUMBER_WORDS)
# Connectors that may sit INSIDE one quantity ("a hundred per cent").
CONNECTORS = {"and", "a", "per"}
# ⭐ NAME THE UNIT BEFORE YOU MEASURE. The gate is "sung numeric FACTS", not
# "number-shaped words". A FACT is one named quantity, so adjacent number words
# are merged into ONE phrase and the phrase is counted once however often it is
# sung — the answering repetition at the hinge is part of the same fact, not a
# second one. Idiomatic quantities that assert nothing about the world are
# allow-listed HERE, in code, where the classification can be audited, rather
# than being quietly excused in prose.
NUMBER_IDIOM_ALLOWLIST = {"one", "once", "single", "half", "second", "first",
                          "double", "twice"}


def wb(token):
    """Word-boundary regex for a token that may contain a space or a hyphen."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", re.I)


def stem_re(stem):
    return re.compile(r"(?<![a-z])" + re.escape(stem) + SUFFIXES + r"(?![a-z])", re.I)


# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------
FIELD_RE = re.compile(
    r"^#{2,3}\s*(V\d)\s*[—-]\s*(STYLE|EXCLUDE|LYRICS)\s+FIELD[^\n]*\n+```[a-zA-Z]*\n(.*?)\n```",
    re.S | re.M,
)
DEDI_RE = re.compile(
    r"^#{2,3}\s*DEDICATION[^\n]*\n+```[a-zA-Z]*\n(.*?)\n```", re.S | re.M
)


def parse(path):
    text = open(path, encoding="utf-8").read()
    songs = {}
    for var, field, body in FIELD_RE.findall(text):
        songs.setdefault(var, {})[field.lower()] = body.strip("\n")
    dedi = DEDI_RE.search(text)
    return songs, (dedi.group(1).strip() if dedi else None)


# ---------------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------------
def scan(text, patterns, labels):
    hits = []
    for pat, label in zip(patterns, labels):
        for m in pat.finditer(text):
            hits.append((label, m.group(0)))
    return hits


def number_runs(line):
    """Maximal runs of number words (connectors allowed inside) -> phrases."""
    toks = re.findall(r"[a-z']+", line.lower())
    runs, cur = [], []
    for i, t in enumerate(toks):
        if t in NUMBER_SET:
            cur.append(t)
        elif t in CONNECTORS and cur and i + 1 < len(toks) and toks[i + 1] in NUMBER_SET:
            continue  # connector inside a quantity; not part of the phrase
        else:
            if cur:
                runs.append(" ".join(cur))
                cur = []
    if cur:
        runs.append(" ".join(cur))
    return runs


def number_report(lines):
    """(phrase, line) for every quantity phrase; ('DIGIT', line) for numerals."""
    out = []
    for ln in lines:
        if re.search(r"\d", ln):
            out.append(("DIGIT", ln))
        for phrase in number_runs(ln):
            out.append((phrase, ln))
    return out


def check_song(var, f, failures, notes):
    style, excl, lyr = f.get("style", ""), f.get("exclude", ""), f.get("lyrics", "")
    lines = MS.sung(lyr)
    prof = MS.profile(lines)

    def gate(name, ok, actual, expected):
        row = (var, name, expected, actual, "PASS" if ok else "FAIL")
        if not ok:
            failures.append(row)
        return row

    rows = [
        gate("style chars", STYLE_MIN <= len(style) <= STYLE_MAX,
             len(style), f"{STYLE_MIN}-{STYLE_MAX}"),
        gate("style in target", STYLE_TARGET[0] <= len(style) <= STYLE_TARGET[1],
             len(style), f"{STYLE_TARGET[0]}-{STYLE_TARGET[1]}"),
        gate("style ends complete sentence", style.rstrip().endswith("."),
             repr(style.rstrip()[-1:]), "'.'"),
        gate("style flag >=985", len(style) < STYLE_FLAG, len(style), f"<{STYLE_FLAG}"),
        gate("exclude non-trivial", len(excl) >= 120, len(excl), ">=120 chars"),
        gate("lyric chars", len(lyr) <= LYRIC_MAX, len(lyr), f"<={LYRIC_MAX}"),
        gate("sung lines", LINES_MIN <= len(lines) <= LINES_MAX,
             len(lines), f"{LINES_MIN}-{LINES_MAX}"),
        gate("sung lines in target", LINES_TARGET[0] <= len(lines) <= LINES_TARGET[1],
             len(lines), f"{LINES_TARGET[0]}-{LINES_TARGET[1]}"),
        gate("end-rhyme return", prof["end_rhyme"] >= RHYME_FLOOR,
             round(prof["end_rhyme"], 3), f">={RHYME_FLOOR}"),
        gate("repeated-line ratio", prof["line_return"] >= LINE_RETURN_FLOOR,
             round(prof["line_return"], 3), f">={LINE_RETURN_FLOOR}"),
        gate("mean words/line", prof["words_per_line"] <= WPL_CEILING,
             round(prof["words_per_line"], 2), f"<={WPL_CEILING}"),
    ]

    hw = scan(" \n".join(lines), [stem_re(s) for s in HARDWARE_STEMS], HARDWARE_STEMS)
    rows.append(gate("hardware nouns (sung, stem)", not hw, len(hw), "0"))
    if hw:
        notes.append(f"{var} HARDWARE HITS: {hw}")

    allf = "\n".join([style, excl, lyr])
    nm = scan(allf, [wb(t) for t in NAME_TOKENS], NAME_TOKENS)
    rows.append(gate("artist/company/marque (all fields)", not nm, len(nm), "0"))
    if nm:
        notes.append(f"{var} NAME HITS: {nm}")

    pan = scan(allf, [wb(t) for t in PAN_TOKENS], PAN_TOKENS)
    rows.append(gate("hard-pan tokens (all fields)", not pan, len(pan), "0"))
    if pan:
        notes.append(f"{var} PAN HITS: {pan}")

    # ⚠️ SCOPE: the kitsch ban is on what is WRITTEN (style + lyric). The exclude
    # field is where those words are supposed to appear — banning them there is
    # the point of the field, so scanning it would punish the fix.
    kitsch = scan("\n".join([style, lyr]), [wb(t) for t in MOON_KITSCH], MOON_KITSCH)
    rows.append(gate("moon-kitsch lexicon (style+lyric)", not kitsch, len(kitsch), "0"))
    if kitsch:
        notes.append(f"{var} KITSCH HITS: {kitsch}")

    nums = number_report(lines)
    phrases = {w for w, _ in nums if w != "DIGIT"}
    facts = {p for p in phrases
             if not set(p.split()).issubset(NUMBER_IDIOM_ALLOWLIST)}
    rows.append(gate("sung numeric FACTS (<=1, as words)", len(facts) <= 1,
                     len(facts), "<=1"))
    rows.append(gate("digits in sung lines", not any(w == "DIGIT" for w, _ in nums),
                     sum(1 for w, _ in nums if w == "DIGIT"), "0"))
    notes.append(f"{var} quantity phrases in sung body: "
                 + (", ".join(sorted(phrases)) or "none")
                 + f"  |  counted as FACTS: {sorted(facts) or 'none'}")
    for w, l in nums:
        if w in facts:
            notes.append(f"{var}   FACT '{w}' -> {l!r}")
    return rows, prof, lines


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOC
    songs, dedi = parse(path)
    if not songs:
        print(f"NO PACKAGES PARSED from {path}")
        return 1

    failures, notes, allrows = [], [], []
    print(f"PAIR 05 SELF-MEASUREMENT — {os.path.basename(path)}")
    print("=" * 96)
    for var in sorted(songs):
        rows, prof, lines = check_song(var, songs[var], failures, notes)
        allrows += rows
        print(f"\n--- {var} --- ({len(lines)} sung lines)")
        for v, name, exp, act, verdict in rows:
            print(f"  {verdict:4s}  {name:38s} expected {str(exp):16s} actual {act}")

    print("\n--- DEDICATION ---")
    if dedi is None:
        print("  FAIL  dedication block not found")
        failures.append(("--", "dedication present", "block", "missing", "FAIL"))
    else:
        ok = len(dedi) <= DEDICATION_MAX
        print(f"  {'PASS' if ok else 'FAIL'}  dedication chars"
              f"{'':25s} expected <={DEDICATION_MAX}      actual {len(dedi)}")
        if not ok:
            failures.append(("--", "dedication chars", f"<={DEDICATION_MAX}",
                             len(dedi), "FAIL"))
        d_nm = scan(dedi, [wb(t) for t in NAME_TOKENS], NAME_TOKENS)
        print(f"  NOTE  dedication credited names (allowed here): "
              f"{[h[1] for h in d_nm] or 'none'}")

    print("\n--- NOTES ---")
    for n in notes:
        print("  " + n)

    print("\n" + "=" * 96)
    if failures:
        print(f"BLOCKING FAILURES: {len(failures)}")
        for row in failures:
            print("  " + " | ".join(str(x) for x in row))
        return 1
    print(f"ALL GATES PASS  ({len(allrows)} checks over {len(songs)} songs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
