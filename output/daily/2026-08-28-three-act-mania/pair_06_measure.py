#!/usr/bin/env python3
"""PAIR 06 self-measurement -- 2026-08-28-three-act-mania.

Reads pair_06_step10_final_package.md, pulls the three PASTED Suno fields out of
each variation, and measures every number in the ICB HOUSE NUMBERS table.

DOCTRINE NOTE -- "reuse the repaired instrument, never re-derive it":
the four canonical return metrics (strict_end_rhyme / line_return /
words_per_line) and the canonical `sung()` line-extractor are IMPORTED from
scripts/measure_soundcraft.py. They are not reimplemented here. Only the scans
that module does not own (hardware-noun stem scan, lexicon scan, numeric-fact
scan, char counts, act split) are written locally.

Usage:  python3 pair_06_measure.py [path/to/pair_06_step10_final_package.md]
Exit 0 = every gate passes. Exit 1 = at least one gate FAILS (reported as a
failure, never rounded into a pass).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from measure_soundcraft import sung, strict_end_rhyme, line_return, words_per_line  # noqa: E402

DEFAULT_TARGET = os.path.join(HERE, "pair_06_step10_final_package.md")

# --------------------------------------------------------------------------
# R5 FALSIFIER A -- the hardware noun. STEM scan, not token scan (ICB R5).
# --------------------------------------------------------------------------
HARDWARE_STEMS = [
    "circuit", "wire", "server", "silicon", "code", "binar", "algorithm",
    "dataset", "prompt", "weight", "train", "model", "machin", "robot",
    "maker", "creator", "program", "boot",
]

# --------------------------------------------------------------------------
# vault/field_token_lexicon.yaml -- companies / artists / spatial. Parsed from
# the vault file itself so this script cannot drift from the coordinator's list.
# --------------------------------------------------------------------------
def load_lexicon():
    path = os.path.join(REPO, "vault", "field_token_lexicon.yaml")
    out, cur = {}, None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if raw.startswith("#") or not raw.strip():
                continue
            if re.match(r"^[a-z_]+:\s*$", raw):
                cur = raw.split(":")[0].strip()
                out[cur] = []
            elif raw.lstrip().startswith("- ") and cur:
                out[cur].append(raw.split("- ", 1)[1].strip())
    return out

# --------------------------------------------------------------------------
# ONE-FACT RULE -- number words, grouped into expressions ("eight hundred feet"
# is ONE numeric fact, not two). Digits are a separate, always-fatal class.
# --------------------------------------------------------------------------
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "dozen", "twice", "once", "half",
    "quarter", "single", "double", "triple", "dual",
}
JOINERS = {"and", "a", "of"}


def numeric_expressions(lines):
    """Return grouped numeric expressions with the line they sit in."""
    found = []
    for ln in lines:
        toks = re.findall(r"[a-z']+|\d+", ln.lower())
        i = 0
        while i < len(toks):
            if toks[i] in NUMBER_WORDS or toks[i].isdigit():
                j, span = i, [toks[i]]
                while j + 1 < len(toks) and (
                    toks[j + 1] in NUMBER_WORDS
                    or toks[j + 1].isdigit()
                    or (toks[j + 1] in JOINERS
                        and j + 2 < len(toks)
                        and (toks[j + 2] in NUMBER_WORDS or toks[j + 2].isdigit()))
                ):
                    j += 1
                    span.append(toks[j])
                found.append((" ".join(span), ln))
                i = j + 1
            else:
                i += 1
    return found


def digits_in(lines):
    return [ln for ln in lines if re.search(r"\d", ln)]


def stem_hits(text, stems):
    low = text.lower()
    return sorted({s for s in stems if s in low})


def token_hits(text, tokens):
    low = text.lower()
    return sorted({t for t in tokens if t in low})


# --------------------------------------------------------------------------
# Package parsing
# --------------------------------------------------------------------------
def fenced(region):
    return re.findall(r"```[a-zA-Z]*\n(.*?)\n```", region, re.S)


def parse(path):
    txt = open(path, encoding="utf-8").read()
    marks = list(re.finditer(r"^# (V\d) . (.+)$", txt, re.M))
    out = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(txt)
        region = txt[m.end():end]
        blocks = fenced(region)
        if len(blocks) < 3:
            print(f"  !! {m.group(1)}: expected 3 fenced blocks, found {len(blocks)}")
            continue
        out.append(dict(vid=m.group(1), title=m.group(2).strip(),
                        style=blocks[0].strip(), exclude=blocks[1].strip(),
                        lyrics=blocks[2].strip()))
    ded = re.search(r"^## DEDICATION.*?```[a-zA-Z]*\n(.*?)\n```", txt, re.S | re.M)
    return out, (ded.group(1).strip() if ded else None)


def act_boundary_words(lyrics):
    """R3(i) checked mechanically: the LAST sung word of each act and the FIRST
    sung word of the next. A word-leap seam is only real if these are the same
    word. Eyeballing this is exactly the class of check that has failed before.
    """
    seq, cur = [], None
    for ln in lyrics.splitlines():
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            u = s.upper()
            for key in ("ACT ONE", "ACT TWO", "ACT THREE"):
                if key in u:
                    cur = key
            continue
        got = sung(s)
        if cur and got:
            seq.append((cur, got[0]))
    out = []
    for a, b in (("ACT ONE", "ACT TWO"), ("ACT TWO", "ACT THREE")):
        last = [t for act, t in seq if act == a]
        first = [t for act, t in seq if act == b]
        lw = re.findall(r"[A-Za-z']+", last[-1])[-1].lower() if last else "?"
        fw = re.findall(r"[A-Za-z']+", first[0])[0].lower() if first else "?"
        out.append((a, b, lw, fw, lw == fw))
    return out


POLE = "the water does not know"


def act_split(lyrics):
    """Sung lines per ACT, and POLE occurrences per ACT, from the act headers.

    R4 requires the pole in ALL THREE acts, identity unchanged. That is also the
    one thing the render prediction says must survive, so it is counted, never
    assumed.
    """
    acts = {"ACT ONE": 0, "ACT TWO": 0, "ACT THREE": 0}
    pole = {"ACT ONE": 0, "ACT TWO": 0, "ACT THREE": 0}
    cur = None
    for ln in lyrics.splitlines():
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            u = s.upper()
            for key in ("ACT ONE", "ACT TWO", "ACT THREE"):
                if key in u:
                    cur = key
            continue
        got = sung(s)
        if cur and got:
            acts[cur] += 1
            if got[0].lower().strip(" .,") == POLE:
                pole[cur] += 1
    return acts, pole


# --------------------------------------------------------------------------
def main():
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    lex = load_lexicon()
    marque = lex.get("companies", []) + lex.get("artists", [])
    spatial = lex.get("spatial_noninstructions", [])
    songs, dedication = parse(target)
    if not songs:
        print("NO PACKAGES PARSED"); return 2

    fails = []
    rows = []
    for s in songs:
        lines = sung(s["lyrics"])
        prof_rhyme = strict_end_rhyme(lines)
        prof_ret = line_return(lines)
        prof_wpl = words_per_line(lines)
        hw = stem_hits(" ".join(lines), HARDWARE_STEMS)
        hw_all = stem_hits(s["style"] + " " + s["exclude"] + " " + s["lyrics"],
                           HARDWARE_STEMS)
        mq = token_hits(s["style"] + " " + s["exclude"] + " " + s["lyrics"], marque)
        sp = token_hits(s["style"] + " " + s["exclude"] + " " + s["lyrics"], spatial)
        nums = numeric_expressions(lines)
        digs = digits_in(lines)
        acts, pole = act_split(s["lyrics"])
        tot = sum(acts.values()) or 1
        style_ok_end = s["style"].rstrip().endswith((".", "!", "?"))

        rows.append(dict(
            vid=s["vid"], title=s["title"],
            style=len(s["style"]), excl=len(s["exclude"]), lyr=len(s["lyrics"]),
            lines=len(lines), rhyme=prof_rhyme, ret=prof_ret, wpl=prof_wpl,
            hw=hw, hw_all=hw_all, mq=mq, sp=sp, nums=nums, digs=digs,
            acts=acts, pole=pole,
            pct={k: 100.0 * v / tot for k, v in acts.items()},
            style_end=style_ok_end,
        ))

        v = s["vid"]
        if not 850 <= len(s["style"]) <= 1000:
            fails.append(f"{v} style chars {len(s['style'])} outside 850-1000")
        if not style_ok_end:
            fails.append(f"{v} style field does not end as a complete sentence")
        if len(s["exclude"]) < 200:
            fails.append(f"{v} exclude field trivial ({len(s['exclude'])} chars)")
        if len(s["lyrics"]) > 5000:
            fails.append(f"{v} lyrics field {len(s['lyrics'])} > 5000")
        if not 70 <= len(lines) <= 120:
            fails.append(f"{v} sung lines {len(lines)} outside 70-120")
        if prof_rhyme < 0.30:
            fails.append(f"{v} rhyme return {prof_rhyme:.3f} < 0.30")
        if prof_ret < 0.20:
            fails.append(f"{v} line return {prof_ret:.3f} < 0.20")
        if prof_wpl > 7.5:
            fails.append(f"{v} mean words/line {prof_wpl:.2f} > 7.5")
        if hw:
            fails.append(f"{v} HARDWARE NOUN in sung lines: {hw}")
        if mq:
            fails.append(f"{v} artist/company token: {mq}")
        if sp:
            fails.append(f"{v} spatial non-instruction token: {sp}")
        if len(nums) > 1:
            fails.append(f"{v} sung numeric facts {len(nums)}: {[n[0] for n in nums]}")
        if digs:
            fails.append(f"{v} DIGITS in sung lines: {digs}")
        for k, pv in rows[-1]["pct"].items():
            if pv < 20.0:
                fails.append(f"{v} {k} is {pv:.1f}% of sung lines (< 20%)")
        for k, pc in pole.items():
            if pc < 1:
                fails.append(f"{v} POLE absent from {k} (R4)")

    hdr = (f"{'':4s} {'style':>6s} {'excl':>5s} {'lyric':>6s} {'lines':>6s} "
           f"{'endRhy':>7s} {'lineRet':>8s} {'w/line':>7s} {'hw':>3s} {'mq':>3s} "
           f"{'num':>4s} {'I%':>5s} {'II%':>5s} {'III%':>5s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['vid']:4s} {r['style']:6d} {r['excl']:5d} {r['lyr']:6d} "
              f"{r['lines']:6d} {r['rhyme']:7.3f} {r['ret']:8.3f} {r['wpl']:7.2f} "
              f"{len(r['hw']):3d} {len(r['mq']):3d} {len(r['nums']):4d} "
              f"{r['pct']['ACT ONE']:5.1f} {r['pct']['ACT TWO']:5.1f} "
              f"{r['pct']['ACT THREE']:5.1f}")

    print("\nNUMERIC FACTS SUNG (must be <= 1 per song, as words, never digits)")
    for r in rows:
        for expr, ln in r["nums"]:
            print(f"  {r['vid']}  '{expr}'   <- {ln}")
        if not r["nums"]:
            print(f"  {r['vid']}  (none)")

    print("\nSEAM WORD-LEAPS (R3(i)) -- last sung word of act N == first sung word of act N+1")
    for s in songs:
        for a, b, lw, fw, ok in act_boundary_words(s["lyrics"]):
            tag = "OK " if ok else "FAIL"
            print(f"  {s['vid']}  {a} ends '{lw}' -> {b} opens '{fw}'   {tag}")
            if not ok:
                fails.append(f"{s['vid']} seam {a}->{b}: '{lw}' != '{fw}'")

    print("\nPOLE PER ACT (R4) -- 'The water does not know', exact, identity unchanged")
    for r in rows:
        print(f"  {r['vid']}  ACT ONE x{r['pole']['ACT ONE']}  "
              f"ACT TWO x{r['pole']['ACT TWO']}  ACT THREE x{r['pole']['ACT THREE']}")

    print("\nHARDWARE-NOUN STEM SCAN (R5 Falsifier A)")
    for r in rows:
        print(f"  {r['vid']}  sung lines: {r['hw'] or 'CLEAN (0)'}   "
              f"all three fields: {r['hw_all'] or 'CLEAN (0)'}")

    if dedication is not None:
        n = len(dedication)
        print(f"\nDEDICATION  {n} chars  (ceiling 500)  "
              f"{'OK' if n <= 500 else 'FAIL'}")
        if n > 500:
            fails.append(f"dedication {n} > 500 chars")
    else:
        print("\nDEDICATION  NOT FOUND")
        fails.append("dedication block not found")

    print()
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  FAIL  " + f)
        return 1
    print("ALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
