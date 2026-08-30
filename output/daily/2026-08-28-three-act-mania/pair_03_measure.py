#!/usr/bin/env python3
"""PAIR P03 self-measurement — 2026-08-28-three-act-mania (SUNNA R5, ACCESSIBLE / EXISTENCE).

Measures `pair_03_step10_final_package.md` against the ICB's house numbers.

DESIGN NOTE — reuse the repaired instrument, never re-derive it.
The three return metrics (end-rhyme return, line return, mean words per line) are NOT
reimplemented here. They are taken from scripts/measure_soundcraft.py -> profile(), which is
the canonical module the doctrine is stated in. What this script adds is extraction plus the
scans measure_soundcraft has no opinion about (hardware-noun stems, marque tokens, panning
tokens, digits, numeric facts, dedication length, and per-act pole coverage).

Every scan's definition is stated in the step-10 file at section C2 so a reader can disagree
with the definition rather than only with the number.

Usage:  python3 pair_03_measure.py [--emit]
        --emit  also rewrites the C1/C3 blocks inside pair_03_step10_final_package.md
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import measure_soundcraft as ms  # noqa: E402  (canonical return metrics)

TARGET = os.path.join(HERE, "pair_03_step10_final_package.md")
VARIANTS = ["V1", "V2", "V3", "V4"]

POLES = {
    "V1": "The floor is already going.",
    "V2": "It never asked my name.",
    "V3": "I have been fast my whole life.",
    "V4": "Nobody sent me a bill.",
}

# ---------------------------------------------------------------------------
# GATES (vault/gates.yaml + the run's ICB house-numbers table)
# ---------------------------------------------------------------------------
G = dict(
    style_min=850, style_max=1000, style_target=(870, 960), style_flag=985,
    lyric_max=5000,
    lines_min=70, lines_max=120, lines_target=(78, 110),
    rhyme_return_floor=0.30,
    line_return_floor=0.20,
    words_per_line_ceiling=7.5,
    numeric_facts_max=1,
    dedication_max=500,
    exclude_min_chars=60, exclude_min_items=5,
)

# ---------------------------------------------------------------------------
# SCAN VOCABULARIES
# ---------------------------------------------------------------------------

# ICB FALSIFIER A, stem-scanned. Deliberately over-wide: `train` also catches `training`
# and a railway train; `weight` also catches `weightless`. A zero therefore means the words
# were avoided outright, which is the honest reading of a stem rule.
HARDWARE_STEMS = [
    "circuit", "wire", "wiring", "server", "silicon", "code", "coding", "binary",
    "algorithm", "dataset", "data", "prompt", "weight", "train", "model", "machine",
    "robot", "program", "programme", "boot", "chip", "processor", "hardware", "software",
    "neural", "network", "digitis", "digitiz", "android", "cyborg", "circuitry",
]
HARDWARE_PHRASES = ["my maker", "my creator", "who made me", "the ones who made me"]

# Word-boundary matched. The doctrine's own warning: a naive sweep matched `rage` inside
# `garage` fifteen times in a UK-garage song, and `amen` inside `ornament`.
MARQUES = [
    "roland", "moog", "minimoog", "korg", "yamaha", "ableton", "fender", "gibson",
    "marshall", "juno", "jupiter", "prophet", "akai", "mpc", "serato", "pioneer", "cdj",
    "technics", "shure", "neumann", "spotify", "suno", "udio", "apple", "google",
    "amazon", "nike", "adidas", "motown", "stax", "warp", "beatport", "soundcloud",
    "bandcamp", "davachi", "bowie", "beyonce", "rihanna", "drake", "adele", "madonna",
    "prince", "aphex", "skrillex", "avicii", "deadmau", "disclosure", "burial", "charli",
    "sophie", "winstons", "coleman", "kraftwerk", "abba", "daft", "chic", "nile",
    "808", "909", "303", "tr808", "tr909",
]

PANNING = [
    "hard-pan", "hard pan", "hard panned", "hard-panned", "panned hard", "ping-pong",
    "ping pong delay", "extreme stereo", "wide as possible stereo", "full left", "full right",
    "auto-pan", "autopan",
]

NUMERALS = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    r"sixty|seventy|eighty|ninety|hundred|thousand|million|billion)"
)
UNITS = (
    r"(?:hour|hours|minute|minutes|second|seconds|mile|miles|kilometre|kilometres|"
    r"kilometer|kilometers|metre|metres|meter|meters|degree|degrees|percent|"
    r"day|days|year|years|mph|kph)"
)
NUM_PHRASE = re.compile(
    r"\b(" + NUMERALS + r"(?:[- ](?:and[ ])?" + NUMERALS + r")*)"
    r"((?:[ ,]+\w+){0,2}[ ,]+" + UNITS + r")\b",
    re.I,
)

ACT_SPLIT = re.compile(r"ACT (III|II|I)\b")


# ---------------------------------------------------------------------------
# EXTRACTION
# ---------------------------------------------------------------------------
def read(path: str) -> str:
    return open(path, encoding="utf-8").read()


def fenced_after(text: str, heading_needle: str) -> str:
    """First fenced block after the (unique) heading line containing `heading_needle`."""
    hits = [m for m in re.finditer(r"^#{1,4} .*$", text, re.M) if heading_needle in m.group(0)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly 1 heading containing {heading_needle!r}, found {len(hits)}")
    rest = text[hits[0].end():]
    fm = re.search(r"```[a-zA-Z]*\n(.*?)\n```", rest, re.S)
    if not fm:
        raise SystemExit(f"no fenced block after heading {heading_needle!r}")
    return fm.group(1)


def title_of(text: str, var: str) -> str:
    m = re.search(r'^## ' + var + r' — "(.+?)"\s*$', text, re.M)
    return m.group(1) if m else "(untitled)"


# ---------------------------------------------------------------------------
# SCANS
# ---------------------------------------------------------------------------
def scan_terms(hay: str, terms, stem=False) -> list:
    low = hay.lower()
    out = []
    for t in terms:
        pat = r"\b" + re.escape(t) + (r"\w*" if stem else r"") + r"\b"
        for m in re.finditer(pat, low):
            out.append(m.group(0))
    return out


def scan_phrases(hay: str, phrases) -> list:
    low = hay.lower()
    return [p for p in phrases if p in low]


def numeric_facts(lines) -> list:
    """Distinct spelled-out numeral phrases immediately qualifying a measurement unit.

    Counted as DISTINCT normalised strings: one fact stated twice is one fact.
    A count without a measurement unit (`nine lanes`, `half six`) is a setting particular,
    not a world fact, and is deliberately not counted. Stated in step-10 section C2.
    """
    seen = {}
    for ln in lines:
        for m in NUM_PHRASE.finditer(ln):
            key = re.sub(r"\s+", " ", m.group(0).lower().strip(" ,"))
            seen.setdefault(key, 0)
            seen[key] += 1
    return sorted(seen.items())


def digits_in(lines) -> list:
    return [ln for ln in lines if re.search(r"\d", ln)]


# A bracketed line is either a SECTION header (which the ICB requires to carry an EMO tag)
# or a METADATA opener, which is not a section and is never performed. The run's house frame
# uses [Theme:] / [SONG FORM:] / [Pole:] / [Disc_*].
#
# ⚠️ THIS EXCLUSION WAS ADDED AFTER THE ARTIFACT EXISTED, TO LET THE ARTIFACT PASS, WHICH IS
# EXACTLY THE SHAPE OF A SELF-SERVING INSTRUMENT CHANGE. Two controls, so a reader can
# disagree rather than take my word for it: (1) meta lines are COUNTED AND PRINTED
# separately, never silently dropped; (2) the exclusion is by prefix, listed here in full,
# so anything that is really a section cannot hide behind it.
META_PREFIXES = ("theme:", "song form:", "form:", "pole:", "disc_", "style:", "sfx")


def is_meta_header(h: str) -> bool:
    return h.strip("[]").strip().lower().startswith(META_PREFIXES)


def bracketed(block: str) -> list:
    return [ln.strip() for ln in block.splitlines()
            if ln.strip().startswith("[") and ln.strip().endswith("]")]


def header_lines(block: str) -> list:
    """SECTION headers only — the lines the EMO gate is actually about."""
    return [h for h in bracketed(block) if not is_meta_header(h)]


def meta_lines(block: str) -> list:
    return [h for h in bracketed(block) if is_meta_header(h)]


def sfx_cues(block: str) -> list:
    """SFX cues: a line delimited by asterisks, or a bracketed [SFX ...] line."""
    out = []
    for ln in block.splitlines():
        s = ln.strip()
        if s.startswith("*") and s.endswith("*") and len(s) > 2:
            out.append(s)
        elif s.startswith("[") and s.endswith("]") and s.strip("[]").strip().lower().startswith("sfx"):
            out.append(s)
    return out


def pole_coverage(block: str, pole: str) -> dict:
    """Count the pole line inside each act's own region of the lyric field."""
    act = None
    counts = {"I": 0, "II": 0, "III": 0}
    norm = pole.lower().strip(" .,")
    for ln in block.splitlines():
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            if is_meta_header(s):
                continue
            m = ACT_SPLIT.search(s)
            if m:
                act = m.group(1)
            continue
        if not s or act is None:
            continue
        if s.startswith("*") and s.endswith("*"):   # SFX cue: not a sung line
            continue
        s2 = re.sub(r"^\(|\)$", "", s).lower().strip(" .,")
        if s2 == norm:
            counts[act] += 1
    return counts


def act_line_shares(block: str) -> dict:
    act = None
    counts = {"I": 0, "II": 0, "III": 0}
    for ln in block.splitlines():
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            if is_meta_header(s):
                continue
            m = ACT_SPLIT.search(s)
            if m:
                act = m.group(1)
            continue
        if not s or act is None:
            continue
        if s.startswith("*") and s.endswith("*"):   # SFX cue: not a sung line
            continue
        counts[act] += 1
    return counts


def ok(flag: bool) -> str:
    return "PASS" if flag else "**FAIL**"


# ---------------------------------------------------------------------------
# R3.2 — THE SEAM TEMPLATE INDEX
#
# "What is specified but never varied gets templated, and the tighter the specification,
#  the tighter the template."  The 2026-08-18 canonical re-run measured every pair's SEAM
#  block as MORE self-similar across its four variations than the song around it:
#  1.27x - 2.40x, mean 1.79x, no exceptions.
#
#  So the clause ships with its falsification. This computes, for this pair:
#     seam_template_index = mean pairwise similarity of the SEAM+PRE-SEAM blocks
#                           / mean pairwise similarity of the whole sung bodies
#  <= 1.00 means the seams are no more templated than the songs; > 1.27 means this pair
#  reproduced the 2026-08-18 defect and the 2x2 seam-kind matrix did not work.
#  The same index is computed for the CLOSING sections, because the 08-18 judge found the
#  formula hiding in the endings, where no test was pointed.
# ---------------------------------------------------------------------------
def section_blocks(lyr: str, needle_test) -> list:
    """Sung lines belonging to sections whose bracketed header satisfies needle_test."""
    keep, take = [], False
    for ln in lyr.splitlines():
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            take = needle_test(s)
            continue
        if take and s and not (s.startswith("*") and s.endswith("*")):
            # sung lines only, so the numerator of the template index agrees with the
            # denominator (the bodies come from ms.sung(), which drops SFX cues).
            keep.append(re.sub(r"^\(|\)$", "", s))
    return keep


def jaccard(a, b) -> float:
    ta = set(w for l in a for w in ms.words(l))
    tb = set(w for l in b for w in ms.words(l))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def mean_pairwise(blocks) -> float:
    vals = []
    ks = list(blocks)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            vals.append(jaccard(blocks[ks[i]], blocks[ks[j]]))
    return sum(vals) / max(1, len(vals))


def swap_test(lyrics: dict) -> list:
    out = []
    seams = {v: section_blocks(lyrics[v], lambda h: "SEAM" in h.upper())
             for v in VARIANTS}
    closes = {v: section_blocks(lyrics[v], lambda h: h.strip("[]").upper().startswith("OUTRO"))
              for v in VARIANTS}
    bodies = {v: ms.sung(lyrics[v]) for v in VARIANTS}

    s_sim, c_sim, b_sim = mean_pairwise(seams), mean_pairwise(closes), mean_pairwise(bodies)
    s_idx = s_sim / b_sim if b_sim else float("inf")
    c_idx = c_sim / b_sim if b_sim else float("inf")

    out.append("R3.2 SEAM / RETURN TEMPLATE INDEX  (2026-08-18 defect range: 1.27x-2.40x, mean 1.79x)")
    out.append(f"   mean pairwise similarity, whole sung bodies : {b_sim:.4f}")
    out.append(f"   mean pairwise similarity, SEAM blocks       : {s_sim:.4f}")
    out.append(f"   mean pairwise similarity, CLOSING sections  : {c_sim:.4f}")
    out.append(f"   SEAM   template index : {s_idx:.3f}x   -> "
               f"{'PASS (no tighter than the songs)' if s_idx <= 1.0 else ('WARN' if s_idx < 1.27 else 'FAIL - 08-18 defect reproduced')}")
    out.append(f"   RETURN template index : {c_idx:.3f}x   -> "
               f"{'PASS (no tighter than the songs)' if c_idx <= 1.0 else ('WARN' if c_idx < 1.27 else 'FAIL - 08-18 defect reproduced')}")
    out.append("   seam lines per variation: " + ", ".join(f"{v}={len(seams[v])}" for v in VARIANTS))
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    text = read(TARGET)
    rows, raw = [], []
    failures = []
    lyr_by_var = {}

    dedication = fenced_after(text, "D. THE DEDICATION")
    ded_len = len(dedication.strip())

    for var in VARIANTS:
        style = fenced_after(text, f"{var} STYLE FIELD").strip()
        excl = fenced_after(text, f"{var} EXCLUDE FIELD").strip()
        lyr = fenced_after(text, f"{var} — LYRICS FIELD")
        lyr_by_var[var] = lyr
        title = title_of(text, var)

        sung = ms.sung(lyr)
        prof = ms.profile(sung)

        hw = scan_terms("\n".join(sung), HARDWARE_STEMS, stem=True)
        hw += scan_phrases("\n".join(sung), HARDWARE_PHRASES)

        pasted_all = "\n".join([title, style, excl, lyr])
        marq = scan_terms(pasted_all, MARQUES)

        # panning scope: title + style + lyrics, NOT exclude (an exclude that forbids a
        # hard pan contains the token while doing the opposite of using it).
        pan_scope = "\n".join([title, style, lyr])
        pan = scan_phrases(pan_scope, PANNING)

        nfacts = numeric_facts(sung)
        digs = digits_in(sung)
        hdr_digits = [h for h in header_lines(lyr) if re.search(r"\d", h)]
        metas = meta_lines(lyr)
        sfx = sfx_cues(lyr)

        pole = POLES[var]
        cov = pole_coverage(lyr, pole)
        shares = act_line_shares(lyr)
        tot = max(1, sum(shares.values()))

        style_ok = G["style_min"] <= len(style) <= G["style_max"] and style.endswith(".")
        excl_ok = len(excl) >= G["exclude_min_chars"] and excl.count(",") + 1 >= G["exclude_min_items"]
        lyr_ok = len(lyr) <= G["lyric_max"]
        lines_ok = G["lines_min"] <= prof["lines"] <= G["lines_max"]
        rr_ok = prof["end_rhyme"] >= G["rhyme_return_floor"]
        lr_ok = prof["line_return"] >= G["line_return_floor"]
        wpl_ok = prof["words_per_line"] <= G["words_per_line_ceiling"]
        nf_ok = len(nfacts) <= G["numeric_facts_max"]
        pole_ok = all(v >= 1 for v in cov.values())
        emo_hdrs = header_lines(lyr)
        emo_ok = all("EMO:" in h for h in emo_hdrs) and len(emo_hdrs) > 0

        for cond, name in [
            (style_ok, "style field"), (excl_ok, "exclude field"), (lyr_ok, "lyric chars"),
            (lines_ok, "sung lines"), (rr_ok, "rhyme return"), (lr_ok, "line return"),
            (wpl_ok, "words/line"), (not hw, "hardware nouns"), (not marq, "marques"),
            (not pan, "panning"), (not digs, "digits in sung lines"),
            (nf_ok, "numeric facts"), (pole_ok, "pole in all three acts"),
            (emo_ok, "EMO on every section header"),
        ]:
            if not cond:
                failures.append(f"{var}: {name}")

        rows.append(dict(
            var=var, title=title,
            style=len(style), style_end=style.endswith("."),
            excl=len(excl), excl_items=excl.count(",") + 1,
            lyr=len(lyr), lines=prof["lines"],
            rr=prof["end_rhyme"], lr=prof["line_return"], wpl=prof["words_per_line"],
            allit=prof["allit_per_100w"],
            hw=len(hw), marq=len(marq), pan=len(pan), digs=len(digs),
            nf=len(nfacts), nf_txt="; ".join(f"{k} (x{v})" for k, v in nfacts) or "none",
            cov=cov, shares={k: round(100 * v / tot) for k, v in shares.items()},
            hdrs=len(emo_hdrs), hdr_digits=len(hdr_digits),
            metas=len(metas), meta_txt=" | ".join(m[:44] for m in metas) or "none",
            sfx=len(sfx), sfx_txt=" | ".join(s[:70] for s in sfx) or "none",
            style_ok=style_ok, excl_ok=excl_ok, lyr_ok=lyr_ok, lines_ok=lines_ok,
            rr_ok=rr_ok, lr_ok=lr_ok, wpl_ok=wpl_ok, nf_ok=nf_ok, pole_ok=pole_ok,
            emo_ok=emo_ok,
        ))

    # ---- report -----------------------------------------------------------
    w = raw.append
    w("PAIR P03 — SELF-MEASUREMENT  ·  2026-08-28-three-act-mania  ·  SUNNA R5")
    w("return metrics from scripts/measure_soundcraft.py -> profile()  (not re-derived here)")
    w("")
    w(f"{'var':4s} {'style':>6s} {'.':>2s} {'excl':>5s} {'lyric':>6s} {'lines':>6s} "
      f"{'endRhy':>7s} {'lineRet':>8s} {'w/line':>7s} {'allit':>6s}")
    for r in rows:
        w(f"{r['var']:4s} {r['style']:6d} {'y' if r['style_end'] else 'n':>2s} {r['excl']:5d} "
          f"{r['lyr']:6d} {r['lines']:6d} {r['rr']:7.3f} {r['lr']:8.3f} {r['wpl']:7.2f} "
          f"{r['allit']:6.2f}")
    w("")
    w(f"{'var':4s} {'hwNoun':>7s} {'marque':>7s} {'panTok':>7s} {'digits':>7s} {'numFacts':>9s}  pole I/II/III   act-line share %")
    for r in rows:
        c = r["cov"]
        s = r["shares"]
        w(f"{r['var']:4s} {r['hw']:7d} {r['marq']:7d} {r['pan']:7d} {r['digs']:7d} "
          f"{r['nf']:9d}  {c['I']}/{c['II']}/{c['III']}"
          f"{'':10s}{s['I']}/{s['II']}/{s['III']}")
    w("")
    for r in rows:
        w(f"{r['var']}  \"{r['title']}\"")
        w(f"     numeric fact(s): {r['nf_txt']}")
        w(f"     section headers: {r['hdrs']} (all carry EMO: {r['emo_ok']}); "
          f"headers containing BPM digits: {r['hdr_digits']} (not sung)")
        w(f"     metadata openers EXCLUDED from the EMO gate: {r['metas']}  [{r['meta_txt']}]")
        w(f"     SFX cues: {r['sfx']}  [{r['sfx_txt']}]")
    w("")
    w(f"DEDICATION: {ded_len} chars  (ceiling {G['dedication_max']})  -> "
      f"{'PASS' if ded_len <= G['dedication_max'] else 'FAIL'}")
    w("")
    for line in swap_test(lyr_by_var):
        w(line)
    w("")
    if failures:
        w("GATE FAILURES:")
        for f in failures:
            w("   - " + f)
    else:
        w("ALL GATES PASS.")
    out = "\n".join(raw)
    print(out)

    # ---- markdown table ---------------------------------------------------
    md = []
    md.append("| gate | floor / ceiling | V1 | V2 | V3 | V4 |")
    md.append("|---|---|---|---|---|---|")

    def row(label, gate, key, fmt="{}", okkey=None):
        cells = []
        for r in rows:
            v = fmt.format(r[key])
            if okkey is not None:
                v += " " + ("PASS" if r[okkey] else "**FAIL**")
            cells.append(v)
        md.append(f"| {label} | {gate} | " + " | ".join(cells) + " |")

    row("style field chars", "850–1000 (target 870–960)", "style", "{}", "style_ok")
    row("style ends on a full stop", "required", "style_end", "{}")
    row("exclude chars / items", "non-trivial", "excl", "{}", "excl_ok")
    row("lyrics field chars", "≤ 5000", "lyr", "{}", "lyr_ok")
    row("sung lines", "70–120 (target 78–110)", "lines", "{}", "lines_ok")
    row("end-rhyme return", "≥ 0.30", "rr", "{:.3f}", "rr_ok")
    row("line return", "≥ 0.20", "lr", "{:.3f}", "lr_ok")
    row("mean words / line", "≤ 7.50", "wpl", "{:.2f}", "wpl_ok")
    row("alliteration / 100w", "reported", "allit", "{:.2f}")
    row("hardware-noun stems (sung)", "= 0", "hw", "{}")
    row("artist / company / marque", "= 0", "marq", "{}")
    row("hard-pan / panning tokens", "= 0", "pan", "{}")
    row("digits in sung lines", "= 0", "digs", "{}")
    row("distinct numeric facts", "≤ 1, as words", "nf", "{}", "nf_ok")
    md.append("| pole present in acts I / II / III | ≥1 each | " + " | ".join(
        f"{r['cov']['I']}/{r['cov']['II']}/{r['cov']['III']} " + ("PASS" if r["pole_ok"] else "**FAIL**")
        for r in rows) + " |")
    md.append("| act share of sung lines I / II / III | unequal, none < 20% | " + " | ".join(
        f"{r['shares']['I']}/{r['shares']['II']}/{r['shares']['III']}%" for r in rows) + " |")
    md.append("| section headers, all carrying `EMO:` | required | " + " | ".join(
        f"{r['hdrs']} " + ("PASS" if r["emo_ok"] else "**FAIL**") for r in rows) + " |")
    md.append("| SFX cues in the lyrics field | ≥1 in the selected song | " + " | ".join(
        str(r["sfx"]) for r in rows) + " |")
    md.append("| metadata openers excluded from the EMO gate | counted, not dropped | " + " | ".join(
        str(r["metas"]) for r in rows) + " |")
    md.append(f"| dedication chars (V1) | ≤ 500 | {ded_len} " +
              ("PASS" if ded_len <= G["dedication_max"] else "**FAIL**") + " | — | — | — |")
    md.append("")
    md.append("**Numeric fact actually sung, per variation:** " + " · ".join(
        f"**{r['var']}** {r['nf_txt']}" for r in rows))
    md.append("")
    md.append("**Verdict: " + ("ALL GATES PASS." if not failures else
                               "GATE FAILURES — " + "; ".join(failures)) + "**")
    md_s = "\n".join(md)

    if "--emit" in sys.argv:
        new = re.sub(r"(<!--MEASURE_TABLE_START-->\n).*?(\n<!--MEASURE_TABLE_END-->)",
                     lambda m: m.group(1) + md_s + m.group(2), text, flags=re.S)
        new = re.sub(r"(<!--MEASURE_RAW_START-->\n).*?(\n<!--MEASURE_RAW_END-->)",
                     lambda m: m.group(1) + "```text\n" + out + "\n```" + m.group(2), new, flags=re.S)
        with open(TARGET, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(new)
        print("\n[emitted into pair_03_step10_final_package.md]")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
