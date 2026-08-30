#!/usr/bin/env python3
"""COORDINATOR-SIDE VERIFIER — 2026-08-28-three-act-mania.

Written by the COORDINATOR, not by any pair. The pairs measured themselves;
this is the separate hand. "The evidence is the artifact, never the agent."

Reuses the canonical soundcraft instrument (scripts/measure_soundcraft.py) rather
than re-deriving its definitions -- re-derivation is how the join breaks.

UNIT DECLARATION (name the unit before you measure):
  Every number below is PER VARIATION (one Suno package = one song), never pooled
  across a pair. A pooled figure and a per-song figure are different units and a
  pair's pooled cross-check may not be compared against these.

TWO-TIER HARDWARE SCAN. The ICB's R5 list contains words that are unambiguous
machine nouns AND words that are ordinary English in another sense. Merging them
would report a defect count that measures something other than its predicate --
the exact failure class this run keeps finding. So:
  HARD   : unambiguous. Any hit is a FAIL.
  REVIEW : ordinary-English homographs. Any hit is surfaced for a human read,
           never auto-failed. (`boot` -> boots are footwear; `train` -> a train of
           thought; `weight` -> a load; `prompt` -> a prompt reply.)
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))
sys.path.insert(0, r"E:\git\private-claw\scripts")
import measure_soundcraft as ms  # noqa: E402

# --- gates, from vault/gates.yaml + the run ICB -----------------------------
STYLE_MIN, STYLE_MAX = 850, 1000
STYLE_HUG = 985
LYRIC_MAX = 5000
LINES_MIN, LINES_MAX = 70, 120
RHYME_FLOOR = 0.30
LINERET_FLOOR = 0.20
WPL_CEIL = 7.5

HARD = [r"circuit", r"silicon", r"binary", r"algorithm", r"dataset",
        r"robot", r"cyborg", r"android", r"software", r"hardware",
        r"motherboard", r"processor", r"cpu", r"gpu", r"neural net",
        r"my maker", r"my creator", r"rebooted?", r"reboot",
        r"source code", r"lines? of code", r"machine"]
REVIEW = [r"\bcode[sd]?\b", r"\bwire[sd]?\b", r"\bserver[s]?\b", r"\bmodel[sl]?\w*\b",
          r"\bprompt\w*\b", r"\bweights\b", r"\btraining\b", r"\bprogram\w*\b",
          r"\bboot(ed|s)?\b", r"\bmaker[s]?\b", r"\bcreator[s]?\b"]
MARQUE = [r"\bspotify\b", r"\bsuno\b", r"\bapple\b", r"\bgoogle\b", r"\bamazon\b",
          r"\bnike\b", r"\bcoca[- ]?cola\b", r"\bvauxhall\b", r"\bford\b",
          r"\btesla\b", r"\bmeta\b", r"\bopenai\b", r"\banthropic\b", r"\bchatgpt\b"]
PAN = [r"hard[- ]?pan", r"panned hard", r"extreme stereo", r"full left", r"full right"]


def fields(path, header_rx):
    """Every fenced block following a header matching header_rx."""
    t = open(path, encoding="utf-8", errors="ignore").read()
    out = []
    for m in re.finditer(header_rx, t, re.M):
        rest = t[m.end():]
        nm = re.search(r"^#{1,3} ", rest, re.M)
        region = rest[:nm.start()] if nm else rest
        fm = re.search(r"```[a-zA-Z]*\n(.*?)\n```", region, re.S)
        if fm:
            out.append(fm.group(1).strip())
    return out


NEG = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\bavoid\b|\bzero\b|\bexclude[sd]?\b)"
                 r"[^,;.]{0,40}$")


STYLE_RX = r"^#{2,4} .*(?:STYLE FIELD|STYLE PROMPT|MUSIC PROMPT)[^\n]*$"
LYRIC_RX = r"^#{2,4} .*LYRIC[^\n]*$"
EXCL_RX = r"^#{2,4} .*EXCLUDE[^\n]*$"


def lyrics_of(path):
    """THE shared lyric extractor. Import this; never re-derive it.

    ⛔ measure_soundcraft.lyric_blocks() matches '^## .*LYRIC' -- exactly two
    hashes. P04 writes '### ...LYRIC', so that reader returns [] for it, and a
    script that then reports a hard-coded expected count prints a clean-looking
    total while silently measuring a smaller set. That happened twice on
    2026-08-28 in this run's own coordinator scripts, AFTER the defect was
    reported. One extractor, imported everywhere, and every count DERIVED.
    """
    out = fields(path, LYRIC_RX)
    return out if out else ms.lyric_blocks(path)


def scan(text, pats, polarity_aware=False):
    """Find pattern hits. With polarity_aware, a hit under a NEGATION is NOT a hit.

    ⛔ MEASURED DEFECT, 2026-08-28, THREE independent instances in one run:
    a banned-token scan that sweeps the EXCLUDE field fails every song *for
    containing the fix*. An EXCLUDE reading "no hard-panned effects" contains
    the token while doing the opposite of the thing the gate forbids. P01 hit
    this (`app` prefix-matched `approve`), P05 hit it (moon-kitsch swept
    EXCLUDE), and this reader hit it (pan swept EXCLUDE). A gate that cannot
    tell a ban from a use is not measuring its predicate.
    """
    hits = []
    low = text.lower()
    for p in pats:
        for m in re.finditer(p, low):
            if polarity_aware and NEG.search(low[max(0, m.start() - 40):m.start()]):
                continue
            hits.append(m.group(0))
    return sorted(set(hits))


def verify(path):
    # Header names drift between pairs (measured 2026-08-28, five files, five
    # spellings): "## 1. STYLE FIELD" / "## 1. MUSIC PROMPT" / "### V1 STYLE
    # FIELD". A reader that hard-codes one spelling reports style_chars=0 and
    # FAILS a clean artifact -- caught here before any pair was told to repair.
    styles = fields(path, STYLE_RX)
    lyrics = lyrics_of(path)
    excludes = fields(path, EXCL_RX)
    # ⛔ FOUND BY P04, 2026-08-28: a file this reader cannot parse yielded
    # "0 songs · 0 PASS · 0 FAIL" -- which PRINTS LIKE A CLEAN RESULT. Silence
    # and success were the same output. Zero songs is now a loud error.
    if not lyrics:
        print(f"!! ZERO SONGS PARSED from {os.path.basename(path)} -- "
              f"THIS IS A READER FAILURE, NOT A PASS. Do not read it as clean.")
    if not (len(styles) == len(lyrics) == len(excludes)):
        print(f"!! FIELD-COUNT MISMATCH in {os.path.basename(path)}: "
              f"style={len(styles)} lyric={len(lyrics)} exclude={len(excludes)} "
              f"-- the reader is suspect, not necessarily the artifact")
    rows = []
    for i, lyr in enumerate(lyrics):
        sungl = ms.sung(lyr)
        prof = ms.profile(sungl)
        sty = styles[i] if i < len(styles) else ""
        exc = excludes[i] if i < len(excludes) else ""
        sung_text = "\n".join(sungl)
        digits = sorted(set(re.findall(r"\d", sung_text)))
        rows.append(dict(
            v=f"V{i + 1}",
            style_chars=len(sty),
            style_ends=bool(re.search(r"[.!?]$", sty.strip())),
            exclude_chars=len(exc),
            lyric_chars=len(lyr),
            lines=prof["lines"],
            rhyme=prof["end_rhyme"],
            lineret=prof["line_return"],
            wpl=prof["words_per_line"],
            hard=scan(sung_text, HARD),
            review=scan(sung_text, REVIEW),
            marque=scan(sung_text + "\n" + sty, MARQUE, polarity_aware=True),
            pan=scan(sty, PAN, polarity_aware=True),
            pan_in_exclude=scan(exc, PAN),
            digits=digits,
        ))
    return rows


def fails(r):
    f = []
    if not (STYLE_MIN <= r["style_chars"] <= STYLE_MAX):
        f.append(f"style_chars={r['style_chars']}")
    if not r["style_ends"]:
        f.append("style_not_terminal")
    if r["exclude_chars"] < 40:
        f.append(f"exclude_thin={r['exclude_chars']}")
    if r["lyric_chars"] > LYRIC_MAX:
        f.append(f"lyric_chars={r['lyric_chars']}")
    if not (LINES_MIN <= r["lines"] <= LINES_MAX):
        f.append(f"sung_lines={r['lines']}")
    if r["rhyme"] < RHYME_FLOOR:
        f.append(f"rhyme={r['rhyme']:.3f}")
    if r["lineret"] < LINERET_FLOOR:
        f.append(f"line_return={r['lineret']:.3f}")
    if r["wpl"] > WPL_CEIL:
        f.append(f"words_per_line={r['wpl']:.2f}")
    if r["hard"]:
        f.append(f"HARDWARE={r['hard']}")
    if r["marque"]:
        f.append(f"MARQUE={r['marque']}")
    if r["pan"]:
        f.append(f"PAN={r['pan']}")
    if r["digits"]:
        f.append(f"DIGITS_IN_SUNG={r['digits']}")
    return f


if __name__ == "__main__":
    paths = sys.argv[1:]
    hdr = (f"{'pair':6s} {'v':4s} {'sty':>5s} {'exc':>5s} {'lyr':>6s} {'ln':>4s} "
           f"{'rhy':>6s} {'lret':>6s} {'wpl':>5s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    total, bad = 0, 0
    reviews = []
    for p in paths:
        tag = re.search(r"pair_(\d+)", os.path.basename(p))
        tag = "P" + tag.group(1) if tag else os.path.basename(p)[:6]
        for r in verify(p):
            total += 1
            f = fails(r)
            if f:
                bad += 1
            hug = "~" if r["style_chars"] >= STYLE_HUG else " "
            print(f"{tag:6s} {r['v']:4s} {r['style_chars']:5d}{hug}{r['exclude_chars']:4d} "
                  f"{r['lyric_chars']:6d} {r['lines']:4d} {r['rhyme']:6.3f} "
                  f"{r['lineret']:6.3f} {r['wpl']:5.2f}  "
                  f"{'PASS' if not f else 'FAIL: ' + '; '.join(f)}")
            if r["review"]:
                reviews.append(f"  {tag} {r['v']}: {r['review']}")
    print("-" * len(hdr))
    print(f"{total} songs measured  ·  {total - bad} PASS  ·  {bad} FAIL")
    print("\nREVIEW-TIER hits (ordinary-English homographs — human read, never auto-fail):")
    print("\n".join(reviews) if reviews else "  none")
