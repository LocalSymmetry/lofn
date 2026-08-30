#!/usr/bin/env python3
"""Cross-pair distinctiveness for a Lofn music run — coordinator-side, single-threaded.

Ceilings from vault/gates.yaml:
    portfolio_max_lyric_similarity  0.42
    portfolio_max_prompt_similarity 0.58
    portfolio_max_ngram_jaccard     0.18

⛔ THE MANDATED SHARED ELEMENTS ARE EXCLUDED BEFORE MEASURING, AND SAID OUT LOUD.
This run's suite binders — the anchor phrase and the folding chair — are shared by
DESIGN (they are what make three clips one work). A similarity validator cannot tell
a deliberate binder from pair bleed, so it would route the run's central device into
repair. They are stripped, the strip is reported, and everything else must still
diverge on its own. That is the difference between a waiver and a blind spot.

L25: print what was extracted before trusting what was concluded; assert cardinality.

Usage: python3 scripts/check_suite_distinctiveness.py output/<run-dir>
"""
import io
import os
import re
import sys

# ⛔ Force UTF-8 on stdout. Without this the script dies with UnicodeEncodeError
# partway through printing on a default Windows console (cp1252) — AFTER emitting a
# plausible-looking partial report. A gate that prints half a clean result and then
# crashes is worse than one that never runs: the operator sees ticks and stops reading.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from difflib import SequenceMatcher
from itertools import combinations

CEILINGS = {"lyric": 0.42, "prompt": 0.58, "ngram_jaccard": 0.18}
SETTLE = 90

# Shared by mandate (metaprompt "THE THREE BINDERS"). Stripped before measuring.
# ⛔ RUN-SCOPED, EMPTY BY DEFAULT. These were hard-coded to the `magnum-opus-suite`
# binders (its anchor phrase and folding chair) and would silently carry into every
# later run — the same defect ANCHOR_STEMS had in restat_pair_packages.py, which
# produced 8 false hard-gate failures before two agents caught it. A strip list from
# another run is either a no-op that lies in the report, or a real strip of content
# this run never mandated. Populate it per run or leave it empty.
MANDATED_SHARED = []


def content_after(text, heading):
    i = text.find(heading)
    if i < 0:
        return None
    start = i + len(heading)
    nxt = re.search(r"^#{1,3}\s", text[start:], flags=re.M)
    sl = text[start:start + nxt.start()] if nxt else text[start:]
    fence = re.search(r"```[a-zA-Z]*\n(.*?)\n```", sl, re.S)
    if fence:
        return fence.group(1)
    lines = [ln for ln in sl.strip().split("\n")
             if ln.strip() and not re.match(r"^\*{1,2}measured", ln.strip(), re.I)
             and not re.match(r"^\*[^*]+\*$", ln.strip())]
    return "\n".join(lines).strip() or None


def split_variations(text):
    anchors = [m.start() for m in
               re.finditer(r"^## 1\. MUSIC PROMPT[ \t]*$", text, flags=re.M)]
    h1 = [m.start() for m in re.finditer(r"^#\s+\S", text, flags=re.M)]
    starts = sorted({max([h for h in h1 if h < a], default=a) for a in anchors})
    if len(starts) != 4:
        return []
    return [text[s:(starts[i + 1] if i + 1 < len(starts) else len(text))]
            for i, s in enumerate(starts)]


def normalize(s):
    for pat in MANDATED_SHARED:
        s = re.sub(pat, " ", s, flags=re.I)
    s = re.sub(r"\[[^\]]*\]", " ", s)      # EMO headers / tags are contract, not content
    s = re.sub(r"\*[^*]*\*", " ", s)       # SFX cues
    return re.sub(r"\s+", " ", s.lower()).strip()


def jaccard_ngrams(a, b, n=5):
    def grams(s):
        w = s.split()
        return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / len(ga | gb) if (ga | gb) else 0.0


def main():
    run_dir = sys.argv[1]
    songs = {}
    skipped = []
    for p in range(1, 7):
        path = os.path.join(run_dir, f"pair_{p:02d}_step10_revision_synthesis.md")
        if not os.path.exists(path):
            skipped.append(f"P{p:02d} (not on disk)")
            continue
        import time
        if time.time() - os.path.getmtime(path) < SETTLE:
            skipped.append(f"P{p:02d} (in flight)")
            continue
        text = io.open(path, encoding="utf-8").read()
        blocks = split_variations(text)
        if len(blocks) != 4:
            skipped.append(f"P{p:02d} (extracted {len(blocks)}/4 — ERROR, not a score)")
            continue
        for i, b in enumerate(blocks, 1):
            ly, mp = content_after(b, "## 2. LYRICS"), content_after(b, "## 1. MUSIC PROMPT")
            if not ly or not mp:
                skipped.append(f"P{p:02d}V{i} (missing payload)")
                continue
            songs[f"P{p:02d}V{i}"] = (normalize(ly), normalize(mp))

    print(f"EXTRACTED {len(songs)} songs: {', '.join(sorted(songs))}")
    if skipped:
        print(f"NOT MEASURED: {'; '.join(skipped)}")
    strip_note = ", ".join(MANDATED_SHARED) if MANDATED_SHARED else "NOTHING - this run mandates no shared element, so every song is measured whole"
    print("stripped before measuring (shared by mandate): " + strip_note)
    print("")
    if len(songs) < 2:
        print("fewer than 2 settled songs — nothing to compare.")
        return 0

    worst = {"lyric": ("", 0.0), "prompt": ("", 0.0), "ngram_jaccard": ("", 0.0)}
    breaches = []
    # Cross-PAIR comparisons only: within-pair variations are 4 takes on one slot and
    # are expected to be closer to each other than to another pair's work.
    for a, b in combinations(sorted(songs), 2):
        if a[:3] == b[:3]:
            continue
        ly = SequenceMatcher(None, songs[a][0], songs[b][0], autojunk=False).ratio()
        mp = SequenceMatcher(None, songs[a][1], songs[b][1], autojunk=False).ratio()
        ng = jaccard_ngrams(songs[a][0], songs[b][0])
        for k, v in (("lyric", ly), ("prompt", mp), ("ngram_jaccard", ng)):
            if v > worst[k][1]:
                worst[k] = (f"{a}↔{b}", v)
            if v > CEILINGS[k]:
                breaches.append((k, f"{a}↔{b}", round(v, 3), CEILINGS[k]))

    n = sum(1 for a, b in combinations(sorted(songs), 2) if a[:3] != b[:3])
    print(f"cross-pair comparisons: {n}")
    for k in ("lyric", "prompt", "ngram_jaccard"):
        pair, v = worst[k]
        print(f"  worst {k:<15} {v:.3f}  (ceiling {CEILINGS[k]})  {pair}")
    print()
    if breaches:
        print(f"⚠️ {len(breaches)} BREACH(ES) — repair trigger, not a run kill:")
        for k, pair, v, c in breaches:
            print(f"   {k} {pair} = {v} > {c}")
    else:
        print("✅ 0 breaches across all three ceilings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
