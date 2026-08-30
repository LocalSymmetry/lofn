#!/usr/bin/env python3
"""PORTFOLIO CROSS-SIMILARITY — coordinator-side, 2026-08-28-three-act-mania.

Gates (vault/gates.yaml):
  portfolio_max_lyric_similarity  0.42   pairwise lyric token Jaccard
  portfolio_max_prompt_similarity 0.58   pairwise style-field token Jaccard
  portfolio_max_ngram_jaccard     0.18   pairwise lyric 5-gram Jaccard

UNIT DECLARATION: every figure is BETWEEN two songs (one variation vs one
variation). Within-song chorus/refrain repetition is a different measure and is
NOT what this reports -- the pole is SUPPOSED to repeat inside a song.

⭐ WITHIN-PAIR vs ACROSS-PAIR are reported separately and judged differently.
Four variations of one concept are expected to be closer to each other than to
another pair's work; the doctrine's portfolio ceilings are about the SHIPPED SET,
so the across-pair number is the one that gates. Within-pair is surfaced as
information for selection (a pair whose four are near-identical offers one song,
not four).
"""
import os
import re
import sys
from itertools import combinations

sys.path.insert(0, r"E:\git\private-claw\scripts")
import measure_soundcraft as ms  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coordinator_verify import lyrics_of, STYLE_RX  # noqa: E402

LYR_CEIL, PROMPT_CEIL, NGRAM_CEIL = 0.42, 0.58, 0.18


def toks(t):
    return set(re.findall(r"[a-z']+", t.lower()))


def ngrams(t, n=5):
    w = re.findall(r"[a-z']+", t.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def styles(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    out = []
    for m in re.finditer(r"^#{2,4} .*(?:STYLE FIELD|STYLE PROMPT|MUSIC PROMPT)[^\n]*$", t, re.M):
        rest = t[m.end():]
        nm = re.search(r"^#{1,3} ", rest, re.M)
        region = rest[:nm.start()] if nm else rest
        fm = re.search(r"```[a-zA-Z]*\n(.*?)\n```", region, re.S)
        if fm:
            out.append(fm.group(1).strip())
    return out


songs = []
for p in sys.argv[1:]:
    tag = re.search(r"pair_(\d+)", os.path.basename(p))
    tag = "P" + tag.group(1) if tag else os.path.basename(p)[:4]
    ly = lyrics_of(p)
    st = styles(p)
    for i, b in enumerate(ly):
        text = "\n".join(ms.sung(b))
        songs.append(dict(id=f"{tag}V{i + 1}", pair=tag, lyr=toks(text),
                          ng=ngrams(text), sty=toks(st[i] if i < len(st) else "")))

print(f"{len(songs)} songs\n")
worst = {"lyr": (0, ""), "ng": (0, ""), "sty": (0, "")}
breaches = []
within = []
for a, b in combinations(songs, 2):
    L, N, S = jac(a["lyr"], b["lyr"]), jac(a["ng"], b["ng"]), jac(a["sty"], b["sty"])
    tag = f"{a['id']}~{b['id']}"
    same = a["pair"] == b["pair"]
    if same:
        within.append((L, N, S, tag))
        continue
    for k, v in (("lyr", L), ("ng", N), ("sty", S)):
        if v > worst[k][0]:
            worst[k] = (v, tag)
    if L > LYR_CEIL or N > NGRAM_CEIL or S > PROMPT_CEIL:
        breaches.append(f"  {tag}  lyric={L:.3f} ngram={N:.3f} style={S:.3f}")

print("ACROSS-PAIR (this is the gate):")
print(f"  max lyric  Jaccard {worst['lyr'][0]:.3f}  (ceil {LYR_CEIL})  {worst['lyr'][1]}")
print(f"  max 5-gram Jaccard {worst['ng'][0]:.3f}  (ceil {NGRAM_CEIL})  {worst['ng'][1]}")
print(f"  max style  Jaccard {worst['sty'][0]:.3f}  (ceil {PROMPT_CEIL})  {worst['sty'][1]}")
print(f"  BREACHES: {len(breaches)}")
for b in breaches:
    print(b)

print("\nWITHIN-PAIR (information for selection, NOT a gate):")
bypair = {}
for L, N, S, tag in within:
    p = tag.split("V")[0]
    bypair.setdefault(p, []).append(L)
for p in sorted(bypair):
    v = bypair[p]
    print(f"  {p}: mean lyric Jaccard across its own 4 = {sum(v) / len(v):.3f}  "
          f"(max {max(v):.3f})")
