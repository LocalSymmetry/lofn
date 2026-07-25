#!/usr/bin/env python3
"""Measure SOUND-CRAFT density in lyrics: the audio-written joys, not the semantics.

The Scientist's charge: the last three runs avoid rhyme but replace it with
nothing — no alliteration, no consonance, no play — so the songs read as
lecture. This measures that claim instead of agreeing with it.

Control group = the proven winners embedded in LOFN-PRIME's own archive
(judge-side reading; this is diagnosis, not generation).
"""
import re, sys, glob, os
from collections import Counter

VOWELS = "aeiou"

def lyric_blocks(path):
    """Pull Suno lyrics fields out of a step10 / enhanced package."""
    t = open(path, encoding="utf-8", errors="ignore").read()
    out = []
    for m in re.finditer(r"^## .*LYRIC[^\n]*$", t, re.M):
        rest = t[m.end():]
        nm = re.search(r"^#{1,3} ", rest, re.M)
        region = rest[:nm.start()] if nm else rest
        fm = re.search(r"```[a-zA-Z]*\n(.*?)\n```", region, re.S)
        out.append(fm.group(1) if fm else region)
    return out

def sung(block):
    lines = []
    for ln in block.splitlines():
        s = ln.strip()
        if not s: continue
        if s.startswith("[") and s.endswith("]"): continue
        if s.startswith("*") and s.endswith("*"): continue
        s = re.sub(r"\*[^*]*\*", "", s)          # inline sfx
        s = re.sub(r"^\(|\)$", "", s.strip())
        if s.strip(): lines.append(s.strip())
    return lines

def words(line):
    return [w for w in re.findall(r"[a-z']+", line.lower()) if w]

def syl(w):
    g = re.sub(r"[^aeiou]+", " ", w).split()
    n = len(g)
    if w.endswith("e") and n > 1: n -= 1
    return max(1, n)

def rime(w):
    """Crude rhyme key: from last vowel cluster to end."""
    m = list(re.finditer(r"[aeiou]+", w))
    if not m: return w[-2:]
    return w[m[-1].start():]

def alliteration(lines):
    """Adjacent-or-near word pairs sharing an initial consonant, per line."""
    hits = tot = 0
    for ln in lines:
        ws = [w for w in words(ln) if w and w[0] not in VOWELS]
        for i in range(len(ws)):
            for j in range(i+1, min(i+4, len(ws))):   # within a 3-word window
                if ws[i][0] == ws[j][0]:
                    hits += 1; break
        tot += max(1, len(ws))
    return hits / tot if tot else 0

def consonance(lines):
    """Repeated non-initial consonant clusters within a line (density)."""
    sc = 0
    for ln in lines:
        cs = re.findall(r"[bcdfgklmnprstvz]", ln.lower())
        if not cs: continue
        c = Counter(cs); top = sum(v for v in c.values() if v >= 3)
        sc += top / max(1, len(cs))
    return sc / max(1, len(lines))

def assonance(lines):
    """Repeated vowel nuclei within a line."""
    sc = 0
    for ln in lines:
        vs = re.findall(r"[aeiou]+", ln.lower())
        if not vs: continue
        c = Counter(vs); rep = sum(v for v in c.values() if v >= 3)
        sc += rep / max(1, len(vs))
    return sc / max(1, len(lines))

def end_rhyme(lines):
    """Fraction of lines whose rime matches another line within a 4-line window."""
    keys = [rime(words(l)[-1]) if words(l) else "" for l in lines]
    hit = 0
    for i, k in enumerate(keys):
        if not k: continue
        for j in range(max(0, i-4), min(len(keys), i+5)):
            if j != i and keys[j] == k:
                hit += 1; break
    return hit / max(1, len(keys))

def internal_rhyme(lines):
    sc = 0
    for ln in lines:
        ws = words(ln)
        if len(ws) < 3: continue
        rs = [rime(w) for w in ws if syl(w) >= 1]
        c = Counter(rs)
        sc += 1 if any(v >= 2 for v in c.values()) else 0
    return sc / max(1, len(lines))

def monosyllable(lines):
    ws = [w for l in lines for w in words(l)]
    return sum(1 for w in ws if syl(w) == 1) / max(1, len(ws))

def analyse(name, blocks):
    all_lines = [l for b in blocks for l in sung(b)]
    if not all_lines: return None
    return dict(
        name=name, songs=len(blocks), lines=len(all_lines),
        end_rhyme=end_rhyme(all_lines),
        internal=internal_rhyme(all_lines),
        allit=alliteration(all_lines),
        cons=consonance(all_lines),
        asson=assonance(all_lines),
        mono=monosyllable(all_lines),
    )

def show(rows):
    print(f"{'corpus':34s} {'songs':>5s} {'lines':>6s} {'endRhy':>7s} {'intRhy':>7s} {'allit':>7s} {'cons':>6s} {'asson':>6s} {'mono':>6s} {'PLAY':>6s}")
    print("-"*104)
    for r in rows:
        if not r: continue
        play = (r['end_rhyme'] + r['internal'] + r['allit'] + r['asson'])
        print(f"{r['name'][:34]:34s} {r['songs']:5d} {r['lines']:6d} "
              f"{r['end_rhyme']:7.3f} {r['internal']:7.3f} {r['allit']:7.3f} "
              f"{r['cons']:6.3f} {r['asson']:6.3f} {r['mono']:6.3f} {play:6.3f}")


# ---------------------------------------------------------------------------
# THE CANONICAL RETURN METRICS
#
# These four functions produce the exact table quoted in the L21 doctrine and
# in lofn-music/SKILL.md § THE RETURN. They were originally computed inline in
# an ad-hoc script, which meant the published module could NOT reproduce the
# published numbers — a reproducibility defect in a file whose whole purpose is
# "check us rather than trust us". Caught by a repair agent that had to rebuild
# them from scratch and calibrate against the quoted figures. Now canonical.
# Thresholds live in vault/gates.yaml (rhyme_return_floor, line_return_floor,
# mean_words_per_line_ceiling, alliteration_per_100w_floor).
# ---------------------------------------------------------------------------

RHYME_WINDOW = 4          # a rime counts as returning if it recurs within N lines

def strict_end_rhyme(lines, window=RHYME_WINDOW):
    """Fraction of lines whose final-word ending recurs within +/-window lines.

    Deliberately crude (last 3 characters), and crude in a *stable* way — the
    point is comparability between corpora, not phonetic truth.
    """
    keys = [words(l)[-1][-3:] if words(l) else "" for l in lines]
    hit = 0
    for i, k in enumerate(keys):
        if not k:
            continue
        lo, hi = max(0, i - window), min(len(keys), i + window + 1)
        if any(keys[j] == k for j in range(lo, hi) if j != i):
            hit += 1
    return hit / max(1, len(keys))

def line_return(lines):
    """Fraction of lines that are exact repeats of another line.

    Choruses COUNT. That is the entire point: this is the measure the harness
    was missing while it optimised similarity ceilings downward.
    """
    c = Counter(l.lower().strip(" .,") for l in lines)
    return sum(v for v in c.values() if v > 1) / max(1, len(lines))

def words_per_line(lines):
    return len([w for l in lines for w in words(l)]) / max(1, len(lines))

def allit_per_100w(lines):
    """Alliteration hits per 100 words — length-independent, unlike per-line."""
    hits = 0
    for l in lines:
        cw = [w for w in words(l) if w and w[0] not in VOWELS]
        for i in range(len(cw)):
            if any(cw[i][0] == cw[j][0] for j in range(i + 1, min(i + 4, len(cw)))):
                hits += 1
    total = len([w for l in lines for w in words(l)])
    return 100 * hits / max(1, total)

def profile(lines):
    """The four numbers the doctrine is stated in."""
    return dict(
        end_rhyme=strict_end_rhyme(lines),
        line_return=line_return(lines),
        words_per_line=words_per_line(lines),
        allit_per_100w=allit_per_100w(lines),
        lines=len(lines),
    )

def profile_file(path):
    return profile([l for b in lyric_blocks(path) for l in sung(b)])

if __name__ == "__main__" and len(sys.argv) > 1:
    print(f"{'file':46s} {'endRhy':>7s} {'lineRet':>8s} {'w/line':>7s} {'allit':>7s} {'lines':>6s}")
    for f in sys.argv[1:]:
        p = profile_file(f)
        print(f"{os.path.basename(f)[:46]:46s} {p['end_rhyme']:7.3f} {p['line_return']:8.3f} "
              f"{p['words_per_line']:7.2f} {p['allit_per_100w']:7.2f} {p['lines']:6d}")
