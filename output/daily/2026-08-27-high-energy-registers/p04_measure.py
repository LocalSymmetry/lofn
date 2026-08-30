#!/usr/bin/env python3
"""p04_measure.py -- PAIR 04 scratch instrument. Reads the fenced pasted fields out of a
pair-04 artifact and reports the numbers the coordinator will re-measure. Read-only."""
import os, re, sys, itertools
sys.path.insert(0, os.path.join("scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from check_field_census import pasted_fields, load_lexicon, hits

path = sys.argv[1]
text = open(path, encoding="utf-8").read()
fields = pasted_fields(text)

# group into songs: music, exclude, lyrics repeating
songs, cur = [], {}
for kind, block in fields:
    if kind in cur:
        songs.append(cur); cur = {}
    cur[kind] = block
if cur:
    songs.append(cur)

TITLES = ["V1", "V2", "V3", "V4"]
NUMWORDS = ("one two three four five six seven eight nine ten eleven twelve thirteen "
            "fourteen fifteen twenty thirty forty fifty hundred thousand million "
            "first second third fourth fifth dozen").split()
EMO_BANNED = ("angry anger furious fury proud pride strong strength loud loudness brave "
              "bravery free freedom rage joy sad sadness happy fear afraid love hate "
              "lonely ashamed embarrassed").split()
GUITAR = ["distorted guitar", "power chord", "palm mute", "palm-mute", "riff", "chug",
          "downtuned", "pinch harmonic"]


def sung_lines(lyr):
    out = []
    for ln in lyr.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") or (s.startswith("*") and s.endswith("*")):
            continue
        out.append(s)
    return out


all_sung = {}
print("=" * 78)
for i, s in enumerate(songs):
    t = TITLES[i] if i < len(TITLES) else "V?"
    mp = s.get("music", "")
    ex = s.get("exclude", "")
    ly = s.get("lyrics", "")
    sl = sung_lines(ly)
    all_sung[t] = sl
    print(f"{t}  music={len(mp)}  exclude={len(ex)}  lyrics_field={len(ly)}  sung_lines={len(sl)}")
    print(f"    music ends: ...{mp[-40:]!r}")
    ok = []
    if not (850 <= len(mp) <= 1000): ok.append(f"MUSIC BAND FAIL {len(mp)}")
    if not (870 <= len(mp) <= 960): ok.append(f"music target miss {len(mp)}")
    if len(mp) >= 985: ok.append("music boundary hug")
    if not mp.rstrip().endswith(("." , "!", "?")): ok.append("music not terminal punctuation")
    if not (400 <= len(ex) <= 900): ok.append(f"EXCLUDE BAND FAIL {len(ex)}")
    if len(ly) >= 5000: ok.append(f"LYRICS FIELD OVER 5000 ({len(ly)})")
    if len(ly) > 4800: ok.append(f"lyrics over 4800 target ({len(ly)})")
    if not (70 <= len(sl) <= 120): ok.append(f"SUNG LINE BAND FAIL {len(sl)}")
    if len(sl) < 78: ok.append(f"sung lines under 78 target ({len(sl)})")
    if len(sl) <= 72: ok.append("sung floor hug")
    # numerals + number words in sung lines only
    joined = " ".join(sl).lower()
    digits = re.findall(r"\d", joined)
    if digits: ok.append(f"DIGITS IN SUNG LINES: {digits}")
    nums = [w for w in NUMWORDS if re.search(r"(?<![a-z])" + w + r"(?![a-z])", joined)]
    print(f"    number-words in sung lines: {nums}")
    emo = [w for w in EMO_BANNED if re.search(r"(?<![a-z])" + w + r"(?![a-z])", joined)]
    if emo: ok.append(f"R2 EMOTION WORDS IN SUNG LINES: {emo}")
    # guitar refusal, positive fields only (music prompt + lyrics)
    pos = (mp + " " + ly).lower()
    g = [t2 for t2 in GUITAR if t2 in pos]
    if g: ok.append(f"REFUSED ELEMENT BREACH (positive fields): {g}")
    # sfx present
    if not re.search(r"(?m)^\*[^*]+\*\s*$", ly): ok.append("no standalone *SFX*")
    if "[Theme:" not in ly: ok.append("no [Theme:]")
    if "[SONG FORM:" not in ly: ok.append("no [SONG FORM:]")
    # mean words/line
    wl = sum(len(l.split()) for l in sl) / max(1, len(sl))
    print(f"    mean words/line: {wl:.2f}  (ceiling 7.5)")
    if wl > 7.5: ok.append(f"words/line over ceiling {wl:.2f}")
    # repeated line ratio
    from collections import Counter
    c = Counter(sl)
    rep = sum(v - 1 for v in c.values() if v > 1) / max(1, len(sl))
    print(f"    repeated-line ratio: {rep:.3f}  (floor 0.20)")
    if rep < 0.20: ok.append(f"line_return below floor {rep:.3f}")
    print("    ->", "; ".join(ok) if ok else "clean")
    print("-" * 78)

# lexicon census
lex = load_lexicon(os.path.join("vault", "field_token_lexicon.yaml"))
print("LEXICON CENSUS (all fenced pasted fields):")
found = False
for kind, block in fields:
    low = block.lower()
    for group, toks in lex.items():
        if group == "exclude_permitted":
            continue
        for tok in toks:
            n = hits(low, tok)
            if n:
                if group == "companies" and kind == "exclude" and tok in lex.get("exclude_permitted", []):
                    continue
                print(f"   HIT {group}:{tok} x{n} in {kind}")
                found = True
if not found:
    print("   0 hits")
print("-" * 78)

# cross-song shared sung lines
print("CROSS-SONG SHARED SUNG LINES (P2):")
tot = sum(len(v) for v in all_sung.values())
shared = set()
for a, b in itertools.combinations(all_sung, 2):
    inter = set(all_sung[a]) & set(all_sung[b])
    if inter:
        print(f"   {a}&{b}: {sorted(inter)}")
    shared |= inter
n_shared = sum(1 for t in all_sung for l in all_sung[t] if l in shared)
print(f"   total sung lines across four: {tot}; lines that are shared with another song: {n_shared}"
      f" ({100.0*n_shared/max(1,tot):.2f}%)")
