#!/usr/bin/env python3
"""p03_measure.py -- PAIR 03's own measurement pass over its fenced Suno fields.

Reads the SAME unit the coordinator reads: the fenced code blocks under a heading
naming one of the three pasted fields. The sung() definition is imported from
scripts/measure_soundcraft.py -- THE canonical definition, never re-derived.

    python p03_measure.py <package.md>

Read-only. Writes nothing, renders nothing, spends nothing.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"))
from measure_soundcraft import sung as canonical_sung  # THE definition

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FENCE = chr(96) * 3
FIELD_KINDS = (("music", "MUSIC PROMPT"), ("exclude", "EXCLUDE"), ("lyrics", "LYRIC"))
NOT_A_FIELD = ("DEDICATION", "CREDIT", "LINEAGE", "VERIFICATION", "REPAIR LOG",
               "PROVENANCE", "DIGEST", "CHECKLIST", "DEVIATION")

# The refused element, PAIR 03's own choice: no second person, no third person,
# no indefinite person-word anywhere in any sung line of any of the four songs.
BANNED_PERSON = ("you", "your", "yours", "they", "them", "their", "theirs",
                 "he", "him", "his", "she", "her", "hers", "we", "us", "our",
                 "ours", "nobody", "anyone", "someone", "everyone", "somebody",
                 "anybody", "everybody", "people", "crowd", "folks")

# R2: no emotion word survives into any sung line.
EMOTION = ("angry", "anger", "furious", "fury", "rage", "proud", "pride",
           "strong", "strength", "loud", "brave", "free", "freedom", "joy",
           "sad", "sadness", "happy", "grief", "fear", "afraid", "love",
           "hate", "shame", "guilt", "hope", "hopeful", "calm", "peace",
           "lonely", "lonely", "brave", "bold", "fierce", "powerful", "power")

# R5 amended: the cost is DISPLACEMENT, not STRAIN. Every strain word re-imports
# the audience the martyr needs.
STRAIN = ("ache", "aching", "burn", "burning", "strain", "sore", "hurt",
          "heavy", "heavier", "tired", "weary", "worn", "exhausted", "drained",
          "endure", "endured", "bear", "burden", "crush", "crushed", "break",
          "breaking", "broken", "suffer", "struggle", "sacrifice", "toil")

# Armour vocabulary (forbidden-default #10).
ARMOUR = ("warrior", "fighter", "survivor", "unbreakable", "stronger",
          "titanium", "phoenix", "roar", "scars", "armour", "armor", "storm")

NUMERALS = ("zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "twenty", "thirty",
            "forty", "fifty", "hundred", "thousand", "first", "second",
            "third", "fourth", "fifth", "once", "twice", "half", "quarter",
            "third", "double", "triple", "both", "dozen")

BODY = ("hand", "hands", "arm", "arms", "shoulder", "shoulders", "back",
        "knee", "knees", "hip", "hips", "chest", "throat", "mouth", "teeth",
        "skin", "bone", "bones", "blood", "breath", "lung", "lungs", "spine",
        "wrist", "finger", "fingers", "palm", "palms", "knuckle", "knuckles",
        "heel", "sole", "foot", "feet", "eye", "eyes", "face", "body")


def pasted_fields(text):
    heads = [(m.start(), m.group(0).upper())
             for m in re.finditer("(?m)^#{1,6}[^" + chr(10) + "]*$", text)]
    out = []
    for m in re.finditer(re.escape(FENCE) + "[a-z]*" + chr(10) + "(.*?)" + re.escape(FENCE),
                         text, re.S):
        prior = [h for h in heads if h[0] < m.start()]
        if not prior:
            continue
        head = prior[-1][1]
        if any(bad in head for bad in NOT_A_FIELD):
            continue
        for kind, needle in FIELD_KINDS:
            if needle in head:
                out.append((kind, head.strip(), m.group(1).rstrip(chr(10))))
                break
    return out


def wordhits(hay, tokens):
    low = hay.lower()
    found = {}
    for t in tokens:
        pat = "(?<![a-z0-9'])" + re.escape(t) + "(?![a-z0-9'])"
        n = len(re.findall(pat, low))
        if n:
            found[t] = n
    return found


def main():
    if len(sys.argv) != 2:
        print("usage: p03_measure.py <package.md>")
        return 2
    text = open(sys.argv[1], encoding="utf-8").read()
    fields = pasted_fields(text)
    if not fields:
        print("NO PASTED FIELDS FOUND -- reported, not passed.")
        return 2

    allsung = []
    print("kind     chars  head")
    print("-" * 78)
    for kind, head, body in fields:
        print("%-8s %6d  %s" % (kind, len(body), head[:52]))
    print()

    for kind, head, body in fields:
        if kind != "lyrics":
            continue
        lines = canonical_sung(body)
        allsung.append((head, lines))
        words = re.findall(r"[a-z']+", "\n".join(lines).lower())
        nw = len(words)
        fp = sum(1 for w in words if w in ("i", "my", "mine", "me"))
        bod = sum(1 for w in words if w in BODY)
        print("=" * 78)
        print(head[:70])
        print("  sung_lines      : %d" % len(lines))
        print("  lyrics_chars    : %d" % len(body))
        print("  sung_words      : %d" % nw)
        print("  first_person    : %d  (%.1f%%)" % (fp, 100.0 * fp / max(nw, 1)))
        print("  body_words      : %d  (%.1f%%)" % (bod, 100.0 * bod / max(nw, 1)))
        joined = "\n".join(lines)
        for label, toks in (("PERSON-WORD (refused elem)", BANNED_PERSON),
                            ("EMOTION (R2)", EMOTION),
                            ("STRAIN (R5)", STRAIN),
                            ("ARMOUR", ARMOUR),
                            ("NUMERAL (one-fact)", NUMERALS)):
            h = wordhits(joined, toks)
            print("  %-26s: %s" % (label, h if h else "0"))

    # cross-variation shared sung lines
    print("=" * 78)
    print("CROSS-VARIATION SHARED SUNG LINES")
    norm = [set(x.strip().lower().rstrip(".,") for x in ls) for _, ls in allsung]
    tot = 0
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            common = norm[i] & norm[j]
            tot += len(common)
            print("  V%d vs V%d: %d shared  %s" % (i + 1, j + 1, len(common),
                                                   sorted(common)[:4]))
    print("  total shared pairs: %d" % tot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
