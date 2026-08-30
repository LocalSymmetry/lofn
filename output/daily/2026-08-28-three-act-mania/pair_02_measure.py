#!/usr/bin/env python3
"""pair_02_measure.py -- PAIR 02's own instrument. Read-only.

Reads the fenced pasted Suno fields out of a P02 artifact and reports every number
the ICB's HOUSE NUMBERS table names, plus the R5 hardware-noun stem scan and the
one-number-per-song rule.

⭐ THE UNIT IS THE FENCED BLOCK, and the heading-matching / trailing-newline rules are
copied deliberately from scripts/check_field_census.py so this pair's self-report and
the coordinator's independent reader are measuring THE SAME STRING. A pair that
measures a different unit than its auditor has not measured anything.

⛔ Renders nothing, publishes nothing, spends nothing. Writes no file.

    python3 pair_02_measure.py <artifact.md> [--ded <file-with-dedication>]
"""
import os
import re
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FENCE = chr(96) * 3
NL = chr(10)

FIELD_KINDS = (("music", "MUSIC PROMPT"), ("exclude", "EXCLUDE"), ("lyrics", "LYRIC"))
NOT_A_FIELD = ("DEDICATION", "CREDIT", "LINEAGE", "VERIFICATION", "REPAIR LOG",
               "PROVENANCE", "DIGEST", "CHECKLIST", "DEVIATION", "MEASUREMENT")

# ---------------------------------------------------------------------------
# R5 FALSIFIER A -- the hardware nouns, STEM-scanned (a word-boundary scan cannot
# see an inflected form; 2026-08-27 F5). Each lemma matches lemma + {s,es,ed,ing,
# er,ers,ry,ies,al}. The ICB list, plus the obvious neighbours a costume-of-an-AI
# song reaches for. Deliberately WIDER than the ICB list: over-refusing costs a
# word, under-refusing costs the gate.
# ---------------------------------------------------------------------------
HW = ("circuit circuitry wire server silicon code binary algorithm dataset prompt "
      "weight train model machine robot maker creator program boot reboot chip "
      "processor cpu neural network digital electronic software hardware data "
      "input output token parameter compute render upload download screen keyboard "
      "sensor motor engine battery switch signal transmit terminal console memory "
      "database protocol interface simulation android automaton mainframe").split()
HW_SUFFIX = "(?:s|es|ed|ing|er|ers|ry|ies|al)?"

NUMWORDS = ("one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen twenty thirty forty fifty hundred "
            "thousand million billion first second third fourth fifth sixth "
            "seventh eighth ninth tenth dozen once twice thrice single double "
            "triple zero").split()


def load_lexicon(path):
    groups, cur = {}, None
    for line in open(path, encoding="utf-8"):
        s = line.rstrip()
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        if not s.startswith(" ") and s.endswith(":"):
            cur = s[:-1].strip()
            groups[cur] = []
        elif cur and s.lstrip().startswith("- "):
            groups[cur].append(s.lstrip()[2:].strip().lower())
    return {k: v for k, v in groups.items() if v}


def pasted_fields(text):
    heads = [(m.start(), m.group(0).upper())
             for m in re.finditer("(?m)^#{1,6}[^" + NL + "]*$", text)]
    out = []
    for m in re.finditer(re.escape(FENCE) + "[a-z]*" + NL + "(.*?)" + re.escape(FENCE),
                         text, re.S):
        prior = [h for h in heads if h[0] < m.start()]
        if not prior:
            continue
        head = prior[-1][1]
        if any(bad in head for bad in NOT_A_FIELD):
            continue
        for kind, needle in FIELD_KINDS:
            if needle in head:
                out.append((kind, m.group(1).rstrip(NL)))
                break
    return out


def sung(block):
    """Sung lines only: no bracketed headers, no standalone *stage directions*."""
    out = []
    for ln in block.split(NL):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            continue
        if s.startswith("*") and s.endswith("*"):
            continue
        out.append(s)
    return out


def words(line):
    return [w for w in re.findall(r"[a-z']+", line.lower()) if w]


def strict_end_rhyme(lines, window=4):
    """measure_soundcraft.strict_end_rhyme, reproduced exactly: last 3 chars of the
    final word, recurring within +/- window lines."""
    keys = [words(l)[-1][-3:] if words(l) else "" for l in lines]
    hit = 0
    for i, k in enumerate(keys):
        if not k:
            continue
        lo, hi = max(0, i - window), min(len(keys), i + window + 1)
        if any(keys[j] == k for j in range(lo, hi) if j != i):
            hit += 1
    return hit / max(1, len(keys))


def line_return_soundcraft(lines):
    """measure_soundcraft.line_return: ALL copies of any line that repeats."""
    c = Counter(l.lower().strip(" .,") for l in lines)
    return sum(v for v in c.values() if v > 1) / max(1, len(lines))


def line_return_strict(lines):
    """The harsher reading: only the SURPLUS copies count."""
    c = Counter(l.lower().strip(" .,") for l in lines)
    return sum(v - 1 for v in c.values() if v > 1) / max(1, len(lines))


def hw_hits(text):
    low = text.lower()
    found = []
    for lemma in HW:
        pat = "(?<![a-z])" + lemma + HW_SUFFIX + "(?![a-z])"
        n = re.findall(pat, low)
        if n:
            found.append((lemma, len(n), sorted(set(n))))
    return found


def numword_hits(lines):
    joined = " ".join(lines).lower()
    found = {}
    for w in NUMWORDS:
        n = len(re.findall("(?<![a-z0-9])" + w + "(?![a-z0-9])", joined))
        if n:
            found[w] = n
    return found


def sfx_lines(block):
    """Standalone stage-direction cues: a whole line wrapped in asterisks.

    ⛔ ADDED POST-QA, 2026-08-28. The ICB HOUSE NUMBERS table does not list this
    gate, so nothing in this pair measured it and V1 shipped with zero cues. A gate
    absent from the table is still a gate; the fix is a reader, not a resolution to
    remember. sung() already drops these lines, so a cue costs no sung line.
    """
    out = []
    for ln in block.split(NL):
        s = ln.strip()
        if len(s) > 2 and s.startswith("*") and s.endswith("*") and "*" not in s[1:-1]:
            out.append(s)
    return out


def sfx_token_lines(block):
    """The OTHER house definition of the same gate.

    scripts/check_generation.py:45 does exactly this and nothing else:
        sfx_cues = [l for l in lyric_lines if '[SFX:' in l or 'SFX:' in l]
    ...i.e. it wants the literal, case-sensitive token SFX: on a lyric line, and it
    is blind to a standalone *...* cue that lacks it. lofn-music/SKILL.md:107 wants
    "at least one standalone *SFX* cue" and is blind to the token.

    ⛔ ONE GATE, TWO DEFINITIONS, TWO FILES — a house defect, not a pair defect
    (declared to The Scientist in the run INDEX, 2026-08-28). Mirrored here rather
    than adjudicated: the cheap move is to satisfy both readers, and the only way to
    know that both are satisfied is to run both.
    """
    return [ln.strip() for ln in block.split(NL) if "SFX:" in ln]


def act_audit(block):
    heads = [l.strip() for l in block.split(NL)
             if l.strip().startswith("[") and l.strip().endswith("]")]
    section_heads = [h for h in heads if h.upper().startswith("[ACT")]
    no_emo = [h for h in section_heads if "EMO:" not in h]
    acts = set()
    for h in section_heads:
        m = re.match(r"\[ACT\s+([IVX]+)", h.upper())
        if m:
            acts.add(m.group(1))
    return section_heads, no_emo, sorted(acts)


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: pair_02_measure.py <artifact.md>")
        return 2
    path = args[0]
    text = open(path, encoding="utf-8").read()
    fields = pasted_fields(text)

    songs, cur = [], {}
    for kind, block in fields:
        if kind in cur:
            songs.append(cur)
            cur = {}
        cur[kind] = block
    if cur:
        songs.append(cur)

    if not songs:
        print("FAIL: no pasted fields parsed. A checker that measures nothing")
        print("      must not return success.")
        return 2

    lexpath = os.path.join("vault", "field_token_lexicon.yaml")
    lex = load_lexicon(lexpath) if os.path.exists(lexpath) else {}

    titles = ["V1", "V2", "V3", "V4"]
    fails = []
    rows = []
    all_sung = {}

    for i, s in enumerate(songs):
        t = titles[i] if i < len(titles) else "V?"
        mp, ex, ly = s.get("music", ""), s.get("exclude", ""), s.get("lyrics", "")
        sl = sung(ly)
        all_sung[t] = sl
        wl = sum(len(l.split()) for l in sl) / max(1, len(sl))
        er = strict_end_rhyme(sl)
        lr_sc = line_return_soundcraft(sl)
        lr_st = line_return_strict(sl)
        hw = hw_hits(" ".join(sl))
        hw_all = hw_hits(mp + " " + ex + " " + ly)
        nums = numword_hits(sl)
        digits = re.findall(r"\d", " ".join(sl))
        heads, no_emo, acts = act_audit(ly)
        sfx = sfx_lines(ly)
        sfx_tok = sfx_token_lines(ly)

        print("=" * 78)
        print(t)
        print("  style field chars   : %d   (band 850-1000, target 870-960)" % len(mp))
        print("  style ends          : ...%r" % mp[-46:])
        print("  exclude chars       : %d" % len(ex))
        print("  lyrics field chars  : %d   (ceiling 5000)" % len(ly))
        print("  sung lines          : %d   (band 70-120, target 78-110)" % len(sl))
        print("  mean words/line     : %.2f (ceiling 7.50)" % wl)
        print("  strict end rhyme    : %.3f (floor 0.30)" % er)
        print("  line return (sc)    : %.3f (floor 0.20)" % lr_sc)
        print("  line return (strict): %.3f (floor 0.20)" % lr_st)
        print("  hardware nouns SUNG : %s" % (hw if hw else "0"))
        print("  hardware nouns ALL  : %s" % (hw_all if hw_all else "0"))
        print("  number-words sung   : %s" % (nums if nums else "{}"))
        print("  digits sung         : %s" % (digits if digits else "0"))
        print("  ACT section headers : %d  acts seen: %s" % (len(heads), acts))
        print("  headers missing EMO : %s" % (no_emo if no_emo else "0"))
        print("  standalone SFX cues : %d  %s" % (len(sfx), sfx))
        print("  SFX: token cues     : %d  (check_generation.py form)" % len(sfx_tok))

        lexhits = []
        for kind, block in (("music", mp), ("exclude", ex), ("lyrics", ly)):
            low = block.lower()
            for group, toks in lex.items():
                if group == "exclude_permitted":
                    continue
                for tok in toks:
                    n = len(re.findall("(?<![a-z0-9])" + re.escape(tok) + "(?![a-z0-9])", low))
                    if n:
                        lexhits.append("%s:%s x%d in %s" % (group, tok, n, kind))
        print("  marque/artist/pan   : %s" % (lexhits if lexhits else "0"))

        if not (850 <= len(mp) <= 1000):
            fails.append("%s style field out of band: %d" % (t, len(mp)))
        elif not (870 <= len(mp) <= 960):
            fails.append("%s style field outside target: %d" % (t, len(mp)))
        if len(mp) >= 985:
            fails.append("%s style field boundary hug: %d" % (t, len(mp)))
        if not mp.rstrip().endswith((".", "!", "?")):
            fails.append("%s style field does not end as a complete sentence" % t)
        if len(ex) < 250:
            fails.append("%s exclude field trivial: %d chars" % (t, len(ex)))
        if len(ly) > 5000:
            fails.append("%s lyrics field over ceiling: %d" % (t, len(ly)))
        if not (70 <= len(sl) <= 120):
            fails.append("%s sung lines out of band: %d" % (t, len(sl)))
        elif not (78 <= len(sl) <= 110):
            fails.append("%s sung lines outside target: %d" % (t, len(sl)))
        if wl > 7.5:
            fails.append("%s mean words/line over ceiling: %.2f" % (t, wl))
        if er < 0.30:
            fails.append("%s rhyme return below floor: %.3f" % (t, er))
        if lr_st < 0.20:
            fails.append("%s strict line return below floor: %.3f" % (t, lr_st))
        if hw:
            fails.append("%s HARDWARE NOUN IN SUNG LINE: %s" % (t, hw))
        if hw_all:
            fails.append("%s hardware noun somewhere in a pasted field: %s" % (t, hw_all))
        if len(nums) > 1:
            fails.append("%s more than one number-word: %s" % (t, nums))
        if digits:
            fails.append("%s DIGITS IN SUNG LINES: %s" % (t, digits))
        if lexhits:
            fails.append("%s marque/artist/panning token: %s" % (t, lexhits))
        if no_emo:
            fails.append("%s section header without EMO: %s" % (t, no_emo))
        if len(acts) != 3:
            fails.append("%s does not carry exactly three ACT headers: %s" % (t, acts))
        if not sfx:
            fails.append("%s NO STANDALONE SFX CUE IN THE LYRICS FIELD" % t)
        if not sfx_tok:
            fails.append("%s NO LITERAL 'SFX:' TOKEN (check_generation.py form)" % t)

        rows.append((t, len(mp), len(ex), len(ly), len(sl), wl, er, lr_sc, lr_st,
                     len(hw), len(nums)))

    print("=" * 78)
    print("CROSS-SONG EXACT SHARED SUNG LINES (divergence check):")
    keys = list(all_sung)
    shared_any = False
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            sa = set(l.lower().strip(" .,!?") for l in all_sung[keys[a]])
            sb = set(l.lower().strip(" .,!?") for l in all_sung[keys[b]])
            common = sorted(sa & sb)
            if common:
                shared_any = True
                print("   %s / %s : %d shared -> %s" % (keys[a], keys[b], len(common), common[:6]))
    if not shared_any:
        print("   0 shared sung lines between any two variations")

    print("=" * 78)
    print("SUMMARY")
    hdr = ("song", "style", "excl", "lyric", "lines", "w/line", "rhyme",
           "ret_sc", "ret_st", "hw", "nums")
    print("%-5s %6s %6s %6s %6s %7s %7s %7s %7s %4s %5s" % hdr)
    for r in rows:
        print("%-5s %6d %6d %6d %6d %7.2f %7.3f %7.3f %7.3f %4d %5d" % r)

    # dedication blocks, counted separately and in code
    for m in re.finditer("(?m)^#{1,6}[^" + NL + "]*DEDICATION[^" + NL + "]*$", text.upper()):
        pass
    ded_blocks = []
    heads = [(m.start(), m.group(0).upper())
             for m in re.finditer("(?m)^#{1,6}[^" + NL + "]*$", text)]
    for m in re.finditer(re.escape(FENCE) + "[a-z]*" + NL + "(.*?)" + re.escape(FENCE),
                         text, re.S):
        prior = [h for h in heads if h[0] < m.start()]
        if prior and "DEDICATION" in prior[-1][1]:
            ded_blocks.append(m.group(1).rstrip(NL))
    for j, d in enumerate(ded_blocks):
        n = len(d)
        print("DEDICATION %d chars: %d  (ceiling 500)" % (j + 1, n))
        if n > 500:
            fails.append("dedication %d over 500 chars: %d" % (j + 1, n))

    print("=" * 78)
    if fails:
        print("FAILURES (%d):" % len(fails))
        for f in fails:
            print("  - " + f)
        return 3
    print("ALL GATES CLEAN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
