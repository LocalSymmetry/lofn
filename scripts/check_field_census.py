#!/usr/bin/env python3
"""check_field_census.py -- an INDEPENDENT count of forbidden tokens in the three
pasted Suno fields, made by the coordinator against the shipped artifacts.

WHY THIS EXISTS (FINDING 12, 2026-08-27). Fresh-context QA on
`2026-08-26-pantheon-rare-registers` found five violations. THREE of them were
sitting directly underneath a verification row that said zero -- and every one of
those rows had been written by the pair being counted. The coordinator copied one
of them into the run-wide table without re-measuring; the selection judge then read
the coordinator table and reported the run clean. One unverified claim wearing three
hats, each hat making the next look like corroboration.

    ⭐ A NUMBER DOES NOT BECOME TRUE BY BEING COPIED FORWARD.

So this reader never reads a verification row. It reads the FIELDS.

⭐ THE UNIT IS THE FENCED BLOCK. Four of this coordinator's scanners failed in one
day by parsing markdown HEADINGS, which vary per pair (`## 2. LYRICS` / `## 2. SUNO
LYRICS` / `## SUNO LYRICS` / `## 2. LYRICS - MAX config...`). The pasted fields are
exactly the fenced code blocks -- 72 of 72, verified independently -- so this reads
those and never needs to know which dialect a pair chose.

A hit named in a pair's `## Major Deviations` is reported DECLARED and does not fail
the run; the open manufacturer-token question is The Scientist's, not this script's.
An UNDECLARED hit fails.

    python3 scripts/check_field_census.py <run-dir> [--lexicon vault/field_token_lexicon.yaml]

Exit 0 = no undeclared hits. Exit 3 = undeclared hits found. Exit 2 = nothing
parseable, reported rather than passed.

Read-only. Writes nothing, renders nothing, spends nothing.
"""
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FENCE = chr(96) * 3


def load_lexicon(path):
    """Tiny YAML subset: `key:` then `  - value` lines. No dependency, no eval."""
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


# ⛔ UNIT CORRECTION, 2026-08-27, ON THIS SCRIPT'S FIRST LIVE RUN. The first version
# took EVERY fenced block as a pasted field and instantly produced four false
# artist-class hits -- `scruggs`, `monroe`, `jamerson` -- by reading the DEDICATION
# fences, which are the one place those names are supposed to be. It was built to fix
# a bad measurement and shipped a bad measurement, in the same hour, on the same axis.
#
# The three pasted Suno fields are the fences under a heading that NAMES one of them.
# Match on the SUBSTRING, never the whole heading: the dialects observed in one run
# were `## 1. MUSIC PROMPT`, `## 1. MUSIC PROMPT - SUNO STYLE PROMPT`, `## 1B. SUNO
# EXCLUDE PROMPT`, `## 2. LYRICS`, `## 2. SUNO LYRICS`, `## SUNO LYRICS` and
# `## 2. LYRICS  - MAX config: ...`. Substring matching survives all seven; exact
# matching survives one.
FIELD_KINDS = (("music", "MUSIC PROMPT"), ("exclude", "EXCLUDE"), ("lyrics", "LYRIC"))
NOT_A_FIELD = ("DEDICATION", "CREDIT", "LINEAGE", "VERIFICATION", "REPAIR LOG",
               "PROVENANCE", "DIGEST", "CHECKLIST", "DEVIATION")


def pasted_fields(text):
    """-> [(kind, block_text)] for the three pasted Suno fields only."""
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
                # ⛔ THE UNIT IS THE STRING A HUMAN PASTES INTO SUNO (FINDING 9,
                # 2026-08-26, P03's correction). The newline before the closing
                # fence belongs to the fence, not to the field -- keeping it makes
                # every character count run +1 and turns a legal 960 into a 961
                # band violation. Irrelevant to a token search, fatal to a count,
                # and this helper is shared. The coordinator reproduced the same
                # +1 error again on 2026-08-27 and caught it only by cross-checking
                # against a pair's independently reported number.
                out.append((kind, m.group(1).rstrip(chr(10))))
                break
    return out


def hits(hay, token):
    """Word-bounded, built from lookarounds so no escape sequence can be mangled
    by a shell on its way into this file. (FINDING 11 -- the same day.)"""
    pat = "(?<![a-z0-9])" + re.escape(token) + "(?![a-z0-9])"
    return len(re.findall(pat, hay))


def main():
    args = [a for a in sys.argv[1:]]
    lex_path = os.path.join("vault", "field_token_lexicon.yaml")
    if "--lexicon" in args:
        i = args.index("--lexicon")
        lex_path = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1:
        print("usage: check_field_census.py <run-dir> [--lexicon <path>]")
        return 2
    run = args[0]
    if not os.path.isdir(run):
        print("FAIL: not a directory: " + run)
        return 2
    if not os.path.exists(lex_path):
        print("FAIL: lexicon not found: " + lex_path)
        return 2

    lexicon = load_lexicon(lex_path)
    icb_path = os.path.join(run, "CREATIVE_CONTEXT.md")
    icb = open(icb_path, encoding="utf-8").read() if os.path.exists(icb_path) else ""

    pkgs = sorted(f for f in os.listdir(run)
                  if f.startswith("pair_") and "final_package" in f)
    if not pkgs:
        print("FAIL: no shipped packages found -- nothing measured.")
        print("      A checker that measures nothing must not return success.")
        return 2

    print("FIELD TOKEN CENSUS -- " + os.path.basename(os.path.abspath(run)))
    print("  lexicon: " + lex_path + "  (" +
          ", ".join(g + "=" + str(len(v)) for g, v in sorted(lexicon.items())) + ")")
    print()

    undeclared, declared, blocks_total = [], [], 0
    for f in pkgs:
        pair = f.split("_")[1]
        text = open(os.path.join(run, f), encoding="utf-8").read()
        body = text[len(icb):] if icb and text.startswith(icb) else text
        declared_zone = ""
        for m in re.finditer("(?im)^#{1,4}[^" + chr(10) + "]*major deviation[^" + chr(10) + "]*$", body):
            declared_zone += body[m.end():m.end() + 4000].lower()
        blocks = pasted_fields(body)
        blocks_total += len(blocks)
        permitted = set(lexicon.get("exclude_permitted", []))
        found = {}
        for kind, blk in blocks:
            low = blk.lower()
            for group, tokens in lexicon.items():
                if group == "exclude_permitted":
                    continue
                for tok in tokens:
                    n = hits(low, tok)
                    # A brand that has become a generic noun is unremarkable in the
                    # NEGATIVE field: "no autotune" is a texture instruction, not an
                    # instruction to imitate a company. QA ruled this de minimis and
                    # the ruling is right -- but only for the exclude field.
                    if n and not (kind == "exclude" and tok in permitted):
                        found[(group, tok)] = found.get((group, tok), 0) + n
        if not found:
            print("  P" + pair + ": clean  (" + str(len(blocks)) + " pasted fields)")
            continue
        print("  P" + pair + ": " + str(len(found)) + " token(s) in "
              + str(len(blocks)) + " pasted fields")
        for (group, tok), n in sorted(found.items()):
            is_declared = tok in declared_zone
            mark = "declared" if is_declared else "UNDECLARED"
            (declared if is_declared else undeclared).append((pair, group, tok, n))
            print("      " + ("OK  " if is_declared else "!!  ") + group + " / "
                  + tok + " x" + str(n) + "  [" + mark + "]")

    print()
    print("  pasted fields scanned: " + str(blocks_total) + "  (dedications and verification tables are NOT fields)")
    if declared:
        print("  declared hits (reported, not failed): " + str(len(declared)))
    if undeclared:
        art = [u for u in undeclared if u[1] == "artists"]
        print("  RESULT: " + str(len(undeclared)) + " UNDECLARED token(s)")
        if art:
            print("  " + str(len(art)) + " in the ARTISTS class, which has no exemption:")
            for p, g, t, n in art:
                print("      P" + p + " / " + t + " x" + str(n))
        print("  Scene and technique in the prompt; people in the dedication.")
        return 3
    print("  RESULT: no undeclared forbidden tokens in any pasted field.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
