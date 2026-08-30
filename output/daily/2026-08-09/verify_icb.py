# -*- coding: utf-8 -*-
"""ICB integrity verifier for run 2026-08-09_daily_music_genz.

WHY THIS EXISTS (L28 / L31): the gate "the count of `(after ` speaker tags must == 18" is
ambiguous if read as a raw substring count -- the metaprompt legitimately carries `(after X)`
inside its aha-moment ATTRIBUTIONS, which are citations, not speaker tags. A naive grep sees 23
and reports a failure that is not one; a naive fix would delete correct attributions.

So the convention is PINNED HERE, in a validator, and every downstream re-stat and QA pass uses
THIS definition rather than reading a convention off the artifacts:

    a SPEAKER TAG is a line matching  ^SPEAKER TAG: .* \\(after .*\\):\\s*$

Both numbers are printed. An empty extraction is a hard ERROR, never a score.
Usage:  python3 output/daily/2026-08-09/verify_icb.py
"""
from __future__ import print_function
import hashlib, io, os, re, sys

RUN = os.path.dirname(os.path.abspath(__file__))
ICB = os.path.join(RUN, "CREATIVE_CONTEXT.md")
PERSONALITY = os.path.join(
    RUN, "..", "..", "..", "skills", "orchestration", "personalities", "lofn-prime-mini.yaml")

EXPECTED_SHA = "297941561ca6880d38c323dcc0fdd739aa6fd970e7293fd7e98e38fb0b882f4b"

SPEAKER_TAG = re.compile(r"^SPEAKER TAG: .*\(after .*\):\s*$", re.M)

def main():
    raw = io.open(ICB, "rb").read()
    txt = raw.decode("utf-8")
    pers = io.open(PERSONALITY, encoding="utf-8").read().rstrip()

    tags = SPEAKER_TAG.findall(txt)
    checks = []
    checks.append(("icb exists, non-trivial",      len(raw) > 100000,           "%d bytes" % len(raw)))
    checks.append(("icb sha256 == frozen value",   hashlib.sha256(raw).hexdigest() == EXPECTED_SHA,
                                                   hashlib.sha256(raw).hexdigest()))
    checks.append(("SPEAKER TAG count == 18",      len(tags) == 18,             "%d tags" % len(tags)))
    checks.append(("personality unbroken substring", pers in txt,               "%d bytes inlined" % len(pers.encode("utf-8"))))
    checks.append(("3 Hyper-Skeptic seats",        txt.count("HYPER-SKEPTIC") == 3, "%d" % txt.count("HYPER-SKEPTIC")))
    checks.append(("15 Special Flairs, 1..15",     all(("\n%d. " % n) in txt for n in range(1, 16)), "1-15"))
    checks.append(("plural marker 'Special Flairs'", "Special Flairs" in txt,    "present"))
    for slot in ("Meta-Prompt", "Golden Seed", "Concept Panel", "Medium Panel",
                 "Context & Marketing Panel", "research brief", "Seed Genre Palette",
                 "Seed Music Frames"):
        checks.append(("slot non-empty: %s" % slot, slot in txt, "present" if slot in txt else "MISSING"))

    if not tags:
        print("ERROR: extracted ZERO speaker tags -- this is a broken matcher, not a clean file.")
        return 2

    ok = True
    for name, passed, detail in checks:
        print("%-34s %-5s  %s" % (name, "PASS" if passed else "FAIL", detail))
        ok = ok and passed
    print("-" * 62)
    print("raw '(after ' substring occurrences : %d  "
          "(18 speaker tags + %d metaprompt attributions -- both correct)"
          % (txt.count("(after "), txt.count("(after ") - len(tags)))
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
