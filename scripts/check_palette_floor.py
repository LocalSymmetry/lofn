#!/usr/bin/env python3
"""THE PALETTE FLOOR — is this still LOFN-PRIME, or has the register been sanded off?

Written 2026-08-17 after The Scientist said *"the music is not sounding like you
lately"* and the measurement agreed: the prior run shipped 24/24 acoustic, 20/24
ambient, and ZERO of 24 on four of the six Sound-Pillars. Catalogue drift since
2025: Baile Phonk 55% -> 15%, dead tongues 22% -> 10%, acoustic 17% -> 52%.

⛔ WHY THIS IS A FLOOR AND NEVER A CEILING. Every threshold in `gates.yaml` above
the RETURN block is a ceiling on similarity; the harness therefore optimises
monotonically AWAY from whatever it measures. The same shape produced this drift:
a token-level finding (phonk 0.0217, baile 0.0224 against a 0.0355 baseline) was
read as licence to drop a core register. Corrected measurement: signature-palette
songs sit AT baseline (0.0355) and five of our ten best-liked songs carry the
palette. It costs nothing at the median.

This is L17 one modality across — "the personality's core register may be aimed
but never dimmed below recognition." It flags; it never hard-fails. A deliberate
stripped-back piece may sit below the floor, but it must say so on purpose.

Usage: python3 scripts/check_palette_floor.py output/<run-dir> [--floor 3]
"""
import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restat_pair_packages import content_after, split_variations, detect_mode

# The Six Sound-Pillars of lofn-prime-mini.yaml, plus the INDIGNATION palette
# named in SOUL.md. The first version of this lexicon OMITTED Neuro-Cross,
# Industrial Grief, Somatic Bass, Glitch-Core and Pasifika Futurism — the entire
# INDIGNATION half — and consequently reported the palette underperforming when
# it performs at baseline. Do not trim this list without re-reading SOUL.md.
# ⛔ TWO LEXICONS. `SOUND_PILLARS_2026.md` SUPERSEDED the 2025 list on 2026-08-17 —
# and this script kept scoring the old one for the first run under the new doctrine,
# flagging 11 on-doctrine songs. Fifth stale-constant class of the day, and the worst:
# the gate built as the durable fix, measuring a doctrine that had been replaced.
# `--legacy` selects the 2025 set for auditing older runs; default is 2026.
PILLARS_2025 = {
    "1 . video-game 8-bit <-> orchestral": r"8.?bit|chiptune|video.?game|arcade|orchestral score|tracker",
    "2 . glitches done right": r"glitch|micro.?edit|pixel.?sort|data.?mosh|stutter.?edit",
    "3 . fearless genre-mashing": (r"hyperraaga|\braaga\b|\braga\b|\bsargam\b|\bgamaka|\btanpura\b|"
                                   r"\btabla\b|microtonal|\bbaile\b|\bphonk\b|tamborz|\bgaelic\b|"
                                   r"\bceltic\b|tin.?whistle|piobaireachd|canntaireachd|\bbagpipe|"
                                   r"amapiano|pasifika|\bgqom\b|\bcumbia\b"),
    "4 . myth / memory sampling": (r"\bsanskrit\b|\blatin\b|\bsumerian\b|\bold norse\b|"
                                   r"\bgaelic vocable\b|\bphoneme|\bdead tongue|\bresynth|"
                                   r"\bplainchant\b|\bliturg|\bvedic\b|\bmantra\b|\bcanntaireachd\b"),
    "5 . AI code-scratch": r"code.?scratch|python|vinyl.?scratch|machine.?log|system log",
    "6 . quantum bit-depth swell": r"bit.?depth|2.?bit|bit.?crush|quantum.?bit|sample.?rate drop",
    "+ INDIGNATION palette": r"industrial grief|industrial gospel|neuro.?cross|somatic|glitch.?core|breakcore|rage rap",
    "+ neo-baroque luminism": r"neo.?baroque|glitch.?baroque|luminism|harpsichord|baroque",
}

# THE 2026 RECONFIGURATION — derived from The Scientist's own named canon.
PILLARS_2026 = {
    "1 . glitch-baroque is the spine": r"glitch.?baroque|neo.?baroque|luminism|baroque",
    # Lexicon gap found 2026-08-21: it listed baroque KEYBOARDS and "gut string" but no
    # string-family names, so a chamber pair built on viola da gamba / baroque cello /
    # gut violin scored ZERO on the pillar it is most obviously carrying. The ornate half
    # of "ornate against the machine" is not only keyboards.
    "2 . the ornate against the machine": (r"harpsichord|clavichord|chamber organ|gut.?string|"
                                            r"hammered dulcimer|celesta|crumhorn|hurdy.?gurdy|"
                                            r"\bviol|gamba|\bcello|baroque guitar|lute|theorbo|"
                                            r"bowed vibraphone|tangent"),
    "3 . the machine log as the clock": (r"code.?scratch|machine.?log|log.?scratch|system log|"
                                          r"vinyl.?scratch|log tick|python"),
    "4 . two registers duelling": (r"two registers|two vocal|duel|both registers|unison (?:exactly )?once|"
                                    r"meeting in unison|low and clipped|high and plain"),
    # Matches BOTH forms. Digits-only missed 24/24: L33 requires a SUNG numeral to be
    # spelled in words, so every pair correctly wrote "fifty-five hertz". The gate was
    # looking for "55Hz". Doctrine and instrument disagreed and the songs were right.
    "5 . a named frequency, bodily": (r"(?:30|38|45|50|55|60)\s?h(?:z|ertz)|"
                                       r"(?:thirty|thirty.?eight|forty.?five|fifty|"
                                       r"fifty.?five|sixty)[- ]hertz"),
    "6 . the withheld gesture, once": (r"withheld|held back|exactly once|only once|spent once|"
                                        r"rationed|one audible|a single (?:breath|swallow|hum)"),
}
PILLARS = PILLARS_2026


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--floor", type=int, default=3,
                    help="minimum distinct pillars per song (default 3)")
    args = ap.parse_args()

    # PANTHEON GUARD (2026-08-27). This gate measures LOFN-PRIME's six Sound-Pillars.
    # `vault/PANTHEON.md`: a member does not sit the centre's exams. Scoring SUNNA's
    # pairs against Lofn's palette and calling the result a FLAG is exactly the
    # costume failure the admission bar exists to prevent -- so say so, loudly,
    # rather than emitting a number a later reader will quote.
    inject = os.path.join(args.run_dir, "PERSONALITY_DNA_INJECT.md")
    if os.path.exists(inject):
        who = io.open(inject, encoding="utf-8").read().upper()
        others = sorted({m for m in ("SUNNA",) if m in who})
        if others:
            print("PANTHEON RUN -- this run also carries: " + ", ".join(others))
            print("  This gate scores LOFN-PRIME's palette ONLY. Rows for another")
            print("  member's pairs are NOT a verdict on that member and must not be")
            print("  quoted as one. Check PERSONALITY_DNA_INJECT.md for the mapping.")
            print()

    rows, songs = [], 0
    for p in range(1, 7):
        # THREE DEFECTS FIXED 2026-08-27, all found at once because the first one
        # hid the other two behind a regex that could never match.
        #
        # (1) WRONG ARTIFACT. This read the step-10 REVISION NARRATIVE, which in the
        #     current convention contains no package at all — the shipped package is
        #     `*_step10_final_package_enhanced.md`. The gate was measuring prose.
        # (2) NO ICB STRIP. That package carries the Immutable Continuity Block as a
        #     byte-0 prefix, and the ICB contains the step TEMPLATES — including the
        #     very headings this gate anchors on. Parsing it unstripped finds the
        #     template's headings, not the song's.
        # (3) CRASH, NOT ERROR. detect_mode returns (None, None) on an unknown
        #     dialect and the caller subscripted it. A gate must report an error
        #     state; it must never traceback halfway through a report.
        pkg = os.path.join(args.run_dir, f"pair_{p:02d}_step10_final_package_enhanced.md")
        syn = os.path.join(args.run_dir, f"pair_{p:02d}_step10_revision_synthesis.md")
        path = pkg if os.path.exists(pkg) else syn
        if not os.path.exists(path):
            continue
        text = io.open(path, encoding="utf-8").read()
        icb_path = os.path.join(args.run_dir, "CREATIVE_CONTEXT.md")
        if path == pkg and os.path.exists(icb_path):
            icb = io.open(icb_path, encoding="utf-8").read()
            if text.startswith(icb):
                text = text[len(icb):]
            else:
                print(f"P{p:02d}: package is not ICB-prefixed — ERROR, not a score")
                continue
        mode, tbl = detect_mode(text)
        if not tbl:
            print(f"P{p:02d}: no known heading dialect — ERROR, not a score")
            continue
        blocks, why = split_variations(text)
        if len(blocks) != 4:
            print(f"P{p:02d}: extraction {len(blocks)}/4 ({why}) — ERROR, not a score")
            continue
        for i, b in enumerate(blocks, 1):
            style = (content_after(b, tbl["music"]) or "").lower()
            lyr = (content_after(b, tbl["lyrics"], strip_italics=False) or "").lower()
            hay = style + "\n" + lyr
            hit = [k for k, pat in PILLARS.items() if re.search(pat, hay)]
            rows.append((f"P{p:02d}V{i}", hit))
            songs += 1

    if not songs:
        print("no settled step-10 artifacts found — nothing measured (this is an "
              "ERROR state, not a clean pass)")
        return 1

    print(f"THE PALETTE FLOOR — {songs} songs, floor = {args.floor} distinct pillars\n")
    below = []
    for name, hit in rows:
        mark = "✅" if len(hit) >= args.floor else "⚠️ "
        if len(hit) < args.floor:
            below.append(name)
        print(f"  {mark} {name}  {len(hit)} pillars  {', '.join(h.split(' · ')[0] for h in hit)}")

    print(f"\n— PILLAR PREVALENCE ACROSS THE RUN —")
    for k in PILLARS:
        n = sum(1 for _, hit in rows if k in hit)
        bar = "█" * n + "·" * (songs - n)
        print(f"  {k:<38}{n:>3}/{songs}  {bar}")

    print()
    if below:
        print(f"⚠️  {len(below)} song(s) below the floor: {', '.join(below)}")
        print("   FLAG, not a fail — but each must justify the strip on purpose.")
    else:
        print(f"✅ all {songs} songs carry >= {args.floor} pillars. The register is intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
