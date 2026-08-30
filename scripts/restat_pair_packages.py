#!/usr/bin/env python3
"""Coordinator-side RE-STAT of a pair's step-10 Suno packages.

The subagent's RETURN envelope is a CLAIM. This is the PROOF (EXECUTION.md §3).

Two rules from COMPETITION_LEARNINGS govern this file's design:
  L28/L31 — MODE DETECTION, never pattern priority. Detect the file's heading
            convention once, then match that table alone. A "canonical-first,
            loose-fallback" matcher does not remove the failure mode, it only
            reorders it.
  L25     — AN EMPTY EXTRACTION IS A HARD ERROR, NEVER A SCORE. Print what was
            extracted before trusting what was concluded, and assert the
            extraction count equals the expected cardinality.

Usage:  python3 scripts/restat_pair_packages.py output/<run-dir> [--pairs 1 2 3]
"""
import argparse
import io
import json
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
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from measure_soundcraft import sung as _canonical_sung  # THE definition — never re-derive

# Seconds a pair artifact must have been untouched before it may be scored.
SETTLE_SECONDS = 90

GATES = {
    "music_prompt_chars": (850, 1000),
    "music_prompt_target": (870, 960),
    "music_prompt_hug_ceiling": 985,
    "exclude_chars": (400, 900),
    "lyrics_field_max": 5000,
    "lyrics_field_target": 4800,
    "sung_lines_band": (70, 120),
    "sung_lines_run_target": (88, 110),   # this run's upper-half mandate
    "mean_words_per_line_ceiling": 7.5,
}
BANNED_OPENERS = ("Begin in", "Begin with", "Use ", "Build the track from",
                  "Chronology:", "For an adult human singer")
# RUN-SCOPED, EMPTY BY DEFAULT - the FIFTH stale constant of this class.
# The four number-strings that lived here arrived at commit cb3956e0 and belong to an
# earlier run. Carried blind into `2026-08-26-opus-masterpiece` they would have condemned
# CORRECT work: that run's P02 is MANDATED by its step-05 isolation brief to sing
# "fourteen" as its one-fact hinge (it appears 18 times in the shipped package), so this
# scanner would have reported forbidden_sung_number FAIL x4 on a passing pair - and the
# repair an agent reaches for is to delete the mandated number to appease the scanner.
# Same class as ANCHOR_STEMS, MANDATED_SHARED and APOLOGY, each documented below.
# A constant from another run either lies in the report or condemns correct work.
# Populate per run or leave empty.
FORBIDDEN_SUNG = []
# RUN-SCOPED, EMPTY BY DEFAULT - the fourth stale constant of 2026-08-17.
# The no-apology gate belongs to `2026-08-17_magnum-opus-suite`, where the Sontag seat
# made "no line apologises" a Somatic condition. Carried blind into the next run it
# flagged "Sorry does not travel up", "I said sorry in every grammar I have" and
# "Everything in here is guilty" - which are that run's SUBJECT (being blamed), not a
# breach. Same class as ANCHOR_STEMS and MANDATED_SHARED: a constant from another run
# either lies in the report or condemns correct work. Populate per run or leave empty.
APOLOGY = []
HOUSE_LEXICON = [
    "more sub and more sky", "small astonished laugh", "triple arch",
    "triple-arch panorama", "small enough to understand", "frost-air pad",
    "starfield percussion", "zodiacal glow", "clear silver tone", "dew-bright",
    "glass harmonica sheen", "crystalline arpeggios", "make my little fear",
]
# The anchor phrase has THREE mandated grammatical states, one per movement. A
# checker keyed to only the first-person stem reports a hard-gate failure on every
# Movement II package, which is the state the metaprompt REQUIRES there. Same class
# as the fence/delimiter misses above: the instrument was keyed to one shape of a
# thing that was specified in three.
# ⛔ RUN-SCOPED, AND EMPTY BY DEFAULT. These were hard-coded to the anchor phrase of
# the `magnum-opus-suite` run and then silently carried into the next run, where they
# reported "anchor ABSENT" on every song of a run that has no anchor phrase at all —
# a false hard-gate failure on 8 correct songs, caught by the P04 agent.
# A check keyed to another run's constant is worse than no check. Pass --anchor to
# enable it; absent, the check is reported as NOT APPLICABLE rather than failed.
ANCHOR_STEMS = []


def count_headings(text, heading):
    """Count REAL headings, anchored to line start/end — never substring mentions.

    Shared helper because this class of bug has now been introduced twice in this
    run, in two different functions: prose that merely *mentions* `## 1. MUSIC
    PROMPT` inside a self-check paragraph inflates an unanchored count. Pair 03's
    enhanced package contains the string three times and the heading once.
    Anything counting headings must call this.
    """
    return len(re.findall(r"^" + re.escape(heading) + r"[ \t]*$", text, flags=re.M))


def content_after(text, heading, strip_italics=True):
    """Return the payload under `heading`, DETECTING whether it is fenced or bare prose.

    ⛔ The step-10 contract pins the heading names but NOT whether the payload sits
    inside a fence. Measured on this run, wave 1:
        pair 02 -> all three payloads fenced (```text ... ```)
        pair 01 -> LYRICS fenced; MUSIC PROMPT and EXCLUDE PROMPT as BARE PROSE
    That second shape is the exact convention L28 recorded on 2026-08-05
    ("`## LYRICS` with music/exclude unfenced"). The contract still does not pin it,
    so it recurred — and a fence-only seeker silently runs PAST an unfenced payload
    and returns the NEXT fence it finds, which is the lyrics block. That yields
    music == exclude == lyrics: an impossible number, and therefore a bug report
    rather than a finding (L25).

    Detection is scoped to the SLICE between this heading and the next heading, so
    the two shapes cannot alias onto each other.
    """
    # ⛔ ANCHORED to line start. `text.find()` also matches a prose MENTION of the
    # heading — pair 02's step-11 package contains a reader note saying "this section
    # sits between `## 1B.` and `## 2. LYRICS`", written so a human would not misread
    # the Disc_Channel placement. An unanchored find grabbed that sentence, sliced to
    # the real heading, and returned 624 chars with 0 sung lines for a complete 42 KB
    # package. Third occurrence of this class in one session (count_headings and
    # split_variations were fixed first) — and this is the function they both call.
    mm = re.search(r"^" + re.escape(heading) + r"[ 	]*$", text, flags=re.M)
    if mm is None:
        return None
    i = mm.start()
    start = mm.end()
    nxt = re.search(r"^#{1,3}\s", text[start:], flags=re.M)
    slice_ = text[start:start + nxt.start()] if nxt else text[start:]

    fence = re.search(r"```[a-zA-Z]*\n(.*?)\n```", slice_, re.S)
    if fence:
        return fence.group(1)
    # Bare-prose mode: drop the agent's own annotations (e.g. "**measured: 959
    # characters**") — commentary about the payload, never the payload.
    # ⛔ `strip_italics` is FALSE for lyrics. A single-asterisk line is an annotation in
    # a prompt field and an *SFX cue* in a lyric — same syntax, opposite meaning. The
    # first version of this filter dropped them unconditionally and then the SFX check
    # reported 0 cues on a pair that had three: the instrument deleted the evidence and
    # then failed the artifact for its absence.
    keep = []
    for ln in slice_.strip().split("\n"):
        s = ln.strip()
        if not s or re.match(r"^\*{1,2}measured", s, re.I):
            continue
        if strip_italics and re.match(r"^\*[^*]+\*$", s):
            continue
        keep.append(ln)
    return "\n".join(keep).strip() or None


def detect_mode(text):
    """Detect the heading convention ONCE. Never fall back to a looser table."""
    tables = {
        "canonical": {"music": "## 1. MUSIC PROMPT",
                      "exclude": "## 1B. SUNO EXCLUDE PROMPT",
                      "lyrics": "## 2. LYRICS",
                      "title": "## 3. TITLE"},
        "legacy_exclude": {"music": "## 1. MUSIC PROMPT",
                           "exclude": "## EXCLUDE PROMPT",
                           "lyrics": "## 2. LYRICS",
                           "title": "## 3. TITLE"},
        "lyrics_prompt": {"music": "## 1. MUSIC PROMPT",
                          "exclude": "## 1B. SUNO EXCLUDE PROMPT",
                          "lyrics": "## 3. LYRICS PROMPT",
                          "title": "## 4. TITLE"},
    }
    for name, tbl in tables.items():
        if all(text.count(h) >= 1 for h in tbl.values()):
            return name, tbl
    return None, None


def split_variations(text):
    """Split a step-10 file into its 4 variation blocks by DETECTING the delimiter.

    ⛔ The step-10 contract pins the SECTION headings (`## 1. MUSIC PROMPT` etc.) but
    does NOT pin the variation delimiter — so every agent invents one. Measured on
    this run, wave 1:
        pair 01 -> `# ⭐ VARIATION 1 — V1 · **THE ARITHMETIC OF WAITING**`
        pair 02 -> `# ===== VARIATION V1`
    Two agents, two conventions, both internally consistent and both correct: an
    agent that invents a delimiter nobody specified is behaving reasonably (L30).

    Per L31 the fix is MODE DETECTION, never pattern priority: collect every level-1
    heading naming a variation, confirm there are exactly 4, and split on those. A
    "strict-first, loose-fallback" chain does not remove the failure, it reorders it —
    the loose arm then over-matches prose headings in a canonical file.
    """
    # STRUCTURAL, not lexical. Anchor on the one thing the contract DOES pin — the
    # `## 1. MUSIC PROMPT` heading — and walk back to the nearest preceding level-1
    # heading. That is the block start under every naming convention, so the matcher
    # never has to know what any agent decided to call a variation.
    music_h = "## 1. MUSIC PROMPT"
    # Anchored to LINE START and END. An unanchored match also fires on prose that
    # merely mentions `## 1. MUSIC PROMPT` inside a self-check paragraph — pair 03
    # reported 6 anchors for 4 real headings for exactly that reason.
    anchors = [m.start() for m in
               re.finditer(r"^" + re.escape(music_h) + r"[ \t]*$", text, flags=re.M)]
    if len(anchors) != 4:
        return [], f"{len(anchors)} `{music_h}` anchors found"
    h1 = [m.start() for m in re.finditer(r"^#\s+\S", text, flags=re.M)]
    starts = []
    for a in anchors:
        prior = [h for h in h1 if h < a]
        starts.append(prior[-1] if prior else a)
    starts = sorted(set(starts))
    if len(starts) != 4:
        return [], f"{len(starts)} distinct level-1 blocks for 4 anchors"
    blocks = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        blocks.append(text[s:end])
    return blocks, "structural (level-1 heading preceding each MUSIC PROMPT)"


def variation_label(block):
    """Recover V1..V4 from whatever heading convention this file used."""
    head = block.split("\n", 1)[0]
    m = re.search(r"\bV(\d)\b", head) or re.search(r"VARIATION\s+(\d)", head)
    return f"V{m.group(1)}" if m else head.strip()[:24]


def sung_lines(lyrics):
    """Lines actually sung — delegated to the CANONICAL shipped definition.

    ⛔ DO NOT REIMPLEMENT THIS. `lofn-music/SKILL.md` is explicit:
    `scripts/measure_soundcraft.py` "is THE definition… Never hand an agent a
    numeric target without the function that computes it."

    A hand-rolled version was written here first and disagreed with the canonical
    one twice, in opposite directions:
      * it dropped `(parenthetical)` lines as "backing vocal", deleting half of
        pair 03 — whose ASSIGNED architecture is call-and-response — and reporting
        55-61 sung lines against a true 91-103;
      * it kept inline `*sfx*` inside otherwise-sung lines, inflating pair 06's
        words-per-line to 7.68 against a true 7.31 and flagging a correct song
        over the 7.5 ceiling.
    Both flags were the instrument, not the song. The canonical function strips
    inline SFX and unwraps parentheses; use it.
    """
    return _canonical_sung(lyrics)


def check_variation(vid, block, mode_tbl, report):
    music = content_after(block, mode_tbl["music"])
    exclude = content_after(block, mode_tbl["exclude"])
    # strip_italics=False: in a lyric an *asterisked line* is an SFX cue, not an annotation.
    lyrics = content_after(block, mode_tbl["lyrics"], strip_italics=False)
    if music is None or exclude is None or lyrics is None:
        report.append((vid, "EXTRACTION", "3 fenced blocks", "MISSING", False))
        return None

    m_len, e_len, l_len = len(music.strip()), len(exclude.strip()), len(lyrics.strip())
    lines = sung_lines(lyrics)
    n = len(lines)
    wpl = (sum(len(x.split()) for x in lines) / n) if n else 0.0

    lo, hi = GATES["music_prompt_chars"]
    report.append((vid, "music_prompt_chars", f"{lo}-{hi}", m_len, lo <= m_len <= hi))
    report.append((vid, "music_prompt_hug", f"<{GATES['music_prompt_hug_ceiling']}",
                   m_len, m_len < GATES["music_prompt_hug_ceiling"]))
    report.append((vid, "music_prompt_terminal_punct", ".!?",
                   music.strip()[-1], music.strip()[-1] in ".!?"))
    bad = [o for o in BANNED_OPENERS if music.strip().startswith(o)]
    report.append((vid, "banned_opener", "none", bad or "none", not bad))
    lo, hi = GATES["exclude_chars"]
    report.append((vid, "exclude_chars", f"{lo}-{hi}", e_len, lo <= e_len <= hi))
    report.append((vid, "lyrics_field_chars", f"<{GATES['lyrics_field_max']}",
                   l_len, l_len < GATES["lyrics_field_max"]))
    report.append((vid, "lyrics_field_target", f"<={GATES['lyrics_field_target']}",
                   l_len, l_len <= GATES["lyrics_field_target"]))
    lo, hi = GATES["sung_lines_run_target"]
    report.append((vid, "sung_lines", f"{lo}-{hi}", n, lo <= n <= hi))
    report.append((vid, "mean_words_per_line", f"<{GATES['mean_words_per_line_ceiling']}",
                   round(wpl, 2), wpl < GATES["mean_words_per_line_ceiling"]))

    low = lyrics.lower()
    # ⛔ The apology and forbidden-number gates are about what is SUNG. Scanning the
    # whole lyrics block also reads EMO headers — which carry performance directions
    # like "steady, no apology in the delivery". That is the gate being ENFORCED, and
    # flagging it would have had me delete a correct direction to appease a broken
    # scanner (L27: when a scan hits a line, fix the SCANNER, not the line).
    sung_only = "\n".join(lines).lower()
    hits = [p for p in FORBIDDEN_SUNG if re.search(p, sung_only)]
    report.append((vid, "forbidden_sung_number", "none", hits or "none", not hits))
    # ⛔ NEGATION WINDOW. "and she was not sorry" is the no-apology gate being ENFORCED,
    # not violated — flagging it invites deleting a correct line to appease a scanner
    # (L27: when a scan hits a line, fix the SCANNER). Look back a few words for a
    # negator before counting a hit.
    NEG = r"(?:\bnot\b|\bnever\b|\bno\b|\bwithout\b|\bdidn'?t\b|\bwon'?t\b|\bisn'?t\b|\bwasn'?t\b)"
    ap = []
    for p in APOLOGY:
        for m in re.finditer(p, sung_only):
            window = sung_only[max(0, m.start() - 40):m.start()]
            if not re.search(NEG + r"[^.]{0,25}$", window):
                ap.append(m.group(0))
    ap = sorted(set(ap))
    report.append((vid, "apology_vocabulary", "none unnegated (sung lines)",
                   ap or "none", not ap))
    hl = [p for p in HOUSE_LEXICON if p in low]
    report.append((vid, "house_lexicon", "none", hl or "none", not hl))
    # No stems configured for this run -> the check does not apply. Reporting a fail
    # here is how another run's constant became 8 false hard-gate failures.
    if ANCHOR_STEMS:
        found = [s for s in ANCHOR_STEMS if s in low]
        report.append((vid, "anchor_phrase", "present (any mandated state)",
                       found or "ABSENT", bool(found)))
    report.append((vid, "sfx_cue", ">=1",
                   sum(1 for x in lyrics.split("\n")
                       if x.strip().startswith("*") and x.strip().endswith("*")),
                   any(x.strip().startswith("*") and x.strip().endswith("*")
                       for x in lyrics.split("\n"))))
    bare = re.findall(r"EMO:\s*(AWE|INDIGNATION)\s*[\]-]", lyrics, re.I)
    report.append((vid, "bare_emo_tag", "none", bare or "none", not bare))
    return {"music": m_len, "exclude": e_len, "lyrics": l_len,
            "sung": n, "wpl": round(wpl, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pairs", nargs="*", type=int, default=list(range(1, 7)))
    args = ap.parse_args()

    grand = {"pass": 0, "fail": 0}
    for p in args.pairs:
        path = os.path.join(args.run_dir, f"pair_{p:02d}_step10_revision_synthesis.md")
        if not os.path.exists(path):
            print(f"\n### PAIR {p:02d} — artifact NOT ON DISK (pending)")
            continue
        # ⛔ IN-FLIGHT GUARD. A pair artifact is only measurable once its agent has
        # STOPPED writing. Measured on this run: pair 05 read 1020 chars mid-write and
        # 964 once settled; pair 03 showed 5 MUSIC PROMPT anchors mid-write and 4 once
        # settled. Both "defects" were the coordinator reading a file under the pen —
        # the same error recorded on 2026-08-09. A stale-by-seconds artifact is
        # indistinguishable from a broken one, so refuse to score it rather than
        # report a finding that will evaporate.
        age = time.time() - os.path.getmtime(path)
        if age < SETTLE_SECONDS:
            print(f"\n### PAIR {p:02d} — IN FLIGHT (last written {age:.0f}s ago, "
                  f"settle threshold {SETTLE_SECONDS}s). NOT SCORED — measuring a file "
                  f"under the pen produces findings that evaporate.")
            continue
        text = io.open(path, encoding="utf-8").read()
        mode, tbl = detect_mode(text)
        if mode is None:
            print(f"\n### PAIR {p:02d} — ERROR: no known heading convention detected. "
                  f"NOT a clean score — extraction failed.")
            grand["fail"] += 1
            continue

        blocks, delim = split_variations(text)
        # L25: assert cardinality BEFORE trusting any measurement.
        if len(blocks) != 4:
            print(f"\n### PAIR {p:02d} — ERROR: extracted {len(blocks)} variation blocks "
                  f"(delimiter mode: {delim}), expected 4. "
                  f"An empty/short extraction is an ERROR, never a score.")
            grand["fail"] += 1
            continue

        report = []
        sizes = {}
        for b in blocks:
            vid = variation_label(b)
            s = check_variation(vid, b, tbl, report)
            if s:
                sizes[vid] = s
        npass = sum(1 for r in report if r[4])
        nfail = len(report) - npass
        grand["pass"] += npass
        grand["fail"] += nfail
        print(f"\n### PAIR {p:02d} — mode={mode} · 4/4 variations extracted · "
              f"{npass} pass / {nfail} fail")
        for vid, s in sizes.items():
            print(f"    {vid}: music {s['music']} · exclude {s['exclude']} · "
                  f"lyrics {s['lyrics']} · sung {s['sung']} · w/l {s['wpl']}")
        for r in report:
            if not r[4]:
                print(f"    ✗ {r[0]} {r[1]}: expected {r[2]}, got {r[3]}")

    print(f"\n=== TOTAL: {grand['pass']} pass, {grand['fail']} fail ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
