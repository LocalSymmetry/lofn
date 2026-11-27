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
# ⛔ 2026-09-13: words-per-line was ALSO being re-derived inline (len(x.split())), and the
# two implementations disagreed by up to 0.16 on the same four songs (Pair 01 measured it).
# One implementation per measurement — the same rule that produced the rhyme-window fix.
from measure_soundcraft import words_per_line as _canonical_wpl

# Seconds a pair artifact must have been untouched before it may be scored.
SETTLE_SECONDS = 90

def _read_gates_yaml(path=None):
    """Tiny-YAML subset over vault/gates.yaml: `key: N` and `key: [A, B]`. No dependency,
    no eval. Returns {} if the file cannot be read -- callers fall back and SAY SO.

    2026-10-08 (run 2026-10-08_lofn-on-three-successor): `path` lets `--gates PATH` point this
    reader at an explicitly declared run-local gates file, mirroring validate_step.py and
    check_brief_line_budget.py. Without it a run-local form exception was invisible here: a
    declared LINE BUDGET of 54 under a run band of 44-96 was clamped to the house band and read
    as an impossible 70-60, failing every correct song -- and the repair an agent reaches for is
    to pad the lyric. A serializer's block-list form (`key:` then `- N` lines) is read too."""
    # 2026-10-01 (run 2026-09-30-caterpillar-survivors, found by pairs 02 and 04): keys are
    # read at COLUMN 0 only. The 2026-09-30 `instrumental:` block nests its own
    # `suno_exclude_chars: [40, 300]`, and the old `\s*` prefix let that nested value
    # overwrite the top-level sung-song band [400, 900], so every sung package failed its
    # exclude row. Every key this reader uses is top-level.
    out = {}
    try:
        with open(path or os.path.join("vault", "gates.yaml"), encoding="utf-8") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines):
            m = re.match(r"([a-z_0-9]+):\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", line)
            if m:
                out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
                continue
            m = re.match(r"([a-z_0-9]+):\s*(true|false)\s*(?:#.*)?$", line, re.I)
            if m:
                out[m.group(1)] = (m.group(2).lower() == "true")
                continue
            m = re.match(r"(length_gate):\s*(words|lines)\s*(?:#.*)?$", line, re.I)
            if m:  # V6-19 (2026-11-25): which measure is pass/fail
                out[m.group(1)] = m.group(2).lower()
                continue
            m = re.match(r"([a-z_0-9]+):\s*(\d+(?:\.\d+)?)\s*(?:#.*)?$", line)
            if m:
                v = m.group(2)
                out[m.group(1)] = float(v) if "." in v else int(v)
                continue
            m = re.match(r"([a-z_0-9]+):\s*(?:#.*)?$", line)
            if m:  # block list: exactly two `- N` items on the following lines
                nums = []
                for nxt in lines[i + 1:i + 3]:
                    mm = re.match(r"-\s+(\d+)\s*(?:#.*)?$", nxt.strip())
                    if not mm:
                        break
                    nums.append(int(mm.group(1)))
                if len(nums) == 2:
                    out[m.group(1)] = (nums[0], nums[1])
    except Exception:
        pass
    return out


_YAML = _read_gates_yaml()
_GATE_SOURCE = {}


def _g(yaml_key, fallback, gate_name):
    """Resolve one threshold from gates.yaml, recording whether it was derived."""
    if yaml_key in _YAML:
        _GATE_SOURCE[gate_name] = ("derived", yaml_key)
        return _YAML[yaml_key]
    _GATE_SOURCE[gate_name] = ("FALLBACK", yaml_key + " MISSING from gates.yaml")
    return fallback


# ⛔ DERIVED, NOT HARDCODED (2026-09-13). These were literals that happened to equal
# gates.yaml by hand-maintained coincidence -- which is exactly how sung_lines_run_target
# drifted to a dead run's number. EXECUTION.md §4: gates.yaml is "the single source of
# numeric truth", and a prose-vs-YAML disagreement is a meta-check the harness must surface.
def _build_gates():
    return {
    "music_prompt_chars": _g("music_prompt_chars", (850, 1000), "music_prompt_chars"),
    "music_prompt_target": _g("music_prompt_chars_target", (870, 960), "music_prompt_target"),
    "music_prompt_hug_ceiling": _g("music_prompt_hug_ceiling", 985, "music_prompt_hug"),
    "exclude_chars": _g("suno_exclude_chars", (400, 900), "exclude_chars"),
    "lyrics_field_max": _g("suno_lyrics_field_max", 5000, "lyrics_field_chars"),
    "lyrics_field_target": _g("suno_lyrics_field_target", 3800, "lyrics_field_target"),
    # V6-19 (2026-11-25): `length_gate: words` makes sung words pass/fail and the line
    # count a FLAG. A gates file that declares `sung_lines` and no `length_gate` was
    # written when lines were the gate, and is read that way (same rule as validate_step.py).
    "length_gate": _YAML.get("length_gate") or ("lines" if "sung_lines" in _YAML else "words"),
    "sung_lines_band": _g("sung_lines", (40, 100), "sung_lines_band"),
    "sung_words_band": _g("sung_words_band", (340, 460), "sung_words_band"),
    "sung_words_hard": (_g("sung_words_hard_floor", 300, "sung_words"),
                        _g("sung_words_hard_ceiling", 520, "sung_words_ceiling")),
    "lyrics_overhead_flag_chars": _g("lyrics_overhead_flag_chars", 1300, "lyrics_overhead_chars"),
    "lyrics_overhead_flag_pct": _g("lyrics_overhead_flag_pct", 40.0, "lyrics_overhead_pct"),
    "disc_channel_in_lyrics_box": _g("disc_channel_in_lyrics_box", False, "disc_channel_placement"),
    # REPAIRED 2026-09-13. This read (88, 110), commented "this run's upper-half
    # mandate" -- a RUN-LOCAL number from a dead run, hardcoded into a SHARED reader, and
    # the SIXTH instance of the defect class documented eight lines below. Against run
    # `music-daily-2026-09-13` (declared budgets 80/79/80/88/92/79) it would have reported
    # FAIL on four of six correct pairs, and the repair an agent reaches for is to pad the
    # song until the scanner goes quiet (2026-09-11: a pair added an outro coda to clear a
    # number that did not apply to it). Now DERIVED, never hardcoded -- see
    # resolve_sung_target() below.
    "sung_lines_run_target": None,
    "mean_words_per_line_ceiling": _g("mean_words_per_line_ceiling", 6.8,
                                      "mean_words_per_line"),
}


GATES = _build_gates()
_GATES_PATH_LABEL = None  # set by --gates; None keeps the default output byte-identical

# Resolved per pair in main(); read by check_variation(). (lo, hi, source_label)
_SUNG_TARGET = ((56, 72), "vault/gates.yaml sung_lines_target (default)")


def _gates_yaml_target(default=(56, 72)):
    """Read sung_lines_target from the loaded gates (vault/gates.yaml, or the file given with
    --gates) -- the single source of numeric truth. 2026-10-08: reads the already-parsed _YAML,
    so a run-local file's target is honoured; it used to re-open vault/gates.yaml directly."""
    v = _YAML.get("sung_lines_target")
    return v if isinstance(v, tuple) and len(v) == 2 else default


def resolve_sung_target(run_dir, pair_no):
    """-> ((lo, hi), source_label).

    The pair's OWN declared LINE BUDGET is the contract and wins. gates.yaml backs it up.
    A tolerance of +/-6 lines lets an intentional near-miss read as a near-miss rather than
    a failure, and the result is still clamped inside the hard band so a declaration can
    never widen the gate beyond doctrine.
    """
    brief = os.path.join(run_dir, "pair_%02d_step05_isolated_brief.md" % pair_no)
    band = _gates_yaml_target()
    if os.path.exists(brief):
        try:
            with open(brief, encoding="utf-8") as fh:
                txt = fh.read()
            m = re.search(r"LINE BUDGET:[^=\n]*=\s*(\d+)", txt)
            if m:
                t = int(m.group(1))
                hard_lo, hard_hi = GATES["sung_lines_band"]
                lo = max(t - 6, hard_lo)
                hi = min(t + 6, hard_hi)
                return (lo, hi), "declared LINE BUDGET %d +/-6 (hard band %d-%d)" % (
                    t, hard_lo, hard_hi)
        except Exception:
            pass
    return band, "vault/gates.yaml sung_lines_target"
# ⛔ SEVERITY MAP, added 2026-09-13. vault/gates.yaml is explicit that part of this
# block is advisory: "These are FLAGs (pass=None), never hard fails: a deliberate
# through-composed piece may sit below them, but it must say so on purpose." This reader
# rendered every one of them as a binary ✗ and counted it as a failure — so a pair whose
# step-05 brief MANDATED long verse lines against short choruses was reported as four
# hard failures for executing its own assignment (Pair 03, this run). A checker that
# cannot say "flag" can only say "fail", and the repair an agent reaches for is to
# deform the song until the number moves.
#
# A FLAG is surfaced, never counted against the pair. A HARD gate still fails closed.
FLAG_CHECKS = {
    "music_prompt_hug",      # gates.yaml: "measured chars >= this -> FLAG boundary_hugging"
    "lyrics_field_target",   # a TARGET; `lyrics_field_chars` is the hard cap and stays hard
    "sung_words_band",       # V6-19: FLAG outside 340-460 unless form-declared in RUN_GATES.yaml
    "sung_lines_advisory",   # V6-19: the line count is a form choice; it does not set length
    "lyrics_overhead_chars", # V6-18: FLAG non-sung bracket overhead > 1300c
    "lyrics_overhead_pct",   # V6-18: FLAG non-sung bracket overhead > 40%
    "disc_channel_placement",# V6-18: FLAG [Disc_*] inside pasted Suno lyrics when default sidecar
    "mean_words_per_line",   # gates.yaml: FLAGs, never hard fails
    "house_lexicon",         # doctrine: "a hit FLAGs to the Somatic read"
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

    # ⛔ INFO-STRING PARITY, 2026-09-20. This read `[a-zA-Z]*` -- letters only. The very
    # same defect was found and fixed in `check_field_census.py` on 2026-09-13, and THE FIX
    # NEVER PROPAGATED BACK TO THE PRIMARY SCORER. Our own frozen ICB carries ```tab-delim
    # fences and `tab-delim` has a HYPHEN, so such a payload is invisible here: the fence
    # branch misses, the function silently falls through to BARE-PROSE mode, and the fence
    # markers themselves are returned as part of the payload. A CommonMark info string is
    # letters, digits, _ + - . -- match that, not a subset.
    fence = re.search(r"```[A-Za-z0-9_+.-]*\n(.*?)\n```", slice_, re.S)
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


def split_variations(text, expect=4):
    """Split a package file into its variation blocks by DETECTING the delimiter.

    ⛔ `expect` ADDED 2026-09-20, AND THE FIRST VERSION OF IT WAS ALSO WRONG.
    The cardinality 4 was a CONSTANT here, so `--enhanced` failed on every step-11 package
    it was added to read (2026-09-13): "extracted 0 variation blocks, expected 4" on a
    well-formed artifact. My first repair swapped it for `1 if enhanced else 4` -- then a
    census of all 519 `*_enhanced.md` files on disk said **156 of them carry FOUR packages
    and 53 carry one.** "Enhanced means one" is not a fact about the tier; it would have
    broken 156 files to fix 53. One wrong constant for another.

    So: `expect=N` ASSERTS a count the contract genuinely pins (step 10 is four, and the
    dispatch says so). `expect=None` DERIVES the count from the file and asserts only what
    is actually an integrity property -- at least one anchor, and every anchor resolving to
    its own level-1 block. That is what L25 was ever protecting: not a house number, but
    the guarantee that no measurement is taken from a half-extracted file.

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
    if not anchors:
        return [], f"0 `{music_h}` anchors found"
    if expect is not None and len(anchors) != expect:
        return [], f"{len(anchors)} `{music_h}` anchors found, expected {expect}"
    want = expect if expect is not None else len(anchors)
    h1 = [m.start() for m in re.finditer(r"^#\s+\S", text, flags=re.M)]
    starts = []
    for a in anchors:
        prior = [h for h in h1 if h < a]
        starts.append(prior[-1] if prior else a)
    starts = sorted(set(starts))
    if len(starts) != want:
        return [], f"{len(starts)} distinct level-1 blocks for {want} anchors"
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
    wpl = _canonical_wpl(lines) if n else 0.0   # canonical, never re-derived

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
    sung_words = sum(len(ln.split()) for ln in lines)
    if GATES["length_gate"] == "lines":
        (lo, hi), _sung_src = _SUNG_TARGET
        report.append((vid, "sung_lines", f"{lo}-{hi}", n, lo <= n <= hi))
    else:
        hlo, hhi = GATES["sung_words_hard"]
        report.append((vid, "sung_words", f"{hlo}-{hhi}", sung_words, hlo <= sung_words <= hhi))
        lo, hi = GATES["sung_lines_band"]
        report.append((vid, "sung_lines_advisory", f"{lo}-{hi}", n, lo <= n <= hi))
    wlo, whi = GATES["sung_words_band"]
    report.append((vid, "sung_words_band", f"{wlo}-{whi}", sung_words, wlo <= sung_words <= whi))
    report.append((vid, "mean_words_per_line", f"<={GATES['mean_words_per_line_ceiling']}",
                   round(wpl, 2), wpl <= GATES["mean_words_per_line_ceiling"]))
    sung_chars = sum(len(ln) for ln in lines)
    ovh_chars = max(0, l_len - sung_chars)
    ovh_pct = round(100.0 * ovh_chars / l_len, 1) if l_len else 0.0
    report.append((vid, "lyrics_overhead_chars", f"<={GATES['lyrics_overhead_flag_chars']}",
                   ovh_chars, ovh_chars <= GATES["lyrics_overhead_flag_chars"]))
    report.append((vid, "lyrics_overhead_pct", f"<={GATES['lyrics_overhead_flag_pct']}%",
                   ovh_pct, ovh_pct <= GATES["lyrics_overhead_flag_pct"]))
    has_disc = bool(re.search(r"^\[Disc_", lyrics, re.M | re.I))
    disc_ok = True if GATES["disc_channel_in_lyrics_box"] else (not has_disc)
    report.append((vid, "disc_channel_placement",
                   "allowed" if GATES["disc_channel_in_lyrics_box"] else "in Production Sidecar (not lyrics box)",
                   "present in lyrics" if has_disc else "clean", disc_ok))

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
    # ⛔ 2026-09-13: the CEILING had no reader. The step-11 contract fails a package with
    # more than three standalone *SFX* cues; this file only ever tested >=1, and Pair 01
    # shipped FIVE per song with nothing flagged. A gate with a floor and no ceiling is
    # half a gate, and the half that is missing is the one that catches excess.
    _sfx_n = sum(1 for x in lyrics.split("\n")
                 if x.strip().startswith("*") and x.strip().endswith("*"))
    report.append((vid, "sfx_ceiling", "<=3", _sfx_n, _sfx_n <= 3))
    report.append((vid, "sfx_cue", ">=1",
                   sum(1 for x in lyrics.split("\n")
                       if x.strip().startswith("*") and x.strip().endswith("*")),
                   any(x.strip().startswith("*") and x.strip().endswith("*")
                       for x in lyrics.split("\n"))))
    bare = re.findall(r"EMO:\s*(AWE|INDIGNATION)\s*[\]-]", lyrics, re.I)
    report.append((vid, "bare_emo_tag", "none", bare or "none", not bare))
    return {"music": m_len, "exclude": e_len, "lyrics": l_len,
            "sung": n, "wpl": round(wpl, 2)}


# ⛔ THRESHOLD PROVENANCE, added 2026-09-13 (promoted from Pair 03's correction).
# Three defects in this file in one run were all the SAME shape: a value that had lost its
# source. A number whose origin is not printed cannot be audited by the person reading the
# output, so a stale constant and a correct one look identical. Every threshold this reader
# enforces now declares WHERE IT CAME FROM and AT WHAT SEVERITY, on every run.
GATE_PROVENANCE = {
    "music_prompt_chars":          ("vault/gates.yaml music_prompt_chars",        "HARD"),
    "music_prompt_hug":            ("vault/gates.yaml music_prompt_hug_ceiling",  "FLAG"),
    "music_prompt_terminal_punct": ("gates.yaml music_prompt_terminal_punctuation", "HARD"),
    "banned_opener":               ("lofn-music SKILL.md banned-opener list",     "HARD"),
    "exclude_chars":               ("vault/gates.yaml suno_exclude_chars",        "HARD"),
    "lyrics_field_chars":          ("vault/gates.yaml suno_lyrics_field_max",     "HARD"),
    "lyrics_field_target":         ("vault/gates.yaml suno_lyrics_field_target",  "FLAG"),
    "sung_lines":                  ("pair's declared LINE BUDGET, else gates.yaml (only under length_gate: lines)", "HARD"),
    "sung_words":                  ("vault/gates.yaml sung_words_hard_floor / _ceiling", "HARD"),
    "sung_lines_advisory":         ("vault/gates.yaml sung_lines (advisory range)",    "FLAG"),
    "sung_words_band":             ("vault/gates.yaml sung_words_band",            "FLAG"),
    "mean_words_per_line":         ("vault/gates.yaml mean_words_per_line_ceiling", "FLAG"),
    "lyrics_overhead_chars":       ("vault/gates.yaml lyrics_overhead_flag_chars", "FLAG"),
    "lyrics_overhead_pct":         ("vault/gates.yaml lyrics_overhead_flag_pct",   "FLAG"),
    "disc_channel_placement":      ("vault/gates.yaml disc_channel_in_lyrics_box", "FLAG"),
    "house_lexicon":               ("vault/gates.yaml house_lexicon",             "FLAG"),
    "forbidden_sung_number":       ("run-scoped, empty by default",               "HARD"),
    "apology_vocabulary":          ("run-scoped APOLOGY list",                    "HARD"),
    "sfx_cue":                     ("lofn-music SKILL.md: >=1 standalone *SFX*",  "HARD"),
    "sfx_ceiling":                 ("step-11 contract: >3 standalone *SFX* = FAIL", "HARD"),
    "bare_emo_tag":                ("EXECUTION.md §4: never bare AWE/INDIGNATION", "HARD"),
    "anchor_phrase":               ("run-scoped ANCHOR_STEMS, empty by default",  "HARD"),
    "EXTRACTION":                  ("L25: an empty extraction is an ERROR",       "HARD"),
}


def print_provenance():
    print("THRESHOLD PROVENANCE — every number this reader enforces, and its source")
    print("  (a value whose origin is not printed cannot be audited; all three defects")
    print("   repaired in this file on 2026-09-13 were values that had lost their source)")
    n_fb = 0
    for name in sorted(GATE_PROVENANCE):
        src, sev = GATE_PROVENANCE[name]
        how, key = _GATE_SOURCE.get(name, ("n/a", ""))
        if how == "FALLBACK":
            n_fb += 1
        tag = {"derived": "[derived]", "FALLBACK": "[⚠ FALLBACK]", "n/a": ""}[how]
        if _GATES_PATH_LABEL and how == "derived":
            src = src.replace("vault/gates.yaml", _GATES_PATH_LABEL)  # 2026-10-08: name the file actually read
        print(f"    {sev:4s}  {name:28s} <- {src} {tag}")
    if n_fb:
        print(f"    ⚠ {n_fb} threshold(s) fell back to a literal because the key is MISSING")
        print("      from vault/gates.yaml. A fallback is a value with no live source —")
        print("      reconcile it rather than letting it drift silently.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pairs", nargs="*", type=int, default=list(range(1, 7)))
    ap.add_argument("--enhanced", action="store_true",
                    help="score the step-11 enhanced package when present (the tier that ships)")
    ap.add_argument("--quiet-provenance", action="store_true",
                    help="suppress the threshold-provenance header (not recommended)")
    ap.add_argument("--gates", default=None,
                    help="an explicitly declared run-local gates file (as validate_step.py --gates)")
    args = ap.parse_args()
    if args.gates:
        global _YAML, GATES, _GATES_PATH_LABEL
        if not os.path.isfile(args.gates):
            print("FAIL: --gates file not found: " + args.gates)
            return 2
        _GATES_PATH_LABEL = args.gates.replace("\\", "/")
        _YAML = _read_gates_yaml(args.gates)
        _GATE_SOURCE.clear()
        GATES = _build_gates()
        print("RUN-LOCAL GATES: " + args.gates + "  length_gate %s, sung_words hard %s, band %s, sung_lines %s" % (
            GATES["length_gate"], GATES["sung_words_hard"], GATES["sung_words_band"], GATES["sung_lines_band"]))

    grand = {"pass": 0, "fail": 0, "unscored": 0}
    if not args.quiet_provenance:
        print_provenance()
    for p in args.pairs:
        # ⛔ 2026-09-13: this name was hardcoded, so the house re-stat could not see a
        # step-11 ENHANCED package — the tier that actually ships. Named, not assumed.
        _cands = ([f"pair_{p:02d}_step10_final_package_enhanced.md"] if args.enhanced else []) + \
                 [f"pair_{p:02d}_step10_revision_synthesis.md"]
        path = None
        for _c in _cands:
            _fp = os.path.join(args.run_dir, _c)
            if os.path.exists(_fp):
                path = _fp
                break
        path = path or os.path.join(args.run_dir, _cands[-1])
        _scored_name = os.path.basename(path)
        global _SUNG_TARGET
        _SUNG_TARGET = resolve_sung_target(args.run_dir, p)
        if not os.path.exists(path):
            print(f"\n### PAIR {p:02d} — artifact NOT ON DISK (pending)"
                  f"   [sung target would be {_SUNG_TARGET[0][0]}-{_SUNG_TARGET[0][1]}"
                  f" from {_SUNG_TARGET[1]}]")
            grand["unscored"] += 1
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
            grand["unscored"] += 1
            continue
        text = io.open(path, encoding="utf-8").read()
        # ⭐ JSON MODE, added 2026-10-08 (run 2026-10-07_lofn-fast-surprise). When a step-10
        # file's `## 4. Complete Step Output` is the JSON contract that scripts/validate_step.py
        # prefers (exactly two revised + two synthesized objects, four keys each), read the four
        # candidates THROUGH THE VALIDATOR'S OWN PARSER and hand each one to the unchanged
        # check_variation() in the canonical heading shape. Before this, the reader looked only
        # for prose variation blocks and extracted 0 of 4 from two correctly formed JSON-first
        # artifacts (P04, P06) — the L25 error fired correctly, and the fix belongs to the reader,
        # never to the artifact. One implementation of the step-10 contract: the validator's.
        _json_items = None
        if not _scored_name.endswith("_enhanced.md"):
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import validate_step as _vs
                _json_items = _vs._step10_json_items(text)
            except Exception as _exc:  # a malformed JSON section is reported, never scored
                print(f"\n### PAIR {p:02d} — step-10 JSON present but unreadable by the validator's "
                      f"parser: {_exc}. NOT a clean score.")
                grand["fail"] += 1
                continue
        if _json_items is not None:
            if len(_json_items) != 4:
                print(f"\n### PAIR {p:02d} — ERROR: step-10 JSON holds {len(_json_items)} candidates, "
                      f"expected 4. An empty/short extraction is an ERROR, never a score.")
                grand["fail"] += 1
                continue
            tbl = {"music": "## 1. MUSIC PROMPT", "exclude": "## 1B. SUNO EXCLUDE PROMPT",
                   "lyrics": "## 2. LYRICS", "title": "## 3. TITLE"}
            blocks = [
                f"# VARIATION V{i}\n\n{tbl['music']}\n\n```text\n{it['music_prompt']}\n```\n\n"
                f"{tbl['exclude']}\n\n```text\n{it['exclude_prompt']}\n```\n\n"
                f"{tbl['lyrics']}\n\n```text\n{it['lyrics_prompt']}\n```\n\n{tbl['title']}\n\n{it['title']}\n"
                for i, it in enumerate(_json_items, 1)]
            mode, delim = "step10_json", "step-10 JSON via validate_step._step10_json_items"
            _pinned = 4
            report = []
            sizes = {}
            for b in blocks:
                vid = variation_label(b)
                s = check_variation(vid, b, tbl, report)
                if s:
                    sizes[vid] = s
            flagged = [r for r in report if (not r[4]) and r[1] in FLAG_CHECKS]
            hard = [r for r in report if r[1] not in FLAG_CHECKS]
            npass = sum(1 for r in hard if r[4])
            nfail = len(hard) - npass
            grand["pass"] += npass
            grand["fail"] += nfail
            grand["flag"] = grand.get("flag", 0) + len(flagged)
            print(f"\n### PAIR {p:02d} [{_scored_name}] — mode={mode} · {len(blocks)}/4 variations "
                  f"extracted ({delim}) · {npass} pass / {nfail} fail / {len(flagged)} flag")
            for vid, s in sizes.items():
                print(f"    {vid}: music {s['music']} · exclude {s['exclude']} · "
                      f"lyrics {s['lyrics']} · sung {s['sung']} · w/l {s['wpl']}")
            for r in report:
                if (not r[4]) and r[1] in FLAG_CHECKS:
                    print(f"    ⚠ {r[0]} {r[1]}: advisory — expected {r[2]}, got {r[3]}"
                          f"   [FLAG, not a failure: gates.yaml]")
            for r in report:
                if (not r[4]) and r[1] not in FLAG_CHECKS:
                    print(f"    ✗ {r[0]} {r[1]}: expected {r[2]}, got {r[3]}")
            continue
        mode, tbl = detect_mode(text)
        if mode is None:
            print(f"\n### PAIR {p:02d} — ERROR: no known heading convention detected. "
                  f"NOT a clean score — extraction failed.")
            grand["fail"] += 1
            continue

        # ASSERT where the contract pins the number; DERIVE where it does not.
        # The step-10 synthesis contract is four packages and the dispatch brief says so,
        # so 4 is asserted. A step-11 enhanced package has no such fixed count on disk
        # (156 carry four, 53 carry one), so its count is read from the file and only its
        # INTERNAL consistency is asserted. Never the flag -- `--enhanced` may have fallen
        # back to the step-10 file, and then the step-10 contract applies again.
        _pinned = None if _scored_name.endswith("_enhanced.md") else 4
        blocks, delim = split_variations(text, expect=_pinned)
        # L25: assert cardinality BEFORE trusting any measurement.
        if not blocks or (_pinned is not None and len(blocks) != _pinned):
            print(f"\n### PAIR {p:02d} — ERROR: extracted {len(blocks)} variation blocks "
                  f"(delimiter mode: {delim}), expected "
                  f"{_pinned if _pinned is not None else 'a self-consistent count'} "
                  f"for {_scored_name}. "
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
        flagged = [r for r in report if (not r[4]) and r[1] in FLAG_CHECKS]
        hard = [r for r in report if r[1] not in FLAG_CHECKS]
        npass = sum(1 for r in hard if r[4])
        nfail = len(hard) - npass
        nflag = len(flagged)
        grand["pass"] += npass
        grand["fail"] += nfail
        grand["flag"] = grand.get("flag", 0) + nflag
        # ⛔ THIS SAID `4/4` AS A LITERAL (found 2026-09-20). The reassuring line was the
        # one that lied: a step-11 package holding ONE variation scored fine and reported
        # "4/4 variations extracted". Third hardcoded 4 in this file, and the only one that
        # printed a number nobody had measured. A reader must never state a count it did
        # not count.
        print(f"\n### PAIR {p:02d} [{_scored_name}] — mode={mode} · "
              f"{len(blocks)}/{_pinned if _pinned is not None else len(blocks)} "
              f"variations extracted · "
              f"{npass} pass / {nfail} fail / {nflag} flag"
              + ("   (flags are advisory: gates.yaml says FLAG, never hard fail)"
                 if nflag else ""))
        for vid, s in sizes.items():
            print(f"    {vid}: music {s['music']} · exclude {s['exclude']} · "
                  f"lyrics {s['lyrics']} · sung {s['sung']} · w/l {s['wpl']}")
        for r in report:
            if (not r[4]) and r[1] in FLAG_CHECKS:
                print(f"    \u26a0 {r[0]} {r[1]}: advisory — expected {r[2]}, got {r[3]}"
                      f"   [FLAG, not a failure: gates.yaml]")
        for r in report:
            if (not r[4]) and r[1] not in FLAG_CHECKS:
                print(f"    ✗ {r[0]} {r[1]}: expected {r[2]}, got {r[3]}")

    # ⛔ THE EXIT CODE USED TO BE A CONSTANT. `return 0` fired whether the run scored 264/264,
    # failed a dozen checks, or silently skipped a whole pair — found 2026-09-14 when this printed
    # `exit=0` with 220 pass and PAIR 06 unscored behind the in-flight guard. Every reader in this
    # house has now produced the same defect: a confident green over ground it never covered.
    # COVERAGE IS PART OF THE VERDICT. A pair that was not scored is not a pair that passed.
    scored = len(args.pairs) - grand["unscored"]
    print(f"\n=== TOTAL: {grand['pass']} pass, {grand['fail']} fail "
          f"| coverage {scored}/{len(args.pairs)} pairs scored ===")
    if grand["unscored"]:
        print(f"  ⛔ {grand['unscored']} pair(s) NOT SCORED — this is NOT a clean run. "
              f"Re-run once they settle or land on disk.")
        return 4
    if grand["fail"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
