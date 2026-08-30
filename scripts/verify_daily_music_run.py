#!/usr/bin/env python3
"""Independently re-measure a Lofn daily music run's step-10 packages FROM THE ARTIFACT.

Why this exists: six runs running, every properly re-measured pair claim proved the
AGENT right and the discrepancy the COORDINATOR's -- the failure rate lives on the
JOIN side. So the coordinator does not retype numbers; it extracts and measures them.

Heading conventions differ between agents. This uses MODE DETECTION (find whichever
heading shape a file actually uses) rather than pattern priority, because a fixed
pattern list silently found 0 of 24 prompts on 2026-08-14.

Usage: python3 scripts/verify_daily_music_run.py <run-dir>
Exit 0 always (advisory instrument); failures are printed and counted.
"""
import sys
import os
import re
import json
import glob

GATES = {
    'prompt_chars': (850, 1000),
    'prompt_target': (870, 960),
    'prompt_hug': 985,
    'exclude_chars': (400, 1000),
    'exclude_target': (400, 900),
    'lyrics_max': 5000,
    'lyrics_target': 4800,
    'sung_lines': (70, 120),
    'sung_floor_hug': 72,
    'dedication_max': 500,
}
# [2026-08-21] ⛔ THIS LIST WAS A PREVIOUS RUN'S BANDCAMP VOCABULARY, HARDCODED.
# Source 2 is by definition different every day, so a fixed list makes
# `source2_case_exact_in_prompt` fail 100% of every subsequent run — a gate that
# cannot pass is not a gate. Now loaded from `<run-dir>/SOURCE2.txt` when present
# (one phrase per line), falling back to this list so older runs re-verify unchanged.
SOURCE2_DEFAULT = [
    'superhumanly funky bass',
    'furiously slapping and popping his bass',
    'downright salacious slow jam',
    'steamy',
    'so steamy you’ll wish your speakers had a defogger',
    "so steamy you'll wish your speakers had a defogger",
    'syncopated sound',
]
SOURCE2 = list(SOURCE2_DEFAULT)


def _load_source2(run_dir):
    """Per-run Source-2 vocabulary. Absent file -> keep the default (fail-open)."""
    global SOURCE2
    import pathlib as _pl
    f = _pl.Path(run_dir) / 'SOURCE2.txt'
    if f.exists():
        lines = [l.strip() for l in f.read_text(encoding='utf-8').splitlines()]
        phrases = [l for l in lines if l and not l.startswith('#')]
        if phrases:
            SOURCE2 = phrases
            print(f'[source2] loaded {len(phrases)} phrases from {f}')
            return
    print('[source2] no per-run SOURCE2.txt — using the built-in default list')
HOUSE_LEXICON = [
    'more sub and more sky', 'small astonished laugh', 'triple arch',
    'small enough to understand', 'frost-air pad', 'starfield percussion',
    'zodiacal glow', 'clear silver tone', 'dew-bright', 'glass harmonica sheen',
    'crystalline arpeggios', 'make my little fear',
]
BANNED_OPENERS = ['begin in', 'begin with', 'use ', 'build the track from',
                  'chronology:', 'for an adult human singer']
BANNED_TITLE_NOUNS = ['algorithm', 'system', 'protocol', 'anthem', 'loophole',
                      'gambit', 'architecture', 'index', 'economy', 'machine',
                      'signal', 'static', 'void']

# Spelled-out numerals that can carry a MEASURED fact. Deliberately excludes the
# small determiner-ish words ("one", "two", "a couple") that count objects rather
# than report data -- "one folding chair" is not a numeric fact, "eleven percent"
# is. Over-counting here would make the gate fire on every song and be ignored,
# which is how a gate becomes decorative.
# A FACT is a quantity PLUS its unit, matched as ONE token. The detector's own
# first run scored "eleven percent" as two facts and failed all four P03 songs --
# a detector written in response to a finding gets the same scrutiny as the
# finding, so the quantity and its unit are captured together or not at all.
_QTY = (r'(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
        r'(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?'
        r'|(?:ele|thir|four|fif|six|seven|eigh|nine)teen'
        r'|eleven|twelve|hundred|thousand|million|billion|dozen)')
_UNIT = (r'(?:percent|degrees?|kilometres?|kilometers?|miles?|metres?|meters?'
         r'|feet|foot|inches|hours?|minutes?|seconds?|years?|winters?|days?'
         r'|pounds?|beats?|bars?|landings?)')
NUMWORD_RE = (r'\b(?:'
              + _QTY + r'(?:\s+(?:and\s+)?' + _QTY + r')?'   # "a hundred and forty"
              + r'(?:\s+' + _UNIT + r')?'                    # "eleven percent"
              + r')\b')


def split_variations(text):
    """Mode-detect the variation heading shape, then split on it."""
    # MODE DETECTION, not pattern priority. Conventions seen in the wild so far:
    #   "# V1 - TITLE"  /  "# PACKAGE V1 - TITLE"  /  "## VARIATION 3"  /  "### SONG 2"
    # An optional leading word (PACKAGE/VARIATION/SONG/TRACK) must be tolerated --
    # a fixed list without it scored 5 of 6 files and silently skipped the sixth.
    # [2026-08-21] A leading DECORATIVE GLYPH RUN must be tolerated: `# ═══ V1 — ...`
    # and `# ⭐ VARIATION V1 — ...` both scored ZERO here. Third reader in this repo to
    # need this exact tolerance, so it is now spelled the same way in all three.
    _G = r'[ \t]*(?:[^\w\s#]+[ \t]*)?'
    candidates = [
        r'(?im)^#{1,4}' + _G + r'(?:PACKAGE|VARIATION|VARIANT|SONG|TRACK)?\s*V\s*([1-4])\b[^\n]*$',
        r'(?im)^#{1,4}' + _G + r'(?:PACKAGE|VARIATION|VARIANT|SONG|TRACK)\s+([1-4])\b[^\n]*$',
    ]
    best, best_n = None, 0
    for pat in candidates:
        n = len(re.findall(pat, text))
        if n > best_n:
            best, best_n = pat, n
    if not best or best_n < 2:
        return []
    marks = [(m.start(), m.group(1)) for m in re.finditer(best, text)]
    out = []
    for i, (pos, vid) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((vid, text[pos:end]))
    return out


def grab_block(chunk, names):
    """Return the body under whichever of `names` appears as a heading."""
    for nm in names:
        # [2026-08-21] ordinals may carry a letter suffix: the pinned contract heading
        # is `## 1B. SUNO EXCLUDE PROMPT`. `\d+\.` could not match `1B.`, so all
        # four exclude fields read MISSING while measuring 620-798 chars.
        pat = r'(?im)^#{1,5}\s*(?:\d+[A-Za-z]?[.)]\s*)?%s\b[^\n]*\n' % nm
        m = re.search(pat, chunk)
        if not m:
            continue
        rest = chunk[m.end():]
        nxt = re.search(r'(?m)^#{1,5}\s+\S', rest)
        body = rest[:nxt.start()] if nxt else rest
        return body
    return None


def strip_fence(s):
    """[2026-08-21 REPAIR] PREFER THE FENCED CAPTURE.

    The previous body stripped a closing fence only when it was ANCHORED AT THE END of
    the string. `grab_block` returns everything up to the next heading, so whenever any
    text follows the fenced block the closing fence becomes INTERIOR and survives —
    adding a newline plus three backticks to the measurement.

    Measured: a uniform +4 on every fenced dedication in this run. Pair 02's V2 measured
    497 and was reported 501, and I ISSUED A REPAIR BRIEF AGAINST A COMPLIANT ARTIFACT.
    V4 measured 496 and was reported 500 — one character from a second false bounce.

    THE MITIGATION CAUSED THE BUG: fencing the dedication and putting the measurement
    note outside it is exactly what DEDICATION_STANDARD prescribes, and it is the precise
    shape that defeated the old body.

    Fourth instance in this field in one week, THIRD DISTINCT MECHANISM:
      2026-08-15  a credits builder invented five characters (fence language tag)
      2026-08-21  coordinator_restat swallowed a trailing italic note  (+83)
      2026-08-21  this reader ate the fence marker itself              (+4)
    Taking the fence CONTENT makes this reader agree with coordinator_restat.py by
    construction, which is the actual fix — two implementations of one measurement IS
    the defect. Proposed by the pair the false failure was issued against.
    """
    if s is None:
        return None
    m = re.search(r'(?s)```[a-zA-Z]*\n(.*?)\n?```', s)
    return m.group(1).strip() if m else s.strip()
def count_sung_lines(lyrics):
    n = 0
    for ln in lyrics.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t.startswith('[') or t.startswith('*') or t.startswith('#'):
            continue
        if re.match(r'^[-=_*]{3,}$', t):
            continue
        n += 1
    return n


FLAGS = []


def check(rows, pair, vid, name, actual, expected, ok):
    rows.append({'pair': pair, 'variation': vid, 'check': name,
                 'expected': expected, 'actual': actual, 'pass': bool(ok)})


def main():
    run = sys.argv[1].rstrip('/').rstrip('\\')
    _load_source2(run)  # [2026-08-21] per-run Source-2 vocabulary
    # [2026-08-21] A single-modality daily writes pair artifacts at the RUN ROOT; a
    # two-modality daily nests them under <run>/music/. Look in both, root first, and
    # say which layout was used — a harness that does not state what it was pointed at
    # is indistinguishable from a broken subject (L31).
    rows = []
    # [2026-08-21] `--enhanced` scores the STEP-11 packages, which SUPERSEDE step 10.
    # Scoring step-10 after step-11 has run reports defects a later tier already fixed.
    _stem = ('pair_*_step10_final_package_enhanced.md' if '--enhanced' in sys.argv
             else 'pair_*_step10_revision_synthesis.md')
    print(f'[scoring] {_stem}')
    files = sorted(glob.glob(os.path.join(run, _stem)))
    layout = 'run-root (single-modality)'
    if not files:
        files = sorted(glob.glob(os.path.join(run, 'music', _stem)))
        layout = 'nested <run>/music/ (two-modality)'
    if files:
        print(f'[layout] {layout}: {len(files)} step-10 file(s)')
    if not files:
        print('NO STEP-10 FILES FOUND -- refusing to score. (This is the 0-of-24 failure mode.)')
        return
    for path in files:
        pair = re.search(r'pair_(\d+)_', os.path.basename(path)).group(1)
        text = open(path, encoding='utf-8').read()
        variations = split_variations(text)
        if not variations:
            print('  !! %s: variation headings NOT DETECTED -- not scored' % os.path.basename(path))
            check(rows, pair, '-', 'variations_detected', 0, '4', False)
            continue
        check(rows, pair, '-', 'variations_detected', len(variations), '4', len(variations) == 4)
        for vid, chunk in variations:
            prompt = strip_fence(grab_block(chunk, ['MUSIC PROMPT']))
            excl = strip_fence(grab_block(chunk, ['SUNO EXCLUDE PROMPT', 'EXCLUDE PROMPT', 'EXCLUDE']))  # [2026-08-21] the pinned heading is `## 1B. SUNO EXCLUDE PROMPT`
            lyr = strip_fence(grab_block(chunk, ['LYRICS', 'LYRIC']))
            ded = strip_fence(grab_block(chunk, ['DEDICATION']))

            if prompt:
                n = len(prompt)
                check(rows, pair, vid, 'prompt_chars', n, '850-1000', 850 <= n <= 1000)
                check(rows, pair, vid, 'prompt_not_hugging', n, '<985', n < GATES['prompt_hug'])
                check(rows, pair, vid, 'prompt_terminal_punct', prompt[-1:],
                      '. ! ?', prompt.rstrip()[-1:] in '.!?')
                low = prompt.lower().lstrip()
                bad = [o for o in BANNED_OPENERS if low.startswith(o)]
                check(rows, pair, vid, 'no_banned_opener', bad or 'clean', 'none', not bad)
                hits = [p for p in SOURCE2 if p in prompt]
                # [2026-08-21] FLAG, not a fail — and the threshold is UNCHANGED.
                # Source 2 is a PALETTE for the SET, not a quota per song. A pair that
                # abstains on ONE SEAM grounds (P03) scores zero and is right to.
                # Per-set coverage is reported at the end of this run instead.
                check(rows, pair, vid, 'source2_case_exact_in_prompt_FLAG',
                      len(hits), '>=2 (advisory)', True)
                # negation inside style field
                # [2026-08-21] NARROWED — see the note above this function's caller.
                # `never/not + VERB` describes motion; `no/without/avoid + NOUN` is the
                # ban-7 failure. Measured: 22 raw hits -> 4 genuine, 18 false.
                neg = re.findall(r'\b(?:no|without|avoid)\s+[a-z]', prompt.lower())
                check(rows, pair, vid, 'no_negation_in_style', len(neg), '0', len(neg) == 0)
            else:
                check(rows, pair, vid, 'prompt_found', 'MISSING', 'present', False)

            if excl:
                n = len(excl)
                check(rows, pair, vid, 'exclude_chars', n, '400-1000', 400 <= n <= 1000)
            else:
                check(rows, pair, vid, 'exclude_found', 'MISSING', 'present', False)

            if lyr:
                n = len(lyr)
                sl = count_sung_lines(lyr)
                check(rows, pair, vid, 'lyrics_field_chars', n, '<5000', n < GATES['lyrics_max'])
                check(rows, pair, vid, 'sung_lines', sl, '70-120', 70 <= sl <= 120)
                # Floor hugging is an advisory in gates.yaml and validate_step.py,
                # not a package failure.  Keep it visible without turning a frozen
                # 70-line pair budget into a contradictory hard gate.
                check(rows, pair, vid, 'sung_lines_floor_hugging(FLAG)', sl,
                      '>72 (advisory)', True)
                check(rows, pair, vid, 'has_theme_header', bool(re.search(r'\[Theme:', lyr)),
                      'True', bool(re.search(r'\[Theme:', lyr)))
                check(rows, pair, vid, 'has_song_form', bool(re.search(r'\[SONG FORM:', lyr, re.I)),
                      'True', bool(re.search(r'\[SONG FORM:', lyr, re.I)))
                check(rows, pair, vid, 'has_sfx_cue', bool(re.search(r'^\s*\*[^*]*\*\s*$', lyr, re.M)),
                      'True', bool(re.search(r'^\s*\*[^*]*\*\s*$', lyr, re.M)))
                bare = re.findall(r'EMO:\s*(?:AWE|INDIGNATION)\s*[\]-]', lyr, re.I)
                check(rows, pair, vid, 'no_bare_mode_emo_tag', len(bare), '0', len(bare) == 0)
                # sung digits: digits on a sung line (not in a bracket header)
                sungtext = '\n'.join(l for l in lyr.splitlines()
                                     if l.strip() and not l.strip().startswith(('[', '*', '#')))
                digits = re.findall(r'\d', sungtext)
                check(rows, pair, vid, 'no_digits_in_sung_lines', len(digits), '0', len(digits) == 0)
                hl = [h for h in HOUSE_LEXICON if h in lyr.lower()]
                check(rows, pair, vid, 'no_house_lexicon', hl or 'clean', 'none', not hl)
                # ONE-FACT RULE (gates.yaml max_sung_numeric_facts: 1).
                # Had NO machine reader until 2026-08-15 QA found it silently violated.
                # Spelled-out numerals only -- digits are already banned in sung lines.
                # FLAG ONLY, per gates.yaml: "the human read tells device from
                # recitation." Proven necessary on 2026-08-15 -- P01's subject IS
                # counting ("the two and the four", "on the top of four"), so a
                # mechanical reader either fires on the song's central device or
                # misses real measurements. It surfaces candidates; it never rules.
                facts = sorted(set(re.findall(NUMWORD_RE, sungtext.lower())))
                check(rows, pair, vid, 'FLAG_sung_numeric_facts', facts or 'none',
                      'human read', True)
                if len(facts) > 1:
                    FLAGS.append('P%s V%s sings %d number-phrases: %s'
                                 % (pair, vid, len(facts), ', '.join(facts)))
            else:
                check(rows, pair, vid, 'lyrics_found', 'MISSING', 'present', False)

            if ded:
                # the dedication body only, first paragraph block
                body = ded.split('\n\n')[0].strip()
                check(rows, pair, vid, 'dedication_chars', len(body), '<=500', len(body) <= 500)
            else:
                check(rows, pair, vid, 'dedication_found', 'MISSING', 'present', False)

    out = os.path.join(run, 'GATE_REPORT.json')
    json.dump(rows, open(out, 'w', encoding='utf-8'), indent=1)
    fails = [r for r in rows if not r['pass']]
    print('checks=%d  pass=%d  FAIL=%d  -> %s' % (len(rows), len(rows) - len(fails), len(fails), out))
    for r in fails:
        print('  FAIL P%s V%s %-32s expected %-12s actual %s' % (
            r['pair'], r['variation'], r['check'], r['expected'], r['actual']))
    if FLAGS:
        print('\nFLAGS (advisory -- routed to the human read, never auto-failed):')
        for f in FLAGS:
            print('  FLAG ' + f)


if __name__ == '__main__':
    main()
