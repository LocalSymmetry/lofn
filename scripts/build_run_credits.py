#!/usr/bin/env python3
"""Build CREDITS.md -- the companion file that `"full credits below"` points at.

`vault/DEDICATION_STANDARD.md` caps a dedication at 500 chars and routes the
overflow to a companion credits file. QA found the phrase shipped in 6 of 24
dedications on 2026-08-15 with NO such file on disk: a promise pointing at
nothing.

⛔ WHY THIS IMPORTS ITS EXTRACTOR RATHER THAN REIMPLEMENTING IT.
The first version of this script measured P05 V2's dedication at 502 chars and
refused to write, reporting an over-cap violation. The verifier measured 497 and
the pair agent measured 497. The dedication sits inside a ```text fenced block:
this script stripped backtick CHARACTERS, which left the language tag "text\\n"
in the string and invented five characters. Two implementations of one
measurement is the defect; a shared one is the fix. Same class as 2026-08-14's
fabricated 611-vs-true-453.

Usage: python3 scripts/build_run_credits.py <run-dir>
"""
import sys
import os
import re
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_daily_music_run import split_variations, grab_block, strip_fence  # noqa: E402

HEADER = """# CREDITS — daily run `{slug}`
**Seed:** {seed} · 6 pairs × 4 variations = 24 songs · LOFN-PRIME throughout.

This file is the referent for the phrase **"full credits below"** in the dedications below.
`vault/DEDICATION_STANDARD.md` caps a dedication at 500 characters and routes the overflow to a
companion credits file — **the file has to exist, or the phrase is a promise pointing at nothing.**

⛔ **Honesty clause, run-wide:** Suno *generates*; it does not sample. Where a figure is named it is
because **we asked for that figure's invention by name in the style field, so we name them** — never
because any recording of theirs is present. No line here is an endorsement by any living person.

⭐ **THE CREDIT FOLLOWS THE FIELD** (derived on this run by four pairs ruling independently, in
isolation): a technique **raised in the STYLE field owes its player**; the same technique **banned in
EXCLUDE owes nobody.** The credit follows which field the token is in, not whether the words appear.

*Extracted from the step-10 artifacts by script. Nothing on this page was retyped.*

---

"""


def title_of(chunk):
    head = chunk.splitlines()[0]
    t = re.sub(r'^#+\s*(?:PACKAGE|VARIATION|SONG|TRACK)?\s*V\s*\d\s*[—\-–:]*\s*', '', head)
    return t.strip().strip('*"“”· ')


def main():
    run = sys.argv[1].rstrip('/').rstrip('\\')
    music = os.path.join(run, 'music')
    out = [HEADER.format(slug=os.path.basename(run),
                         seed='THE LIGHT IS THE DEBRIS')]
    total, over = 0, []
    for path in sorted(glob.glob(os.path.join(music, 'pair_*_step10_revision_synthesis.md'))):
        pair = re.search(r'pair_(\d+)_', os.path.basename(path)).group(1)
        text = open(path, encoding='utf-8').read()
        out.append('## P%s\n' % pair)
        for vid, chunk in split_variations(text):
            ded = strip_fence(grab_block(chunk, ['DEDICATION']))
            if ded is None:
                out.append('### V%s — %s\n\n*(no dedication block found)*\n' % (vid, title_of(chunk)))
                continue
            body = ded.split('\n\n')[0].strip()
            if len(body) > 500:
                over.append('P%sV%s=%d' % (pair, vid, len(body)))
            out.append('### V%s — %s\n**%d chars**\n\n> %s\n'
                       % (vid, title_of(chunk), len(body), body.replace('\n', '\n> ')))
            total += 1
        out.append('')
    if over:
        raise SystemExit('REFUSING TO WRITE — dedications over the 500 cap: ' + ', '.join(over))
    dest = os.path.join(music, 'CREDITS.md')
    open(dest, 'w', encoding='utf-8').write('\n'.join(out))
    print('CREDITS.md: %d dedications, all <=500 chars, %d bytes' % (total, os.path.getsize(dest)))


if __name__ == '__main__':
    main()
