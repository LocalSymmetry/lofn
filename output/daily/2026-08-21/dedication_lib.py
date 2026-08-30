#!/usr/bin/env python3
"""⭐ THE ONE IMPLEMENTATION of the dedication measurement.

WHY THIS MODULE EXISTS
----------------------
`vault/RUN_LEDGER.md`, 2026-08-15:

    "THE FIX WAS NOT THE REGEX — TWO IMPLEMENTATIONS OF ONE MEASUREMENT IS THE DEFECT."

Four instances in eight days, in this one field, by three distinct mechanisms, and
EVERY ONE OVER-COUNTED — which matters because an over-count always bounces a
compliant artifact, never a broken one:

    2026-08-14  a reader fabricated 158 chars (611 reported vs 453 true)
    2026-08-15  a credits builder fabricated   5 chars — stripped backtick characters
                but left the fence's language tag behind
    2026-08-21  coordinator_restat.py fabricated 83 chars — swallowed the trailing
                italic measurement note and the section rule that followed it
    2026-08-21  verify_daily_music_run.py fabricated 4 chars — strip_fence() removed a
                closing fence only when ANCHORED AT END, so any text after the block
                left the fence interior and surviving

⭐ THE 2026-08-21 INSTANCE IS THE ONE THAT PROVES THE RULE, because THE MITIGATION
CAUSED THE BUG: fencing the dedication and putting the measurement note outside it is
exactly what DEDICATION_STANDARD prescribes — and it is the precise shape that defeated
the reader. Diagnosed by Pair 02, against whose compliant artifact the false failure had
been issued, and whose generalisation is better than the rule it was given:

    "Each reader was correct in isolation and only disagreed AT THE JOIN — the same
     shape as the scratchpad collision. Both are join-side failures, and both were
     invisible until two instruments were pointed at one object and made to agree."

So: every reader in this run imports THIS. Nobody re-implements it.
"""
import re

CAP = 500  # vault/DEDICATION_STANDARD.md


def grab_dedication_block(chunk):
    """Return the raw section body under a DEDICATION heading, or ''.

    Tolerates a leading decorative glyph run and a letter-suffixed ordinal, because
    four different heading shapes appeared in one run:
        `## DEDICATION` · `### Dedication - 489 chars`
        `### * DEDICATION *(measured: 447 characters)*` · `## 4. DEDICATION *(<=500)*`
    A reader without that tolerance measured two of four sets and SILENTLY SKIPPED the
    other two: no flag, no finding, no row. A gate that cannot fail is not a gate.
    """
    m = re.search(r"^#{1,4}[ \t]*(?:[^\w\s#]+[ \t]*)?(?:\d+[A-Za-z]?[.)][ \t]*)?"
                  r"DEDICATION.*?$", chunk, re.M | re.I)
    if not m:
        return ""
    rest = chunk[m.end():]
    nxt = re.search(r"^#{1,4}\s", rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def measure_dedication(chunk):
    """The dedication text, cleaned. `len()` of this is THE character count.

    A dedication is the PARAGRAPH. A fence, a trailing italic note, a self-reported
    count in the heading and a section rule are all ABOUT the dedication and are not
    part of it.
    """
    body = grab_dedication_block(chunk)
    if not body:
        return ""
    # Prefer the fenced CONTENT when a fence is present, wherever it sits in the body.
    # `strip-if-anchored-at-end` is the 2026-08-21 +4 defect; do not reintroduce it.
    fenced = re.search(r"(?s)```[a-zA-Z]*\n(.*?)\n?```", body)
    if fenced:
        return fenced.group(1).strip()
    body = re.sub(r"\n\s*-{3,}\s*$", "", body.strip()).strip()          # trailing rule
    body = re.sub(r"\n\s*\*\([^)]*\)\*\s*$", "", body).strip()          # trailing note
    body = re.sub(r"^\s*\*\([^)]*\)\*\s*\n", "", body).strip()          # leading note
    body = re.sub(r"^>\s?", "", body, flags=re.M).strip()               # blockquote marks
    return body


def has_dedication_heading(chunk):
    """True when a heading is PRESENT. An empty measurement against a present heading
    is a HARD ERROR, never a silent skip."""
    return bool(re.search(r"^#{1,4}[^\n]*DEDICATION", chunk, re.M | re.I))


def check(chunk):
    """-> (text, chars, verdict). verdict in {ok, OVER_CAP, EMPTY_AGAINST_HEADING, absent}."""
    text = measure_dedication(chunk)
    if not text:
        return "", 0, ("EMPTY_AGAINST_HEADING" if has_dedication_heading(chunk) else "absent")
    return text, len(text), ("OVER_CAP" if len(text) > CAP else "ok")
