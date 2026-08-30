#!/usr/bin/env python3
"""check_reply_lengths.py — measure drafted public replies against the platform cap.

WHY THIS EXISTS: on 2026-08-21 a hand re-stat fabricated 83 characters across four
compliant dedications, and the shared verifier fabricated 4 more — the third and
fourth instances of hand-counting error in that one field in eight days. The rule
that came out of it is that a character budget is MEASURED, never eyeballed, and
measured on the EXACT string that will be posted.

Reads a markdown file, pulls every blockquote block (the draft form used in
`output/*/PANEL_AND_DRAFTS.md`), and reports the true length of the posted string:
blockquote markers stripped, wrapped lines rejoined, nothing else changed.

    python3 scripts/check_reply_lengths.py output/2026-08-22_suno-comment-sweep/PANEL_AND_DRAFTS.md
    python3 scripts/check_reply_lengths.py FILE --cap 500

Exit 1 if any draft is over cap.

⭐ IT COUNTS UTF-16 CODE UNITS, because that is what the platform counts.
Measured against Suno's own live counter on 2026-08-22: a 462-character draft
(Python `len`) displayed as **463/500**, and the difference was exactly one 💜.
A browser's `String.length` is UTF-16, so every non-BMP emoji costs **2**, not 1.
Counting in Python characters silently under-reports by one per emoji -- fine at
270, fatal at 499.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER_RE = re.compile(r"^\s*>\s?")
LABEL_RE = re.compile(r"^\*\*(?P<label>.+?)\*\*.*?(?:\*\*(?P<claim>\d+)\s*chars\*\*)?\s*$")
CLAIM_RE = re.compile(r"\*\*(\d+)\s*chars\*\*")


def blocks(text: str) -> list[tuple[str, int | None, str]]:
    """Return (label, claimed_len, body) for each blockquote in the file."""
    out: list[tuple[str, int | None, str]] = []
    lines = text.splitlines()
    i = 0
    label = "(unlabelled)"
    claimed: int | None = None
    while i < len(lines):
        line = lines[i]
        if not MARKER_RE.match(line):
            stripped = line.strip()
            if stripped.startswith("**") and stripped.endswith(("**", "chars**")):
                label = re.sub(r"\*\*|\s+—.*$", "", stripped.split("**")[1]).strip() or label
                m = CLAIM_RE.search(stripped)
                claimed = int(m.group(1)) if m else None
            i += 1
            continue
        body: list[str] = []
        while i < len(lines) and MARKER_RE.match(lines[i]):
            body.append(MARKER_RE.sub("", lines[i]).rstrip())
            i += 1
        # A blockquote wrapped for the file is ONE posted line; blank quoted
        # lines are real paragraph breaks and are preserved.
        paras: list[str] = []
        cur: list[str] = []
        for b in body:
            if b.strip():
                cur.append(b.strip())
            else:
                if cur:
                    paras.append(" ".join(cur))
                    cur = []
        if cur:
            paras.append(" ".join(cur))
        out.append((label, claimed, "\n\n".join(paras)))
        label, claimed = "(unlabelled)", None
    return out


def utf16_len(text: str) -> int:
    """Length as the browser counts it -- UTF-16 code units, not characters."""
    return len(text.encode("utf-16-le")) // 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--min-chars", type=int, default=40, help="ignore quoted fragments shorter than this (they are citations, not drafts)")
    args = ap.parse_args()

    text = Path(args.path).read_text(encoding="utf-8")
    bad = 0
    misclaimed = 0
    print("%-28s %6s %6s %7s  %s" % ("label", "len", "claim", "cap", "status"))
    print("-" * 78)
    for label, claimed, body in blocks(text):
        n = utf16_len(body)
        if n < args.min_chars:
            continue
        over = n > args.cap
        wrong = claimed is not None and claimed != n
        if over:
            bad += 1
        if wrong:
            misclaimed += 1
        status = []
        if over:
            status.append("OVER CAP by %d" % (n - args.cap))
        if wrong:
            status.append("CLAIM WRONG (said %d, is %d, off by %+d)" % (claimed, n, claimed - n))
        print(
            "%-28s %6d %6s %7d  %s"
            % (label[:28], n, claimed if claimed is not None else "-", args.cap, "; ".join(status) or "ok")
        )

    print("-" * 78)
    print("%d over cap, %d mis-claimed" % (bad, misclaimed))
    return 1 if (bad or misclaimed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
