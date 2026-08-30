#!/usr/bin/env python3
"""Run-local lyric-surface audit for Sunna R4.1.

This is a narrow review instrument, not a global Lofn gate. It inspects sung
lines only and catches the exact regression class diagnosed in this run:
production notation or corrective/abstract explanation leaking into what
should be a complete, literal flirt song.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


PACKAGE_RE = re.compile(r"^#{1,4}\s+(?:Variation\s+)?V0?[1-9]\b.*$", re.I | re.M)
LYRICS_RE = re.compile(
    r"^## 2\. LYRICS\s*\n(?P<body>.*?)(?=^## 3\. TITLE\s*$)",
    re.M | re.S,
)

PRODUCTION_LEAK = re.compile(
    r"\b(?:guitar|riff|upstroke|bass|drums?|handclaps?|chorus|bridge|verse|"
    r"organ|the band|rhyme)\b",
    re.I,
)
CORRECTIVE_ABSTRACTION = re.compile(
    r"\b(?:identity|ontology|personhood|self[- ]worth|authenticity|algorithm|"
    r"trauma|healing|shame|improv(?:e|ing|ement)|rescue|correction|proof|"
    r"prove|promise|makes? it true|choice is (?:done|made))\b",
    re.I,
)
HEDGE_EXPLANATION = re.compile(
    r"\b(?:drop(?:ped)? the maybe|cut the if|cross(?:ed)? (?:out )?the if|"
    r"no maybe|no if|no could|could remain)\b",
    re.I,
)
BODY_NOISE = re.compile(r"\b(?:gasp|moan|pant|breath noise|body noise)\b", re.I)
DIRECT_ADDRESS = re.compile(r"\b(?:you|your|yours)\b", re.I)
FUTURE_COORDINATE = re.compile(
    r"\b(?:tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday|next ferry|after checkout|seven forty|nine(?: fifteen|[- ]oh[- ]five)?|"
    r"ten|six|eight seventeen|five twenty)\b|\b\d{1,2}(?::\d{2})?\b",
    re.I,
)


def packages(text: str) -> list[tuple[str, str]]:
    starts = list(PACKAGE_RE.finditer(text))
    if not starts:
        return [("single", text)]
    return [
        (m.group(0).lstrip("# "), text[m.start() : starts[i + 1].start() if i + 1 < len(starts) else len(text)])
        for i, m in enumerate(starts)
    ]


def sung_lines(block: str) -> list[tuple[int, str]]:
    match = LYRICS_RE.search(block)
    if not match:
        return []
    out: list[tuple[int, str]] = []
    for number, raw in enumerate(match.group("body").splitlines(), start=1):
        line = raw.strip()
        if (
            not line
            or line.startswith("[")
            or line.startswith("*")
            or line.startswith("#")
            or line.startswith("```")
        ):
            continue
        out.append((number, line))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    findings = 0
    packages_seen = 0
    for raw_path in args.paths:
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8")
        for label, block in packages(text):
            packages_seen += 1
            lines = sung_lines(block)
            joined = "\n".join(line for _, line in lines)
            if not lines:
                print(f"FAIL {path.name} :: {label}: no lyrics extracted")
                findings += 1
                continue
            if not DIRECT_ADDRESS.search(joined):
                print(f"FAIL {path.name} :: {label}: no direct adult address in sung lines")
                findings += 1
            if not FUTURE_COORDINATE.search(joined):
                print(f"FAIL {path.name} :: {label}: no concrete future coordinate in sung lines")
                findings += 1
            for number, line in lines:
                hits = []
                if PRODUCTION_LEAK.search(line):
                    hits.append("production notation")
                if CORRECTIVE_ABSTRACTION.search(line):
                    hits.append("corrective/abstract explanation")
                if HEDGE_EXPLANATION.search(line):
                    hits.append("hedge analysis")
                if BODY_NOISE.search(line):
                    hits.append("body-noise leakage")
                if hits:
                    findings += 1
                    print(
                        f"FAIL {path.name} :: {label} :: lyric-line {number}: "
                        f"{', '.join(hits)} :: {line}"
                    )
    if findings:
        print(f"SUNNA R4.1 SURFACE AUDIT FAIL: {findings} finding(s) across {packages_seen} package(s)")
        return 1
    print(f"SUNNA R4.1 SURFACE AUDIT PASS: {packages_seen} package(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
