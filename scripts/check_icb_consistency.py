#!/usr/bin/env python3
"""check_icb_consistency.py — do the ICB's own sources AGREE with each other?

WHY THIS EXISTS. On 2026-08-26 (`2026-08-26-pantheon-rare-registers`) the frozen
Immutable Continuity Block contained TWO CONTRADICTORY STATEMENTS of the same
assignment: the research brief said P03 was EXISTENCE and P06 was NEWS, while the
pair assignments and both isolation briefs said the opposite. The coordinator had
swapped the two pairs on axis B, updated two artifacts, and left the third stating
the old assignment -- then froze all three together and dispatched them to six
agents.

A pair found it. No gate did, because no gate was looking:

    coord_build_icb.py asserts every source is PRESENT.
    It asserts the seat count is 18.
    It asserts no slot is suspiciously short.
    It asserts the golden-output quarantine holds.
    IT NEVER ASKS WHETHER TWO SOURCES CONTRADICT EACH OTHER.

Presence is not consistency. A block assembled from five independent documents can
be complete, correctly sized, quarantine-clean -- and internally incoherent.

    python3 scripts/check_icb_consistency.py <run-dir>

Exit 0 = sources agree. Exit 3 = a contradiction was found (fail closed, BEFORE
the freeze). Exit 2 = nothing parseable, which is itself reported rather than
passed -- a checker that measures nothing must never return success.
(2026-08-26: a sibling gate returned PASS while extracting zero blocks.)

Read-only. Writes nothing, renders nothing, spends nothing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FIELDS = ("member", "arm", "axis_b", "tempo")
ARMS = ("ACCESSIBLE", "AMBITIOUS")
AXES = ("NEWS", "EXISTENCE")


def norm_dash(s: str) -> str:
    """en/em dashes and hyphens all mean the same thing in a tempo band."""
    return re.sub(r"[‐-―−]", "-", s).replace(" ", "")


def claims_from_row(line: str) -> dict | None:
    """Parse one MARKDOWN TABLE ROW assigning values to exactly one pair.

    ⛔ UNIT DISCIPLINE (2026-08-26). The first version of this parser accepted any
    LINE mentioning a pair, and immediately produced two false contradictions from
    prose sentences that name two pairs and two members at once --

        "P04 is Lofn's by right ... P06 is Sunna's by her admission register"

    -- pairing the first pair with the wrong member. That is the UNIT bug class:
    the matcher found the right words in the wrong structure. A checker that cries
    wolf gets ignored, which is worse than no checker.

    So: table rows only, and exactly ONE pair mentioned in the row.
    """
    if not line.lstrip().startswith("|"):
        return None
    pairs = set(re.findall(r"\bP(0[1-9])\b", line))
    if len(pairs) != 1:
        return None
    out: dict = {"pair": pairs.pop()}
    up = line.upper()
    arms = [a for a in ARMS if re.search(rf"(?<![A-Z]){a}(?![A-Z])", up)]
    axes = [a for a in AXES if re.search(rf"(?<![A-Z]){a}(?![A-Z])", up)]
    whos = [w for w in ("SUNNA", "LOFN") if re.search(rf"(?<![A-Z]){w}(?![A-Z])", up)]
    if len(arms) == 1:
        out["arm"] = arms[0]
    if len(axes) == 1:
        out["axis_b"] = axes[0]
    if len(whos) == 1:
        out["member"] = whos[0]
    # UNIT FIX 2026-08-27: this matched the DATE inside "2026-08-26" and reported
    # a tempo band of "08-26" against two pairs -- on the very run where it had
    # correctly caught the coordinator swapping two arms. A real defect and a false
    # one in the same output, which is the fastest way to get a gate ignored.
    # A tempo is a BPM BAND: both endpoints musical, ascending. "Any two hyphenated
    # numbers" was never the unit.
    for _m in re.finditer(r"\b(\d{2,3})\s*[‐-―−-]\s*(\d{2,3})\b", line):
        _lo, _hi = int(_m.group(1)), int(_m.group(2))
        if 40 <= _lo < _hi <= 220:
            out["tempo"] = norm_dash(f"{_lo}-{_hi}")
            break
    return out if len(out) > 1 else None


def claims_from_brief(path: Path) -> list[tuple[str, str, str, str]]:
    """An isolation brief states its own values in headers, not a table row, and
    names itself `PAIR 03` rather than `P03`. Parse it on its own terms."""
    nn = re.search(r"pair_(\d{2})_", path.name).group(1)
    txt = path.read_text(encoding="utf-8", errors="replace")
    head = chr(10).join(txt.split(chr(10))[:40])
    out = []
    m = re.search(r"\*\*Arm:\*\*\s*([A-Z]+)", head)
    if m and m.group(1) in ARMS:
        out.append((nn, "arm", m.group(1), f"isolation brief {nn}"))
    m = re.search(r"\*\*Axis B:\*\*\s*([A-Z]+)", head)
    if m and m.group(1) in AXES:
        out.append((nn, "axis_b", m.group(1), f"isolation brief {nn}"))
    m = re.search(r"ISOLATION BRIEF\s*[·-]\s*\*\*(SUNNA|LOFN)", head, re.I)
    if m:
        out.append((nn, "member", m.group(1).upper(), f"isolation brief {nn}"))
    m = re.search(r"\*\*(\d{2,3})\s*[‐-―−-]\s*(\d{2,3})\s*BPM\*\*", head)
    if m:
        out.append((nn, "tempo", norm_dash(f"{m.group(1)}-{m.group(2)}"), f"isolation brief {nn}"))
    return out


def collect(path: Path, label: str) -> list[tuple[str, str, str, str]]:
    """-> [(pair, field, value, source_label)]"""
    if not path.exists():
        return []
    got = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        c = claims_from_row(line)
        if not c:
            continue
        for f in FIELDS:
            if f in c:
                got.append((c["pair"], f, c[f], label))
    return got


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_icb_consistency.py <run-dir>")
        return 2
    run = Path(sys.argv[1])
    if not run.is_dir():
        print(f"FAIL: not a directory: {run}")
        return 2

    claims: list[tuple[str, str, str, str]] = []
    for name, label in (("00_research_brief.md", "research brief"),
                        ("05_pair_assignments.md", "pair assignments"),
                        ("04_metaprompt.md", "metaprompt")):
        claims += collect(run / name, label)
    for p in sorted(run.glob("pair_*_step05_isolated_brief.md")):
        claims += claims_from_brief(p)

    if not claims:
        print("FAIL: no pair claims parsed from any source.")
        print("      A checker that measures nothing must not return success.")
        return 2

    by: dict[tuple[str, str], dict[str, set]] = {}
    for pair, field, val, src in claims:
        by.setdefault((pair, field), {}).setdefault(val, set()).add(src)

    conflicts = []
    for (pair, field), vals in sorted(by.items()):
        if len(vals) > 1:
            conflicts.append((pair, field, vals))

    pairs = sorted({p for p, _ in by})
    print(f"ICB SOURCE CONSISTENCY — {run.name}")
    print(f"  pairs seen   : {len(pairs)}  ({', '.join('P'+p for p in pairs)})")
    print(f"  claims parsed: {len(claims)} across {len({s for _,_,_,s in claims})} sources")

    if not conflicts:
        print("  RESULT       : SOURCES AGREE")
        return 0

    print(f"  RESULT       : ⛔ {len(conflicts)} CONTRADICTION(S)\n")
    for pair, field, vals in conflicts:
        print(f"  P{pair} · {field}")
        for v, srcs in sorted(vals.items()):
            print(f"      {v:<12} <- {', '.join(sorted(srcs))}")
    print("\n  A frozen block that contradicts itself will be resolved differently by")
    print("  every agent that reads it. Fix the sources, THEN freeze.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
