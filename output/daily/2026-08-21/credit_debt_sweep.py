#!/usr/bin/env python3
"""CREDIT-DEBT SWEEP — does every living-scene idiom raised in a STYLE field have a
dedication behind it?

WHY THIS EXISTS (2026-08-21, found by a step-11 tier, not by a gate)
--------------------------------------------------------------------
`DEDICATION_STANDARD` and the 16-point gate both make a Lineage & Credit block blocking
for any track whose style field names a living scene. **Nothing read it.** The repo had
a sweep for NAMED PLAYERS and named breaks (`amen` -> Gregory C. Coleman / The Winstons,
`funky drummer` -> Clyde Stubblefield) — but no sweep for the *scene* case, which is the
far more common one and the one a coordinator gets wrong by omission.

On this run the COORDINATOR told Pair 01 and Pair 04 that no credit was owed. Both were
raising living-scene idioms in all four of their style fields:

    Pair 01 -> `gasba`        (Western Algeria; the pair's assigned ONE SEAM)
    Pair 04 -> `dub-poetry`   (Jamaican/British scene)

Pair 01's step-11 tier caught it, refused to invent a credit, refused to drop the idiom,
and escalated. Pair 04's step-10 shipped it uncaught. **Neither pair was at fault.**

⛔ THE RULE THIS ENFORCES, AND WHY IT IS ASYMMETRIC
    "An OVER-credit is visible and embarrassing; an UNDER-credit reads as a completed
     check." (DEDICATION_STANDARD, after 2026-08-13, when two songs shipped `amen` in
     the style field with full multi-artist lineage blocks and NEITHER credited Gregory
     C. Coleman or The Winstons. Living-scene credit is easy, visible and already gated;
     the player who made the figure is invisible, usually dead, and asks for nothing —
     which is why the system forgets exactly the credit that most needs paying.)

    ⭐ A technique raised in the STYLE field owes its scene and its players.
      The same technique banned in EXCLUDE owes nobody.
    ⭐ A name in the prompt is an instruction to imitate; a name in the dedication is a
      debt paid. So the sweep ALSO fails a package that put a player's name in a
      generator-facing field.

Usage:  python3 output/daily/2026-08-21/credit_debt_sweep.py [run-dir]
Exit 0 = no debt outstanding. Exit 1 = at least one uncredited scene. Fail-CLOSED.
"""
import pathlib
import re
import sys

RUN = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "output/daily/2026-08-21")

# Living scenes / idioms whose being NAMED in a style field creates a credit debt.
# Public idioms, not people — naming these in a prompt is legitimate; NOT crediting the
# scene behind them is not.
LIVING_SCENES = [
    "gasba", "funana", "funaná", "rai", "raï", "morna", "semba", "coladeira",
    "amapiano", "gqom", "kwaito", "afrobeats", "baile phonk", "baile funk",
    "tamborzao", "tamborzão", "brazilian-funk", "cumbia", "reggaeton", "dancehall",
    "dub-poetry", "dub poetry", "chaabi", "shaabi", "canntaireachd", "drill",
    "grime", "footwork", "juke", "highlife", "soukous", "mbalax", "qawwali",
]

# Named figures of sound. A NAMED figure in a style field owes THE PLAYER, not just the
# scene — and the player's name must never appear in a generator-facing field.
NAMED_BREAKS = {
    "amen": "Gregory C. Coleman / The Winstons",
    "funky drummer": "Clyde Stubblefield",
    "apache": "the Incredible Bongo Band",
    "think break": "Lyn Collins",
}


def wb(text, tok):
    """Word boundaries are mandatory. `garage` contains `rage`; `ornament` contains
    `amen`; `capitalisation` contains `lisa`. All four of those were real false
    positives in this system, three of them on 2026-08-21 alone."""
    return len(re.findall(r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])", text.lower()))


def style_fields(text):
    return re.findall(r"^#{1,4}[^\n]*MUSIC PROMPT[^\n]*\n(.*?)(?=^#{1,4}\s)", text, re.S | re.M)


def exclude_fields(text):
    return re.findall(r"^#{1,4}[^\n]*EXCLUDE[^\n]*\n(.*?)(?=^#{1,4}\s)", text, re.S | re.M)


def variations(text):
    """Split into per-variation chunks so the debt is reported where it lives.

    [2026-08-21 REPAIR, raised by Pair 04 against its own convenience]
    The first version reported PER FILE. Pair 04 word-boundary swept its own four style
    fields and found `dub-poetry` in **V3 only** — V1 and V4 name no scene at all, and
    V2 names `techno`, a house-palette genre from the ICB's own SLOT 9 which owes
    nobody. My sweep's docstring said "all four of their style fields". It was wrong,
    and the pair corrected it while resolving the finding.

    ⭐ A DEBT IS PER VARIATION. Reporting it per file over-states the problem, and an
    over-stated finding is repaired by over-correction — here, four dedications where
    one was owed, or four idioms dropped where one was.
    """
    marks = list(re.finditer(r"^#{1,4}[ \t]*(?:[^\w\s#]+[ \t]*)?"
                             r"(?:(?:PACKAGE|VARIATION|SONG|TRACK)[ \t]+V?\d|V\d)\b[^\n]*$",
                             text, re.M | re.I))
    if len(marks) < 2:
        return [("all", text)]
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        label = re.search(r"V\s*(\d)", m.group(0), re.I)
        out.append((f"V{label.group(1)}" if label else str(i + 1), text[m.start():end]))
    return out


def main():
    # [2026-08-21] Default to the SHIPPING artifacts. Step 11 SUPERSEDES step 10, so a
    # debt repaired at step 11 must not be re-reported against the artifact it replaced.
    # `--all` re-includes step-10 for an audit trail.
    want_all = "--all" in sys.argv
    files = sorted(p for p in RUN.glob("pair_*.md")
                   if "final_package_enhanced" in p.name
                   or (want_all and "step10_revision_synthesis" in p.name))
    if not files:
        sys.exit("FATAL: no step-10 / step-11 package files found — refusing to report CLEAN "
                 "on an empty extraction.")
    debts, rows = [], []
    for f in files:
        t = f.read_text(encoding="utf-8", errors="replace")
        if not style_fields(t):
            sys.exit(f"FATAL: extracted 0 style fields from {f.name} — HARD ERROR, never a score.")
        for vid, chunk in variations(t):
            styles = "\n".join(style_fields(chunk))
            excls = "\n".join(exclude_fields(chunk))
            if not styles:
                continue
            scenes = sorted({s for s in LIVING_SCENES if wb(styles, s)})
            breaks = sorted({b for b in NAMED_BREAKS if wb(styles, b)})
            banned_only = sorted({s for s in LIVING_SCENES
                                  if wb(excls, s) and not wb(styles, s)})
            has_ded = bool(re.search(r"^#{1,4}[^\n]*DEDICATION", chunk, re.M | re.I))
            owed = bool(scenes or breaks)
            status = "ok"
            if owed and not has_ded:
                status = "DEBT"
                debts.append((f"{f.name} {vid}", scenes, breaks))
            elif not owed and has_ded:
                # [2026-08-21 REPAIR, raised by Pair 04] A dedication that credits NOBODY
                # is not an over-credit — it is the MINIMUM dedication the standard
                # requires of every shipped piece (priority 4, honest AI authorship,
                # which always applies). Only a block naming a THIRD PARTY the track
                # never raised is an over-credit, and only that is worth checking.
                # Labelling a no-credit block "over-credit" invites exactly the wrong
                # repair: deleting a mandatory dedication to satisfy a mislabel.
                body = "\n".join(re.findall(r"^#{1,4}[^\n]*DEDICATION[^\n]*$(.*?)(?=^#{1,4}\s|\Z)",
                                            chunk, re.S | re.M))
                names_third_party = bool(re.search(
                    r"(?i)\b(?:credit|thanks?|owed|borrowed|after|lineage|players?|scene)\b", body)
                    and not re.search(r"(?i)\bnone is claimed\b|\bno (?:break|scene|figure)\b", body))
                status = ("over-credit -- names a party the track never raised, CHECK IT"
                          if names_third_party else
                          "minimum dedication (credits nobody; correct for a track that borrows nothing)")
            rows.append((f"{f.name[:28]} {vid}", scenes, breaks, banned_only, has_ded, status))

    w = max(len(r[0]) for r in rows)
    print(f"{'file':<{w}}  {'scenes in STYLE':<34} {'ded?':<5} status")
    print("-" * (w + 60))
    for name, scenes, breaks, banned_only, has_ded, status in rows:
        raised = ", ".join(scenes + [f"{b}*" for b in breaks]) or "(none)"
        print(f"{name:<{w}}  {raised:<34} {'yes' if has_ded else 'NO':<5} {status}")
        if breaks:
            for b in breaks:
                print(f"{'':<{w}}  [!] NAMED FIGURE `{b}` -> CREDIT THE PLAYER: {NAMED_BREAKS[b]}")
        if banned_only:
            print(f"{'':<{w}}  (banned in EXCLUDE only, owes nobody: {', '.join(banned_only)})")

    print()
    if debts:
        print(f"[DEBT] {len(debts)} FILE(S) CARRY AN UNPAID CREDIT DEBT:")
        for name, scenes, breaks in debts:
            print(f"   - {name}: raises {', '.join(scenes + breaks)} in its style field(s), "
                  f"no DEDICATION block present")
        print("\n   The fix is the CREDIT, not the deletion — unless the idiom is not "
              "structural, in which case dropping it is also legitimate.")
        sys.exit(1)
    print("[OK] no unpaid credit debt: every style field naming a living scene has a dedication behind it.")
    return 0


if __name__ == "__main__":
    main()
