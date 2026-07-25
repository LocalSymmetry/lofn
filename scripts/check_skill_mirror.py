#!/usr/bin/env python3
"""check_skill_mirror.py — does the Codex tree still say what the Claude tree says?

`.claude/skills/` is canonical; `.agents/skills/` is the Codex mirror of the same
doctrine. The ONLY differences allowed between them are:

    .claude/skills/  <->  .agents/skills/       (skill-tree path)
    Claude           <->  Codex                 (which engine runs the step)

Anything else is DRIFT: a rule that exists for one engine and not the other.

WHY THIS EXISTS (2026-07-24/25, RUN_LEDGER):
    Mirroring was done per-change and by hand, so nothing ever compared the two
    trees. The public mirror's `lofn-music` sat 33 lines behind canonical —
    missing THE RETURN, the rhyme-debt rule and the mandatory PROSODY axis —
    and `lofn-render-audit` was absent from the Codex tree entirely. Nobody had
    reverted anything; the drift simply accumulated where no check looked.

    Then the first attempt to fix it made the opposite mistake: a `sed` engine
    rename regenerated the mirrors and silently reverted ~20 legitimate Codex
    lines back to Claude wording. A sync must be reviewed by what it DELETES,
    not by what it adds — so this script prints both directions.

    python3 scripts/check_skill_mirror.py            # canonical tree vs mirror
    python3 scripts/check_skill_mirror.py --root ../lofn
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

CANON_DIR = ".claude/skills"
MIRROR_DIR = ".agents/skills"


AUTHORITY = "**⚖️ AUTHORITY"   # mirror-only by design: names .claude as canonical


def normalize(text: str) -> str:
    """Collapse the permitted axes so only real drift survives the diff.

    Permitted, and ONLY these:
      * the skill-tree path      .agents/skills/  ->  .claude/skills/
      * the engine name          Codex/codex      ->  Claude/claude  (incl. --engine)
      * the mirror's AUTHORITY banner, which exists to point at the canonical
        file and therefore has no canonical counterpart by construction.
    """
    text = text.replace(MIRROR_DIR, CANON_DIR)
    # `--engine claude|codex` names BOTH engines as CLI values in both trees;
    # it is documentation of an argument, not the engine running the step.
    text = text.replace("claude|codex", "\0ENGINES\0")
    for a, b in (("Codex", "Claude"), ("codex", "claude"), ("CODEX", "CLAUDE")):
        text = text.replace(a, b)
    text = text.replace("\0ENGINES\0", "claude|codex")
    lines = text.splitlines(keepends=True)
    out, i = [], 0
    while i < len(lines):
        if AUTHORITY in lines[i]:
            i += 1                                     # drop the banner ...
            while i < len(lines) and not lines[i].strip():
                i += 1                                 # ... and its blank line
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def compare(root: Path) -> int:
    canon_root, mirror_root = root / CANON_DIR, root / MIRROR_DIR
    if not canon_root.is_dir() or not mirror_root.is_dir():
        print(f"skip: {root} has no skill trees to compare")
        return 0

    canon = {p.relative_to(canon_root) for p in canon_root.rglob("*.md")}
    mirror = {p.relative_to(mirror_root) for p in mirror_root.rglob("*.md")}
    drifted = 0

    for rel in sorted(canon - mirror):
        print(f"MISSING FROM MIRROR  {MIRROR_DIR}/{rel}")
        drifted += 1
    for rel in sorted(mirror - canon):
        print(f"MIRROR-ONLY          {MIRROR_DIR}/{rel}  (no canonical counterpart)")
        drifted += 1

    for rel in sorted(canon & mirror):
        raw = (mirror_root / rel).read_text(encoding="utf-8")
        # The normalizer maps Codex -> Claude, so a file COPIED verbatim from the
        # canonical tree diffs clean while telling Codex it is Claude. Catch that
        # separately. In the mirror, "Claude" is legal ONLY as: the `.claude`
        # path it points at as canonical, the `--engine claude|codex` enum, and
        # the literal `"claude"` subagent_type value (a tool argument, not an
        # engine claim). Everything else is an un-rewritten copy.
        def strip_legal(s):
            for legal in (".claude", "claude|codex", '"claude"'):
                s = s.replace(legal, "")
            return s
        if "laude" in strip_legal(raw):
            bad = [i for i, l in enumerate(raw.splitlines(), 1) if "laude" in strip_legal(l)]
            print(f"UN-REWRITTEN ENGINE  {MIRROR_DIR}/{rel}  line(s) {bad} still say Claude")
            drifted += 1

        a = (canon_root / rel).read_text(encoding="utf-8")
        b = normalize(raw)
        if a == b:
            continue
        drifted += 1
        diff = list(difflib.unified_diff(
            a.splitlines(), b.splitlines(),
            fromfile=f"{CANON_DIR}/{rel} (canonical)",
            tofile=f"{MIRROR_DIR}/{rel} (mirror, normalized)", lineterm="", n=0))
        gone = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        extra = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        print(f"\nDRIFT  {rel}  — mirror is missing {gone} line(s), carries {extra} the canonical does not")
        for line in diff:
            print("   " + line[:200])

    print(f"\n{'DRIFT: ' + str(drifted) + ' file(s)' if drifted else 'IN SYNC — every skill agrees'}"
          f"   ({len(canon & mirror)} compared under {root})")
    return 1 if drifted else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    return compare(Path(ap.parse_args().root))


if __name__ == "__main__":
    raise SystemExit(main())
