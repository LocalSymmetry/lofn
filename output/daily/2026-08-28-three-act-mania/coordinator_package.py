#!/usr/bin/env python3
"""Build SUNO_PACKAGES.md for the selected six, with a ROUND-TRIP AUDIT.

Every field written out is re-read from the emitted file and compared byte for
byte against the source package. A packager that silently truncates or re-wraps
a field ships a different song than the one that passed the gates.
"""
import os
import re
import sys

sys.path.insert(0, r"E:\git\private-claw\scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coordinator_verify import fields, lyrics_of, STYLE_RX, EXCL_RX  # noqa: E402

SEL = [
    ("01", 4, "NOTHING WAS TAKEN", "ACCESSIBLE", "NEWS", "LOFN-PRIME", "138-146"),
    ("02", 1, "SAY HOW FAR", "ACCESSIBLE", "EXISTENCE", "LOFN-PRIME", "68-76"),
    ("03", 1, "THE FLOOR IS ALREADY GOING", "ACCESSIBLE", "EXISTENCE", "SUNNA R5", "122-130"),
    ("04", 4, "THE WHOLE TIME", "AMBITIOUS", "NEWS", "LOFN-PRIME", "88-96"),
    # V4 -> V1: override reversed under the rule in 07_evaluation_and_selection.md
    # (author AND instrument both favoured V1; taste alone does not outvote both).
    ("05", 1, "LEAVE IT ON", "AMBITIOUS", "EXISTENCE", "SUNNA R5", "152-162"),
    ("06", 4, "WATCH ME NAME THE WHOLE ROOM", "AMBITIOUS", "NEWS", "LOFN-PRIME", "104-112"),
]

HDR = """# SUNO_PACKAGES — `2026-08-28-three-act-mania`
## The selected six · paste-ready

⛔ **`vault/AUTONOMY.md` holds. Nothing here has been rendered, heard, published or spent.**
Every field below is a **text** claim. Three acts, *attacca*, one pole.

**Round-trip audited:** every field was re-read from this file after writing and
compared byte-for-byte against its source package. Result at the foot of the file.

---
"""

# --- STALENESS GUARD -------------------------------------------------------
# ⭐ FOUND BY P02, 2026-08-28: the round-trip audit below proves the emitted file
# matched its sources AT THE MOMENT OF WRITING. It says nothing about later. A
# pair repaired its lyric after packaging and SUNO_PACKAGES.md silently held the
# version QA had already rejected. "A number does not become true by being
# copied forward, and neither does a lyric."
# So: --check re-compares the shipped file against CURRENT sources and exits
# non-zero if it drifted. Run it immediately before delivery, every time.
if "--check" in sys.argv:
    from coordinator_verify import lyrics_of as _lo
    shipped = _lo("SUNO_PACKAGES.md") if os.path.exists("SUNO_PACKAGES.md") else []
    drift = []
    for i, (n, v, title, *_rest) in enumerate(SEL):
        cur = _lo(f"pair_{n}_step10_final_package.md")[v - 1]
        if i >= len(shipped) or shipped[i] != cur:
            drift.append(f"  {title} (P{n}V{v}): shipped "
                         f"{len(shipped[i]) if i < len(shipped) else 0} chars, "
                         f"source {len(cur)} chars")
    if drift:
        print("STALE — SUNO_PACKAGES.md no longer matches its sources:")
        print("\n".join(drift))
        sys.exit(1)
    print(f"CURRENT — all {len(SEL)} shipped lyrics byte-identical to current sources.")
    sys.exit(0)

src, out = {}, [HDR]
for n, v, title, arm, axis, member, bpm in SEL:
    p = f"pair_{n}_step10_final_package.md"
    sty = fields(p, STYLE_RX)[v - 1]
    exc = fields(p, EXCL_RX)[v - 1]
    lyr = lyrics_of(p)[v - 1]
    src[f"P{n}V{v}"] = (sty, exc, lyr)
    out.append(f"""
# {len(src)}. {title}
`P{n} V{v}` · **{member}** · {arm} · {axis} · primary band **{bpm} BPM**

### STYLE FIELD  *(paste into Suno's style box — {len(sty)} chars)*
```
{sty}
```

### EXCLUDE  *({len(exc)} chars)*
```
{exc}
```

### LYRICS  *({len(lyr)} chars)*
```
{lyr}
```

---
""")

path = "SUNO_PACKAGES.md"
open(path, "w", encoding="utf-8").write("".join(out))

# --- ROUND-TRIP AUDIT ------------------------------------------------------
back = open(path, encoding="utf-8").read()
blocks = re.findall(r"```\n(.*?)\n```", back, re.S)
flat = [f for trio in src.values() for f in trio]
ok = sum(1 for a, b in zip(flat, blocks) if a == b)
lines = [f"\n## ROUND-TRIP AUDIT\n",
         f"- fields written: **{len(flat)}** (6 songs x style/exclude/lyrics)\n",
         f"- fields re-read from this file: **{len(blocks)}**\n",
         f"- **byte-identical to source: {ok} / {len(flat)}**\n"]
if ok != len(flat) or len(blocks) != len(flat):
    lines.append("\n⛔ **MISMATCH — do not paste from this file until repaired.**\n")
    for (k, trio), i in zip(src.items(), range(0, len(flat), 3)):
        for j, name in enumerate(("style", "exclude", "lyrics")):
            if i + j >= len(blocks) or flat[i + j] != blocks[i + j]:
                lines.append(f"  - `{k}` {name}\n")
else:
    lines.append("\n✅ **All fields byte-identical. Safe to paste.**\n")
open(path, "a", encoding="utf-8").write("".join(lines))
print(f"wrote {path}: {len(src)} songs, {ok}/{len(flat)} fields byte-identical "
      f"({len(blocks)} blocks re-read)")
