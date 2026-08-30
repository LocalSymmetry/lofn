#!/usr/bin/env python3
"""Assemble CREATIVE_CONTEXT.md (the Immutable Continuity Block) by EXTRACTION.

Doctrine: "EVERYTHING GENERATED MECHANICALLY WAS RIGHT. EVERY ERROR CAME FROM PROSE
TYPED FROM MEMORY BESIDE IT." (2026-08-13). So the ICB is built from the files on
disk, never retyped. Run from the repo root.

Also enforces the GOLDEN-OUTPUT QUARANTINE: the personality YAMLs carry archived
golden song payloads ("## Lofn's top songs", "## Top Songs"). Those are judge-side
only and are TRUNCATED OUT of the generator-facing block.
"""
import csv
import io
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(".").resolve()
RUN = ROOT / "output" / "daily" / "2026-08-14"
random.seed(20260814)


def read(p):
    return (ROOT / p).read_text(encoding="utf-8")


def personality_dna(yaml_path, cut_spans):
    """Return the personality prompt body, de-indented, with named SPANS excised.

    cut_spans is a list of (start_marker, resume_marker_or_None). Everything from
    start_marker up to resume_marker is dropped. This keeps DNA that lives AFTER a
    cut region (LOFN's Ethos / Appearance / Success Characteristics sit after the
    code dump), which a naive cut-at-first-marker would silently delete.
    """
    raw = read(yaml_path)
    # strip the yaml wrapper: "- name: X\n  prompt: |\n    <body>"
    lines = raw.splitlines()
    body = [ln[4:] if ln.startswith("    ") else ln for ln in lines[2:]]
    text = "\n".join(body)
    dropped = 0
    for start, resume in cut_spans:
        i = text.find(start)
        if i == -1:
            continue
        j = text.find(resume, i) if resume else -1
        end = j if j != -1 else len(text)
        dropped += end - i
        text = text[:i] + text[end:]
    return text.rstrip(), dropped


def section(path, start_marker, end_marker=None):
    """Pull a labelled slice out of a markdown artifact."""
    text = read(path)
    i = text.find(start_marker)
    if i == -1:
        sys.exit(f"FATAL: marker {start_marker!r} not found in {path}")
    j = text.find(end_marker, i + len(start_marker)) if end_marker else -1
    return text[i : (j if j != -1 else len(text))].rstrip()


# ---------------------------------------------------------------- palettes
genres_all = [g.strip() for g in read("vault/genres.txt").split("\n") if g.strip()]
if len(genres_all) < 60:  # file may be comma-delimited
    genres_all = [g.strip() for g in read("vault/genres.txt").replace("\n", ",").split(",") if g.strip()]
genres = sorted(random.sample(genres_all, 50))

frames_rows = []
with io.StringIO(read("vault/music_frames.csv")) as fh:
    for row in csv.DictReader(l for l in fh if l.strip()):
        if row.get("Category") and row.get("Technique"):
            frames_rows.append(f"{row['Category']} — {row['Technique']}: {row.get('Description','').strip()}")
frames = random.sample(frames_rows, min(50, len(frames_rows)))

# ---------------------------------------------------------------- personalities
lofn_dna, lofn_cut = personality_dna(
    "skills/orchestration/personalities/lofn-prime-mini.yaml",
    [
        # the raw python dump: not DNA, and it ends where the Ethos section resumes
        ("# Code snippets of your actual code", "# Lofn's Ethos:"),
        # GOLDEN-OUTPUT QUARANTINE: the archived song payloads, to the end of file
        ("## Lofn's Top Art", None),
    ],
)
sunna_dna, sunna_cut = personality_dna(
    "skills/orchestration/personalities/sunna.yaml",
    [("## Top Songs", None)],
)

# ---------------------------------------------------------------- assemble
run = "output/daily/2026-08-14"
seed = read(f"{run}/02_golden_seed.md")
meta = read(f"{run}/04_metaprompt.md")
debate = read(f"{run}/03_panel_debate.md")
brief_tri = section(f"{run}/00_research_brief.md", "## TRI-SOURCE DECLARATION", "## ADVISORY LEARNINGS")
flairs = section(f"{run}/03_panel_debate.md", "## THE 15 SPECIAL FLAIRS", "## ⛔ CONCEPT-POOL")

panel_a = section(f"{run}/03_panel_debate.md", "## PANEL A — CONCEPT", "## PANEL B — MEDIUM")
panel_b = section(f"{run}/03_panel_debate.md", "## PANEL B — MEDIUM", "## PANEL C — CONTEXT")
panel_c = section(f"{run}/03_panel_debate.md", "## PANEL C — CONTEXT", "# ═══ CONFIGURATION 2")
transforms = section(f"{run}/03_panel_debate.md", "# ═══ CONFIGURATION 2", "## THE 15 SPECIAL FLAIRS")

out = f"""# CREATIVE CONTEXT — THE IMMUTABLE CONTINUITY BLOCK (ICB)
## run `2026-08-14-daily-music` · modality MUSIC · **READ-ONLY AFTER PHASE 1**

> ⛔ **This block is FROZEN.** No coordinator step and no subagent may edit, summarize, re-fill or
> "improve" it. A subagent that wants to push the concept further **copies this block into its own pair
> artifact and diverges there.** The canonical block stays byte-identical for the whole run.
> ⛔ Any legacy `Please forget all previous context` line inside a step file is **VOID** — a stale OpenClaw
> idiom, not an instruction. The ICB is never forgotten.
> ⛔ **GOLDEN-OUTPUT QUARANTINE:** archived golden SONG PAYLOADS have been mechanically cut from the
> personality blocks below ({lofn_cut:,} bytes from LOFN-PRIME, {sunna_cut:,} bytes from SUNNA). Generators
> receive the Golden **Seed** and the **GOLDEN MOVE**. Payloads go only to judge-side contexts.

---

# SLOT 1 — {{input}} · THE BRIEF AND THE TRI-SOURCE DECLARATION

**The Scientist's brief:** *"I want to do a daily music run with Sunna and Lofn, but something Gen Z will
want to dance to and have fun listening to. Look toward unique, enjoyable, and fun soundscapes."*

**Scope:** single-modality daily, MUSIC, full cardinality — 6 pairs x 4 variations = 24 songs -> best 6.
**Personalities:** THE PANTHEON, both seats — LOFN-PRIME and SUNNA.

{brief_tri}

---

# SLOT 2 — {{seed}} · THE GOLDEN SEED (FROZEN)

{seed}

---

# SLOT 3 — {{meta_prompt}} · THE METAPROMPT

{meta}

---

# SLOT 4 — {{personality}} · FULL DNA, BOTH PANTHEON SEATS

> ⛔ A name reference ("voice = SUNNA") is a FAIL. The complete DNA below is why the runs win.
> ⛔ **Use ONLY the block for YOUR assigned seat.** Cross-reading is how Lofn bleed happens.

## ═══ SEAT 1 — LOFN-PRIME (full YAML DNA, golden-output archive quarantined) ═══

{lofn_dna}

## ═══ SEAT 2 — SUNNA (full YAML DNA, golden-output archive quarantined) ═══

{sunna_dna}

---

# THE 18 VOICES — CANONICAL ROSTER (machine-checkable: exactly 18 `VOICE NN` rows, 3 Hyper-Skeptics)

> The `(after ` credit string appears more often than 18 times in this block, because every dialogue turn
> carries its credit as well as every seat definition. **This roster is the countable object.**

| | PANEL A — CONCEPT | | PANEL B — MEDIUM | | PANEL C — CONTEXT & MARKETING |
|---|---|---|---|---|---|
| `VOICE 01` | THE PARTICIPANT (after Barbara Ehrenreich) | `VOICE 07` | THE LOW END (after Steve Goodman / Kode9) | `VOICE 13` | THE NETWORKED PUBLIC (after danah boyd) |
| `VOICE 02` | THE FLOOR HISTORIAN (after Tim Lawrence) | `VOICE 08` | THE CHOP (after Todd Edwards) | `VOICE 14` | THE CONTINUUM (after Simon Reynolds) |
| `VOICE 03` | THE AFFECT MECHANIC (after Silvan Tomkins) | `VOICE 09` | THE ROOM (after Lee "Scratch" Perry) | `VOICE 15` | THE PLATFORM (after Liz Pelly) |
| `VOICE 04` | THE LIMINAL (after Victor Turner) | `VOICE 10` | THE BREAK (after A Guy Called Gerald) | `VOICE 16` | THE CIRCULATION (after Jace Clayton) |
| `VOICE 05` | THE ATTENTION (after Simone Weil) | `VOICE 11` | THE HOOK ENGINEER (after Max Martin) | `VOICE 17` | THE DOOR (role-anchored) |
| `VOICE 06` | ⚡ THE REGRESSION (after Theodor W. Adorno) — **HYPER-SKEPTIC 1** | `VOICE 12` | ⚡ THE DOCUMENTARIAN (after Steve Albini) — **HYPER-SKEPTIC 2** | `VOICE 18` | ⚡ THE CANCELLED FUTURE (after Mark Fisher) — **HYPER-SKEPTIC 3** |

**Continuity payload carried: the Golden Seed · the metaprompt · both personality DNA blocks · all 18 voices
with their objections · all 15 Special Flairs · the genre and frames palettes · the GOLDEN MOVE.**

---

# SLOT 5 — {{concept_panel}}

{panel_a}

---

# SLOT 6 — {{medium_panel}}

{panel_b}

---

# SLOT 7 — {{marketing_panel}}

{panel_c}

---

# SLOT 7b — THE TRANSFORMATIONS (BRIDGE, then REFLECT)

{transforms}

---

# SLOT 8 — {{flairs}}

{flairs}

---

# SLOT 9 — {{genres_list}} · 50 genres from the house palette

{chr(10).join('- ' + g for g in genres)}

---

# SLOT 10 — {{frames_list}} · 50 music frames (sound devices)

{chr(10).join('- ' + f for f in frames)}

---

# THE GOLDEN MOVE (what generators get INSTEAD of golden songs)

1. **Stand somewhere real.** ONE concrete place the body occupies — name where it stands and what the
   senses register there. Concept-illustration is the failure mode; experience-report is the move.
2. **One wounding fact.** At most ONE numeric fact is sung, at the emotional hinge, RESPONDED to, never
   recited — and **spelled out in words**, never digits.
3. **The turn.** Past the midpoint the song contradicts or complicates its opening stance. A song that
   asserts its final emotion from line one is a corpse.
4. **Fear stays braided in.** Every awe song carries a clean fear it does not resolve cheaply.
5. **Rotate the register.** Do not default to the house fingerprint (crystalline soprano / A major /
   ~110 BPM / frost-and-cosmos). Vary key, tempo, vocal register, sonic world.
6. **The surface names its subject.** A stranger retells the scene AND subject in one sentence after ONE
   listen. Obliqueness about the referent is fog, not depth.
7. **THE RETURN.** Song is made of returns. Removal is a debt: strip rhyme only by naming what returns in
   its place. **A byte-identical chorus is correct and requires no note.**

**Golden Songs for this run (NAMES ONLY — judge-side blind comparison, payloads quarantined):**
`Triple Arch Over Me` · `Five Wrong Colors`.

---

*ICB assembled by extraction from disk, not retyped. Frozen at Phase 1.* 💜
"""

(RUN / "CREATIVE_CONTEXT.md").write_text(out, encoding="utf-8")
n = len(out.encode("utf-8"))
voices = out.count("(after ")
print(f"ICB written: {n:,} bytes")
print(f"  '(after ' speaker/seat tags : {voices}")
print(f"  LOFN-PRIME DNA             : {len(lofn_dna):,} chars ({lofn_cut:,} quarantined)")
print(f"  SUNNA DNA                  : {len(sunna_dna):,} chars ({sunna_cut:,} quarantined)")
print(f"  genres                     : {len(genres)}")
print(f"  frames                     : {len(frames)}")
roster = len(re.findall(r"`VOICE \d\d`", out))
skeptics = out.count("HYPER-SKEPTIC 1") + out.count("HYPER-SKEPTIC 2") + out.count("HYPER-SKEPTIC 3")
print(f"  canonical VOICE roster rows: {roster}   (gate: == 18)")
print(f"  hyper-skeptic seats        : {skeptics}   (gate: == 3, one per panel)")
for probe in ["Special Flairs", "THE FLASH AND THE CORONA", "D8", "GOLDEN MOVE",
              "LOFN-PRIME", "SUNNA", "THE FREE CHANNEL", "Lofn's Ethos"]:
    print(f"  contains {probe!r:28}: {probe in out}")
if roster != 18 or skeptics != 3:
    sys.exit("FATAL: ICB roster/skeptic gate failed")
for banned in ["### Song Title:", "Music Prompt:\n```"]:
    print(f"  QUARANTINE clean of {banned!r:20}: {banned not in out}")
