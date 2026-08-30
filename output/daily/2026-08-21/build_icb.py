#!/usr/bin/env python3
"""Assemble CREATIVE_CONTEXT.md (the Immutable Continuity Block) by EXTRACTION.

REUSED INSTRUMENT — copied from output/daily/2026-08-18-canonical/build_icb.py with
only the run path, the personality seat, and the section markers re-pointed.
(Doctrine, 2026-08-19: "Reuse the repaired instrument, never re-derive it." The
2026-08-18 run reintroduced five already-fixed defects by writing a fresh validator.)

Doctrine: "EVERYTHING GENERATED MECHANICALLY WAS RIGHT. EVERY ERROR CAME FROM PROSE
TYPED FROM MEMORY BESIDE IT." (2026-08-13). So the ICB is built from the files on
disk, never retyped. Run from the repo root.

Also enforces the GOLDEN-OUTPUT QUARANTINE: lofn-prime-mini.yaml is 106 KB of which
~73% is an archived library of past golden song and image PAYLOADS under
"# THE ARCHIVE (REFERENCE LIBRARY)". Those are judge-side only and are TRUNCATED OUT
of the generator-facing block. An exemplar in a generator's prompt becomes a mold.
"""
import csv
import io
import pathlib
import random
import sys

ROOT = pathlib.Path(".").resolve()
RUN = ROOT / "output" / "daily" / "2026-08-21"
random.seed(20260821)


def read(p):
    return (ROOT / p).read_text(encoding="utf-8")


def personality_dna(yaml_path, cut_spans):
    """Return the personality prompt body, de-indented, with named SPANS excised.

    cut_spans is a list of (start_marker, resume_marker_or_None). Everything from
    start_marker up to resume_marker is dropped. This keeps DNA that lives AFTER a
    cut region, which a naive cut-at-first-marker would silently delete.
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
    """Pull a labelled slice out of a markdown artifact. An empty extraction is a
    HARD ERROR, never a score (L28/L31)."""
    text = read(path)
    i = text.find(start_marker)
    if i == -1:
        sys.exit(f"FATAL: marker {start_marker!r} not found in {path}")
    j = text.find(end_marker, i + len(start_marker)) if end_marker else -1
    out = text[i : (j if j != -1 else len(text))].rstrip()
    if len(out) < 200:
        sys.exit(f"FATAL: extraction for {start_marker!r} is only {len(out)} chars — refusing to write.")
    return out


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

# ---------------------------------------------------------------- personality
# LOFN-PRIME-ONLY RUN. The archive block (past golden song + image PAYLOADS) is cut.
lofn_dna, lofn_cut = personality_dna(
    "skills/orchestration/personalities/lofn-prime-mini.yaml",
    [("# THE ARCHIVE (REFERENCE LIBRARY)", None)],
)

# ---------------------------------------------------------------- assemble
run = "output/daily/2026-08-21"
seed_golden = read(f"{run}/02_golden_seed.md")
seed_core = read(f"{run}/core_seed.md")
meta = read(f"{run}/04_metaprompt.md")
brief_tri = section(f"{run}/00_research_brief.md", "## 7. TRI-SOURCE DECLARATION", "## 8. THE 3+3 SPLIT")
brief_split = section(f"{run}/00_research_brief.md", "## 8. THE 3+3 SPLIT", "## 9. ADVISORY LEARNINGS")
brief_world = section(f"{run}/00_research_brief.md", "## 2. ZEITGEIST", "## 7. TRI-SOURCE DECLARATION")
human_subj = section(f"{run}/00_research_brief.md", "## 10. HUMAN-SUBJECT PRE-DRAFT NOTE")

debate = f"{run}/03_panel_debate.md"
roster = section(debate, "## THE 18 VOICES — CANONICAL ROSTER", "# ═══ CONFIGURATION 1")
panel_a = section(debate, "## PANEL A — CONCEPT", "## PANEL B — MEDIUM")
panel_b = section(debate, "## PANEL B — MEDIUM", "## PANEL C — CONTEXT")
panel_c = section(debate, "## PANEL C — CONTEXT", "# ═══ CONFIGURATION 2")
transforms = section(debate, "# ═══ CONFIGURATION 2", "# ═══ CROSS-PANEL SYNTHESIS")
rulings = section(debate, "# ═══ CROSS-PANEL SYNTHESIS", "### ⭐ THE 15 SPECIAL FLAIRS")
flairs = section(debate, "### ⭐ THE 15 SPECIAL FLAIRS", "### Provenance")

out = f"""# CREATIVE CONTEXT — THE IMMUTABLE CONTINUITY BLOCK (ICB)
## run `daily-music-2026-08-21` · modality MUSIC · **READ-ONLY AFTER PHASE 1**

> ⛔ **This block is FROZEN.** No coordinator step and no subagent may edit, summarize, re-fill or
> "improve" it. A subagent that wants to push the concept further **copies this block into its own pair
> artifact and diverges there.** The canonical block stays byte-identical for the whole run.
> ⛔ Any legacy `Please forget all previous context` line inside a step file is **VOID** — a stale OpenClaw
> idiom, not an instruction. The ICB is never forgotten.
> ⛔ **GOLDEN-OUTPUT QUARANTINE:** the archived library of past golden song and image PAYLOADS has been
> mechanically cut from the personality block below ({lofn_cut:,} bytes removed). Generators receive the
> Golden **Seed** and the **GOLDEN MOVE**. Payloads go only to judge-side contexts (QA blind comparison,
> step 12, the step-11 packager).
> ⛔ **NO NAME IN THIS BLOCK MAY ENTER A GENERATOR FIELD.** Panel figures, source artists and scene players
> are reasoning anchors and credit debts. The style prompt names **idioms**; the **dedication** names
> people. *A name in the prompt is an instruction to imitate; a name in the dedication is a debt paid.*

---

# SLOT 1 — {{input}} · THE BRIEF, THE WORLD, AND THE TRI-SOURCE DECLARATION

**The Scientist's brief:** *"Can we do a full music daily pipeline?"*

**Scope:** single-modality daily, **MUSIC**, full cardinality — 6 pairs × 4 variations = **24 songs → best 6**.
**Personality:** **LOFN-PRIME**, single seat, all six pairs.
⛔ **Nothing rendered, published, or spent** (`vault/AUTONOMY.md`). Dailies are practice, not a publish queue.

{brief_world}

{brief_tri}

{brief_split}

{human_subj}

---

# SLOT 2 — {{seed}} · THE GOLDEN SEED (FROZEN)

{seed_core}

---

## THE GOLDEN SEED SELECTION RECORD

{seed_golden}

---

# SLOT 3 — {{meta_prompt}} · THE METAPROMPT

{meta}

---

# SLOT 4 — {{personality}} · FULL DNA — LOFN-PRIME

> ⛔ A name reference ("voice = LOFN-PRIME") is a FAIL. The complete DNA below is why the runs win.
> ⛔ **LOFN-PRIME is the only seat in this run.** No other personality's DNA is present in this block.

## ═══ LOFN-PRIME — full YAML DNA, golden-output archive quarantined ═══

{lofn_dna}

---

# THE 18 VOICES — CANONICAL ROSTER (machine-checkable: exactly 18 `VOICE NN` rows, 3 Hyper-Skeptics)

{roster}

**Continuity payload carried: the Golden Seed · the metaprompt · the full LOFN-PRIME DNA block · all 18
voices with their objections · all 15 Special Flairs · the three transformation configurations · the six
binding rulings · the genre and frames palettes · the GOLDEN MOVE.**

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

# SLOT 7b — THE TRANSFORMATIONS (BASELINE → GROUP:REFLECT → SKEPTIC:BRIDGE)

{transforms}

---

# SLOT 7c — THE SIX RULINGS THAT BIND THE RUN

{rulings}

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
   listen. Obliqueness about the referent is fog, not depth. An unnameable subject is `REPAIR — FOG`.
7. **THE RETURN.** Song is made of returns. Removal is a debt: strip rhyme only by naming what returns in
   its place. **A byte-identical chorus is correct and requires no note.**

**Golden Songs for this run (NAMES ONLY — judge-side blind comparison, payloads quarantined):**
`Triple Arch Over Me` · `Blue Screen Lullaby`.

---

*ICB assembled by extraction from disk, not retyped. Frozen at Phase 1.* 💜
"""

(RUN / "CREATIVE_CONTEXT.md").write_text(out, encoding="utf-8")
n = len(out.encode("utf-8"))
voices = out.count("(after ")
voice_rows = out.count("| VOICE 0") + out.count("| VOICE 1")
print(f"ICB written: {n:,} bytes")
print(f"  '(after ' credit strings   : {voices}")
print(f"  countable 'VOICE NN' rows  : {voice_rows}")
print(f"  LOFN-PRIME DNA             : {len(lofn_dna):,} chars ({lofn_cut:,} quarantined)")
print(f"  genres                     : {len(genres)}")
print(f"  frames                     : {len(frames)}")
if voice_rows != 18:
    sys.exit(f"FATAL: expected 18 VOICE rows, got {voice_rows}")
if lofn_cut < 60000:
    sys.exit(f"FATAL: quarantine cut only {lofn_cut} bytes — the archive block was not found.")
print("OK — 18 voices, archive quarantined.")
