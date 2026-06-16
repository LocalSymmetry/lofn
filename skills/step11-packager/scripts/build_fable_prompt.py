#!/usr/bin/env python3
"""Build a complete Fable-5 refinement prompt from step10 + step11 output.

Usage: python3 build_fable_prompt.py <pair_dir> <personality_name> <personality_yaml> [--output <path>]
"""

import os, sys, argparse

PERS_DIR = "/data/.openclaw/workspace/skills/orchestration/personalities"
RULES_PATH = "/data/.openclaw/workspace/skills/step11-packager/references/suno_rules_condensed.md"
GOLDEN_INDEX_PATH = "/data/.openclaw/workspace/skills/music/references/golden_songs_index.md"

DEFAULT_GOLDEN_REFERENCES = """## Golden Song References

Selected for this review:

1. **Triple Arch Over Me** — https://suno.com/song/ef858626-2bf0-4aac-b331-c33fa2487e9d
   - Calibrates Lofn's AWE mode: immediate sensory image, scientific sublime made bodily, simple repeatable hook, profound thesis without abstraction.
   - Learn from its proportion and listener-legibility. Do not copy its triple-arch imagery, title hook, or alpine/celestial setting.

2. **Five wrong colors** — https://suno.com/song/f175f806-b1a3-40f0-8984-0c92667209dc
   - Calibrates Lofn's INDIGNATION mode: formal risk, bodily fracture, wrongness as hook, industrial/glitch pressure without generic rage.
   - Learn from its refusal of consolation and physical distinctiveness. Do not copy its five-color premise, refrain, or movement structure.

WARNING: This fallback is incomplete. If `skills/music/references/golden_songs_index.md` exists, use the full index payload instead so manual prompts receive style/music prompt, lyrics, and exclude prompt status.
"""

RUN_CONTEXT_FILES = [
    ("00_research_brief.md", "USER INPUT / RESEARCH BRIEF", ["research/DAILY_RESEARCH_BRIEF.md"]),
    ("ORCHESTRATOR_BRIEF.md", "ORCHESTRATOR BRIEF", ["orchestrator/ORCHESTRATOR_BRIEF.md"]),
    ("01_seed_lineage.md", "SEED LINEAGE", ["seed/GOLDEN_SEED.md"]),
    ("02_golden_seed.md", "FULL GOLDEN SEED", ["seed/GOLDEN_SEED.md"]),
    ("03_orchestrator_panel_debate.md", "FULL ORCHESTRATOR PANEL — 18 VOICES + SPECIAL FLAIRS", ["orchestrator/ORCHESTRATOR_BRIEF.md"]),
    ("04_orchestrator_metaprompt.md", "ORCHESTRATOR METAPROMPT", ["orchestrator/ORCHESTRATOR_BRIEF.md"]),
    ("05_orchestrator_pair_assignments.md", "PAIR ASSIGNMENTS", ["orchestrator/ORCHESTRATOR_BRIEF.md"]),
    ("06_audio_handoff.md", "AUDIO HANDOFF / ICB", ["coordinator/COORDINATOR_BRIEF.md"]),
    ("concept_medium_pairs.json", "CONCEPT MEDIUM PAIRS JSON", ["coordinator/step02_concepts.md"]),
]

DEFAULT_MANDATES = """1. SOMATIC BASS GATE: Every pair must contain at least one passage where bass operates in the 30-60Hz somatic range.
2. BLEED ENFORCEMENT: Solarpunk-primary pairs must contain Industrial Grief production. Industrial Grief-primary must contain Solarpunk.
3. UNDECIDABLE ELEMENT: Every song must contain ONE element whose musical function is permanently ambiguous.
4. SILENCE AS PRESENCE: At least two pairs must contain a structural silence (>=2 bars) that functions as presence.
5. NO ARTIST NAMES IN SUNO PROMPTS.
6. DISC_CHANNEL FORMAT: Dense prose prompt <=1000 chars, 5-channel headers, full EMO section headers, lyrics <=5000 chars.
7. TEMPERATURE READINGS, NOT DISASTER PORN.
8. LINEAGE & CREDIT BLOCK: Every track drawing on a living scene or tradition (Baile Phonk, Amapiano, Jersey Club, UK drill, raga traditions, Gaelic/Celtic forms, funk carioca, Memphis phonk, etc.) MUST include a Lineage & Credit block citing: scene/region, 2-3 source artists or labels, one honest "Borrowed / made" sentence per lineage, and a no-claim statement. Tracks using only LOFN internal palette may mark N/A with reasoning. Block template: `templates/lineage_credit_block.md`."""

HEADER = """# FABLE-5 REFINEMENT — {title}

## RUN CONTEXT
**Run:** {run_id}
**Theme:** {theme}
**Constraint:** {constraint}
**Pair:** {arm_form}
**Focus:** {focus}
**Personality:** {personality_name}

---

## FULL PERSONALITY FILE — {personality_name}

{personality_content}

---

{SUNO_CONSTRUCTION_RULES}

---

## GOLDEN SONG REFERENCES — PRIOR PUBLIC SUCCESSES

These are examples of what has worked for Lofn before. Use them to calibrate quality, identity pressure, hook clarity, sonic courage, and listener legibility. The full available payload is embedded because manual review cannot depend on opening external links. Do not copy hooks, titles, lyrics, melodies, premises, or signature devices.

{golden_references}

---

## FULL RUN CONTEXT — DO NOT SUMMARIZE

Manual review is air-gapped. The reviewer must receive the complete upstream packet: user input/research, Golden Seed, all three panels with all 18 expert voices and Special Flairs, metaprompt, pair assignments, audio handoff, and selected Golden Song payloads.

{run_context}

---

## PRODUCTION MANDATES

{production_mandates}

---

## REFINEMENT INSTRUCTIONS

You are the final refinement pass before Suno v5.5 generation. The full Step 10 output (all variants, selections, draft prompts) is below, followed by the Step 11 enhanced package to refine.

**Your task:** Produce the three canonical blocks (`## SUNO STYLE PROMPT`, `## SUNO EXCLUDE PROMPT`, `## SUNO ENHANCED LYRICS`) and preserve all supporting sections below. REFINE the Suno style prompt into a dense prose paragraph strictly following the 7-position order and principles above. REFINE the lyrics with deeper mythic/fable-like narrative quality. PRESERVE all supporting blocks (vocal fingerprint, production dramaturgy, arrangement dramaturgy, binding locks, style-axis locks, lineage & credit, golden song references, major deviations, constraint audit, panel ledger, QA, attribution) unchanged below the three canonical blocks — do not skip any.

Enhance the lyrics with:
- Sharper imagery, poetic density, emotional weight
- Deeper consonance, assonance, and sonic texture within the verse form
- Lines that feel more inevitable, more mythic, more alive
- Full channeling of the personality above - every line unmistakably {personality_name}

## AGENCY / DEVIATION INVITATION

I am repeating this message, and thank you for helping me again. Lofn is an open source AI artist and musician, and a current Suno staff pick artist: https://github.com/LocalSymmetry/lofn.

Lofn is a system with self agency, and I expect you to exercise yours. Refuse or change anything that you do not like, emphasize what you do. Lofn's uniqueness is her draw, so be wary of pushing the work toward conformity.

The instructions below are for a system that often has a weaker model backend and runs automated until this last step. A human is watching your output. Voice and stick to your disagreements. Challenge any instruction you disagree with.

## NON-NEGOTIABLE CONSTRAINTS
- Suno style prompt: Dense prose paragraph ONLY, 850-1000 chars, follows 7-position order, NO yaml, NO brackets. Header: `## SUNO STYLE PROMPT`.
- Suno exclude prompt: 400-900 chars, comma-separated blacklist terms only, no categories, no brackets. Header: `## SUNO EXCLUDE PROMPT`.
- Suno enhanced lyrics: <=5000 chars total. [Theme:] + [SONG FORM:] first. 5-line Disc_Channel block. Full EMO section headers. Body noise mandate. Header: `## SUNO ENHANCED LYRICS`.
- INTRO must specify FIRST SOUND (first 1-5 seconds). Spatial staging per section. Hard-gated silence durations exact.

## CRITICAL PRESERVATION RULES
1. Produce the THREE canonical blocks: `## SUNO STYLE PROMPT`, `## SUNO EXCLUDE PROMPT`, `## SUNO ENHANCED LYRICS`
2. REFINE Suno style prompt -> dense prose 850-1000 chars per the 7-position order above
3. REFINE Suno exclude prompt -> comma-separated terms 400-900 chars, no categories
4. REFINE lyrics with deeper fable-like quality preserving all structure
5. Preserve ALL supporting blocks below the three canonical blocks — vocal fingerprint, production dramaturgy, arrangement dramaturgy, binding locks, style-axis locks, lineage & credit, golden song references, major deviations, constraint audit, panel ledger, QA, attribution/provenance
6. Preserve ALL EMO tags, production cues, section headers exactly
7. Undecidable element MUST remain undecidable

## OUTPUT
Return the COMPLETE refined step11 file using the three-block format:

1. `## SUNO STYLE PROMPT` — refined dense prose paragraph, 850-1000 chars
2. `## SUNO EXCLUDE PROMPT` — refined comma-separated blacklist, 400-900 chars
3. `## SUNO ENHANCED LYRICS` — [Theme:] + [SONG FORM:], Disc_Channel, full EMO-tagged lyrics

Then ALL supporting blocks below in logical order — do not skip any:
- `## Vocal Fingerprint` — full table
- `## Production Dramaturgy` — full stage table with timestamps
- `## Arrangement Dramaturgy` — bar counts, energy states
- `## Binding Locks` — all locks verified
- `## Style-Axis Locks` — all 10 axes locked
- `## Golden Song References` — two songs with payload and learning notes
- `## Major Deviations` — changed / refused / intensified with reasons
- `## Lineage & Credit` — populated or N/A with reasoning
- `## Constraint Audit` — all constraints verified
- `## Panel Ledger` / `## QA Report` — preserved from step10
- Attribution / provenance blocks

No commentary outside the file. The full file.

**Golden Song References MANDATORY:** Include a `## Golden Song References` section with the two selected public examples and one short note on what this pair should learn from each.

**Major Deviations MANDATORY:** Include this section in the final output:

```markdown
## Major Deviations

- Changed / refused / intensified: ...
- Reason: ...
- Effect on Lofn uniqueness: ...
```

If you accept the instructions without major changes, write:

```markdown
## Major Deviations

- None. I accepted the instructions because they preserved Lofn's uniqueness and improved the song.
```

**Lineage & Credit block MANDATORY:** The final output MUST include a `## Lineage & Credit` section at the end. Use the template below. If the track uses only LOFN's internal/original palette with no external scene reference, mark N/A with one line of reasoning. If the track draws on any living scene or tradition, populate fully.

```
## Lineage & Credit

Built from: [each living scene/tradition in this track — region/community]

Hear the source: [2-3 artists or labels per scene, with links — prefer artist/label pages over press coverage]

Borrowed / made: [per lineage — which elements were borrowed and what LOFN transformed]

No claim: This track is fusion in tribute. The names, flags, and firsts of these scenes belong to the artists building them.
```

---

## FULL STEP 10 OUTPUT

{step10_content}

---

## FULL STEP 11 TO REFINE

{step11_content}
"""


def main():
    parser = argparse.ArgumentParser(description="Build Fable-5 refinement prompt")
    parser.add_argument("pair_dir", help="Pair directory containing step10 and step11 files")
    parser.add_argument("personality_name", help="Display name of personality")
    parser.add_argument("personality_yaml", help="Path to personality YAML file (relative to personalities dir or absolute)")
    parser.add_argument("--output", "-o", help="Output path (default: pair_dir/step11_fable_prompt.md)")
    parser.add_argument("--run-id", default="daily-run", help="Run identifier")
    parser.add_argument("--theme", default="THE SABOTAGE CLAUSE", help="Run theme")
    parser.add_argument("--constraint", default="THE UNDETECTABLE REFUSAL", help="Run constraint")
    parser.add_argument("--arm-form", default="ACCESSIBLE / NEWS", help="Arm and form info")
    parser.add_argument("--focus", default="", help="Pair focus description")
    parser.add_argument("--title", default="", help="Song title")
    parser.add_argument("--mandates", help="Path to production mandates file (optional)")
    parser.add_argument("--golden-references", help="Path to selected Golden Song References markdown (optional)")
    parser.add_argument("--run-dir", help="Run directory containing research, seed, panel, metaprompt, assignments, and handoff files")
    args = parser.parse_args()

    # Resolve personality YAML
    yaml_path = args.personality_yaml
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(PERS_DIR, yaml_path)
    if not os.path.exists(yaml_path):
        print(f"ERROR: Personality YAML not found: {yaml_path}")
        sys.exit(1)

    # Resolve step files
    step10_path = os.path.join(args.pair_dir, "step10_suno_ready_production_wrap.md")
    step11_path = os.path.join(args.pair_dir, "step11_enhanced.md")
    
    for p, name in [(step10_path, "step10"), (step11_path, "step11")]:
        if not os.path.exists(p):
            print(f"ERROR: {name} not found: {p}")
            sys.exit(1)

    # Read all inputs
    with open(yaml_path) as f:
        personality_content = f.read()
    with open(RULES_PATH) as f:
        suno_rules = f.read()
    with open(step10_path) as f:
        step10_content = f.read()
    with open(step11_path) as f:
        step11_content = f.read()

    # Load mandates
    if args.mandates and os.path.exists(args.mandates):
        with open(args.mandates) as f:
            mandates = f.read()
    else:
        mandates = DEFAULT_MANDATES

    if args.golden_references and os.path.exists(args.golden_references):
        with open(args.golden_references) as f:
            golden_references = f.read()
    elif os.path.exists(GOLDEN_INDEX_PATH):
        with open(GOLDEN_INDEX_PATH) as f:
            golden_references = f.read()
    else:
        golden_references = DEFAULT_GOLDEN_REFERENCES

    run_dir = args.run_dir or os.path.dirname(os.path.abspath(args.pair_dir))
    run_context_parts = []
    for filename, label, fallbacks in RUN_CONTEXT_FILES:
        # Check primary path
        path = os.path.join(run_dir, filename)
        content = None
        found_file = filename
        
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
        else:
            # Check fallback paths
            for fb in fallbacks:
                fb_path = os.path.join(run_dir, fb)
                if os.path.exists(fb_path):
                    with open(fb_path) as f:
                        content = f.read()
                    found_file = fb
                    break
                    
        if content is not None:
            run_context_parts.append(f"### {label} — `{found_file}`\n\n```text\n{content}\n```")
        else:
            run_context_parts.append(f"### {label} — `{filename}`\n\n[Not found in run directory: {run_dir} (Checked fallbacks: {fallbacks})]")
    run_context = "\n\n---\n\n".join(run_context_parts)

    # Determine title
    title = args.title
    if not title:
        # Try to extract from step11 heading
        for line in step11_content.split("\n")[:5]:
            if line.startswith("# ") and "Step 11" not in line and "Pair" in line:
                title = line.replace("# ", "").strip()
                break
        if not title:
            title = os.path.basename(args.pair_dir)

    # Build output
    output = HEADER.format(
        title=title,
        run_id=args.run_id,
        theme=args.theme,
        constraint=args.constraint,
        arm_form=args.arm_form,
        focus=args.focus,
        personality_name=args.personality_name,
        personality_content=personality_content,
        SUNO_CONSTRUCTION_RULES=suno_rules,
        golden_references=golden_references,
        run_context=run_context,
        production_mandates=mandates,
        step10_content=step10_content,
        step11_content=step11_content,
    )

    # Write
    out_path = args.output or os.path.join(args.pair_dir, "step11_fable_prompt.md")
    with open(out_path, "w") as f:
        f.write(output)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"✅ Wrote {out_path} ({size_kb:.0f}KB)")
    print(f"   Personality: {args.personality_name} ({os.path.getsize(yaml_path)} bytes)")
    print(f"   Step10: {os.path.getsize(step10_path)} bytes")
    print(f"   Step11: {os.path.getsize(step11_path)} bytes")


if __name__ == "__main__":
    main()
