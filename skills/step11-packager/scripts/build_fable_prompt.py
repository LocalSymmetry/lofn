#!/usr/bin/env python3
"""Build a complete Fable-5 refinement prompt from step10 + step11 output.

Usage: python3 build_fable_prompt.py <pair_dir> <personality_name> <personality_yaml> [--output <path>]
"""

import os, sys, argparse

PERS_DIR = "/data/.openclaw/workspace/skills/orchestration/personalities"
RULES_PATH = "/data/.openclaw/workspace/skills/step11-packager/references/suno_rules_condensed.md"

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

## PRODUCTION MANDATES

{production_mandates}

---

## REFINEMENT INSTRUCTIONS

You are the final refinement pass before Suno v5.5 generation. The full Step 10 output (all variants, selections, draft prompts) is below, followed by the Step 11 enhanced package to refine.

**Your task:** REFINE the Suno style prompt into a dense prose paragraph strictly following the 7-position order and principles above. REFINE the lyrics with deeper mythic/fable-like narrative quality. PRESERVE everything else unchanged.

Enhance the lyrics with:
- Sharper imagery, poetic density, emotional weight
- Deeper consonance, assonance, and sonic texture within the verse form
- Lines that feel more inevitable, more mythic, more alive
- Full channeling of the personality above - every line unmistakably {personality_name}

## NON-NEGOTIABLE CONSTRAINTS
- Suno prompt: Dense prose paragraph ONLY, <=1000 chars, follows 7-position order, NO yaml, NO brackets
- Lyrics: <=5000 chars total. [Theme:] + [SONG FORM:] first. 5-line Disc_Channel block. Full EMO section headers. Body noise mandate.
- INTRO must specify FIRST SOUND (first 1-5 seconds). Spatial staging per section. Hard-gated silence durations exact.

## CRITICAL PRESERVATION RULES
1. PRESERVE everything EXCEPT the Suno prompt + lyrics blocks
2. REFINE Suno prompt -> dense prose <=1000 chars per the 7-position order above
3. REFINE lyrics with deeper fable-like quality preserving all structure
4. Preserve ALL [11] markers, EMO tags, production cues, section headers exactly
5. Undecidable element MUST remain undecidable

## OUTPUT
Return the COMPLETE refined step11 file - every section from ICB through final verification, with ONLY the Suno prompt and lyrics blocks changed. No commentary. The full file.

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
