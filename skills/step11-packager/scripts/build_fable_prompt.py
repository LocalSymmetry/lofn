#!/usr/bin/env python3
"""General-purpose all-pairs Step 11 prompt builder.

Auto-discovers a run directory, finds all pair step10 files, resolves
personalities from the orchestrator brief, and builds one complete
paste-ready prompt per pair.

Usage:
    # Build all pairs (auto-detect everything):
    python3 build_fable_prompt.py /path/to/run/dir

    # Build all pairs with explicit overrides:
    python3 build_fable_prompt.py /path/to/run/dir --mode fable

    # Build a single pair:
    python3 build_fable_prompt.py /path/to/run/dir --pairs 3

    # Custom output dir:
    python3 build_fable_prompt.py /path/to/run/dir --output-dir /tmp/prompts

Modes:
    fable       — Full refinement prompt with instructions (default, no step11 needed)
    opus        — Same as fable, named for manual Opus paste

Run directory auto-discovery:
    Expects: seed/GOLDEN_SEED.md, orchestrator/ORCHESTRATOR_BRIEF.md,
             audio/pair_XX_step10_production_wrap.md (6 pairs)
    Falls back gracefully for missing optional files.
"""

import os, sys, glob, re, argparse, json

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = "/data/.openclaw/workspace"
PERS_DIR = os.path.join(WORKSPACE, "skills/orchestration/personalities")
RULES_PATH = os.path.join(SKILL_DIR, "references/suno_rules_condensed.md")
GOLDEN_INDEX_PATH = os.path.join(WORKSPACE, "skills/music/references/golden_songs_index.md")
DEFAULT_PERSONALITY = "lofn-prime-mini.yaml"

# These are the canonical run-context files to embed, in order.
# Format: (label, primary_path, fallback_paths)
RUN_CONTEXT_SPEC = [
    ("GOLDEN SEED", "seed/GOLDEN_SEED.md", []),
    ("ORCHESTRATOR BRIEF", "orchestrator/ORCHESTRATOR_BRIEF.md", []),
    ("CONCEPT MEDIUM PAIRS", "coordinator/concept_medium_pairs.json", []),
]

DEFAULT_MANDATES = """1. SOMATIC BASS GATE: Every pair must contain at least one passage where bass operates in the 30-60Hz somatic range.
2. BLEED ENFORCEMENT: Vocal bleed into cathedral reverb (3-7s decay) must create structural tension.
3. UNDECIDABLE ELEMENT: Every song must contain ONE element whose musical function is permanently ambiguous.
4. SILENCE AS PRESENCE: At least two pairs must contain a structural silence (>=2 bars) that functions as presence.
5. NO ARTIST NAMES IN SUNO PROMPTS.
6. DISC_CHANNEL FORMAT: Dense prose style prompt <=1000 chars, exclude prompt 400-900 chars, lyrics <=5000 chars. All three text blocks have equal binding character limits — treat them with the same enforcement as the style prompt. 5-channel headers, full EMO section headers.
7. TEMPERATURE READINGS, NOT DISASTER PORN.
8. LINEAGE & CREDIT BLOCK: Every track drawing on a living scene or tradition MUST include a Lineage & Credit block."""

MANUAL_REFINEMENT_BLOCK = """---

## MANUAL REFINEMENT INSTRUCTIONS

**This is the task block. Execute this LAST, after reading every source artifact and rule provided above.**

### 1. READ EVERYTHING FIRST
Absorb every source artifact before writing a single word. Understand the creative intent, emotional payload, sonic world, and constraint-as-form. Do not skim. Do not proceed until the song's reason for existing is clear.

### 2. REFINE AND ENHANCE ALL COMPONENTS — THE THREE CANONICAL BLOCKS PLUS ALL SUPPORTING SECTIONS
You have complete creative authority. The output MUST use the three-block format:

**## SUNO STYLE PROMPT — PRIMARY:** Tighten to producer-grade density. Rewrite from scratch if vague, narrative, or procedural. Lead with genre/tempo/key/432Hz, then vocalist, instrumentation, arrangement arc, signature device. Dense paragraph, 850-1000 chars. HARD LIMIT 1,000 characters. See Section 5 for construction rules. ONE continuous prose paragraph. The block header is literally `## SUNO STYLE PROMPT` — no variations, no abbreviations.

**## SUNO EXCLUDE PROMPT — PRIMARY:** Concrete comma-separated blacklist terms. 400-900 chars. HARD LIMIT 900 characters. Rewrite if prose-y or thin. No categories, no brackets, no headers — just terms separated by commas. This is a negative-control field for Suno's parser. The block header is literally `## SUNO EXCLUDE PROMPT`.

**## SUNO ENHANCED LYRICS — PRIMARY:** Rewrite, restructure, rebuild. Elevate to literary density. Apply at least one structural transformation. If a verse is generic, replace it wholesale. If the bridge doesn't earn its place, cut it or reinvent it. If the song form fights the emotional arc, change the form. Must open with [Theme:] + [SONG FORM:] lines. Disc_Channel block immediately following. Then full lyrics with EMO tags INTEGRATED INTO EVERY SECTION HEADER. Minimum 60 sung lines. HARD LIMIT: 5,000 characters. Suno rejects lyrics exceeding this bound. Count the entire lyrics block (sung text only, not metadata/headers). The block header is literally `## SUNO ENHANCED LYRICS`.

### CRITICAL: Disc_Channel Format — EXACT EXAMPLE, DO NOT DEVIATE

The Disc_Channel block is a 5-line producer channel strip — each line is a `[Disc_NAME: token | token | ...]` bracket. Pipe-separated production tokens within a single bracket per channel:

```text
[Disc_Rhythm: LinnDrum_100BPM | Gqom_3-3-2_broken_kick | bone_dry_no_fills | Center_Mono]
[Disc_Vocal: dry_sardonic_delivery | ASMR_close_mic | anti-diva_deadpan | breath_on_capsule | Center_Front]
[Disc_Sub: FM_sine_38-42Hz | continuous_swell_+0.5dB_per_8bars | NEVER_RESOLVES | Mono_Sub_Lock]
[Disc_Pad: green_synth_432Hz | El_Niño_Deep_Blue | slow_attack_swell | Stereo_Width_Maximum]
[Disc_Texture: cassette_tape_saturation | telephone_bandpass_break | Wall_of_Sound_layering | Hard_Pan_Right]
```

Five channels minimum: Rhythm, Vocal, Sub, Pad, Texture. Every token is a concrete production decision — oscillator type, BPM, mic technique, processing chain. NOT run metadata. NOT markdown headers.

### CRITICAL: EMO Tag Format — EXACT EXAMPLE, DO NOT DEVIATE

EMO tags MUST be integrated into section headers using `–` (em dash) separators. Emotional states use commas, not `to` connectors. `[11]` markers embed where step11 enhancement touched that section:

```text
[Septet 1 – 0.1°C – EMO:Sardonic Cool, Deflected Warmth [11] – Reluctant Pop Star, dry close-mic, slow internal rhymes]
the water took a breath so slow you thought your skin
had loosed itself and let the silence in
```

Header structure: `[SECTION LABEL – VARIANT/CUE – EMO:State1, State2 [11] – PERSONA NAME, vocal style, delivery notes]` — all separated by ` – ` (space-emdash-space) within a single bracket. EMO values use commas between related emotional states. `[11]` markers embed where the step11 enhancement touched.

**NEVER:** standalone `[EMO=reverence]` on its own line, EMO without persona and delivery style, bare `[EMO]` tags, or `-` (hyphen) where `–` (em dash) is the separator.

### ALL SUPPORTING BLOCKS BELOW THE THREE — DO NOT SKIP ANY:
- EMO Tags: Integrated into every section header. Specific, embodied, emotional arc transforms across the song.
- Disc_Channel Headers: 5-line channel strip per the EXACT format above.
- Vocal Fingerprint: Mic distance, compression, breath placement, spatial assignment. Full table.
- Production Dramaturgy: Every unusual sound has a dramatic job. Full stage table with timestamps.
- Arrangement Dramaturgy: Section-by-section bar counts and energy states.
- Binding Locks: Verify, strengthen, or challenge in Major Deviations.
- Style-Axis Locks: Tempo, mood, instrumentation, lyrics, genre, vocal, rhythm, melody, harmony, production — all 10 locked.
- Lineage & Credit: Every track drawing on a living scene MUST credit it.
- Golden Song References: Two selected songs with full payload and learning notes.
- Major Deviations: Changed / refused / intensified with reasons.
- Constraint Audit: Verify all constraints survived.
- Panel Ledger / QA Report: Preserved from step10.
- Attribution / Provenance blocks: Artist name, run ID, personality, pipeline stage.

### 3. WHOLESALE CHANGES ARE ALLOWED
You are producing, not polishing. Rewrite entire verses. Restructure the song form. Replace the production palette. Change the emotional arc. The ONLY unbreakable bonds: invariant hook phrase, genre string, BPM, key, 432Hz tuning. Everything else is negotiable if the change makes a better song. The song matters more than the provenance.

### 4. FINAL VERIFICATION — THREE-BLOCK + ALL SUPPORTING SECTIONS CHECKLIST
- `## SUNO STYLE PROMPT`: 850-1000 chars, one paragraph, procedural-free, genre-first, 7-position order
- `## SUNO EXCLUDE PROMPT`: 400-900 chars, comma-separated terms only, no categories, no brackets
- `## SUNO ENHANCED LYRICS`: [Theme:] + [SONG FORM:], 5-line Disc_Channel channel strip, integrated EMO section headers using `–` em dashes, >=60 sung lines, sung text <=5,000 chars (Suno hard limit), literary density, structural transformation
- Disc_Channel: 5-line channel strip — PRESENT (above the lyrics)
- EMO Tags: Integrated into section headers using `–` em dash separators — PRESENT on every section
- Vocal Fingerprint: full table — PRESENT
- Production Dramaturgy: full stage table with timestamps — PRESENT
- Arrangement Dramaturgy: bar counts, energy states — PRESENT
- Binding Locks: all locks verified — PRESENT
- Style-Axis Locks: all 10 axes locked — PRESENT
- Lineage & Credit: populated or N/A with reasoning — PRESENT
- Golden Song References: two songs with payload and learning notes — PRESENT
- Major Deviations: stated explicitly — PRESENT
- Constraint Audit: all constraints verified — PRESENT
- Panel Ledger / QA: preserved from step10 — PRESENT
- Attribution / Provenance — PRESENT
- No anti-patterns: no ###, no emoji headers, no summary EMO, no artist names, no procedural openings

### 5. SUNO v5.5 PROMPT CONSTRUCTION RULES
Apply the Seven Core Principles and Mandatory 7-Position Order from the Suno rules above. Character count: 850-1000. HARD LIMIT 1000. ONE continuous dense prose paragraph. NO bracketed [key:value] tags. NO artist names. NO procedural openings. NO bare nouns.
"""


def load_file(path):
    """Load a file, return its content or None."""
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None


def discover_pairs(run_dir):
    """Find all pair step10 files and extract pair numbers and titles.

    Returns list of (pair_num, step10_path, track_title).
    """
    audio_dir = os.path.join(run_dir, "audio")
    if not os.path.isdir(audio_dir):
        return []

    pattern = os.path.join(audio_dir, "pair_*_step10_production_wrap.md")
    files = sorted(glob.glob(pattern))
    pairs = []
    for f in files:
        m = re.match(r".*/pair_(\d+)_step10_production_wrap\.md", f)
        if not m:
            continue
        pair_num = int(m.group(1))
        # Try to extract title from the file
        title = ""
        content = load_file(f)
        if content:
            for line in content.split("\n")[:5]:
                if line.startswith("# Step 10") or line.startswith("# Production Wrap"):
                    # Try to grab the sub-line with title from step10 header
                    for sub in content.split("\n")[1:8]:
                        if sub.startswith("**Pair:**") or " — " in sub:
                            parts = sub.split("—")
                            if len(parts) > 1:
                                title = parts[-1].strip().strip('"')
                                break
                if title:
                    break
            if not title:
                # Fallback: search for "Pair XX" naming
                for line in content.split("\n")[:15]:
                    if f"Pair {pair_num:02d}" in line and ":" in line:
                        title = line.split(":", 1)[1].strip().strip('"')
                        break
        pairs.append((pair_num, f, title or f"Pair {pair_num:02d}"))
    return pairs


def resolve_personalities(run_dir, pairs):
    """Read orchestrator brief to determine personality per pair.

    Returns dict {pair_num: (personality_name, yaml_path)}.
    Falls back to LOFN-PRIME for any pair we can't determine.
    """
    brief_path = os.path.join(run_dir, "orchestrator", "ORCHESTRATOR_BRIEF.md")
    brief = load_file(brief_path) or ""
    
    personalities = {}
    default_yaml = os.path.join(PERS_DIR, DEFAULT_PERSONALITY)
    
    # Strategy: scan for "FULL ... PERSONALITY DNA BLOCK" sections in the brief
    # and pair them with nearby PAIR N: headers
    current_persona = "LOFN-PRIME (AWE mode)"
    current_yaml = default_yaml
    
    # Search for personality assignments
    # Pattern 1: explicit YAML blocks with name
    # Pattern 2: "PAIR N:" sections that may reference a personality
    
    pair_pattern = re.compile(r'(?:###\s*)?PAIR\s*(\d+)[\s:]+(.+?)(?:\n|$)')
    persona_pattern = re.compile(r'##\s*FULL\s+(.+?)\s+PERSONALITY\s+DNA\s+BLOCK', re.IGNORECASE)
    
    # Look for personality blocks tied to pair sections
    sections = brief.split("\n### PAIR ")
    for section in sections:
        m = pair_pattern.match(section) if section.startswith("PAIR") or section[0].isdigit() else None
        if not m and sections.index(section) == 0:
            continue  # header section before first pair
        pair_num = int(m.group(1)) if m else 0
        if pair_num == 0:
            continue
        
        # Check for personality reference in this section
        # Look for "LOFN-PRIME" or named personality
        if "LOFN-PRIME" in section:
            mode = "AWE mode"
            if "INDIGNATION" in section[:500]:
                mode = "INDIGNATION mode"
            personalities[pair_num] = (f"LOFN-PRIME ({mode})", default_yaml)
        else:
            # Try to find a named personality
            p_match = persona_pattern.search(section)
            if p_match:
                name = p_match.group(1).strip()
                yaml_path = os.path.join(PERS_DIR, f"{name.lower().replace(' ', '-')}.yaml")
                if not os.path.exists(yaml_path):
                    yaml_path = default_yaml
                    name = f"{name} (fallback: LOFN-PRIME)"
                personalities[pair_num] = (name, yaml_path)
            else:
                personalities[pair_num] = ("LOFN-PRIME (AWE mode)", default_yaml)
    
    # Fill in any missing pairs
    for pair_num, _, _ in pairs:
        if pair_num not in personalities:
            personalities[pair_num] = ("LOFN-PRIME (AWE mode)", default_yaml)
    
    return personalities


def extract_run_metadata(run_dir):
    """Extract run name, date, theme from seed + orchestrator files."""
    seed = load_file(os.path.join(run_dir, "seed", "GOLDEN_SEED.md")) or ""
    brief = load_file(os.path.join(run_dir, "orchestrator", "ORCHESTRATOR_BRIEF.md")) or ""
    
    metadata = {
        "run_id": os.path.basename(run_dir.rstrip("/")),
        "theme": "",
        "constraint": "Constraint-as-Form",
        "lens": "",
    }
    
    # Extract run title from seed header
    for line in seed.split("\n")[:5]:
        if line.startswith("# ") and "GOLDEN SEED" in line:
            title_part = line.split("—", 1)[-1].strip().strip('"')
            if title_part and title_part != line:
                metadata["theme"] = title_part
        if "Lens:" in line:
            metadata["lens"] = line.split("Lens:", 1)[1].strip()
    
    # Extract from orchestrator
    for line in brief.split("\n")[:20]:
        if "Core Tone:" in line or "AESTHETIC DIRECTIVE" in line:
            continue
        if "Theme:" in line and not metadata["theme"]:
            metadata["theme"] = line.split("Theme:", 1)[1].strip()
    
    if not metadata["theme"]:
        metadata["theme"] = metadata["run_id"]
    
    return metadata


def build_run_context(run_dir):
    """Assemble the full run context block from seed + orchestrator files."""
    parts = []
    for label, primary, fallbacks in RUN_CONTEXT_SPEC:
        content = load_file(os.path.join(run_dir, primary))
        if content is None:
            for fb in fallbacks:
                content = load_file(os.path.join(run_dir, fb))
                if content:
                    break
        if content:
            parts.append(f"### {label}\n\n{content}")
        else:
            parts.append(f"### {label}\n\n[File not found: {primary}]")
    return "\n\n---\n\n".join(parts)


def build_prompt(pair_num, step10_path, title, persona_name, yaml_path,
                 run_dir, metadata, mode):
    """Build a single pair's complete prompt file."""
    
    step10 = load_file(step10_path) or f"[ERROR: Could not read {step10_path}]"
    personality_yaml = load_file(yaml_path) or f"[ERROR: Personality YAML not found: {yaml_path}]"
    suno_rules = load_file(RULES_PATH) or "[ERROR: Suno rules not found]"
    golden_refs = load_file(GOLDEN_INDEX_PATH) or ""
    run_context = build_run_context(run_dir)
    
    mode_label = {"fable": "FABLE/OPUS", "opus": "OPUS"}.get(mode, "FABLE/OPUS")
    
    # Header
    header = f"""# {mode_label} REFINEMENT PROMPT — Pair {pair_num:02d}: "{title}"

## RUN CONTEXT

**Run:** {metadata['theme']}
**Run ID:** {metadata['run_id']}
**Pair:** {pair_num:02d} — "{title}"
**Personality:** {persona_name}
**Mode:** {mode_label}
{f"**Lens:** {metadata['lens']}" if metadata['lens'] else ""}

---
"""

    prompt = (
        header
        + f"\n## PERSONALITY YAML — FULL ARCHIVE FILE ({persona_name})\n\n{personality_yaml}\n\n---\n\n"
        + f"## SUNO v5.5 PROMPT CONSTRUCTION RULES\n\n{suno_rules}\n\n---\n\n"
    )
    
    if golden_refs:
        prompt += f"## GOLDEN SONG REFERENCES INDEX\n\n{golden_refs}\n\n---\n\n"
    
    prompt += (
        f"## FULL RUN CONTEXT\n\n{run_context}\n\n---\n\n"
        + f"## PRODUCTION MANDATES\n\n{DEFAULT_MANDATES}\n\n---\n\n"
        + f"## STEP 10 — PRODUCTION WRAP (PAIR {pair_num:02d})\n\n{step10}\n\n"
    )
    
    # Add refinement instructions (all modes)
    prompt += MANUAL_REFINEMENT_BLOCK
    
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Build all-pair Step 11 manual refinement prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s /path/to/run                    # All pairs, fable mode
  %(prog)s /path/to/run --pairs 1,3,5      # Specific pairs only
  %(prog)s /path/to/run --output-dir /tmp  # Custom output location
        """)
    parser.add_argument("run_dir", help="Run directory (contains seed/, orchestrator/, audio/)")
    parser.add_argument("--mode", default="fable", choices=["fable", "opus"],
                        help="Prompt mode: fable (default) or opus")
    parser.add_argument("--pairs", help="Comma-separated pair numbers (default: all discovered)")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: run_dir/audio/)")
    parser.add_argument("--personality", help="Override personality YAML file (applied to all pairs)")
    parser.add_argument("--mandates", help="Path to production mandates file (overrides defaults)")
    parser.add_argument("--golden-refs", help="Path to golden song references file")
    args = parser.parse_args()
    
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Discover pairs
    all_pairs = discover_pairs(run_dir)
    if not all_pairs:
        print(f"ERROR: No pair step10 files found in {run_dir}/audio/")
        print("Expected pattern: pair_XX_step10_production_wrap.md")
        sys.exit(1)
    
    # Filter by --pairs
    if args.pairs:
        wanted = {int(p.strip()) for p in args.pairs.split(",")}
        pairs = [(n, p, t) for n, p, t in all_pairs if n in wanted]
        if not pairs:
            print(f"ERROR: No pairs matching {args.pairs} found. Available: {[n for n,_,_ in all_pairs]}")
            sys.exit(1)
    else:
        pairs = all_pairs
    
    # Resolve personalities
    personalities = resolve_personalities(run_dir, pairs)
    
    # Override personality if specified
    if args.personality:
        yaml_path = args.personality
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.join(PERS_DIR, yaml_path)
        if not os.path.exists(yaml_path):
            print(f"ERROR: Personality YAML not found: {yaml_path}")
            sys.exit(1)
        for pn in [n for n, _, _ in pairs]:
            personalities[pn] = (os.path.basename(yaml_path).replace(".yaml", "").replace("-mini", "").upper(), yaml_path)
    
    # Extract metadata
    metadata = extract_run_metadata(run_dir)
    
    # Output dir
    output_dir = args.output_dir or os.path.join(run_dir, "audio")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Run: {metadata['theme']}")
    print(f"Mode: {args.mode}")
    print(f"Pairs: {[n for n,_,_ in pairs]}")
    print(f"Output: {output_dir}/")
    print()
    
    mode_label = {"fable": "fable", "opus": "opus"}[args.mode]
    
    for pair_num, step10_path, title in pairs:
        persona_name, yaml_path = personalities.get(pair_num, ("LOFN-PRIME (AWE mode)", 
                                           os.path.join(PERS_DIR, DEFAULT_PERSONALITY)))
        
        prompt = build_prompt(
            pair_num, step10_path, title, persona_name, yaml_path,
            run_dir, metadata, args.mode
        )
        
        filename = f"pair_{pair_num:02d}_step11_{mode_label}_prompt.md"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w") as f:
            f.write(prompt)
        
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  ✅ Pair {pair_num:02d}: {filename} ({size_kb:.0f} KB)  — {persona_name}")
    
    total = sum(os.path.getsize(os.path.join(output_dir, f"pair_{n:02d}_step11_{mode_label}_prompt.md"))
                for n, _, _ in pairs)
    print(f"\n✅ {len(pairs)} prompts built, {total/1024:.0f} KB total -> {output_dir}/")


if __name__ == "__main__":
    main()
