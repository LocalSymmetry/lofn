#!/usr/bin/env python3
"""Claude-native all-pairs Step 11 "fable packager" prompt builder.

Adapted from skills/step11-packager/scripts/build_fable_prompt.py for the
Claude-native Lofn pipeline (.claude/skills) and cross-platform (Windows) use.

What changed vs. the legacy OpenClaw script:
  * No hardcoded /data/.openclaw/workspace — repo root is auto-detected from the
    script location (and from the run dir as a fallback), or set with --repo-root.
  * Run-dir layout matches Claude-native runs: top-level context files
    (core_seed.md, 04_metaprompt.md, 05_pair_assignments.md, 03_panel_debate.md,
    00_research_brief.md, CREATIVE_CONTEXT.md) and per-pair step-10 files under
    <run>/music/pair_NN_step10_package.md. Legacy layouts (seed/, orchestrator/,
    audio/pair_XX_step10_production_wrap.md) still work as fallbacks.
  * Pair discovery uses os.path.basename matching (backslash-safe on Windows).
  * ALL file I/O is utf-8 (legacy used the platform default — crashes on Windows
    cp1252 because the content contains em dashes, ✅, ⛔, ❌, etc.).
  * Reuses the legacy references verbatim (non-destructive): Suno rules,
    personality YAMLs, and the golden songs index are read from their existing
    repo locations — no duplication.
  * If a pair already has a step-11 enhanced draft
    (pair_NN_step10_final_package_enhanced.md), it is embedded as a PRIOR DRAFT
    so the refiner improves on it instead of starting cold.

Usage:
    # Build all pairs for a run (auto-detect everything):
    python skills-or-relative/build_fable_prompt.py output/daily/2026-06-19

    # Opus-named variant:
    python build_fable_prompt.py output/daily/2026-06-19 --mode opus

    # Specific pairs only:
    python build_fable_prompt.py output/daily/2026-06-19 --pairs 1,3,5

    # Custom output dir / personality override / explicit repo root:
    python build_fable_prompt.py output/daily/2026-06-19 -o /tmp/prompts
    python build_fable_prompt.py output/daily/2026-06-19 --personality lofn-prime-mini.yaml
    python build_fable_prompt.py output/daily/2026-06-19 --repo-root E:/git/lofn
"""

# ---------------------------------------------------------------------------
# Self-contained reference resolution. The skill VENDORS its own references
# (Suno rules, golden index, default LOFN-PRIME YAML) so it stands alone; the
# repo archive is only a fallback for named personalities not vendored here.
# ---------------------------------------------------------------------------
import os, sys, glob, re, argparse

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the skill root
LOCAL_REF_DIR = os.path.join(SKILL_DIR, "references")
LOCAL_PERS_DIR = os.path.join(LOCAL_REF_DIR, "personalities")
DEFAULT_PERSONALITY = "lofn-prime-mini.yaml"


def find_repo_root(start):
    """Walk upward from `start` until we find the Lofn repo root.

    The marker is skills/orchestration/personalities (the personality archive).
    Returns None if not found — the skill works without it via vendored refs.
    """
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, "skills", "orchestration", "personalities")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def resolve_repo_root(run_dir, override):
    """Locate the repo root if available. Optional — returns None when absent.

    Used only to fall back to the canonical personality archive for named
    personalities that aren't vendored in the skill's own references/.
    """
    if override:
        root = os.path.abspath(override)
        if not os.path.isdir(os.path.join(root, "skills", "orchestration", "personalities")):
            print(f"ERROR: --repo-root {root} has no skills/orchestration/personalities")
            sys.exit(1)
        return root
    root = find_repo_root(os.path.dirname(os.path.abspath(__file__)))
    if root is None:
        root = find_repo_root(run_dir)
    return root


def resolve_ref(local_name, repo_rel, repo_root):
    """Prefer the vendored copy; fall back to the repo location if present."""
    local = os.path.join(LOCAL_REF_DIR, local_name)
    if os.path.exists(local):
        return local
    if repo_root:
        repo = os.path.join(repo_root, *repo_rel.split("/"))
        if os.path.exists(repo):
            return repo
    return local  # report the (missing) local path so warnings point at the skill


def resolve_personality_yaml(name_stem, repo_root):
    """Resolve a personality YAML: vendored references/personalities/ first,
    then the canonical repo archive. Returns a path (may not exist)."""
    fname = f"{name_stem}.yaml"
    local = os.path.join(LOCAL_PERS_DIR, fname)
    if os.path.exists(local):
        return local
    if repo_root:
        repo = os.path.join(repo_root, "skills", "orchestration", "personalities", fname)
        if os.path.exists(repo):
            return repo
    return local

# Canonical run-context files to embed, in order. Each entry is
# (label, [candidate relative paths], required?). First existing candidate wins.
# Claude-native names first, legacy OpenClaw names as fallbacks.
RUN_CONTEXT_SPEC = [
    ("RESEARCH BRIEF / USER INPUT", ["00_research_brief.md", "00_user_input.md", "input.md"]),
    ("GOLDEN SEED", ["core_seed.md", "02_golden_seed.md", "seed/GOLDEN_SEED.md", "01_seed_lineage.md"]),
    ("ORCHESTRATOR PANEL DEBATE", ["03_panel_debate.md", "03_orchestrator_panel_debate.md",
                                   "orchestrator/ORCHESTRATOR_BRIEF.md"]),
    ("METAPROMPT", ["04_metaprompt.md", "04_orchestrator_metaprompt.md"]),
    ("PAIR ASSIGNMENTS", ["05_pair_assignments.md", "05_orchestrator_pair_assignments.md"]),
    ("CREATIVE CONTEXT / ICB", ["CREATIVE_CONTEXT.md", "06_audio_handoff.md", "06_music_handoff.md"]),
]

# Per-pair step-10 source: first existing pattern wins (relative to run dir).
STEP10_GLOBS = [
    "music/pair_*_step10_package.md",            # Claude-native
    "audio/pair_*_step10_production_wrap.md",    # legacy OpenClaw
    "music/pair_*_step10_*.md",                  # loose fallback
]
# Optional prior step-11 enhanced draft (Claude-native inline enhancement output).
ENHANCED_GLOB = "music/pair_{pair:02d}_step10_final_package_enhanced.md"

DEFAULT_MANDATES = """1. SOMATIC BASS GATE: Every pair must contain at least one passage where bass operates in the 30-60Hz somatic range.
2. BLEED ENFORCEMENT: Vocal bleed into cathedral reverb (3-7s decay) must create structural tension.
3. UNDECIDABLE ELEMENT: Every song must contain ONE element whose musical function is permanently ambiguous.
4. SILENCE AS PRESENCE: At least two pairs must contain a structural silence (>=2 bars) that functions as presence.
5. NO ARTIST NAMES IN SUNO PROMPTS.
6. DISC_CHANNEL FORMAT: Dense prose style prompt <=1000 chars, exclude prompt 400-900 chars, lyrics <5000 chars (target <=4800 — Suno hard render cap on the entire lyrics field). All three text blocks have binding character limits — treat them with the same enforcement as the style prompt. 5-channel headers, full EMO section headers.
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

**## SUNO ENHANCED LYRICS — PRIMARY:** Rewrite, restructure, rebuild. Elevate to literary density. Apply at least one structural transformation. If a verse is generic, replace it wholesale. If the bridge doesn't earn its place, cut it or reinvent it. If the song form fights the emotional arc, change the form. Must open with [Theme:] + [SONG FORM:] lines. Disc_Channel block immediately following. Then full lyrics with EMO tags INTEGRATED INTO EVERY SECTION HEADER. Minimum 60 sung lines. 🚨 HARD CAP: the ENTIRE lyrics field (everything pasted into Suno's lyrics box — [Theme] + [SONG FORM] + Disc_Channel block + every section/EMO header + every *SFX* cue + all sung lines) MUST be < 5,000 characters (target <= 4,800). Measure the exact count; never estimate. A song over 5,000 will not render. If over: cut/merge sung lines first, then tighten headers, then move the Disc_Channel/production metadata to a Production Sidecar OUTSIDE the lyrics field. The block header is literally `## SUNO ENHANCED LYRICS`.

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
- `## SUNO ENHANCED LYRICS`: [Theme:] + [SONG FORM:], 5-line Disc_Channel channel strip, integrated EMO section headers using `–` em dashes, >=60 sung lines, entire lyrics field < 5,000 chars (Suno hard cap — measured, not estimated), literary density, structural transformation
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


def load_file(path, required=False):
    """Load a file as utf-8, return its content or None."""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    if required:
        return None
    return None


def first_existing(run_dir, candidates):
    for rel in candidates:
        p = os.path.join(run_dir, rel)
        if os.path.exists(p):
            return p, rel
    return None, candidates[0]


def extract_title(content, pair_num):
    """Pull a human title from a step-10 file. The Claude-native header is
    `# Pair 01 — "The Dead Tongue Wakes" · Step 10 Package`; legacy uses
    `# Step 10 — "Title"`. A quoted string in the first lines is the title."""
    if not content:
        return f"Pair {pair_num:02d}"
    head = content.split("\n")[:8]
    for line in head:
        m = re.search(r'"([^"]+)"', line)
        if m:
            return m.group(1).strip()
        m = re.search(r'[—–-]\s*([^·|]+?)\s*(?:[·|]|$)', line)
        if line.startswith("#") and m and m.group(1).strip().lower() not in ("step 10", "production wrap"):
            return m.group(1).strip()
    return f"Pair {pair_num:02d}"


def discover_pairs(run_dir):
    """Find per-pair step-10 files. Returns list of (pair_num, step10_path, title)."""
    files = []
    for pattern in STEP10_GLOBS:
        files = sorted(glob.glob(os.path.join(run_dir, pattern)))
        if files:
            break
    pairs = []
    seen = set()
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"pair_(\d+)_step10_.*\.md", base)
        if not m:
            continue
        pair_num = int(m.group(1))
        if pair_num in seen:
            continue
        seen.add(pair_num)
        title = extract_title(load_file(f), pair_num)
        pairs.append((pair_num, f, title))
    return sorted(pairs, key=lambda t: t[0])


def resolve_personalities(run_dir, pairs, repo_root):
    """Determine personality per pair from the pair-assignments / creative-context.

    Claude-native daily runs use LOFN-PRIME with an AWE / INDIGNATION mode per
    pair. One-off `lofn` runs may name a personality. Falls back to LOFN-PRIME.
    YAMLs resolve vendored-first, then the repo archive.
    Returns {pair_num: (name, yaml_path)}.
    """
    assign_path, _ = first_existing(
        run_dir, ["05_pair_assignments.md", "05_orchestrator_pair_assignments.md",
                  "orchestrator/ORCHESTRATOR_BRIEF.md"])
    text = load_file(assign_path) or ""
    default_yaml = resolve_personality_yaml(DEFAULT_PERSONALITY[:-5], repo_root)

    # Split into per-pair sections on "Pair NN" / "PAIR N" headers.
    section_starts = list(re.finditer(r'(?im)^#{0,4}\s*pair\s*0*(\d+)\b', text))
    sections = {}
    for i, mt in enumerate(section_starts):
        pn = int(mt.group(1))
        end = section_starts[i + 1].start() if i + 1 < len(section_starts) else len(text)
        sections[pn] = text[mt.start():end]

    persona_pattern = re.compile(r'##\s*FULL\s+(.+?)\s+PERSONALITY\s+DNA\s+BLOCK', re.IGNORECASE)
    personalities = {}
    for pair_num, _, _ in pairs:
        sec = sections.get(pair_num, "")
        if "INDIGNATION" in sec:
            personalities[pair_num] = ("LOFN-PRIME (INDIGNATION mode)", default_yaml)
        elif "LOFN-PRIME" in sec or not sec:
            personalities[pair_num] = ("LOFN-PRIME (AWE mode)", default_yaml)
        else:
            pm = persona_pattern.search(sec)
            name = None
            if pm:
                name = pm.group(1).strip()
            else:
                # Try a bare named-personality reference matching a YAML file stem.
                for token in re.findall(r'[A-Za-z][A-Za-z0-9\- ]{2,}', sec[:400]):
                    stem = token.strip().lower().replace(" ", "-")
                    if os.path.exists(resolve_personality_yaml(stem, repo_root)):
                        name = token.strip()
                        break
            if name:
                yp = resolve_personality_yaml(name.lower().replace(' ', '-'), repo_root)
                if os.path.exists(yp):
                    personalities[pair_num] = (name.upper(), yp)
                else:
                    personalities[pair_num] = (f"{name} (fallback: LOFN-PRIME)", default_yaml)
            else:
                personalities[pair_num] = ("LOFN-PRIME (AWE mode)", default_yaml)
    return personalities


def extract_run_metadata(run_dir):
    """run_id from dir name; theme/lens from the golden seed header if present."""
    seed_path, _ = first_existing(run_dir, ["core_seed.md", "02_golden_seed.md", "seed/GOLDEN_SEED.md"])
    seed = load_file(seed_path) or ""
    metadata = {
        "run_id": os.path.basename(os.path.normpath(run_dir)),
        "theme": "",
        "constraint": "Constraint-as-Form",
        "lens": "",
    }
    for line in seed.split("\n")[:8]:
        if line.startswith("#"):
            m = re.search(r'[—–-]\s*"?([^"]+?)"?\s*$', line)
            if m and not metadata["theme"]:
                cand = m.group(1).strip()
                if cand and "seed" not in cand.lower():
                    metadata["theme"] = cand
        if "Lens:" in line:
            metadata["lens"] = line.split("Lens:", 1)[1].strip()
    if not metadata["theme"]:
        metadata["theme"] = metadata["run_id"]
    return metadata


def build_run_context(run_dir):
    parts = []
    for label, candidates in RUN_CONTEXT_SPEC:
        path, rel = first_existing(run_dir, candidates)
        content = load_file(path) if path else None
        if content:
            parts.append(f"### {label}\n\n{content}")
        else:
            parts.append(f"### {label}\n\n[File not found — tried: {', '.join(candidates)}]")
    return "\n\n---\n\n".join(parts)


def build_prompt(pair_num, step10_path, title, persona_name, yaml_path,
                 run_dir, metadata, mode, rules_path, golden_path, enhanced_path):
    step10 = load_file(step10_path) or f"[ERROR: Could not read {step10_path}]"
    personality_yaml = load_file(yaml_path) or f"[ERROR: Personality YAML not found: {yaml_path}]"
    suno_rules = load_file(rules_path) or "[ERROR: Suno rules not found]"
    golden_refs = load_file(golden_path) or ""
    enhanced = load_file(enhanced_path) if enhanced_path else None
    run_context = build_run_context(run_dir)

    mode_label = {"fable": "FABLE/OPUS", "opus": "OPUS"}.get(mode, "FABLE/OPUS")

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
    if enhanced:
        prompt += ("---\n\n## PRIOR STEP 11 DRAFT (inline enhancement — IMPROVE ON THIS, "
                   "do not merely echo it)\n\n" + enhanced + "\n\n")
    prompt += MANUAL_REFINEMENT_BLOCK
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Build Claude-native all-pair Step 11 fable-packager prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s output/daily/2026-06-19                 # All pairs, fable mode
  %(prog)s output/daily/2026-06-19 --pairs 1,3,5   # Specific pairs only
  %(prog)s output/daily/2026-06-19 -o /tmp/prompts # Custom output location
        """)
    parser.add_argument("run_dir", help="Run directory (contains core_seed.md, 05_pair_assignments.md, music/)")
    parser.add_argument("--mode", default="fable", choices=["fable", "opus"],
                        help="Prompt mode: fable (default) or opus")
    parser.add_argument("--pairs", help="Comma-separated pair numbers (default: all discovered)")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: <run>/music/)")
    parser.add_argument("--personality", help="Override personality YAML (filename in the archive, or a path)")
    parser.add_argument("--repo-root", help="Repo root (default: auto-detected)")
    parser.add_argument("--no-enhanced", action="store_true",
                        help="Do not embed any existing step-11 enhanced draft as a prior draft")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    repo_root = resolve_repo_root(run_dir, args.repo_root)  # optional — may be None
    # Vendored references first; repo locations only as a fallback.
    rules_path = resolve_ref("suno_rules_condensed.md",
                             "skills/step11-packager/references/suno_rules_condensed.md", repo_root)
    golden_path = resolve_ref("golden_songs_index.md",
                              "skills/music/references/golden_songs_index.md", repo_root)

    all_pairs = discover_pairs(run_dir)
    if not all_pairs:
        print(f"ERROR: No pair step-10 files found under {run_dir}")
        print(f"Tried patterns: {STEP10_GLOBS}")
        sys.exit(1)

    if args.pairs:
        wanted = {int(p.strip()) for p in args.pairs.split(",")}
        pairs = [(n, p, t) for n, p, t in all_pairs if n in wanted]
        if not pairs:
            print(f"ERROR: No pairs matching {args.pairs}. Available: {[n for n,_,_ in all_pairs]}")
            sys.exit(1)
    else:
        pairs = all_pairs

    personalities = resolve_personalities(run_dir, pairs, repo_root)

    if args.personality:
        yaml_path = args.personality
        if not os.path.isabs(yaml_path):
            stem = os.path.basename(yaml_path)
            if stem.endswith(".yaml"):
                stem = stem[:-5]
            yaml_path = resolve_personality_yaml(stem, repo_root)
        if not os.path.exists(yaml_path):
            print(f"ERROR: Personality YAML not found: {yaml_path}")
            sys.exit(1)
        nm = os.path.basename(yaml_path).replace(".yaml", "").replace("-mini", "").upper()
        for pn, _, _ in pairs:
            personalities[pn] = (nm, yaml_path)

    metadata = extract_run_metadata(run_dir)
    output_dir = args.output_dir or os.path.join(run_dir, "music")
    if not os.path.isdir(output_dir):
        output_dir = run_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Repo root: {repo_root or '(not found — using vendored references only)'}")
    print(f"Refs:      rules={'vendored' if rules_path.startswith(LOCAL_REF_DIR) else 'repo'}, "
          f"golden={'vendored' if golden_path.startswith(LOCAL_REF_DIR) else 'repo'}")
    print(f"Run:       {metadata['theme']}  ({metadata['run_id']})")
    print(f"Mode:      {args.mode}")
    print(f"Pairs:     {[n for n,_,_ in pairs]}")
    print(f"Output:    {output_dir}")
    if not os.path.exists(rules_path):
        print(f"WARNING: Suno rules not found at {rules_path}")
    if not os.path.exists(golden_path):
        print(f"WARNING: Golden songs index not found at {golden_path}")
    print()

    mode_label = {"fable": "fable", "opus": "opus"}[args.mode]
    built = []
    for pair_num, step10_path, title in pairs:
        persona_name, yaml_path = personalities.get(
            pair_num, ("LOFN-PRIME (AWE mode)", resolve_personality_yaml(DEFAULT_PERSONALITY[:-5], repo_root)))

        enhanced_path = None
        if not args.no_enhanced:
            cand = os.path.join(run_dir, ENHANCED_GLOB.format(pair=pair_num))
            if os.path.exists(cand):
                enhanced_path = cand

        prompt = build_prompt(
            pair_num, step10_path, title, persona_name, yaml_path,
            run_dir, metadata, args.mode, rules_path, golden_path, enhanced_path)

        filename = f"pair_{pair_num:02d}_step11_{mode_label}_prompt.md"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        size_kb = os.path.getsize(output_path) / 1024
        flag = " +prior-step11" if enhanced_path else ""
        warn = "  ⚠ UNDER 50KB — check truncation" if size_kb < 50 else ""
        built.append(output_path)
        print(f"  [OK] Pair {pair_num:02d}: {filename} ({size_kb:.0f} KB){flag}  — {persona_name}{warn}")

    total = sum(os.path.getsize(p) for p in built)
    print(f"\n[OK] {len(built)} prompts built, {total/1024:.0f} KB total -> {output_dir}")


if __name__ == "__main__":
    main()
