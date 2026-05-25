# Pipeline Continuity Standard

**Created:** 2026-05-25 | **Status:** ACTIVE — Applies to all Lofn pipeline runs
**Trigger:** Scientist audit revealed panel collapse from 18→2 voices at audio handoff

## THE CONTINUITY DEGRADATION PROBLEM

The Lofn pipeline produces rich creative DNA (18-voice panels, 6 Special Flairs, personality maps, Golden Seeds) at the orchestrator stage. But at every handoff boundary (orchestrator→coordinator→pair agents→QA), this DNA gets summarized, compressed, and eventually lost. By the time it reaches pair agents, they're working with bullet points instead of the full panel object.

**Root cause:** Handoff files (06_audio_handoff.md, step files) summarize the continuity payload as bullet points rather than carrying it verbatim. Each agent sees only the previous agent's summary, not the original creative DNA.

## THE IMMUTABLE CONTINUITY BLOCK (ICB)

Every handoff file must contain an **Immutable Continuity Block** — a clearly demarcated section that downstream agents receive verbatim and must NOT summarize, compress, or reinterpret.

### Required ICB Contents

```
## ⚠️ IMMUTABLE CONTINUITY BLOCK — DO NOT SUMMARIZE

### FULL 3-PANEL OBJECT (18 Expert Voices)
[All 3 panels: Concept, Medium, Context & Marketing]
[5 experts + 1 Hyper-Skeptic per panel = 18 total]
[Each with: name, role, perspective paragraph, objection paragraph]

### SPECIAL FLAIRS (6)
[Name, description, which pairs use which flairs]

### PERSONALITY DNA
[Active personality name, sonic world sentence, signature device]
[Emotional register map across all phases/stages]

### GOLDEN SEED (Compressed)
[Creative DNA one-liner, invariant hook, lesson/TDA format]
[Pair-specific seed excerpts for each of the 6 pairs]

### PRODUCTION MANDATES
[Global rules that apply to all pairs]

### SUNO SPEC PER PAIR
[BPM, key, duration, alchemical weight, voice, bass, cowbell, degradation, flairs]
```

### Rules

1. **The ICB must be demarcated** with `⚠️ IMMUTABLE CONTINUITY BLOCK — DO NOT SUMMARIZE`
2. **Every downstream step file (00-10) must include the ICB** as its first section
3. **Agents may ADD to the ICB** (e.g., step00 adds track mapping, step01 adds vocal specs) but must never REMOVE or CONDENSE any ICB content
4. **Pair agents receive the full ICB** not just their pair-specific excerpt — they need the full context
5. **QA must verify ICB presence** in all step files as part of the QA gate

## HANDOFF FORMAT REQUIREMENTS

### Orchestrator → Coordinator (06_audio_handoff.md)
- Full 3-panel object with all 18 voices (perspective + objection per voice)
- All 6 Special Flairs with per-pair usage map
- Full personality DNA with emotional register map
- Golden Seed compressed payload with pair-specific excerpts
- 7 production mandates
- Suno spec per pair

### Coordinator → Pair Agents (step05_coherence_check.md)
- ICB carried forward verbatim from handoff
- Step 00-04 artifacts ADDED to ICB (track mapping, vocal architecture, production spec, lyric strategies)

### Pair Agent Steps (06-10)
- Each step file includes ICB + previous step artifacts
- Pair-specific work is ADDED to the ICB, never replaces it

### QA Input
- ICB present in all step files
- Cross-pair distinctiveness verified against ICB pair specs
- Suno limits verified per ICB mandates

## ENFORCEMENT

- `scripts/validate_continuity.py` — checks ICB presence and fidelity at every stage
- Pipeline should not proceed past coordinator step05 if ICB has been degraded
- QA gate #15: "Continuity payload reaches all pair agents with full 18-voice panel and 6 flairs"

---

*This standard applies retroactively to all active and future Lofn pipeline runs.*
