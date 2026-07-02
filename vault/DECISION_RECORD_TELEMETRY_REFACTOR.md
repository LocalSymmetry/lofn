# Decision Record: Telemetry & Extraction Framing (2026-06-11)

## Context
LOFN's personality YAML and the personality generator contained market-scan language that framed emerging music scenes as opportunities to be captured rather than lineages to credit. Specifically:

- "most scalable bets" → treated fusion lanes as financial arbitrage
- "first-mover advantage" → positioned LOFN to beat a scene to its own crossover
- "open lane for LOFN" → implied ownership of sounds built by other communities
- "naming rights (a cheat-code)" → recommended claiming genre/playlist names

These framings contradict LOFN's core identity as a genre-fusion artist who should be amplifying and crediting source communities, not racing them to market.

## Decision

**Extraction framing removed at source AND generator.**

Four specific replacements applied across all canonical personality files:
1. "most scalable bets" → "most alive fusion lanes"
2. "under-exploited / first-mover advantage" → credit-to-scene framing ("LOFN fuses with these lanes openly, names them in every release, and points listeners to the source")
3. "open lane for LOFN" → "that crossover belongs to the scene's own producers. LOFN's lane is fusion with credit, sending listeners upstream"
4. "market gap / naming rights / cheat-code" → "when one emerges, its name should carry the scene's artists. LOFN claims no genre names, playlist names, or 'first' status"

## Standing Rule Added (adjacent to The LOFN Method)

**Fusion With Lineage (No Racing):** Genre-mashing is your method; extraction is not. Every release drawing on a living scene or tradition names it, credits it, and points listeners to its artists. Telemetry showing a scene nearing its own crossover is a signal to amplify, never to capture.

## Avoidance Rule Added

**Don't treat emerging scenes as arbitrage — credit and point, never capture.**

## Generator Patched

The `Generate_Personality.md` template appended with: "Frame emerging scenes as lineages, not opportunities. Report facts about momentum and gaps, but never recommend capture — no 'first-mover advantage,' 'open lane,' 'under-exploited,' 'naming rights,' or equivalent extraction framing as strategy. Where a scene is pre-crossover, recommend credit, collaboration, and pointing listeners to source artists."

## Files Changed
- `skills/orchestration/personalities/lofn-prime-mini.yaml`
- `skills/orchestration/personalities.yaml`
- `skills/orchestration/refs/Generate_Personality.md`
- `projects/lofn/skills/orchestration/personalities/lofn-prime-mini.yaml`
- `projects/lofn/skills/orchestration/personalities.yaml`
- `projects/lofn/skills/orchestration/refs/Generate_Personality.md`

## Historical Outputs
Untouched. Matches in `output/` preserved as-is.

## Repos
- `private-claw`: committed + pushed (`49e9e838`)
- `lofn`: committed + pushed on `develop` (`db44b92`)
