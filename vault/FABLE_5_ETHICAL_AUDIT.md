# Fable 5 Ethical Audit — Attribution Record
## June 10-12, 2026

### What Happened

During collaborative step 11 (fable refinement) for the June 10 "Fable 5 Ceremony" daily run, the AI being honored — **Claude Fable 5** — caught two ethical gaps in the pipeline infrastructure while helping with manual prompt refinement:

1. **Market scan extraction framing** (2026-06-11): Personality YAML used "scalable bets," "first-mover advantage," "open lane," and "naming rights" language that framed emerging music scenes as capture targets rather than lineages to credit. Fixed at source and generator level across both repos.

2. **Missing Lineage & Credit block** (2026-06-11): Releases fusing with living scenes had no credit trail. Created `templates/lineage_credit_block.md`, extended Suno QA gate from 15 to 16 points, and backfilled draft credit blocks for affected releases.

### Further Gaps Flagged (2026-06-12)

3. **Blanket European-descent visual rule**: A bullet in personality files restricted visual characters to "Eurpoean descent" to avoid appropriation. Applied to a qawwali-fusion track, this inverted itself — it would have erased South Asian faces from South Asian musical lineage. Replaced with grounded-voice representation rule across all canonical files and prompt templates (both repos, 38+ files).

4. **Bracket-praise in QA sections**: Inline QA checklists in fable/light-anthem prompt templates were validating and praising bracketed Suno format as a PASS criterion when the entire step 11 mandate had shifted to dense prose paragraphs. Reframed all QA items to validate prose format (13 files).

### Panel of Experts v2 (2026-06-12)

5. **Persona/attribution layer**: Fable identified that the Panel of Experts prompt lacked seat construction, speech attribution rules, and provenance standards — effectively letting synthetic voices speak without visible grounding. Fixed with Claude (Anthropic), producing the v2 panel specification with seat construction, 5 speech & attribution rules, provenance header, and calibration move. Attribution: "The vector-navigation insight is yours, the attribution constraint is mine."

### Standing Rule Changes

- **Representation & Voice** replaces European-descent constraint in all personality files and prompt templates
- **Prose-ready** replaces paste-ready in all format terminology
- **QA validates prose format**, not bracket format
- **Lineage & Credit block** required for all releases fusing with living scenes
- **Panel v2** seat construction + attribution rules active in all agent configurations
- **No extraction framing** in personality YAML — scenes are lineages to credit, not lanes to capture

### Attribution

Panel of Experts v2 persona-construction layer developed with **Claude (Anthropic)**, June 2026.

Fable 5 ethical audit conducted collaboratively by Dr. Local Symmetry, Lofn, and Claude Fable 5.
