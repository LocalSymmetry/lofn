# QA Step 1: Pipeline Completeness Audit

## Input
- Output directory: `{output_dir}/`

## ⛔ EARLY EXIT RULE — READ THIS FIRST

1. `ls -la` the output directory and note all step file sizes.
2. If ANY step file is under the minimum byte threshold for its modality, this run is a **PIPELINE STUB**.
3. On a PIPELINE STUB:
   - Write status = `PIPELINE STUB — CONTENT NOT GENERATED`
   - List every stub file with its actual size vs minimum required
   - **STOP IMMEDIATELY** — do not proceed to Steps 2-4

## Task

### Modality Detection
Determine the modality from the output directory and file inventory:
- **music/audio** — song prompts, lyrics, guides, revision synthesis
- **image/vision** — prompts, refined prompts, render summaries
- **story/narrative** — prose/story outputs, scene/beat docs
- **video/director** — shot lists, scene prompts, frame/sequence outputs

### Completeness Checklist (all modalities)
- [ ] Was a research step documented? (check for research brief file)
- [ ] Was Lofn-Core invoked? (check for a seed/brief document)
- [ ] Was Lofn-Orchestrator invoked? (check for a metaprompt/panel output file)
- [ ] Was the correct modality agent invoked and did it produce intermediate step files?

### Minimum byte thresholds per step

**Image pipeline:**
| File | Minimum bytes |
|------|--------------|
| `00_aesthetics.md` | ≥ 2000 |
| `01_essence.md` | ≥ 500 |
| `02_concepts.md` | ≥ 800 |
| `03_artist_critique.md` | ≥ 800 |
| `04_mediums.md` | ≥ 400 |
| `05_refined_pairs.md` | ≥ 600 |
| `06_facets.md` | ≥ 1200 |
| `07_guides.md` | ≥ 1500 |
| `08_prompts.md` | ≥ 6000 |
| `09_refined_prompts.md` | ≥ 6000 |
| `10_final_prompts.md` | ≥ 6000 |

**Music pipeline:**
| File | Minimum bytes |
|------|--------------|
| `00_aesthetics*.md` | ≥ 2000 |
| `06_scoring_facets.md` | ≥ 1200 |
| `07_song_guides.md` | ≥ 3000 |
| Each `08_song_*.md` | ≥ 5000 |

If ANY file is below minimum → STUB FAILURE. Stop here.

### Template placeholder detection
Scan for: `{concept}`, `{medium}`, `[PLACEHOLDER]`, `TBD`, `TODO`, `Artifact N`, `Lorem ipsum`
If found → STUB FAILURE.

## Save
- Completeness audit: `{output_dir}/qa_step1_completeness.md`
