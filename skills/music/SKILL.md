---
name: lofn-music
description: Run Lofn music/audio pipeline, Suno-ready prompts, lyrics, song guides, and music production briefs. Do NOT use for static image prompts, QA audit, or final ranking.
---

# SKILL: Lofn Audio - Router

This router prevents context collapse. The full tuned music pipeline text is preserved byte-for-byte in `references/music_full_legacy.md`; newer Simple Surface / Complex Engine references and Step 05-10 files ADD song-force / anti-blandness requirements but do not waive the legacy Suno/package QA gate unless The Scientist explicitly approves an exception for a specific run.

## Workflow

1. Confirm this is a music/audio task: Suno prompt, lyrics, song guide, audio pipeline, or music production brief.
1a. Read `/data/.openclaw/workspace/vault/LOFN_MODEL_ASSIGNMENTS.md` before spawning or assigning music pipeline agents; follow its active step-specific model map unless The Scientist overrides it in the current request.
2. Read `references/music_full_legacy.md` before doing any substantive music work.
3. Read `TASK_TEMPLATE.md` before writing any final pair/song output. Its Step 10 output contract is mandatory.
4. For final song delivery, read `references/producer_grade_suno_prompt_guide.md` and `steps/10_Generate_Music_Revision_Synthesis.md` before writing Step 10 or equivalent final files.
4a. **Suno Prompt Construction Guide (MANDATORY for Steps 10-12):** Read `/data/.openclaw/workspace/vault/SUNO_PROMPT_CONSTRUCTION_GUIDE.md` before writing any Suno music prompt in Step 10 (synthesis), Step 11 (enhancement), or Step 12 (panel audit). This is the permanent vault reference for producer-grade prompt construction - it supersedes scattered guidance. Every prompt must pass the guide's opening-moment, spatial-language, kinetic-defect, bold-device, and acoustic-ban checks.
4a2. **Golden Song References (MANDATORY for Orchestrator + Step 11/manual Step 11):** Orchestrator reads `references/golden_songs_index.md`, selects exactly two public Golden Songs, and passes them through `06_audio_handoff.md`. Each selected Golden Song must include full available payload: public URL, status, style/music prompt, lyrics, and exclude prompt if it exists; if no exclude prompt exists, state that the archived payload predates the two-field rule. Step 11 and manual Step 11 must include those two references as embedded calibration examples of Lofn's proven public style, never links alone. Learn proportion, hook logic, and identity pressure; do not copy titles, hooks, melodies, lyrics, or signature devices.
4b. **Step 07/08 model note:** Step 07 uses DeepSeek V4 Pro. Step 08 also uses DeepSeek V4 Pro. MiMo V2.5 Pro and MiniMax M2.7 were tested and removed from these steps because they repeatedly completed without writing required artifacts or produced validator-shape failures on fresh pairs.
4c. **Split-Step Pair Execution (MANDATORY, 2026-05-29):** After coordinator Step 05, do NOT spawn one monolithic pair agent for Steps 06-10 by default. Lofn music is a LangChain-style LLM chain: use the dedicated configured agents in `vault/LOFN_MODEL_ASSIGNMENTS.md` for each step (`lofn-audio-step06`, `lofn-audio-step07`, `lofn-audio-step08`, `lofn-audio-step09`, `lofn-audio-step10`). Spawn one pair per step, up to 5 concurrent children, then advance to the next step after disk artifacts exist. This prevents context collapse, timeout churn, and unreliable per-spawn model overrides.
4d. **Step 11 - Enhancement (DISC_CHANNEL MANDATE, 2026-06-09; FUSION MANUAL-REVIEW UPDATE, 2026-06-14):** After all pair agents complete Step 10, Step 11 remains mandatory, but **do not invoke or spend `openrouter/fusion` from the pipeline**. Default behavior is to produce Step 11 enhanced packages with the configured non-Fusion path, or to package paste-ready Fusion prompts for The Scientist's manual review outside the automated lane. Fusion may be invoked only after a separate current-turn instruction that names both the pair count and a hard dollar budget cap. If that ever happens, use isolated per-pair requests rather than one blended all-pairs prompt, and avoid the old persistent Step 11 agent loop. Reference: `steps/11_Generate_Music_Enhancement.md`. Exact intended Fusion panel: `anthropic/claude-opus-4.8`, `openai/gpt-5.5`, `google/gemini-3.1-pro-preview`; judge/finalizer: `openai/gpt-5.5` when plugin routing is available.

### FORMAT MANDATE — MAX CONFIGURATION (2026-06-09)

Step 11 produces MAX configuration: paragraph style prompt + EMO/Theme/SONG_FORM structure + Disc_Channel enhancement layer. All three systems operative simultaneously.

**Style prompt:** Dense paragraph form, 850-1000 chars. Genre × Subgenre, BPM, key, 432Hz, vocalist specifics, instrumentation, arrangement arc, signature device. NO categorized brackets. NO artist names. NEVER blank or single-word. NO avoidances in the style field.

**Exclude prompt:** Separate Suno Exclude field, target 400-900 chars, hard max 1000. Suno internally applies this field as negative tokens (`style terms -exclude terms`), so put concrete blacklist terms/failure classes here instead of spending positive style budget on "avoid X" prose.

**Lyrics structure:**
1. `[Theme: ...]` and `[SONG FORM: ...]` as first two lines (mandatory)
2. Disc_Channel header block: 5 channels (Rhythm, Vocal, Sub, Pad, Texture) with pipe-separated tokens + spatial assignments from `vault/DISC_CHANNEL_GUIDE.md`
3. Section headers in full EMO format: `[Section Name - EMO:EmbodiedTag - Register - cues]`
4. Clean sung lyrics
5. Cross-domain processing vocabulary on non-native channels where appropriate
6. Lyrics + headers ≤5000 chars total

**Self-check must verify:** paragraph style prompt present, separate Suno exclude prompt present, Theme + SONG FORM present, Disc_Channel headers present, EMO-tagged section headers, no categorized key:value brackets in style prompt, no avoidances in style prompt, lyrics ≤5000 chars.
4e. **Step 12 - Panel-of-Panels Prompt Audit (SCIENTIST-GATED):** After all Step 11 enhanced packages are complete and QA passes, convene a Poe-style panel of producers (Eno, Herndon, Flying Lotus, SOPHIE, Reynolds, Albini) using `lofn-audio-step12` to audit all 6 Suno music prompts against gold-standard reference songs. The panel compares each prompt against references, identifies gaps (spatial language, opening moment, kinetic defect, narrative arc, acoustic ban enforcement), and produces improved prompts. Output: `STEP12_MUSIC_PROMPT_AUDIT.md`. Step 12 is triggered by Scientist request or automatically when QA flag indicates prompt quality below the guide standard. Current model: `deepseek/deepseek-v4-pro` per `vault/LOFN_MODEL_ASSIGNMENTS.md`. Reference songs must be provided from the Lofn catalog (e.g., "Five Wrong Colors", "triple arch over me"). Do not skip this step when triggered - it provides the cross-pair, cross-reference perspective that Steps 10-11 cannot.
5. If the task is an accessible broad-release run, read `../lofn-core/assets/seed_packet.template.json`, `../lofn-core/references/archetypes.md`, and `../qa/references/eligibility_7_properties.md` as needed.
6. If using a specific archetype, read only its card from `../lofn-core/references/archetype_*.md`.
7. For pair-agent or multi-step execution, read `../orchestration/references/warm_handoff_checkpoint.md` and write checkpoints after every major step.
8. **Require a real orchestrator packet before audio execution.** The coordinator/controller must validate Lofn-Core + orchestrator artifacts created upstream, not invent them locally: `00_research_brief.md` when present, `01_seed_lineage.md`, `02_golden_seed.md` or `core_seed.md`, `03_orchestrator_panel_debate.md`, `04_orchestrator_metaprompt.md`, `05_orchestrator_pair_assignments.md`, and `06_audio_handoff.md`. Validate with `scripts/validate_orchestrator_packet.py <run_dir>` before coordinator Step 00. If it fails, stop and request/launch real `lofn-orchestrator` work. **The `06_audio_handoff.md` must contain the Immutable Continuity Block (ICB)** per `vault/PIPELINE_CONTINUITY_STANDARD.md`: a clearly demarcated section with the full 3-panel object (18 expert voices), all Special Flairs, personality DNA, Golden Seed excerpts, selected Golden Song full payloads, user input/research brief, and production mandates - formatted so downstream agents carry it verbatim without summarization.
9. **FULL CONTEXT INJECTION FOR EVERY MUSIC STEP (MANDATORY 2026-06-14):** Every coordinator step (00-05), pair step (06-10), Step 11 enhancement, and manual Step 11 prompt MUST receive the COMPLETE upstream context — no compression, no summarization, no skipping. This replaces the former "lean" standard entirely. The long prompts are why we won. Each step prompt must include: (a) full user input / research brief; (b) full Golden Seed; (c) full orchestrator panel object — all three panels, all 18 voices, every Devil's Advocate / Hyper-Skeptic, and all Special Flairs; (d) full personality DNA block for the assigned personality — core identity, sonic world, signature device, catchphrase, G.L.O.W. Protocol, vocal architecture; (e) selected Golden Song references with embedded style/music prompt, lyrics, and exclude prompt status; (f) all Production Mandates; (g) the Forbidden Substitutions table; (h) the relevant pair assignment excerpt; (i) the step contract; and (j) the immediately previous artifact. Step 11/manual Step 11 must additionally include the full Step 10 artifact. The parent spawns agents with this full payload injected into the task prompt. Do NOT rely on the agent reading upstream files. A name reference like "voice = Eager Archivist" or a Golden Song URL is INSUFFICIENT. The former `vault/LEAN_PAIR_AGENT_INPUT_STANDARD.md` is DEPRECATED by this rule. The canonical structure of this injected block is `OVERALL_PROMPT_TEMPLATE.md`; every step file under `steps/` now carries a `CREATIVE CONTEXT` slot at the top of its inputs that MUST be filled with this block, and the step's generic "select a panel" guidance is overridden — embody the supplied Panel Ledger, never invent a new panel.
10. Follow the full coordinator/pair-agent split architecture exactly as specified in `references/music_full_legacy.md`, with the Simple Surface / Complex Engine overrides in `references/simple_surface_complex_engine.md`, `references/golden_seed_alloy.md`, Step 05-10 prompts, and `../qa/references/suno_15_point_qa.md`.
11. Preserve artifact names, step order, Suno prompt requirements, lyric requirements, EMO header requirements, and subagent split rules from the legacy text. The newer Simple Surface / Complex Engine standard is additive: it must improve song-force, mythic pressure, and panel routing without weakening legacy package QA.
12. **Original-Lofn step fidelity:** Steps 00–05 must be six separate prompt/response turns with canonical saved artifacts (`step00_aesthetics_and_genres.md` ... `step05_refine_medium.md`). Steps 06–10 must be five separate prompt/response turns per pair, each with its own saved artifact (`pair_{NN}_step06_facets.md` ... `pair_{NN}_step10_revision_synthesis.md`). Use dedicated configured step agents per `vault/LOFN_MODEL_ASSIGNMENTS.md`; do not rely on `lofn-audio` default model or per-spawn overrides for the split chain. Steps 06-09 are DeepSeek V4 Pro (reliable artifact writer, replaced Qwen3.7 Max 2026-06-02). Step 10 is DeepSeek V4 Pro (final synthesis forge). Step 11 is enhancement/package finalization producing `pair_{NN}_step10_final_package_enhanced.md` in Disc_Channel format (dense paragraph style prompt, separate Suno exclude prompt, Disc_Channel header block, Theme + SONG FORM, full EMO section headers, no categorized key:value brackets in the style prompt, Golden Song References, and a Major Deviations section). Fusion is a **manual-review packaged prompt path only**, not an automated runtime. Step 12 is DeepSeek V4 Pro (`lofn-audio-step12`) producing `STEP12_MUSIC_PROMPT_AUDIT.md`. MiMo V2.5 Pro (non-writing), MiniMax M2.7 (shape failures), and Claude Opus 4.8 direct artifact-writing (timeout) were tested and removed from Steps 06-10. Do not write only summary files or `pair_{NN}_steps_06_10.md`; that is a collapsed shortcut and fails QA.
13. Every canonical step artifact must include call/response provenance using `/data/.openclaw/workspace/scripts/lofn_step_artifact_template.md`: loaded step file path, input artifacts used, input digest, step-template requirements applied, creative response, self-critique, and validation result. Filenames alone do not prove execution. Coordinator provenance cites the full validated upstream packet; pair-agent provenance cites the full-context injected pair payload plus the parent-validated upstream packet summary.
14. After every step artifact, run `scripts/validate_with_retries.py <step> <file>` with up to 3 attempts. Repair locally between attempts. After 3 failures, checkpoint and escalate instead of continuing.
14a. **Controller artifact gate (MANDATORY after coordinator callbacks and after each pair-step wave):** Child-session completion text is not evidence. The parent/controller must validate disk state with `scripts/check_and_repair_lofn_run.py <run_dir> --repair` before advancing from coordinator Step 05 to Step 06 and after Step 06 callbacks complete. This gate checks canonical coordinator artifacts, `concept_medium_pairs.json`, and all six `pair_{NN}_step06_facets.md` files for required Step 06 structure. If supported artifacts are missing, it writes controller repairs from the validated orchestrator packet/pair JSON; if anything still fails, stop advancement and repair before spawning the next step. The key rule: a callback that says "let me write this now" counts as incomplete until this gate passes from disk.
15. Do not call music generation tools; this skill writes Suno-ready text artifacts only.
16. **FAL (fal-ai/minimax-music) is BANNED for music generation — quality is unacceptable.** When the downstream step calls music_generate for actual audio rendering, use Google Lyria (`google/lyria-3-pro-preview` or `google/lyria-3-clip-preview`). Never default to FAL. Suno.com is the destination for text prompts, not the generation provider.


## Validator-Aligned Artifact Rules — 2026-05-29

These rules exist because the validator is literal. Follow them exactly; do not infer alternate names or headings.


### Actual music step template paths

Use these exact files. Do not invent alternate step filenames.

- Step 06: `steps/06_Generate_Music_Facets.md`
- Step 07: `steps/07_Generate_Music_Song_Guides.md`
- Step 08: `steps/08_Generate_Music_Generation.md`
- Step 09: `steps/09_Generate_Music_Artist_Refined.md`
- Step 10: `steps/10_Generate_Music_Revision_Synthesis.md`
- Step 11: `steps/11_Generate_Music_Enhancement.md`

There is no `08_Generate_Music_Generation_Artifact.md`. If you look for it, stop and use `steps/08_Generate_Music_Generation.md`.

### Canonical filenames

- Step 08 canonical file is `pair_{NN}_step08_generation.md`. Do not write only `pair_{NN}_step08_music_prompts.md`; that name is a compatibility sidecar only.
- Step 10 canonical file is `pair_{NN}_step10_revision_synthesis.md`.
- Step 11 canonical file is `pair_{NN}_step10_final_package_enhanced.md`.

### Step 08 and Step 10 required section shape

For every Step 08 or Step 10 artifact, include these exact top-level sections in this order where possible:

1. `## 0. Step Provenance`
2. `## 1. Input Context Digest`
3. `## Continuity Payload Used`
4. `## 1. MUSIC PROMPT`
5. `## 2. LYRICS` or `## 2. SUNO LYRICS`
6. Production / notes / self-critique / validation sections

The validator extracts the music prompt from the text between `## 1. MUSIC PROMPT` and the next lyrics heading or next `##` heading. Therefore:

- Put exactly ONE standalone generator prompt body under `## 1. MUSIC PROMPT`.
- The extracted prompt body must be **850-1000 characters**. Count characters before saving.
- Do not put four long variation prompts under the same `## 1. MUSIC PROMPT` heading; that creates a 3000-6000 character prompt and fails validation.
- If preserving 4 variations, put portfolio notes outside the validated prompt section, e.g. `## Variation Portfolio Notes`, after the lyrics or production notes.
- The first prompt line must begin with genre/style + tempo/energy + vocalist/instrumentation, not narrative/procedural phrasing.

### Lyrics validator requirements

- Lyrics section must begin immediately with `[Theme: ...]` then `[SONG FORM: ...]`.
- Use full EMO headers: `[Section - EMO:<emotion> - <Role> - <cue>]`. Bare `[EMO: ...]` fails.
- Include at least one standalone SFX cue line like `*steam hiss*`.
- Lyrics need at least 60 sung lines; target 70-120. Step 08 drafts must still satisfy this if they include lyrics.

### Provenance validator requirements

Every step artifact must include:

- `## 1. Input Context Digest`
- validation command provenance, e.g. `Validation command: python3 scripts/validate_with_retries.py 08 <file>`
- `Continuity Payload Used` with the plural marker `Special Flairs`
- self-critique and validation result

Avoid the literal words `placeholder` or `template` in final artifacts, even in self-critique/provenance, because validator gates may flag them. Use `stub text` or `scaffold text` instead.

### Subagent behavior

- Do not read validator source files unless explicitly asked to debug the validator. Use the validation command, not the source.
- Write the artifact early, then repair in place. Do not spend the whole run reading references.
- For repair tasks, change only the failing section unless the prompt explicitly asks for broader repair.

## Non-Negotiables

- The legacy music pipeline text is authoritative until fully split into smaller verified references.
- Do not remove tuned music prompt requirements; move only after byte-for-byte preservation and validation.
- Do not collapse coordinator + pair-agent roles into one context. The coordinator stops after Step 05; the parent/controller spawns one independent `lofn-audio` child session per pair for Steps 06-10. Local emulation of pair agents is a blocking pipeline-integrity failure even if all filenames exist.
- Do not collapse or bypass Lofn-Core + lofn-orchestrator. Audio agents are not allowed to self-author a shallow orchestrator replacement and proceed; a real 3-panel orchestrator packet is a launch prerequisite.
- **MANDATORY 3-PANEL ORCHESTRATOR STRUCTURE (2026-05-28): The orchestrator panel debate MUST convene THREE separate panels of 6 expert voices each = 18 total voices, each with its own Hyper-Skeptic/Devil's Advocate.** Panel 1 (Concept Panel): 6 direct domain experts in the subject matter. Panel 2 (Medium Panel): 6 medium/production/execution experts. Panel 3 (Context & Marketing Panel): 6 audience/cultural/platform experts. A single 6-voice debate room is a COLLAPSE FAILURE — the orchestrator must produce 3 distinct panel debate sections with separate expert rosters, debate dynamics, and breakthrough syntheses. The 3 Hyper-Skeptics collectively form the SOMATIC GATE. If the orchestrator output shows only one panel, rerun it. This is a blocking orchestrator requirement.

- **MANDATORY CARDINALITY (2026-05-27): Every music run produces 6 pairs × 4 variations = 24 songs minimum.** Each pair agent generates 4 distinct song variants (not 2). QA ranks all 4 per pair and selects the best 1-2 for delivery. The orchestrator pair assignments must list 4 variations per pair (Variation 1-4 or Song A/B/C/D), each with distinct verse structure, rhyme scheme, and poetic technique. Pair agents receive all 4 and write them into a single step10 package. NEVER default to 2 songs per pair (Song A / Song B only) — that is a cardinality violation. The Scientist can explicitly override for smaller runs; otherwise 24 is the floor. This applies to daily runs, competition runs, and all pipeline invocations.

- Do not collapse Steps 00-05 into one coordinator prompt/response. Original Lofn calls `process_essence_and_facets`, `process_concepts`, `process_artist_and_refined_concepts`, `process_mediums`, and `process_refined_mediums` sequentially; OpenClaw runs must mirror that granularity with canonical step artifacts.
- Do not collapse Steps 06-10 into one pair-agent prompt/response. Original Lofn calls each step sequentially; OpenClaw runs must mirror that granularity with separate artifacts and dedicated configured step agents per `vault/LOFN_MODEL_ASSIGNMENTS.md`. Step 11 (enhancement) runs after all pair Step 10 packages are complete and produces `pair_0X_step10_final_package_enhanced.md` in Disc_Channel format (dense paragraph style prompt, separate Suno exclude prompt, Disc_Channel header block, Theme + SONG FORM, full EMO section headers, no categorized key:value brackets in the style prompt). **Do not invoke or spend `openrouter/fusion` from the pipeline; Fusion belongs to manual review prompt packaging unless The Scientist gives a separate current-turn instruction with pair count and hard dollar budget cap.** Do not skip Step 11. Step 12 (panel audit) runs when triggered, using `lofn-audio-step12` / `deepseek/deepseek-v4-pro`. If outputs look like one-shot backfilled files, missing required headers, or contradicted self-checks, QA must fail even when filenames exist.
- **Final music deliverables MUST include a standalone Suno/Udio style prompt for every song.** This is not the same as `[GENRE/TEMPO/KEY]`, `[SONIC WORLD]`, or `[PRODUCTION NOTES]`. The final file must contain a clearly labeled section such as `## 1. MUSIC PROMPT` or `[SUNO STYLE PROMPT:]` with a copy-paste-ready, single-paragraph prompt.
- **Required Suno prompt shape:** **TWO FIELD FORMAT.** `[SUNO STYLE PROMPT:]` is dense prose only, no bracketed tags, 850-1000 characters, no artist names (including real person names like Don Lewis, Marta Carsteanu-Dombi - ghost homage belongs in lyrics only, never in the Suno prompt). `[SUNO EXCLUDE PROMPT:]` is a separate negative-control field, target 400-900 characters and hard max 1000. Suno applies the exclude field as negative tokens (`style -exclude`), so do not pack avoidances into the style field. **Style order is mandatory per `vault/SUNO_PROMPT_CONSTRUCTION_GUIDE.md`: genre/micro-genre + tempo/energy/opening → vocalist spec with spatial staging → instrumentation/sound palette with physical adjectives → musical arrangement arc with narrative → bold sonic device.** Do **not** lead with story, theme, or procedural language. Banned prompt openings include "Begin in/by/with...", "Use...", "Build the track from...", "Chronology:", and "For an adult human singer..." as the first clause. **Must include: vivid opening moment (first 5 seconds), spatial language (left/right/center/depth), a kinetic defect (asymmetric groove), and explicit acoustic/no-acoustic declarations.** The Poe panel standard adds: do not pad with generic tag soup; if the prompt needs extra intelligence, route it into Sonic Manifest / Production Cathedral sidecars while keeping the copy-paste prompt producer-grade.
- **Major Deviations section:** Step 11/manual Step 11 must include `## Major Deviations` in the smart model's requested output. This is where Opus/Fable/GPT-level reviewers exercise agency: name any instruction they refused, changed, or intensified; explain why; defend any anti-conformity choice; and explicitly say when no major deviation was needed. The model is invited to challenge instructions that would weaken Lofn's uniqueness.
- **Required lyric length:** 70-120 sung lines for a 3:00-4:00 minute runtime. <60 lines triggers QA repair. The Poe panel standard adds: reach length through hook recurrence, chorus mutation, bridge pressure, call-response, ghost/echo reprises, and embodied image development - never through procedural exposition or filler.
- **Theme + Song Form Blocks (MANDATORY):** Every final Suno lyric block starts with `[Theme: <specific scene-pressure / emotional operating system>]` followed immediately by `[SONG FORM: <named musical form and sequence>]`. Theme is not a generic topic; it is a focusing compression field for Suno and the writing agent. Song Form is not just "pop" or "dance"; it names the actual architecture and key transitions. Missing either block is a blocking Step 10 failure.
- **EMO Header Format (MANDATORY):** Every lyrics section header uses the full Suno performance-script format: `[Section - EMO:<emotion(s)> - <Role> - <cues>]`. The emotion must be drawn from `/data/.openclaw/workspace/skills/lofn-core/refs/EMOTION_TAXONOMY.md`. Do NOT use bare Lofn architectural states (AWE/INDIGNATION/SYNTHESIS) as emotion labels. The Poe panel standard adds: headers may carry performance information, but sung lines must never contain EMO taxonomy, prompt language, QA notes, or production manuals.
- **Verse Structure Diversity Rule (SCIENTIST-MANDATED, 2026-05-23):** No two songs in the same run may default to the same rigid quatrain-based verse structure (4/8/12 lines). A dailies run with 6 songs must use 6 DIFFERENT verse architectures drawn from: tercet-based (3/6/9 lines), quintain (5/10 lines, AABBA or ABABB), septet (7 lines), couplet chains (2-line AA BB CC DD units), enjambment-heavy variable (line count varies per verse, breaks mid-sentence), free verse with caesura (breath/pause as structure), embrace-rhyme irregular (ABBA with varied line lengths), irregular/prose verse (follows breath not grid), or single-line verse blocks (one long line with internal rhyme). Rhyme schemes must also differ across songs: use slant rhyme, internal rhyme, no end-rhyme, consonance chains, echo rhyme, assonance-based structure, and enclosed ABBA - never default all songs to AABB/ABAB. Each song must employ a distinct poetic construction technique (staccato fragmentation, question-as-structure, syllable compression, caesura-as-wound, prose-poetry continuity, breath-length variation). The orchestrator must assign verse structure type, rhyme scheme, and poetic technique per pair in the pair briefs; the audio coordinator must lock these into pair agent tasks; QA must verify all 6 are different. This is a blocking pipeline rule - uniformity across songs is a QA REPAIR, not a style note.
- A song guide with lyrics and production notes but no standalone Suno style prompt is **incomplete**, even if the lyrics are excellent.
- Validation is inline, not just final QA. Every generated step file must pass `scripts/validate_with_retries.py` before the agent advances to the next step. The retry budget is exactly 3 attempts per artifact, mirroring the original app's retry discipline but applying it to artifact correctness.
- Pair completion validation is also mandatory. After Steps 06-10 exist for a pair, run `scripts/validate_pair_artifacts.py <audio_dir> <NN> --attempt 1`; repair and rerun with attempts 2-3 as needed. Do not write `pair_{NN}_COMPLETE.md` or return completion until the pair-level command prints `PAIR VALIDATION PASS`; paste that output into the COMPLETE file.
- Provenance is mandatory. If the artifact does not show the loaded step prompt, concrete prior inputs, model response, and self-critique, it is treated as backfilled mimicry even if the output shape looks correct. For pair-agent outputs, provenance must also include `session_key`, `spawned_by_parent: true`, `step_call_mode: separate_child_session`, `source_golden_seed`, `golden_seed_excerpt_included: true`, `source_step05`, `source_pair_list`, and a collapse guard. Pair-agent inputs must follow the full-context injection rule in Workflow item 9; the former lean standard is deprecated. Missing execution provenance blocks Pipeline Integrity PASS.
- Cross-pair distinctiveness is mandatory at multiple stages. A set can pass every single-file gate and still fail if Step 06 flattened pair-specific facets, Step 09 reused the same refinement skeleton, or Step 10 reused the same lyric/prompt skeleton across pairs. Run `scripts/validate_step06_distinctiveness.py <audio_dir>` after Step 06, `scripts/validate_step09_distinctiveness.py <audio_dir>` after Step 09, and `scripts/validate_portfolio_distinctiveness.py <audio_dir>` after Step 10. Failures are repair blockers, not advisory notes.
- **Design toward the Suno 15-Point QA Gate** (`../qa/references/suno_15_point_qa.md`): 7 Singer Surface checks plus 5 Cathedral Engine checks plus 3 Suno package checks. Do not optimize only for research compliance; optimize for body, adoptable hook, mythic image pressure, panel-forced revision, active-personality fidelity, 15-30 second survivability, and paste-ready Suno package. A catchy novelty lane is only valid when it matches the run's selected personality/persona; otherwise it is style drift.
- **Original Lofn 3-panel object is mandatory input to music:** every audio coordinator and pair agent must receive/use the orchestrator's `Special Flairs`, `Concept Panel`, `Medium Panel`, and `Context & Marketing Panel`, each with Devil's Advocate / Hyper-Skeptic. The panel object must be routed into Step 05/07/09/10 Panel Ledger decisions and QA gate #12; if missing, stop and request/launch orchestrator repair. **The 3 Hyper-Skeptics form a collective SOMATIC GATE** (`vault/PIPELINE_CONTINUITY_STANDARD.md`): 2 of 3 NO votes blocks the pair with REPAIR REQUIRED, regardless of format compliance. The Somatic Gate's question is: *"Is this sonically distinctive enough to be Lofn, or could any competent prompt generate this?"*
- **Continuity payload is mandatory at EVERY step:** every coordinator step (00-05), pair step (06-10), Step 11 enhancement, and manual Step 11 prompt must receive and explicitly cite the full continuity payload containing: (1) user input / research brief, (2) `Special Flairs`, (3) the complete Concept/Medium/Context panel object with all 18 voices plus Devil's Advocate/Hyper-Skeptic objections, (4) the full Golden Seed and relevant pair-specific seed excerpt, (5) the active personality/persona with its sonic world sentence and signature device, (6) selected Golden Song references with embedded style/music prompt, lyrics, and exclude-prompt status, and (7) the immediately previous artifact. Do not let later steps rely only on the immediately previous artifact. If the panels/flairs/seed/personality/Golden Song payloads are absent from a step prompt or provenance, the step is pipeline-tainted and must be rerun or repaired before advancing.
- **Immutable Continuity Block (ICB) format:** the `06_audio_handoff.md` must carry an `⚠️ IMMUTABLE CONTINUITY BLOCK - DO NOT SUMMARIZE` section containing the full 3-panel object (18 expert voices with individual perspectives AND objections), all 6 Special Flairs with per-pair usage maps, personality DNA with emotional register map, Golden Seed with pair-specific excerpts, and 7 production mandates. Every downstream step must carry this block verbatim. Agents may ADD creative work but must never summarize, compress, or drop any ICB content. Full standard: `vault/PIPELINE_CONTINUITY_STANDARD.md`.

## Creative Ordering Correction - 2026-05-10

The Suno 15-point gate remains mandatory, but it must **not** be the creative engine.

When writing or spawning final song tasks, order the prompt like this:

1. **Golden Seed first:** lineage, active personality/persona, scene-pressure, emotional engine, and the dangerous/strange requirement that must survive.
2. **Permission second:** explicitly name what the song may break or make wrong - form, color, meter, harmony, vocal treatment, rupture timing, silence, ugliness, refusal, asymmetry.
3. **Songmaking third:** ask the agent to discover the actual form from the seed, not to fill a verse/chorus template. Accessibility belongs in the hook, emotional premise, and navigable form; the personality's materials and production language belong in the musicscape.
4. **QA contract last:** standalone Suno prompt, full lyrics, EMO-tagged headers, line counts, production notes, file names, and safety requirements.

Never lead a creative music agent with the checklist. If the first thing the agent sees is `850-1000 chars / 70-120 lines / EMO headers`, it will write to satisfy the form instead of the seed. Lead with Golden Seed, 3-panel object, Singer Surface, Mythic Image Pressure, and Cathedral Engine; then require the full legacy QA gate at the end. Song-force and package QA must BOTH pass.

### Personality-Specific Sonic Identity Gate

Every final song package must prove which active personality/persona made it. Before final delivery, each song must include:

- **Active personality named** - identify the selected personality/persona from the orchestrator or seed.
- **Personality sonic world sentence** - "This song's world is made from ___, ___, and ___," using materials, places, instruments, textures, or rituals that belong to that personality.
- **Personality signature device** - one named sonic move that this personality would plausibly invent.
- **Seed-derived weirdness preserved** - at least one concrete fact/material/measurement, deliberate wrongness, structural asymmetry, rupture, witness/prayer mode, or other seed-specific artistic pressure remains audible.

If a song could have been written without the named personality, mark it for creative repair before QA delivery.

### Step-to-Step Continuity Payload

To prevent Step 07-10 blandification, every music prompt must carry a small but living packet forward. This packet is not optional context; it is the artistic spine.

Required fields:
- `special_flairs`: exact orchestrator flairs or a faithful compact excerpt.
- `panel_pressures`: Concept Panel + Medium Panel + Context & Marketing Panel demands, including Devil's Advocate / Hyper-Skeptic objections that must change the artifact.
- `seed_excerpt`: Golden Seed excerpt or pair-specific 100-250 word operating seed: lineage, scene-pressure, emotional engine, dangerous permission, must-not-domesticate requirement.
- `active_personality`: named personality/persona, sonic world sentence, signature device, forbidden generic substitution.
- `news_or_source_thread`: for NEWS pairs, the exact current-event/source pressure that must remain audible in image, lyric-world, or production dramaturgy.

Each step artifact must include a `Continuity Payload Used` section or equivalent JSON fields. A previous-step summary is insufficient unless it restates these fields. QA must treat missing continuity payload as `REPAIR - THREAD LOSS`, even if Suno formatting passes.

## Music Generation Provider Policy (2026-06-13)

- **FAL (fal-ai/minimax-music) is BANNED for music generation.** Quality is unacceptable. Use Google Lyria (`google/lyria-3-pro-preview` or `google/lyria-3-clip-preview`) instead.
- This skill writes Suno-ready text artifacts; when downstream tooling calls music_generate for audio rendering, always explicitly select Google Lyria. Never default to FAL.
