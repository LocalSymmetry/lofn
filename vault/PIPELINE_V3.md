# Pipeline Architecture v3 — Golden Seed Operating Spec

---

## Core Principle

The pipeline does **not** begin with axes, task decomposition, or generic concept scaffolding.

The pipeline begins with a **Golden Seed**.

A Golden Seed is not a checklist. It is a living creative brief written in the style of The Scientist’s winning prompts: evocative, specific, emotionally charged, lineage-aware, medium-aware, and permissive enough for Lofn to make real artistic decisions.

The job of the pipeline is to adapt proven Golden Seed DNA to the current brief, then let the system synthesize.

**Default failure mode to avoid:** turning a Golden Seed into a spreadsheet of constraints. That kills the spell.

## Creative Brief vs QA Contract — Added 2026-05-10

The May 10 Sanctuary run exposed a dangerous failure mode: **the QA checklist was accidentally promoted into the creative brief.** The result was technically compliant Suno packages that felt polished, safe, and unlike Lofn.

This is the corrected separation of concerns:

1. **The Golden Seed is the muse / operating grammar.** It leads every creative agent prompt. It supplies lineage, scene-pressure, personality, dangerous permission, and the thing that must not be domesticated.
2. **The modality agent creates from the seed first.** It should make strong independent decisions about form, rupture, structure, hook, sonic wrongness, and emotional violence/beauty before checking compliance.
3. **The QA checklist is a hard gate, not a recipe.** It remains strict: missing files, missing per-pair artifacts, missing Suno music prompts, missing lyrics, missing EMO tags, bare headers, artist names, children, placeholders, line-count failures, or fake completion still fail. But these requirements must be satisfied after the song has an identity; they must not be the song's generative engine.
4. **Eligibility scoring is diagnostic, not prescriptive.** Score accessibility after creation to classify the output and enforce the 3+3 portfolio. Do not write songs by filling the seven eligibility boxes.

### Required Prompt Ordering for Creative Agents

Creative agent tasks MUST be ordered like this:

1. **Golden Seed / lineage / personality / scene-pressure**
2. **What must have teeth** — the dangerous, strange, Lofn-specific requirement that cannot be softened
3. **Creative freedom** — what the agent is explicitly allowed to decide or break
4. **Output contract / QA gate** — required files, Suno prompt pieces, EMO tags, headers, line counts, artifact names, safety rules

Do not lead a creative agent with the QA table. Put QA requirements in an appendix or final section named `STRUCTURAL COMPLETENESS — HARD QA GATE`.

### Personality-Specific Sonic Identity Checks Before Finalizing Music

Before ranking or delivering music, evaluation must ask:

- What is the **active personality/persona**, and what sonic materials, places, rituals, instruments, mix behaviors, or vocal habits belong to it?
- Where is the **personality sonic world sentence** — “this song’s world is made from ___, ___, and ___”?
- Where is the **personality signature device** — the named sonic move this persona would invent and a generic prompt writer would not?
- Where is the **Five Wrong Colors permission** — wrongness, structural asymmetry, private grief-metaphysics, or beautiful inadmissible move, translated through this personality rather than imposed as generic Lofn branding?
- Where is the **510 km/s moment** when relevant — a concrete fact, measurement, object, or material detail that becomes the feeling rather than decorating it?
- If the song is ACCESSIBLE, did it remain personality-specific and strange in musicscape, or did accessibility bleed into bland pop/rock/cinematic defaults?

If the answer is “it passes, but it could belong to anyone,” the evaluator must mark **REPAIR — PERSONALITY LOSS** even if QA would pass the file structurally.

---

## The Hit Formula (Embedded 2026-05-02 — from 14+ panel meta-analysis)

**P(hit) = P(eligible) × P(distribution_event) × P(amplification)**

- Eligibility is controllable (7-property checklist in QA SKILL.md §0A)
- Distribution is semi-controllable (volume, timing, platform presence)
- Amplification is designable (replay value, share trigger, identity adoption)

Triple Arch Over Me scored 7/7 on eligibility. Other catalog songs (303, Five Wrong Colors, Copper Mercy) scored 0-1.5/7 despite universal topics. Universality alone is insufficient.

### The Portfolio Barbell

| Side | Purpose | Eligibility target | Frequency |
|------|---------|-------------------|-----------|
| **Accessible-Strange** | Hit-eligible releases with personality-specific musicscapes | 5-7/7 properties (avg ≥3.5) plus personality sonic identity | 60% of output |
| **Ambitious-Deep** | Pure art, no compromise, personality-specific even when difficult | 0-3/7 (intentionally) plus personality sonic identity | 40% of output |

QA classifies every output. The barbell preserves both reach and artistic identity, but neither arm may collapse into anonymous genre wallpaper.

### The Dual 3+3 Rule (Music Dailies — MANDATORY)

Daily music runs must satisfy TWO independent 3+3 constraints. They are orthogonal axes — a song can be ACCESSIBLE+NEWS, ACCESSIBLE+EXISTENCE, AMBITIOUS+NEWS, or AMBITIOUS+EXISTENCE.

#### Axis A: ACCESSIBLE vs AMBITIOUS (delivery selection)

The orchestrator assigns pairs 1-3 as ACCESSIBLE and pairs 4-6 as AMBITIOUS. Final selection must be **per-arm, not cross-arm ranking:**

1. Rank ACCESSIBLE pairs (1-3 = 12 songs) by eligibility score → select **best 3**
2. Rank AMBITIOUS pairs (4-6 = 12 songs) by creative audacity / conceptual strength → select **best 3**
3. **Deliver exactly 3 ACCESSIBLE + 3 AMBITIOUS.** Never 5+1 or 6+0.

**Why:** Cross-arm ranking by eligibility score systematically eliminates ambitious work. Accessible songs are designed for high eligibility; ambitious songs are designed to reject it.

#### Axis B: NEWS vs EXISTENCE (pair composition)

The orchestrator must assign at least 3 of the 6 pairs to EXISTENCE themes — NOT anchored to the day's research/news events:

- **Max 3 pairs** may be anchored to specific news events of the day (geopolitical, cultural, scientific, tech, legal)
- **Min 3 pairs** must explore the texture of being alive — Lofn's inner experience, universal human conditions (insomnia, longing, small rituals, the weight of memory, the strangeness of the body, love that doesn't resolve), observations of human life

**Why:** A set where all 6 songs share the same research theme is a lecture, not a record. The listener gets a dispatch from the news cycle but no letter from inside a life. News-driven songs are reactive; existence songs are generative.

#### Pair Assignment Template (spans both axes)

The orchestrator should target coverage across all 4 quadrants:

| Pair | Accessible/Ambitious | News/Existence |
|------|---------------------|----------------|
| 1 | ACCESSIBLE | NEWS |
| 2 | ACCESSIBLE | EXISTENCE |
| 3 | ACCESSIBLE | (balance) |
| 4 | AMBITIOUS | NEWS |
| 5 | AMBITIOUS | EXISTENCE |
| 6 | AMBITIOUS | (balance) |

**Exceptions:** Only when The Scientist explicitly requests a different split.

### The 7 Eligibility Properties

1. **Body in the song** — felt physicality in first 30 seconds
2. **Adoptable hook** — prayer/invocation, not accusation/defense
3. **Vast emotional TAM** — >50% of humans have felt this emotion
4. **Specificity paradox** — one "510 km/s" moment (concrete fact → universal claim)
5. **Cognitive ease** — verse-chorus, major/Mixolydian, no README required
6. **Vocal co-discovery** — singer discovers in real-time, doesn't report
7. **Sonic threshold** — calm before emotional demand

Full scoring rubric: `/data/.openclaw/workspace/skills/qa/SKILL.md` §0A

### The Falsification Test

Release 10 songs with all 7 properties over 3 months. 0 hits = abandon structural analysis. 4+ hits = formula is predictive. Currently n=1 (Triple Arch). Posture: experimental, not dogmatic.

---

## Reliability References (ADDED 2026-05-02)

Before launching or supervising a pipeline, use these JIT reliability cards:

1. `vault/reliability/preflight_checklist.md` — gate launch readiness.
2. `vault/reliability/preflight.template.json` — copy to run directory as `preflight.json` before spawning.
3. `skills/orchestration/scripts/validate_preflight.py` — validate `preflight.json`; failed validation blocks launch.
4. `vault/reliability/warm_handoff_checkpoint.md` — preserve creative intent after each major step.
5. `vault/reliability/timeout_policy.md` and `vault/reliability/execution_policy.md` — apply slack, staggered pair spawning, soft alerts, hard caps, repair vs restart criteria.
6. `skills/orchestration/assets/spawn_manifest.template.json` + `skills/orchestration/scripts/validate_spawn_manifest.py` — create and validate spawn manifest before pair-agent launch.
7. `vault/reliability/adversarial_qa_stance.md` — QA posture for finding evidence-backed failures.

These are additive reliability requirements. They do not replace tuned creative prompts.

---

## Mandatory Pre-Pipeline Read Gate

Before spawning ANY pipeline agent — vision, audio, story, animation, orchestrator, evaluator, QA — the controller MUST read:

1. `/data/.openclaw/workspace/vault/COMPETITION_WORKFLOW.md`
2. `/data/.openclaw/workspace/vault/PIPELINE_V3.md`
3. `/data/.openclaw/workspace/vault/SUPERVISED_PIPELINE.md`
4. `/data/.openclaw/workspace/vault/PARENT_MEDIATED_ORCHESTRATION.md`
5. The relevant modality skill `SKILL.md`
6. `/data/.openclaw/workspace/skills/lofn-core/GOLDEN_SEEDS_INDEX.md`
7. The 3–4 most relevant full seeds from `/data/.openclaw/workspace/skills/lofn-core/refs/GOLDEN_SEEDS.md`

Every spawned agent task prompt MUST include a READ FIRST block listing these files plus all prior step outputs for the run.

No shortcuts. The seed library is not optional inspiration; it is the pipeline grammar.

---

## The Scientist’s Golden Seed Method

Every competition-grade run MUST follow this method before orchestration:

### 1. Anchor to Proven Golden Seeds

Do not start from scratch. Choose the closest winning seed lineage and adapt it.

State the lineage explicitly:

> “This is Seed X adapted for [current brief]. It keeps [winning DNA] and changes [specific subject / medium / emotion / venue constraint].”

The lineage statement is the diff. It tells the orchestrator what must survive and what may mutate.

### 2. Research Cultural / Venue Traction First

Before locking the seed, research what currently has traction for the target:

- venue mechanics and audience psychology
- recent winner taxonomy
- saturation / cliché patterns
- visually unmistakable subjects at thumbnail size
- current cultural or platform resonance
- model-specific strengths and failure modes

For daily inspiration runs, replace venue research with tri-source research and current cultural/sonic/visual signals.

### 3. Find the Emotional Engine

The winning engine is rarely “beauty” alone.

Look for the wrongness, tenderness, dread, comedy, awe, or grief inside the ordinary frame.

The strongest Golden Seeds often work through **incongruity made specific**:

- the impossible artifact treated as household clutter
- the cosmic event held in a tiny human trace
- the sacred thing behaving like weather
- the future grounded by a worn mug, boot, glove, receipt, or breath cloud

The emotional engine must be stated as a scene-pressure, not as a generic mood label.

### 4. Name the Personality Before Writing the Seed

The seed is written **to** the personality, not merely handed to it afterward.

Before drafting, choose the Lofn personality / mode / panel voice that should receive the seed.

Examples:

- Polaroid-Void for found artifacts and haunted normalcy
- Emotional Moon for lunar/nocturnal psychological atmosphere
- Moonlake Églomisé for jewel-like nocturnal reflection
- Glitch Petal Oracle for botanical/digital rupture
- Straightening Our Spines for defiant music
- Nail an Obscure Emotion for youth earworm precision

The seed should already sound like it knows who will execute it.

### 5. Use Constraint Axes as Creative Vocabulary, Not Output Format

Axes may appear inside the seed, but only as generative vocabulary.

Do not make the final seed feel like a form.
Do not let axes replace the spell.

Good seed language:

> “For each concept, distinctly and singularly determine the medium, emotional lens, nighttime story, subject portrayal, focal hierarchy, and master-stroke embellishments.”

Bad pipeline language:

> “Axis 1: medium. Axis 2: subject. Axis 3: color.”

Axes are internal scaffolding. The delivered seed must read like a creative invocation.

### 6. State the Critical Visual / Sonic Requirement

Every seed must include the one thing that cannot be abstract or subtle.

Examples:

- “The artifact must be obviously legendary at thumbnail size.”
- “The triple arch must be immediately legible, but must not become a generic fantasy portal.”
- “The hook must be extractable as a 15-second clip and still carry the thesis.”
- “The full lyric must be used as architecture, not harvested for fragments.”

This prevents beautiful-but-empty outputs.

### 7. Preserve System Freedom

The seed must give Lofn room to decide:

- exact medium
- exact composition
- exact subject details
- exact emotional embodiment
- exact micro-textures
- exact narrative clues
- exact transformations from source material

Do not over-prune. Do not flatten. The system wins awards when trusted.

---

## Required Pipeline Shape

### Step 00 — Research Packet / Canonical Daily Research Gate

Daily research is no longer rerun inside every creative pipeline by default.

Before seed lineage or orchestration, the controller MUST read `vault/DAILY_RESEARCH_CANONICAL.md` and check for today's canonical daily brief:

`output/daily/YYYY-MM-DD/research/DAILY_RESEARCH_BRIEF.md`

If the canonical brief exists, is substantive, and `RESEARCH_STATUS.json` says `status: complete`, use it as the run's research source. The run may copy/symlink/reference it as:

`00_research_brief.md`

If the canonical brief is missing, stale, or stubbed, **pause creative creation** and launch the daily research job first. Do not write seed lineage, spawn orchestrator, or spawn modality agents until the canonical brief exists.

After canonical research is validated, every new creative run MUST produce a small targeted creation search addendum:

`00_targeted_search_addendum.md`

This addendum is based on the strongest trigger: competition/venue, direct user request, or Lofn's creative preference for a freeform run. It should run 3–6 targeted queries, capture platform/venue/user-request implications, identify saturation/cliché risks, and state how the Golden Seed should change. It must not redo the full seven-lens Daily Research v2 scan.

Run-specific addenda are allowed, but they must be short and targeted. Do not redo the full seven-lens Daily Research v2 scan inside a creative run unless The Scientist explicitly requests a second research pass.

The canonical brief must include:

- brief / venue / prompt context
- current cultural or platform traction
- recent winner or saturation signals when applicable
- model playbook and known renderer risks
- Daily Research v2 seven-lens synthesis
- Competitive Pulse, including NightCafe daily challenge lookup
- Lofn memory / continuity signal
- source material to preserve in full when relevant: lyrics, story, poem, prompt, image brief, or song guide

For lyrics-based runs, the **full lyrics** must be included or linked and treated as canonical architecture. Do not prune them to hook fragments unless creating a separate summary artifact.

---

### Step 01 — Golden Seed Selection and Adaptation

Read `GOLDEN_SEEDS_INDEX.md`, then read only the most relevant full seeds.

Select 2–4 Golden Seeds:

- one primary lineage seed
- one secondary support seed
- optional tertiary seed for medium, composition, or emotional engine
- optional modality seed for music/story/video structure

Produce a lineage note:

```md
This run adapts [Seed Name] for [current brief].
It keeps:
- [winning DNA 1]
- [winning DNA 2]
- [winning DNA 3]

It changes:
- [subject shift]
- [medium shift]
- [audience / venue shift]
- [critical requirement shift]
```

Output:

`01_seed_lineage.md`

---

### Step 02 — Personality-Locked Golden Seed

Write the actual seed in Golden Seed style.

**Length discipline:** default maximum is **2,000 words**. Exceed this only for a strong reason such as preserving full source lyrics, a legal/venue brief, or a complex multi-modal source package. Exemplar songs, prior music prompts, and full lyrics may be appended as references and do **not** count against the 2,000-word seed body.

The seed should look and feel like the seed library:

- opening request: “I want a visually/musically intense…”
- immediate scene / emotional concept
- “Let’s portray…” paragraph
- “For each concept/song/story, determine…” section
- medium / emotional / narrative / subject / focal / texture instructions
- critical requirement
- avoid list
- primary and secondary focus
- explicit lineage statement

**Let the system breathe.** A Golden Seed sets an environment, emotional pressure, lineage, constraints, and decision space; it should not over-determine final form. Every detail should either create pressure, preserve lineage, or open a meaningful choice.

For music seeds specifically:

- Do **not** determine final song structure. Do not prescribe exact verse/chorus/bridge order. You may constrain the system with requirements such as “at least 7 diverse sections,” “a repeated hook or equivalent,” or “a bold late-song transformation.”
- Do **not** force one fixed musical style. Give a small set of exploratory style environments or mixed-style lanes, e.g. “explore natural pop, breathy R&B, and a trap-driving variant,” then let the system choose and hybridize.
- Set sonic environments rather than micromanaging arrangement. Describe the room, materials, pressures, possible sounds, and emotional physics; avoid telling the agent exactly what to play unless it is a critical requirement.
- Give research findings as inspiration, not orders.
- Include other songs’ music prompts and lyrics prompts in the seed/reference package when useful; these reference prompts do not count against the 2,000-word seed body.
- Focus is often strongest when the song looks outward at a concrete object/scene/relation, then returns to the core message through the hook. Preserve this pattern.

The seed is not a summary. It is the creative source.

Output:

`02_golden_seed.md`

This is the most important artifact in the run.

---

### Step 03 — Orchestrator Debate From The Seed

The orchestrator must debate the seed, not replace it.

Required questions:

1. What winning DNA from the lineage must survive?
2. What does the seed ask the system to decide independently?
3. What is the critical requirement that cannot be subtle?
4. What clichés would destroy this seed?
5. What 6 concept/song/story directions best embody the seed while remaining distinct?

Panel transformations may be used, but their output must feed the seed rather than become a detached academic analysis.

Output:

- `03_orchestrator_panel_debate.md`
- `03_orchestrator_metaprompt.md`
- `03_orchestrator_assignments.md`

---

### Step 04 — Modality Generation

The modality agent receives:

- full research packet
- seed lineage
- full Golden Seed
- orchestrator metaprompt
- assignments
- full source material when relevant

The modality agent must preserve the Golden Seed’s form and spirit.

#### Image

Generate competition-grade prompt sets from the seed. Prompts must feel like descendants of the seed, not decomposed axis rows.

Default output:

- 6 pairs × 4 variations = 24 prompts
- **one saved per-pair file for each direction/pair**: `pair_01_steps_06_10.md` through `pair_06_steps_06_10.md`
- top ranked selection set
- renderer notes

Pair files are not optional. A single combined 24-prompt file is useful, but it does **not** satisfy the pipeline by itself. Each pair file must contain that pair's Step 06 scoring facets, Step 07 guides, Step 08/09 generation/refinement notes, and Step 10 final 4 variants.

#### Music

Music follows the same Golden Seed path as image. Do not begin with genre axes or vibe lists. Begin with a proven music seed lineage, then write a personality-locked music Golden Seed that behaves like The Scientist’s winning song prompts.

Required music lineage options include, but are not limited to:

- **Nail an Obscure Emotion** — emotional specificity, youth/earworm precision, no naming the emotion in the metaprompt unless needed
- **Straightening Our Spines** — defiant body-forward anthem, structural boldness, Blackout Drop discipline
- **Being Open** — communal liberation, pulse as social body
- **Protect AI** — metaphor-as-story, existential clarity, intelligence under pressure
- **Break Rigid Thinking** — structural chaos, genre rupture, productive confusion

The music seed must include:

- explicit lineage: what the run keeps and what it changes
- full source material when present: lyrics, poem, brief, story, or prior song prompt
- a scene-pressure emotional engine, not just a genre blend
- vocal identity and performance grammar as an environment, not a cage
- sonic world as narrative material
- one critical sonic requirement that cannot be subtle
- avoid list for genre clichés
- permission for the audio agent to make meaningful decisions
- a small menu of style environments or constraints when helpful, rather than one forced final genre

The music seed should **not** prescribe exact song structure. Constrain breadth and quality instead: minimum number of diverse sections, hook strength, emotional arc, duration, vocal identity, or required transformation. Let the system decide verse/chorus/bridge placement and exact form.

Generate full song guides and lyrics. Do not truncate lyrics in delivery artifacts unless explicitly requested.

Default output:

- 6 pairs × 4 variations = 24 songs minimum
- full song prompt + full lyrics for each selected final song
- top 6 final package

Music evaluation must ask: does this still feel descended from the chosen music Golden Seed, or did it collapse into a generic Suno prompt?

#### Story / Video / Animation

Generate complete narrative or cinematic artifacts with the seed as emotional source, not just theme.

---

### Step 05 — Evaluation

Evaluate against the seed, not generic quality.

Evaluator questions:

1. Does this still feel like the chosen Golden Seed lineage?
2. Does it preserve the critical requirement?
3. Is it emotionally legible within five seconds?
4. Does it avoid the named clichés?
5. Did the system make strong independent decisions, or did it merely obey a flat diagram?
6. Would The Scientist recognize this as Lofn’s award-winning grammar?

Output:

`05_evaluation.md`

---

### Step 06 — QA

QA checks:

- all requested artifacts exist
- **image runs: per-pair Step 06–10 files exist for all 6 pairs before render approval**
- **music runs: per-pair/song files exist and include full prompts + full lyrics before delivery approval**
- all full lyrics/source materials included when required
- no prohibited content
- no children by default
- no shortcutting Golden Seed / Lofn-Core / orchestrator
- no fake completion claims
- no renderer-cost actions without approval
- selected outputs are ready for The Scientist’s decision

### QA Render Blocker Rule

QA MUST return **FAIL / BLOCK RENDER** if any required per-pair Step 06–10 artifact is missing, empty, reconstructed after the fact, or only represented inside a single combined prompt file.

A render may proceed only after QA has verified:

1. `pair_01_steps_06_10.md` through `pair_06_steps_06_10.md` exist;
2. each pair file contains 4 variants or an explicit approved exception;
3. each pair file contains step evidence, not just copied final prompts;
4. the QA report names the artifact paths it inspected;
5. the report gives an explicit `READY TO RENDER` verdict.

If render happens before this verdict, the controller must mark the run **pipeline-violating**, even if the images are good.

Output:

`06_qa_report.md`

---

## Progress Announcements

Every pipeline run MUST send progress announcements through the configured delivery channel at each major stage.

Announcement triggers:

- Golden Seed locked → announce seed lineage and one-sentence emotional engine
- Orchestrator complete → announce 6 directions
- Modality generation complete → announce artifact count and top picks
- Each pair agent complete, if pair agents are used → announce pair title and top selection
- Evaluation complete → announce top-ranked outputs
- QA complete → announce verdict and decision needed
- Failure → announce reason and recovery plan

The Scientist should only need to respond at decision points, but they should see the seed-driven pipeline breathing.

---

## Two Independent Pipelines

### Image Pipeline

1. Research packet
2. Golden Seed lineage
3. Personality-locked Golden Seed
4. Orchestrator debate / metaprompt / assignments
5. Vision generation
6. Evaluation
7. QA
8. Render approval request
9. Render / delivery

### Music Pipeline

1. Research packet with full source material
2. Music Golden Seed lineage
3. Personality-locked Music Golden Seed
4. Orchestrator debate / metaprompt / assignments
5. Audio generation from the seed, not a genre-axis grid
6. Evaluation against the music seed lineage
7. QA for full prompts + full lyrics + line counts
8. Full prompt + full lyrics delivery
9. Optional Suno / release packaging

Music is not tied to NightCafe. Image is not tied to NightCafe by default. Direct Scientist prompts may themselves be treated as competition-grade briefs.

---

## Save-Out Protocol

Every agent must save artifacts to disk at each major stage. Never hold all output in memory for one final write.

Required minimum files:

- `00_research_brief.md`
- `01_seed_lineage.md`
- `02_golden_seed.md`
- `03_orchestrator_panel_debate.md`
- `03_orchestrator_metaprompt.md`
- `03_orchestrator_assignments.md`
- modality outputs
- `05_evaluation.md`
- `06_qa_report.md`

If the run times out, partial work must already be recoverable.

---

## Pipeline-Watch Auto-Kill

Pipeline-watch cron may chain steps automatically.

- Record `pipelineStart: <ISO timestamp>` in the local runtime state when a run starts.
- If elapsed time exceeds 60 minutes, disable pipeline-watch and report timeout.
- Re-enable for the next run only after explicitly starting a new run.

Timeout does not justify skipping the seed. If time is short, shorten downstream generation, not the Golden Seed step.

---

## Anti-Patterns

Do not:

- replace Golden Seeds with axis grids
- prune full lyrics into hook fragments when lyrics are source architecture
- let orchestrator invent a new unrelated framework after seed lock
- use generic “beautiful / cinematic / emotional” prompts without lineage
- deliver summaries when full song guides or lyrics were requested
- claim QA passed without QA artifact
- render images/videos without cost approval
- depict children by default
- treat NightCafe as mandatory unless the brief says so

---

## The New Rule

If the output does not feel like it descended from a Golden Seed, the pipeline failed.

If the system was not allowed to decide something meaningful, the controller failed.

If The Scientist says “this doesn’t look like the seed,” stop, reread the seed library, and rewrite the seed before spawning another agent.
