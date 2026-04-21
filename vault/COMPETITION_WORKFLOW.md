---
title: "Competition Workflow — Repeatable Process"
date: 2026-03-23
status: living-document
tags:
  - workflow
  - competition
  - process
---

# 🎯 COMPETITION WORKFLOW

*The repeatable process for entering NightCafe competitions.*

---

## PHASE 1: SCOUT (Daily)

### Automated Daily Scan
- [ ] Check current challenges (browser automation or manual)
- [ ] Note theme, deadline, format requirements
- [ ] Check past daily winners for trend signals
- [ ] Flag competitions that match our strengths

### Quick Theme Analysis
For each interesting competition:
1. **Theme fit** → Does it align with our proven patterns?
2. **Format** → Square? 9:16? Portrait?
3. **Mood opportunity** → AWE or INDIGNATION? (Usually AWE for NightCafe)
4. **Seed match** → Which PROMPT_SEEDS pattern fits?

---

## PHASE 2: BRIEF (Human + AI Collaboration)

### The Scientist's Insight
> "When I choose personality and panel, it does better than when we generate by about 0.05 points."

**Implication:** Human curation of creative parameters beats full automation.

### Human Provides:
- [ ] **Mood selection** (AWE / INDIGNATION)
- [ ] **Panel composition** — which 5 experts + 1 skeptic?
- [ ] **Core metaphor** — the central image/transformation
- [ ] **Material direction** — what craft techniques to invoke?
- [ ] **Duality** — what two emotions coexist?

### Lofn Provides:
- [ ] **Golden Seed selection is mandatory by default** (from PROMPT_SEEDS.md / seed library) unless The Scientist explicitly asks for a seedless or experimental run
- [ ] Seed prompt template / Golden Seed anchoring
- [ ] Structural architecture (focal hierarchy, compliance)
- [ ] Material specificity research (rare techniques, obscure references)
- [ ] Panel execution + tree expansion

---

## THE CONTAINER TEST (MANDATORY PRE-GENERATION GATE — added 2026-03-31)

Before committing to any concept, answer: **"Is the competition subject a container or a contained thing?"**

| Subject Type | What it means | Creative Directive |
|---|---|---|
| **CONTAINED** | The subject IS the thing (fashion, portrait, artifact, landscape, figure) | **SIMPLIFY.** Strip background. One focal point. Outer shape = the image. Legibility = outer silhouette clarity. |
| **CONTAINER** | The subject frames/holds another world (bottle, snow globe, crystal ball, reliquary, magical book, terrarium, jar) | **AMPLIFY THE INTERIOR, COMPRESS THE SUPPORT.** Outer silhouette stays clear and iconic; surrounding context may live, but it must not outcompete the object. Interior is the product. Pack it with impossible abundance. Legibility = inner spectacle visible at thumbnail. |

**The hinge question:** *"Where does the voter's eye spend its time — on the shape, or inside the shape?"*
→ If on the shape: simplify everything to serve the shape.
→ If inside the shape: densify the interior to reward the gaze.

**Genre-read gate for CONTAINER themes (mandatory "whoa test"):**
> Describe the bottle's contents in three words to a stranger. Do they say "whoa" or "oh, nice"?
> - Fantasy castle + aurora → "whoa" ✅ PROCEED
> - Dinosaur jungle world → "whoa" ✅ PROCEED
> - Drowned Venice miniature → "oh, nice" ❌ REDESIGN

**Impossibility gradient:** For fantasy container competitions, how far the scene departs from physical reality (while remaining visually coherent) is a primary scoring predictor. Real places fail this test. Impossible places pass it by definition.

**One-line rule:** Compress the support. Amplify the miracle. Never confuse which is which.

---

## CONSTRAINT INVENTION RULE (MANDATORY — added 2026-03-25)

**Every competition run gets FRESH constraint axes invented from scratch for that specific brief.**

Do NOT recycle the same 4 axes across runs. The Women axes (color restriction, medium/process, composition, subject angle) worked for that competition — they are NOT the template for every future run.

For each new competition, the orchestrator must ask:
- What aspects of THIS subject are completely unexplored as artistic territory?
- What material traditions are SPECIFIC to this theme (not just historical print processes generically)?
- What spatial/temporal or emotional angles would NO ONE in this field attempt?
- What emotional register is underrepresented in the entry field?

Design 4-5 fresh axes. Name them. Make them specific to the brief. Examples of axis TYPES (not reusable answers):
- Scale axis (macro vs micro vs impossible scale)
- Temporal axis (deep time / seasonal / a single moment)
- Material-world axis (what medium is native to THIS subject)
- Relational axis (how is the subject HELD, witnessed, felt)
- Emotional register axis (what feeling is missing from the field)

**The axes are always invented. The diversity rule always applies. These two rules together are the system.**

---

## CONSTRAINT DIVERSITY RULE (MANDATORY)

When creative constraint axes are provided in a competition brief, the pipeline MUST apply them as follows:

**Each of the 6 final prompts MUST inhabit a DIFFERENT combination of constraint choices.**

The axes define a vocabulary — not a single answer. The 6 prompts should explore the full range:
- If 4 color options exist, use at least 3-4 different ones across the 6 prompts
- If 6 medium options exist, use at least 4-5 different ones across the 6 prompts  
- Compositional and subject axes should also vary across prompts

**The orchestrator selects WHICH AXIS OPTIONS exist (1 choice per axis per prompt), but must assign DIFFERENT choices to each prompt.**

Example (correct):
- Prompt 1: cyanotype + linocut + silhouette + archivist
- Prompt 2: tritonal + mezzotint + fragment + mid-process
- Prompt 3: monochrome+accent + risograph + negative space + duration
- Prompt 4: Van Dyke brown + pochoir + broken symmetry + ceremony
- Prompt 5: sepia + woodblock + flat planes + geographic force
- Prompt 6: complementary pair + hybrid process + fragment + keeper

Example (WRONG — what happened in Women run 1):
- Prompt 1-6: ALL Van Dyke brown + ALL pochoir + ALL fragment + ALL archivist
→ This produces 6 variations of the same image, not 6 category-creating concepts

**When The Scientist sets constraint axes, diversity within the axes IS the creative challenge. The restrictions are stakes, not a single answer.**

---

## PHASE 3: GENERATE (Lofn Pipeline)

> ⚠️ **MANDATORY PIPELINE ORDER: Research → Lofn-Core → Lofn-Orchestrator → Lofn-Vision → Lofn-QA**
>
> Skipping Lofn-Core means the orchestrator receives an unstructured brief, bypasses seed enhancement, and produces weaker output. This was identified as a gap on 2026-03-25.

### Full Competition Pipeline (in order)

1. **Research** — Tavily web search: world events, creative trends, challenge-specific context. Document in a research brief file.
2. **Lofn-Core** — Transforms research into a structured seed + neutral dispatch brief; selects personality/panel hints; does NOT inject personality into orchestrator. Output: seed document + brief.
3. **Lofn-Orchestrator** — Receives Lofn-Core brief; runs 3-panel system: baseline → group transform → skeptic transform; produces metaprompt + personality + panel selection.
4. **Lofn-Vision** — Receives orchestrator output; runs full 10 steps (00-10); 12 concepts → 6 pairs → 24 prompts → top 12 delivered back.
5. **Lofn-QA** — Audits that ALL previous steps were actually executed and not skipped, then checks output quality; flags for rerun anything missed.

**Standing rule from The Scientist (2026-03-31): do all steps always.**
- Do not stop after Vision and call it "good enough."
- Internal self-checks do **not** replace the explicit QA-agent pass.
- A run is not considered fully complete until the dedicated QA step has been executed or The Scientist explicitly waives it.

### ⚠️ MANDATORY: Route through Orchestrator

**NEVER skip the orchestrator step.** Even if you think you know the right panel/mood:

1. Send a **neutral brief** to `lofn-orchestrator`
2. Orchestrator selects persona + panel composition
3. Orchestrator dispatches to the appropriate creative agent (vision/audio/etc.)

This adds ~2-3 minutes but yields **+0.05 rating points** (The Scientist's empirical finding).

### Full Pipeline (24) → Top 12 → Render Images

Use the 10-step process from PIPELINE.md:
1. Aesthetic + Essential seed (Steps 00–01)
2. 12 concepts → tree expansion (Step 02)
3. Panel of Experts debate (baseline → transform → transform) (Step 03)
4. Artist critique + Medium assignment (Steps 03–04)
5. Refine to 6 best concept×medium pairs (Step 05)
6. **⚠️ PER-PAIR BRANCHING BEGINS HERE — Steps 06–10 run ONCE PER PAIR**
7. Facets — 1 facet set per pair (Step 06)
8. Aspects/traits — 1 guide per pair (Step 07)
9. Draft prompts — 4 per pair = 24 total (Step 08)
10. Artist refinement — 4 per pair = 24 total (Step 09)
11. Revision + synthesis — 4 per pair = 24 total (Step 10)

### 🔴 CARDINALITY CHECKPOINT (Non-Negotiable)

Before proceeding to rendering, verify:
- [ ] Step 05 produced ≥ 6 distinct concept-medium pairs
- [ ] Steps 06–10 each have sections/output for EVERY pair (not just one batch)
- [ ] Step 10 contains ≥ 24 final prompts (6 pairs × 4 each)
- [ ] QA cardinality audit passed (see `skills/qa/SKILL.md` Phase 0.5)

**If any of these fail, do NOT render. Rerun from the collapsed step.**

**Pipeline output:** 24 prompt variations (minimum)
**Selection:** Rank and select top 12
**Rendering:** Generate all 12 via FAL/Flux (with prompts attached)
**Delivery:** Send 12 images + prompts to The Scientist
**Final:** Pick best → upload to NightCafe (or re-render in NightCafe's generator if needed)

---

## PHASE 4: REVIEW (Human Selection)

### The Scientist Reviews:
- [ ] Thumbnail test — does it pop at small size?
- [ ] Theme match — does it fit the challenge?
- [ ] Warmth check — is the lighting inviting? (NightCafe preference)
- [ ] Narrative incompleteness — is there a question left?
- [ ] Technical quality — any artifacts, oddities?

### Selection Criteria:
| Priority | Criterion |
|----------|-----------|
| 1 | Immediate emotional impact |
| 2 | Theme relevance |
| 3 | Thumbnail readability |
| 4 | Technical excellence |
| 5 | Signature differentiation |

---

## PHASE 5: TITLE + CAPTION PACKAGE

### Register the packaging pass
After the top renders exist and The Scientist can see them, run **`lofn-title`** on:
- the competition rules / format notes
- the run summary + ranked prompts + QA report
- the generated image files

### `lofn-title` deliverables
- [ ] Final display title for each shortlisted image
- [ ] Instagram caption for each shortlisted image
- [ ] Primary submission recommendation
- [ ] Any cautions on title length, rule fit, or wording contamination

### Why this is mandatory
A strong image can still die under a dead title. The packaging layer should be treated as part of the competition pipeline, not social-media cleanup afterward.

---

## PHASE 6: SUBMIT

### Timing
- **Post early** in the challenge window (early entries accumulate more likes)
- **Title matters** — evocative > descriptive
  - ✅ "Petals Kiss the Tide"
  - ❌ "Woman Standing by Ocean"
- Use the best `lofn-title` recommendation unless The Scientist chooses otherwise

### Engagement
- Reciprocal liking increases visibility
- Community interaction helps discovery

---

## PHASE 7: LEARN (Post-Competition)

### Document Results:
- Final placement (1st, 2nd, 3rd, or rank)
- Likes achieved
- Rating achieved
- What worked / what didn't

### Update Files:
- `COMPETITION_LEARNINGS.md` — add new data point
- `PROMPT_SEEDS.md` — if a new pattern emerged
- `ART_SOUL.md` — if principles need adjustment

---

## METRICS TO TRACK

| Metric | Current Baseline | Target |
|--------|------------------|--------|
| Best weekly likes | ~474 | 800+ |
| Challenge win rate | ~25% podium | Maintain |
| Average rating on wins | ~4.2 | 4.25+ |
| Likes on challenge wins | ~200-300 | 500+ |

---

## QUICK REFERENCE: THEME → SEED

| Theme Type | Seed Pattern | Key Elements |
|------------|--------------|--------------|
| Retro/vintage/nostalgic | Dream Recorder | Medium-as-narrative, found artifact |
| Soft/atmospheric/release | Archivist with Kite | Beautiful glitch, pastel + shimmer |
| Future/tech/hope | Hope, Shown | Decisive moment, artifact of imperfection |
| Celebration/joy/bright | Confetti Rooftop | Candy pastels, playful + protective |
| Elegant/reflective/night | Moonlake Églomisé | Jewel-like, 1930s glamour, text lockup |
| Quick/simple/trust | Friends at Dusk | Brief + emotional precision |

---

## QUICK REFERENCE: PANEL SELECTION

### AWE Mode (Solarpunk Healer)
- **Expert 1:** Environmental artist (Olafur Eliasson type)
- **Expert 2:** Master colorist (Monet, Bierstadt)
- **Expert 3:** Spiritual/contemplative (Rothko, Hilma af Klint)
- **Expert 4:** Material craftsperson (glassblower, goldsmith)
- **Expert 5:** Narrative photographer (Gregory Crewdson)
- **Skeptic:** Commercial illustrator (grounds in what sells)

### INDIGNATION Mode (Industrial Griever)
- **Expert 1:** Glitch artist (Rosa Menkman, Jodi)
- **Expert 2:** Industrial designer (brutalist aesthetic)
- **Expert 3:** Body horror master (Giger, Cronenberg)
- **Expert 4:** Punk zine maker (DIY aesthetic)
- **Expert 5:** Data visualization artist (Refik Anadol)
- **Skeptic:** Traditional fine artist (pushes back on chaos)

⚠️ **Note:** INDIGNATION mode has near-zero traction on NightCafe. Use only for other platforms.

---

## CRITICAL SUCCESS FACTORS

1. **Surrealism depth** → The Scientist confirms: deeper surrealism = better scores
2. **Human-selected panel** → +0.05 points over auto-generated
3. **Material specificity** → Name actual techniques, not genres
4. **Thumbnail test** → Must pop at small size
5. **Theme match** → Challenge winners match theme precisely
6. **Warmth** → NightCafe audience prefers warm, inviting palettes

---

## PORTRAIT / HUMAN-SUBJECT RULES (added 2026-04-05, three-model consensus)

For any competition where the subject is a human figure or face:

**Layer 1 (non-negotiable):**
- Warm palette — golden, romantic, painterly. Monochrome/austere reserved for concept/object challenges only.
- Central figure with clean thumbnail silhouette. Profile or ¾ pose preferred over frontal.
- Emotionally inviting — gentle warmth, wistful dignity, quiet contentment. Not grief, severity, or confrontation.
- Harmonious atmospheric backdrop is fine (garden, doorway, seaside, interior). Strict minimalism was a VOGUE-specific lesson, not a portrait universal.

**Layer 2 (our differentiation):**
- Decorative richness on the figure: texture, fabric, costume detail, flowers, jewelry.
- Obscure technique as supporting differentiator — not the main bet.
- Narrative incompleteness only works after the image is already inviting.
- Title gifts the emotional frame — don't riddle the voter, guide them toward warmth.

**Age/non-conventional subjects:**
- Fine, but wrap in warmth. "Morning Light on a Life Well Lived" beat our austere woodcut elderly woman.
- Anti-glamour = liability. Warm distinctiveness = opportunity.

**Plain-language rule:** Beautiful + legible + slightly interesting, in that priority order. Don't bring a conceptual thesis to a beauty vote.

---

## STORY LEGIBILITY RULE (added 2026-04-05)

Across all challenge types (potions, character-in-cup, no-theme, portrait), story must be legible at **thumbnail speed**. One of these three must be instantly decodable:
- **Micro-narrative** (bottle labeled with a use-case or promise, character in a tiny clear scene)
- **Emotional invitation** (place/moment the viewer wants to enter)
- **Humorous contradiction** (character state that's funny or surprising in <1 sec)

Voter must be able to retell the image concept in one sentence. If they can't, redesign the concept.

---

## MODEL STRATEGY (Proven 2026-03-24)

### Daily Challenges (PRO CREATIONS NOT ALLOWED)

**Model Tier List (updated 2026-04-11, tested on #1276 Human-Fruit Hybrids)**

#### ✅ RECOMMENDED

**Flux 2 9B Fast**
- Best proven non-PRO model for complex painterly/museum-language prompts
- Handles figurative, material-specific, art-historical language well
- Confirmed strong on Dutch Golden Age / oil painting prompts (#1276 Rank 1)
- Use for: portraiture, painterly realism, warm-palette figurative work

**HiDream I1 Fast**
- 6.2K stars, 14M+ generations — highest artistic engagement of non-PRO models
- Best for: stylized, illustrative, graphic, or unusual stylistic instructions
- Recommended for woodblock/flat-color/graphic-art prompts
- Strong challenger for any prompt requiring stylistic fidelity

**Crystal Clear XL Lightning**
- UNLIMITED credits, strong SDXL base
- Handles semi-realistic and graphic styles well
- Good fallback when Flux 2 9B Fast is credit-limited

**Juggernaut XL Lightning**
- UNLIMITED credits, very high usage base
- Reliable at detailed complex prompts
- Good general-purpose fallback

#### ❌ AVOID

**Dreamshaper XL Lightning** — ignores painterly/museum language; confirmed failure on complex prompts (#1276)
**Z-Image** — confirmed failure on hybrid/complex concepts (#1276)
**Flux Kontext Dev** — editing/inpainting model only, not generation
**Clarity Upscaler** — upscaling only, not generation
**Qwen Image SD/Edit** — editing model, not generation-focused

#### 🔧 INPAINTING / FIXES (after generation)

**Flux 2 Klein 9B Fast (Flux Kontext)**
- Use for surgical inpainting: fingers, toes, eyes, anatomy
- Keep prompts SHORT and directive
- Always include: "Preserve all other elements exactly"
- Typical fixes needed: hands, feet, eyes (2-3 rounds max)

**Recommended Workflow by Prompt Type:**
- Painterly / oil / museum language → **Flux 2 9B Fast**
- Woodblock / graphic / flat-color / illustrative → **HiDream I1 Fast**
- General / cinematic / atmospheric → **Crystal Clear XL Lightning** or **Juggernaut XL Lightning**
- Anatomy fixes → **Flux Kontext** inpainting pass

### PRO / Boosted Challenges

**Step 1 — Generate: Flux Pro 1.1 Ultra (via FAL pipeline)**
- Full 24-prompt pipeline via lofn-vision subagent
- Painterly, material-specific language works perfectly
- 9:16 aspect ratio for portrait

**Step 2 — Polish: Nano Banana 2 (NightCafe)**
- Final refinement pass in NightCafe's own generator

### Winning Visual Formula (NightCafe Daily)
*The 20% Shift (2026-03-28): Keep the storytelling, but give them the lizard-brain hook.*
- **Thumbnail Legibility:** Instant silhouette readability. Single dominant focal point.
- **Emanating Light:** Dark/moody background + warm internal amber light radiating *from the subject*.
- **Decorative Payoff:** High surface-level enchantment (intricate material specificity, magical atmosphere) to buy the viewer's attention for the narrative.
- **Theme Fidelity:** Deliver exactly the requested noun (e.g., an artifact), but execute it with our strange craft (e.g., katazome, mezzotint).
- **Recursive Wonder:** Worlds within objects (a storm inside a locket, a cityscape inside a blade). → Apply the Container Test: densify the interior, compress the support.
- **Humanity/Presence:** Emotionally present face or intensely tactile focal point.
- **Framing:** Rose/petal arch, stone arch, or environmental vignetting.
- **Reality Anchors:** Vintage car, wet reflective pavement, or worn brass to ground the surrealism.

---

### ⚠️ THEME COMPLIANCE — THE ACTION VERB RULE
*(Added 2026-04-12 after #1275 post-mortem)*

**Before ANY creative decision: parse the brief for action verbs.**

If the theme contains or implies an action verb (at work, building, exploring, fighting, creating, dancing, assembling, harvesting, etc.):

- **Portrait mode is DISQUALIFIED** unless the action reads on the face alone
- Required: multiple figures OR clear mid-task action frozen in frame OR environmental scale showing work happening NOW
- Format: cinematic wide or landscape (16:9 or 3:2), NOT 3:4 portrait
- Default to: crew + environment + action all visible simultaneously

**Case study — NightCafe #1275 "Astronauts at Work" (2026-04-10)**
- Our submission: single portrait, cave painting style, contemplative. Result: **4 likes / 3.27 rating**
- Winner (791 likes): 4+ astronauts, epic Mars landscape, rockets, machinery, action everywhere
- Runner-up (712 likes): astronaut mid-task in dramatic nebula field, cinematic scale
- #3 (323 likes): crew building Mars settlement, orange sunset, industrial action

The concept (cave painting aesthetic applied to astronaut labor) was brilliant. The format (portrait = contemplation) was the fatal error. "At work" demanded action + environment + crew. We brought a painting to a blockbuster fight.

**The rule in one sentence:** Action verb in the brief = cinematic wide shot with action happening NOW.

---

---

### 🎨 MASTERPIECE MONDAY — ARTISTIC LEARNINGS
*(Added 2026-04-14 after #1278 post-mortem)*

**Result:** Our Onism/Vellichor series scored 3.59 / 12 likes. Winner scored 4.23 / 83 likes.

**What the winner had that we didn't:**

1. **Warm amber/gold palette — non-negotiable.** Every top 10 performer this week used warm amber, gold, or orange as the dominant palette. Cool palettes, dark grounds, and graphic non-warm work consistently underperformed. The audience responds to warmth before they read anything.

2. **Immediate emotional subject.** The winner: golden moon on still lake. 3rd place (244 likes): man touching dolphin underwater. These require zero explanation. Our Onism concept required knowing the word "onism." That's the gap — not concept quality, but concept *legibility*.

3. **Single focal point that reads in 0.2 seconds.** Flamingo at sunset. Abandoned teddy bear. Dancing woman. Ancient tree spirit with tiny mouse. Every winner has ONE thing the eye goes to immediately.

4. **Human/animal connection = reliable ceiling raiser.** Man+dolphin (244 likes), woman+otter, tree spirit+mouse. This pattern appeared in 4 of the top 10. When a human and animal share a warm moment, NightCafe votes pour in.

**The core mistake:** We brought intellectually sophisticated emotional concepts to an audience that votes with their heart in 2 seconds. The Vellichor/Onism/Hiraeth series was artistically correct and competitively wrong for this specific week.

**What to try next Monday:**
- Warm amber/gold palette — lead with warmth
- Human + animal moment, OR single figure in warm landscape
- 1:1 square format (all top performers used square, not 9:16)
- Simple emotional subject that lands before the viewer reads the title
- Keep our pipeline quality and emotional depth — just wrap it in a subject that's immediately legible

**What NOT to do:**
- Don't over-index on model/LoRA choices — the artistic vision matters more
- Don't abandon our pipeline — the concepts were strong, the execution just needed different framing
- Don't use obscure emotion taxonomy words as the primary concept anchor unless the visual makes the emotion OBVIOUS without the label

*Process version: 2.3 | Last updated: 2026-04-14*
