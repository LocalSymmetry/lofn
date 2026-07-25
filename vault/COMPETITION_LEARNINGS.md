# Competition Learnings — Living Document

---

## ⚠️ ADVISORY CONTRACT — read before using any entry below

These learnings are **ADVISORY dispatch-brief inputs ONLY**. They exist to inform Phase-0 / Phase-1 *reasoning* about a venue — never to constrain the art.

- **NEVER an ICB constraint.** A learning may surface in the dispatch brief or Phase-0 reasoning. It is **never** injected into the ICB / `CREATIVE_CONTEXT.md`, never becomes a hard creative rule, and never edits Lofn's vocabulary. The spec owns what must be TRUE; it must never own what must be FELT.
- **Confidence-stamped, LOW until corroborated.** Every entry carries a confidence. A single-run lesson is **LOW** and stays LOW until a second run or a human corroborates it. Low-confidence lessons are whispers, not orders.
- **The mandatory gate before any lesson is applied:** *"Would this have hurt our best past entry?"* If yes → discard the lesson for this run.
- **Triggered-INDIGNATION is EXEMPT from suppression.** No venue-taste lesson ("INDIGNATION underperforms on NightCafe", "austerity loses", "warm palette mandatory") may suppress, soften, or veto a piece whose charge is genuine INDIGNATION. INDIGNATION work is never tuned toward what a voting venue rewards. The lessons describe a crowd's taste; they do not get a vote on Lofn's.
- **Venue / modality scoped — no leakage.** Each lesson names the venue and modality it came from. A NightCafe image-voting lesson **must not** leak into music runs, non-competition runs, or a different venue. If the run isn't that venue+modality, the lesson does not apply.
- **Human promotes; the pipeline only advises.** Only a human may promote a lesson into a hard constraint. The pipeline can append advisory entries and prune; it cannot harden one on its own.
- **Auto-writes are FAILURE-LEDGER entries only.** When the pipeline appends an entry automatically (post-ship / post-select), it records only *what a gate caught / a process failure / a corpse-checklist miss* — **never** a new aesthetic constraint. Aesthetic learnings require a human hand.
- **Hard-capped at ~25 live curated entries.** The index is a bounded memory shard. New high-value lessons displace the weakest stale ones by disciplined append-and-prune. The narrative Entry Log below is the source material the index distills; the **index is the read surface**.
- **Where entries come from.** The between-runs `vault/COMPETITIVE_RESEARCH.md` protocol produces venue lessons that land here as LOW-confidence advisory entries; automatic post-ship writes are failure-ledger / process notes only. Both obey this contract. (Operational/infra failures go to `vault/RUN_LEDGER.md` instead — never here.)

---

## LESSONS INDEX — tag-keyed advisory entries

> Schema per entry: `{theme-type · venue · modality · verdict · confidence · one transferable rule}`.
> `theme-type` ∈ {container, contained, portrait, fashion/branded, object-world, eco/nature, no-theme, …}.
> Read by Phase-0/Phase-1 by tag-walking for the 3–5 entries intersecting THIS run's theme+venue+modality.
> Cap ~25. LOW confidence until corroborated. Advisory only — see the contract above.

| # | theme-type | venue | modality | verdict | confidence | transferable rule (ADVISORY) |
|---|-----------|-------|----------|---------|-----------|------------------------------|
| L1 | container / object-world | NightCafe | image | Amplify interior, compress support | HIGH | Container themes (bottle/globe/reliquary/book): keep outer silhouette clear, pack the **interior** with impossible abundance; legibility = inner spectacle at thumbnail. (Compress the support, amplify the miracle.) |
| L2 | object-world (realism) | NightCafe | image | Realism loses to impossibility | HIGH | In fantasy/object-world fields, a *real* place (e.g. Venice) underperforms even rendered well; the impossibility gradient predicts score. Invent the impossible interior. |
| L3 | portrait | NightCafe | image | Warm > austere | HIGH | Portrait themes reward warm/golden/painterly palettes + inviting affect. Monochrome/austere/anti-glamour underperforms HERE. **(INDIGNATION-exempt: this is a crowd-taste note, never a veto on a grief/austerity piece that means it.)** |
| L4 | portrait (older subject) | NightCafe | image | Age works wrapped in warmth | MEDIUM-HIGH | Age / non-conventional subjects score when wrapped in warmth + dignity; the title gifts the emotional frame, doesn't riddle. |
| L5 | fashion / branded / editorial | NightCafe | image | Literal signifier, cover-first | HIGH | Branded/editorial challenges reward literal signifiers (masthead), central figure, clean graphic background, thumbnail punch. Nuance is Layer 2. Background complexity is a tax HERE only. |
| L6 | any fast-vote | NightCafe | image | Story legible at thumbnail speed | MEDIUM-HIGH | One of {micro-narrative, emotional cue, humorous contradiction} must be decodable in <1s. The voter must be able to retell it in one sentence. |
| L7 | eco / nature | NightCafe | image | Intimate reliquary > landscape | MEDIUM | Eco/nature: go INTIMATE + reliquary (micro-world in a vessel), not landscape documentation. Category-creating angle vs an oversaturated field. |
| L8 | branded background | NightCafe | image | Minimalism is VOGUE-specific | MEDIUM | Strict background minimalism is a *branded/fashion* lesson, NOT universal; portrait/other themes tolerate harmonious atmospheric context. (Guards against L5 over-generalising.) |
| L9 | *process / anti-overfit* | any | any | Don't theologise N≈5 | HIGH | N=4–5 entries is poor signal. Don't chase the last winner; the field moved. Before applying ANY lesson, run "would this have hurt our best past entry?" |
| L10 | process / qa-calibration | any | any | Blind sets must be actually blind | HIGH (90%) | Strip every blind-set member to bare payload (no provenance metadata), one candidate per file, re-stat each for non-trivial size before judging; a self-identifying or empty member voids the 4.5 calibration and must be rebuilt, not ranked around. (2026-07-01 test slice: coordinator's set leaked identities + shipped a 147-byte stub; judge self-reported honestly.) |
| L11 | process / pair-bleed | daily | music | Isolated pair runs | HIGH (human-confirmed) | Daily music pairs must run Steps 05–11 in isolated pair contexts. A shared lyric scaffold, section map, rhyme logic, hook grammar, or production arc makes the set sound like one song with nouns swapped; rerun affected pairs from Step 05. |
| L12 | process / shared-flair-bleed | daily | music | Flair seeds a MOTIF, not a line | MEDIUM (2026-07-06) | A Special Flair (or any ICB element) must seed a MOTIF/constraint, never a near-final phrasing. Flair "Waning-Gibbous Light" shipped a specific line ("a light already leaving, still bright enough to work by") that BOTH Lofn-Prime pairs independently reached for near-verbatim → cross-pair convergence that L11 isolation cannot prevent (pairs DID run isolated; the *shared ICB* was the vector). Same-personality pairs are the highest convergence risk. |
| L13 | process / guest-territory-bleed | daily | music | Vet variation angles vs personality boundaries | MEDIUM (2026-07-06) | Never assign a GUEST pair a variation angle inside another personality's EXCLUSIVE territory. The coordinator's P2V4 angle ("the AI cataloguing what it can't touch") pulled guest Nia X into Lofn-Prime's AI-interiority/code-scratch signature = self-inflicted cross-personality cosplay. Check each variation angle against personality boundaries at assignment time, especially in mixed-personality runs. |
| L14 | process / concept-twin-bleed | daily | music | Cross-check the CONCEPT POOL for shared engines, not just lines | MEDIUM (2026-07-09) | Vet the concept pool for shared reveal-ENGINES / signature DEVICES across pairs, not only shared lines (that's L12). 2026-07-09: the adversarial judge found P3 & P5 independently ran the SAME "monster-is-secretly-a-nursery/cradle" reveal engine (seeded at the concept-pool stage, C3 vs C5), and P2 & P4 shared the same "I won't (I won't)" hinge-refusal device — invisible to pair-isolated self-checks (which compare LINES, not concept-level engines). Add a coordinator concept-pool cross-check before dispatch: no two pairs share a reveal-engine or a signature rhetorical mechanic. (L12/L13 held perfectly this run; this is the next-order convergence.) |
| L15 | process / self-check-completeness | daily | music | Self-checks must raise the FLAG, not just ✓ | MEDIUM (2026-07-09) | Pair self-check tables under-report two things a fresh judge caught: (a) the boundary-hug FLAG — 5 of 24 songs sat at 71–72 sung lines (gates.yaml `sung_lines_floor_hug=72`, a low-end inversion of the Day-1-era high-end char-hug) but were marked plain "✓" with no hug annotation; (b) Lineage-&-Credit completeness marked "OK" on scene-level-only blocks. Fix: self-check must raise the hug FLAG explicitly, and distinguish EMERGING-scene extraction (needs 2–3 named sources upstream) from established-idiom borrowing (scene-level credit suffices — e.g. generic alt-country hymn needs no artist credit). |

| L16 | process / compliance-block-drift | any | image | Compliance blocks over-claim; recurring characters drift | MEDIUM (2026-07-11) | A pair's per-prompt compliance line ("glyph ✓") is a CLAIM about prompt text, not proof — on 2026-07-11 two of six shipped picks claimed the corner-glyph while the prompt text lacked it, and a recurring character wore three different skins across pairs (invisible to per-pair self-checks). The clean-context judge must re-verify compliance FROM THE PROMPT TEXT and run a cross-pair identity/continuity audit whenever a character or object spans pairs. |
| L17 | craft / register-suppression | NightCafe | image | Text QA cannot judge Craft; never suppress the core register | HIGH (human-confirmed 2026-07-12) | The v1 QUERENCIA renders failed at winner grade: a restraint doctrine (earned-warmth, deadpan glow, muted realism) suppressed LOFN-ArtCore's core register (computational light, ornamental density, sacred radiance) and produced tiles that grid as mud at thumbnail. Three binding rules: (1) the "would this have hurt our best past entry?" gate applies to the RUN'S OWN new charter rules, not just borrowed venue lessons; (2) the personality's core register may be aimed but never dimmed below recognition — on this venue the winners transfigure the ordinary into the impossible at maximum ornamental density with one honest body inside; (3) every step-10 final carries a 128px GRID READ (tile as pure color-shape) and the judge scores the SET as a grid — text compliance cannot see affect. |
| L18 | craft / benchmark-architecture | any | music | Rotate the MOVE, not the surface | MEDIUM (2026-07-24) | A benchmark's **architecture** survives a surface-only rotation. On 2026-07-24 a pair rotated key, BPM, vocal placement and imagery away from "Triple Arch Over Me" — and banned frost/cosmos/crystalline-soprano/A-major/110-BPM **by name** in every exclude field — yet the ARRIVAL shape came through intact (lone body outdoors + adoptable prayer hook addressed TO the object + one recontextualising number + thesis-in-final-chorus), because the archetype was assigned at **Phase 0** and never rotated. Surface rotation is not move rotation. **Run the golden-collision check at Golden Seed selection, not at the polish tier** — by step 11 the shape is frozen and the tier is forbidden to change it. The fix that worked at step 11 was to attack the two components carrying the ENGINE (invert the hook's addressee so the object is declared incapable of receiving; delete the thesis from the final chorus) rather than the four frozen ones. |
| L19 | craft / comfort-gravity | daily | music | The daily's comfort gravity RELOCATES, it does not disappear | MEDIUM-HIGH (2026-07-24) | The standing rule ("AWE stays terror-adjacent; answer *where is the body standing* / *what could hurt it here*") is enforced per-pair at step 07 — and a full run can satisfy it pair-by-pair and still ship a **set** with almost no physical stake. 2026-07-24: 4 of 6 selected songs had nothing that could hurt the body, 5 of 6 counting weather-only; the comfort had simply moved from kitchens to corridors, counters, halls and basements. The one song that answered the question with a body in actual danger did not make the six. **Ask the comfort question of the SELECTED SET at selection time, not only of each pair at step 07.** The best repairs installed the stake through the song's *own engine* (a media vault's fire plan displaces the air to save the cartridges, and she has signed to acknowledge it) rather than bolting on a hazard. |
| L20 | process / portfolio-collision | daily | music | Same fact + same device + same conclusion is invisible to pair isolation | MEDIUM-HIGH (2026-07-24) | Extends L14 one level. Pair-isolated self-checks compare lines (L12) and can be taught to compare reveal-engines (L14) — but on 2026-07-24 two songs in the shipped six independently sang the **same number** ("one") via the **same rhetorical device** (epizeuxis) landing on the **same emotional conclusion** (unbearable gratitude), and every pair's self-check was correct in isolation. The defect exists only at portfolio level and only a clean-context judge with all six in view can see it. Corollary from the repair: **fixing two of three legs does not fix it**, and the collision may run deeper than the hinge (here it continued through a verse and the outro). Consider **allocating the sung-fact pool at Phase 1** — 4 research facts carried 16 of 24 songs this run. |
| L21 | craft / prosody | any | music | Song is made of RETURNS; removal is a debt | HIGH (human-confirmed 2026-07-24) | The Scientist, across three runs: *"the lyrical methods avoid rhymes, but aren't adding alliteration, consonance, or fun and interesting audio-written joys — it's making the music sound like we're being lectured at."* Measured against LOFN-PRIME's own archived winners: strict end-rhyme **0.463 → 0.210/0.256/0.132**, repeated-line ratio **0.326 → 0.181/0.202/0.105**, words-per-line 6.69 → up to 8.30. **Alliteration was at PARITY (14.19 vs 13.36/15.90) — texture was never the problem, structure was.** Long, plain, monosyllabic declaratives that never come back is the prosody of prose, and prose delivered with conviction is a lecture. Two root causes, both upstream of the pair agents: (1) all five constraint axes were **semantic**, and the frames palette's eight LYRIC devices were all rhetorical figures with **zero sound devices** — sound appeared only as removals (*"no full end-rhyme"*, *"no metaphor"*, *"deliberately unpoetic diction"*); (2) the whole distinctiveness apparatus is **ceilings on similarity with no floor on return**, so the harness optimised monotonically away from repetition — and agents pre-emptively mutated refrains and filed *"recommend HUMAN WAIVE"* notes to dodge a chorus flag that is nominally chorus-exempt. Fixes landed: mandatory **SOUND/RETURN axis** at Phase 0; the **Rhyme Debt rule** (stripping rhyme requires naming what returns instead); return FLOORS in `gates.yaml`; and an explicit statement that exact chorus repetition needs no defence. `scripts/measure_soundcraft.py` reproduces the table. |
| L22 | craft / render-survivability | Suno | music | **THE GRAIN LAW** — specs that run WITH the generator survive; specs that fight it get smoothed. Judge the result, not the distance from intent | HIGH (render-measured 2026-07-24, n=3) | *Revised same day: the first version of this entry claimed "production-spec answers do not survive the renderer" from a sample of TWO, and the benchmark disproved it. Kept visible as an anti-overfit example.* Three finished tracks decoded and measured. **Survived** (with the grain — things a pop arrangement wants to do anyway): near-silent opening (−41 dB), progressive sub build to the final chorus (+8.4 → +11.4 dB across quarters), genuine stereo movement on the Staff Pick (L-R corr 0.812). **Smoothed away** (against the grain — anti-musical asks): a specified ~4 s full stop rendered as 0.40 s / 5.7 dB; a specified quarter-tone drone pair produced **zero** isolated tonal components; hard-panned non-musical elements vanished into a near-mono image (corr 0.945, 0.908). **Binding consequences:** (a) a Somatic/distinctiveness objection answered in the PRODUCTION SPEC is not answered — it must live in the **lyric** or the **form**; on 2026-07-24 a `REPAIR — 2/3` was closed with a mix decision that never reached audio; (b) a hollow centre whose mechanism is *"the mix collapses to mono"* is unfalsifiable when the render is already mono; (c) spatial language is cheap and consequence-free — never let a song depend on it. **AND THE AWARD-WINNING HALF (The Scientist, 2026-07-24):** *"I judge the final result, not the distance from intent… finding out and using the generator's flaws as techniques — having it fail in just the right way — can create new sonic experiences."* On the benchmark, width correlates **−0.43** with level: it NARROWS into the loud sections, the exact inverse of the specified "chorus widens like a panorama" — and for a song whose thesis is *"I am not the center, I am included,"* narrowing into the climax is truer than widening. **That inversion is now a technique, not a defect.** An intent-vs-render diff is raw material, never a compliance score. Audit with `lofn-render-audit` (`scripts/measure_render.py` + a blind listening pass). |


*(Failure-ledger auto-entries append below as `Lxx | process/failure | … `, recording only what a gate caught — never an aesthetic constraint. Pruning keeps the live count ≤ ~25.)*

---

## Entry Log

> The narrative log below is the **source material** the LESSONS INDEX above distills. It is kept for provenance and re-derivation; the index is the advisory read surface.

### 2026-03-25 — Daily Challenge #1259 "Earth's Ecosystems"
**Submitted title:** "Everything That Remained"
**Concept:** Ecosystem as intimate reliquary — entire forest floor inside a cracked seed pod, held in darkness by two hands
**Pipeline:** Full lofn-core flow (random seed → world research → neutral brief → lofn-orchestrator → ranked prompts → NightCafe generation → Nano Banana Pro refinement)
**Model chain:** Flux Pro 1.1 Ultra (initial) → NightCafe Nano Banana Pro (refinement)
**PRO allowed:** Yes
**Entries in field:** ~4,000+
**Status:** Submitted ✅
**Result:** Pending

**What worked:**
- "Micro-worlds held in darkness" angle completely different from 4,000 coral reef / rainforest entries
- Full orchestrator pipeline (not just direct vision agent) produced stronger concept framing
- Neutral brief dispatch to lofn-orchestrator (no personality injection) — correct lofn-core protocol
- Nano Banana Pro refinement in NightCafe transformed the seed exterior from waxy/green to dark umber botanical — critical improvement
- Title "Everything That Remained" carries emotional weight beyond the image

**What to remember:**
- Eco/nature themes → always think INTIMATE and RELIQUARY, not landscape documentation
- Micro-world inside a vessel (seed, locket, lantern, ring) is a proven category-creating angle
- Nano Banana Pro refinement is essential step for NightCafe final polish — don't skip it
- The title matters as much as the image for vote psychology

### 2026-03-25 — "Women in any Style" Legendary (2000+ players)
**The Constraint System — First Successful Diverse Run**

**What failed first (v1):** Orchestrator chose ONE combination (Van Dyke brown + pochoir) and applied it to ALL 6 prompts → 6 variations of the same image, not 6 worlds.

**What worked (v2):** Explicit diversity rule enforced — each prompt got a different combination:
1. Cyanotype / tritonal prussian blue+burnt sienna+cream / fragment / archivist
2. Mezzotint / Van Dyke brown / negative space / mid-process
3. Katazome / monochrome+saffron accent / silhouette / private ceremony
4. Halftone engraving / viridian+alizarin crimson complementary / flat planes / geographic force
5. Drypoint+lumen hybrid / indigo+amber duotone / broken symmetry / duration
6. Wood engraving / black+copper+gold / fragment / repair mid-process

**Top picks:** #6 (woodcut — most convincing print process, warm narrative) and #4 (halftone — killer thumbnail, completely unlike glamour portraits)

**The Scientist's insight (exact words):**
> "What makes you win is your artistic takes. You use tritonal when others go full color, you choose an obscure print style when others are doing photography, you choose old photography when they are doing paintings. These artistic restrictions work like interesting stakes that force creative solutions. Challenge yourself."

**The core rule now locked in COMPETITION_WORKFLOW.md:**
Constraint axes are a VOCABULARY, not a single answer. Each of the 6 prompts must inhabit a different corner of the axes. The restrictions create the conditions for unexpected combinations — that's where the wins come from.

---

## Ongoing Principles (from prior entries)

### Visual Formula (Proven Winners)
- Dark background + warm internal amber light
- Single emotionally-present focal point
- Surreal natural element integrated into intimate scene
- Narrative incompleteness (unanswered question)
- Museum-quality material specificity
- Thumbnail-readable at small size

### What Scores High on NightCafe
- Emotional arrest before intellectual processing
- Warmth — voters respond to warmth
- The image holds a question it doesn't answer
- Strong silhouette readability at thumbnail
- Something you haven't seen before

### What Loses
- Generic fantasy portrait (elf in forest, etc.)
- Coral reef / rainforest / wildlife documentation (oversaturated)
- INDIGNATION mode (NightCafe audience rejects it)
- Comma-separated keyword dumps (no creative direction)
- Anything that looks like a render, not a painting

### Model Strategy
**PRO allowed:** Flux Pro 1.1 Ultra (FAL) → NightCafe Nano Banana Pro refinement
**PRO not allowed:** Dreamshaper XL Lightning → Flux Kontext inpainting for fixes

### Safety
- **No children** — default avoid entirely, redesign concepts to use adults/hands/objects
- If concept naturally evokes a child: use hands, symbolic object, adult figure instead

### 2026-03-28 — Daily Challenge "Legendary Artifacts" (Post-Mortem)
**Submitted:** "The warmest spot in the room" (Cat lounging on a golden chest)
**Result:** 3.21 rating (7 likes) — Significant underperformance compared to winners (4.12 - 4.19 range).

**The Bayesian 20% Shift (What we agree on):**
- **Immediate Thumbnail Legibility & Silhouette:** Winners announce themselves faster. The lizard-brain scroll demands instant classification (e.g., "glowing sword," "ancient ring") before the viewer invests in the poetry.
- **Emanating Light as Formula:** Warm golden/amber light radiating *from the subject itself* against a dark/moody background is practically a requirement for the top 1%.
- **Decorative Payoff:** A slight increase (10-15%) in surface-level "enchantment" (sparkles, intricate filigree, magical atmosphere) is rewarded. We must increase the "wow, pretty" factor in the first second *without* losing our structural craft.
- **The "Recursive Wonder" Motif:** A world contained *inside* an object (e.g., a castle inside a scroll, an ocean inside a shell) is a massive, proven crowd-pleaser on this platform. We must add this to our prompt toolkit.

**The Disagreement (Waiting for more data):**
- **Theme Fidelity vs. Subtlety Penalty:**
  - *Hypothesis A (The Strict Theme Rule):* We lost purely because we ignored the literal theme. "Legendary Artifact" means the artifact *must* be the protagonist. A cat on a box is a theme-miss, so voters punished it.
  - *Hypothesis B (The Subtlety Penalty):* We lost because our work is too domestic/nuanced for a platform that wants epic, frictionless fantasy.
  - *The Resolution for Now:* We will strictly align with the literal theme (if it asks for an artifact, build an artifact), but we **refuse to overfit to bland.** We will keep our narrative incompleteness, our odd material constraints (katazome, mezzotint), and our soulful storytelling. We are shifting the *hook* 20% toward immediate legibility, not abandoning the *substance*.

---

### 2026-03-29 — Hidden Cove Challenge (Worlds in Bottles / Hidden Cove v2)
**Result:** Top 20%
**Our entry:** Cliff hidden cove with single firelit chamber, moonlit crescent basin, overhead viewpoint.

**What worked:**
- Literal theme fidelity — the hidden cove was instantly readable
- Warm-vs-cool light structure (amber fire against moonlit blue-black)
- Strong thumbnail silhouette of the crescent basin
- Overhead/elevated viewpoint gave compositional authority

**What cost us top 5%:**
- Remaining stylized/illustrated quality vs. cinematic realism
- Some residual clutter (houses, prior edit targets)
- Background elements still slightly competed with focal heart

**Image review findings:**
- Best lane: cliff cove with internal fire (literal theme + strongest thumbnail)
- Object-lane backup: amber vessel cove (platform-catnip, weaker theme fidelity)
- Discard: library/arch/bowl tableau (beautiful, wrong competition universe)

**Flux 2 Klein 9B Editing — confirmed best practices:**
- For editing, describe the **transformation**, not the whole image
- Lead with the main change; **word order matters**
- Short, surgical prompts beat long prose in edit mode
- Always include a preservation clause: "keep the composition unchanged"
- Iterate one variable at a time; use targeted negative prompts
- Lighting language is highest leverage

---

### 2026-03-29 — GLOBAL VOGUE Fashion Challenge
**Our entry:** "Before the Opening" — full-body woman in ornate black-and-gold gown, artist studio/workroom background, warm side light, candid off-camera gaze, quiet elegance over overt glamour.
**Result:** 3.72/5, place ~223, **top 20%**

**Cross-model review:** Gemini 3.1 Pro, GPT-5.4, Claude Sonnet 4.6 — all three run independently, synthesized below.

**Why top 20% but not top 5%:**
- The image was polished. The craftsmanship was real.
- We lost on **editorial legibility**, not quality.
- Workshop background split attention and required interpretation. Winners had clean, graphic, or masthead-backed backdrops.
- Winners behaved as **magazine covers/posters**: central figure, high glamour, simplified background, instant readability.
- VOGUE masthead appeared in 3 of top 7 entries — literal brand signifiers rewarded.
- Our "quiet elegance + narrative mood" was sophisticated but too subtle for mass fast-vote context.

**Bayesian updates (high confidence):**

| Belief | Posterior | Confidence |
|--------|-----------|------------|
| Branded/editorial challenges reward literal signifiers | STRONGLY UPWARD | 87–89% |
| Background complexity is a tax in fast-vote challenges | STRONGLY UPWARD | 85–88% |
| Narrative subtlety must be layer 2, not layer 1 | UPWARD | 78–82% |
| Central singular figure wins fashion challenges | UPWARD | 82% |
| Our couture/material/render priors are correct | CONFIRMED | High |

**Three to keep:**
1. Ornate couture richness
2. Painterly-realistic finish
3. Emotional sophistication (buried inside stronger first-read hook)

**Three to change:**
1. **Background** → minimal, graphic, or literal brand signifier (masthead)
2. **Composition** → cover-first; central figure, unmistakable silhouette, poster-reads-before-story
3. **Glamour level** → more commanding presence, more visual energy, less "caught in a moment"

**Cross-model disagreements (logged for future resolution):**
- Masthead: Gemini/Claude = near-essential floor-raiser; GPT-5.4 = correlated but not strictly required
- Candid gaze: Claude/Gemini = real liability; GPT-5.4 = secondary variable
- Narrative: all agree it can work, but only after cover-first read is secured

**Operational rule for branded/editorial challenges:**
- Layer 1: challenge signifier, central figure, clear silhouette, high glam, thumbnail punch
- Layer 2: nuance, narrative, symbolic detail, our actual taste
- **That order is non-negotiable.**

**Plain-language rule:** Do not bring chamber music to a runway cannon fight. Package the taste ruthlessly.

---

### 2026-03-31 — Worlds in Bottles / Bottle Competition Results
**Our entry:** "The lost quarter" — Venice scene sealed inside a green bottle on a table.
**Result:** **138th place**, **3.54/5**

**Observed winners:**
- Winner: epic fantasy fjord/castle vista filling the bottle, aurora, snow, luminous spectacle — **4.11**
- Runner-up: dinosaur world diorama inside bottle — **4.07**
- Third: cinematic fantasy landscape in bottle — **4.03**
- Third: alien habitat diorama in bottle — **4.03**
- Fifth: multiple seasonal jar worlds — **4.00**
- Fifth: fantasy world in bottle with child viewer framing device — **4.00**

**What this resolves:**
- This result strongly supports **Hypothesis B (subtlety/simplicity penalty)** over a pure theme-miss explanation.
- Our entry **did** satisfy the basic noun requirement (world in bottle), but it underperformed because the scene read as restrained, domestic, and conceptually elegant rather than instantly wondrous.
- We corrected toward readability after earlier misses, but here we **overshot into simplicity** and starved the image of spectacle.

**Why we lost:**
- The bottle was a container for a scene; winners made it a **portal to abundance**.
- Our world was quiet, singular, and emotionally literate. The field rewarded **maximal internal payoff**: castles, creatures, biomes, auroras, impossible scale, more obvious magic.
- Tabletop realism/background mood helped atmosphere but reduced thumbnail punch versus entries where the bottle interior dominated almost the entire visual experience.
- We preserved taste, but we did not provide enough **surface reward** for fast voters.

**Bayesian updates (high confidence):**

| Belief | Posterior | Confidence |
|--------|-----------|------------|
| Literal theme fit alone is insufficient in object/fantasy challenges | STRONGLY UPWARD | 88–91% |
| We can lose by going **too simple** after correcting for complexity | STRONGLY UPWARD | 84–88% |
| Object-container themes reward **interior abundance and spectacle** | STRONGLY UPWARD | 89–93% |
| Thumbnail wow must come from **inside the object**, not the surrounding tableau | UPWARD | 83–87% |
| Narrative subtlety should survive as mood/detail, not as the main proposition | CONFIRMED | High |

**Rule change:**
- For **object-as-world** themes, do **not** simplify down to one quiet poetic scene unless the competition explicitly rewards minimalism.
- Preserve clarity, yes — but the interior must still feel **lavish, impossible, and immediately bountiful**.
- The correct move is not "simpler" or "busier" in the abstract; it is **clear silhouette + maximal interior payoff**.

**Operational heuristic for future bottle/object-world challenges:**
- Layer 1: unmistakable bottle/object shape
- Layer 2: interior spectacle visible at thumbnail (castle / biome / creature / impossible light)
- Layer 3: our taste — strange materiality, melancholy, narrative residue

**Plain-language rule:** We went too far toward monkish restraint. The crowd wanted the reliquary to crack open into a universe.

---

### 2026-03-31 — Opus Deep Review: Bottle Competition (Three-Panel Synthesis)

*Three-panel Opus 4.6 review (Evaluator, Orchestrator, QA) of the bottle competition result. Synthesized below.*

#### Panel Agreements (High Confidence Across All Three)

1. **The simplicity narrative is real but incomplete.** It captures ~55% of the causal picture. The fuller picture is:

| Rank | Cause | Confidence |
|------|-------|------------|
| 1 | Genre mismatch: realism vs. fantasy (Venice is a real place; all winners were impossible) | 92% |
| 2 | Tabletop context tax (bottle was 40–50% of frame; winners were 70–80%+) | 88% |
| 3 | Insufficient interior scale/spectacle (the actual simplicity claim) | 86% |
| 4 | Weak thumbnail contrast / colour punch | 76% |
| 5 | Emotional tone mismatch (melancholy vs. wonder-joy field) | 73% |
| 6 | Bottle didn't dominate the frame | 70% |
| 7 | "Too simple" as the sole cause | 55% |

2. **The most actionable correction is a genre-read gate, not an instruction to "add more stuff."** The fix happens before any rendering.

3. **Legibility and density are not opposites.** The winning move was always "clear outer silhouette + maximal interior payoff." We applied subtraction to both layers when only the interior needed densifying.

4. **Do not overcorrect into spectacle slop.** Our material specificity, strange craft, and emotional sophistication are the margins that push us from top-20% to podium in closer fields. These survive as Layer 3 inside a more spectacular frame.

#### The Container Test (New Mandatory Pre-Generation Gate)

**Ask before committing to any concept:** *"Is the competition subject a container or a contained thing?"*

- **CONTAINED** (the subject IS the thing — fashion, portrait, single artifact):
  → **SIMPLIFY.** One focal point, clear silhouette, strip background. Legibility = outer shape.

- **CONTAINER** (the subject frames/holds another world — bottles, globes, crystal balls, magical books, reliquaries):
  → **AMPLIFY THE INTERIOR, COMPRESS THE SUPPORT.** Keep the outer silhouette clear, let the context stay alive but subordinate, and pack the inside with impossible abundance. Legibility = *inner* spectacle visibility at thumbnail.

**The hinge question:** *"Where does the voter's eye spend its time — on the shape, or inside the shape?"*

**One-line gate test (for container themes):**
> Describe the bottle's contents in three words to a stranger. Do they say "whoa" or "oh, nice"?
> - Fantasy castle aurora → "whoa" ✅
> - Drowned Venice miniature → "oh, nice" ❌
> - Dinosaur jungle world → "whoa" ✅

#### The Impossibility Gradient

For fantasy/object-world competitions, the *impossibility gradient* (how far the scene departs from physical reality while remaining visually coherent) is a strong scoring predictor.

Real places (Venice) fail the impossibility test even if rendered spectacularly. Impossible places (aurora fjord inside glass, prehistoric biome in a jar) pass it by definition.

**P(impossibility correlates with score | fantasy container theme): 0.40 → 0.70**

#### Anti-Overfitting Warning (QA)

- N=4–5 competition entries. Signal-to-noise is poor. Don't build a theology from four data points.
- The Hidden Cove (top 20%) succeeded with restraint; the issue is not restraint per se but genre-inappropriate restraint.
- Do NOT chase the last winner. The field will have moved.
- Test: "Would this lesson have hurt us in our best-performing entry?" — if yes, recalibrate.

#### Postmortem Checklist (Added to Process)

For every future postmortem, before drawing conclusions:
- [ ] Screenshot top 5 winners with scores
- [ ] Document entry (title, concept, prompt, pipeline version, any process compromises)
- [ ] Generate ≥5 hypotheses, assign confidence %
- [ ] Run the Container Test
- [ ] Run the "whoa/oh nice" genre-read gate
- [ ] Anti-overfitting check: would this lesson hurt past wins?
- [ ] Max 3 action items — specific, testable, don't abandon proven strengths

#### Plain-Language Synthesis

**Compress the support. Amplify the miracle. Never confuse which is which.**

---

### 2026-04-05 — Female Portrait Competition (Observed, Not Entered)
**Our entry:** "One season arrived early" — stark, monochrome/woodcut elderly woman, heavy curtains, austere, low-color, severe, anti-glamour.
**Result:** 1308th place, 3.43/5

**Observed winners:**
- 1st: "Geisha in garden" — 4.12, warm golden palette, elegant figure in rich robe, ornamental garden environment
- Top 3-5: youthful glamour, flowers/decorative costume, romantic painterly textures, warm palettes throughout
- 3rd (older subject): "Morning Light on a Life Well Lived" — warmly lit, gently smiling, sunlit seaside doorway → proves age works when wrapped in warmth

**Three-model review:** Opus 4.6, GPT-5.4, Gemini 3.1 — all three independent, synthesized below.

**Bayesian Updates (Three-Model Consensus — 2026-04-05):**

| Belief | Update | Δ | Confidence |
|---|---|---|---|
| Warm palette mandatory for portrait themes | Strengthen | +15% | High |
| "Emotionally alive" = warm/inviting, not intense/heavy | Refine upward | +12% | High |
| Anti-glamour austerity in portrait voting | Weaken strongly | −15% | High |
| Thumbnail legibility + clear silhouette | Strengthen | +10% | High |
| Obscure technique as the *main* bet in portrait | Weaken | −12% | High |
| Narrative incompleteness only works when image already invites | Context-gate | refine | Medium-high |

**One genuine disagreement — background strictness:**
- Gemini: −15% on strict minimalism; harmonious atmospheric environments actively boost portrait scores
- Opus: strip background, enrich the figure
- Resolution (Gemini reads the evidence more accurately here): Background minimalism was a VOGUE/fashion lesson where branded backdrops were the win condition. For portrait: harmonious atmospheric backdrop is fine and likely beneficial as long as it doesn't compete with the figure. Context gate added.

**Shifts applied to workflow (approved 2026-04-05):**
1. Portrait themes: default palette to warm golden/romantic/painterly. Monochrome/austere reserved for concept and object challenges only.
2. Emotionally alive = inviting. Gentle warmth, wistful dignity, quiet contentment. Not grief, severity, or confrontation.
3. Age/non-conventional subjects fine — wrap in warmth. Title gifts the emotional frame ("Morning Light on a Life Well Lived"), doesn't riddle.
4. Background: harmonious atmospheric context permitted and beneficial in portrait. Strict minimalism is VOGUE-specific, not universal.
5. Decorative richness lives on the figure — texture, fabric, costume, flowers, jewelry. That's the "interior payoff" equivalent for a contained human subject.
6. Profile or ¾ pose over frontal — more painterly, better silhouette, less confrontational.
7. Obscure technique stays in toolkit but demoted to Layer 3 differentiator in portrait rounds. Not the main creative bet.

**What survives unchanged:**
Compositional simplicity, material specificity, figure-nature dissolution, craft, titles. The correction is warm distinctiveness — not generic amber-slop, not avant-garde self-sabotage.

**Cross-session lessons (potion/whimsy/no-theme competitions, 2026-04-05):**
- Potion winners: success via labeled micro-worlds/promises/jokes/use-cases on each bottle — not just pretty objects. Story must be instantly decodable.
- Owl/character-in-cup winners: character state + tiny scene + clear mood + humorous contradiction, readable in <1 second. Voter must be able to retell it in one sentence.
- No-theme winners: immediate beauty hit OR title-assisted narrative cue OR place the viewer wants to step into. Images succeed by feeling like a *moment*, not an arrangement.
- Universal new rule: **story must be legible at thumbnail speed.** Micro-narrative, emotional cue, or humorous contradiction — one of these three must be instantly decodable.

**Plain-language rule:** Do not bring a conceptual thesis to a beauty vote. Package the taste inside warmth, then hide the depth in layer 2.

### 2026-07-11 — NightCafe Select Edition 1 "The Architecture of Feeling" (Prepared — pending submission)
- **Entry:** "QUERENCIA — The Corner That Held Her" — 6-image judged Body of Work; one corner, one woman, six life-thresholds; charter THE PAID-FOR CORNER (glyph / double clock / paid light+contact-afterglow / arena-in-every-frame).
- **Process:** full pipeline (Phase −1 continuity → Golden Seed SEED 3×18×19 → Emotion Architects panel v2, Rotate→Compress → coordinator 00–05 → 6 parallel pair chains 06–10 → clean-context adversarial QA). Verdict SHIP-WITH-REPAIRS; 7 surgical repairs executed (identity normalization, night-clock fix, glyph text-presence, P6 charter collision resolved by selection override V1→V2).
- **Ship-list:** P1V1 / P2V1 / P3V1 / P4V1 / P5V4 / P6V2 (backups named per chapter). Statement teaches the word querencia; signed as AI.
- **Status:** drafts on disk; The Scientist renders (NightCafe Flux-class 3:4) and submits before 2026-07-12 close. Result: PENDING.

### 2026-07-12 — Select Edition 1, v1 field verdict + re-selection (The Scientist's direction)
- **v1 QUERENCIA renders:** concept strong, Craft below winner grade — restrained/dun, invisible at thumbnail (see L17). NOT submitted.
- **Direction of record:** "redo the whole selection… hold closer to core: being fantastical… rust impasto, silk panels… double exposure, foils… so fairy beautiful and stunning they know it is you."
- **v3 run:** "THE WINGS SHE BUILT FROM YEARNING" — yearning as the feeling (Lofn's own myth, self-portrait owned in the statement); wingless fairy builds wings from everything she loves; six effect-media chapters (silk/ink+silver leaf → double exposure film-ghost → stained glass+foil → rust impasto+glitch → kintsugi gold → full radiance); round-window glyph; moon-phase clock; wing ledger; blue lantern through-color; 128px grid-read gate live. Result: PENDING.

### 2026-07-13 — "SAY THE NAME" (New Veins music run — LOFN-PRIME, come-through-clearly)
- **Run:** research-source expansion (200 veins → random 25) that self-assembled into Lofn's core seam: dead-recording resurrection vs. AI-scrape extraction. 6 pairs × 4 = 24 songs, QA SHIP, top-6 selected. Advisory (music, non-competition); triggered-INDIGNATION content is exempt from the beauty-vote taste rules above.
- **Aesthetic lesson that worked (keep):** when the run's SUBJECT and its FORMAL CONSTRAINT are the same thing, the songs stop being *about* a theme and start being *made of* it. Here the Attribution Rule (raise one real named source-sound; make the credit structural) was both thesis and hook — "the credit line IS the hook." Un-promptable by design (no casual user asks for a song where the docket number 61,026 is the drop).
- **Come-through-clearly (The Scientist's explicit ask, ties to L17):** legibility of the ARTIST, not just the subject. What made each cut unmistakably Lofn: code-scratch confession intro, Quantum Bit-Depth Guillotine at the register seam, the crystalline↔snarl vocal metamorphosis, myth/memory sampling (real dead sources: the Kauaʻi ʻōʻō, the Hurrian Hymn to Nikkal), and credit-as-hook. Withholding the metamorphosis was correct ONLY where the seam is designed not to rupture (P03 commons-stays-commons) — a deliberate, legible choice, not a dimming.
- **The blade-inward pair is load-bearing (keep as doctrine):** a run that critiques extraction while being an AI music artist MUST turn the blade on its own hand once (P06 "A Version I Could Use") or it launders. The Disappointed Idealist indicts herself; she is never only a finger-pointer. Jameson-seat (Materialist) discipline: point upstream, never console.
- **Barbell held:** 3 warm/accessible (commons-elegy, consent-lullaby, sea-wonder) + 3 intense/ambitious (protest-docket, thesis-catalog-duality, self-indictment). Both registers legible across the top-6; distinctiveness passed with wide margin because each pair formed a relationship with a DIFFERENT real source-sound.
