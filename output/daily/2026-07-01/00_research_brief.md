# Daily Research Brief — 2026-07-01
**Scope: TEST SLICE** — music only, **2 pairs × 2 variations = 4 songs → select per arm**, library personality + panel. First run on the geared pipeline (Golden Move, cut ledger, REDIRECT, one-fact rule, per-pair variation angles, blind golden+decoy QA). Model tiering for this run: **Sonnet-5 generates (Phase 0/1, steps 00–10), Fable refines (step 11) and judges (QA)** — generator and judge share no weights.
**Down-scaling is explicit:** 2 pairs (1 ACCESSIBLE + 1 AMBITIOUS; 1 NEWS + 1 EXISTENCE), 2 variations each. QA reads this as intended cardinality.

## Verified facts (fetched live this session)
- **F4 (USGS):** No significant quakes in today's feed. Context: last week's deadly Venezuelan earthquakes still resolving — see F20.
- **F6/F7 (NASA APOD, Jul 2): "Sibling Supernova Remnants."** Two overlapping supernova remnants ~6,000 ly away: the younger Jellyfish Nebula (yellow, center) layered over the older G189.6+3.3 (purple filaments arcing across). A bright triple star (Propus) at the right edge. **Image structure:** two expansions of different ages superimposed — the young loud shell over the old slow one.
- **F18 (NOAA SWPC):** Solar wind 373 km/s at 2026-07-02T04:04Z — a quiet sun today.
- **F19 (Hacker News top):** "Bring Back Crappy Forums" · "Searchable directory of 22k+ products from worker-owned co-ops" · "Building an Open-Source Robot Vacuum – Meet Oomwoo" · "ZCode – Harness for GLM-5.2" · "Senior SWE-Bench."
- **F20 (BBC World):** At least eight killed in major missile/drone strikes on Kyiv · **Aunt of Venezuelan boy (2) pulled from rubble on day six says she will give him "mother's warmth"** · Trump's unprecedented $2.2bn White House-year income · **Hong Kong pulled an AI-generated anti-drug K-pop video after backlash that it made drugs look appealing** · Ukrainian charged in Germany over Nord Stream.
- **F13 (Color API #0701):** **#770011 "Venetian Red"** — dried-blood brick red; old wounds, kiln heat, rust.
- **F-moon (moongiant):** Waning Gibbous, **97%** — the night after fullness; light already leaving.
- **F9/F10 (Bandcamp Daily, exact sonic-texture quotes):**
  - GiGi FM (minimal jungle): *"rooted in the bittersweet mix of rave sunrise bliss and bluesy soul sadness."*
  - Peter Kan (glitch-dub): *"like they're being transmitted from some alien dimension. But as they unfold, and deep, deep dub bass flows through them, there's a sense of bubbling fertile life about them."*
  - Cresfenn (breakcore): *"all the noise barrage of classic breakcore… but there's also a demented joy to the melodies and delight in applying slithering pitch-shift to everything."*
  - Also live on Bandcamp Daily: "The Endless Orbit of Spacemen 3" (feature title — the orbit motif is in the air).
- **F17 (Oblique Strategies):** UNAVAILABLE (source down). Continue without.

## ⚠️ Human-subject discipline (binding, pre-draft)
The Venezuelan rescue (F20) involves a **real, named, recent child victim**. Per `vault/HUMAN_SUBJECT_STANDARD.md`: the piece may carry the **charge** — being lifted out of the dark on day six; warmth given by someone who is not your mother — but **every person, name, place, and circumstance must be invented**. No name, no Venezuela, no "day six of the earthquake." Same for Kyiv: pattern, not people. REAL GRIEF IS NOT RAW MATERIAL.

## Tri-Source Declaration
- **Source 1 — CONTENT / stakes:** the sibling supernova (a survivor star orbiting the expanding shell of the twin whose explosion lights it) = EXISTENCE anchor. The AI anti-drug video that made the harm gorgeous (the slop economy's machinery aestheticizing what it claims to warn against) = NEWS anchor, triggered INDIGNATION. Waning gibbous 97% and Venetian Red available as atmosphere.
- **Source 2 — SONIC VOCABULARY (exact Bandcamp language):** "rave sunrise bliss and bluesy soul sadness" · "deep, deep dub bass… bubbling fertile life" · "demented joy… slithering pitch-shift." Import these words, not genre labels.
- **Source 3 — MATERIAL STRUCTURE (APOD form rule, mandatory):** **two superimposed expansions of different ages** — every song carries an OLD, slow structure running underneath a YOUNG, bright one (e.g., a slow 4-chord shell cycling beneath a faster verse engine; the old layer surfaces alone at the bridge). The overlap is the form.

## Pair split (test slice: 1+1)
- **Pair 01 — ACCESSIBLE / EXISTENCE / AWE (terror-adjacent):** the survivor of a binary star, standing in the light of the sibling whose death illuminates it. Body stands somewhere REAL (an observatory catwalk, a cold field at 3am — the pair decides; not a kitchen). ONE wounding fact sung, responded to (candidates: 6,000 light-years; or 97% and waning). The clean fear: everything that lights you cost somebody an explosion.
- **Pair 02 — AMBITIOUS / NEWS / INDIGNATION:** the machine that makes the poison beautiful — an official warning rendered so glossy it becomes an advertisement. Body stands somewhere REAL (a train car under the ad screen; a corridor of loops). Breakcore/glitch-dub vocabulary from Source 2 fits the register. The fear: your own eye liking it. Invented institution, invented video, no real-person likeness.

## EXISTENCE prompts (interior-life questions today's songs can answer)
1. What do you owe the version of you that had to explode for you to exist?
2. Who kept warmth for you when your own source went dark?
3. Is a small ugly room where people know your name worth more than an infinite polished feed?
4. What does the body do on the sixth day of waiting?
5. When the thing built to warn you makes the danger beautiful, who failed you?

## Advisory learnings consulted
Tag-walk of `vault/COMPETITION_LEARNINGS.md`: **0 venue lessons intersect** (L1–L8 are NightCafe/image-scoped; this is a music practice run — no leakage). L9 (anti-overfit) noted. **Session-scoped process lessons that DO apply** (from the 2026-07-01 regression review, advisory): rotate the register away from the calcified house fingerprint (crystalline-soprano / A-major / ~110 BPM / frost-and-cosmos palette — see `gates.yaml → house_lexicon`); AWE stays terror-adjacent, not domestic reassurance; at most ONE sung numeric fact. INDIGNATION exempt from all venue-taste suppression; advisory-only.

## Publish policy
This is **practice**. Nothing from this run publishes without the full rig + cross-model step-11 review + the Scientist's ear, borderline defaulting to HOLD.

---
## FULL-RUN EXTENSION (added after test-slice review, same date/controller)
**Scope now: FULL DAILY** — music 6 pairs × 4 = 24 → best 6 (3+3 arms), image 24 prompts → top 12 → top 6. Full-run artifacts land in `music/` and `images/` subdirs; root artifacts are the test-slice record. Tiering: **Opus generates (assignments, steps 00–10), Fable refines (step 11) and judges (QA).**
**F1–F3 (NightCafe daily challenge): UNAVAILABLE** (JS-gated; fetch + search attempted). Image lane runs on the day's own themes, non-competition; NightCafe venue lessons apply as advisory only where tagged.
**⚠️ THE SCIENTIST'S TEST-SLICE VERDICT (binding on every pair brief):** structure and composition were right; **emotional connection failed on lyric coherence** — an AWE song about a survivor star read as being about a troubled family member ("did an uncle do something wrong?"). The fix is Golden Move **rule 6 — the surface names its subject**: a stranger retells scene AND subject in one sentence after one listen; the subject appears PLAINLY at least once (title or early verse); depth lives in the RESPONSE to the named thing, never in withholding it. An unnameable subject is `REPAIR — FOG` at the Somatic Gate. The two test HOLDs' counter-moves carry forward: terror-adjacency must be REALIZED not asserted (P1V2), and sibling variations may not share payoff lines (P2V2, 28% overlap).
