# 06 — MUSIC HANDOFF (ICB carrier)

**Run:** `2026-08-26-opus-masterpiece` · **Modality:** music · **Cardinality:** 6 pairs × 4 = 24 → best 6
**ICB:** `CREATIVE_CONTEXT.md` — **87,216 bytes (LF-normalised), 18 `(after ` seat tags, frozen.**
**Personality payload inside the ICB:** `lofn-prime-mini.yaml` **lines 1–326 = 27,822 bytes.**

---

## GOLDEN SONGS — **NAMES ONLY**

Per `EXECUTION.md` §3 GOLDEN-OUTPUT QUARANTINE, and reinforced by yesterday's breach in `2026-08-26-music-daily-changes` ("one P02 executor consumed Golden payloads beyond its isolated brief"):

1. **`Five wrong colors`** — https://suno.com/song/f175f806-b1a3-40f0-8984-0c92667209dc
2. **`The Date On The Door`** — the 2026-08-05 `the-junk-and-the-light` P05 V1 selection

**Selected because they are the two songs this run was commissioned to rival**, and because they are the clearest surviving evidence of the catchable-operation engine. They are the **standard, not the template.**

⛔ **THEIR PAYLOADS DO NOT EXIST IN ANY GENERATING CONTEXT ON THIS RUN.** Not in the ICB, not in any pair packet, not in step 11. `coord_build_icb.py` asserts this mechanically on every build and the build fails closed. Payloads are available **judge-side only**: `lofn-qa` blind comparison, step 12, and `lofn-step11-packager`.

⚠️ **Family guard.** P04 works in the `Date On The Door` family and P05 in the `Five wrong colors` family. Each **must change the operating dimension** — P04 substitutes **pronouns** where that song substituted verbs; P05 exits on **distance** where that song exited on colour. A pair that reproduces the exemplar's own dimension is a `REPAIR`, and neither pair has ever seen the exemplar.

---

## THE GOLDEN MOVE (what generators get INSTEAD of the golden songs)

1. **Stand somewhere real.** One concrete place the body occupies — name where it stands and what the senses register there. Concept-illustration is the failure mode; experience-report is the move.
2. **One wounding fact.** At most **ONE** numeric fact sung, at the emotional hinge, and the lyric **responds** to it, never recites it. All other research stays atmosphere.
3. **The turn.** Past the midpoint the song contradicts or complicates its opening stance. A song that asserts its final emotion from line one is a corpse.
4. **Fear stays braided in.** AWE is terror-adjacent sublime, not domestic reassurance.
5. **Rotate the register.** Not the house fingerprint — crystalline soprano / A major / ~110 BPM / frost-and-cosmos. Vary key, tempo, vocal register and sonic world per pair.
6. **The surface names its subject.** A stranger retells the scene AND subject in one sentence after ONE listen; the subject appears **plainly** at least once. Obliqueness is fog, not depth.
7. ⭐ **THE RETURN (L21).** Song is made of returns. **Removal is a debt** — a pair stripping end-rhyme names what returns in its place. **A byte-identical chorus is correct and needs no defence; say nothing about it.**

---

## ⭐ THIS RUN'S SEVEN ADDITIONAL BINDING RULES

| # | rule | source | enforcement |
|---|---|---|---|
| **R1** | **LITERAL, NOT METAPHORICAL.** The operation is a real condition of the singer, never a figure for one. | Ruling 1 | a metaphorical operation is `REPAIR → step 07` |
| **R2** | **THE ARMS SPLIT.** AMBITIOUS (P04–P06) holds the **flat register** — level, distance and warmth identical; EMO marks the ROOM, never the DELIVERY. ACCESSIBLE (P01–P03) rotates fully into appetite. **No averaging.** | Ruling 2 | a build/lift in P04–P06 is `REPAIR` |
| **R3** | **THE BODY TEST — pass/fail.** Temporal operations specified in **body units** (a beat in a four-bar phrase), never notation units. And: *would the singer still be inside this operation if nobody were recording?* | Ruling 3 | a sixteenth-note drift is `REPAIR` |
| **R4** | **THE NAMING BAN + DELETION TEST.** No lyric names its own operation. **Strike every line referring to the structure — if the operation vanishes, the song is a caption.** | Ruling 4 | `REPAIR → step 07`; run the test and state the result |
| **R5** | **NO CEILING ON THE FUN.** ≥3 of 6 enjoying themselves, ≥2 outright fun. The clean-fear slot does not police the others. | Ruling 5 | portfolio-level, checked at selection |
| **R6** | **ONE SEAM, N=1.** One dominant material + at most one intruder + one seam, **and the subject must be doing something across it.** The seam is a **bodily event** — a moment the singer's body changes what it is doing — never an edit point. | L38 + Derbyshire seat | two collisions in one song is `REPAIR` |
| **R7** | **ASYMMETRIC RETURN.** Whatever returns comes back changed **on one side only**. Symmetrical return is decoration. | SOURCE 3 / F07 | checked at step 10 |

---

## THE SUNO OUTPUT CONTRACT (hard gate — non-waivable)

- **`## 1. MUSIC PROMPT`** — dense **prose paragraph, 850–1000 chars, target 870–960** (pinning the cap is a FLAG). Must **end as a complete sentence**. Mandatory order: genre/micro-genre + tempo/energy/opening → vocalist spec with spatial staging → instrumentation with physical adjectives → arrangement arc → bold sonic device. Must contain a vivid **opening moment (first 5 s)**, **spatial language** (left/right/centre/depth), a **kinetic defect** (asymmetric groove), and explicit acoustic / no-acoustic declarations. ⛔ Banned openers: *"Begin in/with…"*, *"Use…"*, *"Build the track from…"*, *"Chronology:"*, *"For an adult human singer…"*.
- **`## 1B. SUNO EXCLUDE PROMPT`** — separate negative field, **400–900 chars** (hard max 1000). Concrete blacklist terms, not avoidance prose.
- **Lyrics** — open `[Theme: …]` then `[SONG FORM: …]`; full EMO headers `[Section - EMO:<emotion> - <Role> - <cue>]` with emotion from `EMOTION_TAXONOMY`, ⛔ **never bare AWE/INDIGNATION**; ≥1 standalone `*SFX*` cue; **70–120 sung lines**.
- 🚨 **LYRICS-FIELD HARD CAP: < 5000 chars, target ≤ 4800.** Counts *everything pasted into Suno's lyrics box* — Theme + SONG FORM + Disc_Channel block + every header + every SFX + all sung lines. **Measure exactly; never estimate.** If over: cut/merge sung lines, then tighten headers, then move Disc_Channel + metadata to a `## Production Sidecar` **outside** the field. **The 70–120-line target yields to this cap.**
- ⛔ **NO REAL ARTIST, GROUP, PANEL-SOURCE OR COMPANY NAMES IN ANY OF THE THREE FIELDS.** Name the SCENE and the TECHNIQUE (public idioms); name the PEOPLE only in the dedication. *A name in the prompt is an instruction to imitate; a name in the dedication is a debt paid.* **Verify by scanning the three pasted fields against the run's credited roster — never by a proper-noun heuristic.**
- **No house-lexicon self-copying** (`gates.yaml → house_lexicon`).
- **Female vocals default.** Prove the active personality: sonic-world sentence + signature device + seed-derived weirdness.
- **Living-scene genres require a Lineage & Credit block.** ⛔ **If the prompt names a break, credit the drummer** (amen → Gregory C. Coleman / The Winstons). A generated track is **not** a sample — write *"we asked for this figure by name, so we name him."*

---

## PER-PAIR STEP CONTRACT (subagents produce all six files for their pair)

| step | contract file | artifact |
|---|---|---|
| 06 | `skills/music/steps/06_Generate_Music_Facets.md` | `pair_{NN}_step06_facets.md` |
| 07 | `skills/music/steps/07_Generate_Music_Song_Guides.md` | `pair_{NN}_step07_song_guides.md` — **opens with the two pre-draft questions:** *Where is the body standing?* · *What could hurt it here?* Guides DIRECT; they do not draft. |
| 08 | `skills/music/steps/08_Generate_Music_Generation.md` | `pair_{NN}_step08_generation.md` — 4 variations |
| 09 | `skills/music/steps/09_Generate_Music_Artist_Refined.md` | `pair_{NN}_step09_artist_refined.md` |
| 10 | `skills/music/steps/10_Generate_Music_Revision_Synthesis.md` | `pair_{NN}_step10_revision_synthesis.md` |
| 11 | `skills/music/steps/11_Generate_Music_Enhancement.md` | `pair_{NN}_step10_final_package_enhanced.md` (MAX config + Disc_Channel + `## Major Deviations`) |

⛔ **NO-SKIP:** steps **07, 09, 10** are the editorial spine. A pair missing any of them is **NON-CANONICAL** — cannot SHIP, cannot be published.
⛔ **PAIR ISOLATION:** no shared lyric skeleton, section map, rhyme scheme, hook grammar or production arc across pairs. Validators and file writers may be shared; creative bones may not.
⛔ **SCRATCH FILES ARE PAIR-PREFIXED `p{NN}_*`** — 2026-08-21 collision watch item. An instrument that can be replaced under you while still returning numbers is the gate-that-cannot-fail class.
⛔ **WRITE ONLY INTO YOUR OWN `pair_{NN}_*` NAMESPACE.** Never the INDEX, never RUN_STATE, never another pair's file.

**Describe-render self-check (one capped pass, inside the existing max-3 loop):** before returning step 10, predict in 2–3 sentences what the Suno prompt would literally **PRODUCE** — the opening 5 s, the vocal placement, the groove, where the bold device lands — then diff that against the Golden Seed and the GOLDEN MOVE, phrased adversarially: **"name the one way this would render generic."** Self-repair **once** if it drifts. No render-verifier subagent, no recursion.

---

## RETURN BLOCK (metadata only, LAST, after the files are written)

```
RETURN (metadata only — no creative payload):
  pair_id: NN
  artifacts: [ ...six canonical paths... ]
  gate: PASS | FAIL
  icb_bytes_injected: <int>
  personality_yaml_bytes: <int>
  special_flairs_marker: present
  measured_binding_constraint: "MUSIC PROMPT <n> chars | lyrics field <n> chars | <n> sung lines"  (per variation)
  deletion_test_result: "<what vanished when structure-referring lines were struck>"
  body_test_result: "<the operation stated in body units>"
  confidence: <0.0–1.0>
  top_2_risks: [ "...", "..." ]
```

**The return is a CLAIM. The coordinator re-stats every file from disk before recording `done`** — existence, byte size, recomputed binding constraint, and the ICB prefix as an unbroken substring. **The subagent's number is the claim; the coordinator's stat is the proof.**
