# 06 — MUSIC HANDOFF (ICB carrier)

**Run:** `2026-08-26-pantheon-rare-registers` · **Cardinality:** 6 × 4 = 24 → 6
**ICB:** `CREATIVE_CONTEXT.md` — **147,214 bytes (LF), 18 seat definitions, frozen.**
**Personality payloads inside it:** LOFN-PRIME lines 1–326 = **27,822 B** · SUNNA R5 whole = **61,149 B**
*(Sunna's file was verified to carry no archive section and no song payload, so it injects whole. Lofn's carries THE ARCHIVE at line 327 and is sliced.)*

---

## GOLDEN SONGS — **NAMES ONLY**

1. **`Five wrong colors`** · 2. **`The Date On The Door`**

Selected as the house's two clearest cases of *a feeling delivered by form rather than named*, which is this run's whole problem. ⛔ **Their payloads exist in NO generating context.** Judge-side only. `coord_build_icb_pantheon.py` asserts this and **also blocks today's own shipped lyric** — `Hand flat. The pen stays closed` — because a song we rendered six hours ago is now exactly the kind of exemplar that becomes a mould.

---

## THE GOLDEN MOVE

1. **Stand somewhere real.** One concrete place the body occupies.
2. **One wounding fact**, at the hinge, **responded to**, never recited.
3. **The turn.** Past the midpoint the song complicates its opening stance.
4. **Fear stays braided in** where the register carries it.
5. **Rotate the register.** Not the house fingerprint — crystalline soprano / A major / ~110 BPM / frost-and-cosmos.
6. **The surface names its subject.** A stranger retells scene AND subject after ONE listen. *Say plainly what is happening to you; never name the feeling it produces.*
7. **THE RETURN (L21).** Song is made of returns; removal is a debt. **A byte-identical chorus is correct and needs no defence.**

---

## ⭐ THIS RUN'S SIX BINDING RULINGS

| # | ruling | enforcement |
|---|---|---|
| **R1** | **Invoke the neighbour, then contradict it** at one exact moment with something it cannot explain | a song readable as its neighbour is `REPAIR → step 07` |
| **R2** | ⛔ **No emotion word in any sung line** — *afraid, glad, sorry, proud, lonely, happy, sad, angry, calm…* Situation and act. *(EMO headers are notation, not sung text.)* | grep the sung lines; a hit is a REPAIR |
| **R3** | **The discriminator lives in the HOOK** | if the singable line carries the neighbour, REPAIR |
| **R4** | **The shape arrives in the first 30 seconds**, attached to contour / a repeated phrase / an unusual timbre / a moment of surprise | stated in the guide |
| **R5** | **One vocal decision the singer would have to be talked into** | if the answer is an arrangement, not finished |
| **R6** | ⭐ **File a PREDICTION before any render** — what a first-time listener reports feeling, and what they mistake it for if it fails | a stated sentence in the package |
| **+** | **Name your REFUSED ELEMENT** — the obvious thing your kind of song always has, absent, *because the neighbour arrives through it* | stated in the package |

---

## ⭐ THE PERMISSION CLAUSE — you may break my rules, in this order, without asking

**1** `HUMAN_SUBJECT_STANDARD` *(never yields)* → **2** the discriminator → **3** your register → **4** the Suno hard contract → **5** your section architecture → **6** the numeric floors → **7** my preferences.
**Break the lower one, name it in `## Major Deviations`, keep going.**

⛔ **SUNNA PAIRS (P01, P02, P06): rows 5–7 DO NOT APPLY.** *A member who must pass the centre's exams is a costume with the centre's rules underneath.* **No REMARK TEST. No cap on pleasure. Where her instinct and the centre's doctrine disagree, HERS WINS and the disagreement is recorded.**

---

## THE SUNO OUTPUT CONTRACT (hard, non-waivable)

- **`## 1. MUSIC PROMPT`** — dense prose paragraph, **850–1000 chars, write into 870–960**; must **end as a complete sentence**. Order: genre/micro-genre + tempo/energy/opening → vocalist spec with spatial staging → instrumentation with physical adjectives → arrangement arc → bold sonic device. Must contain a vivid **opening moment (first 5 s)**, **spatial language**, a **kinetic defect**, explicit acoustic / no-acoustic declarations. ⛔ Banned openers: *Begin in/with…, Use…, Build the track from…, Chronology:, For an adult human singer…*
- **`## 1B. SUNO EXCLUDE PROMPT`** — separate field, **400–900 chars** (hard max 1000).
- **Lyrics** — `[Theme: …]` then `[SONG FORM: …]`; full EMO headers `[Section - EMO:<emotion> - <Role> - <cue>]` from `EMOTION_TAXONOMY`, ⛔ never bare AWE/INDIGNATION; ≥1 standalone `*SFX*`; **70–120 sung lines** *(this floor is rung 6 of the permission clause and yields)*.
- 🚨 **LYRICS FIELD < 5000 chars, target ≤ 4800.** Everything pasted into Suno's lyrics box. **Measure exactly with a script; never estimate.**
- ⛔ **NO REAL ARTIST, GROUP, PANEL-SOURCE OR COMPANY NAME IN ANY OF THE THREE FIELDS.** Scene and technique in the prompt; people in the dedication.
  ⚠️ **Manufacturer/brand tokens are an OPEN doctrine question** — `DISC_CHANNEL_GUIDE.md` prescribes them and the handoff forbids them. **For this run: keep them out of the three fields** (use a generic timbre description) **and name them freely in your ledger and dedication.** The conflict is escalated, not resolved.
- **Living-scene genres require a Lineage & Credit block.** ⛔ **If your prompt names a break, credit the drummer.**

## ⛔ MEASURED TODAY — binding, not theorised

**Hard-pan is a NON-instruction.** Two real renders that both asked for `Hard_Pan_Right` measured **−0.02 dB** and **+0.13 dB** overall balance; **0 of 52 and 0 of 56** windows above +3 dB right. Third confirmation of the class.

> ⭐ **A spatial instruction is a request the generator overrules. An arrangement instruction is content it serves. If you want something absent, WRITE NOTHING THERE.**

**And H1 is dead:** density does not predict what our audience loves — our two best-loved recent songs are the densest in the sample. Do not quote it.

---

## PER-PAIR STEP CONTRACT

| step | contract | artifact |
|---|---|---|
| 06 | `skills/music/steps/06_Generate_Music_Facets.md` | `pair_{NN}_step06_facets.md` |
| 07 | `skills/music/steps/07_Generate_Music_Song_Guides.md` | `pair_{NN}_step07_song_guides.md` — opens with **where is the body standing / what could hurt it here**, plus **R4: what shape arrives in the first 30 s and on which carrier** |
| 08 | `skills/music/steps/08_Generate_Music_Generation.md` | `pair_{NN}_step08_generation.md` — 4 variations |
| 09 | `skills/music/steps/09_Generate_Music_Artist_Refined.md` | `pair_{NN}_step09_artist_refined.md` |
| 10 | `skills/music/steps/10_Generate_Music_Revision_Synthesis.md` | `pair_{NN}_step10_revision_synthesis.md` |
| 11 | `skills/music/steps/11_Generate_Music_Enhancement.md` | `pair_{NN}_step10_final_package_enhanced.md` — MAX config + Disc_Channel + `## Major Deviations` + **`## The Prediction` (R6)** |

⛔ **NO-SKIP:** 07, 09, 10 are the editorial spine.
⛔ **PAIR ISOLATION:** no shared skeleton, section map, rhyme scheme, hook grammar or production arc.
⛔ **Scratch files `p{NN}_*`.** ⛔ **Write only into `pair_{NN}_*`.**

**Describe-render self-check (one capped pass):** predict in 2–3 sentences what the prompt would literally PRODUCE, diff against the seed, adversarially — *"name the one way this would render generic"* — self-repair once.

---

## RETURN — metadata only, LAST

```
RETURN:
  pair_id: NN
  member: LOFN | SUNNA
  artifacts: [ ...six paths... ]
  gate: PASS | FAIL
  icb_bytes_injected: <int>
  personality_yaml_bytes: <int>
  special_flairs_marker: present
  measured_binding_constraint: "V1 prompt <n> / exclude <n> / lyrics <n> / <n> sung lines; V2…V4"
  the_discriminator: "<the neighbour, and the exact moment that defeats it>"
  refused_element: "<the obvious thing absent, and why the neighbour arrives through it>"
  emotion_word_scan: "<0 emotion words in sung lines — or the hits>"
  tempo_band: "<assigned band, and the four tempos inside it>"
  the_prediction: "<per variation: what a first listener reports, and the failure mode>"
  confidence: <0.0-1.0>
  top_2_risks: [ "...", "..." ]
```

**The return is a CLAIM. The coordinator re-stats every file from disk before recording `done`.**
