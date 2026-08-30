# step04 — MEDIUM · `2026-08-14-daily-music`

**Continuity Payload Used:** the frozen ICB (Golden Seed · metaprompt · both personality DNA blocks · all 18
voices · all 15 **Special Flairs** · palettes · the GOLDEN MOVE).
**Inputs:** `step02_concepts.md` · `step03_artist_and_critique.md` · `05_pair_assignments.md` (the beds).

The medium here is **a Suno two-field package**: a dense-prose style prompt, a separate negative field, and
a lyrics field with EMO headers. Everything below is about what that medium can and cannot carry.

---

## 1. WHAT THE MEDIUM WILL NOT DO (measured, not assumed — L22)

From the render audit, `n=8`, **these are facts about the generator, not opinions about it**:

| asked for | what came back | class |
|---|---|---|
| a ~4-second full stop | **0.40 s at −5.7 dB** | ⛔ interrupting silence gets **filled** |
| a quarter-tone drone pair | **zero isolated tonal components** | ⛔ anti-musical asks get smoothed |
| hard-panned non-musical elements | **near-mono image** | ⛔ spatial language is cheap |
| programmed tempo transformations | **steady grids, 2 of 2** | ⛔ tempo *dramaturgy* smooths |
| opposed vocal characters in one song | **one blended lead, 2 of 2** | ⛔ |
| a near-silent opening | **arrived, −46.8 dB** | ✅ with the grain |
| steady BPM target | **within ±5 BPM, 6 of 6** | ✅ |
| progressive sub build to the final chorus | **arrived** | ✅ |
| terminal silence | **arrived, 1 of 1** | ✅ |

### ⛔ THE CONSEQUENCE FOR THIS RUN'S FORM RULE

**THE FLASH AND THE CORONA was designed against this table, and it survives it — but only in one direction.**

- **THE FLASH** ✅ — a ≤2-bar bright/low hit at a transition is a thing a pop arrangement *wants to do*.
  With the grain. It will arrive.
- **THE CORONA** ⚠️ — *"a quiet element revealed by a dropout"* is **exactly the smoothed class.** An
  interrupting void gets filled; dynamic dramaturgy flattens.

> ⭐⭐ **THEREFORE, BINDING ON ALL SIX PAIRS: the corona is carried by the LYRIC, and the mix is its
> accompaniment — never the reverse.** The reveal must be **countable in the text**: a line that was a
> fragment and is now finished · a word that was buried in a stacked chorus and is now bare · a second voice
> that answers where nothing answered before · a repeat count that changes. **Nobody can quantise a line
> count away.** *(Albini's standing veto: an objection answered only in the production spec is not answered.)*

**And the productive half (The Scientist, 2026-07-24):** *"I judge the final result, not the distance from
intent… having it fail in just the right way can create new sonic experiences."* On the benchmark, width
**narrowed** into the loud sections — the exact inverse of the spec — and for that song it was truer.
⭐ **So: an intent-vs-render diff on this run is RAW MATERIAL, never a compliance score.** If the corona
arrives as something other than specified, we judge what arrived.

---

## 2. THE SIX BEDS AS MEDIUM DECISIONS

⛔ **THE INVERSION IS THE MEDIUM RULE OF THIS RUN** *(2026-08-13, hers; Fisher's veto, today)*:
**the modern genre is the WORLD; the historical element is the INTRUSION; one seam.** A historical bed named
as *dominant* smooths to pastiche, **because the generator resolves to its strongest prior and its strongest
prior for "1979 two-tone" is a pastiche of 1979 two-tone.**

| slot | the WORLD (dominant, modern) | the INTRUSION (one, historical/borrowed) | where the seam goes |
|---|---|---|---|
| **P01** | contemporary ska-punk / two-tone revival, live-room drums, horns | **rocksteady half-time**, arriving once | at the moment she joins — the bounce changes gear under her |
| **P02** | modern power-pop, 2026 production, guitars up | **sixties girl-group handclap + stacked "ooh"**, as a visitor | when the tape is playing — the past enters as sound, not as style |
| **P03** | contemporary bass-forward dub-adjacent electronic | **spring-reverb / tape-echo dub gestures**, sparingly | at the dropout, which is the corona |
| **P04** | 2026 jungle / breakbeat, modern low end | **the chopped break itself** — a borrowed four bars | at the point the loop reveals its internal figure |
| **P05** | speed garage / bassline, current | **the gear-change into half-time dub**, once | at the change, placed to ruin the flow on purpose |
| **P06** | Jersey / Baltimore club, current, chopped | **Glitch-Baroque ornament** — a harpsichord figure bitcrushed | at the wrong entry in the catalogue |

⭐ **One seam each. Never two.** *(In-lane N=1: `sunna.yaml`'s ONE DEVICE PER SONG — if you cannot name the
device in four words, it is more than one device.)*

---

## 3. THE STYLE-PROMPT CONTRACT AS A MEDIUM CONSTRAINT

**Dense prose paragraph, 850–1000 chars, written into 870–960** *(pinning the cap is a boundary-hug FLAG —
every late-June prompt measured 988–999 against a 1000 ceiling, which is what optimisation pressure looks
like)*. **Must end as a complete sentence.** Mandatory internal order:

`genre/micro-genre + tempo/energy/opening` → `vocalist spec with spatial staging` → `instrumentation and
palette with PHYSICAL adjectives` → `arrangement arc` → `the bold sonic device`.

Must contain: a vivid **opening moment (first 5 s)**, **spatial language**, a **kinetic defect** (an
asymmetric groove — `sunna.yaml`'s kick nudged **+8 to +14 ms late** with the bass on the grid is the house
example), and explicit **acoustic / no-acoustic** declarations.
⛔ **Banned openers:** *Begin in/with… · Use… · Build the track from… · Chronology: · For an adult human
singer…* ⛔ **No real-artist names in the style field** — ghost-homage lives in the lyric and the credit
block, never in the generator's instructions.

**EXCLUDE field, separate, 400–900 chars:** concrete blacklist tokens, not avoidance prose. *(Avoidance
prose inside the style field is read as content — Suno applies EXCLUDE as negative tokens and the style
field as positive ones, so "no airhorn" in the style field asks for an airhorn.)*

---

## 4. SOURCE-2 AS MEDIUM VOCABULARY (the floor, and where it is counted)

The day's Bandcamp texture language enters the **style prompts as written**, not paraphrased into genre
labels. Suggested distribution *(pairs may re-choose, but the floor of ≥2 phrases per generator-facing
prompt is fixed)*:

`"low-slung bass, swelling synths"` · `"high-voltage guitars and Close Encounters synths"` ·
`"guitar diverts into a weepy countermelody"` · `"subdued dynamics, liminal melodies, and quiet-storm R&B
rhythms"` · `"gorgeously restrained beats"` · `"bone-rattling breakdown with battering drums and explosions
of guitar"` · `"natural entropy — recording hum, dissonant scrapes, the odd messy recording"` ·
`"shambling rock 'n' roll"` · `"rambunctious, unwieldy"` · `"glitchy and frenetic"`.

⛔ **Counted in the SHIPPED PROMPT TEXT, never in the design document.** *(2026-08-12: a pair shipped zero
Source-2 phrases in all four generator-facing prompts against a floor of 2, because its self-check counted
them where it had listed them. **SPEC ≠ ARTIFACT.**)*
⚠️ **And the repair caveat, measured:** folding identical mandatory phrases into four prompts **collides
them** — the 2026-08-12 fix pushed distinctiveness from 0.523 to 0.669, over a 0.58 ceiling. **Vary which
two phrases each variation uses, and vary where in the paragraph they sit.**

---

## 5. THE LYRICS FIELD AS A PHYSICAL OBJECT

🚨 **< 5000 characters, target ≤ 4800, measured exactly and stated.** The field holds **everything pasted
into Suno's lyrics box**: `[Theme:]` + `[SONG FORM:]` + the Disc_Channel block + every EMO header + every
`*SFX*` cue + all sung lines. **A song over 5000 will not render**, and the 70–120-line target yields to the
cap — a renderable 64-line song beats an unrenderable 110-line one. If over: cut/merge sung lines, then
tighten headers, then move the Disc_Channel block to a `## Production Sidecar` **outside** the field.

⛔ **EMO headers:** `[Section - EMO:<emotion(s)> - <Role> - <cues>]`, emotion drawn from the taxonomy in
`step00`. **Never bare AWE / INDIGNATION** — those are categories. *(104 bare mode-word tags shipped across
4 of 6 pairs on 2026-08-13 in exactly this field, because it was the one thing nothing counted. It is
counted this run.)*

⛔ **Any SUNG numeral is spelled out in words** — *"eight seconds"*, *"four hours"*, *"two hundred and
twenty-five"*. Digits stay legal in `[Theme:]`, `[SONG FORM:]`, EMO headers, BPM and key: **none of those is
voiced.** *(A listener told us this in 2026-05 and we sat on it for thirteen weeks.)*

---

## Self-critique

This step is heavy on prohibition. ⚠️ **That is the L21 tell** — *"a pair brief written entirely in
negatives will produce a lecture no matter how good the concept is."* Counted honestly: this file carries
**14 prohibitions and 9 positive constructions**, which is the wrong ratio for a document pair agents read.
**Mitigation, deliberate:** the positive vocabulary lives where the writing actually happens — the six axis
options, the SOUND/RETURN assignment, the Source-2 phrase bank, and the per-slot variation angles, all of
which are **additive** and all of which sit in `05_pair_assignments.md` and the isolation briefs. **The
prohibitions are in the medium file; the invitations are in the brief.** That separation is intentional and
it is checkable.

*Step file: `skills/music/steps/04_Generate_Music_Medium.md`. No golden-song payload consulted.*
