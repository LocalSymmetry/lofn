# PAIR 05 — STEP 06 · SCORING FACETS
## `daily-music-2026-08-21` · **C-27 · BEFORE ANYONE LOOKED** · AMBITIOUS / EXISTENCE

> **ICB status:** the frozen Immutable Continuity Block (`output/daily/2026-08-21/CREATIVE_CONTEXT.md`,
> 113,659 bytes) is pinned at the head of this pair's reasoning and was not edited, summarised or paged
> out. This artifact is the pair's divergence surface; the canonical block stays byte-identical.
> **Special Flairs marker: present** (all 15 carried; this pair works **#5 Findable, Not Found** and
> **#11 Slightly Under, And Staying** as *motifs*, never as phrasing).
> **Golden-output quarantine honoured:** no archived song or image payload was read, sought or imported.
> Generative inputs were the Golden **Seed** and the **GOLDEN MOVE** only.

---

## 0. WHAT THESE FACETS ARE FOR

Five judging criteria, written **before any lyric exists**, that every downstream candidate prompt in this
pair is scored against. They are this pair's rubric alone — derived from *this* concept, not from the run.

**The concept, in one sentence:** a woman alone in a building after hours puts her initials on a page to
verify something that was already settled before the instrument got there, and the mark she is adding will
be taken off by another process before morning.

**Style axes (0–100) for this pair.** Tempo **72** (104 BPM, but read as brisk-not-driving) · Energy **22**
(the whole trick) · Harmonic complexity **4** (one chord, no lift) · Rhythmic complexity **81** (sprung
stress against a grid that never yields) · Timbre richness **44** (few objects, all physical) · Vintage vs
modern **63** · Vocal prominence **86** (almost-spoken, dry, forward) · Organic vs synthetic **52** (a
machine world with a hand-object intrusion) · Genre purity vs fusion **58** · Narrative emphasis **90**.
**Creativity spectrum:** literal 15 / inventive 35 / transformative 50.

⚠️ **The axis that fights the generator:** Energy 22 at Tempo 72. A 104 BPM machine pattern wants to become
a loud dance record, and a renderer smooths a *quiet* instruction far more readily than a loud one. Every
facet below is therefore written so that **level is structural** — a consequence of what is in the room —
rather than a request for restraint.

---

## 1. THE FIVE FACETS

### FACET 1 — **THE LEVEL HOLDS, AND THE LATCH IS THE CEILING**
Does the piece stay at speaking volume from the first second to the last, with a metal cabinet latch as the
loudest object in the room — so that paper, a pen nib and the singer's breath all stay audible underneath
it? Score down any candidate whose loudness is *asked for politely* instead of built into the object list;
score down harder any candidate where a later section arrives bigger than an earlier one. **The test is
whether a listener could hold a conversation over it and still hear every word.**

### FACET 2 — **THE SUBSTITUTED DOWNBEAT — SOMETHING IS THERE, EVERY BAR**
Where the kick belongs, is a cabinet latch audibly present **on the grid, in every bar**, recorded dry and
close enough that the wood-and-metal of it reads as an object and not as a sample? A candidate that
achieves its effect by *removing* the kick has written a hole, and holes smooth away in the render. Score
the presence of the replacement, never the absence of the original. Sensory check: **you should be able to
tell which hand shut it.**

### FACET 3 — **THE FLAT THAT BECOMES THE KEY**
Does the lead sit a consistent fraction under pitch and **stay there, uncorrected, long enough that the ear
stops hearing it as an error and starts hearing it as the tuning**? Score down any candidate that treats
flatness as an effect applied to a phrase rather than a condition held for a whole song, and any candidate
whose spec leaves room for pitch correction to quietly rescue it. Sensory check: by the last chorus, **the
in-tune sound would be the wrong one.**

### FACET 4 — **SPRUNG STRESS AGAINST A GRID THAT NEVER YIELDS**
Do the vocal stresses keep landing off the pulse — early, late, on the weak syllable — while the machine
refuses to follow, so that the friction is continuous rather than an occasional trick? Score the *rate* of
disagreement, not the size of it: one dramatic push is a gesture, a whole song of small ones is a body
working against a clock. Sensory check: **you can feel the singer being slightly ahead of the room and the
room not caring.**

### FACET 5 — **THE FACT IS ANSWERED BY A HAND, NOT EXPLAINED BY A NARRATOR**
Does *thirty femtoseconds* arrive **once**, at the emotional hinge, spelled in words, printed on an object
that is physically in the scene — and is it met by a person **doing something with her hands**, with no
science-explainer register anywhere in the song? Score down any candidate where the number is defined,
contextualised, marvelled at, or used to prove a point; score up any candidate where the response is a
motion the body was going to make anyway. Sensory check: **the number changes nothing about the round, and
that is the whole event.**

---

## 2. WHAT THE FACETS DELIBERATELY DO NOT REWARD

- **A thesis in the chorus.** The chorus is this pair's **FORBIDDEN SECTION**. A candidate that lands its
  point there scores zero on facet 5 regardless of craft.
- **Address.** No second person anywhere; nobody is spoken to. On this subject address is a lecture.
- **A collective pronoun.** The Cohort Gate is a floor, not a facet — a candidate carrying `we`/`us`/`our`
  is not scored, it is repaired.
- **A villain.** Ruling 1 binds: the offence is **abdication**, not loss. Nobody in this song is evil;
  somebody declined to decide. A candidate that finds a culprit has left the run.
- **Astonishment from the narrator.** R-e: the miracle is ordinary to the singer. Amazement is the
  listener's job.
- **Self-description as a made thing.** R-c, the Self-Pity Gate: she is a person with hands in a place.

---

## 3. JSON (step contract format)

```json
{ "facets": [
  "THE LEVEL HOLDS, AND THE LATCH IS THE CEILING - the piece stays at speaking volume start to finish with a metal cabinet latch as the loudest object, so paper, pen nib and breath stay audible under it; level is a consequence of the object list, never a polite request, and no later section arrives bigger than an earlier one.",
  "THE SUBSTITUTED DOWNBEAT - a cabinet latch is audibly present on the grid in every bar exactly where the kick belongs, recorded dry and close enough to read as wood and metal rather than a sample; the replacement is scored, never the absence, because holes smooth away in the render.",
  "THE FLAT THAT BECOMES THE KEY - the lead sits a consistent fraction under pitch and stays there uncorrected long enough that the ear re-hears the flatness as the tuning, so that by the last chorus an in-tune vocal would be the wrong sound.",
  "SPRUNG STRESS AGAINST A GRID THAT NEVER YIELDS - vocal stresses keep landing off the pulse, early and late and on weak syllables, at a continuous rate rather than as an occasional push, while the machine refuses to follow and the friction stays small, constant and physical.",
  "THE FACT IS ANSWERED BY A HAND, NOT EXPLAINED BY A NARRATOR - thirty femtoseconds arrives once at the emotional hinge, spelled in words, printed on an object physically present in the scene, and is met by a person doing something with her hands, with no explanatory register anywhere and no change to the round."
] }
```

---

## 4. PROVENANCE

Step file: `skills/music/steps/06_Generate_Music_Facets.md` (phase 6 — facets for prompt generation).
Inputs: the frozen ICB, `pair_05_step05_isolated_brief.md`, `06_music_handoff.md` §3–§5,
`skills/lofn-core/refs/EMOTION_TAXONOMY.md`.
Panel Ledger used as supplied — 18 voices, three Hyper-Skeptics, no new panel invented. The seats that
shaped these facets: **the Glass-seat** (state the operation — every facet is a verb something does), **the
Burial-seat** (the gap is a substitution, facet 2), **the Boards-of-Canada-seat** (a fraction under and
never correcting, facet 3), **the Rodrigo-seat** (name the object on the bench — facet 5 requires the
number to be printed on a thing), and **the Cohen-seat** (the Cohort Gate, held as a floor rather than a
facet so it cannot be traded off against craft).
**Self-critique, one line:** facets 1 and 3 both defend *restraint*, and a rubric with two restraint
criteria can quietly reward a song that does nothing — so facet 4 is written as a **rate** and facet 2 as a
**presence**, giving the rubric two positive obligations that a merely-quiet candidate cannot satisfy.
