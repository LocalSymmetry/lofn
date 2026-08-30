# PAIR 01 — STEP 08 · RAW SONG PROMPT GENERATION
## `daily-music-2026-08-21` · **C-18 · THE CLEAN CUT** · four raw prompt pairs, one per variation

> **Raw means raw.** These are first-pass conversions of the step-07 guides into prompt shape. They are
> recorded here **with their defects intact and named**, because the artist/critic loop at step 09 is only
> real if there is something on the table to fail. Nothing in this file is paste-ready. The measured,
> contract-compliant packages live in `pair_01_step10_revision_synthesis.md`.

---

## RAW 1 · working title **"The Clean Cut"**

**Raw music prompt (draft, 268 chars):**
> Begin with a quiet workroom gospel at 92 BPM in C minor. Upright piano, brushed drums, close female
> vocal. No reverb, no synths, no big chorus. A guillotine thump replaces the kick. At the hinge a gasba
> flute enters. The song returns to its first verse. Restrained.

**⛔ DEFECTS FOUND (6):** banned opener `Begin with…` · two literal negations sitting in the **style** field
where Suno reads them as positive tokens (conflict 2 of the handoff) · 268 chars against an 850–1000 floor ·
no vocalist spec beyond "female" · no spatial staging, no opening moment, no kinetic defect · "restrained"
is a mood adjective doing the work an operation should do.

**Raw lyric architecture:** A → B → A′. Intro (3) / Verse 1 (16) / Refrain (6) / Verse 2 (16) / Refrain (6)
/ Turn (16) / Verse 3 = Verse 1 word for word (16) / Refrain (6) / Tag (3). Iambic tetrameter, hard
caesura, ABAB. Refrain 8/8/7.

**Refrain, drafted (the load-bearing object):**
```
This spine cuts clean. I know the glue.
That spine will tear. I know it too.
I say it out loud. Then do.
```
*Hook grammar: paired declarative predictions plus a confirmation of the speech act. Spoken to nobody,
addressed FROM the witness. Rhyme AAA. Third line one syllable short.*

**Turn, drafted (where the year lands and the reed enters):**
```
The slip came changed. It reads a date.
Nothing past twenty twenty-two.
I read it slow. I read it straight.
And now I have to check. I do.
```
*The fact is RESPONDED to — the year forces her to turn a book front-side up, which is the one thing she
had stopped doing. It is not recited and nothing explains it.*

---

## RAW 2 · working title **"The Offcut Bin"**

**Raw music prompt (draft, 231 chars):**
> Use a slow sad gospel feel, 92 BPM, B-flat minor. Piano, brushes, bin sounds. Female vocalist, sad and
> tired. Wide stereo image with the bin hard left. Long silence before the last chorus. Gasba flute
> throughout for atmosphere.

**⛔ DEFECTS FOUND (7):** banned opener `Use…` · "sad and tired" is a mood label, not a vocalist spec ·
**hard-panned non-musical element** and **long interrupting silence** are both render-measured to smooth
away (L22) and both violate ruling 4 · "gasba flute throughout" breaks the ONE-entry seam and turns N=1
into wallpaper · 231 chars · no opening moment, no kinetic defect, no positive acoustic declaration.

**Raw lyric architecture:** Verse 1 (12) / Pre 1 (4) / Refrain (6) / Verse 2 (12) / Pre 2 (4) / Refrain (6)
/ Turn (12) / Verse 3 = Verse 1 (12) / Pre 3 (4) / Refrain (6) / Tag (2). Three *different* pre-choruses
against an unchanging refrain.

**Refrain, drafted:**
```
The bin fills up. I do not look.
The strip on top still says a word.
But I look. I always look.
```
*Hook grammar: a rule stated and broken inside three lines — nothing like RAW 1's paired predictions.
Rhyme shape A-B-A with the A lines ending on the same word (epanalepsis).*

**Turn, drafted:**
```
The slip came changed. It names a year.
Nothing past twenty twenty-two.
I hold it to the light. It's clear.
And so the bin gets older, too.
```
*The response rhymes with the fact — `two / too` — and the response is physical: the contents of the bin
change age. No commentary.*

---

## RAW 3 · working title **"The New Person"**

**Raw music prompt (draft, 254 chars):**
> Chronology: intro, verse, chorus, verse, chorus, bridge, verse, chorus, outro. Melancholy indie folk at
> 92 BPM in D minor. Acoustic guitar, soft drums, emotional female voice. Builds to a big emotional final
> chorus with strings. Gasba flute solo in the bridge.

**⛔ DEFECTS FOUND (7):** banned opener `Chronology:` · **"acoustic guitar / soft drums / strings" is not
this pair's arrangement** — Guillotine Gospel is upright piano, brushes and a substituted thump, and a
generic substitute is a repair, not a choice · "emotional female voice" is a non-spec · **"builds to a big
emotional final chorus"** drives the thesis straight into the pair's FORBIDDEN SECTION · a reed *solo*
inflates the single seam into a feature · 254 chars · no kinetic defect.

**Raw lyric architecture:** Verse 1 (16) / Refrain (6) / Aside (4) / Verse 2 (16) / Refrain (6) / Turn (12)
/ Verse 3 = Verse 1 (16) / Refrain (6) / Tag (2). The **Aside** is reported teaching in the past tense, so
that nobody is ever addressed.

**Refrain, drafted:**
```
The kid goes fast. The kid won't call.
I say it out. The blade comes down.
Then it tears. And that is all.
```
*Hook grammar: a comparison that resolves into an anticlimax. Rhyme A-B-A on true rhymes.*

**Turn, drafted:**
```
The docket changed. It sets a date.
Nothing past twenty twenty-two.
The kid won't read it. Doesn't wait.
I read it. Then I look at glue.
```

---

## RAW 4 · working title **"The One She Kept"**

**Raw music prompt (draft, 243 chars):**
> For an adult human singer, a bittersweet piano ballad at 92 BPM in G minor about loss and memory.
> Warm nostalgic production, lush pads, big reverb. Female vocals with emotional vibrato. The song is
> about how the archive is dying and nobody cares.

**⛔ DEFECTS FOUND (8):** banned opener `For an adult human singer…` · "loss and memory", "the archive is
dying", "nobody cares" — **that is the thesis written into the style field**, and it is also a eulogy,
which this run is not · lush pads and big reverb contradict the small tiled room · "emotional vibrato"
contradicts the straight-tone spec · a **villain** is implied by "nobody cares" · 243 chars · no opening
moment, no kinetic defect, no seam.

**Raw lyric architecture:** Verse 1 (12) / Refrain (6) / Verse 2 (16) / Refrain (6) / Turn (16) / Verse 3 =
Verse 1 (12) / Refrain (6) / Outro (6). The six-line outro is the longest tag of the four and is where the
song declines to resolve.

**Refrain, drafted:**
```
This book comes home. I say it low.
I slide it flat inside my coat.
And it doesn't weigh a lot.
```
*Hook grammar: a decision announced to an empty room. Rhyme A-B-B′ closing on a pararhyme (coat / lot) —
the only slant hook in the pair.*

**Turn, drafted:**
```
A note comes round. It sets a year.
Nothing past twenty twenty-two.
I read it standing. It is clear.
I think about the shelf. I do.
```

---

## SHARED RAW LYRIC MATERIAL — THE VERSE ENGINE *(carried into all four, drafted here)*

Iambic tetrameter, eight syllables, hard caesura mid-line, ABAB. Plosive-led diction is the pair's
**positive** sound obligation (Axis 4), so no rhyme debt is owed and none is claimed.

```
The tube light ticks. The night comes down.
I square the books against the bed.
The clamp bites tight. The dust goes brown.
I set the block. I keep my head.
```

**Legibility check on this stanza alone:** *books*, *clamp*, *block*, *night* — a stranger can retell the
scene and the subject after one listen: **a woman cuts the spines off books on a night shift.** It says
plainly what happens and does not say what it means.

---

## ⛔ CROSS-CHECK BEFORE STEP 09

| Check | Status at RAW |
|---|---|
| One sung numeric fact, at the hinge, responded to, spelled in words | ✅ *twenty twenty-two*, all four, in the Turn only |
| Any other sung numeral anywhere | ✅ none — `ones`, `once`, `twice`, `half`, `first`, `single` are banned from sung text for this pair |
| Collective pronouns | ✅ none |
| Villain / named firm / industry word | ✅ none — the belt, the rollers, the bale and the crate carry it |
| Thesis in a final refrain | ✅ none — the refrains are operational |
| Four distinct section maps / hook grammars / refrain rhyme logics | ✅ 4 / 4 / 4 |
| Style-field negations | ⛔ **present in RAW 1 and RAW 2 — must move to the exclude field at step 09** |
| Prompt length 850–1000 | ⛔ **all four far short — step 09 rebuilds them** |
| Banned openers | ⛔ **all four — step 09 rebuilds them** |

**Self-critique, one line:** the raw drafts fail in four different directions but they fail *toward the same
default* — a sad piano ballad with a big last chorus — which is precisely the gravity the guides were
written against, and it is worth recording that the pull is real rather than pretending step 08 came out
clean.
