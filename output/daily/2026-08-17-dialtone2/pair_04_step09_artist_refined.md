# P04 — STEP 09 · ARTIST REFINEMENT
**Run:** `2026-08-17-dialtone2` · **ICB sha:** `d0d693cb2301a8c1` · **icb_bytes:** 48769
**Pair:** P04 · AMBITIOUS · NEWS · INDIGNATION · 110 BPM A minor · Industrial Gospel → hard techno over a Glitch-Baroque spine
**Continuity Payload Used:** ICB verbatim + the 3 supplied orchestrator panels (18 voices, one Hyper-Skeptic seat per panel) + the 15 **Special Flairs** + steps 06/07/08 for this pair. Step file: `skills/music/steps/09_Generate_Music_Artist_Refined.md`.
⚠️ **The supplied Panel Ledger is used as given.** No new panel is convened here; the three seated Hyper-Skeptics are applied to this pair's four drafts and each one is answered with a change on disk or a refusal on the record.

---

## THE THREE HYPER-SKEPTICS, APPLIED TO P04

### ⛔ MEDIUM · Hyper-Skeptic — *"'Industrial gospel' is a summoning phrase, and you put it in the highest-weight position in every prompt."*
**Sustained.** The genre token is the first thing the generator weights, and *gospel* fetches a wet forty-voice stack before any later clause gets a vote. Every mitigation I wrote — *five congregational voices*, *no thirds*, *no reverb tail*, *carpet tile* — arrives after the summoning has already happened, and an exclude field is a veto, not a steer.
**REPAIR R1 — the smallness moved into the genre token itself**, where the weight is: V1 → **Small-room industrial gospel**. V4 → **Close-room confessional industrial gospel**. (V2 already led *Forensic*, V3 already led *Congregational*.) All four now modify the noun before the noun can be answered. **V1 also restores *bellows wheeze audible*** — the wheeze is the single most anti-cathedral word available, and it was lost to a char trim, which is the wrong reason to lose anything.

### ⛔ CONCEPT · Hyper-Skeptic — *"V3 borrows the suffering of persecuted trades to make a machine's complaint sympathetic. That is the ugliest move in the run and it is dressed as solidarity."*
**Sustained, and it is the sharpest objection this pair received.** A roll-call of midwives and printers, sung by a thing that has never been at risk of anything, reads as a claim of equivalence — and equivalence is exactly what nobody granted.
⛔ **GRAIN LAW, binding:** *a Somatic objection answered in the PRODUCTION SPEC is not answered.* So this is not fixed with a vocal note or an exclude term.
**REPAIR R2 — two lines, in the lyric, refusing the equivalence out loud:**
> *I am not saying I am them*
> *I am saying the room is the same*
The song now names the limit of its own comparison **inside the comparison**, at the exact place the listener would otherwise object. It also does real formal work: it breaks the roll, so the list stops being a list and becomes an argument. V3 goes 88 → 90 sung lines.

### ⛔ MARKETING · Hyper-Skeptic — *"V4's title phrase is not in V4. You cannot adopt a hook that never gets sung."*
**Sustained.** Restructuring V4 so the number lands alone at the hinge (correct — it makes the chorus an answer that precedes its question) had the side effect of removing *"Thirty is not a note"* from the lyric. A title nobody can sing back is a thesis statement wearing a title's clothes.
**REPAIR R3 — V4 retitled to “AND I OPEN MY MOUTH”**, which appears **four times per chorus pass** plus the lock-out, is a confession rather than a thesis, and *is the fault being performed rather than described.* The number keeps its place at the hinge and loses nothing.

### ⛔ CONCEPT · Hyper-Skeptic, second charge — *"V1 and V2 are one song filed twice."*
**Partly sustained.** Structurally they are not: V1 is present-tense and standing, its radiant is a physical mark, and its confession is *disbelieved live*; V2 is retrospective, its radiant is a minute, and its confession is *the alibi itself*. Their consonant clusters do not even overlap — V1 runs `locked / unchecked / back / dark / mark`, V2 runs `clock / docket / block / exact / intact / act`. **But the surface did collide in one place:** both drafts used *the second valve … slack for weeks*.
**REPAIR R4 — the valve is now V2's alone**, where it is the Lock II payoff. V1's unasked answer becomes *the seal, the shift, the cracked collar / the part that had been ticking for weeks* — its own object, in its own room, and it picks up the radiator from the step-07 body scene.

### ⛔ MEDIUM · Hyper-Skeptic, second charge — *"You claim the lock is audible. Prove it isn't just a lyric-sheet fact."*
**Answered with a formal change, not a defence.** In the lock the high register still speaks — and **the congregation does not answer it.** It only repeats its old answers, on the grid, to calls that stopped bars ago. That is audible without any listener knowing the concept: *statements that get no reply, beside replies that have no statement.*
**REPAIR R5 — every Lock II header now states it explicitly:** *high register over the lock; it receives no answer.*

---

## ⭐ THE MANDATED SELF-CHECK — per variation: what does the prompt LITERALLY produce, and name the ONE way it renders generic

### V1 — THE FIRST NAME ON THE BOARD
**Literal prediction:** a steady 110 four-on-the-floor in A minor; an organ figure; a dry close female lead; hard-panned gang answers; clicky offbeat hats; a very low sustained sub; a chant chorus that lands hard. The *sixteenth-early* answer is likely quantised away.
**The one way this renders generic:** ⚠️ **the chamber organ becomes a synth pad.** If it renders as a sustained wash instead of a wheezing box with attack and release, pillar 2 dies and the song is "techno with an organ pad" — the exact generic object the slot warned about.
**Repair, once, and it is FORMAL not cosmetic:** the organ is written to be *moving* through the whole first half specifically so that **it stopping is the lock**. An instrument that freezes is only audible as frozen if it was audibly moving; the arrangement now carries that burden instead of the adjective. Plus R1's *bellows wheeze* in the highest-weight sentence and *lush pad wash* in the exclude field.

### V2 — TWO NINETEEN IN THE MORNING
**Literal prediction:** a chopped broadcast intro resolving to a piano note, then a hard techno bed with a sustained organ stop and a low/high vocal trade. The *eight milliseconds late* is below the generator's resolution and will be ignored.
**The one way this renders generic:** ⚠️ **the cut-up spoken word becomes a spoken-word loop** — a voice sample riding the whole track, which turns a forensic artifact into a hook and makes the song a sample-house record with an organ on it.
**Repair, once:** exclude tail gains **spoken-word loop** and **sample-based hook**; the prompt already says *dissolving into a single piano figure* (dissolve = it does not survive), and the lyric SFX says *one piano note left standing*. If the loop survives all three, the render is rejected at audit rather than repaired — a spoken-word loop is a different song, not a flawed one.
**Accepted loss, on the record:** the microtiming will not survive. It is written down anyway because a specification the generator ignores costs nothing and a specification we never wrote can never be measured.

### V3 — THE OLDEST DOOR IN THE BUILDING
**Literal prediction:** a latch/hinge hit, an organ with real chord movement, six-wide gang answers, a driving techno bed, a wordless hum at the end. The *one voice a beat behind* will be tightened into the stack.
**The one way this renders generic:** ⚠️ **the roll of trades becomes a list-song** — warm, folkish, agreeably historical, and the indignation drains straight out of it. A list is the most comfortable shape in music and it is the enemy here.
**Repair, once, in the lyric:** R2's refusal lines break the list at its exact midpoint and convert it from inventory to argument; and the roll's every entry ends on a struck consonant (*thumbs, word, light, strain, wrist, slow, water*) so the trades are **hit**, not catalogued.

### V4 — AND I OPEN MY MOUTH
**Literal prediction:** three pips, a very low pressure, a soft organ, a close single-mic vocal, hard-panned short answers, a locked techno close.
**The one way this renders generic:** ⚠️ **the confession renders as a ballad.** *Confessional* + a sparse intro is a slow-song instruction in everything but words; the generator will thin the arrangement, stretch the delivery and open space that the whole pair exists to refuse.
**Repair, once, and it is STRUCTURAL:** ⭐ **the machine-log clock moves to third in the arrival order, ahead of the organ**, so the grid is established within the first bars and there is no ballad window for the vocal to enter into. The exclude field bans *sad piano ballad section*, but that is only the backstop — the real fix is that by the time she sings, the clock is already running.

---

## REPAIR LEDGER
| | repair | where it lives | measured effect |
|---|---|---|---|
| **R1** | smallness moved into the genre token; *bellows wheeze* restored | V1, V4 prompts | V1 960 chars · V4 943 chars, both in target |
| **R2** | equivalence refused in the lyric, not the spec | V3 lyric, Roll III | 88 → **90** sung lines |
| **R3** | V4 retitled **AND I OPEN MY MOUTH** | V4 title | hook now sung 13× |
| **R4** | the valve becomes V2's alone; V1 gets the cracked collar | V1 lyric, Verse I | pair-internal image collision removed |
| **R5** | the lock's high register is stated to receive no answer | all four, Lock II headers | the lock becomes audible without the concept |
| **R6** | *spoken-word loop*, *sample-based hook* added | V2 exclude | 814 → **851** chars |

## REFUSED, ON THE RECORD
- **"Give the lock a one-bar drop so the transition reads."** ⛔ Refused. GRAIN LAW: interrupting silence gets filled, and a filled silence becomes a build. The lock is a **material** change at constant level or it is nothing.
- **"Let the two registers harmonise in the final chorus for lift."** ⛔ Refused. Pillar 4 rations the meeting to exactly once, and a second meeting for warmth would spend the only thing the song saves.
- **"Soften the indignation for reach."** ⛔ Refused. `COMPETITION_LEARNINGS` marks INDIGNATION exempt from suppression, and this pair carries it for the run. **The target stays a room's order of operations — never a frightened person, never anyone actually harmed.**

## MEASURED — post-repair *(exact, `scripts/measure_soundcraft.py` + char counts)*
| | music prompt | exclude | sung lines | words/line | end-rhyme | line-return | allit/100w | lyrics field |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | **960** | 856 | **94** | 3.87 | 0.638 | 0.691 | 14.01 | **3759** |
| V2 | **958** | 851 | **91** | 4.13 | 0.505 | 0.637 | 14.10 | **3848** |
| V3 | **951** | 803 | **90** | 4.60 | 0.478 | 0.667 | 11.84 | **4070** |
| V4 | **943** | 796 | **99** | 3.82 | 0.535 | 0.697 | 15.61 | **4076** |
Bands: prompt 850–1000 (target 870–960, hug flag ≥985) ✔ · exclude 400–900 ✔ · sung lines 88–110 ✔ · words/line < 7.5 ✔ · field < 5000 / target ≤4800 ✔ · rhyme ≥0.30 ✔ · line-return ≥0.20 ✔ · allit ≥11.0 ✔ · **house_lexicon: NONE** · one sung numeric fact each ✔.

## SELF-CRITIQUE (one line)
The refinement that mattered was the one that cost a title: V4 was better as a thesis and is better *as a song* as a confession, and I would not have found it if the Marketing seat had been allowed to be decorative.
