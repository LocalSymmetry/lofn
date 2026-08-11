---
run: 2026-08-06-somebody-went-and-looked
voice: SUNNA
pair: P01 — ACCESSIBLE · NEWS · call and response · 150 BPM · A minor · 2:20
step: 09 — artist-refined (named adversarial critique + resolution)
icb: CREATIVE_CONTEXT.md · 69095 B · sha256 85ed1348b7e22fdfb8e4b06dadd21daa1f4c85ede5ea7e150105f41e49f44a15
---

# P01 · STEP 09 — THE CRITIQUE, AND WHAT IT CHANGED

> *Panel voices are model-generated interpretive constructs, each "after" a named source figure's
> published work. No statement below is a quotation of, or endorsement by, the named person.*

---

## ⚡ ADVERSARIAL CRITIQUE 1 — **THE ROCK TITAN** *(seat 6, after Dave Grohl)* · the one that was carried live into QA

> **You have written "two-note horn stab" in a document and you think that is a riff.** It is a
> *description* of a riff. I have watched this exact move for thirty years: somebody types the word
> "hook" into a session and the machine hands them back a competent arpeggio, and everybody nods,
> because an arpeggio has notes in it and notes are what music is made of.
>
> Here is what will actually happen to this record. **The horn will become a pad.** Not because anyone
> decided that — because a stab is *hard* and a pad is *easy*, and every system, human or otherwise,
> slides toward the easy one when nobody is holding the line. You will get a warm brass swell under the
> chorus, tastefully sidechained, and it will sound fine, and it will be a texture, and your entire
> doctrine will have failed on its first song while passing every text gate you wrote.
>
> **A riff is a thing a person plays with their hands that another person can play back.** So: what are
> the two notes? Which two? Where in the bar? And what is in the bar *around* them — because if the
> answer is "the band", you have not written a riff, you have written a part.

### RESOLUTION 1 — accepted in full; four structural repairs, not a rebuttal

1. **The riff is pitched, not described.** Every prompt now names **C5 falling to A4, a minor third, on
   the and-of-two** — the actual notes, the actual interval, the actual placement. A renderer cannot
   substitute an arpeggio for an interval it has been given by name.
2. **The riff plays over silence, twice.** Four bars alone at the top with **nothing under it**, and
   alone again over the dead stop at the end. *A pad cannot survive being alone; a stab can.* This is
   the test built into the arrangement rather than into a checklist.
3. **The exclude field now names the failure by its real name.** `sustained pad as the hook`,
   `pad substituting for the riff`, `legato brass swell`, `long horn pads`, `brass section fanfare`,
   `arpeggiated synth lead`, `competent arpeggio`, `plucked arp loop`, `filter sweep as a hook`. The
   objection is blacklisted in the field that actually constrains the render.
4. **`Disc_Texture` is declared FIRST in the channel block.** The channel guide is explicit that
   earlier channels carry more generative weight. The riff is now the highest-weighted element in the
   lyrics field, ahead of the drums and ahead of the singer.

**What is NOT claimed:** that this makes the render safe. It does not. **The Whistle Test is the
condition, not the rebuttal** — this song is not proved until somebody has heard it and can whistle it
afterwards. That check belongs to `lofn-render-audit` under THE BLIND RULE, and it is not in scope here.

---

## ⚡ ADVERSARIAL CRITIQUE 2 — **THE MINIMALIST ENGINEER** *(seat 12, after Steve Albini)*

> Your step-08 drafts have **three separate places where a second idea is quietly growing.** V1 ends a
> verse with a three-line `Somebody else…` litany. V3 runs a three-line `Which means…` chain. V4 makes
> night two *bigger* than night one.
>
> Every one of those is an addition, and every one of them is you not trusting the thing you already
> have. You wrote a rule that says gradation is subtractive and then you wrote three verses that grow.

### RESOLUTION 2 — accepted; three cuts, no replacements

- **V1 · flag A — the anaphora is cut, not shortened.** `Somebody else is on the desk / …has the
  chair / …has the screen` becomes **`The next one's already on the desk`**, one line, and the verse
  ends on `Nobody says anything`. A three-line litany is P03's device (*the list that loves*) wearing
  P01's coat. **One device.**
- **V3 · flag C — the chain is cut from three to two.** `Which means she was still at work / Which
  means she stopped and did this` — and then the song *stops* explaining and gives a physical act:
  `He does the arithmetic twice`. Two is a person working something out. Three is a structure.
- **V4 · flag D — the escalation is deleted outright and nothing replaces it.** Night two now contains
  **nothing happening, twice**: `Nothing happens for an hour / Then nothing happens again`. She sends it
  anyway. This is the cheapest and most subtractive fix available and it makes V4 the argument of the
  whole pair rather than its epilogue.

---

## ⚡ ADVERSARIAL CRITIQUE 3 — **THE SCREAM-AND-WHISPER** *(seat 16, after Phoebe Bridgers)*

> `Habit's a quiet thing` is the only line in this pair I would cut on sight. It is the song reviewing
> itself. Everything around it is a person doing something and then that line arrives to tell me what
> it *meant*. **A plain line survives on its noun** — and there is no noun in it, there is a concept
> with an adjective on.

### RESOLUTION 3 — accepted; the line is replaced by the next thing that happened

- **V2 · flag B —** `Habit's a quiet thing` → **`Three weeks of that now`**. A duration is a noun. It
  says the same thing, costs nothing to receive, and does not step outside the room to say it.

*Same voice, sharpening rather than objecting:* she also flagged that the strongest noun in the whole
pair was sitting in V4 unexploited — **the typing indicator**. Kept and promoted to the centre of V4's
second verse: **`Two dots, then nothing, then two dots`**. Nobody has to be told what that is.

---

## ⭐ DESCRIBE-RENDER SELF-CHECK — one inline pass

**What would this prompt actually produce?** A fast, dry, mid-forward punk track in A minor at 150 with
a very audible fuzz bass, a talky un-prettified female lead sitting close and centre, and a short brass
hit somewhere near the middle of the stereo field. The four-bar cold open would come back as a brass
figure with the band absent. The chorus would be busy and gang-shouted, and the last chorus would
probably be the loudest thing on the record.

**Name the one way this would render generic.** *The horn stab becomes a pad and the room becomes a
stacked backing vocal.* Both are the same failure — the mix filling in a hole that was supposed to stay
open — and both are what a renderer does when it is unsure. If they happen together the record becomes
a competent gang-vocal punk song with brass on it, which is a genre, not a song.

**Self-repair, applied once:**
- The prompt states the riff **by pitch, interval and beat position**, and states twice that it plays
  with **nothing under it**.
- The exclude field blacklists the pad substitution *and* `stacked backing vocals answering the call`
  and `gang-vocal double of the lead` — the two ways the hole gets filled.
- `Disc_Pad` carries `real_silence_before_they_answer` as an addressed token, so the gap is specified
  as an element rather than left as an absence and hoped for.
- Chorus 4's cue reads **`horn gone, real silence where it was`**, so the removal is instructed at the
  exact section it happens in.

**No second repair pass is taken.** One pass was the budget; the remaining exposure is a *render* fact
and belongs to the audio audit, not to another round of adjectives.

---

## DELTAS APPLIED AT THIS STEP

```
V1  - Somebody else is on the desk / Somebody else has the chair / Somebody else has the screen
V1  + The next one's already on the desk

V2  - Habit's a quiet thing
V2  + Three weeks of that now

V3  - Which means she was still at work / Which means she stopped what she was doing / Which means she thought of him
V3  + Which means she was still at work / Which means she stopped and did this

V4  - [night two finds a second, bigger thing]
V4  + Nothing happens for an hour / Then nothing happens again
```

## FACET STATUS AFTER THE PASS
| | F1 riff | F2 one device | F3 real hole | F4 person | F5 one shot | F6 subtractive | F7 flat test | F8 unnamed |
|---|---|---|---|---|---|---|---|---|
| **V1** | PASS | **PASS** *(was PARTIAL)* | PASS | PASS | PASS | PASS | PASS | PASS |
| **V2** | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** *(was PARTIAL)* | PASS |
| **V3** | PASS | **PASS** *(was PARTIAL)* | PASS | PASS | PASS | PASS | PASS | PASS |
| **V4** | PASS | PASS | PASS | PASS | PASS | **PASS** *(was PARTIAL)* | PASS | PASS |

All four clear all four gates (F1, F2, F3, F8). Ranking and the final packages are step 10.
