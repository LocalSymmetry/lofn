# PAIR 05 — STEP 09 · ARTIST REFINEMENT
## `daily-music-2026-08-22-sunna-offbeat` · SLOT P05 · four variations

**The artist pass over `pair_05_step08_generation.md`.** Every change below is recorded with the line it
replaced, the gate that caught it, and what the change cost or bought. ⛔ Nothing here is a preference;
each entry names **THE FLAT TEST**, **THE HUM TEST** or **THE REMARK TEST**, and the two structural
sweeps this slot owns — **DISTINCTNESS ACROSS THE FOUR** and **THE ONE-FACT RULE**.

**Measured before refinement** (per variation, counted in code, not by eye):
prompt 962 / 938 / 942 / 962 · exclude 904 / 785 / 812 / 784 · lyrics field **4991 / 4865 / 5088 / 5065**
· sung **82** · spoken tercets **9** · tannoy **1** · validator raw line count **92**.

⚠️ **Two of the four draft lyrics fields were over the 5,000 hard render cap and all four were over the
4,800 target.** That is a real defect, not a rounding note: a lyrics field over cap does not render.
Section 5 below is the repair, and it is a **container** repair — no sung line was deleted to make room.

---

## 1. V1 — *Say It Back To Me*

**① THE FLAT TEST — the opening quatrain had no rhyme and no content.**
*Was:* `The one who reads the pump and not the screen. / The one who counts the sharps and counts them twice. / The one who reads the label out in full. / The one who says it twice and says the same.`
*Now:* the fourth line is **`The one who tells you twice without advice.`** and the label item moves down
into the third quatrain.
**Why:** *says it twice and says the same* is a sentence that survives being read but dies being said —
it is short, not simple, and it rhymes on nothing. *Tells you twice without advice* is one specific,
recognisable act of a good colleague: the repeat without the lecture. **Bought:** an ABAB the quatrain
did not have (*screen / mean*, *twice / advice*), and a competence the draft did not contain.

**② THE FLAT TEST — an ambiguous item of praise.**
*Was:* `The one who left the good tape on the shelf.`
*Now:* **`The one who put the good tape back on the shelf.`**
**Why:** the draft reads as *left it behind*. The praise intended is *restocked it for the next person* —
the smallest generosity in the building. One word changes it from a hint into a fact.

**③ Container — the last line of the fourth verse ran long against a 100 BPM bar.**
*Was:* `I start at the top and I don't miss one thing in between.` (57 characters)
*Now:* **`I start at the top. I don't miss one in between.`** (47)
**Why:** at 100 BPM a line of that length forces the singer to crowd the bar, and crowding is the one
thing that would make the handover sound panicked rather than fast. Same meaning, one breath.

**④ THE HUM TEST — moved from prose into the field the model reads.**
The step 07 guide says the bass states the hook melody before the vocal enters. That instruction lived
only in the guide, which the generator never sees. It is now in the **music prompt**, in words.
**Why:** ⛔ answering a claim only in the production spec is not answering it, and answering it only in a
document the model cannot read is worse.

---

## 2. V2 — *Whoever Signed It*

**① SUNNA'S FIRST REFUSAL — a joke that required a butt.**
*Was:* `Whoever went back up and had a shower.`
*Now:* **`Whoever went back up and got a coffee. Fine.`**
**Why:** the shower line is contempt wearing a joke, and it also breaks the couplet rhyme. The coffee
line is flatter, funnier and **worse for the people it describes**, because it declines to caricature
them: they went and got a coffee, which is exactly what happened, and *Fine.* is the whole verdict.
**Bought:** the *time / Fine* rhyme the quatrain was missing.

**② THE FLAT TEST — a word that was only there to rhyme.**
*Was:* `It's the second time somebody checks and stands.`
*Now:* **`It's the second time that anybody checks.`**
**Why:** *and stands* does nothing but reach for *hands*. This is the thesis quatrain of the whole
variation — the sentence that got a person removed, said by somebody nobody is watching — and it has to
be the plainest line in the song. ⭐ **The rhyme is deliberately not repaired.** One unrhymed quatrain in
twelve, at the exact point where plainness is the content, and the return debt is already paid three
times over by a byte-identical chorus and forty-eight lines of anaphora.

**③ DISTINCTNESS — an item that belonged to another variation.**
*Was:* `the pump that under-reads, the door, the sink`
*Now:* **`the cuff that reads too high, the door, the sink`**
**Why:** V1 opens on *reads the pump and not the screen*. Two songs praising the same machine is the
start of writing one song four times.

---

## 3. V3 — *Twelve, Fourteen*

**① THE ONE-FACT RULE — a second number, and not the interesting one.**
*Was:* `Sixty-one steps to the top. I have done it.`
*Now:* **`The lift says coming. The lift has said it all year.`**
**Why:** this song is allowed exactly one number and it is *the thirteenth floor*. A step count competes
with it and loses. The replacement does the same job — she is not waiting — with a fact about the
building instead of a fact about her legs.

**② THE FLAT TEST — the one heroic word in a technical song.**
*Was:* `The one on twelve has never lost a war`
*Now:* **`The one on twelve has never lost a set`** with `and never once has left one on the floor.`
**Why:** ⛔ nothing in this pair is allowed to praise anybody for heroism. *Never lost a set* is stock
control, which is what the praise actually is, and it rhymes properly against *corridor*.

**③ DISTINCTNESS — the sung fact was arriving the same way in three songs.**
*Was:* `She said the thirteenth floor. I said there's none. / She said: you're on it. Look down at your feet.`
*Now:* **`The thirteenth floor is where I'm standing now. / She told me that in March and I said no. / The panel skipped a number, not a floor. / I've been up here all year. I ought to know.`**
**Why:** V1, V3 and V4 were all answering the fact with *you're standing on it*. Now V3 answers it with
**her own authority** — she has worked the landing for a year — which is the only one of the four
responses that is a claim of competence rather than a correction.

**④ DISTINCTNESS — two wolf lines opened identically.**
*Was:* `On the Tuesday they came up and walked her down.`
*Now:* **`They walked her down the whole of it on the Tuesday.`**
**Why:** V1's wolf begins *On the Tuesday they came down…*. Two of four wolves sharing an opening clause
is the formula the run's third falsification test exists to catch, and it is cheaper to fix here than to
have QA find it.

---

## 4. V4 — *It Still Opens Everything*

**① THE REMARK TEST and the slot's named failure mode — adolescent affect in the chorus.**
*Was:* `It opens every door in here and I don't care.` … `I've got a whole shift in my head and time to spare.`
*Now:* **`It opens all of it. It isn't mine to own.`** … **`It isn't a principle. It's just how I was shown.`**
**Why:** ⛔ *and I don't care* turns a practice into a pose, and a pose is exactly what this bed drags a
song toward. The refusal is not defiance; it is **a habit somebody taught her**, and saying so is both
truer and colder. ⭐ It also moves the song's meaning out of attitude and into inheritance, which is the
subject. **Bought:** *own / shown*, a clean rhyme where the draft had *care / spare*.

**② THE FLAT TEST — a line that read as an actual key.**
*Was:* `They took the person. They forgot the key.`
*Now:* **`They took the person and they left the key.`**
**Why:** *forgot* invites the listener to picture a key ring. *Left* states the administrative fact:
the permissions were never revoked. The indignation in this song is procedural, and it needs the verb
that says so.

**③ DISTINCTNESS — the sung fact, and the closing image.**
The fact now arrives **in writing** and is answered **by a gesture**: `She wrote the thirteenth floor
down on a sheet. / I said there isn't one. She didn't fight. / She tapped the floor we both were
standing on, / then went and did the drugs round for the night.` And the final quatrain's sky image
(`The sky over the car park's going white`) is replaced by **`The day shift's in the car park, coming
through.`**, because V3 already owns the sky over that car park.
⭐ The last line becomes **`and a strap in my hand, and one thing I won't do.`** — which is the variation's
whole closing move, said once, plainly, at the end, and never explained.

**④ DISTINCTNESS — the wolf, and a teaching that collided with V2.**
Wolf *was:* `They took the badge off her and then they posted it.` *Now:* **`The bag came back on
Thursday with the badge still live.`** — V1 already owns the badge being taken; V4 owns it coming back.
Teaching *was:* `She taught me: write the hour you really did it.` *Now:* **`She taught me: sign the
thing you actually saw.`** — V2 owns *the true time in the notes*.

---

## 5. THE CONTAINER REPAIR — measured, and it deleted no song

⚠️ **The defect:** all four lyrics fields were over the 4,800 target and **V3 and V4 were over the 5,000
hard cap**. A field over cap does not render, so this was the highest-severity finding of the step.

**⛔ What was NOT done:** no sung line was cut, no verse was shortened, no chorus was touched, no tercet
was dropped. Cutting song to fit a container is how a gate ends up editing the record.

**What was done — three container edits, all outside the sung text:**

| edit | before | after | saved per song |
|---|---|---|---|
| **section headers** compressed to four tight segments each, still `[Section - EMO:… - voice - cue]` | 12 headers averaging **77–81 chars** | 12 headers averaging **~60 chars** | ~215–250 |
| **`[Theme: …]`** trimmed to the situation and nothing else | 172–183 | ~140 | ~35 |
| **`[SONG FORM: …]`** trimmed to form, tempo, key, stanza shape | 145–165 | ~115 | ~35 |

⭐ **One header edit is a content fix, not a container fix:** the third tercet in every variation did not
carry the word **spoken**, which meant the one register instruction that keeps this song from becoming a
two-character duet was missing from the section where the thesis lands. All three tercets now read
`alto spoken, clipped`. That defect was found by counting in code — by eye the headers looked identical.

---

## 6. WHAT THE ARTIST PASS DELIBERATELY LEFT ALONE

- **The byte-identical chorus, three times, in all four.** ⛔ It needs no defence and no note.
- **The anaphora.** Forty-eight lines opening on a function-epithet is the device; thinning it to avoid
  monotony would delete the song's only argument.
- **V1's opening quatrain returning byte-identical in the fourth verse.** Return is free.
- **All four wolves landing in the third verse over a full band.** ⭐ They land in four *different events*
  at four different distances from her — a badge taken at a desk · a rota changed three buildings away ·
  a walk down a stair nobody counted · a bag posted back with the permissions still live. The placement
  is shared on purpose; the events are not interchangeable, which is the property the run's third
  falsification test is measuring.
- **V3's counted floors in the outro.** *Ten. Eleven. Twelve.* is the variation's assigned device, not a
  recitation of research. The song's one sung numeric **fact** remains *the thirteenth floor*, once.
- **The tannoy.** It is one line, in the intro, and it is the building's voice, not a character. Cutting
  it would leave a hole that was never a thing — ⛔ *you cannot drop what was never established.*

---

## 7. POST-REFINEMENT SELF-CHECK — what the artist pass could NOT verify

⛔ **Every instrument used in this step is textual.** Nothing here can hear whether the render puts the
bass in front of the guitars, and that is the single failure this slot was warned about. The music prompt
now says it three separate ways — *bass front of mix*, *a hair louder than the guitars*, *a line you can
hum* — and the exclude field names the failure class outright: **no chorus where the guitars are louder
than the bass**. ⭐ **That is the ceiling of what text can do.** If the render comes back with a guitar
record, the take is re-rolled, not accepted, and the finding belongs in the render audit under THE BLIND
RULE — not in this file, which has heard nothing.
