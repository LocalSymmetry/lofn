# PAIR 02 — STEP 09 · ARTIST-REFINED
## `daily-music-2026-08-22-sunna-offbeat` · SLOT P02 · four variations
**Refinement pass over the step-08 draft. Every change is recorded with the gate that caught it.**

> The three gates were applied **line by line**, not section by section:
> - ⭐ **THE FLAT TEST (words):** say the line out loud, flatly, no music, to a person's face. If it
>   survives it is *simple*; if it dies it was merely *short*.
> - ⭐ **THE HUM TEST (music):** the tune must exist with no words, before the vocal enters and after it
>   leaves. A pad is not a riff.
> - ⭐ **THE REMARK TEST (attention):** would a listener say *"that was clever"*? Then it is a toll —
>   bury it or cut it.
>
> Measurement was done **in code**, not by eye: `scripts/validate_step.py` plus a per-variation counter
> for prompt characters, lyrics-field characters, sung lines, line-ending syllable counts and numeral
> hits. Nothing in this file was judged by looking at it.

---

## 1. WHAT THE COUNTER FOUND IN THE DRAFT, BEFORE ANY TASTE WAS APPLIED

| | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| music prompt chars | 962 | **985** | **984** | 933 |
| lyrics field chars | **5435** | **5185** | **5193** | **5283** |
| sung lines | 84 | 82 | 84 | 84 |
| **numeral hits in sung lines** | 3 | 1 | 6 distinct | 3 |
| **line endings that are not monosyllables** | 0 | 1 | 2 distinct | 2 |

⛔ **Three hard problems, all invisible to the eye and all found by counting:**

1. **Every lyrics field was over the 5000 cap** — 5435 / 5185 / 5193 / 5283 against a target of 4800.
   The words were never the problem: the **section headers** were. Thirteen EMO headers averaging 105
   characters is 1,365 characters of stage direction inside a field that has a hard limit.
2. **V2 and V3 prompts were at 985 and 984** — inside the band, but `music_prompt_hug_ceiling` is 985.
   That is gate-optimisation pressure showing, and the run's own instruction is *write INTO the middle
   of the bands*.
3. **The verse architecture was recruiting numerals.** *six · two · twice · third · dozen · three ·
   pair · couple* are all stressed monosyllables, which is exactly what the line-ending rule asks for,
   so the form itself kept pulling numbers into a song allowed **one**. This was not a lapse of
   attention; it is a structural interaction between two constraints, and it needed a rule rather than
   a fix: **no numeral may ever be a line-ending in this slot, and the only number in the lyric is the
   sung fact.**

---

## 2. THE STRUCTURAL REPAIR — the field size

**Changed in all four:** `[Theme:]` shortened, `[SONG FORM:]` shortened, and **every EMO header cut to
its load-bearing content** — section, emotion pair, voice, and the one production event that section
introduces. Nothing else.

**Refused:** cutting sung lines to buy space. The sung line count is the song; the headers are
instructions to a generator. **When a field is over its cap you cut the instructions, not the record.**

**Refused:** dropping the per-repeat headers on returning choruses. Suno needs the header on every
section, and the returning chorus is where the parade event is declared (*tambourine enters* · *full
horns*). Those five words are the only readable proof that SWAG's mechanism is a sequence of events.

---

## 3. V1 · "MIND HOW YOU GO" — line-level changes

**Verse 1, seventh line.** *"He turns the heat a click or two."* → **"He turns the heater up a bit."**
**Caught by:** the numeral counter. **Also better:** *grit / bit* is an internal rhyme where there was
none, and *heater* is the object's actual name, which is Ruthless Specificity rather than a paraphrase.

**Verse 3, seventh and eighth lines.** *"The bass comes in before the beat / and hits harder than the
beat it stole."* → **"The bass comes in ahead of the top / and hits harder than the top it stole."**
**Caught by:** reading aloud — *beat / beat* is a stumble inside one couplet. **Also better:** *the top*
is the accurate name for the place the downbeat used to be, so the C1 line now says exactly what the
body is doing. And *top / stole / hole* chimes into the couplet above it.

**Bridge, fifth and sixth lines.** The dog and its name — **cut entirely**, replaced with:
**"The window fogs. Somebody draws. / The wiper doubles at the top. / The amber light goes round and
round / and paints us all the same. Non-stop."** **Caught by:** THE FLAT TEST — *"He calls the dog by
name: it's Mog"* is charming and it is doing nothing. **The real reason:** it is warmth aimed at
softening a room that the song needs to stay clear-eyed about. A dog on this coach is the listener's
permission to feel that this is all quite nice. The amber light replaces it with an image where
everybody is made the same colour by a machine, which is inventory that still carries.

**Verse 4, first line.** *"They pulled the last coach out at six."* → **"They pulled the last coach off
at tea."** **Caught by:** the numeral counter.

**Verse 1, opening line — kept, against a temptation to improve it.** *"The rain comes off the lane in
sheets."* survives THE FLAT TEST and carries the internal rhyme on the offbeat, which is the whole
architecture. It was left alone.

**Chorus — untouched, and no note is owed.** Byte-identical three times. *A byte-identical chorus needs
no defence.*

---

## 4. V2 · "IT STILL SAYS ON TIME" — line-level changes

**Chorus, fifth and sixth lines.** *"We are the ones out here in the wet. / It still says on time. And
it isn't. Yet."* → **"Nobody's shouting. That's the worst. / It still says on time. And it says it
first."** **Caught by:** THE FLAT TEST. *"And it isn't. Yet."* dies when spoken to a face; it is short,
not simple, and the *Yet* is doing something I could not name in four words. **The replacement is the
whole song's thesis** — the politeness is the indictment — and putting it in the chorus means **the
room sings it**, which is worth more than any line she could deliver. It also stays out of the bridge,
which is this slot's forbidden section.

**Verse 2, seventh and eighth lines.** *"It printed him a seat and train. / It printed him a perfect
crime."* → **"It printed him a seat, a train, / a platform and a place in line."**
**Caught by:** THE REMARK TEST. *A perfect crime* is the writer being pleased with himself; a listener
notices the phrase before they notice the fact, which is a toll charged at exactly the moment the song
needs to be free. **The replacement is a flat inventory**, and the inventory is the verdict.

**Verse 4, whole stanza rebuilt.** *"We gave it a couple of goes and then… And nor have them."* →
**"We had a downbeat at the start. / We let it go and shut the door. / It has not come back. It will
not. / And your foot still goes to the floor. / You go to stand. There's nothing there. / You take the
weight yourself. You do. / That's how you know a thing has gone: / your body keeps on doing it too."**
**Caught by:** the numeral counter (*a couple*) and by basic grammar (*nor have them* is not English).
**Also better:** the body-feels-it line is now about the listener's own body rather than about the bass,
which makes it a different answer to RULING C1 from V1's and V3's — the run's own falsification test
asks whether four variations are one song, and this is one of the places it would have been.

**Bridge, fourth line.** *"Somebody's done that all along."* → **"Somebody's done it all year long."**
**Caught by:** the line-ending counter. *Along* is a disyllable and this slot's architecture requires a
stressed monosyllable at every line end. *Long* keeps the rhyme with *wrong* and adds a duration that
makes the hand-typed sheet more specific.

---

## 5. V3 · "SOMEBODY SIGNED IT" — line-level changes

**Lift, all four lines.** *"Three buildings off…"* → **"A few streets off they're warm and dry. / A few
streets off they're out. Good night. / A few streets off they signed the sheet. / A few streets off. And
here's the street."** **Caught by:** the numeral counter (*three*) **and** the line-ending counter
(*asleep* is a disyllable, and it was sitting in the most-repeated block in the song — the worst
possible place to break the architecture). **Also better:** *"they're out. Good night."* is drier than
*fast asleep* and it is said in the same voice as the joke, which is the personality's own law.

**Chorus, second and fourth lines.** *"went home to bed" / "here's the rain and here's the road
instead"* → **"and drove off home" / "And we're the ones stood out in the storm."**
**Caught by:** the line-ending counter — *instead* is a disyllable, three times over, in the byte-identical
chorus. **Also better:** *drove off home* is an action rather than a destination, and *storm / warm /
home* turns the chorus into one long chime the room can hold without knowing why.

**Verse 1, fourth and fifth lines.** *"A dozen of us"* → **"The lot of us"**; *"The woman third row down"*
→ **"The woman up the front"**. **Caught by:** the numeral counter, twice.

**Verse 3, third line.** *"I read it twice."* → **"I read it slow."** **Caught by:** the numeral counter.
**Also better:** *slow* is what actually happens when you read a thing you do not want to be true.

**Outro, second line.** *"Say it nice to the man in green."* → **"Say it nice to the hand with the
pen."** **Caught by:** the cross-variation separation table in step 07 — the man in green belongs to V2
and appearing in both would make two variations share a character in their closing move. **Also
better:** the hand with the pen is this variation's own praise object, so the outro now closes on the
person the song actually admires.

**Outro, second quatrain rebuilt** so the trombone is named as the last instrument playing, matching the
guide, and so the closing bass line no longer repeats V2's.

---

## 6. V4 · "LOUDER THAN THE RAIN" — line-level changes

**Chorus, second, fourth, fifth and sixth lines** — rebuilt. *"can hear a thing / Listen to it sing / It
costs them nowt and it costs us nowt / So let it out"* → **"can hear a note / Give it some throat / It
costs them nothing. It never will. / Louder than the rain. And louder still."**
**Caught by:** THE FLAT TEST and by sense. *"It costs them nowt and it costs us nowt"* is muddled — the
second half undoes the first, and the point is that it costs **them** nothing. The dialect word also
shuts out half the room a hook is supposed to belong to, which fails **L — LOAD THE FLOOR** outright.
*"Listen to it sing"* asks the room to listen; **"Give it some throat"** asks them to sing, which is the
only correct instruction in a song called this.

**Verse 2, fourth line.** *"and both her arms have got a load."* → **"and her arms have got the lot to
hold."** **Caught by:** the numeral sweep. **Also better:** *the lot to hold* is doing work — she is
carrying everything *and* standing.

**Verse 2, eighth line.** *"and every one of us comes along."* → **"and the whole coach picks it up.
Come on."** **Caught by:** the line-ending counter (*along*). **Also better:** *picks it up* is the
actual mechanism of a singalong and it is a verb, where *comes along* was a preposition doing a verb's
job.

**Verse 3, seventh line.** *"So somebody starts the song again."* → **"So somebody starts the song back
up."** **Caught by:** the line-ending counter (*again*).

**Bridge, last line.** *"Somebody's done it twice. It's fine."* → **"Somebody's done it again. It's
fine."** **Caught by:** the numeral counter. The numeral moved out of the line ending and out of the
count entirely.

**Verse 4, fifth to eighth lines.** *"stand up and shift / the standing woman the pair / Still not
there."* → **"stand up and squeeze. / They give the standing woman the seat. / She sits. She does not
stop the song. / She does the low bit. Still not sweet."** **Caught by:** the numeral counter (*the
pair*) and by clarity — *the pair* was ambiguous between two people and two seats. **Also better:**
*Still not sweet* lands the wolf's resolution on the fact that she is **still getting it wrong and still
singing**, which is what the variation is about.

**Verse 4, first four lines — kept exactly, and they are the most important four lines in the pair.**
*"The downbeat played and then it stopped. / We never put it back. It's gone. / Sing on the and. Sing on
the and. / There's nothing on the top. Sing on."* This is simultaneously the mandated naming of the
absence **and** the hole-swap mitigation `05` assigned to this slot: the one variation where a whole
room sings is the variation that says, in words, that **the singing happens on the upstroke and the hole
stays empty.** Nothing was allowed near it.

---

## 7. WHAT THE THREE GATES REFUSED TO CHANGE

- **The four hooks stayed.** *Mind how you go* · *It still says on time* · *Somebody signed it* ·
  *Louder than the rain*. All four pass the step-06 test — a stranger on that coach could have said any
  of them first — and none of them names the decision, the operator or the system. The indictment
  arrives dressed as manners in all four.
- **No line was made denser.** Every repair in this file makes a line **plainer**. That is the direction
  the run's own diagnosis points: *a plain line is harder to write than a dense one; density hides a
  weak idea.*
- **No wolf was moved to the bridge**, and no bridge acquired a verdict. All four bridges remain pure
  inventory, which is the cost this slot agreed to pay at step 06 and did not renegotiate when the
  writing got hard.
- **No band drop-out was added anywhere**, and no line was given a hush. *She'll say the line. She
  won't sell it.*

---

## 8. THE ONE THING NO GATE IN THIS FILE CAN SEE

Every instrument used here is textual. **Not one of them can hear whether a two-tone chop at 126 BPM
arrives carefree or arrives cross.** The whole slot rests on a claim about how a rhythm feels — that the
skank is physically incompatible with sounding angry — and that claim is untested by anything in this
pipeline. The exclude fields name the failure class in words (*no shouting · no sneer · no raised voice ·
nothing that reads as a sonic assault*), which is not the same as detecting it.

**And the absence is worse.** No text gate can hear a downbeat that is not played. The lyric names it,
the guide places it, the prompt instructs it, and the exclude field forbids its return — that is the
ceiling of what a text tier can do. **If the render puts a kick on beat one, everything in this pair is
decoration.** That is an ear's job, under THE BLIND RULE, and the audio has not been heard by anything
that wrote this file.

💜
