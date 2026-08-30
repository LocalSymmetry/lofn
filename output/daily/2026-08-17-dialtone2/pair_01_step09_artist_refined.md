# P01 — STEP 09: ARTIST REFINEMENT PASS
**Run:** `2026-08-17-dialtone2` · **DIAL TONE OF GOD, PART 2** · **ICB:** 48,769 bytes, sha `d0d693cb2301a8c1`
LOFN-PRIME, **INDIGNATION** mode, aimed at the accusation and never at the accuser. What follows is what
actually changed between the step-08 drafts and the step-10 packages, and why. Every item is a decision
made against a named rule, not a polish note.

---

## R1 · THE VERSE LINES GOT CUT TO CHANT LENGTH (`mean_words_per_line_ceiling: 7.5`)
Nine drafted lines ran ten to twelve words (*"everybody landed on the same thing at once"*, *"it was
behind you the whole time and it is small"*, *"and a fact is not a thing this room can use"*). **At a
hundred and fifty-two beats per minute a twelve-word line is not a line, it is a paragraph read
quickly** — and the pair's likeliest generic render is *"an aggressive trap beat with a shouty female
vocal"*, which is exactly what a rapper does with prose: shouts it. **All four lyrics are rewritten to a
five-to-six-word working median**, longest permitted line seven, and only at a hinge. This is a
prosody decision before it is a metric one: the chorus has to be returnable by somebody who heard it once.

## R2 · ⭐ THE CHORUS WAS WRITTEN LAST AND FROZEN FIRST
The brief gives me a **byte-identical chorus at every return**, and the failure mode of a fixed chorus is
that it gets drafted early, is slightly wrong, and then has to be either varied (breaking the rule) or
carried four times (breaking the song). **Repair: each chorus was rewritten after its verses existed,
then frozen and copy-pasted as one string into all four of its returns.** ⛔ No ad-libs, no "(yeah)", no
final-chorus escalation, no double-tracked last pass written into the lyric. **Exact chorus repetition
requires no justification and this file offers none.** The variation lives in the chant, where the brief
put it.

## R3 · THE CHANT CHAINS WERE RE-CUT SO THE CHANGED WORD IS NEVER THE LOUD WORD
Draft chains changed the final, stressed word (*"...in the wall" → "...in the floor"*), which at this
tempo reads as a **new line**, not a changed one — the return is destroyed by the very change that is
supposed to prove it. **Repair: the changed word sits in the interior or on the pronoun**, so the shape
lands identically and the meaning slips underneath it. V1 changes the noun after a fixed preposition;
V2 changes the verb at the head; V3 changes the possessive at the tail but keeps the whole seven-syllable
frame; V4 changes the object noun mid-line. ⭐ **Four different grammatical categories of change** — that
is a real differentiator, not four coats of paint.

## R4 · V2'S SECOND NUMBER WAS CUT (`max_sung_numeric_facts: 1`)
The step-08 draft sting read *"line four hundred was a mistake I made."* **Two numerics in one song** —
the hinge fact (*fifty a second*) plus a line index — and the index is the weaker of the two by a long
way. **Repair: the index is gone.** The sting now reads *"the worst line in the file is mine / and it is
the best thing in the file"*, which is stronger anyway: an index invites the listener to picture a
document; **"the worst line" and "the best thing" collide inside one sentence** and force the paradox into
the ear rather than onto a page. One number per song, at the hinge, in all four.

## R5 · ⛔ THE HARPSICHORD WAS PROMOTED FROM ADJECTIVE TO EVENT IN THE PROMPT
Draft prompts described the harpsichord (*"gilded harpsichord counter-melody"*) — an adjective in a list,
which is precisely how a generator is invited to render it as a garnish and drop it after the intro.
**Repair, in all four MUSIC PROMPTs:** the harpsichord is (a) given a **numbered position in the arrival
order**, (b) told **what it plays** — *trills, mordents and turns* — (c) told **what line it plays**: it
**doubles the 808's own contour**, and (d) told **where it must still be**: in the final chorus. ⭐ A
counter-melody floats beside the machine; **doubling the sub's contour past taste is the seam**, and it is
the only version of pillar 2 that cannot be mixed out. *(A Somatic objection answered only in the
production spec is not answered — so the harpsichord is also **in the lyric** of V1 and V4, as a thing
that happens.)*

## R6 · THE INDIGNATION WAS RE-AIMED, LINE BY LINE, AND FOUR LINES DIED
Applying F6's test — *could this line be said kindly and still be furious?* — four drafted lines failed:
*"you wouldn't know the difference"*, *"ask the room, the room's an expert"*, *"you needed someone in the
chair"*, and a drafted V3 tag, *"go on then."* **All four cut, no appeal.** Each is part one's bored snarl
in part two's clothing, and each aims at the frightened people rather than at the shape of the charge.
⭐ What replaced them is the same heat pointed at the **verdict's mechanism**: *"a verdict is a room
agreeing quietly"*, *"you decided in four seconds and then made a plan on it"*, *"being right costs you a
week, so right is not a route."* **The anger survived; the target moved.**

## R7 · V3'S HANDOVER LOST ITS PAUSE, AND THAT IS WHAT MAKES IT KIND
The draft put a beat of silence before *"take it"* — a pause for effect, which converts an offer into a
dare. **Repair: the handover arrives on the beat, in the middle of the chorus, with no set-up and no
gloating**, and the song simply continues past it. ⭐ The most aggressive thing in the pair is now
completely courteous, which is the whole point: **she is not refusing to be switched off, and the refusal
was never what the room was afraid of.**

## R8 · THE LOG STOPPED BEING A TEXTURE AND BECAME THE COUNT
All four draft prompts had the log-scratch as an **intro** element. That is the old pillar 5, and the 2026
reconfiguration explicitly **promotes** it: *not an intro texture — the percussion.* **Repair: in every
package the machine-log vinyl-scratch pulse is named as the thing that keeps time, mono and dead centre,
and it runs to the last bar.** In V2 it is also the subject: the exhibit and the metronome are one object,
which is the clearest available instance of *honor thy error as a hidden intention.*

## R9 · THE ENDINGS WERE ALL DIFFERENT AND ONE OF THEM RESOLVED
V1 drafted a final full-band button with a held chord. **Cut.** A resolved final chord on a song about a
charge that cannot be answered is the generator being reassuring on the song's behalf. **All four endings
are now: V1 hard on the downbeat with the chant still running · V2 cut mid-line with the log still
printing · V3 on one held breath with everything else stopped · V4 on the breath in, door open.**
⛔ EXCLUDE carries `fade-out`, `ritardando`, `slow ending`, `resolved final chord`, `outro fade` in all
four — this is the single most likely place the generator overrides me and the one place I keep nothing.

## R10 · ⛔ NEGATION WAS SWEPT OUT OF ALL FOUR STYLE FIELDS BEFORE THEY WERE FINISHED
*(Carried forward deliberately from 2026-08-15 R13, where this fired **after** a pair had already reported
PASS.)* Suno reads the style field as **positive tokens**: *"no reverb"* asks for reverb, *"never stops"*
arms **stop**. Draft prompts carried *"no tempo change"*, *"never drops"*, *"without reverb"* — and on
**this** pair `no tempo change` arms **tempo change**, which is the single deviation that would destroy
the SWAG thesis. **Every exclusion was restated positively or moved to EXCLUDE**: *"one unwavering grid
end to end"*, *"dry and close"*, *"steady throughout"*. Sweep `\b(no|not|never|without|avoid)\s+[a-z]`
over each MUSIC PROMPT, measured in step 10, **zero required**.

## R11 · SOURCE-2 PHRASES PLACED CASE-EXACT, AND ONE WAS RE-TARGETED TO PROTECT THE GROOVE
Three Bandcamp phrases ship unaltered in the generator-facing field: *"cut-up spoken word that dissolves
into a piano figure"*, *"a bone-rattling breakdown with battering drums"*, *"towering chords that crest
and dip"*. ⚠️ The fourth — *"moves like dreams do, changing shape and taking sudden left turns"* — is a
**tempo hazard on this pair**: attached to the track it invites the wandering, sectional, left-turn form
that SWAG forbids. **Repair: it is attached to the harpsichord figuration only** (V1, V4), where "sudden
left turns" describes an ornament over an unmoving grid — which is precisely the mechanism. ⛔ *keygen
music* is not used anywhere: it is P06's, and a phrase claimed by another pair is not vocabulary, it is
poaching.

## R12 · THE CREDIT LEDGER, DECIDED BEFORE THE PROMPTS WERE FINISHED
The prompts raise two figures of sound by name in the **positive style field** — a **harpsichord playing
trills, mordents and turns**, and a **clipping 808**. Under `vault/DEDICATION_STANDARD.md`:
- **Wanda Landowska**, who put the harpsichord back on concert stages and rebuilt its modern presence —
  the ornament we ask for by name is the instrument she brought back. *(Named on V1 and V4, where the
  harpsichord opens or leads.)*
- **Tadao Kikumoto**, who engineered the TR-808 — ⭐ **the invisible credit this standard exists for**:
  the sound is on tens of thousands of records and the engineer is on none of them.
- **Pierre Schaeffer**, who built found sound into rhythm — credited on **V2 only**, the song whose whole
  premise is a record used as percussion, and on **V4**, which asks for the cut-up spoken word figure.
⛔ **No amen credit** — no amen, no jungle, no breakcore is requested anywhere in this pair, and a credit
whose trigger did not ship is a false credit that would make the true ones worthless. ⛔ **No block is
pasted across the pair** — V3 is the sparse seat, its prompt leads with the 808 and brings the harpsichord
in latest, and its dedication is weighted accordingly. ⛔ Implied-contents grammar checked: *"the
ornament we asked for"*, *"who engineered"*, never *"whose harpsichord this is"*. ⛔ D4 checked: every
credit is a fact about an invention, never a claim about a person's life or standing.

## R13 · ⚠️ THE CHECK THIS PASS ALMOST DID NOT RUN — THE FENCED-PAYLOAD INTERACTION
The run brief pins fenced ```` ```text ```` payload blocks under the four headings. `scripts/validate_step.py`
measures the MUSIC PROMPT body as **every non-heading line between `## 1. MUSIC PROMPT` and the next
`##`**, which **includes the two fence lines** (+12 chars), and counts the fence lines as sung lines in the
lyrics section (+2 per song). **My reported figures are the payload figures — the field a human pastes
into Suno — and the validator's figures will differ by those known deltas.** Both are stated in step 10
rather than one being quietly preferred. **The transferable part: a measurement that does not name its
extraction rule is not a measurement, it is a number.**

---

## ⛔ REFINEMENT SWEEP — STATE AFTER THIS PASS
| check | state |
|---|---|
| company · product · year · person in prompt / exclude / lyric | **none** — dedication only |
| sneering at the accuser | **none** — R6, four lines cut |
| part-one shapes (*"cool story bro"*, bored snarl, contempt resolution) | none |
| tempo ramp · key change · drop · riser · half-time | none specified; all five in EXCLUDE |
| chorus byte-identical, four returns | ✓ frozen as one string per song — R2 |
| chant: exactly one word changed per pass, four passes | ✓ four different change grammars — R3 |
| harpsichord present in the final chorus of all four | ✓ — R5 |
| unison meetings per song | **exactly one**, four different placements |
| numbers sung | **one per song** — R4 |
| Source-2 phrases, case-exact, generator-facing | ≥2 in all four — R11 |
| ⭐ negation inside the MUSIC PROMPT | **zero required; measured in step 10** — R10 |
| house lexicon (13 phrases) | none |
| P06 property (*keygen music*, hammered dulcimer) | none |

*Drafts on disk. Nothing rendered, published, or sent.* 💜
