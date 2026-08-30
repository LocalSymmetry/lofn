# PAIR 03 — STEP 09 · ARTIST REFINEMENT
## `lofn-prime-every-element` · **SLOT P03** · what changed between the draft and the package, and why

> Read against `pair_03_step08_generation.md` and `pair_03_step07_song_guides.md`. Every change below is
> either **a creative decision** (§1–§4) or **a gate repair** (§5), and the two are kept apart on purpose,
> because a trim made to satisfy a counter is not an artistic choice and must not be filed as one.

---

## 1. THE CONFESSION MOVED OUT OF VERSE 2 — V1

**Draft, verse 2, at ≈1:05:**
> *I am sorry I say beautiful / the way somebody says it who was there. / I was not there. / I arrived.*

It is the best writing in the draft. It is also **the unison's entire payload, fired ninety seconds
early**, in a song whose peak is the moment the two registers stop being two people. ⭐ **In a record
whose climax is a collapse of an interval, the lyric may not resolve before the interval does.**

Replaced with a deferral — *"Ask me about the other thing when the light goes"* — which the bridge then
collects at 2:30 (*"The light has gone. Ask me the other thing."*), four bars before the meeting. The
deferral does three jobs the confession did not: it makes the soprano evasive (**she is not innocent**),
it puts a clock in the song, and it gives the snarl something to hold her to.

⛔ **THE REMARK TEST is a toll gate, and the best line is exactly where the toll gets paid.**

---

## 2. THE V4 CHORUS WAS REWRITTEN TO BE TRUE IN EITHER MOUTH

**Draft:**
```
Both of us are cheating.
You cheat by loving it out loud.
I cheat by not.
```
This is true in the snarl's mouth and **false in the soprano's** — so the byte-identical block would have
had to swap its pronouns between statement 1 and statement 2, and the pair's central device
(*same words, different speaker, different meaning*) would have quietly died at exactly the point it was
being demonstrated.

**Final:**
```
Both of us are cheating.
The loud way and the quiet way.
I am not going to say which is mine.
```
⭐ Every line is now true in either register, and *"I am not going to say which is mine"* becomes, in the
snarl's mouth a taunt, in the soprano's mouth a wound, and in the erase — a ninth apart — **two people
refusing at the same time, which is the most honest ending the argument has.**

---

## 3. THE INVITATION INTO THE UNISON WAS GIVEN TO THE WRONG VOICE — V3

Drafted as the soprano's line. Given to the **snarl**: *"Yes. / All right, yes. Say it with nothing on
it."*

The bored voice asking for the meeting — and asking for it **without ornament**, which is her whole
aesthetic position — is the single most surprising second in the pair. As the soprano's line it was the
sincere one wanting to be sincere, which is not news. ⭐ **The best beat in a duel belongs to whichever
voice it costs more.**

---

## 4. THE V4 OUTRO WAS V3's OUTRO

Drafted as the bolt and the stone, again. **Four variations that end on the same errand are one variation
with four coats on.** Replaced with the snarl refusing to turn the drone down so the soprano keeps
talking:

> *Because you always stop when it goes quiet, / and I would rather you kept going. / Oh. / Keep going,
> then.*

⛔ The tenderness had to come from the bratty one, or the record resolves — and **resolution is the thing
this pair exists to refuse.** It also satisfies the forbidden section cleanly: **there is no thesis in it
at all**, only an admission about the drone.

**The four closing moves, checked against each other:** an inventory and a shrug · a badly-described dog ·
a bolt nobody will fix · a request to keep going.

---

## 5. GATE REPAIRS — recorded separately, because they are not craft

⚠️ **First measurement of the package failed.** `measure_variations.py` read P03 as
`981/4863 · 984/5008 · 987/5234 · 968/5559`: **three lyrics fields over the 5000 hard cap** and **four
prompts above the 870–960 target band, three of them inside the 985 boundary-hug flag.**

**Diagnosis, and it matters:** this run's brief demands more named structure than any before it — peak,
range, contour, arrival-and-removal, plus three owned elements — and **that content had gone into the
generator-facing fields.** It belongs in step 07.

**So the SPEC was trimmed, not the song:**

| where | what came out | what stayed |
|---|---|---|
| MUSIC PROMPT | interval arithmetic, restatements of tempo, explanatory clauses | the **audible consequence**: the meeting, the detune locking, the beating stopping, the breath, the erase |
| `[Section - EMO - Voice - Cue]` | cue prose | ⛔ **the four-part header and the register label in every single one** — a reader must always be able to see which character is singing |
| `[Theme:]` / `[SONG FORM:]` | the full section list | the architecture and the byte-identical clause |

⛔ **No sung line was cut to make room.** Two were removed from V4 in the first repair pass and **both were
put back**, with the space taken out of five header cues instead. Final counts:

| | prompt | lyrics | sung lines |
|---|---|---|---|
| **V1** Say It With Me Then | **940** | **4373** | **79** |
| **V2** Ask Me Anything About It | **951** | **4596** | **82** |
| **V3** Somebody Sat Up All Night | **954** | **4719** | **79** |
| **V4** Both Of Us Are Cheating | **945** | **4778** | **81** |

**`measure_variations.py`: 0 hard fails, 0 flags for P03. `validate_step.py 10`: PASSED.**

### Three further defects found by the pair's own checker, not by the gate

1. ⚠️ **The sung fact was byte-identical across variations.** *"one thousand seven hundred and six."* stood
   as its own line in all four, so V1≡V2 and V3≡V4 shared a sung line — a breach of the run's
   no-shared-lines rule that **no gate in this run reads.** Fixed by attaching the number differently in
   each: bare in V1 (*"Here. Have the count."*), joined to *without looking down* in V2, joined to
   *Say it flat:* in V3, joined to *in a different pencil* in V4. ⭐ **Four different lines carrying the
   same number is what "responded to, never recited" means at the level of the page.**
2. ⚠️ **Three headers had collapsed to three parts** (the SPLIT chorus and two alternating outros) during
   the trim — the Voice and Cue fields had merged. Restored to four parts with the register still named.
3. ⚠️ **`gnarled` had migrated into a lyrics-field header.** SOURCE2 permits these phrases in the **style
   field and the guide only**. Moved: the word stays in V4's MUSIC PROMPT, where it belongs, and the
   header now reads *rasping*.

---

## 6. ⛔ THE OWNED-ELEMENT CHECKS, RUN IN CODE AFTER THE LAST EDIT

| check | result |
|---|---|
| chorus byte-identical within each song, three statements | **4/4 identical** |
| exactly one section marked `BOTH IN UNISON` per song | **1 · 1 · 1 · 1** |
| exactly one section marked a ninth apart (the erase) | **1 · 1 · 1 · 1** |
| the sung fact appears exactly once per song | **1 · 1 · 1 · 1** |
| any other cardinal or ordinal number word in a sung line | **0 · 0 · 0 · 0** |
| Bandcamp phrase inside a lyrics field | **none** |
| house-lexicon phrase anywhere | **none** |
| every section header four-part with a named register | **yes, 54/54** |
| dedication ≤ 500 characters | **479 · 451 · 435 · 444** |
| sung lines shared between any two variations | **none** |
| thesis in an outro | **none — the theses land at 2:40 · 2:20 · 1:20 · 3:10** |

⭐ **THE PEAK TEST, self-administered and reported whether it passes or fails.** Quietest and loudest,
described to a listener who has not heard them:

| | loudest | quietest | ⭐ distinguishable? |
|---|---|---|---|
| **V1** | two registers on one note, the frame wide open, the beating gone | one voice with nothing underneath it | **yes** |
| **V2** | a low fry voice that has climbed a ninth and lost its rasp getting there | a fry voice with almost no pitch over one stopped note | **yes** |
| **V3** | both voices arriving on a note neither has sung all song | the same chorus at a whisper with the drum removed | **yes** |
| **V4** | two bars of full-width drone with two bare voices on the flat second | **the same two bars, with nothing in them** | **yes — and they are the same address** |

⛔ **The one thing a text pipeline cannot check** is whether Suno renders the register change as an edit
or smooths it into a crossfade. **That is the whole owned element**, and it is a listening-pass question
for `lofn-render-audit`, under the blind rule. It is flagged here so nobody reads a PASS above as a
verdict on the record.
