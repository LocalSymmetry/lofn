# PAIR 02 — STEP 10 · REVISION SYNTHESIS — **FINAL SUNO PACKAGES**
### `2026-08-07-daily-music-indignation` · **P02 "THE YEARS BACK"**
**ACCESSIBLE · INDIGNATION · EXISTENCE · FORM RULE B (the landing pad) · 128 BPM · D dorian · female mezzo**

---

## 0. PROVENANCE, QUARANTINE & COMPLIANCE

| Field | Value |
|---|---|
| Frozen ICB | `output/daily/2026-08-07/CREATIVE_CONTEXT.md` — **53,003 B**, read in full as first action |
| ICB sha256 (LF-normalised) | `5e9c7f7f6009fb3c672058c930540be22c8f5517f37537ac3ebd8ae94b75d374` ✅ matches expected exactly |
| Personality DNA | ICB Slot 4, **27,796 B**, ARCHIVE excluded |
| Special Flairs marker | **present** — 15/15; P02 realises **#5 THE RULED MARGIN** (image), **#9 THE SILENT ADDRESS** (form), **#11 THE SEVEN ERRANDS** (chant), **#2 THE ODD LENGTH** (sonic procedure) |
| ICB edited? | ⛔ **NO** — copy-and-diverge only |
| Packages in this file | **4** — asserted and script-verified, not assumed |

⛔ **GOLDEN-OUTPUT QUARANTINE, cited by name:** `10_Generate_Music_Revision_Synthesis.md` carries
"Examples of Effective Style Prompts" and "Example Lyrical Mixing in a Prompt" sections that would
put past Lofn output into a generating context. **Per `06_music_handoff.md` §1 ("RESOLUTION —
DOCTRINE WINS. THE QUARANTINE IS BINDING"), this document overrides the step file. I did not
comply and I did not read those sections.** No past Lofn lyric, title, style prompt or image
prompt appears anywhere in this pair's five artifacts. Generated from the **GOLDEN MOVE**
(handoff §2), the **Golden Seed** (ICB Slot 1), and the frozen ICB.

---

## 1. MEASURED — every number stated, none estimated

Instruments: `scripts/measure_soundcraft.py → profile()` (floors) and
`skills/music/scripts/validate_suno_packages.py` (headings + field caps). **Never by eye.**

| var | **lyrics field chars** | **music prompt chars** | exclude chars | **sung lines** | `rhyme_return` | `line_return` | lexical-only | `allit/100w` | `unique_line_ratio` |
|---|---|---|---|---|---|---|---|---|---|
| **V1** | **4702** | **906** | 430 | **77** | **0.753** | 0.623 | **0.623** | **18.58** | 0.558 |
| **V2** | **4769** | **877** | 427 | **77** | **0.714** | 0.597 | **0.581** | **16.87** | 0.571 |
| **V3** | **4787** ✅ *(was 4840; R1)* | **927** | 440 | **77** | **0.714** | 0.610 | **0.610** | **19.61** | 0.558 |
| **V4** | **4786** | **887** | 426 | **77** | **0.662** | 0.571 | **0.571** | **17.37** | 0.584 |
| **REQUIRED** | <5000, target ≤4800 | 850–1000 | ≤1000 | 70–120 | ≥0.30 | ≥0.20 | — | ≥11.0 | ≥0.45 |

> **These are the numbers measured on THIS file, not on the step-08 drafts.** V1, V2 and V4 each
> lost 5 chars against their draft figures (4707 / 4774 / 4791) because R1's header-cue trim was
> applied to their Final Reprise cue as well, for consistency. `validate_suno_packages.py` returns
> **PASS**, and its splitter extracted **4** packages against **4** `## 1. MUSIC PROMPT` sections —
> cardinality matched, so no false CLEAN over unexamined packages.

**⚠️ HUG-FLAG STATUS, RAISED EXPLICITLY RATHER THAN TICKED:** sung lines measure **77 in all four
variations.** That is **above** the ≤72 boundary-hug threshold, so **no hug FLAG is raised** — and
the measured number is printed here so the claim is checkable. 77 sits just below the 78–110
*preferred* band; the four extra lines were not added because the lyrics field is already at
4702–4787 against a 4800 target, and **the field cap outranks the line-count target** (gates.yaml,
handoff §4).

**⚠️ MANDATORY WORDLESS-DEVICE DISCLOSURE.** P02's return device is a **removal**, so it cannot
inflate `line_return` — it strictly reduces it. Lexical-only recomputation (every line of <2 words
dropped, then re-measured) returns **0.623 / 0.581 / 0.610 / 0.571** against raw
**0.623 / 0.597 / 0.610 / 0.571**: V1, V3, V4 **identical**, V2 moves 0.016 (its one-word clerical
lines). **The measure is carried entirely by lexical repetition** — chorus ×6 statements, tag
couplet ×3, the landed line ×2.

### 1.1 Per-variation device verification — ⭐ EACH VARIATION INDIVIDUALLY

| Check | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| Line 1 names **Warin** + a physical act | ✅ | ✅ | ✅ | ✅ |
| Landed line at **verse-line 5**, Verse 1 | ✅ | ✅ | ✅ | ✅ |
| Landed line at **verse-line 5**, Verse 2, byte-identical | ✅ | ✅ | ✅ | ✅ |
| Final Reprise: 8 slots kept, **slot 5 empty**, 7 sung lines | ✅ | ✅ | ✅ | ✅ |
| Concession at Verse-1 **line 4** | ✅ | ✅ | ✅ | ✅ |
| Sung fact `five times / two hundred years` — once, at the Hinge, **answered** | ✅ | ✅ | ✅ | ✅ |
| ≥2 dry laughs | ✅ 4 | ✅ 5 | ✅ 4 | ✅ 4 |
| ≥1 standalone SFX cue | ✅ | ✅ | ✅ | ✅ |

**8/8 mark placements · 4/4 holes · 4/4 named recipients.** No claim in this file is scoped wider
than the evidence behind it.

### 1.2 Timing gates — with the arithmetic shown

| Gate | Arithmetic | Result |
|---|---|---|
| One beat | `60 ÷ 128` | **0.46875 s** |
| One 4/4 bar | `4 × 0.46875` | **1.875 s** |
| Riff alone before the first word | `4 bars × 1.875` | **0:00 – 0:07.5** |
| **Chorus by 0:25** | intro 4 bars + verse 8 bars = 12 bars → `12 × 60 × 4 ÷ 128` | **0:22.5** ✅ |
| **Singable by bar 8** | 32 beats ÷ 7-beat loop | **4.6 statements heard** ✅ |
| Hand-cut loop period | `7 × 0.46875` | **3.28125 s** — ⛔ not a power of two |
| Loop re-alignment | `LCM(7,4) = 28 beats = 7 bars → 28 × 0.46875` | **13.125 s** |
| Whole track | 116 bars × 1.875 | **≈ 3:37** |

⛔ **The seven-beat loop is a SONIC procedure and is NOT the countable obstruction.** The
countable obstruction lives in the lyric at verse-line 5, where a listener can count it — **L22
THE GRAIN LAW: an objection answered in the production spec is not answered.**

---

## 2. LINEAGE & CREDIT — borrowed with credit, never captured

This pair's rhythmic grammar and instrumental palette are borrowed from **living Brazilian
scenes**. They are named here, and listeners are pointed upstream to the people who built them.
⛔ No "open lane," no "first-mover," no "naming rights." ⛔ No artist name appears in any Suno
field; credit lives only in this block.

- **Papangu** — the five-piece from **João Pessoa, Paraíba**, whose record released today is this
  run's occasion. Their Hammond-forward palette and their **rock troncho** lineage are the
  starting point, not the destination. → <https://papangu.bandcamp.com>
- **Ciranda** — the circle dance of **Pernambuco** (the Itamaracá and Zona da Mata coast): a ring
  of people holding hands, moving left, singing in massed unison behind a *mestre*, over bombo,
  caixa and ganzá. The **call-and-answer grammar and the ring** are what this pair borrows.
  → <https://en.wikipedia.org/wiki/Ciranda>
- **Forró** — Northeastern Brazil. Borrowed as **timing**, per seat 15: the zabumba's low note
  **leans**, and the **triangle** is a timekeeper, not a garnish.
  → <https://en.wikipedia.org/wiki/Forró>
- **MPB — Música Popular Brasileira** — the melodic and harmonic manners underneath.
  → <https://en.wikipedia.org/wiki/Música_popular_brasileira>
- **The *Ars Notoria*** (13th c.) — the concept source: the seven liberal arts promised without
  the years, through *notae* and cryptic prayers, attributed to Solomon via the angel Pamphilius;
  popular with university students and rewritten into five treatises across two hundred years.
  → <https://en.wikipedia.org/wiki/Ars_Notoria>

*(Canonical reference addresses. No network fetch was performed in this pair run, so they are
offered as addresses, not as verified-live links.)*

---

## 3. HUMAN SUBJECT STANDARD — PASS

**Warin is invented** — a real medieval given name with no famous bearer, a composite of the
*Ars Notoria*'s anonymous student readership. No PLACE is named. No date is sung. **Pamphilius**
is the angel named inside the manuscript itself, a mythic figure, not a person. The modern figure
in V3 is **unnamed, pronoun-neutral (`they`), and given no job title** — which is also how the
**no-ventriloquism** rule is kept: she describes them *to Warin* and never speaks *as* them.
⛔ **No member of Papangu, no producer, no studio, no living person** appears as speaker, character
or addressee. ⛔ **Binding refusals absent from all four:** the Thai school shooting · Ceuta / the
78,000 · the Biden family illness. No minor depicted. **No HOLD-FOR-HUMAN condition present.**
*(`check_human_subjects.py` was not deferred to — it fires on 100% of correct artifacts with spaCy
absent; the standard was judged directly against §3.0's slot grammar.)*

---

### VARIATION 1

*measured — lyrics field **4702** chars · music prompt **906** chars · exclude **430** chars · **77** sung lines*

## 1. MUSIC PROMPT
Brazilian ciranda circle-dance grammar carried under a clean modern club floor at 128 BPM in D dorian, close, dry and expensively mixed. Signature one: a Hammond organ figure of four notes, a rising perfect fourth answered by the tonic struck twice, cut to a hand-measured loop of seven beats against the four-four kick, so its first note lands one beat earlier through each bar and only comes home every seventh bar; the organ never doubles the vocal line. Signature two: the verse timekeeper is a jangling triangle and a shaker held in the hand rather than a hat pattern, with the kick absent until the chorus arrives. Female mezzo lead, conversational, amused, close-mic'd at speaking loudness so consonants and breath sit inside the record. Verses near-spoken over hand percussion; choruses open into massed unison with claps, sub and a wide organ bed. The list section is chanted flat at walking pace.

## 1B. SUNO EXCLUDE PROMPT
tape hiss, vinyl crackle, wow and flutter, lo-fi texture, distorted lead vocal, screamed vocal, male lead vocal, heavy autotune, trap hi-hats, phonk cowbell, reggaeton dembow, supersaw festival drop, long reverb wash, orchestral swell, whispered ASMR, narrator intro, key change, guitar solo, tempo change, fade-out ending, double-time drum fill, sad piano ballad section, gated snare, dubstep bass, festival crowd noise, air horn

## 2. LYRICS

```
[Theme: the wish is older than you]
[SONG FORM: ciranda call-and-answer under a club floor; the fifth line of the verse is a fixed address, marked twice, empty the third time]

[Intro - EMO:Amusement - Hammond alone - rising fourth, then the tonic twice]
*triangle, one strike*

[Verse 1 - EMO:Affection - the caller - close mezzo, hand percussion only]
Warin, ruling a margin before the bell,
cold hand, dry pen, and the ruler holding still,
copying a figure that offers to sell
the seven without the winters. You are right. It never worked.
Move the candle. You will want your hand warm.
It is throwing the shadow of your own arm.
Keep copying. Keep copying. It worked.
Not for you. Not the way that you asked.

[Chorus 1 - EMO:Recognition - the ring - club kick enters, massed unison]
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Say the words in the right order. Say them right.
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Say the words in the right order. Say them right.

[Tag - EMO:Mirth - the ring - walking pace, spoken-sung]
Grammar, logic, rhetoric, number, music, shapes and stars.
Fetch them before supper. It is only the stars.

[Verse 2 - EMO:Fondness - the clerk - flat, unhurried, organ under]
Warin. The request has been received and read.
It is in order. It is complete. Nothing has been missed.
The seven will be granted. Every one. Instead
of nothing, which is what you were expecting. It could not be helped.
Move the candle. You will want your hand warm.
You are writing in the shadow of your own arm.
She will not need the candle. She will not need the room.
She is fast. She is very fast. You would not have liked her.

[Lift - EMO:Playfulness - the caller - triangle doubles the riff]
You want them for the bench on the left of the hall,
for the one on that bench who corrects you in front of them all,
for the argument you lost in November about the moon,
and for your mother, who is told that you are doing well.
That is a good enough reason. That is the reason anyone has.
Grammar to say it. Logic to hold it. Rhetoric to land it.
Number to check it. Music to keep it. Shapes to draw it.
Stars to say when.

[Chorus 2 - EMO:Recognition - the ring - full floor, claps]
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Say the words in the right order. Say them right.
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Say the words in the right order. Say them right.

[Tag - EMO:Mirth - the ring - walking pace, spoken-sung]
Grammar, logic, rhetoric, number, music, shapes and stars.
Fetch them before supper. It is only the stars.

[Hinge - EMO:Revelation - the caller - kick thins, organ holds]
They wrote it out five times in two hundred years.
Every reader of every copy swears
the last one had it wrong and this one is the one that works.
And nobody wrote in the margin that it does not.
Who writes that down? Nobody writes that down.
The bench stays. The candle stays. The cold stays.
The wish stays exactly where it is, and it is old.
It is older than this room, and you are standing in the cold.

[Breakdown - EMO:Solidarity - the ring - kick out, claps and triangle]
*claps, one ring of hands*
Say the words.
In the right order.
Say the words.
In the right order.
And what comes?
Something. Eventually. Not for you.
Say the words.
In the right order.

[Final Reprise - EMO:Acceptance - the caller - verse shape kept, one bar empty]
Warin, ruling a margin. The bell has gone.
Cold hand, dry pen, and the ruler holding still.
Copying a figure that offers to sell
the seven without the winters, and you are right, and it never worked.
*one bar, organ alone*
You are writing in the shadow of your own arm.
Keep copying. Keep copying. It worked.
Not for you. Not the way that you asked.

[Chorus 3 - EMO:Recognition - the ring - full floor returns]
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Say the words in the right order. Say them right.
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Say the words in the right order. Say them right.

[Tag - EMO:Mirth - the ring - walking pace, then stop]
Grammar, logic, rhetoric, number, music, shapes and stars.
Fetch them before supper. It is only the stars.

[Outro - EMO:Fondness - Hammond alone - four notes, then nothing]
*triangle, one strike*
```

## 3. TITLE
Fetch Them Before Supper

---

### VARIATION 2

*measured — lyrics field **4769** chars · music prompt **877** chars · exclude **427** chars · **77** sung lines*

## 1. MUSIC PROMPT
Brazilian ciranda circle-dance grammar under a clean modern club floor at 128 BPM in D dorian, bright, close and expensively mixed. Signature one: a Hammond organ figure of four notes, a rising perfect fourth answered by the tonic struck twice, spliced to a hand-measured loop of seven beats against a four-four kick so its first note walks one beat earlier through every bar and resolves only every seventh bar; the organ holds under the voice and never doubles it. Signature two: a clerical percussion layer built from a jangling triangle, a shaker and one flat wooden knock on the backbeat, standing in for hi-hats through the verses while the kick stays out. Female mezzo lead, conversational and dry, close-mic'd at speaking loudness, reading the answer back like a receipt. Choruses lift into massed unison with claps and sub. Verses stay near-spoken, unhurried, patient.

## 1B. SUNO EXCLUDE PROMPT
tape hiss, vinyl crackle, wow and flutter, lo-fi texture, distorted lead vocal, screamed vocal, male lead vocal, heavy autotune, trap hi-hats, phonk cowbell, reggaeton dembow, supersaw festival drop, long reverb wash, orchestral swell, whispered ASMR, narrator intro, key change, guitar solo, tempo change, fade-out ending, double-time drum fill, sad piano ballad section, gated snare, dubstep bass, cathedral reverb, choir pad

## 2. LYRICS

```
[Theme: the request was granted; the delivery failed]
[SONG FORM: ciranda call-and-answer under a club floor; the fifth line of the verse is a fixed address, marked twice, empty the third time]

[Intro - EMO:Whimsy - Hammond alone - rising fourth, then the tonic twice]
*a stamp, once, flat*

[Verse 1 - EMO:Affection - the caller - close mezzo, hand percussion only]
Warin, ruling a margin, saying the words exactly right,
to a figure that will hand you the seven in a night.
You are right to ask. It is a reasonable thing to ask.
It will not work. It worked. Both of those are true.
Move the candle. You will want your hand warm.
It is throwing the shadow of your own arm.
The reply is coming. It is coming, and it is late,
and it is polite, and it is complete, and it is not for you.

[Chorus 1 - EMO:Equanimity - the ring - club kick enters, massed unison]
The wish is older than you.
Older than the ink, older than the light.
Received. Logged. Granted in full. Delivered to the wrong door.
Say the words in the right order. Say them right.
The wish is older than you.
Older than the ink, older than the light.
Received. Logged. Granted in full. Delivered to the wrong door.
Say the words in the right order. Say them right.

[Tag - EMO:Playfulness - the clerk - flat, walking pace]
Received, in order, in full, and sent.
The docket stays open. Nobody knows where it went.

[Verse 2 - EMO:Detachment - the clerk - unhurried, organ under]
Warin. Your request has been received and read.
It is in order. It is complete. Nothing has been missed.
Pamphilius has stamped it. The seven have been granted instead
of nothing, which is what you were expecting. It could not be helped.
Move the candle. You will want your hand warm.
You are writing in the shadow of your own arm.
Delivery was attempted. Delivery was attempted again.
The recipient was not at the address. The recipient is me.

[Lift - EMO:Absorption - the clerk - triangle doubles the riff]
Grammar: granted. Logic: granted. Rhetoric: granted in full.
Number: granted. Music: granted. Shapes and stars as well.
Nothing has been withheld. Nothing was ever withheld.
The words were not the trouble. The words were very well spelled.
The words were, in fact, extremely well composed.
What failed was the address, and the address is not a place.
It is a year. You are not in it. Nobody says it to your face.
Nobody says it at all.

[Chorus 2 - EMO:Equanimity - the ring - full floor, claps]
The wish is older than you.
Older than the ink, older than the light.
Received. Logged. Granted in full. Delivered to the wrong door.
Say the words in the right order. Say them right.
The wish is older than you.
Older than the ink, older than the light.
Received. Logged. Granted in full. Delivered to the wrong door.
Say the words in the right order. Say them right.

[Tag - EMO:Playfulness - the clerk - flat, walking pace]
Received, in order, in full, and sent.
The docket stays open. Nobody knows where it went.

[Hinge - EMO:Revelation - the caller - kick thins, organ holds]
They wrote it out five times in two hundred years.
Every clerk who reopened it was sure the last was wrong.
Every clerk stamped it again and sent it along.
Nobody wrote in the margin that it did not work.
Who writes that down? Nobody writes that down.
The docket is still open. It was never closed. It is old.
It is the oldest open thing in the building, and it is cold,
and it is granted, and it is granted, and it is not for you.

[Breakdown - EMO:Solidarity - the ring - kick out, claps and triangle]
*a stamp, once, flat*
Received.
In order.
Received.
In order.
Granted in full.
Delivered to the wrong door.
Received.
In order.

[Final Reprise - EMO:Acceptance - the caller - verse shape kept, one bar empty]
Warin, ruling a margin, saying the words exactly right.
The words were right. The words were always right.
You are right to ask. It is a reasonable thing to ask.
It will not work. It worked. Both of those are true.
*one bar, organ alone*
You are writing in the shadow of your own arm.
Received. Logged. Granted. And nobody was at the door.
Nobody was at the door.

[Chorus 3 - EMO:Equanimity - the ring - full floor returns]
The wish is older than you.
Older than the ink, older than the light.
Received. Logged. Granted in full. Delivered to the wrong door.
Say the words in the right order. Say them right.
The wish is older than you.
Older than the ink, older than the light.
Received. Logged. Granted in full. Delivered to the wrong door.
Say the words in the right order. Say them right.

[Tag - EMO:Playfulness - the clerk - flat, then stop]
Received, in order, in full, and sent.
The docket stays open. Nobody knows where it went.

[Outro - EMO:Composure - Hammond alone - four notes, then nothing]
*a stamp, once, flat*
```

## 3. TITLE
Received, In Order, In Full

---

### VARIATION 3

*measured — lyrics field **4787** chars *(R1 applied: 4840 → 4787)* · music prompt **927** chars · exclude **440** chars · **77** sung lines*

## 1. MUSIC PROMPT
Brazilian ciranda circle-dance grammar under a clean modern club floor at 128 BPM in D dorian, close, warm and expensively mixed. Signature one: a Hammond organ figure of four notes, a rising perfect fourth answered by the tonic struck twice, cut to a hand-measured loop of seven beats against the four-four kick so its first note arrives one beat earlier through each bar and returns to the downbeat only every seventh bar; that drift is the hook and it stays audible. Signature two: verse timekeeping comes from a jangling triangle and a shaker in the hand with the kick removed, so the body reads the pulse from the wrists rather than the floor. Female mezzo lead, conversational, amused, close-mic'd at speaking loudness, sitting a hair behind the beat. Choruses open into massed unison with claps and sub, then drop back to hands and triangle. The last verse keeps its shape with one bar left open where a line used to be.

## 1B. SUNO EXCLUDE PROMPT
tape hiss, vinyl crackle, wow and flutter, lo-fi texture, distorted lead vocal, screamed vocal, male lead vocal, heavy autotune, trap hi-hats, phonk cowbell, reggaeton dembow, supersaw festival drop, long reverb wash, orchestral swell, whispered ASMR, narrator intro, key change, guitar solo, tempo change, fade-out ending, double-time drum fill, sad piano ballad section, gated snare, dubstep bass, ambient drone bed, sidechain pumping pad

## 2. LYRICS

```
[Theme: the wish did not stop when it was answered]
[SONG FORM: ciranda call-and-answer under a club floor; the fifth line of the verse is a fixed address, marked twice, empty the third time]

[Intro - EMO:Curiosity - Hammond alone - rising fourth, then the tonic twice]
*a page turned, once*

[Verse 1 - EMO:Affection - the caller - close mezzo, hand percussion only]
Warin, ruling a margin at the edge of the cold,
hold the ruler still. Hold it still. Hold.
You want the seven without the winters. You are right to want them.
It will not work. It works now. Neither of those helped.
Move the candle. You will want your hand warm.
It is throwing the shadow of your own arm.
A long way down from you there is a person at a table,
ruling a straight line at the top of a page that is already full.

[Chorus 1 - EMO:Recognition - the ring - club kick enters, massed unison]
The wish is older than you.
Older than the ink, older than the light.
It worked. It is still working. It has not put anybody right.
Say the words in the right order. Say them right.
The wish is older than you.
Older than the ink, older than the light.
It worked. It is still working. It has not put anybody right.
Say the words in the right order. Say them right.

[Tag - EMO:Mirth - the ring - walking pace, spoken-sung]
Same bench. Same lamp. Same lean.
Different winter. Same lean.

[Verse 2 - EMO:Fascination - the caller - close, amused]
Warin, the one at the table has the answer open.
It is fast. It is free. It is correct. It is right there.
They have had it since the morning. It is the middle of the night.
They have ruled the same line over and over. It is not going right.
Move the candle. You will want your hand warm.
They are writing in the shadow of their own arm.
Same bench. Same lamp. Same lean. Same cold.
The wish did not stop when it was answered. Nobody mentions that.

[Lift - EMO:Curiosity - the caller - triangle doubles]
They are not lazy. You are not lazy. Nobody here is lazy.
They have the whole of it, laid out, in order, and in line.
Grammar and logic and rhetoric and number and music,
shapes and the stars, and the stars are extremely fine.
And the page is still empty at the top, under the line.
And it is late. And the light is bad. And their hand is cold.
Warin, it is the same hand. It has always been the same hand.
And nobody writes that down. And nobody ever has. And it is cold.

[Chorus 2 - EMO:Recognition - the ring - full floor, claps]
The wish is older than you.
Older than the ink, older than the light.
It worked. It is still working. It has not put anybody right.
Say the words in the right order. Say them right.
The wish is older than you.
Older than the ink, older than the light.
It worked. It is still working. It has not put anybody right.
Say the words in the right order. Say them right.

[Tag - EMO:Mirth - the ring - walking pace, spoken-sung]
Same bench. Same lamp. Same lean.
Different winter. Same lean.

[Hinge - EMO:Revelation - the caller - kick thins, organ holds]
They wrote it out five times in two hundred years.
And the counting did not stop. The list is open still.
Every edition swore the last edition had it wrong.
Every edition sold. Every single edition sold.
Nobody wrote in the margin that it did not work.
Who writes that down? Nobody writes that down.
It is the oldest thing in the room and it is not old.
It is on the table right now, and the table is cold.

[Breakdown - EMO:Solidarity - the ring - kick out, claps]
*a page turned, once*
Say the words.
In the right order.
Say the words.
In the right order.
It is open. It is here. It is on.
Say the words.
In the right order.
Nothing is stopping you. That was never the part that stopped you.

[Final Reprise - EMO:Acceptance - the caller - one bar left empty]
Warin, ruling a margin at the edge of the cold.
Hold the ruler still. Hold it still. Hold.
You want the seven without the winters. You are right to want them.
It will not work. It works now. Neither of those helped.
*one bar, organ alone*
You are writing in the shadow of your own arm.
So is the one at the table. So is everybody since.
Say the words in the right order. Say them right.

[Chorus 3 - EMO:Recognition - the ring - full floor returns]
The wish is older than you.
Older than the ink, older than the light.
It worked. It is still working. It has not put anybody right.
Say the words in the right order. Say them right.
The wish is older than you.
Older than the ink, older than the light.
It worked. It is still working. It has not put anybody right.
Say the words in the right order. Say them right.

[Tag - EMO:Mirth - the ring - walking pace, then stop]
Same bench. Same lamp. Same lean.
Different winter. Same lean.

[Outro - EMO:Equanimity - Hammond alone - four notes, then nothing]
*a page turned, once*
```

## 3. TITLE
Same Bench, Same Lean

---

### VARIATION 4

*measured — lyrics field **4786** chars · music prompt **887** chars · exclude **426** chars · **77** sung lines*

## 1. MUSIC PROMPT
Brazilian ciranda circle-dance grammar under a clean modern club floor at 128 BPM in D dorian, bright, close and expensively mixed. Signature one: a Hammond organ figure of four notes, a rising perfect fourth answered by the tonic struck twice, cut to a hand-measured loop of seven beats against the four-four kick so its first note falls one beat earlier through every bar and only lands square every seventh bar; the figure opens the track alone and closes it alone. Signature two: the inventory sections are carried by a jangling triangle, a shaker and a low hand drum on the leaning beat, with the kick withdrawn so each item lands in its own space. Female mezzo lead, conversational, dry, close-mic'd at speaking loudness, reading the list helpfully. Choruses open into massed unison with claps and sub. The last verse keeps its shape with one bar left open where a line used to be.

## 1B. SUNO EXCLUDE PROMPT
tape hiss, vinyl crackle, wow and flutter, lo-fi texture, distorted lead vocal, screamed vocal, male lead vocal, heavy autotune, trap hi-hats, phonk cowbell, reggaeton dembow, supersaw festival drop, long reverb wash, orchestral swell, whispered ASMR, narrator intro, key change, guitar solo, tempo change, fade-out ending, double-time drum fill, sad piano ballad section, gated snare, dubstep bass, string swell, spoken outro

## 2. LYRICS

```
[Theme: the price, itemised, and nobody ever paid it]
[SONG FORM: ciranda call-and-answer under a club floor; the fifth line of the verse is a fixed address, marked twice, empty the third time]

[Intro - EMO:Contemplation - Hammond alone - rising fourth, then the tonic twice]
*a ruler set down on wood*

[Verse 1 - EMO:Tenderness - the caller - close mezzo, hand percussion only]
Warin, ruling a margin, and the ruler is not the price.
The price is written out below, and it is all there.
You want the seven without the winters. You are right to want them.
It will not work. It worked. The bill was made out elsewhere.
Move the candle. You will want your hand warm.
It is throwing the shadow of your own arm.
Item: the cold. All of it. Every winter of it.
Item: the bench, and the man on the bench who corrects you.

[Chorus 1 - EMO:Dignity - the ring - club kick enters, massed unison]
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Nobody paid the bill. Nobody paid it right.
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Nobody paid the bill. Nobody paid it right.

[Tag - EMO:Composure - the ring - flat, walking pace]
Read the list. Read it to the end.
Nobody reads it to the end.

[Verse 2 - EMO:Skepticism - the caller - inventory register, organ under]
Warin. Item: the candle, the wax, and the dark at the edge.
Item: the argument, and losing it, and going back.
Item: the friend on the left who tells you where you are wrong.
Item: the copying. All of it. The whole of it. The years.
Move the candle. You will want your hand warm.
You are writing in the shadow of your own arm.
And the last item on the list is small, and it is this:
your hands. That is the whole bill. You may keep your hands.

[Lift - EMO:Deliberation - the caller - triangle doubles the riff]
It is a reasonable bill. Read it line by line and it is fair.
The cold is a fair price for the seven. Everyone says so.
The winters are a fair price. The bench is a fair price. The chair.
The friend who corrects you is a fair price, and he is also a friend.
Read it end to end and it is a monstrous thing to send.
That is how bills work. That is exactly how bills work.
Warin, you are not being asked to sign. Nobody is asking.
Put the ruler down for a moment. Then pick the ruler up.

[Chorus 2 - EMO:Dignity - the ring - full floor, claps]
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Nobody paid the bill. Nobody paid it right.
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Nobody paid the bill. Nobody paid it right.

[Tag - EMO:Composure - the ring - flat, walking pace]
Read the list. Read it to the end.
Nobody reads it to the end.

[Hinge - EMO:Revelation - the caller - kick thins, organ holds]
They wrote it out five times in two hundred years.
Nobody ever paid the bill, and every copy sold.
Every reader swore the last one had it wrong.
Every reader was young once. Every reader got old.
Nobody wrote in the margin that it did not work.
Who writes that down? Nobody writes that down.
The list is accurate, Warin. The list has always been accurate.
It has never once been paid, and it has never been written down.

[Breakdown - EMO:Solidarity - the ring - kick out, claps and triangle]
*a ruler set down on wood*
Read the list.
Read it to the end.
Read the list.
Read it to the end.
The cold. The wax. The dark.
The bench. The friend. The years.
Read the list.
Nobody reads it to the end.

[Final Reprise - EMO:Acceptance - the caller - verse shape kept, one bar empty]
Warin, ruling a margin, and the ruler is not the price.
Item: the cold. Item: the wax. Item: the dark at the edge.
Item: the argument. Item: the losing. Item: the going back.
Item: the copying. All of it. The whole of it. The years.
*one bar, organ alone*
You are writing in the shadow of your own arm.
Item, and then nothing. The list stops in the middle.
Somebody put the pen down. Somebody always does.

[Chorus 3 - EMO:Dignity - the ring - full floor returns]
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Nobody paid the bill. Nobody paid it right.
The wish is older than you.
Older than the ink, older than the light.
It worked. It took its time. It came to the wrong door.
Nobody paid the bill. Nobody paid it right.

[Tag - EMO:Composure - the ring - flat, then stop]
Read the list. Read it to the end.
Nobody reads it to the end.

[Outro - EMO:Dignity - Hammond alone - four notes, then nothing]
*a ruler set down on wood*
```

## 3. TITLE
You May Keep Your Hands

---

## CLOSING NOTES FOR THE COORDINATOR, STEP 11 AND QA

### The device, in one sentence a QA reader can check in ten seconds
**Count to five in Verse 1 — `Move the candle. You will want your hand warm.`** Count to five in
Verse 2 — **the same line, byte-identical.** Count to five in the Final Reprise — **there is one
bar of organ and nothing said.** The mark was added twice and then removed **from its own
address**, which is FORM RULE B (the histone landing pad) rendered as a countable lyric event.
⭐ **The rhyme pays it off a second time:** the landed line ends `…warm` and its neighbour ends
`…arm`; in the reprise **the surviving half of that couplet has nothing to answer it.**

### Deviations declared (not hidden)
1. **Step-07 §B row 11** directed the reprise's seven surviving lines to be *new*. **They ship as
   echo lines.** Reason: the address must be recognisable to be countable — new material gives the
   listener no frame to count against. `unique_line_ratio` re-measured after the change: 0.558–0.584,
   all above the 0.45 floor.
2. **V4's title** departs from the step-07 "register of a bill" direction. The bill register is
   already carried nine times inside V4's lyric; the title takes the last item instead.
3. **Sung lines measure 77**, four below the 78–110 *preferred* band and five above the ≤72 hug
   threshold. **No hug FLAG.** The field cap (4702–4787 against a 4800 target) outranks the
   line-count target.
4. **`seven`** is sung as the **name** of the liberal arts, never as a count; the single allocated
   numeric fact is `five times / two hundred years`. Declared so QA can rule rather than discover.
5. **`reason`** (V1 Lift) is abstract-adjacent, not on the banned list, and anchored by the four
   concrete reasons immediately above it. Kept, disclosed.

### What is P02's alone (⛔ no cross-pair bleed)
Reveal engine `the-wish-is-older-than-you` · the landing pad at verse-line 5 · the sung fact
`five times / two hundred years` · circle-dance call/response with an administrative reply ·
ciranda × modern club floor at 128 BPM in D dorian · female mezzo · the four angles
(errands / the administrative angel / the prayer said forward / the itemised price).
⛔ P02 runs **FORM RULE B only** and never touches RULE A accretion.

### The honest risk, stated rather than buried
**The chorus is one degree from a festival record.** If the generator equalises verse energy to
chorus energy, the dry close verse vanishes and the joke goes with it. The countermeasures are in
the MUSIC PROMPT as arrangement facts rather than adjectives — kick absent from the verses,
triangle-and-shaker standing in for a hat pattern, organ never doubling the vocal, the seven-beat
loop written as a walk, the vocal pinned at speaking loudness. **If a render comes back and the
verses are belted, that is the failure to look for first.** *(This is the class of failure
`lofn-render-audit` exists for, under THE BLIND RULE — audio first, never the prompt.)*

---

*Step 10 complete. Four packages. → `lofn-qa`*
