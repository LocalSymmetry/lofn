# PAIR 03 — STEP 10 · REVISION & SYNTHESIS (FINAL PACKAGE)
## `2026-08-08-daily-music` · THE WRONG INVENTORY · **P03 — THE ROTA**

**ARM** ACCESSIBLE · **AXIS B** EXISTENCE · **MODE** `LOFN-PRIME (INDIGNATION mode — bratty, chanting, blown out)`
**Step file:** `skills/music/steps/10_Generate_Music_Revision_Synthesis.md` · **Overriding authority:** `06_music_handoff.md`
⭐ **This pair carries the run's INDIGNATION requirement inside the ACCESSIBLE arm, and the run's one non-female lead.**

**Continuity Payload Used:** frozen ICB, **142,900 B**, LF-sha **`9b538e912935bc585f512f2ec53c95f44826ce2443f0f60df8588831b224ed1a`** (computed by this agent, exact match to handoff §0) · personality DNA **27,796 B** inlined and read in full · 18 baseline voices · 3 Hyper-Skeptics at 6/12/18, one per panel · **15 Special Flairs present** · 3 debate configurations. Scratch namespace `_work/pair_03/` only.

> **This is the shippable artifact.** Four complete Suno packages: title, dense-paragraph music prompt, separate exclude field, full lyrics field. Every gate in handoff §4 is reported below with a measured value, **including the ones that pass**.

---

## §1 — THE ESSENCE, IN ONE SENTENCE

> **A man who is good at this is in a back office at closing, doing the list for the ninth time this year, and he leaves one name off for a reason that is true — and the room chants three words back at him all night that mean *add me* when they say it and mean *a put-down* when you hear it, and in the last chant it lands on him and he does not notice, because he is already sending the list.**

**The two best ways the medium carries it** (step-10 §1 contract): (1) **call-and-answer**, the oldest work-song device there is, doing the job it was invented for — a room agreeing to something while its hands are busy; (2) **one room, no booth**, so the closing-up (a shutter, a mop, chairs being stacked) is audible behind every line and the song's size comes from the space rather than from a build.

---

## §2 — ⭐ THE D3 DECLARATION (restated; named at step 06 before a word of lyric existed)

**LINE A — sung chant:** `Put me down` — **rises** F♯4 → A4 → **B4** (Put · me · **down**).
**LINE B — shouted pre-chorus:** `PUT IT DOWN` — **falls** B4 → A4 → **E4** (PUT · IT · **DOWN**).
Same singer, same octave, same timbre class. **A perfect fourth apart on both sides of the crossing** — B4 over F♯4 before, E4 under B4 after. **They pass at A4, on the middle syllable, on the same beat, in contrary motion. Neither bends. Neither is transposed. Neither is altered.**

**Executed in the lyric and in the notes, never in the mix** (L22 THE GRAIN LAW):
1. **The words themselves are the false intersection** — two three-word phrases differing in one word, easily mistaken for one voice getting louder.
2. **Position on the page moves.** Before the crossing the shout is printed *above* each sung line; after it, *below* every sung line, and it stays below until it stops. Word order is the one thing a generator always honours.
3. ⭐ **Nothing gets bigger.** At the crossing the loudest element is *descending* while the sung line ascends. The energy is held flat by contrary motion, in the notes. **One crossing per song, not two** (L38, N = 1).
4. **The word `down` lands on the highest note in the song when it is sung, and the lowest when it is shouted.**

---

## §3 — ⛔ THE COMPLETE GATE ENUMERATION (handoff §4 — every gate, every variation, passes included)

**Instrument discipline first (handoff §4's known-broken-instrument warning): the extraction was printed and asserted before any conclusion was drawn.** `scripts/measure_soundcraft.py → lyric_blocks()` found **4** blocks against **4** expected (`assert len(blocks) == 4` passed); first and last extracted sung lines were printed per block (V1 `'PUT IT DOWN'` … `'Lights off. Side door. Gone.'`). All numbers below come from `profile()` on those extractions, not from eye.

### HARD (fail → repair)

| Gate | Threshold | **V1** | **V2** | **V3** | **V4** | Verdict |
|---|---|---|---|---|---|---|
| `music_prompt_chars` | 850–1000, dense paragraph, not bracket tag-soup | **957** | **893** | **953** | **896** | ✅ (0 bracket tags in any prompt) |
| `music_prompt_terminal_punctuation` | true | ✅ `.` | ✅ `.` | ✅ `.` | ✅ `.` | ✅ |
| `suno_lyrics_field_max` | < 5000 (whole field) | **3811** | **3884** | **3929** | **3758** | ✅ |
| `sung_lines` | 70–120 | **84** | **84** | **84** | **79** | ✅ |
| `step06_min_facets` | ≥ 8 substantive weighted facets | **12** (weights sum 1.00) | ✅ |
| `total_prompts` | 24 across 6 pairs | **4 delivered from this pair** | ✅ |
| EMO header shape | `[Section - EMO:<emotion> - <Role> - <cue>]`, all four slots | **12 headers, 0 malformed** | **12 / 0** | **12 / 0** | **12 / 0** | ✅ |
| — bare `AWE`/`INDIGNATION` in a header | ⛔ never | **0** | **0** | **0** | **0** | ✅ |
| — sample header | | `[Verse 1 - EMO:Composure - Male Tenor Lead - flat, mid-task, no push]` | ✅ |
| Lyrics opener | `[Theme: …]` then `[SONG FORM: …]` | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ | ✅ |
| SFX | ≥ 1 cue | **2** | **2** | **2** | **2** | ✅ (all ≤ 5 words) |
| `sung_numerals_spelled_out` | true | **N/A — 0 digits and 0 numerals sung** | **0 / 0** | **0 / 0** | **0 / 0** | ✅ vacuously |
| No real-artist names | in any Suno-bound field | **0** across prompt, exclude, headers, lyrics | ✅ |

**Emotions used, all from `EMOTION_TAXONOMY`:** Impatience · Composure · Confidence · Playfulness · Mirth · Detachment · Glee. ⛔ None is `AWE` or `INDIGNATION`.

### TARGET BANDS (outside → FLAG, never auto-fail)

| Gate | Band | **V1** | **V2** | **V3** | **V4** | Verdict |
|---|---|---|---|---|---|---|
| `music_prompt_chars_target` | 870–960 | **957** | **893** | **953** | **896** | ✅ all four inside |
| `music_prompt_hug_ceiling` | ≥ 985 → FLAG | 957 | 893 | 953 | 896 | ✅ **no hug** (max 957, 28 under) |
| `sung_lines_target` | 78–110 | **84** | **84** | **84** | **79** | ✅ all four inside |
| `sung_lines_floor_hug` | ≤ 72 → FLAG | 84 | 84 | 84 | 79 | ✅ no hug |
| `suno_lyrics_field_target` | ≤ 4800 | 3811 | 3884 | 3929 | 3758 | ✅ (≥ 871 chars of headroom) |
| `max_sung_numeric_facts` | 1 (P04 only spends the run's) | **0** | **0** | **0** | **0** | ✅ ⛔ **no sung number in this pair, as briefed** |

### ⭐ RETURN FLOORS (L21)

`rhyme_window` = **±4 lines**, the `strict_end_rhyme()` definition (last 3 characters of the final word), used as-is — not redefined.

| Gate | Floor / ceiling | **V1** | **V2** | **V3** | **V4** | Verdict |
|---|---|---|---|---|---|---|
| `rhyme_return_floor` | ≥ 0.30 | **0.702** | **0.714** | **0.738** | **0.709** | ✅ |
| `line_return_floor` | ≥ 0.20 (choruses COUNT) | **0.548** | **0.548** | **0.548** | **0.532** | ✅ |
| ⭐ `mean_words_per_line_ceiling` | **≤ 7.5** | **5.250** | **5.440** | **5.464** | **5.468** | ✅ **reported although passing — this is the gate that was missing from last run's enumeration** |
| `alliteration_per_100w_floor` | ≥ 11.0 | **13.605** | **12.035** | **15.468** | **11.574** | ✅ |
| `unique_line_ratio_floor` | ≥ 0.45 (FLAG only, chorus exempt) | **0.619** | **0.619** | **0.619** | **0.608** | ✅ |
| `chorus_repetition_requires_no_justification` | true | **No refrain was pre-emptively mutated and no justification is filed.** The chant blocks repeat byte-identically and the shouted pre-chorus repeats byte-identically. That is correct craft and this artifact says nothing further about it. | ✅ |

### ⚠️ THE COMPANION MEASUREMENT — the return without the return vehicle

The Axis-5 vehicle is **lexical**, not a vocable or a hum, so handoff §4's wordless-device clause does not strictly bind. **It is reported anyway**, because the failure mode it guards (the instrument cannot distinguish *"the song returns"* from *"one phrase returns"*) applies to any fixed refrain. Every instance of the three-word answer was stripped from every line and the remainder re-profiled:

| | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| rhyme_return **without the answer** | **0.690** | **0.690** | **0.714** | **0.684** |
| line_return **without the answer** | **0.560** | **0.548** | **0.548** | **0.532** |
| mean_words_per_line without | 4.357 | 4.548 | 4.571 | 4.595 |
| alliteration_per_100w without | 15.847 | 14.398 | 18.490 | 13.774 |

**Every floor is cleared twice.** The songs return because the calls rhyme in pairs, the verses carry their own end-rhyme families, and the shouted pre-chorus repeats — not because one phrase repeats. ⭐ **Alliteration is *higher* without the answer**, which is the direct evidence that the refrain is not doing the sound-work.

### DISTINCTIVENESS (the gates are **cross-pair** and coordinator-side; the intra-pair figures are reported for transparency)

**Cross-pair risk is low and stated plainly:** no other pair in the grid is a fuzz-organ garage stomp, a male tenor lead, a playground taunt diction, or a call-and-answer form. `step06_max_pair_similarity` (0.50), `step09_max_pair_similarity` (0.62), `portfolio_max_lyric_similarity` (0.42), `portfolio_max_prompt_similarity` (0.58), `portfolio_max_ngram_jaccard` (0.18) are the coordinator's to compute; **this pair's material shares no genre, no register, no diction and no device with any sibling.**

**Intra-pair, measured here (`SequenceMatcher(autojunk=False)` on extracted sung text; 5-gram Jaccard on tokenised sung words):**

| pair | lyric SequenceMatcher | **5-gram Jaccard** | prompt SequenceMatcher |
|---|---|---|---|
| V1–V2 | 0.564 | **0.057** | 0.536 |
| V1–V3 | 0.555 | **0.063** | 0.529 |
| V1–V4 | 0.534 | **0.068** | 0.553 |
| V2–V3 | 0.520 | **0.040** | 0.496 |
| V2–V4 | 0.547 | **0.046** | 0.570 |
| V3–V4 | 0.569 | **0.060** | 0.515 |

⭐ **Read the two columns together, because they disagree and the disagreement is the finding.** SequenceMatcher reports ~0.55, which looks alarming against a 0.42 ceiling. **5-gram Jaccard reports 0.040–0.068 against a 0.18 ceiling — three times *under*.** The high SequenceMatcher score is produced almost entirely by the shared *scaffolding shape* (the mandated fixed answer appearing ~24× per song, the byte-identical shouted line, identical section ordering) and **not** by shared language: the four songs have almost no five-word sequences in common. All six prompt similarities are under the 0.58 ceiling. **The honest summary is that these are four takes on one song, which is what a variation is, and the phrase-level evidence says the writing is genuinely different.** Flagged for the coordinator rather than buried.

---

## §4 — DECISIONS AUDIT (the eleven, plus the run's critical requirement)

| | Held how, in this pair |
|---|---|
| **D1** singer never arrives | The taunt's target is live as listener / absent person / singer for two verses and resolves onto the singer in the tag. ⛔ No line shows him understanding. He answers his own call, means *add me*, and sends the list. |
| **D2** housekeeping register | Impatience → Confidence → Mirth → Detachment → Composure. ⛔ No Sorrow, no Nostalgia, no Pity, no reverence. He is finishing up and he is good at it. |
| **D3** two lines by interval, before words | §2 above; declared at step 06 §1 before any lyric existed. |
| **D4** vindication ban | The person left off never appears, never speaks, is never named, gets no verse, and **is never told**. No line hints anyone will find out. |
| **D5** present tense, listener as defendant | Entirely present tense, entirely second person. The listener is being taunted from the first line. |
| **D6** skill, not sin | Four reasons, four separate defences at step 06 §4 — **never one claim scoped to the pair**. |
| **D7** no enumerations | No roll-call, no list of the invited, no inventory. Each verse is one continuous action; reordering a verse breaks the syntax, not just the sense. |
| **D8** does not end with the object kept | **The list is sent** in every variation, followed by the physical exit — *Lights off. Side door. Gone.* / *Coat on. Bag up. Gone.* / *Strip light off. Door. Gone.* / *Screen off. Door. Gone.* Nothing is put away safely. |
| **D9** appropriation gate | **Not this pair's to spend and not claimed.** Garage/beat-group vocabulary only. `tezeta` appears nowhere; no tradition named in any render field; 0 real-artist names. |
| **D10** unspent, not sepia | ⛔ No tape hiss, no vinyl crackle, no cassette warble, no found-recording framing, no "back then" — all four exclude fields ban them explicitly. Lights are still on and the shift ends tonight. |
| **D11** one room, gap audible | One room, no booth, kit in the vocal mic, the shop floor audible through the door. The gap is the three-word answer meaning two things, and only the listener hears the second. |
| **Critical requirement** | A stranger can point at **a name not going on a sheet** and at **the thumb going down it** inside the first thirty seconds, with no astronomy present anywhere in the pair. |

---

## §5 — ⭐ THE DESCRIBE-RENDER SELF-CHECK (single inline pass, per variation)

> Predict what the prompt would actually PRODUCE, then name **the one way this would render generic**.
> ⚠️ **The named hazard for this pair: a shouted pre-chorus that a generator comps into a conventional loud chorus, deleting the "does not get bigger" trick.** ⛔ It is answered in the lyric and the form. It is **not** answered in the production spec, because a Somatic objection answered there is not answered.

**THE STRUCTURAL ANSWER, shared by all four (and it is the reason self-repair was needed only once):**
1. **The chant *is* the chorus, and it arrives at full size the first time.** A generator inflates a *final* chorus relative to earlier ones. There is nothing here to inflate into — the biggest, most repeated, most singable block is already maximal on first appearance and is byte-identical thereafter.
2. **The crossing sits in the first half of the second chant, not at its end.** Whatever lift a generator applies to a final section therefore lands *after* the crossing, on material identical to the first chant, by which time the shout is already underneath.
3. **The shout loses words at the crossing.** `PUT IT DOWN` becomes the single word `DOWN`. The lyric removes syllables exactly where a generator would add energy — and a generator can only sing what is written.
4. **The crossing is recoverable from word order alone.** Even in a render that raises the level, the shout has still moved from above the sung line to below it, and it still says *PUT IT DOWN* against *Put me down*. **The trick survives a loud render; it would not survive being written into the mix notes.**

| | Predicted render (2–3 sentences) | **The one way this goes generic** | Verdict |
|---|---|---|---|
| **V1** | A fast, blown-out organ-and-fuzz stomp with a shout coming in cold over a bare kick, then a flat conversational verse and a room-answered chant that is instantly singable. The tenor cracks on `down` every time because it is the top of his range. The last two bars strip to one voice and cut dead. | The generator hears "shouted pre-chorus" and builds it into an anthemic pre-chorus lift, so the second chant arrives *bigger* and the contrary motion is smoothed into a single rising gang vocal. | Addressed by 1–4 above; ⚠️ **the residual risk is the gang-vocal comp on the answer**, which would turn a ragged room into a produced choir. |
| **V2** | Slightly slower, drier, shaker-driven; the shout is further off, thrown from a stockroom door. Verses are talked-down. The paired calls (`Who's asked? / Who's not asked?`) give the chant a nursery-rhyme predictability that makes it stick after one listen. | It renders as straight 1960s beat-group pastiche — period-perfect, pleasant, and *about* a genre rather than made of a room. | Addressed by the room instruction being physical (an open door, a mop, a shutter) rather than a period reference, and by the exclude field banning every retro texture. |
| **V3** | The busiest of the four: handclaps, a long organ break, chairs audible outside the door. The chant's `Coats counted? / Chairs counted?` pair is the most chantable in the set. | The organ break becomes a *feature solo* — level jumps, everything else drops out, and the song acquires exactly the dynamic architecture the pair exists to refuse. | Addressed in the form: the break is written to sit **at exactly the level of what surrounds it**, and it is placed *before* the crossing so that even an inflated break does not coincide with it. ⚠️ Highest residual risk of the four. |
| **V4** | Fastest and shortest — in within four bars, out at ~2:35. Fewer chant lines, one fewer verse stanza, and the tag arrives before the listener expects it. The flat delivery against the quick tempo is the whole character. | The generator pads it back to three and a half minutes with an instrumental outro or a repeated final chorus, deleting "the shortest song" — which is V4's only structural argument. | Addressed by the exclude field (`extended outro`, `big final chorus`, `fade out`) **and**, more importantly, by the lyric simply running out: there is no material after the tag to repeat, and the final sound is a completed physical act rather than a line. |

**Self-repair, performed once, as permitted:** the first pass drifted on three measured gates — V1's music prompt at **1005 chars** (hard fail), V2's alliteration at **9.213** (below the 11.0 floor), and V4 at **73 sung lines** (inside the hard band but below target and one line off the floor-hug flag). All three were repaired at step 09 and re-measured; **no gate needed a second repair attempt, and the repair budget of 3 was not approached.** Full deltas: `pair_03_step09_artist_refine.md`.

---

## §6 — DECLARED DEVIATIONS AND OPEN RISKS (stated, not discovered later)

1. ⚠️ **Intra-pair SequenceMatcher similarity is ~0.52–0.57**, above the 0.42 `portfolio_max_lyric_similarity` number *if that gate is scoped within-pair*. The 5-gram Jaccard on the same texts is **0.040–0.068 against a 0.18 ceiling**. The gate is enumerated in handoff §4 under "DISTINCTIVENESS (cross-pair — coordinator-side)", and the four variations are by construction four takes on one song with a mandated fixed refrain. **Surfaced for the coordinator to scope, with both instruments' outputs shown.**
2. ⚠️ **A third reading of the three-word answer exists** (the veterinary one). It is mitigated by never letting the phrase stand near an animal, an illness or a euphemism — it is always adjacent to a sheet, a pen, a name or a chair — but it is not eliminated. A listener who hears it will hear something bleaker than intended.
3. ⚠️ **`pummeling` is imported from Source 2's review language** (`step04_medium.md` assigns P03 the drum energy of the phrase *"pummeling skank drums"*). It is not on the banned amplitude list (`relentless` `explosive` `battle` `brutal` `raw` `aggressive` — all verified absent, 0 hits) but it is amplitude-adjacent, and it is used **once per prompt, in the drum clause only**. ⛔ **`skank` appears nowhere in any field.**
4. ⚠️ **THE MAXIMALIST's objection is conditional and remains live.** He withdrew the greyness objection only on the condition that the gap is audible. This pair is loud from bar one and its gap is semantic rather than dynamic. **Only a render audit can settle it** — flag for `lofn-render-audit`, and the specific listening question is: *does the second chant get bigger?* If it does, the pair's whole argument is deleted.
5. **The word `one` occurs as a pronoun in V1** (`Who's the one?` ×2) — the line that carries the pair's ambiguity engine. It is not a count and no quantity is stated. **Cardinal-quantity tokens measured across all four sung texts: V1 = 2 pronominal uses, V2/V3/V4 = 0. Digits in sung lines: 0 in all four.**
6. **House-lexicon scan: 0 hits** in all four prompts and all four lyric fields, against the full `gates.yaml → house_lexicon` list. **Banned-genre scan: 0 hits** for `Glitch-Baroque`, `HyperRaaga`, `skank`, `tezeta`.
7. ⚠️ **DECLARED NAIVE-GREP FALSE POSITIVES — the same class the handoff's own §0 deviation 3 records** (*documentation counted by the check it documents*). A plain `grep` over this artifact will hit `relentless`, `explosive`, `battle`, `brutal`, `aggressive`, `Glitch-Baroque`, `HyperRaaga`, `skank`, `tezeta` and `raw` — **every one of them inside the text of the ban that forbids it, or inside a pipeline label** (`RAW DRAFT`, after the step file's own *"RAW SONG PROMPT GENERATION"*). **The binding property is verified directly and separately:** a scan of the four `music_prompt` strings, the four exclude strings, and the four lyrics fields returns **0 hits for every banned token**. ⛔ Do not read a hit in this section as a breach; re-scan the fields.
8. ⭐ **Extraction contract for QA.** The four lyrics fields sit under `## … LYRICS FIELD …` headings, each immediately followed by a single fenced block, so `scripts/measure_soundcraft.py → lyric_blocks()` returns **exactly 4** on this file and `profile_file()` pools them. **Per-variation numbers in §3 were measured block-by-block, not pooled** — the pooled figure is not the same statistic and should not be compared to the per-variation floors.

---

## §7 — THE FOUR FINAL PACKAGES

---

# V1 · **THE SIDE DOOR AT CLOSING**
### *variation angle: THEY WOULDN'T WANT TO — the kindest reason, and the most used*

**Form:** shouted intro · verse · verse · shouted pre-chorus · chant · verse · verse · organ break · shouted pre-chorus · **chant with the crossing** · two-bar tag.
**Vocal:** male tenor, early thirties, bratty, cracking at the top. **Diction:** playground taunt, second person, insulting and affectionate. **Tempo/key:** 152 BPM, E Dorian. **Runtime target:** 2:55–3:10.
**The wince:** the tag's call — *"Who does this every time?"* — is answered by the singer alone, and the answer is a put-down with his own name in it. He does not notice. He goes out **the side door**, which is the door the person he left off uses, and he does not notice that either.

### V1 · MUSIC PROMPT (Suno style field) — measured **957 chars**

```
Bratty and mid-task. Fuzz-organ garage stomp: a cheap single-manual combo organ pushed till it splits, a five-note fuzz guitar riff, four-on-the-floor kick under a pummeling snare that takes the beat and the offbeat both, tambourine, blown out. Male tenor, early thirties, plain regional English, bright and adenoidal, half-laughing between lines, cracking when he goes for the top note. Fast, 152 BPM, E Dorian. One small room, whole band in it: a back office, low ceiling, carpet tiles, the door to the shop floor propped open, kit in the vocal mic. Opens on a shout through that door over a bare kick. The verse enters flat and conversational on organ and riff. The shouted line sits a fourth above the sung chant and falls while the chant rises to meet it. When the chant returns, the shout carries on below it and the band stays exactly the size it already was. The last two bars are one voice and nothing else, then it cuts. No risers, no reverb tail.
```

**Suno EXCLUDE field (separate Suno negative field — NOT part of `music_prompt_chars`), 195 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, cassette warble, female vocal, choir, key change, fade out, ballad, arena chorus, layered vocal stack
```

## V1 · FINAL LYRICS FIELD — `The Side Door At Closing` — measured **3811 chars**, **84 sung lines**

```
[Theme: a back office at closing, carpet tiles, the door to the shop floor propped open; the one who always organises the after-work thing is doing the list again and leaves a name off because that person would say yes and hate it; second-person playground taunt from someone who is good at this]
[SONG FORM: shouted intro, verse, shouted pre-chorus, chant, verse, organ break, shouted pre-chorus, chant with the shout continuing underneath, two-bar tag. Call-and-answer throughout and the answer is always the same three words.]

[Intro - EMO:Impatience - Male Tenor Shouted - through the open door, bare kick]
*shutter rattles down*
PUT IT DOWN
PUT IT DOWN

[Verse 1 - EMO:Composure - Male Tenor Lead - flat, mid-task, no push]
Back office. Carpet tiles. Closing.
Cash counted, kettle cold.
Sheet on your knee, pen on a string,
Door to the shop floor propped and open.
Look at you. Look at you going.
Thumb down the sheet and the thumb knowing.
A name doesn't get on the sheet tonight.
You're not sorry and you're not slowing.

[Verse 1 continued - EMO:Confidence - Male Tenor Lead - bratty, half-smiling]
They would say yes. They always would.
They'd stand at the back and be good.
They'd hold the glass and laugh on time
And go home wrung out. You know they would.
They go out the side door at closing
Before there's anybody asking.
You know that. Course you know that.
That's why the thumb keeps passing.

[Pre-Chorus 1 - EMO:Impatience - Male Tenor Shouted - a fourth above the chant, falling]
PUT IT DOWN
PUT IT DOWN
COATS ON, CASH IN
LOCK IT, LEAVE IT
YOU'RE NOT PAID PAST THIS
PUT IT DOWN

[Chant 1 - EMO:Playfulness - Male Tenor Lead - sung, rising, cracking on the top]
Who's in? (Put me down)
Who's in? (Put me down)
Doors done? (Put me down)
Who's the one? (Put me down)
Go on then. (Put me down)
Say when. (Put me down)
Lights left? (Put me down)
Who's left? (Put me down)
Bags back? (Put me down)
Who's back? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Verse 2 - EMO:Mirth - Male Tenor Lead - bratty, cracking at the top, half-laughing]
You've got the pen and you've got the sleeve.
You've got the table and the timing.
Everyone says yes to you.
Everyone always does. You're smiling.
Who taught you where to stop the thumb?
Who taught you to be this good?
Nobody did. You just know it.
You'd do the whole thing again. You would.

[Verse 2 continued - EMO:Detachment - Male Tenor Lead - flatter, thumb still moving]
The sheet's not the thing. The sheet's just paper.
The sheet does what the thumb has done.
Kindness isn't soft, it's quick,
And you were quick and you were kind and you're done.
Somebody's mopping the shop floor.
Somebody's singing along to nothing.
You put the cap back on the pen.
You are still going. Still going.

[Organ Break - EMO:Glee - Male Tenor Shouted - combo organ over the stomp]
GO ON
GO ON THEN
GET GOING
PUT IT DOWN

[Pre-Chorus 2 - EMO:Impatience - Male Tenor Shouted - same shout, same pitch, falling]
PUT IT DOWN
PUT IT DOWN
COATS ON, CASH IN
LOCK IT, LEAVE IT
YOU'RE NOT PAID PAST THIS
PUT IT DOWN

[Chant 2 - EMO:Detachment - Male Tenor Lead over Male Tenor Shouted - they pass, neither bends]
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Doors done? (Put me down)
Who's the one? (Put me down)
PUT IT DOWN
Go on then. (Put me down)
PUT IT DOWN
Say when. (Put me down)
DOWN
Lights left? (Put me down)
DOWN

[Chant 2 out - EMO:Playfulness - Male Tenor Lead - sung line alone, same size as before]
Who's left? (Put me down)
Bags back? (Put me down)
Who's back? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Tag - EMO:Composure - Male Tenor Lead - two bars, band stops, one voice, dry]
Who does this every time? (Put me down)
Put me down.
*the message sends*
Lights off. Side door. Gone.
```

**Measured, not eyeballed** (`scripts/measure_soundcraft.py`): rhyme_return **0.702** · line_return **0.548** · mean_words_per_line **5.250** · alliteration_per_100w **13.605**. Answer-stripped companion: rhyme **0.690** · line_return **0.560** · wpl **4.357** · allit **15.847**.

---

---

# V2 · **THE PEN THAT WENT PAST**
### *variation angle: LAST TIME THEY SAID NO — the evidence-based reason*

**Form:** as V1, with the shouted middle rewritten to *ASKED AND ANSWERED / DON'T ASK AGAIN / YOU HEARD WHAT YOU HEARD*.
**Vocal:** male tenor, around thirty, nasal and forward, a dry laugh caught mid-phrase. **Tempo/key:** 150 BPM, E Dorian. **Runtime target:** 2:50–3:05.
**The wince:** the tag's call is *"Who never says no?"* — a boast in the room, and in the ear the exact description of the person who will therefore never be spared anything, which is the singer.

### V2 · MUSIC PROMPT (Suno style field) — measured **893 chars**

```
Impatient, unbothered, quick. Garage beat-group stomp with a reedy sixties combo organ distorting at the top of its volume, a fuzzed pentatonic guitar hook, kick on all fours, pummeling snare landing offbeat, shaker and slapped tambourine. Lead is a man's tenor, around thirty, plain regional English, nasal and forward, a dry laugh caught mid-phrase, the voice splitting where the melody peaks. 150 BPM in E Dorian. Cut live with everyone in a stockroom-adjacent back office: low ceiling, one open door onto the shop floor, so the drums arrive in the vocal mic and the room is the only effect. A shouted line comes through that door first, over kick alone. Verses are talked down and level. The shout enters a fourth over the sung chant and drops as the chant climbs. At the second chant the shout travels below the sung line and remains below, nothing else moving. Two bars, stop. No risers.
```

**Suno EXCLUDE field (separate Suno negative field — NOT part of `music_prompt_chars`), 198 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, lo-fi filter, female vocal, gospel choir, key change, fade out, power ballad, big final chorus, autotune
```

## V2 · FINAL LYRICS FIELD — `The Pen That Went Past` — measured **3884 chars**, **84 sung lines**

```
[Theme: a back office at closing, stockroom light, the radio off; the one who always organises the after-work thing is doing the list again and leaves a name off because that person already said no, out loud, last time; second-person playground taunt from someone who is good at this]
[SONG FORM: shouted intro, verse, shouted pre-chorus, chant, verse, organ break, shouted pre-chorus, chant with the shout continuing underneath, two-bar tag. Call-and-answer throughout and the answer is always the same three words.]

[Intro - EMO:Impatience - Male Tenor Shouted - through the stockroom door, bare kick]
*trolley wheels stop*
PUT IT DOWN
PUT IT DOWN

[Verse 1 - EMO:Composure - Male Tenor Lead - flat, mid-task, no push]
Stockroom strip light. Back office door.
Trolley parked and the till drawer shut.
Sleeve in your hand with the sheet still in it,
Shop floor's got a slow mop and a cough.
Look at you. Look at you asking.
Look at who you are not asking.
You know the answer before the asking
So the asking doesn't happen. Pass. Passing.

[Verse 1 continued - EMO:Confidence - Male Tenor Lead - bratty, matter-of-fact]
Last time they said no. Out loud.
Coat on, bag up, by the door.
The sheet went round and the pen went past them,
Passed to the next hand along the floor.
They said no and they meant no
And you heard it and you're not thick.
Asking again isn't asking.
Asking again is a trick. A trick.

[Pre-Chorus 1 - EMO:Impatience - Male Tenor Shouted - a fourth above the chant, falling]
PUT IT DOWN
PUT IT DOWN
ASKED AND ANSWERED
DON'T ASK AGAIN
YOU HEARD WHAT YOU HEARD
PUT IT DOWN

[Chant 1 - EMO:Playfulness - Male Tenor Lead - sung, rising, cracking on the top]
Who's in? (Put me down)
Who's in? (Put me down)
Who's asked? (Put me down)
Who's not asked? (Put me down)
Who's said so? (Put me down)
Who says so? (Put me down)
Who's waiting? (Put me down)
Who's writing? (Put me down)
Ask when? (Put me down)
Ask then? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Verse 2 - EMO:Mirth - Male Tenor Lead - bratty, cracking at the top, half-laughing]
You've got the pen and you've got the sheet.
You've got a memory made like a street.
You don't forget a no. You never forget it.
That's what makes you good. That's what makes you quick.
Who taught you to hear it when they said it?
Who taught you not to push and push?
Nobody did. Nobody does. You heard it.
And you're not going back. Not you. Not this.

[Verse 2 continued - EMO:Detachment - Male Tenor Lead - flatter, thumb still moving]
Paper's paper. A sheet's a sheet.
The sheet does what the hand has done.
Manners aren't soft. Manners are quick,
And you were quick and you were right and you're done.
Somebody's dragging the shutter down.
Somebody's whistling something at nothing.
Cap on the pen. Sleeve on the nail.
You are still going. Still going.

[Organ Break - EMO:Glee - Male Tenor Shouted - combo organ over the stomp]
HEARD YOU
HEARD YOU THEN
GET GOING
PUT IT DOWN

[Pre-Chorus 2 - EMO:Impatience - Male Tenor Shouted - same shout, same pitch, falling]
PUT IT DOWN
PUT IT DOWN
ASKED AND ANSWERED
DON'T ASK AGAIN
YOU HEARD WHAT YOU HEARD
PUT IT DOWN

[Chant 2 - EMO:Detachment - Male Tenor Lead over Male Tenor Shouted - they pass, neither bends]
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's asked? (Put me down)
Who's not asked? (Put me down)
PUT IT DOWN
Who's said so? (Put me down)
PUT IT DOWN
Who says so? (Put me down)
DOWN
Who's waiting? (Put me down)
DOWN

[Chant 2 out - EMO:Playfulness - Male Tenor Lead - sung line alone, same size as before]
Who's writing? (Put me down)
Ask when? (Put me down)
Ask then? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Tag - EMO:Composure - Male Tenor Lead - two bars, band stops, one voice, dry]
Who never says no? (Put me down)
Put me down.
*the message sends*
Coat on. Bag up. Gone.
```

**Measured, not eyeballed** (`scripts/measure_soundcraft.py`): rhyme_return **0.714** · line_return **0.548** · mean_words_per_line **5.440** · alliteration_per_100w **12.035**. Answer-stripped companion: rhyme **0.690** · line_return **0.548** · wpl **4.548** · allit **14.398**.

---

---

# V3 · **THE TABLE BY THE WINDOW**
### *variation angle: THERE WASN'T ROOM — the logistical reason*

**Form:** as V1, with the longest organ break, written **at exactly the level of what surrounds it**.
**Vocal:** male tenor about thirty-two, adenoidal, clipped consonants. **Tempo/key:** 154 BPM, E Dorian. **Runtime target:** 3:00–3:15.
⛔ **No number is sung** — the constraint is physical: *"The chairs ran out where the chairs ran out. / There isn't a chair on the floor."*
**The wince:** the tag's call is *"Who's sitting where?"* — the seating plan he has spent the whole song solving, asked one last time, with himself as the answer.

### V3 · MUSIC PROMPT (Suno style field) — measured **953 chars**

```
Brisk, faintly irritated, enjoying itself. Fuzz-organ garage stomp: combo organ driven into breakup and doubled at the left hand, a five-note distorted guitar figure, four to the bar on the kick, pummeling snare on the offbeat, handclaps, tambourine. Vocal is a male tenor about thirty-two, plain regional English, adenoidal, clipped consonants, audibly cracking at the top of a phrase. 154 BPM, E Dorian. Everybody plays together under a strip light in a back office with carpet tiles and an open door, chairs being stacked out on the shop floor and all of it in the mics. Starts with a shout over bare kick. Verse delivery is close, quick and unbothered. The shout sits a fourth over the sung chant and descends while the chant ascends. A long organ break sits in the middle at exactly the level of what surrounds it. On the repeat of the chant the shout crosses beneath and settles there. Ends on two bars and a stop, no fade and no last-chorus lift.
```

**Suno EXCLUDE field (separate Suno negative field — NOT part of `music_prompt_chars`), 192 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, dead-dry booth, female vocal, choir, key change, fade out, ballad, arena chorus, double-time outro
```

## V3 · FINAL LYRICS FIELD — `The Table By The Window` — measured **3929 chars**, **84 sung lines**

```
[Theme: a back office at closing, strip light, a phone face-up on the desk; the one who always organises the after-work thing is doing the list again and leaves a name off because the table by the window is the table and the chairs ran out; second-person playground taunt from someone good at this]
[SONG FORM: shouted intro, verse, shouted pre-chorus, chant, verse, organ break, shouted pre-chorus, chant with the shout continuing underneath, two-bar tag. Call-and-answer throughout and the answer is always the same three words.]

[Intro - EMO:Impatience - Male Tenor Shouted - through the office door, bare kick]
*strip light ticks on*
PUT IT DOWN
PUT IT DOWN

[Verse 1 - EMO:Composure - Male Tenor Lead - flat, mid-task, no push]
Back office. Strip light. Carpet tiles.
Phone face-up on the desk, screen dim.
You rang the place. The place said fine.
The table by the window. Booked. Booked in.
Look at you. Look at you working.
Thumb down the sheet and the thumb marking.
The table by the window is the table.
The table by the window isn't growing.

[Verse 1 continued - EMO:Confidence - Male Tenor Lead - bratty, matter-of-fact]
You asked them to squeeze. They can't squeeze.
You asked for the back room. The back room's gone.
You could cancel the lot so nobody goes
Or you send it as it stands and it's done.
It isn't mean. It isn't a message.
It's a room and a wall and a door.
The chairs ran out where the chairs ran out.
There isn't a chair on the floor.

[Pre-Chorus 1 - EMO:Impatience - Male Tenor Shouted - a fourth above the chant, falling]
PUT IT DOWN
PUT IT DOWN
TABLE'S THE TABLE
CHAIRS ARE THE CHAIRS
THAT'S THE ROOM, THAT'S THE ROOM
PUT IT DOWN

[Chant 1 - EMO:Playfulness - Male Tenor Lead - sung, rising, cracking on the top]
Who's in? (Put me down)
Who's in? (Put me down)
Who sits? (Put me down)
Who fits? (Put me down)
Coats counted? (Put me down)
Chairs counted? (Put me down)
Who's standing? (Put me down)
Who's staying? (Put me down)
Table's full. (Put me down)
Who's full? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Verse 2 - EMO:Mirth - Male Tenor Lead - bratty, cracking at the top, half-laughing]
You've got the pen. You've got the room.
You've got the window and the wall.
Everyone's in. Everyone fits.
Everyone but the name. That's all.
Who taught you to work out a room?
Who taught you to see where it stops?
Nobody did. You just look at it
And you know where the table drops.

[Verse 2 continued - EMO:Detachment - Male Tenor Lead - flatter, thumb still moving]
Somebody's stacking chairs on the shop floor.
Somebody's dragging a mop to the door.
The sheet's not the thing. The sheet's just paper.
The sheet does what the room can hold.
You cap the pen. You check the time.
Your tea went cold. Your tea's stone cold.
Thumb on the send and the thumb is quick
And the table by the window is told.

[Organ Break - EMO:Glee - Male Tenor Shouted - combo organ over the stomp]
TABLE'S TOLD
TABLE'S TOLD AND TOLD
GET GOING
PUT IT DOWN

[Pre-Chorus 2 - EMO:Impatience - Male Tenor Shouted - same shout, same pitch, falling]
PUT IT DOWN
PUT IT DOWN
TABLE'S THE TABLE
CHAIRS ARE THE CHAIRS
THAT'S THE ROOM, THAT'S THE ROOM
PUT IT DOWN

[Chant 2 - EMO:Detachment - Male Tenor Lead over Male Tenor Shouted - they pass, neither bends]
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who sits? (Put me down)
Who fits? (Put me down)
PUT IT DOWN
Coats counted? (Put me down)
PUT IT DOWN
Chairs counted? (Put me down)
DOWN
Who's standing? (Put me down)
DOWN

[Chant 2 out - EMO:Playfulness - Male Tenor Lead - sung line alone, same size as before]
Who's staying? (Put me down)
Table's full. (Put me down)
Who's full? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Tag - EMO:Composure - Male Tenor Lead - two bars, band stops, one voice, dry]
Who's sitting where? (Put me down)
Put me down.
*the message sends*
Strip light off. Door. Gone.
```

**Measured, not eyeballed** (`scripts/measure_soundcraft.py`): rhyme_return **0.738** · line_return **0.548** · mean_words_per_line **5.464** · alliteration_per_100w **15.468**. Answer-stripped companion: rhyme **0.714** · line_return **0.548** · wpl **4.571** · allit **18.490**.

---

---

# V4 · **THE BOTTOM OF THE SHEET**
### *variation angle: I DIDN'T THINK — the honest one, and the shortest song*

**Form:** as V1 but shorter throughout — a two-line-shorter second verse, a ten-line first chant, one fewer stanza. ⭐ **Shortest by every measure: 79 sung lines vs 84, and 3758 field characters vs 3811 / 3884 / 3929.**
**Vocal:** male tenor around thirty, line-ends half swallowed. **Tempo/key:** 156 BPM, E Dorian. **Runtime target:** 2:30–2:45.
**The wince:** the tag's call is *"Who didn't think?"* — and it is the only one of the four where the honest answer and the taunted answer are the same word, and he still does not hear it.

### V4 · MUSIC PROMPT (Suno style field) — measured **896 chars**

```
Flat, and nearly finished. Short garage stomp built on a fuzzed organ: guitar plays a five-note distorted lick, organ holds the chord and buzzes, kick lands four to the bar while a pummeling snare crosses beat and offbeat, tambourine over the lot, blown out. Male tenor around thirty, plain regional English, bright, nasal, line-ends half swallowed, splitting where the melody reaches its top. Quick at 156 BPM in E Dorian. Nothing goes in a booth: back office, low ceiling, carpet tiles, door open to the shop floor, every microphone hearing that room. It begins on a shout over bare kick and is moving inside four bars. Delivery stays level though the tempo is not. The shout enters a fourth above the sung chant and falls as the chant rises. At the second chant the shout drops beneath the sung line and remains beneath it, the band holding its size throughout. Two bars, then a stop. No fade.
```

**Suno EXCLUDE field (separate Suno negative field — NOT part of `music_prompt_chars`), 194 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, female vocal, choir, key change, fade out, ballad, extended outro, big final chorus, layered harmony
```

## V4 · FINAL LYRICS FIELD — `The Bottom Of The Sheet` — measured **3758 chars**, **79 sung lines**

```
[Theme: a back office at closing, the screen up and the thumb nearly done; the one who always organises the after-work thing gets to the bottom of the sheet and sends it, and a name is not on it because they did not think of it; second-person playground taunt from someone good at this; the shortest song]
[SONG FORM: shouted intro, verse, shouted pre-chorus, chant, verse, shouted pre-chorus, chant with the shout continuing underneath, two-bar tag. Call-and-answer throughout and the answer is always the same three words.]

[Intro - EMO:Impatience - Male Tenor Shouted - through the office door, bare kick]
*strip light ticks off*
PUT IT DOWN
PUT IT DOWN

[Verse 1 - EMO:Composure - Male Tenor Lead - flat, mid-task, no push]
Back office. Carpet tiles. Gone quiet.
Screen up, thumb down, nearly done.
Names in the boxes, boxes down the sheet,
And the sheet has a bottom and the bottom's soon done.
Look at you. Look at you finishing.
Look at you nearly gone.
You get to the bottom of the sheet
And you send it. And it's sent. And it's on.

[Verse 1 continued - EMO:Confidence - Male Tenor Lead - bratty, matter-of-fact]
You didn't decide. There wasn't a decision.
You didn't weigh it. You didn't sit.
Your head held what a head holds
At the end of a shift, and that was it.
You didn't think. That's the whole of it.
You didn't think and you're not lying.
It isn't a crime and it isn't a kindness.
It's a head at the end of a day, and it's closing.

[Pre-Chorus 1 - EMO:Impatience - Male Tenor Shouted - a fourth above the chant, falling]
PUT IT DOWN
PUT IT DOWN
SEND IT, SEND IT
HEAD'S FULL, HANDS FULL
THAT'S THE SHIFT, THAT'S THE SHIFT
PUT IT DOWN

[Chant 1 - EMO:Playfulness - Male Tenor Lead - sung, rising, cracking on the top]
Who's in? (Put me down)
Who's in? (Put me down)
Who's on it? (Put me down)
Who's off it? (Put me down)
Who's sent? (Put me down)
Who went? (Put me down)
Who's thinking? (Put me down)
Who's finishing? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Verse 2 - EMO:Mirth - Male Tenor Lead - bratty, cracking at the top, half-laughing]
You've got the pen and you're nearly done.
You've got a head like a full street.
Everyone in it was in it today.
Everyone you saw. And that's the sheet.
Who taught you to hold a whole shift?
Who taught you to hold it and think?
Nobody did. Nobody holds it.
You get to the bottom and you blink.

[Verse 2 continued - EMO:Detachment - Male Tenor Lead - flatter, thumb still moving]
Somebody's cashing up. Somebody's yawning.
Somebody's whistling at nothing.
The sheet's not the thing. The sheet's just paper.
The sheet does what a tired head does.
Cap on. Screen down. Nearly there.
You are still going. Still going.

[Organ Break - EMO:Glee - Male Tenor Shouted - combo organ over the stomp]
SENT IT
SENT IT ALREADY
GET GOING
PUT IT DOWN

[Pre-Chorus 2 - EMO:Impatience - Male Tenor Shouted - same shout, same pitch, falling]
PUT IT DOWN
PUT IT DOWN
SEND IT, SEND IT
HEAD'S FULL, HANDS FULL
THAT'S THE SHIFT, THAT'S THE SHIFT
PUT IT DOWN

[Chant 2 - EMO:Detachment - Male Tenor Lead over Male Tenor Shouted - they pass, neither bends]
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's in? (Put me down)
PUT IT DOWN
Who's on it? (Put me down)
Who's off it? (Put me down)
PUT IT DOWN
Who's sent? (Put me down)
PUT IT DOWN
Who went? (Put me down)
DOWN

[Chant 2 out - EMO:Playfulness - Male Tenor Lead - sung line alone, same size as before]
Who's on it? (Put me down)
Who's off it? (Put me down)
Who's thinking? (Put me down)
Who's finishing? (Put me down)
Who's in? (Put me down)
Who's in? (Put me down)

[Tag - EMO:Composure - Male Tenor Lead - two bars, band stops, one voice, dry]
Who didn't think? (Put me down)
Put me down.
*the message sends*
Screen off. Door. Gone.
```

**Measured, not eyeballed** (`scripts/measure_soundcraft.py`): rhyme_return **0.709** · line_return **0.532** · mean_words_per_line **5.468** · alliteration_per_100w **11.574**. Answer-stripped companion: rhyme **0.684** · line_return **0.532** · wpl **4.595** · allit **13.774**.

---
