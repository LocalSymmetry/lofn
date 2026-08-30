# PAIR 03 — STEP 09 · ARTIST-REFINED PROMPTS (the critic loop)
## `2026-08-08-daily-music` · THE WRONG INVENTORY · **P03 — THE ROTA**

**Continuity Payload Used:** frozen ICB, LF-sha `9b538e91…b224ed1a`, **142,900 B** · personality DNA **27,796 B** inlined · 18 baseline voices · 3 Hyper-Skeptics (6/12/18) · **15 Special Flairs present** · 3 debate configurations.
**Step file:** `skills/music/steps/09_Generate_Music_Artist_Refined.md` · **Input:** `pair_03_step08_generation.md` (raw) · **Output:** the refined set, carried to `pair_03_step10_revision_synthesis.md`.

> **Step-09 discipline, taken literally:** *"make as minimal a change as you can to get your desired result, allowing the hard work in previous parts of the process to shine through."* Three measured gate misses are repaired below and **nothing else was touched** except where a critic named a specific defect. Every change is itemised with its before and after.

⚠️ **Step-file instruction adapted, and the adaptation declared.** The step file asks for each prompt to be *"rewritten in a chosen artist's signature style."* ⛔ Under the run's D9 and the hard output contract, **no real-artist name may appear in any Suno-bound field**, and panel figures are *"after"* constructs whose source figures have not said, reviewed or endorsed anything here. The refinement is therefore performed **in the voice of two seats from the supplied Panel Ledger**, and no name of any kind enters a prompt, an exclude field, a header or a lyric. **No new panel was invented.**

---

## §1 — THE TWO CRITICS, CHOSEN AND JUSTIFIED (from the supplied 18)

**Critic A — THE HARD SECTION** *(after Thomas Brenneck, seat 7, Medium Panel).* Guitarist/founder in an analogue, live-to-tape, band-in-a-room lineage; temperament hostile to anything that needs a screen to exist. **Why him for this pair:** P03's entire production argument is that size comes from the room and never from a build, and his is the seat that said so in the baseline debate — *"the size does not come from the arrangement, it comes from the room the tape was made in."* He is also the seat that read the differentiation note aloud and named six close-miked vocals in a dead booth as what the house has actually been shipping.

**Critic B — THE SMALL ROOM** *(after Bob Boilen, seat 16, Context & Marketing Panel).* Creator of a stripped format; audience-first; allergic to anything that needs a press release to land. **Why him for this pair:** he set the run's format test — *"could this be sung at a desk, unamplified, and still work?"* — and this is the ACCESSIBLE arm, where that test is the whole assignment. A chant either travels without explanation or it does not.

⚠️ **Both Hyper-Skeptics with live claims on this pair were also heard:** **THE TRADITION CUSTODIAN** *(after Wynton Marsalis, seat 6)*, whose suspicion is not withdrawn and whose condition is D3; and **THE MAXIMALIST** *(after Kamasi Washington, seat 12)*, whose greyness withdrawal is conditional on the gap being audible.

*Panel voices are model-generated interpretive constructs, each "after" a named source figure's published work. No statement is a quotation of, or endorsement by, the named person.*

---

## §2 — THE CRITIQUE, IN THE CRITICS' VOICES

**THE HARD SECTION, on all four raw prompts:** The room instruction is right and it is the only part of these I would not change — a propped door, carpet tiles, a low ceiling, the kit arriving in the vocal mic. That is a real place and a generator can hear it. **What is wrong is V1, and it is wrong in a boring way: it is over the cap.** A thousand and five characters against a thousand. I do not care that it reads well; a prompt that does not fit is not a prompt. And the excess is all connective tissue — *"the band plays in one small room, a back office with a low ceiling and carpet tiles and the door…"* is four clauses doing one clause's work. Cut the joins, keep the nouns.

**Second point, and this one is craft rather than arithmetic.** All four of these say the same thing in the same order in nearly the same words. That is what happens when four takes come out of one session, and it is fine in a room — it is not fine on a page a machine reads, because four near-identical prompts will give you four near-identical records and the variation was the whole point of doing four. Change the *language*, not the band.

**THE SMALL ROOM, on the lyrics:** I applied the desk test to all four chants and three of them pass on the first read — I could answer them without being taught, which is the only test that matters for a call-and-answer. **V2 is the one that does not sit right, and I could not say why until I counted.** Its calls are the flattest of the four: *Who said yes / Who says yes* is two ways of saying nothing, and the verses have gone smooth in the mouth. The instrument agrees with me — **9.213 against a floor of 11.0.** That is not a taste note, it is a measurement, and it means the words have stopped having edges.

**Third defect, mine:** **V4 is meant to be the shortest song and it has been made short by removing structure rather than by being short.** Seventy-three lines is inside the hard band and one line off the flag. Shortest should be *decisive*, not *thin* — give it its break back and let it end early because it has finished, not because it ran out.

**THE TRADITION CUSTODIAN, briefly:** the two lines are named by interval, they cross once, and neither is altered. **That was my condition and it is met, at step 06, before a word existed.** My suspicion is not withdrawn and I am not required to withdraw it. What I will say is that the crossing is written in the notes and the words and not in a mix note, which is the first time this room has done that without being reminded.

**THE MAXIMALIST:** loud from the first bar to the last, no ramp, and the gap is semantic rather than dynamic. I accept that on paper. ⚠️ **My objection is live again if the render inflates the second chant**, and nothing written here can settle that. Send it to the render audit and ask one question: *does the second chant get bigger?*

---

## §3 — THE REPAIRS (three gates, one attempt each; budget of 3 not approached)

### REPAIR 1 — `music_prompt_chars`, V1: **1005 → 957.** ⛔ Hard fail → ✅ inside target band (870–960).
Two clauses compressed; **no content removed.**
- `The band plays in one small room, a back office with a low ceiling and carpet tiles and the door to the shop floor propped open, kit bleeding into the vocal mic.` → `One small room, whole band in it: a back office, low ceiling, carpet tiles, the door to the shop floor propped open, kit in the vocal mic.`
- `At the second chant the shout keeps going underneath the sung line and the band stays exactly the size it already was. Two bars alone at the end, then a clean stop.` → `When the chant returns, the shout carries on below it and the band stays exactly the size it already was. The last two bars are one voice and nothing else, then it cuts.`

### REPAIR 2 — `alliteration_per_100w`, V2: **9.213 → 12.035.** ⛔ Below floor 11.0 → ✅ clear.
Eight edits, all of them putting consonant pairs back into lines that had gone smooth. **No line was added for the metric's sake; every edit also does lyric work.**

| Raw | Refined | What it also fixes |
|---|---|---|
| `Stockroom light. Back office door.` | `Stockroom strip light. Back office door.` | names the light, which is the pair's Flair 9 |
| `Trolley parked, radio off.` | `Trolley parked and the till drawer shut.` | a shut till is a closing-time fact; a radio being off is not |
| `Sleeve in your hand with the sheet inside it,` | `Sleeve in your hand with the sheet still in it,` | *still* is the point — it has not gone round yet |
| `Shop floor's got a mop and a cough.` | `Shop floor's got a slow mop and a cough.` | pace |
| `So the asking doesn't happen. Passing.` | `So the asking doesn't happen. Pass. Passing.` | the pen and the act, in one word |
| `You've got a memory like a street.` | `You've got a memory made like a street.` | built, not inherited |
| `You don't forget a no. You log it.` | `You don't forget a no. You never forget it.` | removes filing-cabinet diction, which was drifting toward D10 |
| `Nobody did. You just heard it` | `Nobody did. Nobody does. You heard it.` | the anaphora the chant already uses |

**Two further V2 edits made on the critics' notes rather than the metric:**
- Calls `Who said yes? / Who says yes?` → **`Who's said so? / Who says so?`** — a playground taunt (*says who?*) instead of two ways of saying nothing, and it rhymes on itself so the room can predict the second call.
- Verse-2-continued opened `The sheet's not the thing. The sheet's just paper.` → **`Paper's paper. A sheet's a sheet.`** — the line was identical to V1's and the critic's cross-take note applies.
- Organ break `GO ON / GO ON THEN` → **`HEARD YOU / HEARD YOU THEN`** (V2's own reason), and in V3 → **`TABLE'S TOLD / TABLE'S TOLD AND TOLD`** (V3's own reason). The break had been the same in three variations.

### REPAIR 3 — `sung_lines_target`, V4: **73 → 79.** ✅ Inside 78–110, and clear of the `sung_lines_floor_hug` 72 flag.
Six lines restored as *structure*, not padding: V4's own organ break (`SENT IT / SENT IT ALREADY / GET GOING / PUT IT DOWN`) and two chant-out calls (`Who's on it? / Who's off it?`). ⭐ **V4 remains the shortest song by every measure — 79 lines against 84, and 3758 field characters against 3811 / 3884 / 3929.**

### THE TWO NUMERAL RESIDUES (not a `gates.yaml` breach; a breach of this pair's own ⛔ NO SUNG NUMBER)
- V2: `Asking twice isn't asking twice. / Asking twice is a trick.` → **`Asking again isn't asking. / Asking again is a trick. A trick.`** and `hear it the first time` → **`hear it when they said it`**.
- V3: `Everyone but the one.` → **`Everyone but the name.`** — and it is better, because *the name* is the thing that does not go on the sheet.
- V4: `the bottom's the one` → **`the bottom's soon done`**.
- V1: `Who's the one?` **retained.** It is a pronoun, not a count, and it is the line the pair's whole ambiguity runs through. Declared at step 10 §6 rather than silently kept.

### THE CROSS-TAKE DIVERGENCE (the Hard Section's second point)
All four music prompts were rewritten with different vocabulary and different clause construction while holding the band, the room, the interval and the crossing identical. **Measured effect on intra-pair prompt similarity: worst pair 0.739 → 0.570; all six pairs now under the 0.58 ceiling.** V1 also lost its shared verse line (`the hand has done` → `the thumb has done`).

---

## §4 — POST-REPAIR MEASUREMENT (re-run, not asserted)

Extraction printed and asserted first — **4 blocks found, 4 expected.**

| | V1 | V2 | V3 | V4 | Gate |
|---|---|---|---|---|---|
| `music_prompt_chars` | **957** | **893** | **953** | **896** | 850–1000 hard · 870–960 target · ✅ all four in target, none ≥ 985 |
| `alliteration_per_100w` | 13.605 | **12.035** | 15.468 | 11.574 | ≥ 11.0 ✅ |
| `sung_lines` | 84 | 84 | 84 | **79** | 70–120 hard · 78–110 target ✅ |
| `rhyme_return` | 0.702 | 0.714 | 0.738 | 0.709 | ≥ 0.30 ✅ |
| `line_return` | 0.548 | 0.548 | 0.548 | 0.532 | ≥ 0.20 ✅ |
| `mean_words_per_line` | 5.250 | 5.440 | 5.464 | 5.468 | ≤ 7.5 ✅ |
| `suno_lyrics_field` | 3811 | 3884 | 3929 | 3758 | < 5000 hard · ≤ 4800 target ✅ |
| sung numerals / digits | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | ⛔ no sung number in this pair ✅ |

**No gate required a second repair attempt.** The complete enumeration — every gate in handoff §4, passes included — is in `pair_03_step10_revision_synthesis.md` §3.

---

## §5 — THE REFINED MUSIC PROMPTS AND EXCLUDE FIELDS (full text)


**V1 · The Side Door At Closing** — 152 BPM, E Dorian.

**MUSIC PROMPT — 957 chars, ends on terminal punctuation, 0 bracket tags:**

```
Bratty and mid-task. Fuzz-organ garage stomp: a cheap single-manual combo organ pushed till it splits, a five-note fuzz guitar riff, four-on-the-floor kick under a pummeling snare that takes the beat and the offbeat both, tambourine, blown out. Male tenor, early thirties, plain regional English, bright and adenoidal, half-laughing between lines, cracking when he goes for the top note. Fast, 152 BPM, E Dorian. One small room, whole band in it: a back office, low ceiling, carpet tiles, the door to the shop floor propped open, kit in the vocal mic. Opens on a shout through that door over a bare kick. The verse enters flat and conversational on organ and riff. The shouted line sits a fourth above the sung chant and falls while the chant rises to meet it. When the chant returns, the shout carries on below it and the band stays exactly the size it already was. The last two bars are one voice and nothing else, then it cuts. No risers, no reverb tail.
```

**Suno EXCLUDE field (separate negative field, not counted in `music_prompt_chars`) — 195 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, cassette warble, female vocal, choir, key change, fade out, ballad, arena chorus, layered vocal stack
```


**V2 · The Pen That Went Past** — 150 BPM, E Dorian.

**MUSIC PROMPT — 893 chars, ends on terminal punctuation, 0 bracket tags:**

```
Impatient, unbothered, quick. Garage beat-group stomp with a reedy sixties combo organ distorting at the top of its volume, a fuzzed pentatonic guitar hook, kick on all fours, pummeling snare landing offbeat, shaker and slapped tambourine. Lead is a man's tenor, around thirty, plain regional English, nasal and forward, a dry laugh caught mid-phrase, the voice splitting where the melody peaks. 150 BPM in E Dorian. Cut live with everyone in a stockroom-adjacent back office: low ceiling, one open door onto the shop floor, so the drums arrive in the vocal mic and the room is the only effect. A shouted line comes through that door first, over kick alone. Verses are talked down and level. The shout enters a fourth over the sung chant and drops as the chant climbs. At the second chant the shout travels below the sung line and remains below, nothing else moving. Two bars, stop. No risers.
```

**Suno EXCLUDE field (separate negative field, not counted in `music_prompt_chars`) — 198 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, lo-fi filter, female vocal, gospel choir, key change, fade out, power ballad, big final chorus, autotune
```


**V3 · The Table By The Window** — 154 BPM, E Dorian.

**MUSIC PROMPT — 953 chars, ends on terminal punctuation, 0 bracket tags:**

```
Brisk, faintly irritated, enjoying itself. Fuzz-organ garage stomp: combo organ driven into breakup and doubled at the left hand, a five-note distorted guitar figure, four to the bar on the kick, pummeling snare on the offbeat, handclaps, tambourine. Vocal is a male tenor about thirty-two, plain regional English, adenoidal, clipped consonants, audibly cracking at the top of a phrase. 154 BPM, E Dorian. Everybody plays together under a strip light in a back office with carpet tiles and an open door, chairs being stacked out on the shop floor and all of it in the mics. Starts with a shout over bare kick. Verse delivery is close, quick and unbothered. The shout sits a fourth over the sung chant and descends while the chant ascends. A long organ break sits in the middle at exactly the level of what surrounds it. On the repeat of the chant the shout crosses beneath and settles there. Ends on two bars and a stop, no fade and no last-chorus lift.
```

**Suno EXCLUDE field (separate negative field, not counted in `music_prompt_chars`) — 192 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, dead-dry booth, female vocal, choir, key change, fade out, ballad, arena chorus, double-time outro
```


**V4 · The Bottom Of The Sheet** — 156 BPM, E Dorian.

**MUSIC PROMPT — 896 chars, ends on terminal punctuation, 0 bracket tags:**

```
Flat, and nearly finished. Short garage stomp built on a fuzzed organ: guitar plays a five-note distorted lick, organ holds the chord and buzzes, kick lands four to the bar while a pummeling snare crosses beat and offbeat, tambourine over the lot, blown out. Male tenor around thirty, plain regional English, bright, nasal, line-ends half swallowed, splitting where the melody reaches its top. Quick at 156 BPM in E Dorian. Nothing goes in a booth: back office, low ceiling, carpet tiles, door open to the shop floor, every microphone hearing that room. It begins on a shout over bare kick and is moving inside four bars. Delivery stays level though the tempo is not. The shout enters a fourth above the sung chant and falls as the chant rises. At the second chant the shout drops beneath the sung line and remains beneath it, the band holding its size throughout. Two bars, then a stop. No fade.
```

**Suno EXCLUDE field (separate negative field, not counted in `music_prompt_chars`) — 194 chars:**

```
orchestral, strings, synth pad, trap hats, EDM riser, gated reverb, tape hiss, vinyl crackle, female vocal, choir, key change, fade out, ballad, extended outro, big final chorus, layered harmony
```

---

**Self-critique.** The critic loop earned its keep three times and each time the instrument found it before the ear did — the over-length prompt, the flat V2 verses, the thin V4. That is the correct order of operations and it is also slightly humbling: on a first read all four sounded fine. ⚠️ **What the loop could not test is the only thing that actually decides this pair** — whether a generator honours a shouted line that goes *under* instead of over. Everything available at this layer has been spent on it (word order, contrary motion, the chant already maximal at first appearance, the shout losing syllables at the crossing) and it is still a bet. ⛔ It was deliberately **not** answered in the production spec, because L22 says a Somatic objection answered there is not answered. **Route to `lofn-render-audit` under THE BLIND RULE with one question: does the second chant get bigger?**

*Step 09 complete. Step 10 is the shippable package.*
