# PAIR 01 — STEP 11 · ENHANCED FINAL PACKAGES

## `2026-08-08-daily-music` · THE WRONG INVENTORY · **P01 THE END OF THE REEL**

**Continuity.** Frozen ICB re-hashed on entry, LF-normalised: **142,900 B**, sha256 `9b538e912935bc585f512f2ec53c95f44826ce2443f0f60df8588831b224ed1a` — **matches the canonical value in `06_music_handoff.md` §0.** Personality DNA **27,796 B** inlined · 18 baseline seats · 3 Hyper-Skeptics.
**Inputs:** `pair_01_step10_revision_synthesis.md` (four final packages) · `06_music_handoff.md` (**overrides every step file**) · `skills/music/steps/11_Generate_Music_Enhancement.md`.
**Calibrated against the GOLDEN MOVE**, instructions only: a real place a body stands in named in the first thirty seconds; one wounding fact responded to and never recited; a mid-song turn the singer performs and does not understand; a register rotated away from anything this house shipped recently.
**This is a fresh context.** It did not write the step-10 packages and has no stake in defending them.

---

## §A — QUARANTINE CHECK, AND TWO CONTRACT CONFLICTS RESOLVED IN THE OPEN

**⛔ GOLDEN-OUTPUT QUARANTINE — checked rather than assumed, per instruction.** `skills/music/steps/11_Generate_Music_Enhancement.md` was read in full and grepped for embed instructions. **The 2026-08-07 patch holds.** The file's head-of-file note, its §"Context" item, and its two output clauses (lines 74–77, 251, 263) now all forbid Golden Song payloads and route them judge-side. **There is nothing in this step file to refuse, and nothing golden was loaded.** No past shipped lyric, style prompt or title entered this context; the two Golden Song names in the handoff were read as names and not looked up.
*This is reported as a negative result, not skipped. The 2026-08-06 incident happened because a step file contradicted itself and the pair followed the local instruction; the check is cheap and the failure was expensive.*

**Conflict 1 — the output contract.** The step file mandates a two-section shape (`## SUNO STYLE PROMPT` / `## SUNO LYRICS`, "missing either = FAIL") in its §1, and a numbered shape (`## 1. MUSIC PROMPT`, `## 1B. SUNO EXCLUDE PROMPT`) in its §"Output MUST include". **The step file contradicts itself.** `skills/music/scripts/validate_suno_packages.py` — which is what actually gates this artifact — parses only the numbered form. **Resolved to the numbered form**, per the coordinator's normalisation requirement and L30 (the run-level instruction beats the step file). All six step-10 artifacts failed the validator for exactly this reason: six agents, six heading conventions, and a step-10 contract that pinned none.

**Conflict 2 — Disc_Channel placement.** The step file's Gate 13a puts the five-channel Disc block **inside** the lyrics field and calls its absence a FAIL; the same file's line 148 tells you to move it to a production sidecar when the field is tight. ⭐ **Resolved by the single harness decision: the Disc_Channel block goes in `## 4. PRODUCTION SIDECAR`, outside the lyrics field, for all six pairs.** On 2026-07-24 keeping it inside pushed four packages to 5105–5514 chars against a 5000 hard cap and six agents each improvised a different workaround. Moving it bought **127–153 chars per variation**, which paid for the repairs below, and it let the block be written in the proper **five-channel** form (`Disc_Rhythm` … `Disc_Texture`) that would never have fitted inside the cap. The render field is the field that wins.

---

## §B — ⭐ THE ANDON CORD

**Verdict: 4 ENHANCED · 0 REJECTED.** One variation came close and the reasoning is set out in full, because "no rejects" is only worth reading if you can see what was nearly rejected and on what evidence.

| | Verdict | Basis |
|---|---|---|
| **V1 · THE LOG BOOK** | **ENHANCE** | Two L22 defects, both in the spec, both movable. The lyric already carries the crossing, the two lines, the direction and the stasis. |
| **V2 · THE DOOR OPEN** | **ENHANCE** | One prompt/lyric contradiction and one missing lyric fact. Both one line. |
| **V3 · WHAT THE ROOM DID** | ⚠️ **ENHANCE — near reject, four defects, see below** | Repairable inside the lyric and the form without touching the architecture. Had it not been, this was a step-07 return. |
| **V4 · DRIVING HOME** | **ENHANCE** | One subtractive event in the prompt. The lyric was already the strongest-built of the four. |

### ⚠️ V3 — the near reject, stated in full

The coordinator's instruction was conditional: *"If V3 cannot be made audible in the words, REJECT it."* The condition was tested, not assumed, and here is the arithmetic.

**The charge.** V3's step-10 rebuild installed its host with four things: an opening SFX, a rewritten `[Theme:]`, an intro header cue, and one added lyric line. **Three of those four are things a listener never hears** — the pair says so itself in its own self-critique, and it is right. And the run's critical requirement (`core_seed.md` §5, restated in handoff §3) is that *a stranger must be able to point at the thing being set aside and at the hand setting it aside, inside the first thirty seconds.*

**The measurement.** At 78 BPM in four, one bar is **3.08 s**. V3's prompt and intro cue both specified **eight bars of hum alone** — **24.6 s**, or **82 % of the thirty-second window**, spent on a wordless vocable. The first sung line landed at ~24.6 s and the second at ~27.7 s, so by second thirty a stranger had heard exactly this: *"Red light. Nobody counts it in. / Nobody lifts a hand."* **No hand doing anything. No object. No book.** The two lines that follow were *"A hand on the desk, not moving"* and *"The needle's not moving"* — two dead-still images in a row, the second a near-duplicate of the first.

**Why it is not a reject.** The failure was *not* in the variation's architecture. It was in **two form rules that were living in the production spec** — the eight-bar count and the "nothing at all underneath" — plus **one weak line**. All three are step-11 territory, and none of the fixes touch the AABA form, the impersonal register, or the rule that she does not say *I* until the final head:

1. **`A hand on the desk, not moving.` → `A hand in the van, filling in a book.`** The hand and the object are now named in the first A-section, in the words, at roughly **13 s**, with no *I* and no metaphor. It also improves the line: *"Nobody lifts a hand"* / *"A hand in the van, filling in a book"* is an antithesis where there were two still-lifes.
2. **The wordless opening drops from eight bars to two** (**6.2 s**), so the first sung line lands at ~6 s and the tag *"It's room tone on the tail"* at ~28 s — **both inside the window.**
3. ⭐ **The rule moved into `[SONG FORM:]`**, which is inside the render field, rather than sitting in the style paragraph where the renderer reads it as flavour. That is the L22 remedy applied to the exact defect L22 describes.

**What would have made it a reject.** If the only available repair had required her to be present in the live room, or to speak in the first person before the final head, that is a rebuild of the variation's premise — step 07, not step 11 — and it would have gone back. It did not require that. **The line that decided it was line three of verse one**, and it was already the weakest line in the variation.

**What is still not settled, and cannot be by a text gate.** Whether a listener can tell that the hum in V3's intro (hers, in the van) and the hum at V3's crossing (a stranger's, by the wall) are two different people humming the same figure. In the render they will both be the same contralto. **The words now say who is where; the voice may not.** This is a listening-pass question and it is flagged for `lofn-render-audit` under THE BLIND RULE, not resolved here.

---

## §C — ⛔ THE L22 SCAN

**THE GRAIN LAW: a Somatic or form objection answered in the PRODUCTION SPEC is not answered.** Every place a *form rule* was living in the music prompt or the production notes instead of in the lyric or the form. **Nine defects across four variations.** ⭐ Every repair is **accretive** — an element added, never a hole requested — because subtractive specs get smoothed and accretive ones survive.

| # | | Defect — the rule and where it was living | Repair |
|---|---|---|---|
| 1 | V1 | *"upright bass … that sits out one section **with the gap left unfilled**"* — a **hole requested** in the prompt and repeated in the Verse-4 header cue. A generator fills gaps; asking for silence is asking to be overruled. | Prompt → *"sits out one section **with the room audible instead**"*; cue → *"the room left audible in its place"*. The room is the song's subject, so the addition is the argument. |
| 2 | V1 | **Intro length unpinned.** *"It opens on a closed-mouth hum alone"* with no duration — the run's thirty-second legibility window left entirely to the generator's whim. | Pinned to **two bars, voice in by bar three**, and the same rule written into `[SONG FORM:]` inside the render field. |
| 3 | V2 | ⭐ **The prompt contradicted the lyric.** Prompt: *"**Halfway** a side door opens."* Lyric: the door opens in **Verse 1** (*"Door goes back. He gets in it"*). Two different songs in two fields of the same package. | Prompt → *"**Early** a side door goes back and gravel stays in the same microphone **to the end**"*. The field now describes the form the lyric actually has. |
| 4 | V2 | ⭐ **The second sound's refusal to move existed only in the prompt.** *"holds one low note … the whole way"* is in the style paragraph; the lyric never said the amp doesn't shift. **That refusal is the entire counterpoint** — without it the crossing is a harmony, not an unavailable unison. V1, V3 and V4 all carry it in the words; V2 was the one gap. | One line, in the visitor's mouth, where V2's crossing already lives: *"He means the amp. **It's done that all night.**"* A fact he supplies; she concludes nothing (D1 intact). |
| 5 | V3 | ⭐ **Eight-bar wordless opening**, specified in the prompt *and* the intro cue — **24.6 s of a 30 s window**. A form rule the lyric could not defend itself against. | Two bars; rule relocated into `[SONG FORM:]`. See §B. |
| 6 | V3 | *"with **nothing at all** underneath it"* — subtractive, stated **twice** (prompt + cue). | → *"with only **the air handling** under it"*. Same sparseness, produced by an addition the lyric already names in Verse 2 (*"The air comes on. It's on a clock."*). |
| 7 | V3 | **The host was inaudible.** Hand and object absent from the entire first head; installed by a Theme line, an SFX and a cue — three things nobody hears. | One line substitution in Verse 1. See §B. |
| 8 | V3 | ⭐ **The anti-chord-pad pin was missing here.** The pair's own declared risk #1 is the held organ note rendering as a chord pad — which deletes the second line and the crossing with it. V1's prompt pins *"at one steady level, same pitch throughout"*; **V3's did not**, and V3 is the sparsest and most pad-prone of the four. | Pin added to V3. `sustained organ chord pad, organ chord swell, string pad` added to **all four** exclude fields, where a thing you do not want belongs. |
| 9 | V4 | ⭐ *"one of the two low sounds **simply stops**"* — a subtractive event **with no cause in the prompt**. The lyric supplies the cause (*"Somebody pulls the fire door to"*); the style field asked for a disappearance, which a generator renders as a fade. | Prompt → *"a heavy door **is pulled to** on the lower of the two and only the hum carries on"*, and `fade-out` added to the exclude field. **Sounds do not stop. Somebody stops them.** |

### Scanned and cleared — reported because a scan that only lists hits is not a scan

- ⭐ **The crossing is in the lyric in all four**, with **both lines named, the direction named, and the second line's stasis named**: V1 *"Mine goes under it. It sits still. / It doesn't move for mine."* · V2 (repaired, #4) · V3 *"Theirs goes under it. Neither's moving."* · V4 *"Mine goes down under it. / It stays where it is."* The pair got the hardest thing right and did not need help with it.
- **The interval magnitude** (*a minor third*) stays in the instrumentation, which is where a pitch relationship belongs. The lyric carries the **event**; the prompt carries its **realisation**. That is the correct division, not a defect.
- **V4's declared risk #2** — *"the disappearance is inaudible because the second sound was never established"* — is answered **in the words three times before the crossing** (Verse 2 *"The hum's coming out the fire door"*, Verse 3 *"It's still coming out of it. / I'm still doing it, under it"*, Verse 4 *"Both of them going together"*). Verified, not taken on trust.
- **Accretive doctrine sentences** (*"It grows by addition, never by level"*, *"Everything added, nothing turned up"*, *"Brushes keep time the whole way and never stop"*) are genuine production instructions with no lyric home. **Reported as cannot-move**, and all three are phrased positively.

---

## §D — WHAT CHANGED, AND WHAT DELIBERATELY DID NOT

### ⭐ Prompt length — the discipline this step keeps failing

A refinement step has over-lengthened the prompt on **seven consecutive runs**. **The first drafts of all four prompts here came out at 981–1023 chars — three of them over the 1000 hard cap outright, the fourth four chars under the boundary-hug flag.** They were cut by **48–68 chars each** before anything else was done, and the cuts are recorded here rather than quietly absorbed.

| | step 10 | step 11 | Δ | Why |
|---|---|---|---|---|
| V1 | 958 | **939** | **−19** | Two repairs added, more words cut than added. |
| V2 | 957 | **948** | **−9** | The door-timing fix is shorter than the sentence it replaced. |
| V3 | 942 | **955** | **+13** | ⭐ Bought the anti-chord-pad pin (defect 8) and the air-handling substitution (defect 6). Paid for with two trims elsewhere. **Load-bearing, not decorative.** |
| V4 | 916 | **933** | **+17** | ⭐ Bought the door-closing cause (defect 9). The old clause named no agent; the new one does. |

**All four sit inside the 870–960 target band; none reaches the 985 boundary-hug flag.** Net across the pair: **+2 chars.** Two got shorter, two bought something specific and named.

### Lyric changes — four, total

| | Change | Why |
|---|---|---|
| V1 | `[Theme:]` *"A tape operator"* → *"The operator on the session"* | ⚠️ **A re-stat correction.** Step 10 §9 asserts *"the word 'tape' does not appear in any of the four lyrics at all."* **That claim is false.** It appears in V1's `[Theme:]` — measured, not eyeballed. It is a job title and not a D10 breach, but the Theme is the first thing a renderer reads and D10's retro trap is the run's named ⛔. Removed; the claim is now true. |
| V2 | Verse 5: *"He means the amp they left on. / Then it's going again under it."* → *"He means the amp. It's done that all night. / Then mine's going under it."* | L22 defect 4. Adds the stasis, keeps the section at eight lines, keeps it in his mouth. |
| V3 | Verse 1 line 3 | L22 defect 7 and the near-reject. See §B. |
| V3 | Verse 5: *"It's been on the whole night."* → *"It's done the same note all night."* | The amp was established as *present* but not as *unmoving*; **the same note** is the plainest available way to say a pitch that will not come to meet hers. Same syllable count, no musician-speak, still an observation and not a conclusion. |

### Structural rules moved into the render field

`[SONG FORM:]` in all four now carries **the wordless-opening length and the thirty-second legibility rule**. That is a form rule living in the form, inside the lyrics box, instead of in a style paragraph.

### EMO dramaturgy — one deliberate change, applied to all four

**Outro `EMO:Ennui` → `EMO:Absorption`.** Every other wordless-hum section in every variation is tagged `Absorption`; the Outro — the longest wordless hum in the record — was the sole inconsistency. It is also the right reading: **the hum has her, she does not have it.** The last emotional address in the song is now the one thing the song is actually about. All emotions verified against `EMOTION_TAXONOMY.md`: *Composure, Equanimity* (Serenity) · *Absorption* (Fascination) · *Detachment* (Apathy) · *Ennui, Listlessness* (Ennui) · *Impatience* (Anger) · *Warmth* (Happiness). ⛔ No bare `AWE` / `INDIGNATION` / `SYNTHESIS`.

### ⛔ What was deliberately NOT done, and why

**No new literary device was bolted on.** The step file asks for "at least one distinctive literary or structural device" per pair. **This pair already has four, all load-bearing, and adding a fifth would be decoration on a song whose whole argument is that nothing is added:**

- **Epistrophe** — a byte-identical three-line tag closing all seven A-sections of every variation. ⛔ Per `gates.yaml`, a byte-identical refrain **needs no justification and none is filed**; it was not pre-emptively mutated.
- **Antanaclasis** on the pair's own noun — V1's *"And that's the whole take. Nothing to take."*
- **Tag mutation exactly once** — V3's first two tag lines change in the final head, at the moment the singer arrives, and never again.
- **A wordless five-note hook** carrying the return without a word.

**No title was changed.** All four titles come out of their own refrains, which is where titles should come from. Churning them to look busy is the same failure as lengthening the prompt.

---

## §E — ⭐ THE COMPLETE GATE ENUMERATION (handoff §4 — every gate, including the passes)

All thresholds from `vault/gates.yaml`, cited not restated. Instrument: `scripts/measure_soundcraft.py → profile()` over `sung()`; prompt and field lengths measured on the exact strings below.
⭐ **Extraction asserted before conclusions**, per handoff §4's warning that these validators have failed in both directions: **V1 80 sung lines · V2 80 · V3 81 · V4 80**, of which **14 are vocable (hum) lines in each**. Every return figure is reported **twice — over all sung lines / over lexical lines only.**

### HARD (fail → repair)

| Gate | Threshold | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|
| `music_prompt_chars` | 850–1000, dense paragraph | **939** ✅ | **948** ✅ | **955** ✅ | **933** ✅ |
| `music_prompt_terminal_punctuation` | true | ✅ `.` | ✅ `.` | ✅ `.` | ✅ `.` |
| `suno_lyrics_field_max` | < 5000, whole field | **4191** ✅ | **4308** ✅ | **4405** ✅ | **4480** ✅ |
| `sung_lines` | 70–120 | **80** ✅ | **80** ✅ | **81** ✅ | **80** ✅ |
| `step06_min_facets` | ≥ 8 | **10** ✅ (step 06 §6, weights sum 1.00) | | | |
| `total_prompts` | 24 run-wide | **4 from this pair** ✅ | | | |
| EMO header shape, all four slots | taxonomy emotion, ⛔ never bare | ✅ **13/13** | ✅ 13/13 | ✅ 13/13 | ✅ 13/13 |
| — sample header | | `[Verse 5 - EMO:Equanimity - Solo Female Contralto - the crossing, vibraphone octave added, nothing removed]` | | | |
| Lyrics opener | `[Theme:]` then `[SONG FORM:]` | ✅ | ✅ | ✅ | ✅ |
| SFX | ≥ 1 | **2** ✅ | **2** ✅ | **2** ✅ | **2** ✅ |
| `sung_numerals_spelled_out` | true | ✅ | ✅ | ✅ | ✅ |
| No real-artist names in **any** Suno-bound field | true | ✅ | ✅ | ✅ | ✅ |

### TARGET BANDS (outside → FLAG, never auto-fail)

| Gate | Band | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|
| `music_prompt_chars_target` | 870–960 | **939** ✅ | **948** ✅ | **955** ✅ | **933** ✅ |
| `music_prompt_hug_ceiling` | ≥ 985 → FLAG | no flag ✅ | no flag ✅ | no flag ✅ | no flag ✅ |
| `sung_lines_target` | 78–110 | **80** ✅ | **80** ✅ | **81** ✅ | **80** ✅ |
| `sung_lines_floor_hug` | ≤ 72 → FLAG | no flag ✅ | no flag ✅ | no flag ✅ | no flag ✅ |
| `suno_lyrics_field_target` | ≤ 4800 | **4191** ✅ | **4308** ✅ | **4405** ✅ | **4480** ✅ |
| `max_sung_numeric_facts` | 1 (**P04 spends the run's one**) | **0** ✅ | **0** ✅ | **0** ✅ | **0** ✅ |

⚠️ **`sung_numerals_spelled_out` and the numeral count, stated precisely rather than repeated.** Step 10 claimed *"0 digits and 0 number-words in any sung line."* Re-measured: **0 digits and 0 cardinal or ordinal number-words** (`one … thousand`, `once`, `twice`, `first`, `second`, `third`) across all four. **The claim as written was slightly overstated**: the determiners **`both`**, **`neither`** and **`last`** do occur (V4 *"Both of them going together"*, V2 *"the last case"*). They are quantifiers, not numeric facts, and none is a number *responded to or recited* — but they are reported rather than swept, since the gate is about a number reaching a listener's ear. **⛔ No sung number in this pair. P04's one stands unchallenged.**

### ⭐ RETURN FLOORS — the L21 floors

| Gate | Floor / ceiling | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|
| `rhyme_window` | ±4 lines | used as `strict_end_rhyme()` defines it; not redefined | | | |
| `rhyme_return_floor` | ≥ 0.30 | **0.500** ✅ | **0.463** ✅ | **0.481** ✅ | **0.475** ✅ |
| ⭐ *rhyme_return — LEXICAL ONLY* | ≥ 0.30 | **0.424** ✅ | **0.379** ✅ | **0.403** ✅ | **0.394** ✅ |
| `line_return_floor` | ≥ 0.20, **choruses COUNT** | **0.438** ✅ | **0.438** ✅ | **0.407** ✅ | **0.438** ✅ |
| ⭐ *line_return — LEXICAL ONLY* | ≥ 0.20 | **0.318** ✅ | **0.318** ✅ | **0.284** ✅ | **0.318** ✅ |
| ⭐ **`mean_words_per_line_ceiling`** | **≤ 7.5** | **5.88** ✅ | **5.84** ✅ | **5.16** ✅ | **5.79** ✅ |
| ⭐ *words/line — LEXICAL ONLY* | ≤ 7.5 | **6.21** ✅ | **6.17** ✅ | **5.34** ✅ | **6.11** ✅ |
| `alliteration_per_100w_floor` | ≥ 11.0 | **18.94** ✅ | **18.42** ✅ | **19.14** ✅ | **17.93** ✅ |
| ⭐ *alliteration — LEXICAL ONLY* | ≥ 11.0 | **13.17** ✅ | **12.53** ✅ | **12.57** ✅ | **11.91** ✅ |
| `unique_line_ratio_floor` | ≥ 0.45, FLAG only, chorus-exempt | **0.637** ✅ | **0.637** ✅ | **0.667** ✅ | **0.727**\* ✅ |
| *unique_line_ratio — LEXICAL ONLY* | ≥ 0.45 | **0.727** ✅ | **0.727** ✅ | **0.761** ✅ | **0.727** ✅ |
| `chorus_repetition_requires_no_justification` | true | **honoured.** The three-line tag is byte-identical across all seven A-sections of every variation. ⛔ **No refrain was pre-emptively mutated and no justification is filed.** | | | |

\* V4 all-lines figure is **0.637**; the 0.727 in that cell is its lexical-only companion — both clear.

⚠️ **THE COMPANION MEASUREMENT, which is the point of this pair.** The return vehicle is a **vocable**, and `line_return` cannot distinguish *"the song returns"* from *"one syllable returns"* (measured 2026-08-06: stripping one hum dropped a run's `line_return` 0.289 → 0.044). **Stripping all fourteen hum lines drops `line_return` 0.438 → 0.318 and `rhyme_return` 0.500 → 0.424, and every floor still clears with margin in every variation.** ⭐ **The words carry the return unaided; the hum sits on top of a floor it does not need.**

### ⚠️ THE GATE THAT CARRIES NO INFORMATION

`scripts/check_human_subjects.py` was **not** run as evidence and its output is **not** reported in either direction — it returns `HOLD_FOR_HUMAN` on 100 % of correctly-written artifacts in this checkout because spaCy is absent and its regex fallback reads capitalised bracket tokens as person names (handoff §4). **The human-subject judgement is a content judgement, made per handoff §5:** every person in all four songs is invented; **Messier and Tempel appear nowhere, by name or allusion**; ⛔ **neither of the two real deaths in today's feed is alluded to, obliquely or otherwise.**

### Banned-token sweep — `music_prompt` + `exclude` + `title` + the whole lyrics field including every EMO header

Practitioner names **0** · tradition / place / language / mode names **0** · amplitude vocabulary (`relentless` `explosive` `battle` `brutal` `raw` `aggressive`) **0** — ⭐ **including inside the exclude fields, which are Suno-bound and were checked as such** · banned engines (`Glitch-Baroque`, `HyperRaaga`) **0** · `gates.yaml → house_lexicon`, all thirteen phrases, **0** · astronomy **0** · retro-trap tokens (`tape hiss`, `vinyl crackle`, `warble`, `sepia`, `found recording`) **0 in any prompt or lyric** — they appear **only inside the exclude fields**, which is where a thing you do not want belongs. ⭐ **The word "tape" now appears in none of the four lyrics fields** (it did in V1's; see §D).

---

## §F — DISTINCTIVENESS, WITH ITS FRAME STATED

⚠️ `portfolio_max_lyric_similarity 0.42` · `portfolio_max_prompt_similarity 0.58` are **BETWEEN-pair** ceilings (`gates.yaml`: *"these measure similarity BETWEEN pairs"*). Everything below is **WITHIN** this pair and is surfaced in advance rather than discovered.

| | V1~V2 | V1~V3 | V1~V4 | V2~V3 | V2~V4 | V3~V4 |
|---|---|---|---|---|---|---|
| lyric SequenceMatcher | **0.445** | 0.322 | 0.286 | 0.363 | 0.330 | 0.363 |
| **style prompt only** | 0.568 | 0.554 | **0.623** | 0.578 | 0.588 | 0.557 |
| exclude field only | 0.870 | 0.778 | **0.936** | 0.785 | 0.891 | 0.799 |
| prompt + exclude concatenated | 0.671 | 0.630 | **0.730** | 0.648 | 0.691 | 0.640 |
| lyric 5-gram Jaccard | 0.030 | 0.044 | 0.015 | 0.020 | 0.014 | 0.016 |

⭐ **A rise this tier caused, declared rather than buried.** Step 10 measured **prompt + exclude concatenated** at 0.537–0.611. This tier standardised the exclude fields into a shared failure-class blacklist — the D10 retro trap, the chord-pad risk, the fade-out — and the concatenated figure rose to **0.630–0.730**. **Splitting the two fields shows where the rise is: the creative field is unmoved (prompt-only 0.554–0.623, against step 10's concatenated 0.537–0.611), and the entire increase sits in the blacklist (0.778–0.936).**

**Four variations of one song share their failure modes, so their blacklists should be near-identical.** ⛔ **The guards were not thinned to improve a number** — that is optimising the instrument instead of the work, which is the exact behaviour handoff §4 warns about. **The recommendation to the coordinator is to measure the style field alone and treat the exclude field as a control surface, not as creative content.** The 5-gram figure, least fooled by shared scaffolding, sits at **0.014–0.044 against a 0.18 ceiling.**

---

## §G — LINEAGE & CREDIT · D9 APPROPRIATION GATE (1 of 2 permitted pairs)

⛔ Function used, **label never printed**. No practitioner, tradition, place, language or mode name appears in any Suno-bound field. **This block ships with the release, never in a render field.**

⭐ **All nine links were fetched and confirmed live in THIS session, not inherited.** The coordinator's instruction was to verify rather than trust, because **a dead credit is worse than no credit**.

| Practitioner / compiler | What was taken | Link — **fetched and confirmed 2026-08-08** |
|---|---|---|
| **Mulatu Astatke** (b. 1943, working) | the vibraphone-and-combo-organ instrumental grammar | <https://en.wikipedia.org/wiki/Mulatu_Astatke> ✅ resolves |
| **Getatchew Mekurya** (1935–2016) | a solo line standing complete before anything joins it | <https://en.wikipedia.org/wiki/Getatchew_Mekurya> ✅ resolves |
| **Mahmoud Ahmed** (b. 1941) | the low conversational placement, phrases landing behind the beat | <https://en.wikipedia.org/wiki/Mahmoud_Ahmed> ✅ resolves |
| **Alemayehu Eshete** (1941–2021) | the soul-side phrasing of the same scene | <https://en.wikipedia.org/wiki/Alemayehu_Eshete> ✅ resolves |
| **Emahoy Tsege Mariam Gebru** (1923–2023) | how much silence a low left hand can hold | <https://en.wikipedia.org/wiki/Emahoy_Tsege_Mariam_Gebru> ✅ · <https://mississippirecords.net/> ✅ both resolve |
| **Hailu Mergia** (b. 1946, working) | the combo organ as a second person in the room | <https://en.wikipedia.org/wiki/Hailu_Mergia> ✅ resolves |
| **Francis Falceto** · the **Éthiopiques** series (Buda Musique) | the document that carried the scene out | <https://en.wikipedia.org/wiki/%C3%89thiopiques> ✅ · <https://www.budamusique.com/> ✅ both resolve |

**One correction made during verification.** Step 10 printed the name as *Getatchew Mekurya* but linked `…/Getatchew_Mekuria`. **Both URLs resolve** — the second is a redirect — but the canonical article title is **Getatchew Mekurya**, so the direct URL is used here. **A redirect is not a dead link and this is not a defect**; it is corrected because a credit should point straight at its subject.

⚠️ **The Simon seat did not withdraw, and this block does not pretend otherwise:** *"the intent is never the issue."* Four of the six practitioners above are or were working artists whose scene supplied this pair's grammar. **The release note points upstream to them before it says anything about this song.**

---

## §H — MAJOR DEVIATIONS

- **Changed / refused / intensified:** (1) **Refused the step file's two-section output shape** (`## SUNO STYLE PROMPT` / `## SUNO LYRICS`) in favour of the numbered contract the validator actually parses; the step file contradicts itself and L30 gives the run-level instruction the win. (2) **Refused the step file's Gate 13a** requirement to place the Disc_Channel block inside the lyrics field; it goes in the sidecar, per the single harness decision. (3) **Refused to bolt on a new literary device** despite the step file's explicit instruction to apply one per pair. (4) **Refused to lengthen any prompt**, and cut all four first drafts by 46–68 chars each to get back inside the target band.
- **Reason:** the first two are self-contradictions inside the step file, resolved in writing rather than by silent choice. The third is a song whose entire argument is that nothing is added; a fifth device would be decoration, and the four existing ones are load-bearing. The fourth is a named, seven-run-old failure mode of this exact step.
- **Effect on Lofn uniqueness:** the four devices that make this pair a pair — the byte-identical tag, the wordless hook, the antanaclasis, the single tag mutation — are preserved untouched. **Nine form rules were moved out of the production spec and into the lyric, the form, or an accretive instruction**, which is the difference between a pair that has an argument and a pair whose argument gets smoothed away in the render.

---

## §I — SELF-CRITIQUE · WHAT QA AND THE RENDER AUDIT SHOULD ATTACK FIRST

1. ⭐ **The most useful thing in this artifact is defect 8**, and it is unglamorous: the pair named the chord-pad risk as its own #1 declared risk, wrote the pin into V1's prompt, and **then did not write it into V3's** — the sparsest, most pad-prone variation of the four. **A risk declared in a self-critique is not a risk mitigated in a field.** Worth checking on every pair in this run.
2. ⚠️ **Attack V3 first, still.** Its host is now audible in the words and the arithmetic is in §B, but **the question of whether a listener can distinguish two people humming the same figure in one contralto voice cannot be answered by any text gate.** If a blind listener cannot say who is where, V3 passed on paper again. This is the pair's live risk and it is named, not managed.
3. ⚠️ **The Maximalist's condition is still unsettled and nothing here settles it.** The pair is quiet, level and unhurried in a run that fears greyness, and nothing builds. He withdrew **conditionally**, on the gap being audible (D11). **Only `lofn-render-audit` under THE BLIND RULE can rule on that** — and per THE GRAIN LAW, it should judge the result, not the distance from intent.
4. ⚠️ **Attack the diction claim second**, as step 10 asked. *Zero metaphor* is asserted throughout and the closest call is V1's **`You can't print a room.`** — where *print* is the literal trade verb for selecting a take and the second meaning belongs entirely to the listener. **I believe it holds and I did not touch it.** If a re-stat rules it figurative, the pair's own diction rule has one exception and should say so rather than be quietly amended.
5. ⚠️ **The closest thing to a D1 breach in the enhanced set is V4's `Nothing to put it beside now.`** It is a state, not a principle — she reports that the reference sound is gone, never what the adjacency meant — and it is where the listener's wince lives. **It is also one word away from being the same defect the Custodian caught twice at step 10.** Left standing, flagged deliberately.
6. ⚠️ **`both` / `neither` / `last` survive in the sung lines** and step 10's "zero number-words" claim was overstated. They are determiners and not numeric facts, but the gate exists because a number reaching an ear is the failure. **If the coordinator's re-stat counts them, V4's `Both of them going together` is the line to cut.** I did not cut it, because it is the plainest statement of the two sounds coexisting in the whole pair.
7. **What I did not verify and am not claiming.** The distinctiveness figures in §F are **within-pair**; I have not seen the other five pairs and cannot speak to portfolio-level similarity. The render-side risks (chord pad, fade-out, ambient drift) are **mitigated in the fields and unproven in audio**. A describe-render check is still a prediction, not a measurement.

---

### VARIATION 1

## 1. MUSIC PROMPT

```text
Unhurried and matter-of-fact instrumental soul, a thirty-two-bar form stated three times. Vibraphone with the motor off, washed-out combo organ holding one low note at one steady level, same pitch throughout, upright bass in halves that sits out one section with the room audible instead, brushes on a slack snare, breezy acoustic guitar on the back of the beat. A woman in her forties, very low contralto, half-spoken at the bottom, no vibrato, dropped line-ends, breath in. Seventy-eight beats per minute, no key change. Five notes, a lowered second, no third, nothing resolving upward. Two bars of closed-mouth hum open it, band entering by ear off the downbeat, voice in by bar three. Midway a vibraphone octave arrives over that hum as it steps beneath the held organ note, settling a minor third under. Live in one small room the size of a van, doors shut, traffic outside. It grows by addition, never by level. End on the hum alone.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, cassette warble, wow and flutter, found-recording framing, sepia nostalgia pastiche, sustained organ chord pad, organ chord swell, string pad, risers, build into a final chorus, orchestral swell, key change, big drum fill, fade-out, gated reverb, wide stereo synth pads, belted vocal, heavy vibrato, melisma runs, choir stack, male lead vocal, child vocals, autotune gloss, spoken-word rap, trap hi-hats, sidechain pumping, four-on-the-floor kick, EDM drop
```

## 2. LYRICS

```text
[Theme: end of a 14-hour day. The operator on the session sits in a van with the doors shut and the engine off, logging takes in a ruled book on her knee. The last thing on the reel was rolled only to mark where the reel stopped — nobody played on it. She writes N.G. beside it and shuts the book. Present tense, trade nouns, no metaphor. She never finds out.]
[SONG FORM: AABA 32-bar song-form stated three times. Head, wordless hum hook, head with the crossing in its second A-section, hum hook, final head. The wordless opening is two bars and the first sung line lands inside the first half-minute. One fixed tempo, no key change, no build by level. The hook is a five-note wordless figure, not a chorus.]

[Intro - EMO:Composure - Solo Female Contralto - two bars of closed-mouth hum alone, band enters by ear]
*traffic passing outside*
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.

[Verse 1 - EMO:Composure - Solo Female Contralto - conversational, half-spoken at the bottom]
Door shut. Engine off.
Book open on my knee.
Car park lights come on a clock.
Then I can see him working the lock.
Just the tail left to take.
It's not a take.
So I put down N.G.,
in the log book on my knee.

[Verse 2 - EMO:Composure - Solo Female Contralto - same low placement, pen keeping time]
The column above it's full to the top.
Keepers, and the rest they ran again.
Every mark on it's mine.
Every line of it's fine.
Then this at the tail to take.
It's not a take.
So I put down N.G.,
in the log book on my knee.

[Bridge - EMO:Impatience - Solo Female Contralto - unrhymed, spoken closer to the mic]
Out the glass they're wheeling cases.
He wants the book before he locks up.
She wants the van keys.
I've got the tail and then I'm gone.
I'm not slow. I'm just last.

[Verse 3 - EMO:Detachment - Solo Female Contralto - band drops to organ and brushes]
So I run the tail back.
Wind it back to the black mark.
Nobody counts it in.
Nobody comes in.
Nothing anybody would want to take.
It's not a take.
So I put down N.G.,
in the log book on my knee.

[Hook - EMO:Absorption - Solo Female Contralto - wordless five-note hum, no words, vibraphone shadowing]
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.

[Verse 4 - EMO:Absorption - Solo Female Contralto - bass out for the whole section, the room left audible in its place, held organ note under the voice]
The air comes on. It's on a clock.
He moves in the chair. Wood on the floor.
Then he coughs. He'd been holding that
since the light went red on the door.
And that's the whole take. Nothing to take.
It's not a take.
So I put down N.G.,
in the log book on my knee.

[Verse 5 - EMO:Equanimity - Solo Female Contralto - the crossing, vibraphone octave added, nothing removed]
I'm humming while I write. (mm-hmm)
There's a hum on it as well. (mm-hmm)
Same sort of low. Same sort of nothing.
Mine goes under it. It sits still.
It doesn't move for mine. Nothing to take.
It's not a take.
So I put down N.G.,
in the log book on my knee.

[Bridge 2 - EMO:Composure - Solo Female Contralto - unrhymed, flat, entirely reasonable]
They rolled it to mark the end.
That's the whole job of it.
You can't print a room.
Anybody would put the same.
I'd put it again tomorrow.

[Verse 6 - EMO:Ennui - Solo Female Contralto - band returns by ear, no level change]
So I write it in the last box.
Letters. Doesn't take long.
Wrist on the edge of the book.
Cap off, cap on, and I don't look.
Nothing on it to take.
It's not a take.
So I put down N.G.,
in the log book on my knee.

[Hook 2 - EMO:Absorption - Solo Female Contralto - wordless five-note hum, no words, full band underneath]
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.

[Verse 7 - EMO:Ennui - Solo Female Contralto - last head, the book closes inside the section]
I'm still doing it in my teeth.
Pen pushed back in the spine.
Book shut. Both hands on the book.
In the door pocket. I don't look.
Nothing there to take.
It's not a take.
So I put down N.G.,
in the log book on my knee.

[Outro - EMO:Absorption - Solo Female Contralto - hum alone over the held organ note, players leave by ear]
*van door, then traffic*
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.
```

## 3. TITLE

The Log Book On My Knee

## 4. PRODUCTION SIDECAR

⛔ **NOT part of the Suno lyrics field.** Single harness decision for all six pairs: the Disc_Channel block lives here, outside the render field, after the title. The lyrics field carries Theme + SONG FORM + EMO headers + SFX + sung lines only.

**Disc_Channel**

```text
[Disc_Rhythm: brushed slack snare | pen tapping the ruled page | no fills | centre, close]
[Disc_Vocal: low female contralto | half-spoken at the bottom | breath left in, dropped line-ends | centre, one foot from the mic]
[Disc_Sub: upright bass in halves | out for one whole section | no sub reinforcement | centre, low]
[Disc_Pad: one held combo-organ note | one steady level, same pitch first bar to last | no chord movement | centre, behind the voice]
[Disc_Texture: vibraphone, motor off | octave added at the crossing | breezy acoustic guitar behind the beat | narrow, close]
```

**Locks** — 78 BPM · no key change · five-note set with a lowered second and no third · AABA 32-bar stated three times · one small room the size of a van, doors shut.

**Vocal fingerprint** — a woman in her forties, working, near the end of a fourteen-hour day. Very low contralto, half-spoken at the bottom of the range, no vibrato, line-ends dropped rather than landed, breath audible between phrases. She is mid-task and slightly irritated. **Reverence is the failure mode.**

**Body noise** (all three already in the lyric; none added at step 11)

| # | Location | Body noise | Function |
|---|---|---|---|
| 1 | Intro | closed-mouth hum, two bars, alone | Establishes line one of the counterpoint **before** the listener knows it is a line. |
| 2 | Hook / Hook 2 | wordless five-note figure | The adoptable hook. Carries the return without a single word, twice. |
| 3 | Verse 7 / Outro | *"still doing it in my teeth"*, then hum alone | She leaves with it and cannot account for it. The one thing she takes and the one thing she cannot log. |

**Production dramaturgy** — the held organ note is not atmosphere, it is the second voice; if it becomes a chord the song has no argument. The bass sitting out for one section is not a drop, it is the room being let in. The vibraphone octave at the crossing is the only addition in the record and it adds, it never swells.

**Style-axis lock** — accretive only. It grows by addition, never by level. No build, no riser, no final-chorus lift.

### VARIATION 2

## 1. MUSIC PROMPT

```text
Warm and talkative instrumental soul. Thirty-two bars, three times through, unhurried. A washed-out combo organ holds one low note in the left hand at one level throughout and answers the singer in the gaps with the right, like a second person. Vibraphone with the motor off, upright bass in halves, brushes on a slack snare, breezy acoustic guitar close. A woman in her forties, very low contralto, talking more than singing at the bottom, no vibrato, consonants bitten. Seventy-eight beats per minute, no key change. Five notes, a lowered second, no third, no phrase pulling home. Two bars of wordless hum begin it, voice in by bar three. Early a side door goes back and gravel stays in the same microphone to the end; midway the guitar plays out and a vibraphone octave arrives over the hum as it passes under the held note. Live in a small warm space the size of a van. Everything added, nothing turned up. Finish on the hum, organ still going.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, cassette warble, lo-fi filter, found-recording framing, sepia nostalgia pastiche, sustained organ chord pad, organ chord swell, string pad, filter sweep, riser, build into a final chorus, key change, orchestral swell, double-time outro, fade-out, gated reverb, wide stereo synth pads, gospel choir stack, belted vocal, heavy vibrato, melisma runs, male lead vocal, child vocals, autotune gloss, trap hi-hats, sidechain pumping, four-on-the-floor kick, EDM drop
```

## 2. LYRICS

```text
[Theme: same van, same night, the door opens. Somebody comes over from the load-out and stands in the doorway with a cup, talking, while she keeps filling in the log on her knee. She is nearly at the tail of the reel. The last thing on it was rolled only to mark where the reel stopped. She writes N.G. and shuts the book while he is still talking. Present tense, trade nouns, no metaphor.]
[SONG FORM: AABA 32-bar song-form stated three times. Head, wordless hum hook, head with the crossing in its second A-section, hum hook, final head. The wordless opening is two bars and the first sung line lands inside the first half-minute. One fixed tempo, no key change. The visitor's lines fall outside the rhyme so his speech sounds like speech against her sung column.]

[Intro - EMO:Composure - Solo Female Contralto - two bars of closed-mouth hum alone, band enters by ear]
*van door handle, then traffic*
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.

[Verse 1 - EMO:Composure - Solo Female Contralto - conversational, half-spoken at the bottom]
Door shut. Engine off.
Book open. Biro out.
Down to the tail of the reel.
Somebody's boot's on the back wheel.
Door goes back. He gets in it.
Nobody put a note on it.
So I put down N.G.
still somebody at the van door.

[Verse 2 - EMO:Warmth - Solo Female Contralto - organ right hand answers in the gaps after each line, left hand still holding the low note]
He's got a cup he's not drinking.
Leans on the frame, still talking.
Asks if the last case goes in the front.
It does. There's no room in the back.
I put the pen back on it.
Nobody put a note on it.
So I put down N.G.
still somebody at the van door.

[Bridge - EMO:Warmth - Solo Female Contralto - unrhymed, closer to the mic, ordinary speech]
Weather's turning, he says, for the drive.
The other van's gone already.
He asks am I nearly done.
I am. I'm on the last of it.
He doesn't go.

[Verse 3 - EMO:Detachment - Solo Female Contralto - band drops to organ and brushes]
Wind the tail down to the mark.
Nobody counts it in.
Nothing comes in after.
Just the room, and it's on it.
He's saying something. I've lost it.
Nobody put a note on it.
So I put down N.G.
still somebody at the van door.

[Hook - EMO:Absorption - Solo Female Contralto - wordless five-note hum, no words, vibraphone shadowing]
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.

[Verse 4 - EMO:Absorption - Solo Female Contralto - thinnest arrangement, held organ note under the voice]
Air on it. Then the chair.
Then a cough he'd been holding.
He's asking me something.
I nod. I've heard it again.
I write while he's saying it.
Nobody put a note on it.
So I put down N.G.
still somebody at the van door.

[Verse 5 - EMO:Equanimity - Solo Female Contralto - the crossing, vibraphone octave added, nothing removed]
He says I've got a hum on that. (mm-hmm)
I stop. I think he means me.
He means the amp. It's done that all night.
Then mine's going under it.
He's still talking. I'm nearly through it.
Nobody put a note on it.
So I put down N.G.
still somebody at the van door.

[Bridge 2 - EMO:Composure - Solo Female Contralto - unrhymed, flat, entirely reasonable]
The can's got to go back tonight.
The log has to match the can.
Nothing played, so nothing prints.
Anybody would put the same.
He'd put it. He'd put it quicker.

[Verse 6 - EMO:Ennui - Solo Female Contralto - band returns by ear, no level change]
Pen back on the page.
Letters in the last box.
He reads it upside down.
He says right. He gets down.
Doesn't say anything about it.
Nobody put a note on it.
So I put down N.G.
still somebody at the van door.

[Hook 2 - EMO:Absorption - Solo Female Contralto - wordless five-note hum, no words, full band underneath]
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.

[Verse 7 - EMO:Ennui - Solo Female Contralto - last head, the door closes inside the section]
Door goes. Cold comes in and out.
Cap on. Book shut. Both hands.
Still doing it down in my teeth.
He never said what he meant by it.
And he's gone, and I'm still on it.
Nobody put a note on it.
So I put down N.G.
still somebody at the van door.

[Outro - EMO:Absorption - Solo Female Contralto - hum alone over the held organ note, players leave by ear]
*boots on gravel, going*
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.
```

## 3. TITLE

Somebody At The Van Door

## 4. PRODUCTION SIDECAR

⛔ **NOT part of the Suno lyrics field.** Single harness decision for all six pairs: the Disc_Channel block lives here, outside the render field, after the title.

**Disc_Channel**

```text
[Disc_Rhythm: brushed slack snare | boots on gravel through the open door | no fills | centre, close]
[Disc_Vocal: low female contralto | talking more than singing | consonants bitten, breath left in | centre, one foot from the mic]
[Disc_Sub: upright bass in halves | steady, unemphatic | no sub reinforcement | centre, low]
[Disc_Pad: combo-organ left hand, one held low note | one level throughout | no chord movement | centre, behind the voice]
[Disc_Texture: combo-organ right hand answering in the gaps | vibraphone, motor off, octave at the crossing | breezy acoustic guitar close | narrow, close]
```

**Locks** — 78 BPM · no key change · five-note set with a lowered second and no third · AABA 32-bar stated three times · a small warm space the size of a van, side door open from the first head.

**Vocal fingerprint** — the same woman, with somebody standing in the doorway. She talks more than she sings. His lines fall outside the rhyme so his speech reads as speech against her sung column; she never stops working while he is talking.

**Body noise**

| # | Location | Body noise | Function |
|---|---|---|---|
| 1 | Intro | closed-mouth hum, two bars, alone | Her line, established before the visitor arrives to talk over it. |
| 2 | Verse 4 | *"a cough he'd been holding"* | The room's own body noise, on the reel — the thing the whole song is about, sung as an aside. |
| 3 | Verse 7 / Outro | *"still doing it down in my teeth"*, then hum alone | He goes; the hum does not. Neither of them noticed it and neither of them will. |

**Production dramaturgy** — the crossing is executed **by a man talking**, not by a fader. The organ's right hand answering in the gaps is what makes the room feel occupied; its left hand is the second line and must not move. Gravel enters the same microphone when the door goes back and stays there — nothing sweeps, nothing filters.

**Style-axis lock** — everything added, nothing turned up.

### VARIATION 3

## 1. MUSIC PROMPT

```text
Level and slow-moving instrumental soul, a thirty-two-bar form stated three times, no build. A held low combo-organ note runs first bar to last at one level, same pitch throughout. Vibraphone with the motor off, left to decay, upright bass entering late, out for whole sections, breezy acoustic guitar barely there. A woman in her forties narrates in a very low contralto, flat, half-speaking the bottom, no vibrato, no line weighted more than another. Seventy-eight beats per minute, no key change. Five notes, a lowered second, no third, never resolving upward. Two bars of hum open it with only the air handling under it, voice in by bar three. Midway the organ and brushes carry it alone with the air handling up in the same microphone; a second hum enters at the same low pitch, passes under the held note, and a vibraphone octave is added, nothing taken away. Brushes keep time the whole way and never stop. One small close room. It ends on the hum.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, cassette warble, found-recording framing, sepia nostalgia pastiche, ambient drone pad, untuned noise bed, field-recording collage, sound-collage montage, sustained organ chord pad, organ chord swell, string pad, string swell, risers, build into a final chorus, key change, orchestral swell, fade-out, gated reverb, wide stereo synth pads, belted vocal, heavy vibrato, melisma runs, choir stack, male lead vocal, child vocals, autotune gloss, trap hi-hats, EDM drop
```

## 2. LYRICS

```text
[Theme: the take itself, while it is happening. The red light is on in the live room and nobody is playing. It was rolled only to mark where the reel stopped. Impersonal, present tense, from inside the room: the air on its timer, a chair, a held cough, an amplifier left switched on. She is out in the van with the reel running and the book open the whole time, and does not say I until the last section. Trade nouns, no metaphor.]
[SONG FORM: AABA 32-bar song-form stated three times, narrated from inside the live room. The wordless opening is two bars; the hand and the book are named in the first A-section and the first sung line lands inside the first half-minute. The three-line tag is byte-identical every time; its first two lines change exactly once, in the final head, when the singer arrives. The hook is a five-note wordless figure, not a chorus.]

[Intro - EMO:Detachment - Solo Female Contralto - two bars of her hum in the van over the air handling, then the band enters by ear]
*van door, then air handling*
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.

[Verse 1 - EMO:Detachment - Solo Female Contralto - flat narration, no emphasis anywhere]
Red light. Nobody counts it in.
Nobody lifts a hand.
A hand in the van, filling in a book.
The needle's not moving.
Nobody's saying anything.
Nobody's playing.
The light's still red.
It's room tone on the tail.

[Verse 2 - EMO:Detachment - Solo Female Contralto - same flat placement, brushes only]
The air comes on. It's on a clock.
Nothing asked it to.
It gets into everything.
It gets on the reel.
It's the loudest thing running.
Nobody's playing.
The light's still red.
It's room tone on the tail.

[Bridge - EMO:Listlessness - Solo Female Contralto - unrhymed, closer to the mic]
Through the glass there's somebody with a cup.
Somebody's coat still on.
Nobody's told them to stop.
The tail's going round with nothing on it.
That's what it's for.
None of it goes in the book.

[Verse 3 - EMO:Ennui - Solo Female Contralto - organ and brushes carry it alone, air handling up in the same mic, bass sits out]
He moves in the chair. Wood on the floor.
The air made him do it.
He's been sat since the light went red.
He's not been asked for anything.
He's got nothing to be doing.
Nobody's playing.
The light's still red.
It's room tone on the tail.

[Hook - EMO:Absorption - Solo Female Contralto - wordless five-note hum, no words, vibraphone shadowing]
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.

[Verse 4 - EMO:Absorption - Solo Female Contralto - thinnest arrangement, held organ note under the voice]
Then he coughs.
He'd been holding that.
Held it since the light went red.
Held it while nothing was happening.
It's the only thing anybody's doing.
Nobody's playing.
The light's still red.
It's room tone on the tail.

[Verse 5 - EMO:Equanimity - Solo Female Contralto - the crossing, vibraphone octave added, nothing removed]
By the wall somebody's humming. (mm-hmm)
They don't know they're doing it. (mm-hmm)
The amp's still on behind them.
It's done the same note all night.
Theirs goes under it. Neither's moving.
Nobody's playing.
The light's still red.
It's room tone on the tail.

[Bridge 2 - EMO:Composure - Solo Female Contralto - unrhymed, flat, entirely reasonable]
They rolled it to mark the end.
Nothing else was going to happen.
Somebody has to stop the reel.
Somebody has to write it down.
That's the job. That's all it is.

[Verse 6 - EMO:Ennui - Solo Female Contralto - band returns by ear, no level change]
The tail runs out on the spool.
Nothing comes in after.
The needle isn't moving.
The air goes off on its clock.
That's the whole of it, ending.
Nobody's playing.
The light's still red.
It's room tone on the tail.

[Hook 2 - EMO:Absorption - Solo Female Contralto - wordless five-note hum, no words, full band underneath]
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.

[Verse 7 - EMO:Composure - Solo Female Contralto - the singer arrives, first person, tag lines change once]
Then it's me, in the van.
The whole reel in front of me.
Nothing on the tail of it.
Biro. Letters. Book shut.
Nothing anybody was playing.
Nobody played on it.
So I put down N.G.
It's room tone on the tail.

[Outro - EMO:Absorption - Solo Female Contralto - hum alone over the held organ note, players leave by ear]
*door pulled to, then traffic*
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.
```

## 3. TITLE

Room Tone On The Tail

## 4. PRODUCTION SIDECAR

⛔ **NOT part of the Suno lyrics field.** Single harness decision for all six pairs: the Disc_Channel block lives here, outside the render field, after the title.

**Disc_Channel**

```text
[Disc_Rhythm: brushed slack snare, never stopping | air handling on its timer | no fills | centre, close]
[Disc_Vocal: low female contralto | flat narration, no line weighted more than another | breath left in | centre, one foot from the mic]
[Disc_Sub: upright bass entering late | out for whole sections | no sub reinforcement | centre, low]
[Disc_Pad: one held combo-organ note, first bar to last | one level, same pitch throughout | no chord movement | centre, behind the voice]
[Disc_Texture: vibraphone, motor off, left to decay | octave added at the crossing | breezy acoustic guitar barely there | narrow, close]
```

**Locks** — 78 BPM · no key change · five-note set with a lowered second and no third · AABA 32-bar stated three times · one small close room, air handling audible.

**Vocal fingerprint** — the same woman, narrating flat and impersonally from inside the live room she is not in. No emphasis anywhere. She does not say **I** until the final head, where the first two lines of the tag change once and never again.

**Body noise**

| # | Location | Body noise | Function |
|---|---|---|---|
| 1 | Intro | closed-mouth hum, two bars, over the air handling | Her hum, in the van, bracketing a song she is otherwise absent from. |
| 2 | Verse 4 | the held cough, released | *"It's the only thing anybody's doing."* The room's one event in fourteen hours. |
| 3 | Outro | hum alone over the held note | The figure survives the reel, the book and the shift. Nobody claims it. |

**Production dramaturgy** — ⚠️ **this is the sparsest of the four and therefore the one closest to becoming weather.** Two guards, both accretive: brushes keep time the whole way and never stop; the air handling is a named element that is added, not a hole left open. A pulse that never stops cannot become ambience.

**Style-axis lock** — level throughout, no build, no arrival. The only addition in the record is the vibraphone octave at the crossing.

### VARIATION 4

## 1. MUSIC PROMPT

```text
Plainspoken and level instrumental soul, a thirty-two-bar form taken three times, nothing added at the end. Underneath everything two low sustained notes a minor third apart, one a held combo-organ note at one level. Above it, vibraphone with the motor off, upright bass in halves, brushes on a slack snare, breezy acoustic guitar close and dry. A woman in her forties carries it in a very low contralto, half-speaking the bottom, no vibrato, phrases landing behind the beat and never corrected. Seventy-eight beats per minute, no key change. Five notes, a lowered second, no third, never resolving upward. Two bars of closed-mouth hum open it, voice in by bar three. Midway a vibraphone octave joins the hum as it steps down beneath the held note; straight after, a heavy door is pulled to on the lower of the two and only the hum carries on. Cab-sized warm room, wet road outside. Fuller by addition, never louder. Ends on the hum.
```

## 1B. SUNO EXCLUDE PROMPT

```text
tape hiss, vinyl crackle, cassette warble, wow and flutter, found-recording framing, sepia nostalgia pastiche, sustained organ chord pad, organ chord swell, string pad, risers, build into a final chorus, power ballad chorus, key change, orchestral swell, big drum fill, fade-out, gated reverb, wide stereo synth pads, belted vocal, heavy vibrato, melisma runs, choir stack, male lead vocal, child vocals, autotune gloss, trap hi-hats, sidechain pumping, four-on-the-floor kick, EDM drop
```

## 2. LYRICS

```text
[Theme: after. The book is already shut and in the door pocket. She is in the cab in the yard with the engine running, the fire door propped open at the loading bay, and the room still going in there. She pulls out. The other sound is gone and only hers is left, and she cannot say where she got it. She would write the same thing again tomorrow. Trade nouns, no metaphor. She never places it.]
[SONG FORM: AABA 32-bar song-form stated three times. The wordless opening is two bars and the first sung line lands inside the first half-minute. The room is a van that is not moving for the first two heads and a van that is moving for the last. The crossing is in the second A-section of the second head, and immediately afterwards a door is pulled to on one of the two sounds. The hook is a five-note wordless figure, not a chorus.]

[Intro - EMO:Composure - Solo Female Contralto - two bars of closed-mouth hum alone, band enters by ear]
*heater, then a wet road*
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.

[Verse 1 - EMO:Composure - Solo Female Contralto - conversational, half-spoken at the bottom, every phrase landing just behind the beat]
Last case in. Doors banged.
Kit in the back, strapped tight.
Fire door propped. Nobody's shut off the light.
Room's still running in there.
Haven't opened it since I shut it.
Can't place it. Can't say it.
The book's shut in the door.
It's the hum in the cab.

[Verse 2 - EMO:Composure - Solo Female Contralto - same low placement, engine under everything]
Nothing to do now but go.
Hands on the wheel. Waiting. Not gone.
The hum's coming out the fire door.
Low, and it's coming through the floor.
Can't say which of them started it.
Can't place it. Can't say it.
The book's shut in the door.
It's the hum in the cab.

[Bridge - EMO:Impatience - Solo Female Contralto - unrhymed, closer to the mic]
Somebody bangs the back panel and waves.
Somebody's got the last of the cable.
Nobody's waiting on me now.
I could go. I'm going to go.
I'm sat here with the heater on.

[Verse 3 - EMO:Detachment - Solo Female Contralto - organ and brushes, bass sits out]
Reverse lights run on the wall.
Handbrake off. Nothing yet.
The propped door's still open.
It's still coming out of it.
I'm still doing it, under it.
Can't place it. Can't say it.
The book's shut in the door.
It's the hum in the cab.

[Hook - EMO:Absorption - Solo Female Contralto - wordless five-note hum, no words, vibraphone shadowing]
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.

[Verse 4 - EMO:Absorption - Solo Female Contralto - thinnest arrangement, held organ note under the voice]
Windows up. It's still in here.
Both of them going together.
Same low. Same sort of nothing.
Neither of them doing anything.
Nobody in the yard hearing it.
Can't place it. Can't say it.
The book's shut in the door.
It's the hum in the cab.

[Verse 5 - EMO:Equanimity - Solo Female Contralto - the crossing, vibraphone octave added, nothing removed]
Mine goes down under it. (mm-hmm)
It stays where it is. (mm-hmm)
Somebody pulls the fire door to.
And there's only mine left going.
Nothing to put it beside now.
Can't place it. Can't say it.
The book's shut in the door.
It's the hum in the cab.

[Bridge 2 - EMO:Composure - Solo Female Contralto - unrhymed, flat, entirely reasonable]
It was rolled to mark the end.
Nothing on it to write up.
I'd write the same tomorrow.
Anybody would.
Out the gate. You don't go back for that.

[Verse 6 - EMO:Ennui - Solo Female Contralto - band returns by ear, no level change]
A-road. Wipers on and off.
Radio on. Radio off.
Doesn't shift it. It's under all of it.
Sat at the lights with it going.
Green, and I'm still doing it.
Can't place it. Can't say it.
The book's shut in the door.
It's the hum in the cab.

[Hook 2 - EMO:Absorption - Solo Female Contralto - wordless five-note hum, no words, full band underneath]
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.

[Verse 7 - EMO:Ennui - Solo Female Contralto - last head, the indicator clicks off inside the section]
Off the roundabout. Indicator on.
Off again. Click, and it stops.
Nothing on the road. Nothing but it.
I try it against the radio. Nothing.
Give it up. Get on and drive.
Can't place it. Can't say it.
The book's shut in the door.
It's the hum in the cab.

[Outro - EMO:Absorption - Solo Female Contralto - hum alone over the held organ note, players leave by ear]
*wipers, then a wet road*
Mm-hmm, mm-mm-hmm,
mm-hmm, mm-mm.
Mm-hmm, mm-mm-hmm,
mm-mm, mm.
```

## 3. TITLE

The Hum In The Cab

## 4. PRODUCTION SIDECAR

⛔ **NOT part of the Suno lyrics field.** Single harness decision for all six pairs: the Disc_Channel block lives here, outside the render field, after the title.

**Disc_Channel**

```text
[Disc_Rhythm: brushed slack snare | wipers, intermittent | no fills | centre, close]
[Disc_Vocal: low female contralto | half-spoken at the bottom, every phrase behind the beat and never corrected | breath left in | centre, one foot from the mic]
[Disc_Sub: upright bass in halves | two low sustained notes a minor third apart underneath everything | no sub reinforcement | centre, low]
[Disc_Pad: held combo-organ note at one level | the lower of the two low sounds | no chord movement | centre, behind the voice]
[Disc_Texture: vibraphone, motor off | octave added at the crossing | breezy acoustic guitar close and dry | narrow, close]
```

**Locks** — 78 BPM · no key change · five-note set with a lowered second and no third · AABA 32-bar taken three times · cab-sized warm room; **parked for the first two heads, moving for the last** — the crossing happens while the vehicle is still stopped.

**Vocal fingerprint** — the same woman, an hour later, driving. Every phrase lands fractionally behind the beat and is never corrected. She has the tune and not its source, and she says so in the plainest available words.

**Body noise**

| # | Location | Body noise | Function |
|---|---|---|---|
| 1 | Intro | closed-mouth hum, two bars, alone | The hum is already in the cab before the song admits it is the subject. |
| 2 | Verse 3 | *"I'm still doing it, under it"* | Names her own line and its position without naming what it means. |
| 3 | Outro | hum alone over the held note | She drives off with it. Nobody in the world can tell her where she got it. |

**Production dramaturgy** — ⭐ **the disappearance must have a hand on it.** Both low sounds are named in the instrumentation and established in the lyric three times before the crossing; at the crossing a heavy door is pulled to by somebody who will never know what they ended, and only the hum carries on. **Sounds do not stop — somebody stops them.** No fade-out; the exclude field bans one.

**Style-axis lock** — fuller by addition, never louder. Nothing is added at the end.
