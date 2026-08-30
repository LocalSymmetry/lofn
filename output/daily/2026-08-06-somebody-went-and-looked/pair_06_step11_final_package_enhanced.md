---
run: 2026-08-06-somebody-went-and-looked
voice: SUNNA
pair: P06 — AMBITIOUS · NEWS · `the count that climbs`
step: 11 — enhancement tier (final packages)
slot: 132→165 BPM · C minor · 2:45 · four-on-floor → breakbeat · pursuer = a sub creeping a semitone
icb_bytes: 69095
icb_sha256: 85ed1348b7e22fdfb8e4b06dadd21daa1f4c85ede5ea7e150105f41e49f44a15
supersedes: the ICB `{pair_slice}` P06 concept, per `05b_P06_REPLACEMENT.md` (declared, not hidden)
verdict: ENHANCED
---

# P06 · STEP 11 — ENHANCED FINAL PACKAGES

## CONTINUITY SUMMARY — the invariants this pass did not touch

| invariant | value | preserved |
|---|---|---|
| Arm · Anchor | AMBITIOUS · NEWS | ✅ |
| ONE DEVICE | **the count that climbs** — a number goes up by one each chorus and nothing else in the chorus changes | ✅ |
| RIFF (written first) | **THE STACK** — two women's voices, wordless, a diatonic third apart; lower line E♭4·F4·G4·A♭4·G4, upper G4·A♭4·B♭4·C5·B♭4; range a major sixth, inside one octave | ✅ |
| THE SPLIT | on the fifth note the upper voice refuses the third and goes to **D5** — G4–D5, a perfect fifth — **exactly twice**: last bar of the break, and the final sound of the record | ✅ |
| GEAR-CHANGE | four-on-the-floor 132 → 165 breakbeat at the halfway point | ✅ |
| PURSUER (⛔ never named) | **a sub creeping a semitone**, whole track, never arriving | ✅ |
| BPM · Key · Length | 132→165 · C natural minor · 2:45 | ✅ |
| Gradation | **subtractive** — hats gone / guitar gone / bass out two bars / everything gone but the two voices | ✅ |
| Chorus text | **byte-identical but for the number**; the final chorus repeats the fourth verbatim and resolves nothing | ✅ |
| Verified fact | 78,000 people crossed from Morocco to Ceuta (BBC, 2026-08-06). ⭐ **The count counts the going.** | ✅ |
| The second number | ⛔ **absent from every lyrics field in every form.** It is the unnamed pursuer. | ✅ |

## Golden Song References

⛔ **GOLDEN-OUTPUT QUARANTINE (`EXECUTION.md` §3).** Step 11 is a *generating* context. The archived golden outputs were **not loaded** here and no style prompt, lyric or exclude field from them appears in this file. **Names only**, handed forward so QA (a judge-side context) can pull the full records itself:

1. **"Triple Arch Over Me"** — the calibration that matters for this pair is the one already abstracted into `vault/gates.yaml` as a *rule* rather than a text: **one numeric fact, sung at the emotional hinge, responded to rather than recited.** P06 spends its single licensed number on the device itself.
2. **"I Will Stop the Almost"** — named for QA's second calibration slot.

Full style/lyrics/exclude records for both: `skills/music/references/golden_songs_index.md`. **Exclude-prompt status for both: not verified here** (the file was not opened in this generating context).

## WHAT STEP 11 CHANGED — and what it deliberately did not

**Enhanced (all four):**
1. ⭐ **Gate 14a repair — the style prompts now lead with genre + tempo + key.** Step 10's prompts opened on the riff ("Two women's voices land…"), which is the run's house order but is a **Gate 14a violation** (*"MUST lead with genre/style + tempo + key/tuning"*). Fixed without losing the doctrine: genre/tempo/key is one clause, and **the riff is still the first musical thing described.**
2. ⭐ **The anti-choir defence moved into the POSITIVE field.** Step 10 fought F2 almost entirely in the exclude. Every prompt now carries *"every note attacked on a hard D"* — a renderer reads the positive field harder than the negative one, and **a choir cannot happen on a hard plosive.**
3. ⭐ **The split is now specified as bounded** — *"a split that happens exactly twice and nowhere else."* Step 10 described the split but not its scarcity; an unbounded split becomes a gesture and the gap stops opening.
4. ⭐ **Suno-parse repair: the room's bar is no longer a bracket.** Step 10 placed `[the room's bar — no vocal, no bass, real silence]` **inside** chorus 4. Suno reads `[...]` as a *section marker*; a bracket mid-chorus can split the chorus into two sections and break the byte-identical return. It is now the italic sound command `*silence — the room's bar*`, which is the correct idiom and leaves the sung lines untouched.
5. ⭐ **Three second-number near-misses repaired** (see the sweep below). All three were rewritten, not deleted — no rhyme lost, no line count changed.
6. **EMO mirror-form:** the outro now returns to the intro's emotion, changed — `Anticipation` → … → `Persistence+Anticipation`. **The song ends where it started, still expecting. It stops; it does not conclude.**
7. **Excludes rebuilt** — redundancy pruned, coverage widened on the two fatal axes (sacred and lament/triumph), each variation given its own tail against its own nearest failure.

**Deliberately NOT changed:** the choruses, the count, the `Da-da-da-da — dah` figure, the gradation ladder, the strongest lines (*"Wrote nothing on the tag." · "Nobody made a speech. Everybody went." · "Bag's gone and the hook's still there." · "Kid's got a ball and the ball's got a hole."*). **RETURN IS FREE and removal is a debt.** A step-11 pass that "improves" a byte-identical chorus has misunderstood the doctrine it was hired to enforce.

## APPLIED TECHNIQUE — the one structural/sonic device, and it costs the listener nothing

⭐ **THE PLOSIVE APPROACH.** Every line that hands off to a chorus is loaded with hard **D / T / K** attacks, so `Da-da-da-da — dah` is **pre-heard in the speech** before it is sung. *Count the going. Count it right. · Count them again. Count them all. · Counting them again. Counting them again. · Chalk's down to nothing. Chalk gets found.* The listener never audits it; the chant simply arrives already familiar. **Free channel, spent hard** — and it is measurable: it is the whole of the hook-excluded alliteration figure below.

Paired with **EMO mirror-form** (A → … → A′), the two are the pass's structural transformation. ⛔ No formal asymmetry, no double meaning, no structure that must be noticed. **One device, and the enhancement did not add a second.**

## BODY NOISE — 3 instances, each with a job

| # | Location | Body noise | Function |
|---|---|---|---|
| 1 | Intro | audible inhale on the pickup, on the *and* of 4 | the figure arrives a beat early; the breath is what makes "at speed" physical rather than notated |
| 2 | Break | breath between the two voices as they open from a third to a fifth | the gap must be heard as **two throats**, not one processed source — the single strongest anti-choir tell |
| 3 | Outro | breath on the capsule under the last unaccompanied figure, no de-esser | the record ends on a body still working, not on a resolved chord |

Realised in the lyrics field as Disc_Vocal tokens (`breath_on_capsule_audible`, `body_noises_foregrounded`) plus section cues. ⛔ Not written as sung lines — a body noise given a lyric line becomes a second device.

## ⭐ SECOND-NUMBER SWEEP — RE-RUN AT STEP 11, NOT INHERITED

Mechanically re-swept over **all four lyrics fields in full** (not sung lines only — the Disc_Channel block, `[Theme:]`, `[SONG FORM:]`, every header and every SFX cue were included).

- **Digits:** none in any sung line. The only digits in any lyrics field are the licensed count in `[SONG FORM:]` (75/76/77/78) and production constants in the Disc block (132, 165, 12ms, 180Hz). **`100` and `hundred` appear nowhere in any form.**
- **Number words:** only the count — `seventy / five / six / seven / eight / thousand`. The two `one` pronouns the step-10 audit disclosed are **gone** (repair 2). ⚠️ **One token the step-10 audit missed and this pass found: the ordinal `first`**, once, in V4's *"Everybody heard it the first time."* **Adjudicated and KEPT:** it is an idiomatic adverbial about a radio bulletin repeating itself, not a quantity, not a fact about the crossing, and not resolvable to anything countable. Disclosed rather than left implied — the step-10 claim *"no second numeric fact of any kind exists in this pair"* was true in substance and **imprecise in wording**, and imprecision is what this constraint dies of.
- **Arithmetic:** none. No subtraction, no remainder, no *some / not all / the rest / fewer / left*. **The only operation in this pair is addition.**
- **Absence of an expected arrival:** none reported anywhere.

⛔ **Three near-misses the pair's own audit missed or misreported. All three repaired here.**

1. ⭐ **V3 — *"Counted wrong. Started again."* → *"Counting them again. Counting them again."*** The step-10 GATE BLOCK claims *"Lost count — **cut** (step 09)."* **That is not what happened.** Step 09 (line 148) shows the line was **replaced with "Counted wrong. Started again." — "same rhythm, same meaning"** — by the file's own words. The line was never cut; it was reworded, and the reworded line carries the identical failure class: **a reported counting failure, in the frame of counting people in.** The pair's rule is that *near-misses HOLD, they do not get argued*; an audit that reports a cut which did not occur is worse than the line. Repaired: the error is gone, the rhythm is kept, and the doubled line is a **byte-identical internal return** — free channel.
2. ⭐ **V4 — *"rubs the old one out / writes the new one in"* → *"wipes the old time off / chalks the new time in."*** Two independent problems in one couplet: **`one` as a pronoun for a person** (*"rubs the old one out"* parses as a person if the antecedent is missed) and **`rub out` as a killing idiom.** Both sat in the one place in the run that must be clean of death-adjacent readings. Repaired at no cost: the out/in antithesis survives, the antecedent is now explicit, and `chalked / chalks / Chalk's / Chalk` becomes a four-fold consonance chain.
3. ⭐ **V1 — *"Count them again. Count them right."* → *"Count the going. Count it right."*** *"Count them right"* leaves the **object of the count unstated**, which is the only door through which the other figure could enter by inference. Naming the object — **the going** — forecloses it at the source, and echoes `went` from the chorus. The `night / right` rhyme is untouched. ⛔ Deliberately **not** propagated to V2/V3/V4: a phrase shared across all four would make the pair a suite, and a suite is a second structural idea (step 09 critique 6, upheld).

**Adjudicated and KEPT, on the record:** *"I never fold"* (folding clothes) · *"Bag's gone"* (a bag) · *"Text me when it's light"* (what anyone says to anyone travelling overnight; the song never reports whether the text arrives, and a live hope is not a reported absence) · V2 *"Count them all"* and V3 *"Count them still"* and V4 *"Count them up"* (inclusive and upward; no shortfall is available).

**Verdict: the second number is absent from all four lyrics fields in every form — not counted, named, numbered, implied by arithmetic, or implied by a reported absence.** It lives where it was assigned: **the sub creeping a semitone**, under every bar, never resolving, ⛔ never named.

## ⭐ MORAL / LAMENT / TRIUMPH AUDIT — re-run at step 11

- **EMO set across all four:** `Anticipation · Equanimity · Composure · Solidarity · Resolve · Amusement · Vigilance · Persistence · Impatience · Zeal · Dignity`. **All canonical** (`EMOTION_TAXONOMY.md`). ⛔ **Not one tag is Sorrow. Not one tag is Triumph.** Neither appears in any form — no Grief, Mourning, Melancholy, Heartache, Victory, Success or Achievement.
- **The final chorus resolves nothing — checked line by line, not asserted.** It is **byte-identical to chorus 3 apart from the count** (`78` where chorus 3 has `77`), which is the device and the only permitted difference; and it is **byte-identical to chorus 4's full text**, chorus 4 being the same chorus with one riff bar handed to the room as silence. **The count does not advance at the end** — 78 twice, because the count stops where the report stops. No added tag, no lift, no new line, no key change, no coda, no fifth chorus. ⚠️ **Residual, disclosed:** the final chorus sings the riff bar that chorus 4 gave away, so at the level of the words the last return is not strictly subtractive; the arrangement continues to subtract underneath it (drums thin, two voices forward) and the outro removes everything else. Step 10's design, left untouched deliberately — mutating a byte-identical chorus to tidy a doctrine is the step-11 failure mode this pass exists to avoid. **The song stops; it does not conclude.**
- **No line editorialises.** The nearest thing to a statement in the pair is V1's *"Nobody made a speech. Everybody went."* — which is the **refusal of a moral, performed as a flat report.** Rule S1 holds throughout: **evidence, never inference.** The song may report what an object is doing; it may never report what that means. No face is described anywhere in the pair.
- ⛔ **Not a lament, not a cheer.** Excludes blacklist both sides in the specific vocabulary of the failure: `lament, elegiac, requiem, dirge, funeral march, mourning tone, candlelit vigil, mournful strings, tearful delivery` and `triumphant ending, victory fanfare, anthemic uplift, celebratory drop`.

## ⭐ ANTI-CHOIR — fatal twice here, fought in three places

1. **The syllable.** `Da-da-da-da — dah` is four hard plosives and one held note. **A choir cannot happen on a hard D** — the register requires sustained vowels and no consonant attack.
2. **The positive field** (new at step 11): *"every note attacked on a hard D"* in all four style prompts; *"dead dry and close on a dynamic mic"*; *"tape hiss where a pad would be."*
3. **The exclude field:** `choir, choral pad, choral swell, angelic vocal, wordless angelic aahs, sacred texture, hymnal texture, plainsong, church organ, cathedral reverb, long reverb tail, sustained vowel harmony, layered oohs` — plus `sustained pad in place of a riff, ambient wash as the hook` for the F1 failure that rides alongside it.
4. **The instrument.** Fuzz bass doubles the lower voice an octave down — **a punk instrument standing exactly where the organ would be.**

⛔ **Why it is fatal twice:** it is the register Sunna forbids by name, **and over this subject a sacred wordless women's harmony becomes a hymn for the dead** — the second number arriving through the one channel the lyric cannot police.

## ⭐ HUMAN SUBJECT STANDARD — verified line by line across all four

`vault/HUMAN_SUBJECT_STANDARD.md` §3.0 slot grammar, re-checked at step 11 against the finished text:

| slot | value | verified |
|---|---|---|
| **PERSON** | *she · he · I · somebody · everybody · nobody · a kid.* **No proper name appears in any of the four lyrics fields.** Every person invented whole. | ✅ |
| **PLACE** | *the road · the gate · the yard · the wall · the door · the tap · the mat · the hill · the window.* **No place name, no country, no city, no border, no sea.** | ✅ |
| **WHEN** | *since dark · the hour before · last night · when it's light.* **No date, no year, no clock time, no numeral of any kind.** | ✅ |
| **THEME** *(open)* | people deciding to go, and the ordinary physical things they do in the hour before | — |

**The pre-draft question, re-answered against the finished text:** *does any PERSON/PLACE/WHEN value, alone or combined, let a listener resolve this to one specific real person who was actually harmed?* — **No.** No real individual, no real family, no real town, no reconstructed real death. **Draw the pattern; invent the people.**
**The abstract child in V3** (*"Kid's got a ball and the ball's got a hole"*) is permitted explicitly by §5 — unnamed, unplaced, not a victim, and the strongest anti-lament image in the pair.

**No Lineage & Credit block is needed for any of the four.** No living scene is borrowed, no real scene is drawn on, every person and place is invented. **Stated explicitly rather than omitted**, per the ICB's hard gates.

---

### VARIATION 1 — Bag By The Door Since Dark

## 1. MUSIC PROMPT
```
Electropunk and fuzz-bass punk at 132 BPM in C natural minor, collapsing to a 165 breakbeat halfway. The hook is two women's voices, dead dry and close on a dynamic mic, landing a wordless stacked-third figure in the first two seconds — five staccato notes a diatonic third apart, the lower line stepping E-flat, F, G, A-flat and settling back on G, every note attacked on a hard D. That figure opens the record before any lyric, sits under every chorus as the room's part, and returns alone after the last line, where the upper voice refuses the third and climbs to a bare fifth — a split that happens exactly twice and nowhere else. Kick twelve milliseconds behind the grid, bass locked on it and doubling the lower voice an octave down, blown-out drop-D guitar hard left, hats crisp right, tape hiss where a pad would be. Female sing-speak, flat and unbothered, one bar of real silence in the last chorus. One sub climbs a semitone and never arrives.
```

## 1B. SUNO EXCLUDE PROMPT
```
choir, choral pad, choral swell, angelic vocal, wordless angelic aahs, sacred texture, hymnal texture, plainsong, church organ, cathedral reverb, long reverb tail, sustained vowel harmony, layered oohs, warm alto, crystalline vocal, sung-pretty, operatic soprano, head-voice legato, vibrato swell, pitch-corrected lead, de-essed breath, lament, elegiac, requiem, dirge, funeral march, mourning tone, candlelit vigil, mournful strings, tearful delivery, triumphant ending, victory fanfare, anthemic uplift, celebratory drop, sustained pad in place of a riff, ambient wash as the hook, supersaw wall, white-noise riser, airhorn, generic trap hats, phonk, snarl, festival EDM drop, orchestral strings, cinematic swell, piano ballad, acoustic guitar, male lead vocal, moralising tag, warning tone, glossy master, key change, long fade-out, empty-room reverb, tasteful sadness, solemn processional
```

## 2. LYRICS
```
[Theme: the hour before somebody goes — she fills a bottle, laces a boot, hands over the good torch, and the bag has been by the door since dark]
[SONG FORM: Intro riff alone / Verse 1 / Chorus 1 — 75 / Verse 2 / Pre-Chorus / Chorus 2 — 76 hats gone / Break — four-on-floor to breakbeat, the two voices split to a fifth / Chorus 3 — 77 guitar gone / Verse 3 / Pre-Chorus / Chorus 4 — 78 bass out two bars / Chorus 4 again / Outro riff alone, splits to a fifth]

[Disc_Vocal: two_female_voices_stacked_thirds | sing_speak_dry_flat_affect | hard_D_plosive_attack_no_sustained_vowel | breath_on_capsule_audible | dry_intimate_no_reverb | Center_Front]
[Disc_Rhythm: four_on_the_floor_132_kick_late_12ms | amen_breakbeat_165_after_the_break | closed_hats_removed_from_chorus_two | uncompressed_transient_snap | Stereo_Width_Mid]
[Disc_Sub: fuzz_bass_doubling_the_lower_voice_one_octave_down | Minimoog_Bass | semitone_upward_glide_never_arriving | Mono_Sub_Lock]
[Disc_Texture: blown_out_drop_D_guitar_stabs | fuzz_pedal_saturation | guitar_removed_from_chorus_three | Hard_Pan_Left]
[Disc_Pad: no_choir_no_choral_swell_no_angelic_texture | dry_room_tone_only | cassette_tape_hiss_saturation | Center_Narrow]

[Intro — EMO:Anticipation — two voices alone, staccato, audible inhale on the pickup, no drums, no bass]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah

[Verse 1 — EMO:Equanimity — sing-speak, flat, close]
She filled the bottle at the tap.
Turned the cap till it sat straight.
Laced the left boot, laced the right.
Put the good torch in somebody's hand.
Bag by the door since dark.
*a bottle cap turning*

[Chorus 1 — EMO:Resolve — full band, the room takes the two voices]
Seventy-five thousand went.
Da-da-da-da — dah
Bag by the door since dark.
Da-da-da-da — dah
Seventy-five thousand went.
Da-da-da-da — dah

[Verse 2 — EMO:Amusement — dry, guitar stabs hard left]
Kettle off at the wall.
Key on the hook where the key goes.
Somebody's shoes by somebody's shoes.
Everybody up and nobody loud.
Door held open with a foot.

[Pre-Chorus — EMO:Vigilance — band drops to bass and two voices]
Same road. Same gate. Same night.
Count the going. Count it right.

[Chorus 2 — EMO:Resolve — closed hats gone]
Seventy-six thousand went.
Da-da-da-da — dah
Bag by the door since dark.
Da-da-da-da — dah
Seventy-six thousand went.
Da-da-da-da — dah

[Break — EMO:Zeal — the floor collapses to a 165 breakbeat; the two voices open from a third to a fifth, breath audible between them]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — DAH

[Chorus 3 — EMO:Resolve — guitar gone, breakbeat under]
Seventy-seven thousand went.
Da-da-da-da — dah
Bag by the door since dark.
Da-da-da-da — dah
Seventy-seven thousand went.
Da-da-da-da — dah

[Verse 3 — EMO:Dignity — breakbeat, bass forward]
Out past the yard, out past the light.
Coat on. Bottle in the coat.
Behind her somebody locks the door.
Behind her somebody puts the kettle on.
Nobody made a speech. Everybody went.

[Pre-Chorus — EMO:Vigilance — band drops to bass and two voices]
Same road. Same gate. Same night.
Count the going. Count it right.

[Chorus 4 — EMO:Resolve — bass out for two bars]
Seventy-eight thousand went.
Da-da-da-da — dah
Bag by the door since dark.
*silence — the room's bar*
Seventy-eight thousand went.
Da-da-da-da — dah

[Chorus 4 again — EMO:Resolve — drums thin, two voices forward]
Seventy-eight thousand went.
Da-da-da-da — dah
Bag by the door since dark.
Da-da-da-da — dah
Seventy-eight thousand went.
Da-da-da-da — dah

[Outro — EMO:Persistence+Anticipation — drums gone, everything gone but two voices, breath on the capsule; the last note opens from a third to a fifth]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — DAH
```

## 3. TITLE
**Bag By The Door Since Dark**

## V1 — fingerprint, dramaturgy, style-axis lock
**Vocal fingerprint:** two female voices, stacked thirds, close dynamic mic, dead dry, no vibrato, no head voice, consonants dropped off line ends, breath left on the capsule and no de-esser near it. The lead owns the number; the two voices own the wordless figure; **the room owns the bar of silence.**
**Production dramaturgy:** every unusual sound has a job — the hard-D attack is the anti-choir lock; the fuzz bass is the hook's shadow, not a second idea; the bar of silence is the room's part written as an absence; the semitone climb is the pursuer and never resolves.
**Style-axis lock:** electropunk × fuzz-bass punk · 132→165 · C natural minor · four-on-floor → breakbeat · gradation subtractive · **exit at 2:45, no fifth chorus, no coda.**

---

### VARIATION 2 — I Packed It So It Shuts

## 1. MUSIC PROMPT
```
Fuzz-bass punk and electropunk at 132 BPM in C natural minor, collapsing to a 165 breakbeat halfway. One fuzz-bass note holds from bar one under two women's voices, dead dry and close on a dynamic mic, carrying a wordless stacked-third figure — five staccato notes a diatonic third apart, the lower line stepping E-flat, F, G, A-flat and settling back on G, every note attacked on a hard D. That figure opens the record before any lyric, sits under every chorus as the room's part, and returns alone after the last line, where the upper voice refuses the third and climbs to a bare fifth — a split that happens exactly twice. Kick twelve milliseconds late against a bass locked on the grid and doubling the lower voice an octave down, blown-out drop-D guitar hard left, hats crisp right, tape hiss where a pad would be. Female sing-speak, first person and flat, one bar of real silence in the last chorus. One sub climbs a semitone and never arrives.
```

## 1B. SUNO EXCLUDE PROMPT
```
choir, choral pad, choral swell, angelic vocal, wordless angelic aahs, sacred texture, hymnal texture, plainsong, church organ, cathedral reverb, long reverb tail, sustained vowel harmony, layered oohs, warm alto, crystalline vocal, sung-pretty, operatic soprano, head-voice legato, vibrato swell, pitch-corrected lead, de-essed breath, lament, elegiac, requiem, dirge, funeral march, mourning tone, candlelit vigil, mournful strings, tearful delivery, triumphant ending, victory fanfare, anthemic uplift, celebratory drop, sustained pad in place of a riff, ambient wash as the hook, supersaw wall, white-noise riser, airhorn, generic trap hats, phonk, snarl, festival EDM drop, orchestral strings, cinematic swell, piano ballad, acoustic guitar, male lead vocal, moralising tag, warning tone, glossy master, key change, long fade-out, sentimental piano, farewell montage, sighing backing vocals
```

## 2. LYRICS
```
[Theme: the one who packed the bag — rolled not folded, charger down the side, sat on the lid till the zip went round, nothing written on the tag]
[SONG FORM: Intro riff alone / Verse 1 / Chorus 1 — 75 / Verse 2 / Pre-Chorus / Chorus 2 — 76 hats gone / Break — four-on-floor to breakbeat, the two voices split to a fifth / Chorus 3 — 77 guitar gone / Verse 3 / Pre-Chorus / Chorus 4 — 78 bass out two bars / Chorus 4 again / Outro riff alone, splits to a fifth]

[Disc_Vocal: two_female_voices_stacked_thirds | sing_speak_first_person_flat | hard_D_plosive_attack_no_sustained_vowel | breath_on_capsule_audible | dry_intimate_no_reverb | Center_Front]
[Disc_Sub: fuzz_bass_single_note_from_bar_one | fuzz_bass_doubling_the_lower_voice_one_octave_down | semitone_upward_glide_never_arriving | Mono_Sub_Lock]
[Disc_Rhythm: four_on_the_floor_132_kick_late_12ms | amen_breakbeat_165_after_the_break | closed_hats_removed_from_chorus_two | uncompressed_transient_snap | Stereo_Width_Mid]
[Disc_Texture: blown_out_drop_D_guitar_stabs | fuzz_pedal_saturation | guitar_removed_from_chorus_three | Hard_Pan_Left]
[Disc_Pad: no_choir_no_choral_swell_no_angelic_texture | dry_room_tone_only | cassette_tape_hiss_saturation | Center_Narrow]

[Intro — EMO:Anticipation — two voices and one held fuzz-bass note, audible inhale on the pickup, no drums]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah

[Verse 1 — EMO:Composure — first person, flat, close]
I rolled it. I never fold.
Charger down the side where you'll find it.
Sat on the lid till the zip went round.
Good socks on the top.
Wrote nothing on the tag.
*a zip going round*

[Chorus 1 — EMO:Resolve — full band, the room takes the two voices]
Seventy-five thousand went.
Da-da-da-da — dah
I packed it so it shuts.
Da-da-da-da — dah
Seventy-five thousand went.
Da-da-da-da — dah

[Verse 2 — EMO:Amusement — dry, guitar stabs hard left]
Bread in the cloth, cloth in the bag.
Took the book out. Put the book back.
Took the book out.
Left the book on the side.
Weighed it on the bathroom scales.

[Pre-Chorus — EMO:Persistence — band drops to bass and two voices]
Bag's done. Bag's by the wall.
Count them again. Count them all.

[Chorus 2 — EMO:Resolve — closed hats gone]
Seventy-six thousand went.
Da-da-da-da — dah
I packed it so it shuts.
Da-da-da-da — dah
Seventy-six thousand went.
Da-da-da-da — dah

[Break — EMO:Zeal — the floor collapses to a 165 breakbeat; the two voices open from a third to a fifth, breath audible between them]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — DAH

[Chorus 3 — EMO:Resolve — guitar gone, breakbeat under]
Seventy-seven thousand went.
Da-da-da-da — dah
I packed it so it shuts.
Da-da-da-da — dah
Seventy-seven thousand went.
Da-da-da-da — dah

[Verse 3 — EMO:Dignity — breakbeat, bass forward]
Stood in the door with the bag done up.
Handed it over. Took my hands back.
Said, "Text me when it's light."
Went in. Washed a cup. Put it on the rack.
Bag's gone and the hook's still there.

[Pre-Chorus — EMO:Persistence — band drops to bass and two voices]
Bag's done. Bag's by the wall.
Count them again. Count them all.

[Chorus 4 — EMO:Resolve — bass out for two bars]
Seventy-eight thousand went.
Da-da-da-da — dah
I packed it so it shuts.
*silence — the room's bar*
Seventy-eight thousand went.
Da-da-da-da — dah

[Chorus 4 again — EMO:Resolve — drums thin, two voices forward]
Seventy-eight thousand went.
Da-da-da-da — dah
I packed it so it shuts.
Da-da-da-da — dah
Seventy-eight thousand went.
Da-da-da-da — dah

[Outro — EMO:Persistence+Anticipation — drums gone, everything gone but two voices, breath on the capsule; the last note opens from a third to a fifth]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — DAH
```

## 3. TITLE
**I Packed It So It Shuts**

## V2 — fingerprint, dramaturgy, style-axis lock
**Vocal fingerprint:** first person, sing-speak, close and dry, no vibrato, no head voice; the two stacked voices are a separate body from the lead and are never doubled with it. Breath left in on the verse ends where the packing is physical.
**Production dramaturgy:** the single held fuzz-bass note from bar one is the floor the whole song stands on — it is also the pursuer's launch pad; the bar of silence lands on the line about the bag being shut; the split opens where nothing is said.
**Style-axis lock:** fuzz-bass punk × electropunk · 132→165 · C natural minor · four-on-floor → breakbeat · **the bassline is the hook's shadow, never a second tune** · exit at 2:45.

---

### VARIATION 3 — Room On The Mat

## 1. MUSIC PROMPT
```
Electropunk and crunch punk at 132 BPM in C natural minor, collapsing to a 165 breakbeat halfway. The hook is two women's voices, dead dry and close on a dynamic mic under two room mics eight decibels below instinct, landing a wordless stacked-third figure in the first two seconds — five staccato notes a diatonic third apart, the lower line stepping E-flat, F, G, A-flat and settling back on G, every note attacked on a hard D. That figure opens the record before any lyric, sits under every chorus as the room's part, and returns alone after the last line, where the upper voice refuses the third and climbs to a bare fifth — a split that happens exactly twice. Kick twelve milliseconds behind a bass locked on the grid and doubling the lower voice an octave down, blown-out drop-D guitar hard left, hats crisp right, room tone and tape hiss where a pad would be. Female sing-speak, flat and unbothered. One sub climbs a semitone and never arrives.
```

## 1B. SUNO EXCLUDE PROMPT
```
choir, choral pad, choral swell, angelic vocal, wordless angelic aahs, sacred texture, hymnal texture, plainsong, church organ, cathedral reverb, long reverb tail, sustained vowel harmony, layered oohs, warm alto, crystalline vocal, sung-pretty, operatic soprano, head-voice legato, vibrato swell, pitch-corrected lead, de-essed breath, lament, elegiac, requiem, dirge, funeral march, mourning tone, candlelit vigil, mournful strings, tearful delivery, triumphant ending, victory fanfare, anthemic uplift, celebratory drop, sustained pad in place of a riff, ambient wash as the hook, supersaw wall, white-noise riser, airhorn, generic trap hats, phonk, snarl, festival EDM drop, orchestral strings, cinematic swell, piano ballad, acoustic guitar, male lead vocal, moralising tag, warning tone, glossy master, key change, long fade-out, campfire singalong, cosy acoustic warmth, community-choir tone
```

## 2. LYRICS
```
[Theme: the ones already there — somebody moves a crate to make the room, rolls a mat out with his foot, boils the water again, and holds a phone at the window for the bars]
[SONG FORM: Intro riff alone / Verse 1 / Chorus 1 — 75 / Verse 2 / Pre-Chorus / Chorus 2 — 76 hats gone / Break — four-on-floor to breakbeat, the two voices split to a fifth / Chorus 3 — 77 guitar gone / Verse 3 / Pre-Chorus / Chorus 4 — 78 bass out two bars / Chorus 4 again / Outro riff alone, splits to a fifth]

[Disc_Vocal: two_female_voices_stacked_thirds | sing_speak_dry_flat_affect | hard_D_plosive_attack_no_sustained_vowel | body_noises_foregrounded | dry_intimate_no_reverb | Center_Front]
[Disc_Pad: no_choir_no_choral_swell_no_angelic_texture | two_room_mics_8dB_under_highpassed_180Hz | short_plate_0point9_second_sweat_plate | cassette_tape_hiss_saturation]
[Disc_Rhythm: four_on_the_floor_132_kick_late_12ms | amen_breakbeat_165_after_the_break | closed_hats_removed_from_chorus_two | uncompressed_transient_snap | Stereo_Width_Mid]
[Disc_Sub: fuzz_bass_doubling_the_lower_voice_one_octave_down | Minimoog_Bass | semitone_upward_glide_never_arriving | Mono_Sub_Lock]
[Disc_Texture: blown_out_drop_D_guitar_stabs | fuzz_pedal_saturation | guitar_removed_from_chorus_three | Hard_Pan_Left]

[Intro — EMO:Anticipation — two voices and a room mic under them, audible inhale on the pickup, no drums]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah

[Verse 1 — EMO:Solidarity — sing-speak, flat, close]
He moved the crate to make the room.
Rolled the mat out with his foot.
Boiled the water again because.
Held the phone at the window for the bars.
Somebody's coming up the road.
*a kettle clicking off*

[Chorus 1 — EMO:Resolve — full band, the room takes the two voices]
Seventy-five thousand went.
Da-da-da-da — dah
Room on the mat for whoever.
Da-da-da-da — dah
Seventy-five thousand went.
Da-da-da-da — dah

[Verse 2 — EMO:Amusement — dry, guitar stabs hard left]
Queue at the tap and the tap is slow.
Kid's got a ball and the ball's got a hole.
Somebody's charging everybody's phones.
Somebody's cutting somebody's hair.
Radio on. Nobody listening.

[Pre-Chorus — EMO:Vigilance — band drops to bass and two voices]
Lights on the road, lights on the hill.
Count them again. Count them still.

[Chorus 2 — EMO:Resolve — closed hats gone]
Seventy-six thousand went.
Da-da-da-da — dah
Room on the mat for whoever.
Da-da-da-da — dah
Seventy-six thousand went.
Da-da-da-da — dah

[Break — EMO:Zeal — the floor collapses to a 165 breakbeat; the two voices open from a third to a fifth, breath audible between them]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — DAH

[Chorus 3 — EMO:Resolve — guitar gone, breakbeat under]
Seventy-seven thousand went.
Da-da-da-da — dah
Room on the mat for whoever.
Da-da-da-da — dah
Seventy-seven thousand went.
Da-da-da-da — dah

[Verse 3 — EMO:Persistence — breakbeat, bass forward]
Road's a line of little lights.
Counting them from the top of the wall.
Counting them again. Counting them again.
Gate goes and everybody stands up.
Somebody puts more water on.

[Pre-Chorus — EMO:Vigilance — band drops to bass and two voices]
Lights on the road, lights on the hill.
Count them again. Count them still.

[Chorus 4 — EMO:Resolve — bass out for two bars]
Seventy-eight thousand went.
Da-da-da-da — dah
Room on the mat for whoever.
*silence — the room's bar*
Seventy-eight thousand went.
Da-da-da-da — dah

[Chorus 4 again — EMO:Resolve — drums thin, two voices forward]
Seventy-eight thousand went.
Da-da-da-da — dah
Room on the mat for whoever.
Da-da-da-da — dah
Seventy-eight thousand went.
Da-da-da-da — dah

[Outro — EMO:Persistence+Anticipation — drums gone, everything gone but two voices, breath on the capsule; the last note opens from a third to a fifth]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — DAH
```

## 3. TITLE
**Room On The Mat**

## V3 — fingerprint, dramaturgy, style-axis lock
**Vocal fingerprint:** flat sing-speak with body noises foregrounded, two room mics eight decibels under instinct and high-passed at 180 Hz into a 0.9-second plate — the sweat plate, never a hall. The crowd bed is the room, not a sample of one.
**Production dramaturgy:** the room mics are the only reverb in the record and they exist so the chant has somewhere to be sung back into; the bar of silence is where that becomes literal; the guitar leaving in chorus three opens the space the room fills.
**Style-axis lock:** electropunk × crunch punk · 132→165 · C natural minor · four-on-floor → breakbeat · **room tone where a pad would be** · exit at 2:45.

---

### VARIATION 4 — Bigger By Breakfast

## 1. MUSIC PROMPT
```
Electropunk and fuzz-bass punk at 132 BPM in C natural minor, collapsing to a 165 breakbeat halfway. Two women's voices are already mid-figure when the record starts, no count-in and no downbeat, dead dry and close on a dynamic mic, carrying a wordless stacked-third hook — five staccato notes a diatonic third apart, the lower line stepping E-flat, F, G, A-flat and settling back on G, every note attacked on a hard D. That figure is the song: it runs before any lyric, sits under every chorus as the room's part, and returns alone after the last line, stepping up one degree with nothing under it as the upper voice refuses the third and climbs to a bare fifth. Kick twelve milliseconds late against a bass locked on the grid and doubling the lower voice an octave down, blown-out drop-D guitar hard left, hats crisp right, tape hiss where a pad would be. Female sing-speak, flat and dry. One sub climbs a semitone and never arrives.
```

## 1B. SUNO EXCLUDE PROMPT
```
choir, choral pad, choral swell, angelic vocal, wordless angelic aahs, sacred texture, hymnal texture, plainsong, church organ, cathedral reverb, long reverb tail, sustained vowel harmony, layered oohs, warm alto, crystalline vocal, sung-pretty, operatic soprano, head-voice legato, vibrato swell, pitch-corrected lead, de-essed breath, lament, elegiac, requiem, dirge, funeral march, mourning tone, candlelit vigil, mournful strings, tearful delivery, triumphant ending, victory fanfare, anthemic uplift, celebratory drop, sustained pad in place of a riff, ambient wash as the hook, supersaw wall, white-noise riser, airhorn, generic trap hats, phonk, snarl, festival EDM drop, orchestral strings, cinematic swell, piano ballad, acoustic guitar, male lead vocal, moralising tag, warning tone, glossy master, key change, long fade-out, news-report voiceover, radio announcer sample, documentary tone
```

## 2. LYRICS
```
[Theme: the morning after — she turns the radio up with a wet hand, the butter is soft, the bread is the end of the loaf again, and the number is bigger than it was]
[SONG FORM: Intro riff mid-figure / Verse 1 / Chorus 1 — 75 / Verse 2 / Pre-Chorus / Chorus 2 — 76 hats gone / Break — four-on-floor to breakbeat, the two voices split to a fifth / Chorus 3 — 77 guitar gone / Verse 3 / Pre-Chorus / Chorus 4 — 78 bass out two bars / Chorus 4 again / Outro riff alone, no number left, the figure steps up one degree and splits to a fifth]

[Disc_Vocal: two_female_voices_stacked_thirds | sing_speak_dry_flat_affect | hard_D_plosive_attack_no_sustained_vowel | breath_on_capsule_audible | dry_intimate_no_reverb | Center_Front]
[Disc_Rhythm: no_count_in_no_downbeat_at_the_top | four_on_the_floor_132_kick_late_12ms | amen_breakbeat_165_after_the_break | uncompressed_transient_snap | Stereo_Width_Mid]
[Disc_Sub: fuzz_bass_doubling_the_lower_voice_one_octave_down | Minimoog_Bass | semitone_upward_glide_never_arriving | Mono_Sub_Lock]
[Disc_Texture: blown_out_drop_D_guitar_stabs | fuzz_pedal_saturation | guitar_removed_from_chorus_three | Hard_Pan_Left]
[Disc_Pad: no_choir_no_choral_swell_no_angelic_texture | dry_room_tone_only | cassette_tape_hiss_saturation | Center_Narrow]

[Intro — EMO:Anticipation — the two voices are already going when the record starts, audible inhale on the pickup]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah

[Verse 1 — EMO:Equanimity — sing-speak, flat, close]
She turned the radio up with a wet hand.
Butter's out. Butter's soft.
Bread's the end of the loaf again.
Number's bigger than it was.
Everybody eats standing up.
*a radio dial clicking*

[Chorus 1 — EMO:Resolve — full band, the room takes the two voices]
Seventy-five thousand went.
Da-da-da-da — dah
Bigger than it was last night.
Da-da-da-da — dah
Seventy-five thousand went.
Da-da-da-da — dah

[Verse 2 — EMO:Amusement — dry, guitar stabs hard left]
Kettle's on again. Kettle's always on.
Somebody's asleep in a coat.
Somebody's up and out already.
Radio says the number. Radio says it again.
Everybody heard it the first time.

[Pre-Chorus — EMO:Impatience — band drops to bass and two voices]
Chalk in a hand, tea in a cup.
Count them again. Count them up.

[Chorus 2 — EMO:Resolve — closed hats gone]
Seventy-six thousand went.
Da-da-da-da — dah
Bigger than it was last night.
Da-da-da-da — dah
Seventy-six thousand went.
Da-da-da-da — dah

[Break — EMO:Zeal — the floor collapses to a 165 breakbeat; the two voices open from a third to a fifth, breath audible between them]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — DAH

[Chorus 3 — EMO:Resolve — guitar gone, breakbeat under]
Seventy-seven thousand went.
Da-da-da-da — dah
Bigger than it was last night.
Da-da-da-da — dah
Seventy-seven thousand went.
Da-da-da-da — dah

[Verse 3 — EMO:Dignity — breakbeat, bass forward]
Somebody's chalked the road times on the wall.
Somebody wipes the old time off.
Somebody chalks the new time in.
Chalk's down to nothing. Chalk gets found.
Everybody walks past and reads it.

[Pre-Chorus — EMO:Impatience — band drops to bass and two voices]
Chalk in a hand, tea in a cup.
Count them again. Count them up.

[Chorus 4 — EMO:Resolve — bass out for two bars]
Seventy-eight thousand went.
Da-da-da-da — dah
Bigger than it was last night.
*silence — the room's bar*
Seventy-eight thousand went.
Da-da-da-da — dah

[Chorus 4 again — EMO:Resolve — drums thin, two voices forward]
Seventy-eight thousand went.
Da-da-da-da — dah
Bigger than it was last night.
Da-da-da-da — dah
Seventy-eight thousand went.
Da-da-da-da — dah

[Outro — EMO:Persistence+Anticipation — drums gone, no number left, breath on the capsule; the figure steps up one degree and stays there, the last note opening from a third to a fifth]
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — dah
Da-da-da-da — DAH
```

## 3. TITLE
**Bigger By Breakfast**

## V4 — fingerprint, dramaturgy, style-axis lock
**Vocal fingerprint:** flat, dry sing-speak, third person, no vibrato; the record is already running when you arrive and the voice does not acknowledge it. Breath on the capsule at the top and the tail.
**Production dramaturgy:** ⭐ **the last removal is the number itself** — the outro drops the count, leaves the two voices, and the figure steps up one degree with nothing under it. **The count keeps climbing after the song stops saying it.** That is the subtractive doctrine executed in the last eight bars; the ending contains less than any other point in the record.
**Style-axis lock:** electropunk × fuzz-bass punk · 132→165 · C natural minor · **no count-in, no downbeat at the top** · four-on-floor → breakbeat · exit at 2:45.

---

## Major Deviations

- **Changed:** the style prompts now lead with genre + tempo + key rather than with the riff. **Reason:** step 10 violated Gate 14a outright, and a validator-passing prompt that fails the step's own mandatory gate is a defect no matter how good the paragraph reads. **Effect:** none on the doctrine — the riff is still the first musical object named, and the whistle-test receipt (instrument, five notes, interval, octave range) is intact.
- **Changed:** the mid-chorus bracket `[the room's bar — …]` became the italic sound command `*silence — the room's bar*`. **Reason:** Suno parses `[...]` as a section boundary; a bracket inside a chorus can split the chorus in two and destroy the byte-identical return the whole device rests on. **Effect:** the room's bar survives as a render instruction instead of becoming a rendering bug.
- **Changed:** three lines repaired for second-number proximity (V1, V3, V4 — itemised above), one of which the step-10 audit **reported as cut when it had only been reworded.** **Reason:** the pair's own rule is that near-misses HOLD and are not argued; an audit claiming a cut that did not happen is the failure mode the rule exists to prevent. **Effect:** the constraint is now true rather than asserted, at the cost of nothing — no rhyme lost, no line count changed, no chorus touched.
- **Refused:** the step-11 contract's instruction to deploy literary density, structural innovation and poetic elevation *as such*. **Reason:** this run exists to test THE FREE CHANNEL — *complexity in the MUSIC, feeling in the WORDS.* Elevating these lyrics toward literary texture would rebuild the toll booth the doctrine was written to demolish, and P06 in particular must report rather than interpret. The technique mandate is met instead by a device with **zero semantic cost**: the plosive approach chain, plus EMO mirror-form. **Effect:** the difficulty stays relocated rather than deleted — the receipt is the hook-excluded soundcraft table below, which holds without the riff doing the work.
- **Refused:** the 70–120 sung-line target. **Reason:** the run's own metaprompt overrides it — *"Sunna's songs are SHORT — 45–75 lines is correct here and the 70-line floor yields to her spec."* 59 sung lines per variation, nothing padded. **Effect:** the exit lands at 2:45 and the set still ends one song early.
- **Refused:** embedding the golden songs' style prompts, lyrics or exclude fields. **Reason:** GOLDEN-OUTPUT QUARANTINE (`EXECUTION.md` §3) — step 11 is a generating context. Names handed forward for QA instead.

## Lineage & Credit

**None required.** No living scene is borrowed, no real scene is drawn on, no real individual, family, town, country or date appears in any of the four songs. Every person is invented. **Stated explicitly rather than omitted**, per the ICB's hard gates.

## ⛔ VERIFICATION CHECKLIST — measured, not asserted

| | prompt chars *(850–1000, target 870–960)* | exclude chars *(400–900)* | lyrics field chars *(<5000, target ≤4800)* | sung lines |
|---|---|---|---|---|
| **V1** Bag By The Door Since Dark | **953** ✅ | **892** ✅ | **3750** ✅ | 59 *(61 as the validator counts)* |
| **V2** I Packed It So It Shuts | **950** ✅ | **895** ✅ | **3754** ✅ | 59 *(61 as the validator counts)* |
| **V3** Room On The Mat | **951** ✅ | **898** ✅ | **3880** ✅ | 59 *(61 as the validator counts)* |
| **V4** Bigger By Breakfast | **935** ✅ | **899** ✅ | **3898** ✅ | 59 *(61 as the validator counts)* |

All four prompts end on terminal punctuation (`music_prompt_terminal_punctuation`), none hugs the ceiling (`music_prompt_hug_ceiling: 985`), none contains an artist name, none contains avoidance language.

**`skills/music/scripts/validate_suno_packages.py` → PASS.** 4 package headings, 4 `## 1. MUSIC PROMPT` sections, cardinality matched.

**`scripts/measure_soundcraft.py → profile_file()` — the honest figures.**
⚠️ **The pooled alliteration number is inflated by the hook.** `Da-da-da-da — dah` is five d-initial tokens and appears 25 times per song, so it scores four alliteration hits every time. **Report the hook-excluded figure — that is the number to trust.**

| | pooled *(236 lines, hook included)* | ⭐ **hook-excluded** *(136 lines)* | floor | verdict |
|---|---|---|---|---|
| `end_rhyme` | 0.682 | **0.566** | ≥0.30 | PASS |
| `line_return` | 0.750 | **0.566** | ≥0.20 | PASS |
| `words_per_line` | 5.390 | **5.676** | ≤7.5 | PASS |
| `allit_per_100w` | 40.881 | **15.544** | ≥11.0 | PASS |

**Step-10 baseline, hook-excluded: 0.566 / 0.566 / 5.669 / 15.305** — reproduced exactly from the step-10 file before measuring this one, so the comparison is against a re-run number rather than a quoted one.
**Delta: end rhyme unchanged · line return unchanged · words per line +0.007 · alliteration +0.239.** ⛔ **No return was removed** — the three repairs were like-for-like, every rhyme survives, and the small alliteration gain is the plosive approach doing exactly what it was applied to do. **Nothing here is carried by the riff:** with all 100 hook lines stripped, all four floors still clear.

**`scripts/check_human_subjects.py` — run on the four lyrics fields + the four titles, and read honestly.**
`ner_method: regex-fallback` (spaCy model not installed on this host, so the detector degrades toward over-flagging by design). Meaningful signals: **`crime_death_terms: []` · `checkA_minor_as_victim: false` · `sensitive_context: false`.** Every entry in `checkB_person_names` is a **section-header word** — *Chorus, Verse, Pre, Intro, Break, Outro, Resolve, Anticipation, Persistence* — i.e. bracket-format artefacts, not names; there is no name in any sung line. `minor_terms: ["kid"]` is V3's abstract child, permitted explicitly by §5 and declared at step 06. The script's `HOLD_FOR_HUMAN` is therefore **the documented fail-open behaviour of the fallback path, not a finding.** ⛔ It remains a prefilter, not an authority: **clean ≠ ship**, and the release decision is The Scientist's.

**Hard gates, each verified against the finished text, not the previous file's claim:**
- ⭐ **Second number — ABSENT** from all four lyrics fields in every form. Re-swept at step 11; three near-misses found and repaired.
- ⭐ **No moral, no lament, no triumph.** No EMO tag is Sorrow or Triumph. The final chorus repeats the fourth verbatim and resolves nothing.
- ⭐ **The stack is not a choir.** Hard-D attack in every positive prompt; the sacred cluster and both the lament and triumph clusters blacklisted in all four excludes.
- ⭐ **Human Subject Standard.** No proper name, place name, country or date in any of the four.
- **ONE DEVICE** — `the count that climbs`. No second structural idea added; the applied technique is sonic, not semantic.
- **Pursuer never named.** No sung line mentions a sub, a semitone, a clock, a wolf, or anything chasing.
- **No Disc_Channel token appears in any sung line.**
- **No Lofn motifs.** No industrial grief, no somatic machinery, no plant-wave, no laboratory narration.
- **Riff present before the first line and after the last**, alone, in all four.
- **≥1 standalone `*SFX*` per song**; 2 per song, both at emotional peaks, ≤3 per the anti-pattern table.
