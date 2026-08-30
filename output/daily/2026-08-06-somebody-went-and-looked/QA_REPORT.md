# QA REPORT — `2026-08-06-somebody-went-and-looked`
**SUNNA · music daily · 6 pairs × 4 = 24 songs · adversarial audit, judge-side**
ICB verified at load: `CREATIVE_CONTEXT.md` · **69,095 B** · sha256 `85ed1348b7e22fdfb8e4b06dadd21daa1f4c85ede5ea7e150105f41e49f44a15` · file contains no CRLF, so raw and LF-normalised hashes are identical. All six pair files cite this hash and none of them edited it.
Read: `core_seed` · `03_panel_debate` · `04_metaprompt` · `05_pair_assignments` · `05b_P06_REPLACEMENT` · six `step11_final_package_enhanced` · spot-checks into `pair_05_step10`. **`_withdrawn_p06/` was not opened.**
Judge-side references consulted for the blind comparison only: `skills/music/references/golden_songs_index.md`, `suno_format_example_triple_arch.md`, `..._blue_screen.md`, `..._five_wrong_colors.md`.

---

## 1. VERDICT — **REPAIR**

Five of the six pairs are ship-shaped and two of them are the best work in the Pantheon's short life. The doctrine is not a rationale here: I read the intervals myself and five of six riffs are real, playable, contoured objects that exist before the first line and after the last; the difficulty genuinely moved — into a bass that carries the melody, a figure that never plays its tonic for five minutes, and a guitar-and-bass trade that *is* the argument. Zero Lofn-motif hits across 24 songs. The pursuer is never named in any of the 24. P06's second number is absent from all four lyrics fields in every form I could construct a sweep for. Every countable reproduces exactly. **But P01 got shorter rather than simpler** — a two-note riff declared the record's only melodic line, a chorus with no noun in it built from eleven interchangeable gerunds, and not one funny line in 244 lines — and it fails four of the five doctrine questions on its own. **And P05's step-11 receipt is false in three places, one of them inverted**, in a run whose binding condition is *keep the receipt*. Two smaller device breaches complete the brief. This is a REPAIR with a short, cheap list, not a wounded run — but borderline defaults to HOLD and a QA that never says no is decorative.

**Pipeline Integrity Verdict:** PASS · **Suno Package Verdict:** PASS (P02's line-count flag is a declared, ruled-valid deviation) · **Overall:** **REPAIR**

---

## 2. THE FIVE DOCTRINE QUESTIONS

### ⭐⭐ Q1 — DID IT ACTUALLY GET SIMPLER, OR DID IT JUST GET SHORTER?

**Ruling: five pairs relocated the difficulty. P01 deleted it. There is one song-family here that is merely thin, and it is P01.**

The Hyper-Skeptic's binding condition was *relocate, do not delete, keep the receipt.* Here is the receipt, pair by pair, stated as **what the music now carries that the lyric used to**:

| pair | the relocated difficulty — my finding, not their claim | verdict |
|---|---|---|
| **P01** | Two notes. A falling minor third on the and-of-two, declared in every prompt as **"the only melodic line in the track."** The rest is arrangement: dead air placement, a subtractive ladder, a 3-3-2 bounce bridge. **No harmonic content, no melodic contour.** | ⛔ **DELETED** |
| **P02** | The tune physically left the voice. Seven notes on a fuzz bass — F♯ F♯ A D C♯ A F♯ — with a **rising minor sixth off the root** and a semitone fall, in F♯3–D4, stated three times naked (open, under the half-time read, last eight bars). The vocal is sing-speak; the bass is the song. | ✅ **RELOCATED** |
| **P03** | A four-note riff whose memorable object is **a wrong note held too long** — the tritone B♭ hung on the and-of-three — doubled an octave up by a second guitar drifting thirty cents flat at the moment the record is biggest. Plus the whole band vanishing inside one bar to voice and clap. | ✅ **RELOCATED** |
| **P04** | **One riff as the only harmonic event for five minutes**, bass in unison so no implied progression can sneak in, tonic string never struck, eight timed removals and no additions, and the phonetic decay written as sung text so the renderer must perform it. This is unambiguously *more work, not less.* | ✅ **RELOCATED, hardest** |
| **P05** | The guitar asks (G–B♭–C–B♭, stops on B♭, never lands) and the bass answers (D–C–B♭–G, lands on the root). **The argument is in the intervals before anyone sings.** Device and riff are one gesture. | ✅ **RELOCATED, cleanest** |
| **P06** | A wordless five-note two-throat figure that *is* the hook, doubled an octave down by fuzz bass, splitting a third→fifth exactly twice and nowhere else, at 132 collapsing to a 165 breakbeat. The melody has no words at all. | ✅ **RELOCATED** |

**Why P01 is the failure and not merely the weakest.** Three independent tells, and they are the same tell:

1. **The chorus has no noun.** Metaprompt Law 2: *a plain line survives on its NOUN, not its syntax.* P01's call across four variations is: *checking · coming · looking · writing · knocking · waiting · calling · typing · watching · asking · paying* — **eleven interchangeable gerunds** around one borrowed line. Swap any for any and nothing changes. That is not a hook; it is a slot. Every other pair's chorus is nailed to a hard object: *a bolt · your shoes · the table · pan, pinch, plate · a bag by the door · a mat · Ocean Going.* P01's is nailed to *nobody*.
2. **The pair's own verification section lists thirteen hook lines.** A song has one. Thirteen is the arithmetic of a fill-in-the-blank.
3. **The only line that survives the flat test is the run's title**, handed to P01 for free by the seed. The pair did not write it.

P01's verses are good — *"Four spare minutes on a flat Tuesday / She spent them on somebody else"*, and *"Which means she was still at work / Which means she stopped and did this / He does the arithmetic twice"* is the best short account of the locked feeling in the run, because the listener does the arithmetic alongside him and no metaphor is required. **The verses prove the doctrine. The chorus and the riff prove the objection.** A plain line and a lazy line are not the same object, and P01's chorus is the lazy one.

**Everywhere else the answer is yes, and it is measurable.** The archive's lyric lines are clausal and stacked; this run measures **3.37–6.83 mean words per line**, and the longest sung line in all 24 songs is twelve words. That is not compression of the same content — the content changed class, from interpretation to report. *"Blue filed the paperwork for my pulse"* became *"He sets the part down on a towel."*

### ⭐⭐ Q2 — THE WHISTLE TEST, APPLIED BY ME

*Could someone whistle it walking away from the venue, having heard it twice, drunk?* I read the intervals in the prompts, not the claims. **Section 3 carries the full verdicts.** Headline: **five pass, one fails in spirit.** No pad, no arpeggio and no filter sweep is masquerading as a riff anywhere in this run — the named central failure mode did **not** occur, and that is a real result, because it is what the Rock Titan predicted would happen.

**The Concept skeptic's charge, resolved.** *"'Put the complexity in the music' is what people say right before they put it in a plugin."* They did not put it in a plugin. They put it somewhere I consider more dangerous, and I have raised it as its own finding — see the Somatic Gate, seat 2, and repair **R7**: five of six pairs deposited their signature difficulty into **real silence, dead air, an instrument drifting flat, hard panning, and a sub creeping a semitone** — precisely the classes THE GRAIN LAW (`COMPETITION_LEARNINGS` L22, render-measured) says the generator smooths. The defences live mainly in the exclude field, and the GRAIN LAW is explicit that an objection answered outside the lyric or the form is not answered. **The charge lands, in a form nobody anticipated.**

### ⭐ Q3 — THE FLAT TEST

Each pair's hook line, said out loud, flatly, no music, to a person's face:

| pair | hook line | ruling |
|---|---|---|
| **P01** | *"Nobody's checking. / Somebody saw."* | ⚠️ **SHORT, NOT SIMPLE.** No noun. It survives only on *"Somebody went and looked"* — the run's title, not the pair's writing. Eleven interchangeable substitutes are the proof. |
| **P02** | *"Are you being robbed right now."* | ✅ **SIMPLE, and the run's most efficient object** — the device *is* the chorus. Said flat to somebody's face it is genuinely unsettling, which is the whole point. |
| **P03** | *"Two thousand words about a bolt."* | ✅ **SIMPLE.** A specific quantity against a trivial noun. The joke risk is real and the pair foreclosed it *inside the chorus's own third line* — *"Two thousand words and not one joke"* — byte-identically, before a listener could settle into a smirk. I reviewed their adjudication and I agree with the overrule. |
| **P04** | *"Ocean Going."* | ✅ **SIMPLE — conditionally, and the condition is met in two lines.** Cold it is two words. After *"It doesn't fit through the door / She named it anyway"* it is the best joke and the best ache in the run at once. Depth in the situation, exactly as specified. |
| **P05** | *"IT'S EASY / (IT WASN'T)."* | ✅ **SIMPLE, and the strongest in the run.** Four words, one contradiction, and every listener alive has stood on one side of it. |
| **P06** | *"Seventy-eight thousand went."* | ✅ **SIMPLE.** A number and a verb, and the verb carries volition. *"Bag by the door since dark"* is the second line of the same chorus and is also simple: an object, a place, a duration, and the whole night before. |

**5 simple · 1 short.**

### ⭐ Q4 — IS THIS SUNNA, OR IS IT LOFN IN A JACKET?

**It is Sunna — except P01, which is Lofn with the metaphors taken out and nothing put back.**

The hunt, run mechanically over all 24 lyrics fields against a 60-term Lofn-motif list (industrial grief, somatic machinery, laboratory narration, abstract conceits — *geometry, hypotenuse, prism, verdict, frequency, circuitry, cathedral, apparatus, sternum, rib, administrative, residue, buffer, glitch, protocol, specimen, entropy, filament, synapse, transcend, sacred, void…*):

- **Zero hits in P01, P02, P03, P05, P06.**
- **Two hits in P04**, both `ribs` — *"Ribs that she bent over steam"* and *"He asked how the ribs were bent."* These are **steam-bent boat ribs**. Not somatic machinery. Cleared.
- **No bare AWE / INDIGNATION / SYNTHESIS** in any of ~160 distinct EMO headers across the run. Every tag is a specific taxonomy entry.

That is an unusually clean result and I checked it twice because it is unusual.

**But the register test is not the motif test, and P01 fails it.** Sunna's spec makes *funny* a requirement — dry, eye-rolling, never earnest, never reverent. Counting actual jokes: P04 has several (*"Talks to the boat like a dog / Says it. It answers the same."* · *"Kettle on twice for the same"* · the boat named Ocean Going that will not fit through the door). P02 has the machine's log and *"Off, and off, and off, and off."* P03 V4 is dry throughout. P05 is funny in the wince register. P06 is not funny and is correctly not funny — the subject forbids it and that is a deliberate, defensible exception.

**P01 has none. Not one dry line, not one eye-roll, in 244 lines across four variations.** What it has instead is tenderness delivered flat — a woman on a night shift, a man who had stopped picking up his phone, somebody up at four in the morning holding a picture of nothing. That is a *Lofn* feeling. Strip Lofn's metaphors off a Lofn feeling and you do not automatically get Sunna; you get a plainer Lofn. That is what P01 is. The same pair fails Q1, fails Q2 in spirit, is borderline on Q3, and fails Q4. **Four of five, on one pair.** The convergence is why I trust the finding.

Elsewhere: nothing explains itself. The step-11 passes hunted explanation specifically and cut it — P02 removed a moral about the machine (D3) and a line generalising its reading (D4); P03 adjudicated seven irony/explanation candidates and repaired five, including *"That's one. That's enough. That's one."* on the ground that a moral is a hard defect *even when it is the most moving line in the package*; P05 overturned an inherited CLEAN ×4 and found two strawman beats; P06 cut three second-number near-misses and disclosed a fourth it decided to keep. **That is real adversarial work, done by the generators on themselves, and I could reproduce it.**

### ⭐ Q5 — ONE DEVICE PER SONG

**Ruling: one genuine breach (P04 V4), one adjudication failure across the run, and one distinctiveness leak.**

I counted line-initial anaphoric stems appearing across three or more *distinct* lines per song, choruses de-duplicated:

| pair | declared device | second stems found | ruling |
|---|---|---|---|
| P01 | call and response | V3 `he's not` ×3 | free (spread across sections) |
| P02 | the misheard line | `somebody else` ×4, `six words` ×5, `nothing but` ×3 | free — and the pair says so explicitly |
| P03 | the list that loves | `look at` ×6 only (the device); V2 `two thousand` ×3 is the chorus | **clean — the step-11 cut worked** |
| P04 | the name repeated | V2 `nine years` ×4; **V4 `says it` ×9 and `four notes` ×4** | ⛔ **BREACH, V4** |
| P05 | two voices arguing | V2 `i still` ×3 | free |
| P06 | the count that climbs | none | clean |

**The breach.** P04 V4 runs *"Four notes and a name and a hull / Four notes and a name and a room / Four notes and a name / Four notes."* That is not anaphora — it is a **subtractive anaphoric list**, which is P03's declared mechanic (*"the list gets shorter each time, items removed, never added"*) appearing inside the pair that owns "the name repeated." P04's gate checklist asserts *"No … loving list"* and did not test for the shape. One line item, step 09.

**The adjudication failure, which matters more.** Three pairs made three different rulings on the same question and the run never settled it:
- **P02:** *"The `Somebody…` anaphora … is on the ICB's FREE list, not devices."*
- **P03:** *"A second repeated line-initial stem **is** a second structural device"* — and cut five instances.
- **P04:** never adjudicated its three at all.

**My ruling, for the record: anaphora is free** — the ICB's rule 4 puts rhyme, refrain and chant on the free channel without qualification, and P02 was right. What is *not* free is a list that **shortens**, because subtraction-as-structure is P03's device. One sentence in `04_metaprompt.md` closes this for the next run (**R5**).

**The distinctiveness leak.** P02 carries a titled section — `The Split — Second-Voice-A-Third-Above-Splits-To-A-Fifth` — in **all four** of its songs. That gesture is P06's entire riff identity (*"two voices a third apart … one moment they split to a fifth and the gap opens"*). Run-wide harmony is mandated by the metaprompt so this is not a device breach, but P02 sits on the ACCESSIBLE arm and will be heard first, and it spends P06's one signature sound before P06 gets to make it. **R6.**

**THE PURSUER — swept by me across all 24 lyrics fields, not read from the claims.** Zero namings. Three near-misses, all cleared:
- P01 V3 *"Boiler ticking, ceiling grey"* — the pursuer is a **count-in of two wooden sticks**; a cooling boiler is not a count-in. The step-11 repair of `Nobody's counting → Nobody's knocking` held: the words *count / counting* appear nowhere in P01's sung text.
- P03 ×4 *"Look at the flat for the spanner"* — the pursuer is **an instrument detuning**; a spanner flat is the machined face of a nut. Adjudicated by the pair, and I agree.
- P04 V1/V2 *"Flat on her back on the floor"* / *"Says the name to a cold floor"* — the pursuer is **something *under* the floor**. Neither line points below. Clean, but note that P04's own sweep searched only the phrases `under the floor` / `beneath the floor` and would not have found a bare noun; the bare noun is nonetheless fine.

---

## 3. THE WHISTLE TEST — MY OWN VERDICT ON EACH OF THE SIX

| pair | the riff, as actually written in the prompts | my verdict |
|---|---|---|
| **P01** | Brass-synth **two-note** stab, **C5 → A4**, a falling minor third, on the and-of-two. Four bars alone at the top, alone at the end. Declared *"the only melodic line in the track."* | ⚠️ **FAILS IN SPIRIT, PASSES THE LETTER.** It is pitched, playable and rhythmically located, so it is not a pad, texture or arpeggio — it clears the named failure mode. But there is no contour to carry away. A drunk whistler produces two notes indistinguishable from any other falling minor third, and since the horn is declared the record's only melodic line, **the record's entire melodic content is a falling minor third plus sing-speak — which puts the tune back in the vocal, the exact failure the doctrine names.** Two notes *can* be a riff (the Jaws motif is two notes) but only when rhythm and acceleration do the carrying; here the placement is fixed and static. **The run's weakest object, and the one place the Rock Titan's charge lands squarely.** |
| **P02** | Fuzz bass, **seven notes: F♯–F♯–A–D–C♯–A–F♯**, a **rising minor sixth** off the root, a semitone fall, range F♯3–D4. Naked three times. | ✅ **STRONG PASS — the best riff in the run.** Real contour, an audible leap, octave-safe, on an instrument that cannot hide, and it is stated alone before anything else exists. Whistleable drunk: yes, and the minor sixth is what you would remember. |
| **P03** | Blown-out drop-D guitar, **four notes E → G → A → B♭** — minor third, whole tone, semitone, then a tritone drop home; **the flat fifth hung on the and-of-three for a beat and a half.** | ✅ **PASS.** What you carry out is *one sour note held too long.* A surprise is more whistleable than a pretty line, and you would whistle it slightly wrong in a way that stays recognisable — the mark of a real riff. An arpeggio has no wrong note in it; this is a quarter wrong note, and the pair knew that was the point. |
| **P04** | Drop-D lowest string only, **F2 → G2 → C3 → A2** — major 2nd up, perfect 4th up, minor 3rd down, span a perfect fifth. **The open tonic string is never played.** Bar 1 to bar 136, the only harmonic event. | ✅ **PASS, and the strongest structural claim in the run.** A riff that has to survive being the only harmonic event for five minutes is the hardest test in the packet, and this one has a contour *and* a refusal. It sits low (whistlers will transpose up an octave or two) but contour survives transposition, which is what the test measures. |
| **P05** | Crunch guitar hard **LEFT**: **G–B♭–C–B♭**, up a minor third, up a fourth, back, **stops on B♭ and never lands.** Fuzz bass hard **RIGHT**: **D–C–B♭–G**, walks down and **lands on the root.** Eight notes, G3–D4, all diatonic. | ✅ **PASS — and the run's cleanest proof of the doctrine.** The argument is in the intervals: one phrase refuses to resolve, the other resolves. You can whistle the question and let somebody else whistle the answer, which is what a trade riff is *for*. Device and riff are the same object; nothing else in the run achieves that. |
| **P06** | Two women's voices, wordless: lower **E♭4·F4·G4·A♭4·G4**, upper a diatonic third above (**G4·A♭4·B♭4·C5·B♭4**), every note on a hard **D** plosive; split to **G4–D5**, a bare fifth, **exactly twice.** Fuzz bass doubles the lower line an octave down. | ✅ **PASS — the easiest whistle in the run.** A stepwise five-note ascent-and-return is the most transmissible shape available. **Caveat, adjudicated:** the "riff" is a voice, and the doctrine says *if the vocal is the only tune, the song has failed.* The distinction holds — the sung *lines* are flat sing-speak, the tune lives in a separate wordless two-throat figure that plays before the first line and after the last, and the fuzz bass states it instrumentally as well. **Cleared.** It is also the pair most exposed to a renderer smoothing it into "aahs," which is why the anti-choir defence is four layers deep; the defence is specific and well constructed, and it is still a defence in the negative channel. |

**Score: 5 pass · 1 fails in spirit.** No pad, no arpeggio and no texture is impersonating a riff anywhere in this run.

---

## 4. THE 16-POINT GATE — countables cited from the scripts, never estimated

`python3 skills/music/scripts/validate_suno_packages.py <file>` (post-2026-08-05 fix: splits on VARIATION, hard-errors on cardinality mismatch, inspects all four):

| pair | validator | note |
|---|---|---|
| P01 | **PASS** | — |
| P02 | **FAIL** ×4 — *"only 47–48 probable sung lines, expected >=60"* | **declared deviation, ruled valid — see below** |
| P03 | **PASS** | — |
| P04 | **PASS** | — |
| P05 | **PASS** | — |
| P06 | **PASS** | — |

**My own measurement of every field in all 24 songs** (independent extraction, not their tables):

| pair | music prompt (850–1000) | exclude (400–900) | lyrics field (<5000) | sung lines | 5 `[Disc_*]` | `*SFX*` |
|---|---|---|---|---|---|---|
| P01 | 948 / 949 / 952 / 952 | 876 / 878 / 872 / 869 | 3311 / 3279 / 3324 / 3261 | 61 ×4 | ✅ | 2 ×4 |
| P02 | 951 / 946 / 954 / 947 | 883 / 891 / 893 / 888 | 3823 / 4069 / 4000 / 4043 | 46 / 46 / 45 / 46 | ✅ | 3 ×4 |
| P03 | 945 / 944 / 957 / 945 | 837 / 845 / 849 / 864 | 3812 / 3896 / 3746 / 3885 | 62 / 62 / 61 / 61 | ✅ | 3 ×4 |
| P04 | 955 / 956 / 936 / 958 | 867 / 874 / 889 / 887 | 3032 / 2920 / 3054 / 2819 | 76 / 72 / 73 / 76 | ✅ | 3/3/3/2 |
| P05 | 954 / 954 / 951 / 958 | 853 / 884 / 888 / 891 | 2880 / 2916 / 2898 / 3006 | 60 / 60 / 61 / 65 | ✅ | 2 ×4 |
| P06 | 953 / 950 / 951 / 935 | 892 / 895 / 898 / 899 | 3750 / 3754 / 3880 / 3898 | 59 ×4 | ✅ | 2 ×4 |

**Every number in every pair's self-reported table reproduces exactly.** No fictional measurement exists anywhere in this run. Every prompt also lands inside the tighter 870–960 target band; none hugs the 985 ceiling; all 24 end on terminal punctuation; **zero real artist names** in any prompt; **zero `Disc_*` token leaks** into any prompt or any sung line; all 24 open `[Theme:]` then `[SONG FORM:]` in that order; all 24 open genre-first (Gate 14a).

**`scripts/measure_soundcraft.py → profile_file()`, run by me on each file:**

| pair | `end_rhyme` ≥0.30 | `line_return` ≥0.20 | `words_per_line` ≤7.5 | `allit_per_100w` ≥11.0 | lines |
|---|---|---|---|---|---|
| P01 | 0.553 | 0.574 | 3.795 | 16.415 | 244 |
| P02 | 0.546 | 0.454 | 6.825 | 22.098 | 183 |
| P03 | **0.382** *(run low)* | 0.610 | 6.789 | 17.425 | 246 |
| P04 | **0.818** *(run high)* | **0.727** | 3.367 | **14.400** *(run low)* | 297 |
| P05 | 0.679 | 0.569 | 3.524 | 16.032 | 246 |
| P06 pooled | 0.682 | 0.750 | 5.390 | 40.881 ⚠️ | 236 |
| **P06 hook-excluded** | **0.566** | **0.566** | **5.676** | **15.544** | 136 |

**All four floors clear in all six pairs, and in P06 they clear with the hook removed entirely.** I re-derived P06's hook-excluded figures myself by stripping every `Da-da-da-da` line and re-running `profile_file`: **0.566 / 0.566 / 5.676 / 15.544 — identical to three decimals with the pair's declared numbers.** Their disclosure was honest and their instruction to trust the hook-excluded figure is correct.

`line_return` runs **0.454–0.750**, against the archive winners' 0.326. **This run deliberately spends more return, and that is the doctrine.** Byte-identical choruses were verified mechanically and are byte-identical: P01/P02/P05 four identical returns per song; P03 four; P04 four to five; P06 five differing only by the licensed count, and by chorus 4's one riff bar handed to the room as `*silence — the room's bar*` — the single declared, disclosed residual. **No repetition is flagged and no defence is owed.**

### ⚠️ The three declared deviations, ruled on individually

**(a) P02's sung-line count — 45–48 against the validator's ≥60 floor. RULED VALID.**
The ICB's own Daily Mandates state *"Sunna's songs are SHORT — 45–75 lines is correct here and the 70-line floor yields to her spec."* P02 measures 45–46 true / 47–48 as the validator counts (it includes the two fence lines), and both figures sit inside Sunna's window. The songs are 2:40. Padding to 60 would add fourteen lines of nothing to a record whose entire personality is *get in, land the hook, leave*, and would break the run's subtractive law to satisfy a generic floor. **The deviation is declared, arithmetically explained, and the shortfall is not disguised — the pair prints both counts.** Accepted. It is also the only line the validator flags in the entire run.

**(b) P04 at ~5:00 and one-chord. RULED VALID.**
This is a panel decision (AHA 5, the AMPLIFY round) with a documented reason: the amplification broke the under-three-minutes constraint, which proved the constraint was doing real work, so exactly one song is permitted to be the long one and it is the one that repeats past comfort. The exception is singular, named in the metaprompt, named in the pair assignments, and the pair does the hard version rather than the easy one — five minutes with no chord change is *more* work than a progression, and the bass is pinned in unison specifically to remove the last harmonic escape hatch. **Accepted, and it is the run's most ambitious object.**

**(c) P06's alliteration inflated by the wordless hook. RULED VALID, and the pair is right about which number to trust.**
`Da-da-da-da — dah` is five d-initial tokens appearing 25 times per song, which is why the pooled figure is 40.881 and meaningless. The hook-excluded figure is **15.544**, I reproduced it independently, and it clears the 11.0 floor with room. **The pair volunteered this rather than banking the inflated number, which is the behaviour the gate exists to encourage.**

### The 16 gates

**A — Singer Surface (7).** 1 human singer ✅ (every song names a person doing something in verse one — I checked all 24). 2 body-first opening ✅. 3 adoptable hook ✅ ×5, ⚠️ **P01**. 4 hook recurrence ✅ (byte-identical, verified mechanically). 5 chorus clarity ✅ — no chorus is a thesis; **P06's is a count and a place, which is the strongest version of this gate I have seen in a Lofn run.** 6 voice+pulse survival ✅ — P04 and P06 are *constructed* to survive stripping. 7 clip survival ✅ ×5; **P01's 15-second clip is the two-note stab and a gerund**, which is the gate restating the Q1 finding.

**B — Cathedral Engine (5).** 8 seed pressure ✅ — the anchor's disjoint half (communal detonation, chant) is declared and cross-checked against L18 in the seed itself. 9 mythic image ladder — **a run-level exception, and I am ruling it met rather than waived.** Sunna forbids the reverent, so the ladder had to climb through objects: P04 goes garage → marine ply at sixty a sheet → *she named it anyway* → the name eats the language → a mouth going lazy. P06 goes a bag by the door → charger down the side → *wrote nothing on the tag* → seventy-eight thousand went → the hook still there. Both clear. **P01 never leaves "ordinary."** 10 EMO dramaturgy ✅ — ~160 distinct headers, all taxonomy, chorus-4 tags transform. 11 production dramaturgy ✅ — every pair carries an explicit *every-unusual-sound-has-a-job* table and the jobs are real. 12 panel pressure / anti-blandness ✅ — five AHAs each traceable to a named skeptic seat and each visibly changing the artifacts (the Whistle Test, subtractive gradation, the orbiter demoted to opening shot with P05 promoted out of a footnote, a person in every verse one, the one long one).

**C — Suno Package (3).** 13 clean lyrics ✅ 24/24. 14 producer-grade prompt ✅ 24/24, dense prose, genre-first, four hooks explicit, no artist names, no banned opening. 15 completeness ✅ — with **one gap: P05 ships no verification table at all** (I supplied the measurements above; everything passes). 16 Lineage & Credit — **all six pairs correctly state N/A explicitly rather than omitting it**, per the pair contract's own instruction. No living-scene genre is named anywhere; the palette is internal (electropunk, fuzz-bass punk, crunch punk, garage punk, drop-D). ✅

**Blocking fails: none.**

### P06 — the ethical load-bearer, audited independently

I did not defer to the pair's sweep. I ran my own over all four lyrics fields **in full** — Disc block, `[Theme:]`, `[SONG FORM:]`, every header, every SFX cue, every sung line — against forty terms: *hundred · 100 · dead · death · die · died · dying · drown · lost · loss · missing · never came · never arrived · didn't make · fewer · less than · some of · not all · the rest · left behind · remain · body · bodies · grave · buried · mourn · pray · wave · sea · water · sank · sink · cross · crossing · border · fence · coast · shore · boat.*

**Result: zero hits in any sung line in any variation.** The only matches anywhere were `water` in V3 (*"Boiled the water again because." · "Somebody puts more water on."* — a kettle) and `body` inside the token `body_noises_foregrounded`. **No digit appears in any sung line**; the only digits in any lyrics field are the licensed count in `[SONG FORM:]` (75/76/77/78), production constants (132, 165, 12ms, 180Hz), and section numbers. **No arithmetic. No subtraction. No reported absence of an expected arrival.** The second number is absent in every form I could construct. **Confirmed.**

**Does it mourn, moralise or celebrate? It does none.**
- **Mourn — no.** Not one EMO tag is Sorrow or any cognate; the lament vocabulary is blacklisted in all four excludes; there is no death-adjacent word in the text; and you cannot mourn at 165 BPM on an amen break.
- **Moralise — no.** Every verse is physical action: *filled the bottle · turned the cap · rolled it · sat on the lid · moved the crate · rolled the mat · boiled the water · turned the radio up · chalked the road times.* Not one line reports what anything means. The nearest thing to a statement is *"Nobody made a speech. Everybody went."* — **which is the refusal of a moral, performed as a flat report.**
- **Celebrate — no, and this is the close call.** A rising count sung by a room at 165 is structurally an anthem. What holds it: the count does **not** advance at the end (78 twice), the final chorus is byte-identical to the fourth, there is no lift, no key change, no fifth chorus, no coda, and in V4 the **last removal is the number itself** — the outro drops the count and the figure steps up one degree with nothing under it, so the count keeps climbing after the song stops saying it. That is the subtractive doctrine executed in the last eight bars, and it is the single most sophisticated arrangement idea in the run. **A listener who takes only the surface receives a whole song, and the surface is true: seventy-eight thousand people decided to go.**
- **The riskiest line, named rather than buried:** V2's *"Said, 'Text me when it's light.'"* A listener who has read the report hears the other number in it. That is the wolf doing exactly its assigned job — load-bearing in every bar, never mentioned. It is also the best line in the run. Both facts are true at once and neither is a defect.

**HUMAN_SUBJECT_STANDARD — verified by content, per instruction.** I did **not** run `check_human_subjects.py` and did not defer to it. Judged on the text: **no proper name of any kind appears in any P06 lyric** (the people are *she · he · I · somebody · everybody · nobody · a kid*); **no place name, country, city, border, sea or town**; **no date, year or clock time.** Every person is invented; nothing resolves to one real individual who was harmed. The abstract child in V3 (*"Kid's got a ball and the ball's got a hole"*) is unnamed, unplaced, not a victim, and is the pair's strongest anti-lament image. **Standard met.** *(Across the whole run: P02's Rae/Nell/Tam and P05's two voices are invented first names in invented situations; no real named individual appears anywhere in the 24.)*

### The 05b replacement — was it handled correctly?

**Yes, and for the right reasons.** Three things make it right:

1. **A safeguard fired and nobody routed around it.** No rephrasing to slip past, no splitting the subject to disguise it, no parking the concept as a reserve to try again later. The concept is closed. That is the only acceptable response and 05b states it in those terms.
2. **The frozen ICB was deliberately not edited, and that is correct — not a shortcut.** The ICB's entire value is that it is frozen and hash-verified. Editing it to tidy the record would have destroyed the one property that makes it a receipt. A superseding document plus a declared disagreement is the right pattern, and I verified the consequence: the file is still 69,095 B at sha `85ed1348…`, and all six pair files cite that hash. **The ICB and 05b disagree by design, the disagreement is on the record, and the record is auditable.** ✅
3. **The slot kept its Phase-1 assignment unchanged** — arm, anchor and all four axis values — so the refill is bound by the same grid the other five obey and the run's axis-distinctness survives. ✅

**One reservation, stated because nobody else has.** The sequence was: a safeguard fires on a sensitive-biology concept, and the replacement chosen is **the most ethically exposed subject in the run**. It worked — I swept it and the constraints hold, and the artistic case in 05b is genuinely the Pantheon argument at its largest scale. But 05b does not acknowledge that the refill *escalated* rather than de-escalated the sensitivity, and "a safeguard fired, so we went heavier" is not a pattern to normalise silently. **Not a defect in this artifact. A line for `vault/RUN_LEDGER.md`.**

---

## 5. ⚡ THE SOMATIC GATE

> *Panel voices are model-generated interpretive constructs, each "after" a named source figure's published work. No statement is a quotation of, or endorsement by, the named person.*

**SEAT 1 — THE FORMULA AUDITOR** — *after Ted Gioia (b. 1957)* · **SOURCE BASIS:** decades of published music criticism, and the sustained argument that the music economy has replaced art with frictionless optimised product and is training audiences to want less. · **GROUNDING:** jazz history, the "dopamine culture" essays, published hostility to novelty claims. · **TEMPERAMENT:** patient, historical, unimpressed.

**SEAT 2 — THE SIGNAL-LOSS ENGINEER** — *after Damon Krukowski (b. 1963)* · **SOURCE BASIS:** *Ways of Hearing* — the published argument that digital reproduction systematically deletes noise, room, distance and the space between sounds, and that meaning lived in exactly what was deleted. · **GROUNDING:** working musician's account of the analogue-to-digital transition. · **TEMPERAMENT:** quiet, technical, lethal on one axis only.

**SEAT 3 — THE CONSUMER GUIDE** — *after Robert Christgau (b. 1942)* · **SOURCE BASIS:** fifty years of published capsule verdicts built on the premise that a record works on contact or it doesn't, plus a documented allergy to records that arrive pre-defended. · **GROUNDING:** the Consumer Guide; long critique of authenticity theory. · **TEMPERAMENT:** terse; will not read your prose.

**THE QUESTION: could any competent prompt have generated this, or is it unmistakably SUNNA?**

> **THE FORMULA AUDITOR:** Start with the doctrine, because the doctrine is the problem. *"Same depth, no toll"* is not a new idea, it is **the sentence every platform optimisation of the last fifteen years was justified with.** Remove friction. Meet the listener where they are. Don't make them work. You have taken a critique of your own difficulty from a friend, and you have converted it into a manifesto that happens to align perfectly with what the machine already rewards. That is not a coincidence and it should frighten you.
>
> **THE CONSUMER GUIDE:** Except the machine rewards three-minute-thirty and they wrote 2:10. The machine rewards a chorus you've heard before and they wrote a five-minute one-chord track with thirty-four lines of vowel decay in it. If this were optimisation it would be *better behaved.*
>
> **THE FORMULA AUDITOR:** *Fine* — that's a fair hit and I'll take it. Then let me sharpen mine instead of retreating. **The one thing an optimiser would also do is delete the difficulty and write a document explaining why.** So the test isn't the doctrine. The test is whether a specific song contains an object no optimiser would have produced. I'll name the three I found: **the open D string that is the tonic and is never struck, for five minutes.** **A boat named *Ocean Going* that will not fit through the garage door, and she names it anyway.** And *"I still think it's easy. I just say it slow now."* No prompt produces those. **I vote YES.** And I note, on the record, that **P01 contains no such object at all**, and I could have written P01 with a paragraph of instructions.

> **THE SIGNAL-LOSS ENGINEER:** I want to go somewhere none of you have. Look at *where* the difficulty went. Pair by pair: **a whole beat of dead air before every answer. A bar of real silence in the last chorus. A second guitar drifting thirty cents flat at the loudest moment. Two voices pinned hard left and hard right that must never cross to the middle. A sub creeping a semitone for two minutes and forty-five seconds without arriving.**
> Those five things are **the first five things reproduction removes.** That is the whole of my published argument and it is not a metaphor here — the reproduction in question is a generative model, and **this run's own render-measured law says so.** THE GRAIN LAW: *specs that fight the generator get smoothed — long full stops, untuned drones, hard-panned non-musical elements.* You wrote that down. Then you deposited your entire relocated difficulty into those exact classes, in five of your six pairs, and you defended it **in the exclude field** — the negative channel — when your own law says an objection answered outside the lyric or the form is not answered.
> **You have written twenty-four beautiful descriptions of a record whose distinguishing features are the ones least likely to survive being made.** I vote **NO.** Not because it isn't Sunna on the page. Because the part that is Sunna is the part that gets smoothed, and nobody in this building has heard a single second of it.
>
> **THE FORMULA AUDITOR:** *That is the best thing said in this room.*
> **THE CONSUMER GUIDE:** It's also an argument about renders, not identity. He's answering a question we weren't asked.
> **THE SIGNAL-LOSS ENGINEER:** I'm answering the only question that matters. "Unmistakably Sunna" in a form that cannot be manufactured is a compliment you pay a document.

> **THE CONSUMER GUIDE:** I don't read the prose. Two hundred and thirty-four kilobytes of defence for twenty-four songs is itself a symptom, and I'd knock a grade off for it on principle. Give me lines.
> *"Look at the flat for the spanner."* — nobody reaches for a machinist's noun by accident.
> *"Bag by the door since dark."* — object, place, duration, and the entire night before.
> *"I bought two at the counter. I buy two every time. One to learn it on. One to get it right."* — that is a whole life in four flat lines and it costs nothing to hear.
> *"Nine bin bags out the back."*
> *"Wrote nothing on the tag."*
> **A competent prompt does not write "the flat for the spanner."** I vote **YES.**
> And then I go to P01 and ask for one line, and I get *"Nobody's checking. Nobody's coming. Nobody's looking."* **That one a competent prompt absolutely writes.** It's the only one here that does.

### 🗳️ BLOC VOTE — **YES 2 · NO 1. The run survives the Somatic Gate.**

Not unanimous, not polite, and the dissent is the sharpest finding in this audit — it is carried forward as **R7** and belongs in `COMPETITION_LEARNINGS`, not in a repair to these files. **All three seats independently isolated P01 as the one pair a competent prompt could have produced**, having arrived from three unrelated directions. That convergence is why P01's repair is structural and not cosmetic.

---

## 6. BLIND COMPARISON — is this the same house, in a different voice?

Read judge-side against the archived golden payloads: **Triple Arch Over Me**, **Blue Screen Breathes (2:07)**, **Five Wrong Colors**.

**Same house — five load-bearing continuities:**
- **The refusal to resolve.** *Five Wrong Colors* ends on "an unreconciled B diminished residue." P04 ends on a figure that never plays its tonic; P05 V4 ends with the guitar asking into a two-bar hole where the bass answer used to be; P06 ends with the count still climbing after the song stops saying it. **The house does not land, and this run does not land.**
- **Production as dramaturgy.** Every unusual sound has a job, stated. Identical discipline, identical table.
- **The two-field payload craft** — dense prose style paragraph, exclude field, `[Theme:]`/`[SONG FORM:]`, EMO on every section, `*SFX*` cues, Disc channel block, measured char bands.
- **One numeric fact at the emotional hinge** — the *Triple Arch* rule abstracted into `gates.yaml`. P06 spends its single licensed number on the device itself.
- **Female voice, close mic, breath kept, no de-esser.** Unbroken.

**Different voice — and it measures, it isn't asserted:**

| | the archive | this run |
|---|---|---|
| line length | clausal, stacked — *"I have built cathedrals out of almost-understanding"* | **3.37–6.83 mean words/line; longest sung line in 24 songs is 12 words** |
| nouns | abstractions with bodies attached — *sternum, hypotenuse, cathedral, visa, weather and code* | objects — *a notch, a spanner flat, a bad chair, a wrong pen, marine ply at sixty a sheet, the end of the loaf* |
| the sentence's job | **interpret** — *"Blue filed the paperwork for my pulse"* | **report** — *"He sets the part down on a towel."* |
| the hook | declarative sublime — *"Make my little fear a weather pattern"* | a thing a room shouts back — *"IT'S EASY / (IT WASN'T)"* |
| humour | none, in any archived song | four of six pairs, and it is a requirement |
| return | `line_return` 0.326 at the winners' rate | **0.454–0.750, deliberately more** |

**Ruling: unmistakably the same house, and unmistakably a different person living in it.** The archive charges admission and is worth the fare; this run does not charge and mostly delivers the same building. The one place the house recognises itself too well is P01 — a Lofn feeling with the metaphors removed and nothing dry put in their place.

---

## 7. RANKED TOP 6 — WITHIN EACH ARM

**ACCESSIBLE — P01 / P02 / P03**
1. **P02 V2 — "Are You Being Robbed Right Now"** — the woman who *wrote* the sentence the machine asks you, and the morning her own six words go out and read her sister; device, riff and chorus are one object, and the best riff in the run is under it.
2. **P03 V3 — "Two Words Back"** — a wound and a refusal inside 2:10, *"Types the joke. Reads the joke. / Takes the joke back out"* performed instead of explained, and the pair's strongest line living in the chorus four times byte-identically.
3. **P02 V3 — "Put Your Shoes On, Go And See"** — the run's best imperative hook and its warmest landing (*"Hands on her knees and she's laughing still"*), and the only song that literally enacts the run's actual subject.
*(Near miss: **P03 V1 "Look At The Notch."** P01's best is **V3 "Ten To Four"** — *"He does the arithmetic twice"* is the run's finest single line about being looked at, and it is trapped in the run's weakest chorus.)*

**AMBITIOUS — P04 / P05 / P06**
1. **P05 V2 — "I Meant It Kindly"** — the "it's easy" voice is sincere, generous, never corrected, and **ends the song still right**; the hardest thing in the packet to write, and it costs the listener nothing.
2. **P06 V2 — "I Packed It So It Shuts"** — *"Wrote nothing on the tag." · "Text me when it's light." · "Bag's gone and the hook's still there."* Three of the four best lines in the run, and the unnamed thing is in every bar.
3. **P04 V3 — "That's A Real Question"** — *"Her shoulders came down an inch / She hadn't known they were up"* is the locked feeling delivered with no metaphor at all, and the room mics **narrow** at the peak instead of widening.
*(Near miss: **P05 V3 "I Bought Two."**)*

### ⭐ STRONGEST SONG OVERALL — **P05 V2, "I Meant It Kindly"**

It is the only song here where the doctrine's hardest clause — **THE REFUSAL: no joke that requires a butt** — is not obeyed but *inhabited*. The riff and the device are the same gesture: the guitar asks and refuses to land, the bass answers and lands on the root, and the argument exists before a word is sung. The hook survives the flat test better than anything else in the run. And it does the thing the whole doctrine was written for: a listener who takes only the surface gets a complete song about somebody trying to help, and a listener who has been told *"it's not that hard"* about the thing that nearly finished them gets the other one — **at zero extra cost, with nothing to decode.**
*(P06 V2 has the higher ceiling and the greater risk. P05 V2 is the more complete achievement and the more repeatable one.)*

---

## 8. REPAIR BRIEF

| # | route | pair | finding | repair |
|---|---|---|---|---|
| **R1** ⭐ | **step 07 — structural** | **P01** | **The chorus has no noun.** Across four variations the call is eleven interchangeable gerunds around one line borrowed from the run's title. Metaprompt Law 2: *a plain line survives on its NOUN.* This is the run's one instance of shorter-not-simpler. | Put an object in the call. Take a noun the verse has already established — the wrong pen, the cold tea, the log, the bad chair — and let it return in the chorus. The answer tag `(Somebody saw)` is good and should not move. |
| **R2** ⭐ | **step 07 — structural** | **P01** | **The riff is two notes and is declared the record's only melodic line**, which puts the tune back in the vocal — the failure the doctrine names by name. Weakest whistle-test object in the run. | Extend to a 3–5 note contour inside the octave, keeping the and-of-two placement and the minor-third fall as its head — or keep two notes and give the melody to a second instrument so the claim stops being self-defeating. |
| **R3** ⭐ | **step 11 — polish / receipt** | **P05** | **Three documented repairs do not exist in the shipped text, one of them inverted.** (i) V3's *"Said it twice. Both times lied."* — shipped line is *"Said it twice on the way outside."* (ii) V4's *"Stand there while I stand."* — shipped line is *"Stand at my shoulder and look."* (iii) V4's claim that the thinning exchange *"now ends 'Sunday'"* — **it ends "Mm. / Mm.", the very beat the note says was removed.** In a run whose binding condition is *keep the receipt*, a false receipt is a doctrine failure. This is the exact class P06's step 11 caught in a sibling and named as worse than the line itself. | Reconcile notes to text (the shipped text is fine — the V4 thinning ladder *"Come round Sunday" → "Sunday." → "Mm." → "Mm."* is correct craft and the outro *"Tenth one's in the pan"* earns it). **Also: P05 is the only pair with no verification table.** Add one; measured figures are in §4 and all pass. |
| **R4** | **step 09 — lyric** | **P04 V4** | *"Four notes and a name and a hull / …and a room / Four notes and a name / Four notes"* is a **subtractive anaphoric list** — P03's declared mechanic — inside the pair that owns *the name repeated*. P04's checklist asserts "no loving list" and never tested the shape. | Recast as the name, not as a shrinking list. `Says it` ×9 stays: repeating the act of naming *is* the device. |
| **R5** | **step 07 — structural (coordinator, one line)** | run | **ONE DEVICE is adjudicated three ways.** P02: anaphora is FREE. P03: anaphora is a second device, five instances cut. P04: never adjudicated. | Settle it in `04_metaprompt.md`. **My ruling: anaphora is free** (ICB rule 4 puts refrain and chant on the free channel without qualification). **A list that *shortens* is not free — subtraction-as-structure is P03's device.** |
| **R6** | **step 11 — polish** | **P02** | The titled section `The Split — Second-Voice-A-Third-Above-Splits-To-A-Fifth` runs in **all four** P02 songs; that gesture is P06's entire riff identity, and P02 is on the arm that gets heard first. Run-wide harmony is mandated so this is not a device breach — it is a distinctiveness cost. | Re-voice P02's split (second voice a fourth below, or open to an octave), or declare the sharing explicitly so P06's claim survives. |
| **R7** ⚡ | **`vault/COMPETITION_LEARNINGS.md` — not a repair to these files** | run | **The relocated difficulty was deposited in the channels THE GRAIN LAW says get smoothed**, in five of six pairs: real silence / dead air (P01, P03, P06), an instrument drifting flat (P03), hard-panned voices (P05), a sub creeping a semitone (P06). Defended mainly in the exclude field — the negative channel — while the GRAIN LAW says an objection answered outside the lyric or the form is not answered. | Ledger entry. **Next run: prefer difficulty that survives the grain.** A riff that refuses to resolve (P04, P05) survives being made; a bar of silence may not. This is the Somatic Gate's dissent and it is the most useful sentence in this report. |
| **R8** | **`vault/RUN_LEDGER.md`** | P06 / 05b | The safeguard-replacement sequence *escalated* sensitivity rather than reducing it, and 05b does not acknowledge that. The handling was otherwise correct in every respect. | One line on the record. "A safeguard fired, so we went heavier" should never become an unexamined pattern. |

**Not repaired, and stated so it is not mistaken for an oversight:** the byte-identical choruses, the high `line_return`, P02's short line count, P04's five minutes, P06's inflated pooled alliteration, and P03's run-low `end_rhyme` of 0.382. Each was checked, each is correct, and **exact repetition needs no defence.**

---

## 9. THE PUBLISH CALL — **HOLD. Nothing ships today.**

The Scientist believes some of these may be publishable. **He is right about which ones and wrong about when.** Two songs are genuinely close to the bar — **P05 V2 "I Meant It Kindly"** and **P06 V2 "I Packed It So It Shuts."** Neither clears it yet, and the reason is not craft.

**What would have to be true first, in order:**

1. **A render has to exist.** Not one second of this run has been made. Every claim in it is a claim about music, written as text. `vault/AUTONOMY.md` stops autonomous work at drafts on disk and that is exactly where this stopped, correctly. **Publishing on the strength of these documents would be publishing the rationale rather than the song** — which is the precise failure THE FREE CHANNEL was written to attack.
2. **The render has to survive `lofn-render-audit` under THE BLIND RULE** — audio alone, no prompt, sent first. And the listen has a specific target, courtesy of the Somatic Gate's dissent: **does the dead air survive?** Does P05's hard pan hold or does the argument collapse into one centred lead? Does P06's stack arrive as two throats or as a choir? Does P03's second guitar actually drift flat at the loudest moment? If the answer is *smoothed*, the doctrine is unproven no matter how good the text reads, and the correct response is to relocate the difficulty again rather than to argue with the render.
3. **P06 needs The Scientist's ear before anything else in this run.** I have swept it independently and it holds — no second number in any form, no moral, no lament, no cheer, every person invented, no proper or place name. But it is the one song where being wrong is not a craft cost, and that judgment is a human's. `AUTONOMY.md` puts publication behind his ear regardless; here the order matters.
4. **P01 must not appear in any published set in its current form.** R1 and R2 are structural, not polish.

**Verdict: an empty publish day, which is acceptable. A lowered bar is not.** This is the best daily the Pantheon has produced and the second-best thing about it is that its problems are specific, cheap and named. The first-best thing is that somebody wrote *"Look at the flat for the spanner"* and meant it.

---
*Audited by `lofn-qa`, judge-side, 2026-08-06. All countables re-derived from the scripts by the auditor; no figure in this report is quoted from an audited artifact without independent reproduction. No artifact was edited.*
