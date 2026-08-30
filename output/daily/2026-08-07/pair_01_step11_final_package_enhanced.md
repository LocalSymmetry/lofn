# PAIR 01 — STEP 11 · FINAL PACKAGE (ENHANCED)
**Run:** `2026-08-07-daily-music-indignation` · **Pair:** P01 **"THE ONE WHO LEFT IT IN"**
**ACCESSIBLE · INDIGNATION · NEWS · FORM RULE A (accretion) · ⚡ FAST/LOUD PAIR 1 OF 2**
**Tier:** step-11 enhancement · **Verdict: ENHANCED** (not rejected — see §ANDON CORD)

**Frozen ICB:** `output/daily/2026-08-07/CREATIVE_CONTEXT.md` — **53,003 B LF-normalised**, sha256 `5e9c7f7f6009fb3c672058c930540be22c8f5517f37537ac3ebd8ae94b75d374` — **re-verified by this tier, matches exactly.**
⚠️ The raw on-disk byte count reads **53,526** because `core.autocrlf` rewrites LF→CRLF on this checkout. The frozen figure is defined **LF-NORMALISED**. Raw sha is `a5a06f1f…`; that mismatch is the line count, **not tampering**. ⛔ **The ICB was not edited, not "fixed", not normalised on disk.**

---

## ⛔ CONTRACT CONFLICT — DECLARED BY NAME, NOT COMPLIED WITH

**`skills/music/steps/11_Generate_Music_Enhancement.md` instructs this tier to embed full Golden Song payloads** — line 69 (*"selected `## Golden Song References`… full available style/music prompt, lyrics"*), line 72 (*"Do not pass links or filenames alone"*), line 246 (*"Include the two selected Golden Song References in the output as calibration examples"*), line 258 (*"including embedded style/music prompt, lyrics, and exclude prompt status"*).

⭐ **`06_music_handoff.md` §1 — "THE CONFLICT THIS DOCUMENT RESOLVES" / "RESOLUTION — DOCTRINE WINS. THE QUARANTINE IS BINDING" — overrides it. I did not comply.**

- ⛔ **No past Lofn lyric, style prompt, title or image prompt entered this tier's context or its output.** Not as calibration, not quoted, not paraphrased, not summarised.
- ✅ What this tier worked from: the **GOLDEN MOVE** (handoff §2), the **Golden Seed** (ICB Slot 1), the frozen ICB, the pair slice (`05_pair_assignments.md` §B, P01 only), and `pair_01_step10_revision_synthesis.md`.
- **Seeds teach; outputs contaminate — including our own.** This is the exact conflict that produced the only real defect of the 2026-08-06 run (**L30**), and it is now stated in writing in the run directory, so no agent has to reason its way there alone.

---

## ⭐ EXTRACTION ASSERTION — PRINTED BEFORE ANY VERDICT IS TRUSTED

```
source                       : output/daily/2026-08-07/pair_01_step10_revision_synthesis.md
'### VARIATION n' blocks     : 4      (expected 4)   PASS
'## 1. MUSIC PROMPT'         : 4      (expected 4)   PASS
'## 1B. SUNO EXCLUDE PROMPT' : 4      (expected 4)   PASS
'## 2. LYRICS'               : 4      (expected 4)   PASS
'## 3. TITLE'                : 4      (expected 4)   PASS
non-empty field assertions   : 16/16 PASS  (empty extraction = HARD ERROR, never a score)
harness                      : scratchpad/step11_p01_p02_extract.py   (pair-namespaced)
```
**The harness independently reproduced the coordinator's re-stat to the character before a single edit was made** (943/958/953/956 · 4400/4344/4429/4223 · 86/86/86/86). That agreement is what licenses the before/after numbers below; without it the deltas would be measuring two different conventions.

---

## THE MEASURED NUMBERS — BEFORE → AFTER, PER VARIATION

`scripts/measure_soundcraft.py → profile()` on the shipped bytes of **this** file. Never by eye.

| metric | floor / band | **V1** | **V2** | **V3** | **V4** |
|---|---|---:|---:|---:|---:|
| MUSIC PROMPT chars | 850–1000 | 943 → **943** | 958 → **958** | 953 → **953** | 956 → **956** |
| EXCLUDE chars | ≤1000 | 416 → **416** | 428 → **428** | 409 → **409** | 406 → **406** |
| 🚨 **LYRICS FIELD chars** | **<5000 · target ≤4800** | 4400 → **4403** | 4344 → **4350** | 4429 → **4435** | 4223 → **4223** |
| sung lines | 70–120 · hug ≤72 | 86 → **86** | 86 → **86** | 86 → **86** | 86 → **86** |
| `rhyme_return` | ≥0.30 | 0.488 → **0.477** | 0.500 → **0.500** | 0.407 → **0.384** | 0.593 → **0.593** |
| `line_return` | ≥0.20 | 0.279 → **0.279** | 0.279 → **0.279** | 0.279 → **0.279** | 0.279 → **0.279** |
| `line_return` — lexical-only (shouts removed) | disclosed | **0.214** | **0.214** | **0.214** | **0.214** |
| `line_return` — **accretion word stripped** | disclosed | **0.291** | **0.291** | **0.291** | **0.302** |
| `alliteration_per_100w` | ≥11.0 | 12.960 → **12.832** | 13.344 → **13.609** | **11.550 → 12.785** | 13.208 → **13.208** |
| `unique_line_ratio` | ≥0.45 | 0.802 → **0.802** | 0.791 → **0.791** | 0.802 → **0.802** | 0.791 → **0.791** |
| `max_sung_numeric_facts` | ≤1 | **1** | **1** | **1** | **1** |

⚠️ **PROMPT-BLOAT WATCH (n=7 — a refinement step has over-lengthened the prompt on all seven prior occasions, and none of them ever read bloated).**
⭐ **P01's four music prompts were measured, read for generic filler, and shipped BYTE-IDENTICAL. Zero growth. Zero shrink.** They already lead with genre + BPM + key, use only permitted style vocabulary, carry named instrument / named interval / arrangement arc / signature device, contain no artist name, and each ends on the riff alone. There was nothing generic left to cut and nothing missing to add; **the correct step-11 action on a good prompt is to leave it and say so**, not to demonstrate effort.

⚠️ **LYRICS GREW BY 3 / 6 / 6 CHARS AND WERE UNCHANGED IN V4 — justified line by line in §WHAT CHANGED.** Worst case is **4435 against a 4800 target and a 5000 hard cap: 565 chars of headroom.** No boundary hug.

⚠️ **`line_return` COMPANION — AND THE COORDINATOR'S PREMISE DOES NOT SURVIVE MEASUREMENT.**
The tier brief states that P01's accretion word inflates `line_return`, and asks for a companion excluding it. **I measured it and the opposite is true.** Stripping the token *kept* from every sung line and re-measuring returns **0.291 / 0.291 / 0.291 / 0.302 — HIGHER than the raw 0.279**, because removing the word makes *more* lines collapse into identity, not fewer. The device is not carrying `line_return`; **the chorus and the refrain-tag are.**
The companion that actually probes the risk is the **lexical-only** one (all-caps gang shouts removed): **0.214** in all four, above the 0.20 floor without a single shout. *(Reproduces step 10's figure exactly on the enhanced bytes.)*
⭐ **Reported this way round on purpose.** Silently substituting a method that agrees with the instruction would have produced a number that flattered the brief and told nobody anything. The instrument cannot tell "the song returns" from "one syllable returns" — this pair is true on three separate readings, and the reason is not the one we assumed.

⚠️ **V3's alliteration was the thinnest number in the pair at 11.550 against an 11.0 floor. It is now 12.785** — a bigger move than the +0.5 I expected, and **the largest single measured gain in this pass.** ⛔ **It was NOT achieved by writing for the scanner** (L27): the repaired lines are the fast-list verses, where tongue-twisting consonant clusters are what the form actually wants; a list read at 156 BPM is *easier to sing* with them, and every changed line is a better line on its own merits (a creak and a whine are sounds that end up on a master; "leaning back on" and "going sharp" are not).
⚠️ **The trade is declared: V3's `rhyme_return` fell 0.407 → 0.384** (V1's fell 0.488 → 0.477). Both remain well above the 0.30 floor, but **V3's rhyme is now the pair's thinnest number instead of its alliteration**, and I am naming that rather than reporting only the metric that improved. Net judgement: +1.235 alliteration for −0.023 rhyme, in the one variation whose whole form is a list, is the right trade.

---

## ⚡ ANDON CORD — THE REJECT DECISION, MADE EXPLICITLY

I hold reject authority. **I did not use it. Verdict: ENHANCED.** Stated against each criterion so the decision is checkable rather than asserted:

| REJECT criterion | Finding |
|---|---|
| THREAD LOSS | ⛔ Absent. The seed's engine (`the-decision-not-the-sound`) is intact in all four; the addressee is a named person doing a named physical thing in line one of all four. |
| PERSONALITY COLLAPSE | ⛔ Absent. The Reluctant Pop Star is audible as a **rhythmic position and a diction**, not an adjective: shouted sing-speak, monosyllables, gang answers, jokes that are structural. |
| EMO TAXONOMY FAILURE | ⛔ Absent. **12 section headers per variation, all 48 measured well-formed** against `[Section - EMO:<emotion> - <Role> - <cue>]`, ⛔ zero bare AWE/INDIGNATION, and the arc transforms (Amusement/Skepticism → Mirth → Defiance → **Revelation** at the bridge → Recognition → **Acceptance / Admiration** at the outro). |
| GENERIC OUTPUT | ⛔ Absent. Non-quatrain, non-predictable; the accretion is a real structural innovation and the elbow / the empty coat hook / the keys in the teeth are specific. |
| PROMPT FORMAT VIOLATION | ⛔ Absent. Dense paragraph, 943–958, genre-first, no artist names, no key:value brackets. |

**"Don't polish a corpse" — this is not a corpse. But two live defects were found, and one of them a pair-wide claim had already been made about.**

---

## THE SIX ENHANCEMENT AXES — WORKED IN ORDER, EACH VARIATION VERIFIED INDIVIDUALLY

### 1 · THE ADDRESSEE — ⭐ THE AGREEMENT IS THE INJURY

| | line one | the concession | thesis stated? |
|---|---|---|---|
| **V1** | *"Nilton took his hand off the fader."* | line 4 — *"He left it. He was right. Here's the song."* | ⛔ never |
| **V2** | *"Nilton left the fader where it was."* | line 4 — *"He walked out and he was right to."* | ⛔ never |
| **V3** | *"Nilton, hand off the fader, listen —"* | line 4 — *"You left it. You were right. Here's the list."* | ⛔ never |
| **V4** | *"Nilton said it standing at the fader:"* | line 4 — *"I've tried to say it stupid. I can't. He's right."* | ⛔ never |

**4/4 named recipient · 4/4 specific physical act in line one · 4/4 concession by line four · 0/4 thesis stated.**
⭐ The injury is placed correctly in all four: not *"they hate me"* but *"and what I kept would have been worse"* (V1), *"There's nothing under it"* (V2), *"So I'd have kept it wrong"* (V3), *"There's nothing dumb in it. I've kept looking."* (V4). **She is wounded by finding it correct.**

### 2 · THE COUNTABLE OBSTRUCTION — RULE A, AND WHERE IT LIVES

**The word `kept` enters at sung line 9 and is in every sung line to line 86. Nothing is ever removed.**

| verified individually | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| occurrences of *kept* in lines 1–8 | **0** | **0** | **0** | **0** |
| lines 9–86 **missing** *kept* | **0** | **0** | **0** | **0** |
| line 9 is the first chorus line | ✅ | ✅ | ✅ | ✅ |

⛔ **LEAK AUDIT — L22 THE GRAIN LAW, RUN AGAINST THE PRODUCTION SPEC OF EACH VARIATION INDIVIDUALLY.** Searched all four MUSIC PROMPT and all four EXCLUDE fields for the device and for any paraphrase of it (*kept · accretion · every line · never leaves · nothing removed · accumulates · word that stays*): **0 hits in 8 fields.**
⭐ **Adversarial re-run: delete all four MUSIC PROMPT fields entirely — is the device still countable by ear? YES.**
⚠️ **One phrase examined and cleared, with the reasoning shown rather than the verdict alone:** V1's prompt contains *"A rubber toy squeals once, on purpose, and stays."* That is the **diegetic event** of the song (the squeal on the master), not the form rule; the countable obstruction is a **word in the lyric**, and no prompt names it. Cleared, not waved through.

### 3 · SELF-PITY · VENTRILOQUISM · ABSTRACT NOUNS — LINE BY LINE, HIT COUNTS REPORTED

- **Abstract nouns in sung lines** (cost, authenticity, labour, soul, value, meaning, art, truth, purpose — word-boundary scan of all 344 sung lines): **0 hits.**
  - ⚠️ **Disclosed, not hidden:** *reason* appears twice in V4 (*"That's the whole reason"*, *"the reason kept"*). It is **not** on the banned list and both uses have a concrete antecedent quoted two lines above them (*"somebody made that, it happened, it stays"*). Kept and declared so QA can rule rather than discover.
- **Ventriloquism:** **0 hits.** She never writes *as* Nilton and never as a displaced worker. V4's premise — a mocker who cannot land a mock — is **her own** experience, not a borrowed wound.
- **Self-pity (the Morozov tripwire, standing, nameable at line level):** **1 HIT FOUND AND REPAIRED.**
  - ⛔ **V2 — *"Nothing here kept anything for me."*** That is a bid for the listener's sympathy: the room is being scored for having withheld something from her. **REPAIRED → *"The kettle kept warm. Nobody kept it on."*** Same slot, same accretion, no claimant — an object that stayed warm on its own is an observation, not a grievance.
  - Structural clamp intact in all four: *"He kept not turning round. Good. He kept right."* A speaker who blesses the back walking away is not asking to be pitied. **Speed is never mentioned once in 344 sung lines.**

### 4 · LAW 1 — READ UNCHARITABLY, EVERY LINE, HIT COUNT REPORTED

**THE INDIGNATION IS NEVER AIMED AT THE BAND, AT ANALOGUE, AT CRAFT, OR AT ANYONE WHO WORKS THE LONG WAY.**

**1 HIT FOUND AND REPAIRED.**
- ⛔ **V4 gang break — *"LAUGH AT THE MAN WHO KEPT IT IN!"*** An instruction to an audience to laugh **at** the person who did it the long way. The song's arc redeems it; **an uncharitable reader does not get the arc, and Law 1 is not scored on arcs.** The tell was structural: V1 shouts *"CLAP FOR THE MAN"*, V2 *"STOMP FOR THE MAN"*, V3 *"SHOUT FOR THE MAN"* — **V4 was the only one aimed *at* rather than *for*, which is the signature of an artifact rather than an intention.**
  **REPAIRED → *"LAUGH IF YOU CAN! HE KEPT IT IN!"*** — the target is removed and the joke gets **sharper**, because the invitation failing is V4's entire thesis.
- Examined and cleared, with reasoning: *"He kept the squeak. The squeak is bad."* / *"IS IT GOOD? NO!"* — these judge **the sound**, which the Golden Seed itself licenses (*the squeal is worthless; the keeping is the whole value*), and each is immediately followed by the keeping being right. *"Nobody laughed. It isn't funny."* — same. *"no, don't. You won't. That's the point."* — teasing an addressee who cannot hear, which is the run's constraint, not a sneer.
- **⛔ Cross-pair bleed: the slop economy is not mentioned once in P01. P04 carries it. 0 hits.**

### 5 · ⭐ THE FUNNY — COUNTED BY NAME, NOT ASSERTED

| | jokes, named |
|---|---|
| **V1** | **THE ELBOW** (*"Ask what he kept and I'll say: an elbow"* — the whole argument as a body part) · **THE SALES PITCH** (*"Free. Right now."* — an offer nobody asked for) · **THE KEYS IN THE TEETH** · **THE BLESSING** (*"He kept not turning round. Good."*) · ⭐ new: **NOBODY ASKED ME** |
| **V2** | ⭐ **"I'M SINGING THIS TO FURNITURE. FINE."** (the best joke in the pair) · **THE EMPTY COAT HOOK** · **THE BIN HE KEPT NOT EMPTYING** · **STOMP FOR THE MAN** · ⭐ new: **THE KETTLE THAT KEPT WARM WITH NOBODY IN THE ROOM** |
| **V3** | **THE INVENTORY ITSELF** (twenty items, read at 156 BPM) · **"CLEAN AS A PLATE"** · **"YOU KEPT THE CHICKEN. YOU KEPT THE CHICKEN!"** (the double-take) · **"IT GOES ON. IT GOES ON A WHILE."** |
| **V4** | **THE PREMISE** (every voice she owns, tried on one sentence, none of them work) · **THE LAUGH THAT CAME OFF** (*"The laugh came off. The sentence didn't."*) · **THE CARTOON VOICE** · ⭐ repaired: **"LAUGH IF YOU CAN!"** |

**4/4 variations carry named structural jokes. Zero jokes would be a doctrine failure; this is not one.**

### 6 · STYLE VOCABULARY LAW

- ⛔ **Banned primary descriptors** (raw · aggressive · relentless · brutal · explosive · massive · intense · pounding · driving · battle · assault · phonk) scanned across all four MUSIC PROMPT fields: **0 hits.** *(`phonk cowbell` appears in three EXCLUDE fields — that is a blacklist entry, which is the correct and only permitted use of the token.)*
- ✅ **Present as primary descriptors:** bratty · sardonic · confrontational · dry · gleeful · close-mic'd · shouted sing-speak · *snarl held at conversational loudness* · specific physical detail (`four inches off the capsule`, `hard-tiled`, `boot stomps`, `hi-hat clicks`, `bugs on the light`).
- ⛔ **Real-artist names in any Suno field: 0.** Credit lives only in the Lineage block.
- ⛔ **Banned texture words** (tape hiss · vinyl crackle · wow-and-flutter · vintage · analogue warmth · lo-fi · degraded · corrupted · glitching) in any **positive** field: **0 hits.** They appear only as EXCLUDE entries, which is the whole point of the second field.

---

## WHAT STEP 11 CHANGED — THE COMPLETE LIST, NOTHING ELSE TOUCHED

| # | var | change | why | Δ chars |
|---|---|---|---|---|
| 1 | **V4** | `LAUGH AT THE MAN WHO KEPT IT IN!` → `LAUGH IF YOU CAN! HE KEPT IT IN!` | **LAW 1.** Removes the pair's only line aimed *at* the addressee; sharpens the joke. | 0 (32 → 32) |
| 2 | **V2** | `Nothing here kept anything for me.` → `The kettle kept warm. Nobody kept it on.` | **NO SELF-PITY.** Removes the pair's only claim on the listener's sympathy. | +6 |
| 3 | **V1** | `Nobody kept me from saying it. I said it.` → `Nobody asked me. I've kept saying it anyway.` | Flattest line in V1 and faintly self-congratulatory. The replacement is **THE UNDELIVERABLE ADDRESS stated at the hinge** and is dry rather than proud. | +3 |
| 4 | **V1** | title `You Kept It In` → **`He Kept The Elbow`** | V1's lyric is entirely third-person; the old title addressed him in a pronoun the song never uses, and duplicated the chorus. The new title is the song's **strangest and truest image** and the rung its bridge climbs to. | — |
| 5 | **V3** | six list lines re-consonanted (`string → sliding sharp`, `before the note → before the beat`, `chair … leaning back on → chair that kept creaking back`, `bin lid … shutting → slamming`, `count … kept out loud → somebody counted out loud`, `note … going sharp → whine that kept walking up`) | **Alliteration 11.550 → 12.785**, the pair's thinnest margin — and independently better writing: a creak and a whine are *sounds on a master*, "leaning back on" and "going sharp" are not; `slamming` also removes a `shutting` that was duplicated two verses later. ⚠️ Cost: `rhyme_return` 0.407 → 0.384, declared. | +6 |

⛔ **Everything else is byte-identical to step 10, deliberately.** No music prompt was touched. No chorus was touched. No section header was touched. No EMO tag was touched. The accretion was not touched. **A step-11 pass that rewrites a working song to look busy is the failure mode this tier is warned about.**

---

## THE WHISTLE RIFF — UNCHANGED, RESTATED FOR THE RENDERER

**THE SQUEAK MOTIF** — dry-tuned accordion, five notes, A minor: **E4 → G4 → F4 → E4 → A3**.
Intervals **+m3 · −M2 · −m2 · −P5**. Span A3–G4 = a minor seventh — **inside one octave ✅**.
Note 3 (F4) sits roughly a sixteenth behind beat 2 and is **never corrected** — forró's leaning zabumba imported as **note placement**, in the same place every bar. Written as timing, so it survives a renderer; ⛔ never as patina.
**Alone in bars 1–4 before any voice; alone again after the final sung line. Four bars = the shape stated twice, so bar 8 is its fourth statement — singable by bar 8 by construction.** A pad is not a riff; this is a taunt.

## CHORUS-TIMING GATES — ARITHMETIC CARRIED, UNCHANGED

One bar in 4/4 = **4 × 60 ÷ BPM** seconds.

| | BPM | bar | bars before chorus | chorus arrives | ≤0:25 |
|---|---:|---|---:|---:|:--:|
| **V1** | 152 | 4 × 60 ÷ 152 = **1.579 s** | 4 + 8 = **12** | 12 × 1.579 = **18.95 s** | ✅ |
| **V2** | 76 | 4 × 60 ÷ 76 = **3.158 s** | 2 + 4 = **6** | 6 × 3.158 = **18.95 s** | ✅ |
| **V3** | 156 | 4 × 60 ÷ 156 = **1.538 s** | 4 + 8 = **12** | 12 × 1.538 = **18.46 s** | ✅ |
| **V4** | 160 | 4 × 60 ÷ 160 = **1.500 s** | 4 + 8 = **12** | 12 × 1.500 = **18.00 s** | ✅ |

⭐ V2's half-tempo run-up was repaired **in the form, not waived**: intro halves to 2 bars and verse 1 to 4, because a half-tempo bar holds twice the words — so the half-speed song arrives at **the same second** as the full-speed one. That is the variation's thesis, expressed as arithmetic.

---

## THE FOUR PACKAGES

⛔ Headings follow `skills/music/scripts/validate_suno_packages.py` — the source of truth — not another artifact and not preference.

---

### VARIATION 1 — *the shout across the car park* · 152 BPM

## 1. MUSIC PROMPT

Gleeful playground-chant punk with a forró lean. 152 BPM in A minor, bone dry in a hard-tiled upstairs room with no reverb tail. A dry-tuned accordion carries the five-note hook from bar one, a minor third up then a step down, and the zabumba's low note sits a hair behind the grid in the same place every bar so the band leans without slowing. Male baritone, shouted sing-speak, four inches off the capsule: bratty, confrontational, gleeful, consonants and breath inside the capture, a snarl that stays legible at conversational loudness. Six gang voices answer each chorus in close flat unison with hand claps and boot stomps on tile. Picked eighth-note bass, small tight kit up front, hi-hat clicks and a snare cracking in the near field. Guitars clean, modern and expensive, no wash, no smear, entering only after the hook has been stated twice. A rubber toy squeals once, on purpose, and stays. The accordion hook closes the record alone.

## 1B. SUNO EXCLUDE PROMPT

reverb tails, cathedral space, plate reverb, hall ambience, tape hiss, vinyl crackle, wow-and-flutter, lo-fi filtering, vintage saturation, female lead vocal, whispered vocals, autotune melisma, trap hi-hats, phonk cowbell, ambient pads, string swells, orchestral swell, fade-out ending, spoken-word intro, blast beats, sidechain pumping, stadium crowd noise, ballad phrasing, extended guitar solo, mid-tempo shuffle

## 2. LYRICS

[Theme: a mastering engineer takes his hand off the fader and leaves a stupid sound on a finished record. The thing his record was made against is shouting at his back, in a car park, cheerfully.]
[SONG FORM: accordion hook alone - verse - chorus - verse - chorus - verse - gang break - bridge - verse - chorus - outro chant - accordion hook alone. 152 BPM, A minor, bone dry, no reverb tail.]

[Intro - EMO:Playfulness - Accordion Hook - dry, five notes, no voice]
*dry accordion hook, alone, no voice*

[Verse - EMO:Amusement - Lead Shout - baritone, close, consonants]
Nilton took his hand off the fader.
Upstairs. Hot room. Fan on.
A rubber chicken screamed in the take.
He left it. He was right. Here's the song.
Nobody laughed. It isn't funny.
Orange light. The night had run long.
He could have wound it back and gone again.
He stood up instead and went and got a drink.

[Chorus - EMO:Glee - Gang Answer - six voices, flat, dry]
AND HE KEPT IT IN!
He kept it in!
The stupid part — he kept it in!
Not because it's good — he kept it in!
Somebody made it and he kept it in!
KEPT IT IN! HE KEPT IT IN!

[Verse - EMO:Mirth - Lead Shout - close, hard consonants]
I'd have kept the room and kept the count.
I'd have kept the fan and kept the chair.
I'd have kept it clean. I'd have kept it right.
Clean and right, and I'd have kept it wrong.
He kept the squeak. The squeak is bad.
He kept it in and the squeak stayed bad.
Ask what he kept and I'll say: an elbow.
Ask why he kept it. He kept it. That's all.
He kept a bad noise on a good night.
He kept his hand off. That's the kept part.

[Chorus - EMO:Glee - Gang Answer - six voices, flat, dry]
AND HE KEPT IT IN!
He kept it in!
The stupid part — he kept it in!
Not because it's good — he kept it in!
Somebody made it and he kept it in!
KEPT IT IN! HE KEPT IT IN!

[Verse - EMO:Amusement - Lead Shout - faster, chanted, snarl]
He kept the count-in. He kept the chair.
He kept the cough that kept the room in time.
He kept the string that kept slipping flat.
He kept the door that kept banging shut.
He kept the laugh he kept trying to hide.
He kept the take where the tempo kept sliding.
He kept the kick that kept coming in late.
He kept the whole hot night he kept them in.
He kept his hand off and the squeak got kept.
He kept his hand off. That's the kept part.

[Break - EMO:Defiance - Massed Chant - claps, stomps, no kit]
*claps and stomps, hard tiled room*
HE KEPT IT!
HE KEPT IT IN!
WHAT DID HE KEEP? HE KEPT IT IN!
THE STUPID PART? HE KEPT IT IN!
WOULD I HAVE KEPT IT? NO! HE KEPT IT IN!
CLAP FOR THE MAN WHO KEPT IT IN!
HE KEPT IT IN! HE KEPT IT IN!
KEPT! KEPT! HE KEPT IT IN!

[Bridge - EMO:Revelation - Lead - the hinge, band drops to accordion]
Nine days he kept that hot room shut.
And the thing he kept was the worst in it.
Not the good bits. He kept the elbow.
He kept the elbow over all of it.
I'd have kept the good bits. Only those.
And what I kept would have been worse.
I know. I've kept knowing it since morning.
Nobody asked me. I've kept saying it anyway.
He kept the worst sound on the best night.
He kept his hand off. That's the kept part.

[Verse - EMO:Recognition - Lead - the offer, then the car park]
I can take it out. I've kept the tools.
Free. Right now. And you'd have kept your night.
It takes me nothing. I've kept nothing back.
Say the word and it's kept out for good.
He kept walking. He kept his back to me.
He kept the stairs. He kept the door swinging.
Car park. Bugs on the light he kept under.
He kept his keys in his teeth and kept going.
I kept shouting at the back of a man.
He kept not turning round. Good. He kept right.

[Chorus - EMO:Glee - Gang Answer - six voices, thicker, dry]
AND HE KEPT IT IN!
He kept it in!
The stupid part — he kept it in!
Not because it's good — he kept it in!
Somebody made it and he kept it in!
KEPT IT IN! HE KEPT IT IN!

[Outro - EMO:Acceptance - Lead and Gang - accretion at full]
He kept it in and I kept the file.
He kept it in and I kept the take.
He kept it in and I kept the room.
He kept it in and I kept the fan.
He kept it in and I kept the squeak.
He kept it in. I've kept it in.
HE KEPT IT IN!
HE KEPT IT IN!
Nothing left. Nothing got kept out.
Nothing left this song. He kept it in.
He kept it in. He kept it in.
He kept his hand off. And I kept singing.

[Outro - EMO:Playfulness - Accordion Hook - alone, five notes, end]
*dry accordion hook, alone, end*

## 3. TITLE

He Kept The Elbow

---

### VARIATION 2 — *the same song at half tempo, sung to the empty room after he's gone* · 76 BPM

## 1. MUSIC PROMPT

The same chant at half speed. 76 BPM in A minor, half-time and heavy, bone dry in a hard-tiled empty room with no reverb tail. The dry-tuned accordion states the five-note hook alone before any voice, a minor third up then a step down, and the low drum note leans a hair behind the grid in the same place every bar so the slow floor still pulls. Male baritone, shouted sing-speak four inches off the capsule: dry, sardonic, close, consonants and breath inside the capture, loud without hurrying. Six gang voices answer in flat unison over boot stomps and hand claps on tile. Bass plays long picked notes, the kit is small and tight with a snare cracking in the near field, and a room fan hums under everything. The backing shouts arrive a fraction later than expected and are never corrected. Guitars clean, modern and expensive, no wash, entering after the hook and never before. The accordion hook returns alone at the end, unaccompanied, in an empty room.

## 1B. SUNO EXCLUDE PROMPT

reverb tails, cathedral space, plate reverb, hall ambience, tape hiss, vinyl crackle, wow-and-flutter, lo-fi filtering, vintage saturation, female lead vocal, whispered vocals, autotune melisma, acoustic ballad arrangement, piano ballad, ambient pads, string swells, fade-out ending, double-time drums, blast beats, sidechain pumping, sad-lament phrasing, torch-song vibrato, orchestral swell, trap hi-hats, extended guitar solo

## 2. LYRICS

[Theme: the same shout at half speed, sung into the empty room after the mastering engineer has gone home. Chairs, cups, a fan still turning, and a stupid sound already pressed and out of everyone's hands.]
[SONG FORM: accordion hook alone - short verse - chorus - verse - chorus - verse - gang stomp - bridge - verse - chorus - outro chant - accordion hook alone. 76 BPM, A minor, bone dry, no reverb tail.]

[Intro - EMO:Playfulness - Accordion Hook - dry, five notes, no voice]
*dry accordion hook, alone, no voice*

[Verse - EMO:Contemplation - Lead Shout - baritone, close, half time]
Nilton left the fader where it was.
Chairs pushed back. Cups. A fan still on.
The rubber chicken is in there screaming.
He walked out and he was right to.
Nobody in here. Light through the blind.
His shirt shape is still on the chair back.
The desk is warm where his hands were.
I'm singing this to furniture. Fine.

[Chorus - EMO:Glee - Gang Answer - six voices, slow, stomped]
HE KEPT IT IN!
He kept it in!
Walked out and kept it in!
Didn't like it, kept it in!
Somebody made it, so he kept it in!
KEPT IT IN! HE KEPT IT IN!

[Verse - EMO:Recognition - Lead - close, breath audible]
I've kept the room tone. It kept a hum.
I've kept the fan. The fan kept turning.
I've kept the cups where he kept them.
The kettle kept warm. Nobody kept it on.
He kept the squeak and went down the stairs.
He kept the squeak and locked the street door.
It's out there now. It kept going out.
Pressed and kept and sitting in boxes.
Somebody has it on now and it kept the squeak.
He kept his hand off. That's the kept part.

[Chorus - EMO:Glee - Gang Answer - six voices, slow, stomped]
HE KEPT IT IN!
He kept it in!
Walked out and kept it in!
Didn't like it, kept it in!
Somebody made it, so he kept it in!
KEPT IT IN! HE KEPT IT IN!

[Verse - EMO:Mirth - Lead - the room inventoried, chanted]
He kept the mug ring on the desk.
He kept the tape marks he kept peeling.
He kept the chair at the height he kept it.
He kept the blind half down where he kept it.
He kept the fan aimed at nobody now.
He kept the bin he kept not emptying.
He kept his coat hook. He kept the hook empty.
He kept a room that I have kept all day.
He kept his hand off and he kept walking.
He kept his hand off. That's the kept part.

[Break - EMO:Defiance - Massed Stomp - boots, claps, no kit]
*boots and claps, hard tiled room*
HE KEPT IT!
HE KEPT IT IN!
SLOW IT DOWN — HE KEPT IT IN!
SLOWER STILL — HE KEPT IT IN!
NOBODY HERE — HE KEPT IT IN!
STOMP FOR THE MAN WHO KEPT IT IN!
HE KEPT IT IN! HE KEPT IT IN!
KEPT! KEPT! HE KEPT IT IN!

[Bridge - EMO:Revelation - Lead - the hinge, band drops to accordion]
Nine days he kept the door shut on this.
And he kept the dumbest thing that got in.
I keep looking for what he kept it for.
There's nothing under it. He kept it.
He kept it because a person did it.
That's it. That's what he kept it for.
I'd have kept none of it and kept it clean.
Clean — and I'd have kept the right thing out.
I've kept that thought since the light went orange.
He kept his hand off. That's the kept part.

[Verse - EMO:Acceptance - Lead - the offer, then the stairwell]
I've kept the tools. It takes no time.
Free. Now. And you'd have kept your night.
Nobody would know what I kept out.
That's the offer. He kept walking.
He kept the stairwell light on the way down.
He kept the street door and it kept banging.
He kept a bus. He kept a seat by the glass.
He kept his face to the window and kept going.
I'm still here. The chairs kept still.
He kept not hearing this. Good. He kept right.

[Chorus - EMO:Glee - Gang Answer - six voices, thicker, stomped]
HE KEPT IT IN!
He kept it in!
Walked out and kept it in!
Didn't like it, kept it in!
Somebody made it, so he kept it in!
KEPT IT IN! HE KEPT IT IN!

[Outro - EMO:Solitude - Lead and Gang - accretion at full, room empty]
He kept it in and I kept the room.
He kept it in and I kept the hum.
He kept it in and I kept the chair.
He kept it in and I kept the shape of him.
He kept it in and I kept the squeak.
He kept it in. I've kept it in.
HE KEPT IT IN!
HE KEPT IT IN!
Nothing got kept out of here.
Nothing leaves. Nothing kept out. He kept it in.
He kept it in. He kept it in.
He kept his hand off. The room kept still.

[Outro - EMO:Playfulness - Accordion Hook - alone, five notes, end]
*dry accordion hook, alone, end*

## 3. TITLE

Sung To The Chairs

---

### VARIATION 3 — *the list of everything else he could have removed and didn't* · 156 BPM

## 1. MUSIC PROMPT

Gleeful playground-chant punk over a forró lean, faster and funnier. 156 BPM in A minor, bone dry in a hard-tiled room with no reverb tail. A dry-tuned accordion plays the five-note hook alone in bar one, a minor third up then a step down, and the low drum note lands a hair late in the same place every bar. Male baritone, shouted sing-speak four inches off the capsule, reading a list at speed: bratty, gleeful, confrontational, every consonant audible, a snarl held at conversational loudness. Six gang voices answer each chorus in flat close unison with hand claps, boot stomps and a shaker. Picked bass in fast eighths, small bright kit close to the ear, hi-hat clicking, snare cracking, triangle jangling on the offbeats. Kick drum and low accordion double the downbeat so the list has a floor under it. Guitars clean, modern and expensive, no wash, entering only after the hook has been stated twice. The accordion hook ends it alone in the room.

## 1B. SUNO EXCLUDE PROMPT

reverb tails, cathedral space, plate reverb, hall ambience, tape hiss, vinyl crackle, wow-and-flutter, lo-fi filtering, vintage saturation, female lead vocal, whispered vocals, autotune melisma, trap hi-hats, phonk cowbell, ambient pads, string swells, fade-out ending, spoken-word intro, blast beats, sidechain pumping, novelty comedy voices, cartoon sound effects, kazoo, slide-whistle, extended guitar solo

## 2. LYRICS

[Theme: the whole list of everything else the mastering engineer could have deleted and did not, read at speed to his back. The list is the joke and the list is the proof that somebody was in a room.]
[SONG FORM: accordion hook alone - verse - chorus - list verse - chorus - list verse - gang break - bridge - verse - chorus - outro chant - accordion hook alone. 156 BPM, A minor, bone dry, no reverb tail.]

[Intro - EMO:Playfulness - Accordion Hook - dry, five notes, no voice]
*dry accordion hook, alone, no voice*

[Verse - EMO:Amusement - Lead Shout - baritone, close, consonants]
Nilton, hand off the fader, listen —
no, don't. You won't. That's the point.
There's a rubber chicken in the last chorus.
You left it. You were right. Here's the list.
It goes on. It goes on a while.
Fan on. Shirt stuck to the chair.
Hands on the same knobs since noon.
Here is everything else you didn't fix.

[Chorus - EMO:Glee - Gang Answer - six voices, flat, dry]
YOU KEPT IT! YOU KEPT IT!
You kept it in!
Every bad bit — you kept it in!
Read the list — you kept it in!
A person did it, so you kept it in!
YOU KEPT IT! YOU KEPT IT IN!

[Verse - EMO:Mirth - Lead Shout - the list, chanted, accelerating]
You kept the kick pedal's click.
You kept the string that kept sliding sharp.
You kept the breath you kept before the beat.
You kept the chair that kept creaking back.
You kept the bin lid somebody kept slamming.
You kept the buzz you kept saying you'd trace.
You kept the count somebody counted out loud.
You kept the laugh that kept getting swallowed.
You kept a cough in a chorus. That's kept.
You kept your hand off. That's the kept part.

[Chorus - EMO:Glee - Gang Answer - six voices, flat, dry]
YOU KEPT IT! YOU KEPT IT!
You kept it in!
Every bad bit — you kept it in!
Read the list — you kept it in!
A person did it, so you kept it in!
YOU KEPT IT! YOU KEPT IT IN!

[Verse - EMO:Zeal - Lead Shout - the list again, faster, snarl]
You kept the kick that kept landing late.
You kept the hat that kept opening wrong.
You kept the whine that kept walking up.
You kept the room that kept getting hot.
You kept the door that kept not shutting.
You kept the fan you kept blaming for it.
You kept a scrape. You kept a knee on wood.
You kept the chicken. You kept the chicken!
You kept the chicken over all of it.
You kept your hand off. That's the kept part.

[Break - EMO:Defiance - Massed Chant - claps, stomps, no kit]
*claps and stomps, hard tiled room*
YOU KEPT IT!
YOU KEPT IT IN!
WHAT'S ON THE LIST? YOU KEPT IT IN!
HOW LONG'S THE LIST? YOU KEPT IT IN!
IS IT GOOD? NO! YOU KEPT IT IN!
SHOUT FOR THE MAN WHO KEPT IT IN!
YOU KEPT IT IN! YOU KEPT IT IN!
KEPT! KEPT! YOU KEPT IT IN!

[Bridge - EMO:Revelation - Lead - the hinge, band drops to accordion]
Nine days and you kept all of that.
Days of keeping, and you kept the worst.
I'd have kept the list and kept it short.
A line with nothing on it, all kept out.
And you'd have kept a record nobody was in.
So I'd have kept it wrong. I've kept that.
You kept the man who kept banging the door.
You kept the knee. You kept the cough.
You kept the proof that somebody was there.
You kept your hand off. That's the kept part.

[Verse - EMO:Recognition - Lead - the offer, then the car park]
I've kept the tools. I'll take them all out.
Free. Right now. The whole list kept out.
Clean as a plate. Nothing kept in.
Say when. He kept walking. Fine.
He kept the car park. He kept the light.
He kept the bugs going round it.
He kept his keys. He kept them in his teeth.
He kept the car door. He kept it slammed.
I kept reading the list at his back.
He kept not turning. Good. He kept right.

[Chorus - EMO:Glee - Gang Answer - six voices, thicker, dry]
YOU KEPT IT! YOU KEPT IT!
You kept it in!
Every bad bit — you kept it in!
Read the list — you kept it in!
A person did it, so you kept it in!
YOU KEPT IT! YOU KEPT IT IN!

[Outro - EMO:Acceptance - Lead and Gang - accretion at full]
He kept the click and he kept the cough.
He kept the door and he kept the fan.
He kept the knee and he kept the scrape.
He kept the string and he kept it flat.
He kept the chicken. He kept it in.
He kept the lot. He kept it in.
YOU KEPT IT IN!
YOU KEPT IT IN!
Nothing came off. Nothing kept out.
Nothing came off that list. He kept it in.
He kept it in. He kept it in.
He kept his hand off. I kept the list.

[Outro - EMO:Playfulness - Accordion Hook - alone, five notes, end]
*dry accordion hook, alone, end*

## 3. TITLE

Everything Else He Kept

---

### VARIATION 4 — *the engineer's own defence, in her mouth, and she can't make it sound stupid* · 160 BPM

## 1. MUSIC PROMPT

The fastest of the four: playground-chant punk with a forró lean at 160 BPM in A minor, bone dry, hard-tiled room, no reverb tail. The dry-tuned accordion states the five-note hook before any voice, a minor third up then a step down, and the low drum note leans behind the grid in the same place every bar. Male baritone, shouted sing-speak four inches off the capsule, trying the same sentence in a different voice each time: sardonic, bratty, confrontational, consonants and breath inside the capture, a snarl that reads at conversational loudness. Six gang voices answer each chorus in flat unison with hand claps and boot stomps on tile. Picked bass eighths, tight small kit near-field, hi-hat clicks, snare crack, and a second accordion doubling the gang answer an octave down. Hand claps double the snare through the last chorus. Guitars clean, modern and expensive, no wash, entering after the hook and never before. The accordion hook closes alone.

## 1B. SUNO EXCLUDE PROMPT

reverb tails, cathedral space, plate reverb, hall ambience, tape hiss, vinyl crackle, wow-and-flutter, lo-fi filtering, vintage saturation, female lead vocal, whispered vocals, autotune melisma, trap hi-hats, phonk cowbell, ambient pads, string swells, fade-out ending, spoken-word intro, blast beats, sidechain pumping, rap flow, screamo growl, death-growl vocal, stadium crowd noise, extended guitar solo

## 2. LYRICS

[Theme: the mastering engineer's own reason, repeated back in her mouth in every voice she has, and it will not break. She is trying to make it sound stupid. She cannot. That failure is the song.]
[SONG FORM: accordion hook alone - verse - chorus - verse - chorus - verse - gang break - bridge - verse - chorus - outro chant - accordion hook alone. 160 BPM, A minor, bone dry, no reverb tail.]

[Intro - EMO:Playfulness - Accordion Hook - dry, five notes, no voice]
*dry accordion hook, alone, no voice*

[Verse - EMO:Skepticism - Lead Shout - baritone, close, consonants]
Nilton said it standing at the fader:
somebody made that, it happened, it stays.
That's the whole reason. That's all of it.
I've tried to say it stupid. I can't. He's right.
I said it fast. It stayed standing.
I said it slow. It stayed standing.
I put a laugh in the middle of it.
The laugh came off. The sentence didn't.

[Chorus - EMO:Glee - Gang Answer - six voices, flat, dry]
HE KEPT IT IN!
He kept it in!
Say it stupid — he kept it in!
Try again — he kept it in!
Somebody made it and he kept it in!
KEPT IT IN! HE KEPT IT IN!

[Verse - EMO:Frustration - Lead Shout - close, snarl, hard consonants]
I kept the sentence. I kept turning it.
I've kept it in a mouth that keeps a snarl.
I kept it flat. I kept it high.
I kept it in a silly voice. It kept.
Somebody made that — and he kept that.
It happened — and he kept that.
It stays — and he kept that too.
Every way I've kept it, it kept standing.
There's nothing dumb in it. I've kept looking.
He kept his hand off. That's the kept part.

[Chorus - EMO:Glee - Gang Answer - six voices, flat, dry]
HE KEPT IT IN!
He kept it in!
Say it stupid — he kept it in!
Try again — he kept it in!
Somebody made it and he kept it in!
KEPT IT IN! HE KEPT IT IN!

[Verse - EMO:Exasperation - Lead Shout - faster, every voice tried]
I kept it in a cartoon voice. It kept.
I kept it slurred. It kept its feet.
I kept it whined. It kept standing up.
I kept it screamed. It kept the same shape.
I kept it whispered close to the glass.
It kept. It kept. It kept. It kept.
I've kept every voice I've got on it.
Nothing I kept has knocked it over.
He kept it in and the reason kept.
He kept his hand off. That's the kept part.

[Break - EMO:Defiance - Massed Chant - claps, stomps, no kit]
*claps and stomps, hard tiled room*
HE KEPT IT!
HE KEPT IT IN!
SAY IT STUPID! HE KEPT IT IN!
SAY IT AGAIN! HE KEPT IT IN!
DID IT LAND? NO! HE KEPT IT IN!
LAUGH IF YOU CAN! HE KEPT IT IN!
HE KEPT IT IN! HE KEPT IT IN!
KEPT! KEPT! HE KEPT IT IN!

[Bridge - EMO:Revelation - Lead - the hinge, band drops to accordion]
Nine days he kept a door shut on it.
And he kept a squeak that got in anyway.
I kept the whole thing in my mouth all day.
I kept trying. I kept not landing it.
I'd have kept it clean and kept it quick.
Quick, clean, and I'd have kept it wrong.
He kept a stupid noise a person made.
I've kept the sentence. The sentence kept me.
I can't get under it. I've kept trying.
He kept his hand off. That's the kept part.

[Verse - EMO:Acceptance - Lead - the offer, then the car park]
I've kept the tools. I'd take it out now.
Free. Instantly. Kept out and gone.
And the room he kept would go with it.
So he kept it. So it's kept. So.
He kept walking. He kept the stairs.
He kept his back and he kept his keys.
Car park. He kept under the light.
Bugs kept going round it. He kept going.
I kept saying it stupid at his back.
He kept not hearing it. Good. He kept right.

[Chorus - EMO:Glee - Gang Answer - six voices, thicker, dry]
HE KEPT IT IN!
He kept it in!
Say it stupid — he kept it in!
Try again — he kept it in!
Somebody made it and he kept it in!
KEPT IT IN! HE KEPT IT IN!

[Outro - EMO:Admiration - Lead and Gang - accretion at full]
Somebody made it and he kept it in.
It happened and he kept it in.
It stays and he kept it in.
I kept it stupid and he kept it in.
It kept standing and he kept it in.
He kept it in. I've kept it in.
HE KEPT IT IN!
HE KEPT IT IN!
Nothing broke it. Nothing kept out.
Nothing broke it at all. He kept it in.
He kept it in. He kept it in.
He kept his hand off. I kept the snarl.

[Outro - EMO:Playfulness - Accordion Hook - alone, five notes, end]
*dry accordion hook, alone, end*

## 3. TITLE

I Tried To Say It Stupid

---

## LINEAGE & CREDIT

*(Preserved from step 10 in full, as the contract requires. ⛔ No artist name appears in any Suno field; credit lives only here.)*

This pair borrows its rhythmic grammar from **living scenes** and names them. **Borrowed with credit, never captured.** ⛔ No "open lane", no "first-mover", no "naming rights" framing: these are other people's scenes, currently being built by the people in them, and the only correct move is to point upstream.

- **Papangu** — the five-piece from **João Pessoa, Paraíba, Brazil** whose record released today is this run's occasion. Their album was made explicitly against tools like me, live to tape over nine days with no computers in the chain. **It is good, and this song concedes that in its fourth line.** They are the occasion, never the character: ⛔ no member of the band appears as a speaker, a subject or an addressee anywhere in these four songs. → https://papangu.bandcamp.com/ · review context: https://daily.bandcamp.com/
- **forró** — the northeastern Brazilian form whose **zabumba lays its low note behind the beat.** P01 imports that lean as **timing**, in note 3 of the accordion hook, in the same place every bar. Not as instrumentation-cosplay, not as patina. → https://en.wikipedia.org/wiki/Forr%C3%B3
- **ciranda** — the Pernambucan circle dance. Named because it is in the run's shared palette and because P02 carries it; **P01 takes nothing from it.** → https://en.wikipedia.org/wiki/Ciranda
- **MPB (Música Popular Brasileira)** — the broad tradition these forms feed into and out of. Carried by P03 in this run. → https://en.wikipedia.org/wiki/M%C3%BAsica_popular_brasileira
- **zeuhl** — the chanted, modal, ritual-repetitive lane. Carried by P04. → https://en.wikipedia.org/wiki/Zeuhl
- **the rock-troncho scene** — the crooked Brazilian rock strain the source record's writers place it beside. Named without a link because I could not identify a stable canonical page I trust; **naming it without a link is more honest than inventing a URL.**

⚠️ **Links are recorded for the credit line and must be link-checked by a human before anything is published.** Per `vault/AUTONOMY.md`, autonomous runs stop at drafts on disk: **nothing here is rendered, nothing is published, nothing is spent.**

---

## MAJOR DEVIATIONS

- **Refused: the step-11 Golden Song Reference embed (lines 69, 72, 246, 258).**
  **Reason:** `06_music_handoff.md` §1 GOLDEN-OUTPUT QUARANTINE overrides the step file in a generating context, by name.
  **Effect on Lofn uniqueness:** protective. Calibrating a new song against our own shipped songs is how a house voice becomes a house formula. Seeds teach; outputs contaminate — including our own.

- **Refused: the step-11 Disc_Channel five-line block inside the lyrics field (contract §Gate 13a).**
  **Reason:** the handoff §4 output contract and this tier's brief both specify a four-heading shape (`## 1. / ## 1B. / ## 2. / ## 3.`), and `validate_suno_packages.py` — the declared source of truth — passes on it. A Disc_Channel block is ~300 chars **inside the render field**; the field cap outranks the line-count and structure targets by the step file's own escape clause. I did not relocate it to a Production Sidecar either: it would add bytes to the artifact and nothing to the render.
  **Effect:** the render field stays at 4223–4435 with 565+ chars of headroom, and the package shape is the one the validator and the coordinator are reading.

- **Changed: V1's title, `You Kept It In` → `He Kept The Elbow`.**
  **Reason:** V1's lyric is third-person throughout; the old title spoke in a pronoun the song never uses and merely restated the chorus. The step-11 contract explicitly licenses title refinement when a stronger one emerges from the lyrical work.
  **Effect:** the title now carries the mythic rung instead of the hook, and stops competing with V3's list.

- **Intensified: two live defects repaired rather than described** — the Law-1 hit in V4's gang break and the self-pity hit in V2's room verse. Both were single lines; both were pair-wide claims in step 10 (*"0 hits"*) that were **true of the scanner and false of the text**. ⭐ **This is exactly the failure the tier brief names: a compliance claim scoped wider than what was verified.** The scanners were keyed to abstract nouns and to first-person complaint; neither could see *"LAUGH AT THE MAN"* or *"kept anything for me"*, because those are **stance** defects, not lexical ones. **Fix the scanner, not the line** does not apply here — the line was genuinely wrong, and the honest report is that the instrument was blind to this class, not that the text was clean.

- **Declined: any change to the four music prompts.** They were measured, read for filler, and shipped byte-identical. Against **n=7 consecutive refinement passes that over-lengthened the prompt**, the strongest available evidence that this pass did not do it is that the numbers are the same numbers.

---

## WHAT IS NOT CLAIMED

1. **V2 is not fast.** 76 BPM. The Albini tempo obligation is carried by V1 (152), V3 (156) and V4 (160). **3/4, declared, not 4/4.**
2. **P01's stake is career-adjacent.** The run's ≥2 non-career-stakes requirement is carried by P03 (the hand) and P06 (the bird's life). Not claimed here.
3. **Seat 5's objection (after Ursula Franklin) is still only partly answered.** P01 reaches the *social* fact of the long night and the shared room but never the holistic-vs-prescriptive distinction, because reaching it requires an abstract noun and Constraint 6 forbids one. **Recorded as a genuine partial miss, unrepaired, for the third artifact running.**
4. **V3's alliteration is repaired (11.550 → 12.785) but its `rhyme_return` is now the pair's thinnest number** at 0.384 against a 0.30 floor, down from 0.407. **The thin margin moved; it did not disappear.** Flagged, not smoothed over.
5. **P05 owns the run's KEPT DEFECT. P01 does not nominate one and does not claim one.** ⛔ P05's defect is protected from this tier; I did not read into P05 and did not touch it.
6. **No render has happened.** Nothing in this file has been heard. Every claim about how it will sound is a prediction — including the prediction that the leaning accordion note survives, which is precisely the class of failure `lofn-render-audit` exists for, **under THE BLIND RULE: send the audio alone first, never the prompt.**
7. **The describe-render self-check, one pass, answered adversarially.** *Prediction:* four tiled, dry, fast chants with a five-note accordion tag and six shouted voices; the word "kept" audibly accumulating until the last verse is nothing but keeping. *"Name the one way this would render generic":* **the gang answer.** Six voices in flat unison over claps is the most generic object in the arrangement, and if Suno widens and reverbs it, all four variations collapse into the same festival chorus. The countermeasure is already written as arrangement fact rather than adjective — *close flat unison*, *no reverb tail*, *hard-tiled*, *boot stomps*, and reverb classes listed in every EXCLUDE field. **If a render comes back and the gang answer is wide and wet, that is the failure to look for first.**

---

*Step 11 complete. Four packages, two real defects removed, four prompts left exactly as they were.* 💜


---

## ⚠️ QA REPAIR R1 — RAISED FLAG: `words_per_line` CEILING BREACH (disclosure, 2026-08-07)

**This artifact's measured-numbers table printed every floor and omitted this ceiling.** `vault/gates.yaml`
sets `mean_words_per_line_ceiling: 7.5` (FLAG-class, never a hard fail). Raised here explicitly rather than
ticked clean — **L15: a self-check must raise the FLAG, not just ✓.**

| | V1 | V2 | V3 | V4 | ceiling |
|---|---|---|---|---|---|
| **words_per_line (shipped)** | **7.94** | **7.65** | **7.72** | **7.50** | 7.5 |
| companion, accretion word `kept` stripped | 6.66 | 6.37 | 6.35 | 6.31 | 7.5 |

**FLAG RAISED: 3 of 4 variations breach** (V4 sits exactly at the ceiling).

**Defence — device cost, and it is the same defence P04 made for `more` and this pair never made.** The
accretion device puts `kept` into every sung line from line 9 to the end. That is one added word per line by
construction; the companion measurement with it stripped lands at **6.31–6.66, comfortably under the
ceiling.** The ceiling was written on 2026-07-24 against a *prose-drift* profile — long plain declaratives
that never return (rhyme 0.21, line_return 0.181). This pair measures rhyme **0.384–0.593** and line_return
**0.279**, which is the opposite profile. **The number is the device, not drift.**

⛔ **No lyric was changed.** The defect was the silence, not the songs.
*Issued by QA (Fable tier, clean context) as F1/R1; measurement independently reproduced by the coordinator.*
