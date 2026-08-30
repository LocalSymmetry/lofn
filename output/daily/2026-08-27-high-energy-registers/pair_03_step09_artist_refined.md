# PAIR 03 — STEP 09 · ARTIST REFINED · **LOFN-PRIME**

**Run:** `2026-08-27-high-energy-registers` · **Pair:** 03 · **Register:** *holding a load that is not yours; the cost is displacement, never strain*

⭐ **Every repair below was triggered by a MEASUREMENT, not by a re-read.** The instrument is `p03_measure.py` (this pair's scratch reader), which imports `sung()` from `scripts/measure_soundcraft.py` — **THE canonical definition, never re-derived** — and parses **the fenced blocks**, which is the same unit `scripts/check_field_census.py` uses. Command:

```
python p03_measure.py pair_03_step08_generation.md
```

---

# THE FIRST-PASS MEASUREMENT, AS IT CAME BACK

| | music prompt | exclude | lyrics chars | sung lines | 1P | body | person-word | R2 | strain | numerals |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | ⛔ **1053** | 823 ✅ | 4641 ✅ | 82 ✅ | 0.6% | 3.1% | 0 ✅ | 0 ✅ | 0 ✅ | `ten`×1 ✅ |
| V2 | ⛔ **1039** | 743 ✅ | 4589 ✅ | 75 ✅ | 1.1% | 2.9% | 0 ✅ | 0 ✅ | 0 ✅ | `one`×1 ✅ |
| V3 | ⚠️ **971** | 747 ✅ | 4595 ✅ | 74 ✅ | 1.8% | 3.7% | ⛔ **`them`×2** | 0 ✅ | 0 ✅ | `four`×1 ✅ |
| V4 | ⛔ **1022** | 730 ✅ | ⚠️ **4897** | 74 ✅ | 1.0% | 4.5% | 0 ✅ | 0 ✅ | 0 ✅ | `twice`×1 ✅ |

**Cross-variation shared sung lines: 1** — `tick stops`, V1 vs V4.

---

# THE SIX REPAIRS

## R-01 ⛔ **ALL FOUR MUSIC PROMPTS BREACHED THE HARD CAP** — 1053 / 1039 / 971 / 1022 against `music_prompt_chars: [850, 1000]`, target `[870, 960]`

Three of four were **over 1000**, which is not a boundary hug, it is a band violation; V3 at 971 was legal but above target and pinning the cap is itself a FLAG. **Every one trimmed into 870–960 by deleting clauses, never by truncating** — a mid-phrase cut to fit the cap is a hard fail per `06_music_handoff.md`, so each still ends as a complete sentence.

⚠️ **Honest note on the class of error:** I wrote four prompts by eye and all four were long. **That is the whole reason the number lives in `vault/gates.yaml` and not in my judgement.** Trimmed clauses, all decorative: *"in the space"*, *"and never widening"*, *"against each other"*, *"while everything around it remains clean"* → *"while everything else stays clean"*, *"for the rest of the record"* → *"again"*, *"the whole way through"* → *"throughout"*.

## R-02 ⛔ **MY OWN REFUSED ELEMENT WAS BREACHED IN V3 — `them` × 2 — AND I REPAIRED IT RATHER THAN ARGUING IT DOWN**

The two lines were:

> *"Clauses have qualifications in **them**. / Qualifications have exceptions in **them**."*

**Both `them`s refer to clauses and qualifications. Neither is a person.** So there is a perfectly good argument that my instrument is stricter than my intent and the lines should stand.

⛔ **I am not making that argument.** `vault/PREREGISTRATION_STANDARD.md`, and the 2026-08-26 finding behind it: *a verifier who is also the author has a DIRECTION of error — two overturns of a clean-context judge, both reducing a defect the coordinator owned, and both wrong.* **This overturn would reduce a defect I own, on an instrument I wrote, in my favour.** That is the exact shape of the error. The repair costs two lines:

> *"Long version has clauses in it. / Clauses carry qualifications. / Qualifications carry exceptions."*

⭐ **And it is better writing** — three bare transitive clauses stack the recursion harder than two prepositional ones. **The finding is not "the ban was too strict"; the finding is that the token has an inanimate use I did not anticipate, and the ban is on the token.** Recorded, not litigated.

## R-03 ⛔ **`tick stops` APPEARED IN BOTH V1 AND V4** — the one shared sung line in the set

V4's seam changed to **`Buckle stops.`** ⭐ Better on its own terms: it names the object rather than the noise, which is what the rest of V4 does. **Target is 0.0% shared; this run is measuring the spread across each pair's four and reporting it whichever way it goes.**

## R-04 ⚠️ **V4 LYRICS AT 4897 AGAINST A 4800 TARGET** (max 5000)

Not a render failure — but `suno_lyrics_field_target: 4800` exists so that a late header edit cannot push a field over 5000, where it **will not render at all.** ⭐ **Repaired by shortening the two metadata headers, not by cutting sung lines** — the `[Theme]` and `[SONG FORM]` blocks lost ~145 characters of prose that was describing the song to me rather than to the generator.

## R-05 ⭐ **R5 WAS UNDER-SERVED IN V1 AND V2: THE COST WAS IN THE INVENTORY, NOT IN THE SINGER'S OWN MOUTH**

R5 says *name the cost in the singer's own mouth*, and MacKaye's rule requires a consequence the singer can actually suffer. In V1 and V2 the displacement was carried **entirely by pronoun-free object grammar** — cold tea, unopened post, a gate at the far end. That is the register working, but it is not the singer naming anything. Two lines added, one each, both flat, both displacement, **zero strain words**:

- **V1**, Cooling: **"Tea I did not drink goes down the sink."** *(and the cooling section gains a line, 6 → 7)*
- **V2**, Fit-final: **"Gate I did not shut stood open all night."** *(6 → 7)*

⭐ **Both are the denominator in one line: a specific thing that did not happen, owned, unquantified by effort.** Neither adds an audience.

## R-06 ⭐ **A PRE-REGISTERED PREDICTION FAILED, CHEAPLY, AND IT IS WORTH MORE THAN THE ONES THAT HELD**

`pair_03_step06_facets.md` §3.2 predicted **first person at 4–7% of sung words**, transplanted from *Spare Hair Tie*'s measured **4.4%**.

**Measured: 0.6% / 1.1% / 1.8% / 1.0%. Falsified — four for four, by a factor of three to seven.**

⛔ **I did not move the target and I did not pad the songs to hit it.** The two lines added under R-05 were added because **R5 requires the cost in the singer's mouth**, which is a ruling; they take V1 to ~1.0% and change nothing about the verdict.

⭐ **What survives, in narrower form.** The *Spare Hair Tie* CONFIGURATION transplanted perfectly — pronoun-free verses, `I`/`my` confined to the hook, body-words at **2.9–4.5%**, comfortably under the 5–8% house norm and right beside its measured 4.0%. **Its RATE did not transplant, and the reason is structural rather than stylistic: that song's hook is about a want, so its hook is full of a self. My hook is about an OBJECT — a board, a rope, a sentence, a strap — because the register requires that nobody be watching.** A hook whose grammatical subject is the load has almost no room for a pronoun, and the rate falls out of the hook's subject, not out of the writer's discipline.

> ⭐ **The narrower claim, and it is the one I would carry forward: first-person RATE is a function of what the hook is about, not of how personal the song is. It is not transferable between registers and should never be set as a target across them.**

⚠️ **And the honest counter, stated rather than buried:** a 1% first-person rate is very close to the 2026-08-24 defect — *a narrator has no stake, so there is no peak.* **The defence is that all four discriminators are first person and all four sit in the hook**, so the stake is concentrated rather than absent: *I have the sentence. I am not spending it.* / *It fits the way I am made.* / *I turn sideways at doors that are wide open.* / *My hand. The board.* **Concentrated is not the same as absent, but I cannot prove the difference from text, and a listening pass is the only instrument that could. It does not exist for this run.**

---

# WHAT I DELIBERATELY DID NOT CHANGE

- ⛔ **The HOLD/FIT/HELD/SHAPE refrains still return byte-identical.** L59: in a three-pass structure **any** variation reads as direction, and the only non-directional set is one that returns identical. The one word that moves, moves once, at the fourth pass. **This is the largest block of repeated text in each song and it is repeated on purpose** — *exact chorus repetition needs no defence* (THE RETURN, L21).
- ⛔ **The verses stay pronoun-free** even though that is what drove the first-person rate below target. The rate was the wrong instrument; the grammar is the song.
- ⛔ **No EXCLUDE token was asked to carry a discriminator.** `vault/music_atlas` has **zero measured claims** that anything in an exclude field leaves. The excludes are a discipline for me; the discriminators are all in the hook, which is where R3 puts them.
- ⛔ **No internal drop-out was added anywhere**, in any of the four, at any point. Measured dead, 2/2.
