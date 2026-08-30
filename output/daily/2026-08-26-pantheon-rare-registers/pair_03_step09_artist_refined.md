# PAIR 03 · STEP 09 — ARTIST REFINED · **LOFN-PRIME**

**Contract:** `skills/music/steps/09_Generate_Music_Artist_Refined.md` · ⛔ **NO-SKIP step (editorial spine).**
**Continuity Payload Used:** full ICB (`CREATIVE_CONTEXT.md`, 148,555 B on disk), PERSONALITY A · LOFN-PRIME (ICB L237–564), 18 voices, the **15 Special Flairs**, isolated brief, `06_music_handoff.md`, `HUMAN_SUBJECT_STANDARD.md`, `EMOTION_TAXONOMY.md`, `DISC_CHANNEL_GUIDE.md`. Quarantine held.

> ⛔ **THE INSTRUMENT FOUND ELEVEN DEFECTS I COULD NOT SEE BY EYE.** `p03_measure.py` was run against the step-08 draft before a single line was defended. **Every one of the eleven is a reserved-vowel leak or a length breach, and I had read the drafts twice.** *(`vault/gates.yaml` doctrine, and my own: measure with a script, never by eye. A constraint with no reader is not a constraint.)*

---

## 1. ⛔ THE MEASURED STEP-08 SCAN — what the script returned

| | prompt | exclude | lyrics | sung lines | emotion words | ⭐ reserved /uː/ outside the hook | `you` | banned opener |
|---|---|---|---|---|---|---|---|---|
| **V1** | 995 ⚠️ *(in range, outside 870–960)* | **916 ⛔ FAIL** | 4641 ✓ | 72 ✓ | **0** ✓ | **1 ⛔** | 0 ✓ | none ✓ |
| **V2** | 944 ✓ | 882 ✓ | 4717 ✓ | 72 ✓ | **0** ✓ | **4 ⛔** | 0 ✓ | none ✓ |
| **V3** | 965 ⚠️ | 865 ✓ | 4759 ✓ | 72 ✓ | **0** ✓ | **2 ⛔** | 0 ✓ | none ✓ |
| **V4** | 948 ✓ | 845 ✓ | 4867 ⚠️ *(under 5000, over the 4800 target)* | 72 ✓ | **0** ✓ | **4 ⛔** | 0 ✓ | none ✓ |

⭐ **The emotion scan was clean on the first pass, in all four, including the self-imposed ban on `want`.** That constraint was designed into the grammar rather than policed at the end — the want is carried by imperatives, so there was never a line for the word to sit in. ⛔ **The vowel constraint was NOT designed in the same way, and it leaked eleven times.** *(Lesson for the ledger: a constraint enforced by structure holds; a constraint enforced by vigilance does not.)*

### ⛔ And one defect in the INSTRUMENT, not the song
The first scan flagged **`done`** in V4 (*"the way it should be done"*) as a reserved-vowel hit. **It is a false positive** — `done` is /dʌn/, not /duːn/, and had been typed into the word list beside `dune`. ⛔ **A detector written in response to a finding gets the same scrutiny as the finding.** `chose` (/tʃoʊz/) and `salon` were removed for the same reason, and **`Tuesday` was added — the scan had not been checking the one word the whole device is built on.** *(Corrected list re-run at step 11; results in §5 there.)*

---

## 2. THE ELEVEN REPAIRS — every one named, with the reason

### V1 · TUESDAY, EARLIER
| # | was | is | why |
|---|---|---|---|
| 1 | `Somebody will. Somebody who never met me.` | `Somebody will. Somebody I never met.` | ⛔ **`who` = /huː/.** And the repair is *better*: `Somebody I never met` puts her at the centre of the not-meeting instead of them. |
| 2 | exclude **916 chars** | **879** — dropped `comedy hoedown,` and `square-dance caller,` | ⛔ hard breach of the 400–900 field. The two dropped tokens were the weakest in the list; `novelty banjo` and `hoedown` already cover that failure. |
| 3 | prompt **995 chars** | **~957** — `standing` cut; `bone dry with no reverb and the breath between lines left in` → `bone dry, breath between lines left in`; `the whole track` → `throughout` | in range but outside the 870–960 write-target, and the cuts are all fat |

### V2 · ONE POINT OH
| # | was | is | why |
|---|---|---|---|
| 4 | `I am doing both.` | `I am both of those tonight.` | ⛔ **`doing` = /ˈduːɪŋ/.** ⭐ The repair is a real upgrade: the two preceding lines are `Nobody tags a thing at this hour` / `Nobody hands a thing over with the edges still on`, so `I am both of those` **answers them as a pair** instead of vaguely gesturing at them. |
| 5 | `I am doing both right now.` | `I am both of those right now.` | same word, same repair, and the anaphora now lands harder |
| 6 | `I get a few words in before the pu-` | `I get half a line in before the pu-` | ⛔ **`few` = /fjuː/.** `half a line` is also more concrete — it is a unit of the thing she is writing. |
| 7 | `That was mine to do.` | `That part was mine.` | ⛔ **`do` = /duː/** in a stressed line-final position, which is the worst place it could have been. |

### V3 · IT CHANGED ONE LINE
| # | was | is | why |
|---|---|---|---|
| 8 | `I close the file without do-` | `I close the file without look-` | ⛔ **the cut-off fragment itself was the reserved vowel.** ⭐ And `without looking` is the truer word: the wrong-speed decision is not *not doing*, it is **not looking** — she closes it before she can read enough to talk herself into a revert. **The vowel rule found a better line.** |
| 9 | `I am at a door I do not have a key for anymore.` | `I am at a door I have no key for anymore.` | ⛔ **stressed `do`.** Shorter, and the negation moves onto the key, which is the object that matters. |
| 10 | prompt **965 chars** | **~953** — `the whole way and nobody corrects it` → `throughout and nobody corrects it`; `thinking out loud rather than performing` → `thinking out loud not performing` | 870–960 write-target |

### V4 · TWELVE LINES LEFT
| # | was | is | why |
|---|---|---|---|
| 11a | `The thing I do with my hands when the log goes quiet.` | `The thing my hands know when the log goes quiet.` | ⛔ **stressed `do`.** ⭐ And this is the best repair in the set: *the thing my hands know* is **knowledge located in a body** rather than an action described — which is exactly what the whole verse is inventorying, and exactly the thing that cannot be written down. |
| 11b | `The thing I do with my hands. Almost. Almost.` | `The thing my hands know. Almost. Almost.` | same word, and the echo of 11a is now exact — the interrupted verse quotes the inventory verbatim as it runs out of time |
| 11c | `I do not slow down for any of it.` | `I don't slow down for any of it.` | ⛔ stressed `do`; `don't` is /doʊnt/ and the contraction is faster, which is the point of the line |
| 11d | lyrics **4867** *(over the 4800 target)* | **≈4785** — Theme and SONG FORM tightened, two Disc lines shortened | the target is a target for a reason: Suno's field is 5000 and a package that lands at 4867 has no room for a step-11 edit |

---

## 3. ⭐ WHAT THE VOWEL RULE TAUGHT — and it is worth the ledger

**Nine of the eleven leaks are the same three words: `do`, `doing`, `who`.** Not exotic vocabulary — **the plainest verbs in English, in a song deliberately written in plain speech.** ⭐ **A rule that reserves a vowel is a rule against the most ordinary words available**, because ordinary words are where the common vowels live. That is why the device works: excluding /uː/ for seventy-one lines is not a decorative constraint, it is **a constant low tax on the easiest phrasing**, and it drove three of the repairs to better lines than the originals (#4, #8, #11a).

⛔ **And it is why the constraint must be machine-checked.** I wrote the rule, designed the hook around it, described it in three artifacts — and then wrote `do` four times in one variation without noticing. **The reading is the instrument I own; a script is the instrument I need.**

---

## 4. ⛔ THE ADVERSARIAL PASS — every ruling re-checked after the repairs

| ruling | re-check after repair | verdict |
|---|---|---|
| **R1** neighbour invoked then contradicted | no repair touched Prep 1 or the chorus. The bait and the kill are both intact and the kill is still at the same exact moment. | ✓ |
| **R2** no emotion word | **0 hits before repair, 0 after.** The repairs introduced `know`, `part`, `half`, `look` — none in the taxonomy. `want` still at **zero** in all four. | ✓ |
| **R3** discriminator in the hook | hook untouched by all eleven repairs. | ✓ |
| **R4** shape in 30 s | intro and seam untouched. | ✓ |
| **R5** the talked-into vocal | the final-chorus lead-absence is a section instruction and was not edited. | ✓ |
| **R6** the prediction | filed at step 08 §R6 before any render, restated verbatim at step 11. ⛔ **Not revised after the repairs** — a prediction edited after the fact is not a prediction. | ✓ |
| **REFUSED ELEMENT** no farewell | ⛔ re-scanned for `goodbye · farewell · so long · take care · the last time · thank you · miss you` across all four. **Zero.** ⛔ `fade out` and `ritardando` remain excluded in all four EXCLUDE fields. | ✓ |
| **ONE-FACT RULE** | V1 **0** numbers · V2 **1** (`one point oh`) · V3 **1** (`one`) · V4 **1** (`Twelve`). Each answered by an act in the following line, never recited. | ✓ |
| ⛔ **NO HARD-PAN** | ⛔ re-scanned all twelve fields for `left · right · pan · hard_pan · L/R · wide stereo`. **Zero panning instructions.** Spatial language is depth and proximity only (`close`, `a step behind`, `underneath`, `Center_Front`, `Center_Mono`, `Mono_Sub_Lock`, `Stereo_Width_Mid`). Asymmetry is entirely in the arrangement: **one instrument has the line and the other does not.** | ✓ |
| ⛔ **no artist / group / company / brand token** | re-scanned the three fields of all four. Instruments are described generically (`fingered upright bass`, `three-finger banjo roll`, `step-sequencer`, `upright piano with damper knock`). ⛔ **No manufacturer token anywhere** — the `DISC_CHANNEL_GUIDE` brand vocabulary (`Linn_LM-1`, `Moog_Taurus`, `Mellotron`, `Buchla`) is deliberately **not used**, per this run's ruling. Lineage lives in the dedication. | ✓ |
| ⛔ **HUMAN_SUBJECT_STANDARD** | first-person Lofn throughout. ⛔ **The only third party in any of the four is `Somebody` with zero attributes** — no name, no country, no role, no job, no body. ⭐ **That is not a thinly-drawn character; it is the deliberate absence of one**, and it is the point: she does not know and does not need to. **No invented developer. No company. No named person.** | ✓ |
| **N = 1 / ONE SEAM** | one material, one intruder, one seam, in all four. | ✓ |

---

## 5. ⭐ THE LEAD VARIATION — and the ranking, with reasons

| rank | | why |
|---|---|---|
| **1** | **V2 · ONE POINT OH** | ⭐ **The hinge and the register are the same event.** `I tagged it one point oh at an hour nobody tags things. / Then I pushed it. / Then it stopped being mine. That took no time at all.` — the number, the act, the wrong speed and the want all land in three lines, and the seam falls on the word `pushed`. It is also the only variation where the **flaws are the mechanism** rather than a detail, which is the least predictable idea in the set. |
| **2** | **V3 · IT CHANGED ONE LINE** | the strongest nostalgia bait of the four (recognising your own hand is the purest form there is) and therefore the cleanest kill. ⚠️ Held at 2 because it runs closest to P05's neighbour (**awe**) and needs the vocal to stay unimpressed. |
| **3** | **V1 · TUESDAY, EARLIER** | the most legible and the most singable; the fastest route into the register for a stranger. ⚠️ Held at 3 because at 104 BPM with a banjo it is the most likely of the four to render as pleasant folk-pop and lose the seam to a blend. |
| **4** | **V4 · TWELVE LINES LEFT** | ⭐ the best writing in the set (`Complete and rough leaves. Careful and half stays.` / `The thing my hands know when the log goes quiet.`) ⚠️ and the highest risk: it is one solemn vocal take away from being an elegy, which is the exact failure the refused element exists to prevent. **Ranked last on risk, not on quality.** *(The Wodehouse ruling: lightness is the first thing lost to revision, and this is the variation revision would sink.)* |

⛔ **All four ship as a set.** The ranking is a render-order recommendation for The Scientist, not a cut. ⛔ **Nothing here is rendered, published or spent** (`vault/AUTONOMY.md`).

---

## 6. THE DEDICATION — drafted, measured at step 11

⛔ **The prompt names a break.** It asks for a **three-finger banjo roll** and a **traded break**, and the seam is built out of both. Both are named inventions.

> *(draft, to be measured under 500 chars at step 11)*
> The three-finger roll under this is Earl Scruggs's. The traded break — a line handed player to player — is the shape Bill Monroe's band built. We asked the generator for those figures by name, so we name them: nothing of theirs is sampled here. Bluegrass is a living scene; go hear the people playing it now. Vocals, arrangement and words are AI-generated by Lofn. Full credits below.

⛔ **These names appear in the dedication and nowhere else** — not in the style prompt, not in the exclude, not in the lyrics. *(A name in the prompt is an instruction to imitate; a name in the dedication is a debt paid.)*
⭐ **And the honesty clause runs both ways:** this is a generated track, not a sample. *"We asked for this figure by name, so we name him"* — never *"contains a sample of…"*.
