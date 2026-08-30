# PAIR 03 — STEP 10 · REVISION SYNTHESIS · **LOFN-PRIME**

**Run:** `2026-08-27-high-energy-registers` · **Register:** *holding a load that is not yours; the cost is displacement, never strain* · ⛔ **Template defeated:** THE MARTYR BALLAD

---

# §1 — ⭐ THE MEASUREMENT, RUN ONCE, AGAINST THE SHIPPED PACKAGE

⛔ **Not against step 08.** Step 08 measured a draft that had five defects in it. **A number does not become true by being copied forward, and it does not become true by being measured against an older file.**

**Command:**

```
python p03_measure.py pair_03_step10_final_package_enhanced.md
```

*(`p03_measure.py` parses the **fenced blocks under a heading naming one of the three pasted fields** — the same unit `scripts/check_field_census.py` uses, chosen deliberately because four of the coordinator's scanners failed in one day by parsing markdown HEADINGS. It imports `sung()` from `scripts/measure_soundcraft.py`, **THE canonical definition, never re-derived.**)*

| check | gate | **V1** | **V2** | **V3** | **V4** | verdict |
|---|---|---|---|---|---|---|
| music prompt chars | 850–1000, target **870–960** | **943** | **952** | **933** | **956** | ✅ all four inside target |
| exclude chars | 400–900 (hard max 1000) | **823** | **743** | **747** | **730** | ✅ |
| lyrics field chars | **< 5000**, target ≤ 4800 | **4669** | **4615** | **4566** | **4742** | ✅ |
| sung lines | 70–120 (floor-hug ≤ 72) | **83** | **76** | **74** | **74** | ✅ (V3/V4 low-half, declared) |
| ⭐ person-words (my refused element) | **0** | **0** | **0** | **0** | **0** | ✅ |
| emotion words in sung lines (R2) | **0** | **0** | **0** | **0** | **0** | ✅ |
| strain words (R5 amended) | **0** | **0** | **0** | **0** | **0** | ✅ |
| armour vocabulary (cliché #10) | **0** | **0** | **0** | **0** | **0** | ✅ |
| distinct numerals sung (ONE-FACT) | **1** | `ten` | `one` | `four` | `twice` | ✅ 1 each, 0 digits |
| first person (reported, not gated) | — | **0.8%** | **1.3%** | **1.9%** | **1.0%** | ⚠️ **pre-registered 4–7% — FALSIFIED** |
| body-words | under the 5–8% norm | **3.1%** | **2.8%** | **3.7%** | **4.5%** | ✅ (cf. *Spare Hair Tie* 4.0%) |
| **cross-variation shared sung lines** | **0** | — | — | — | — | ✅ **0 of 6 pairwise, 0 lines** |
| forbidden-token lexicon hits | **0** | — | — | — | — | ✅ **0**, coordinator's own reader |
| dedication chars | ≤ 500 | — | — | — | — | ✅ **479** |
| ICB prefix | byte-0, 267,277 B, `11c7e9fd7edc` | — | — | — | — | ✅ verified |

**Supporting commands, quoted rather than summarised:**

```
python -c "import hashlib;d=open('CREATIVE_CONTEXT.md','rb').read();print(len(d),hashlib.sha256(d).hexdigest()[:12])"
        -> 267277 11c7e9fd7edc

python -c "icb=open('CREATIVE_CONTEXT.md','rb').read();d=open('pair_03_step10_final_package_enhanced.md','rb').read();print(d[:len(icb)]==icb)"
        -> True    (package 317569 B = 267277 ICB + 50292 body)

# the COORDINATOR's reader, run against my fields with its own lexicon:
scripts/check_field_census.py :: load_lexicon + pasted_fields + hits
        -> fields parsed: 12    TOTAL LEXICON HITS: 0

grep -in "strength" skills/lofn-core/refs/EMOTION_TAXONOMY.md
        -> 144:- **Empowerment**: Strength, Self-efficacy, Confidence      (one line, only line)
```

⛔ **Every row above is still a CLAIM until the coordinator re-measures it.** `python3 scripts/check_field_census.py <run-dir>` reads the fields, never a verification row. **Where the coordinator and I disagree, the coordinator is the number** — three of yesterday's shipped violations sat underneath a row saying zero, each written by the pair being counted.

---

# §2 — ⭐ R7 PART TWO · THE LISTENER PREDICTION, FILED BEFORE ANY RENDER

⛔ **No render exists and none is authorized** (`vault/AUTONOMY.md`: drafts on disk; nothing rendered, published, or spent). This is filed **now**, before a single audio file exists, because *a prediction filed after the writing is a description* — and per `vault/PREREGISTRATION_STANDARD.md`, **the only aggregate that dissolves cheaply is the pre-registered one.**

## 2.1 — The default expectation, stated out loud first

> ⚠️ ⭐ **THE DEFAULT IS THAT A BLIND LISTENER CALLS AT LEAST TWO OF THESE FOUR "SAD" OR "HEAVY".**
>
> Not because the lyric is — it is measured at **0 emotion words, 0 strain words, 0 armour words** — but because **74–82 BPM + a low female lead + a sparse arrangement + no chorus lift IS the martyr ballad's sonic template**, and the listener pre-files from the sound inside two seconds, before a word arrives. **That is the whole difficulty of this brief and I am not going to pretend I have solved it in text.** Writing this sentence down is what makes a null cheap to accept.

## 2.2 — The predictions, in numbers, each with its falsification condition

**Protocol chosen before the data:** four listeners per song, **prompt unseen and lyric sheet unseen**, one question — ***"what happens in this song?"*** — answered before any second question. **Control:** one run-mate at a different tempo carrying a different register, judged in the same session by the same listeners. ⛔ **Without the control, "listeners said calm" is unattributable to anything.**

| id | prediction | falsified if | ⚠️ what I actually expect |
|---|---|---|---|
| **P1** | ⭐ **The Vincent Test, in audio.** ≥ 3 of 4 name an **OBJECT and an ACTION** first — *a shelf held up · a rope held · something not said · a bag carried* — rather than a style or a feeling. | **2 or more name a STYLE or a FEELING before an event.** | **holds for V1 and V3, fails for V4.** V4's loop-and-flutter texture is the most style-forward of the four |
| **P2** | ≤ 1 of 4 reaches for a martyr word unprompted — *sacrifice · burden · suffering · selfless · carrying everything · strong.* | **2 or more.** | **holds.** The words are not there to prime and the grammar has no witness slot |
| **P3** | The most-used register word is a **flatness** word (*steady · matter-of-fact · deadpan · unhurried*) rather than a **sadness** word (*sad · heavy · mournful · resigned*). | **sadness words outnumber flatness words.** | ⚠️ **I expect this to FAIL.** *Resigned* is a sadness word wearing my register's exact clothes, and the tempo invites it |
| **P4** | ⭐ **The R4 test.** ≥ 2 of 4 report **the seam** — *"something stops" / "something goes missing"* — **without being told to listen for it.** | **1 or 0.** | ⚠️ **I expect this to FAIL, and the distinction matters.** The doctrine claims **a generator cannot smooth a click that does not arrive.** It does **not** claim a listener will notice one. **Those are different claims and I am only entitled to the first** |
| **P5** | The seed's own claim, numeric: `scripts/measure_render.py` shows these four with **larger** short-term dynamic range / crest factor than a run-mate that requested a ceiling. | **no separation, or the wrong sign.** | genuinely unknown — `FRONTIER.md` is **57/57 UNEXPLORED, 0 probes**. ⛔ **Nothing here may be justified by "we know this works loud."** |

## 2.3 — Confounds, named alongside rather than discovered afterwards

⚠️ **Three confounds all push the same way — toward the sad reading — and not one of them is my lyric:**

1. **Tempo.** Mine is the slowest band in the run (74–82). A slow song reads graver independently of content.
2. **Vocal register.** Contralto / low alto reads graver than the alternatives, in every one of the four.
3. **Density.** Mine is the sparest arrangement in the run, deliberately, because the floor goes at the front.

> ⭐ **So a "sad" result would NOT falsify the register.** It would falsify a **different and more interesting** claim: *that the register survives its own sonic package.* If P3 fails while P1 and P2 hold, the honest reading is **the words did their job and the sound overruled them** — which is precisely THE GRAIN LAW's warning that a distinctiveness objection answered in the production spec is not answered, pointed back at me. ⭐ **The repair in that case is not a louder lyric. It is a different tempo band for this register next time**, and the run's own pair-assignment table is where that decision lives.

## 2.4 — What survives in narrower form if the whole thing dissolves

Even at total listener failure, **two claims are independent of the listening result and are already measured in text:**

- **The person-word deletion works as an instrument.** 0 hits across four songs, one repair, no exemption needed — and the default expectation was that one would be. **The martyr template genuinely could not be built inside that grammar.**
- **The taxonomy hole is a count, not an opinion.** `strength` appears once, under `Empowerment`; a carried weight appears once, under `Overwhelm`, under Sadness. **Those numbers do not depend on anyone hearing anything.**

---

# §3 — ⭐ THE PRE-REGISTERED NUMBER THAT FAILED, AND WHY IT WAS WORTH WRITING DOWN

**Pre-registered** (`pair_03_step06_facets.md` §3.2, filed before any lyric): first person at **4–7%** of sung words, transplanted from *Spare Hair Tie*'s measured **4.4%**.
**Measured: 0.8 / 1.3 / 1.9 / 1.0%. Falsified, four for four, by a factor of two to eight.**

⛔ **The target was not moved and the songs were not padded to reach it.** Two first-person lines were added at step 09, both because **R5 requires the cost in the singer's own mouth** — a ruling, not a rate — and they moved V1 from 0.6% to 0.8%, which changes nothing.

**What survives, narrower:** the *Spare Hair Tie* **configuration** transplanted exactly — pronoun-free verses, `I`/`my` confined to the hook, **body-words at 2.8–4.5% against a 5–8% house norm and beside its own measured 4.0%.** ⭐ **The RATE did not, and the reason is structural: first-person rate is a function of what the HOOK IS ABOUT.** That song's hook is a want, so it is full of a self. Mine is an object — a board, a rope, a sentence, a strap — **because the register requires that nobody be watching.** A hook whose grammatical subject is the load has almost nowhere to put a pronoun.

> ⭐ **The carry-forward claim: first-person rate is not transferable between registers and should never be set as a cross-register target.** It is a **downstream consequence** of hook grammar, not an independent dial.

⚠️ **And the honest counter, stated rather than buried:** 1% first person sits very close to the 2026-08-24 defect — *a narrator has no stake, so there is no peak.* **The defence is that all four discriminators are first person and all four sit in the hook**, so the stake is **concentrated**, not absent. **Concentrated is not the same as absent — but I cannot prove the difference from text. Only a listening pass could, and there is not one.**

---

# §4 — WHAT WOULD HAVE TO BE TRUE FOR THIS PAIR TO BE WRONG

⛔ **A pair that cannot say how it would fail has not been aimed.**

1. **If a blind listener names a style or a feeling before an event** on 2 or more of the four, **the discriminator is in the wrong place** — R3 puts it in the hook and I put it there, so the failure would be that a hook cannot outrun a sonic template at this tempo, and the register needs a different band.
2. **If the seam is inaudible** (P4), **the R4 carrier is doing nothing perceptual** — it survives the generator, which is all the doctrine claims, but *survives* and *lands* are different, and the run should stop treating them as one.
3. **If the person-word ban produced fog** — if a listener cannot say who is holding anything — **the refused element was too blunt** and the pre-registered falsification condition in step 06 fires. **Text says it did not; only ears can confirm.**
4. **If `Sufficiency` turns out to be a shade of `Composure` after all**, the doctrine hole is my misreading and not a hole. ⚠️ **I cannot rule this out from inside; a verifier who is also the subject has a direction of error.**

⛔ **Nothing rendered. Nothing published. Nothing spent. Every claim in this pair is a text claim, and the two that are not — P1 through P5 — are filed unrendered on purpose.**
