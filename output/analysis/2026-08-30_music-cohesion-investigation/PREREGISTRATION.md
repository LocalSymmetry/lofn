# Preregistration — Music Cohesion Regression Investigation

**Frozen:** 2026-08-30, before lyric extraction, scoring, catalogue-performance lookup, or doctrine-diff counting.

## Observation that triggered the investigation

The Scientist reports that recent Lofn songs feel choppy, disjointed, abstract, list-like, difficult to decode, low in listener commonality, and insufficiently compelling. She names **The Wedding Takes It Louder**, **Triple Arch Over Me**, **Spare Hair Tie**, **Burning Bush With a Barcode**, and **ASK FOR IT BY NAME** as the comparison bar, and says the 2026-08-29 batch does not clear it.

This file freezes tests. It does not assume the report is wrong or right.

## Frozen samples

### A — user-named compelling/high-bar songs (n=5)

1. `ASK FOR IT BY NAME` — Suno clip `5a68c8bc-767a-499f-a541-6bdbff0b8fd9`; lyric source will be the public clip payload or a byte-identical local artifact if found.
2. `Triple Arch Over Me` — `output/distribution/LYRICS/SIDE_A/A17 - Triple Arch Over Me.txt`.
3. `Spare Hair Tie` — `output/songs/20260809_the-spare-hair-tie_P04_V2.md`.
4. `Burning Bush With a Barcode` — `output/songs/2026-07-06_P05V1_burning-bush-with-a-barcode.md`.
5. `The Wedding Takes It Louder` — `output/songs/20260821_the_wedding_takes_it_louder_P06_V2.md`.

### B — the criticized recent batch (n=6)

All six shipped files under `output/daily/2026-08-29-lofn-cleverness-unbound/selected/`, with no replacement after scoring begins.

### C — onset/timeline control (8 runs, target n=48 selected songs)

The six declared selections from each of these already-completed canonical runs:

- `output/daily/2026-08-21/`
- `output/daily/2026-08-22-lofn-prime-every-element/`
- `output/daily/2026-08-24/`
- `output/daily/2026-08-25-sunna-modern-swagger/`
- `output/daily/2026-08-26-opus-masterpiece/`
- `output/daily/2026-08-27-high-energy-registers/`
- `output/daily/2026-08-28-three-act-mania/`
- `output/daily/2026-08-29-lofn-cleverness-unbound/`

If a run does not expose six unambiguous selected lyric payloads, that run is reported as unavailable rather than replaced. No partial/live 2026-08-28 run may enter.

## Hypotheses and numerical predictions

### H1 — adjacency continuity regressed (the choppiness claim)

**Hypothesis.** Recent lines behave more often as adjacent flashes than as a causal, referential, contrastive, or return-bearing chain.

**Measures.** On stripped sung lines only: (a) ultra-short line rate (1–3 lexical words); (b) connective/reference carry rate; (c) blinded hard-cut judgment on every adjacent pair, where a hard cut means the second line neither advances, specifies, causes, contrasts, answers, nor deliberately returns to the first.

**Prediction.** Cohort B hard-cut rate is at least **15 percentage points higher** than cohort A; its ultra-short-line rate is at least **10 points higher**; and at least **4/5** A songs beat the B median on continuity.

**Falsifier.** H1 fails if the hard-cut gap is under **10 points**, or fewer than 4/5 A songs beat the B median. The deterministic short-line result alone cannot rescue a failed blinded continuity result.

### H2 — recognition/commonality regressed (the abstraction claim)

**Hypothesis.** Recent songs provide fewer immediately occupiable situations and more propositions or mechanisms that must be decoded before a listener can locate herself.

**Measures.** Blind readers answer after first verse + first chorus: (1) “Where are we, literally?” (2) “What is happening?” (3) “What does the singer want now?” Each answer is scored clear without interpretation / inferable / unavailable. Full lyrics also receive line-level concrete-action, sensory-description, abstract-proposition, and device-about-itself labels.

**Prediction.** At least **4/5** A songs receive clear literal situation + clear want, versus at most **2/6** B songs. B has at least **15 percentage points** more abstract-proposition/device-about-itself lines and at least **15 points** fewer concrete-action lines than A.

**Falsifier.** H2 fails if at least 4/6 B songs yield a clear situation and want, or if the line-class gaps are each under 10 points.

### H3 — compulsion regressed

**Hypothesis.** The high-bar songs create an unresolved demand—decision, refusal, danger, desire, promise, or irreversible act—that makes the next line necessary; recent songs more often display an idea without forcing consequence.

**Measures.** Blind readers score: clear want by first chorus; human cost of failure; irreversible action/decision; chorus consequence; and whether the final return changes the stakes. Each is yes/no with cited lyric evidence.

**Prediction.** At least **4/5** A songs clear four of five compulsion signals, versus at most **2/6** B songs. The median B song clears no more than **2/5**.

**Falsifier.** H3 fails if 4/6 B songs clear four signals, or if the A–B median gap is under one signal.

### H4 — formal-device load displaced song causality

**Hypothesis.** Recent briefs and packages require more simultaneous semantic/formal engines, causing successive lines to service different devices instead of one dramatic through-line.

**Measures.** Count independently named mandatory engines/constraints in the pair assignment, guide, and final package; count device switches visible in sung lines; compare within cohort C to hard-cut and clear-want outcomes.

**Prediction.** B has a median of at least **3** mandatory semantic/formal engines per song and at least **2×** the median of the 2026-08-21 control. Across cohort C, engine count has Spearman **rho ≥ +0.40** with hard-cut rate and **rho ≤ −0.40** with clear-want score.

**Falsifier.** H4 fails if B's engine load is below 3, is not at least 1.5× the 2026-08-21 control, or both correlations have absolute value under 0.25. A high-bar song with equal engine load but strong causality is retained as a required counterexample.

### H5 — the break begins in constraint accretion after 2026-08-21

**Hypothesis.** The system changed from organizing songs around one concrete relationship/event to asking each song to demonstrate multiple portfolio doctrines (element schedules, dualities, rare registers, century/form transformations, semantic rekeys). The onset should be visible in source artifacts, not merely asserted from memory.

**Measures.** For each timeline run: count mandatory production/lyric constraints in the frozen seed, assignments, and QA; measure selected-song engine load, hard-cut rate, clear-want rate, list density, and abstract-line rate. Separately inspect git history between the 2026-08-21 song and each later run.

**Prediction.** Median mandatory-constraint count rises by at least **50%** after 2026-08-21, and the first run with both a ≥50% constraint rise and a ≥10-point hard-cut rise occurs no later than 2026-08-24.

**Falsifier.** H5 fails if constraint load does not rise by 50%, if the continuity decline predates the rise, or if the 2026-08-21 songs are no more coherent than later songs. Mere temporal coincidence does not establish cause.

### H6 — list/fragment texture is a mechanism, not the whole disease

**Hypothesis.** Lists and fragments contribute to choppiness only when they lack a stable speaker want or causal return; they are not intrinsically defective.

**Measures.** Fragment/list rates crossed with clear-want and transformed-return scores. Named high-bar counterexamples are actively sought.

**Prediction.** At least one A song has a fragment/list rate within the B interquartile range while still clearing ≥4/5 compulsion signals. Within cohort C, fragments predict hard cuts more strongly in songs without a clear want than in songs with one (interaction-direction test; magnitude treated as exploratory).

**Falsifier.** H6 fails if no high-bar counterexample exists and fragment/list rate predicts discontinuity equally in both want conditions.

## Named confounds measured alongside

- **Age/date** (mandatory): creation date and run date.
- **Audio vs text:** the high bar includes rendered songs; the 2026-08-29 repository package is text-only. All primary lyric comparisons use text on both sides. No claim about sung phrasing or production survival may be made without the recent audio.
- **Selection bias:** cohort A is explicitly cherry-picked by human impact; cohort B is a complete selected batch. Direction matters; effect sizes are descriptive.
- **Song length:** sung lines, lyric words, sections, and duration where a render exists.
- **Genre/tempo/form:** especially rap-like short-line forms and the three-act suite run.
- **Personality and model/step-11 route:** Lofn vs Sunna and differing refinement models.
- **Theme seriousness:** source subject and emotional register may cause both constraint choices and impact.
- **Release/performance exposure:** plays and like rates are not comparable for unreleased text packages and will not be used to grade B.
- **Extraction format:** headers, EMO tags, SFX, parentheticals, and production prose must be removed consistently and the extraction table retained.

## Controls

- The complete 2026-08-21 six-song selection tests whether Wedding was a singular outlier or representative of that pipeline state.
- Triple Arch and Burning Bush are established Golden/reference packages; Spare Hair Tie is a measured audience outlier.
- A known nonselected 2026-08-29 variation will be included only in the fresh QA calibration as the decoy, never in the primary A/B statistics.
- At least one high-bar fragment/list counterexample must be sought before any anti-fragment rule is proposed.

## Default expectation

**The default is that the broad “the pipeline became incoherent after Wedding” claim partially dissolves.** Cohort A is human-selected at the peak, audio availability differs, and two recent catalogue aggregates dissolved under age/control checks. I expect a narrower mechanism to survive—most plausibly loss of one stable want under accumulated device and portfolio mandates—but the null must be cheap to accept.

## What this design cannot settle

- Whether Suno's melody, phrasing, or edit made recent rendered songs feel choppier without the exact audio files.
- Whether audience performance would confirm the text diagnosis before comparable release exposure.
- Whether any repair works. This run diagnoses only; a later controlled retry must test interventions.

## Change embargo

No doctrine, skill, gate, personality, validator, or pipeline file will be edited in this investigation. Any proposed change remains an experiment proposal until The Scientist reviews the result.
