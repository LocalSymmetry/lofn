# P02V03 Soundcraft Repair Report — `Leave It Crooked`

## Provenance and Scope

- Session: `/root/sunna_p02`; `spawned_by_parent: true`.
- Binding brief: `output/daily/2026-08-24-sunna/music/SOUNDCRAFT_REPAIR_BRIEFS.md`.
- Numeric authority: `vault/gates.yaml`, `rhyme_window: 4`, `rhyme_return_floor: 0.30`, `line_return_floor: 0.20`, `mean_words_per_line_ceiling: 7.5`, `alliteration_per_100w_floor: 11.0`.
- Repair targets: strict end-rhyme `>=0.32`, exact line return `>=0.22`, alliteration `>=12.0/100w`, words/line `<=7.5`.
- Modified selected blocks only in `pair_02_step09_artist_refined.md`, `pair_02_step10_revision_synthesis.md`, and `pair_02_step10_final_package_enhanced.md`; the Step 09/10 count rows and Step 11 V03 measurement/image-ladder notes were updated to remain truthful.
- No sibling artifact was opened for creative use. Portfolio validators performed their required deterministic aggregate reads. No render, upload, publication, spend, commit, or push occurred.

## Verdict

**PASS — binding repair achieved with margin.** Final Step 11 P02V03 measures strict end-rhyme **0.351351**, exact line return **0.337838**, mean words per sung line **4.878378**, and alliteration **13.296399 per 100 words** across **74** sung lines. The repair adds local `-icket`, `skin`, `it`, and `tap` returns plus speakable `h`, `s`, `p`, `b`, `c`, `w`, `n`, and `r` clusters. It introduces no production talk or message explanation.

## Anchor Preservation

- **Damp claim ticket:** `[Theme:]` still names the damp claim ticket; `Hold this damp claim (with me)` remains unchanged, and the opening now physically returns to `ticket`.
- **Washer glass:** `Hold it to the glass (this glass)` remains unchanged; `Where washer glass sweats` makes the same surface more mouth-active.
- **Blue-square removal:** `What's that blue square? (right there)` remains unchanged; `Let blue square slip, let it run` and `Blue square slips through neon water` make removal explicit without explaining its meaning.
- **Retained crescent:** `My red crescent curves (kiss me)` and `My red crescent curves and stays` preserve the chosen mark through the wash.
- **Fold → keep:** both `Fold it with me` lines in Return 1 and both `Keep it with me` lines in Return 2 remain byte-identical.
- **Missing-center brass:** all three music prompts, exclude prompts, titles, and missing-center/outer-note production instructions remain unchanged; the lyric still ends the second return with `Leave one note out`.

## Canonical Soundcraft Measurement

Measured with the shipped definitions from `scripts/measure_soundcraft.py`: `lyric_blocks()` → selected P02V03 block → `sung()` → `profile()`. `strict_end_rhyme` uses the binding last-three-characters / ±4-line window.

| Artifact | Metric | Old | New | Repair target | Verdict |
|---|---|---:|---:|---:|---|
| Step 09 | strict end-rhyme | 0.229730 | 0.351351 | >=0.32 | PASS |
| Step 09 | exact line return | 0.337838 | 0.337838 | >=0.22 | PASS |
| Step 09 | words/line | 4.675676 | 4.851351 | <=7.5 | PASS |
| Step 09 | alliteration/100w | 8.092486 | 13.370474 | >=12.0 | PASS |
| Step 10 | strict end-rhyme | 0.229730 | 0.351351 | >=0.32 | PASS |
| Step 10 | exact line return | 0.337838 | 0.337838 | >=0.22 | PASS |
| Step 10 | words/line | 4.675676 | 4.851351 | <=7.5 | PASS |
| Step 10 | alliteration/100w | 8.092486 | 13.370474 | >=12.0 | PASS |
| Step 11 | strict end-rhyme | 0.229730 | 0.351351 | >=0.32 | PASS |
| Step 11 | exact line return | 0.337838 | 0.337838 | >=0.22 | PASS |
| Step 11 | words/line | 4.729730 | 4.878378 | <=7.5 | PASS |
| Step 11 | alliteration/100w | 8.000000 | 13.296399 | >=12.0 | PASS |

Other deterministic package measures remain valid:

| Artifact | Lyrics-field chars | Sung lines | Full EMO headers | SFX | Disc_Channel lines |
|---|---:|---:|---:|---:|---:|
| Step 09 | 2,829 | 74 | 7 | 1 | 0 |
| Step 10 | 2,885 | 74 | 7 | 1 | 0 |
| Step 11 | 3,411 | 74 | 7 | 1 | 5 |

## Exact Changed Lyric Lines

The following 25 substitutions are the complete sung-line repair. Step 09 and Step 10 use the same repaired lines; Step 11 retains its pre-existing body-noise prefix where noted.

| # | Before | After |
|---:|---|---|
| 1 | `My heel stops the basket` | `My heel hooks the basket` |
| 2 | `Your hands wait by mine` | `Your hand holds the ticket` |
| 3 | `Pink sticks to my thumb` | `Pink sticks, slick on skin` |
| 4 | `Don't fold it yet (why not?)` | `Don't fold; let it kiss skin` |
| 5 | `That shirt can spin (let it)` | `That shirt spins (let it)` |
| 6 | `Your eyes stay here` | `Your eyes stay on it` |
| 7 | `Tap by the punch hole (kiss me)` | `Tap the pink punch hole (kiss me)` |
| 8 | `One ruby little moon` | `One ruby moon for me` |
| 9 | `Basket taps my heel (tap-tap)` | `Basket bumps my heel (tap-tap)` |
| 10 | `My hip taps it back` | `My hip bumps back (tap-tap)` |
| 11 | `I have plans for it` | `I have hot plans` |
| 12 | `Cold on the claim` | `Cold corner on the card` |
| 13 | `Too flat for my kiss` | `Too flat, too flawless for my kiss` |
| 14 | `Where the washer sweats` | `Where washer glass sweats` |
| 15 | `Your palm keeps it still (for you)` | `Your palm pins it still (for you)` |
| 16 | `My nail lifts the edge` | `My nail nicks the edge` |
| 17 | `I want you closer` | `I pull you closer` |
| 18 | `Past every straight side` | `Past straight, slick sides` |
| 19 | `My red keeps its curve (kiss me)` | `My red crescent curves (kiss me)` |
| 20 | `Around the little hole` | `Round the ruby punch hole` |
| 21 | `Good, lean right there` | `Good—grin, lean right there` |
| 22 | `Let the square run` | `Let blue square slip, let it run` |
| 23 | `Peel slow, pretty thing` | `Peel blue, pretty thing` |
| 24 | `My crescent stays` | `My red crescent curves and stays` |
| 25 | `Blue leaves in water` | `Blue square slips through neon water` |

Step 11 variants of two substitutions:

- `Mm, your hands wait by mine` → `Mm, your hand holds the ticket`.
- `Blue drains out of the night` → `Blue square slips through neon water`.

No Return 1 or Return 2 line was changed. Exact chorus return remains **0.337838** rather than being inflated with filler.

## Selected-Block Hashes

SHA-256 is computed over the complete selected variation block in each artifact, from its variation heading up to—but not including—the next variation heading or global post-variation section.

| Artifact | Old P02V03 SHA-256 | New P02V03 SHA-256 |
|---|---|---|
| Step 09 | `0bff5a73164458760ba66750c2574aa70991ff14bffc909d29d7b9c3968ece3e` | `83e1fbc5eb628b8c95ee95c286baa4ea21ec0c9287198758cfac504ccb6e4f00` |
| Step 10 | `3a5788dd9a8ac197e2f15eb05d976c4c967a8a6e27d42e6215c31aa60a9fb038` | `c0aca19b86605e8d24328b57d059e2884e480434aa1c7f837b767c04d71c9a6e` |
| Step 11 | `116970d56f3e085e09b19912ab55a9d237a2b3113e9a5da61280519ca1b2d158` | `ca4be5e80f33826b84b97219e18eb98b1861ff2e1b4b8c629b92aa21f598e87c` |

## Byte-Identity Proof for Untouched Variations

Hashes use the same variation-only boundaries. Every untouched block has an identical pre/post SHA-256.

| Artifact | Variation | Pre-repair SHA-256 | Post-repair SHA-256 | Verdict |
|---|---:|---|---|---|
| Step 09 | V01 | `6444796bb9899ee0d56ee14bce075860967f12850a2a1a1563f1b9ca5fd17820` | `6444796bb9899ee0d56ee14bce075860967f12850a2a1a1563f1b9ca5fd17820` | IDENTICAL |
| Step 09 | V02 | `19dc615f6734720627ac5204a9ca37fe2fd5a8ce428bff12b774621021311606` | `19dc615f6734720627ac5204a9ca37fe2fd5a8ce428bff12b774621021311606` | IDENTICAL |
| Step 09 | V04 | `64f348249409746df2b9f8bc046d05d17afe08a0a68cbd9a8e5554556a9ea232` | `64f348249409746df2b9f8bc046d05d17afe08a0a68cbd9a8e5554556a9ea232` | IDENTICAL |
| Step 10 | V01 | `2d327a6fc873a250f7412e1e4dd3fc6bdabc76a98a4ee9b3bb11ecc9da61f55e` | `2d327a6fc873a250f7412e1e4dd3fc6bdabc76a98a4ee9b3bb11ecc9da61f55e` | IDENTICAL |
| Step 10 | V02 | `64ccc6b017967e23122060e5d6e178b9b8505262a7aa0093c25439f9b694205d` | `64ccc6b017967e23122060e5d6e178b9b8505262a7aa0093c25439f9b694205d` | IDENTICAL |
| Step 10 | V04 | `e6fe0b0fdb4a3370c1f6a0d7b1b93af7f2c0e393c4d50141dd33939ee0b336f1` | `e6fe0b0fdb4a3370c1f6a0d7b1b93af7f2c0e393c4d50141dd33939ee0b336f1` | IDENTICAL |
| Step 11 | V01 | `f639ff749bb5c18824155b84bd16c1fe8af8eab7031dd276b1f4c6334e176521` | `f639ff749bb5c18824155b84bd16c1fe8af8eab7031dd276b1f4c6334e176521` | IDENTICAL |
| Step 11 | V02 | `2022757b261246c6b8c7ef5d2911360793b8a49441818cf76b04860af445edb5` | `2022757b261246c6b8c7ef5d2911360793b8a49441818cf76b04860af445edb5` | IDENTICAL |
| Step 11 | V04 | `3629cb9a7a21c0a5e768f847d8bfd3615c6f1c2d9e9d4a687ad87935a902c93e` | `3629cb9a7a21c0a5e768f847d8bfd3615c6f1c2d9e9d4a687ad87935a902c93e` | IDENTICAL |

Global count-summary rows were updated only where they report V03; these rows are outside all variation blocks.

## Dedication Re-measurement

P02V03 Step 11 dedication length: **296 characters**. Limit: **<=500**. Verdict: **PASS**. Dedication text is byte-identical.

## Validator Outputs

### Canonical step validators

```text
STEP 09 PASSED: output\daily\2026-08-24-sunna\music\pair_02_step09_artist_refined.md
VALIDATION PASS on attempt 1/3

STEP 10 PASSED: output\daily\2026-08-24-sunna\music\pair_02_step10_revision_synthesis.md
VALIDATION PASS on attempt 1/3

STEP 11 PASSED: output\daily\2026-08-24-sunna\music\pair_02_step10_final_package_enhanced.md
VALIDATION PASS on attempt 1/3
```

### Pair validator

```text
PAIR VALIDATION START pair_02 in output\daily\2026-08-24-sunna\music
STEP 06 PASSED: output\daily\2026-08-24-sunna\music\pair_02_step06_facets.md
VALIDATION PASS on attempt 1/3
STEP 07 PASSED: output\daily\2026-08-24-sunna\music\pair_02_step07_song_guides.md
VALIDATION PASS on attempt 1/3
STEP 08 PASSED: output\daily\2026-08-24-sunna\music\pair_02_step08_generation.md
VALIDATION PASS on attempt 1/3
STEP 09 PASSED: output\daily\2026-08-24-sunna\music\pair_02_step09_artist_refined.md
VALIDATION PASS on attempt 1/3
STEP 10 PASSED: output\daily\2026-08-24-sunna\music\pair_02_step10_revision_synthesis.md
VALIDATION PASS on attempt 1/3

PAIR VALIDATION PASS
```

### Portfolio distinctiveness

```text
extracted 6 pair files: pair_01_step09_artist_refined.md (14178 chars), pair_02_step09_artist_refined.md (16145 chars), pair_03_step09_artist_refined.md (14605 chars), pair_04_step09_artist_refined.md (13947 chars), pair_05_step09_artist_refined.md (12959 chars), pair_06_step09_artist_refined.md (13752 chars)
STEP 09 DISTINCTIVENESS PASS

extracted 24 finalist packages from 6 pair files
PORTFOLIO DISTINCTIVENESS PASS

extracted 24 finalist packages from 6 pair files
PORTFOLIO DISTINCTIVENESS PASS
```

The second and third portfolio verdicts are the Step 10 and Step 11 runs respectively; all three commands exited `0`.

## Completion State

- Soundcraft targets: PASS.
- Structural validators: PASS.
- Pair validator: PASS.
- Step 09/10/11 distinctiveness: PASS.
- Untouched variation identity: PASS.
- Dedication cap: PASS.
- Repair sidecars: none.

