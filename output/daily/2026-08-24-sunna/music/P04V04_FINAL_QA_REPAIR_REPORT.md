# P04V04 Final QA Repair Report — Again Before Credits

## 1. Scope and verdict boundary

- Repair source: `suno_render/QA_REPORT.md` and `TOP_6_FINAL_SELECTION.md`, both read completely.
- Selected source only: P04V04 `Again Before Credits` in Steps 09, 10, and 11.
- Execution provenance: `session_key: /root/sunna_p04`; `spawned_by_parent: true`.
- This is a text-package repair. It is not a render, listening verdict, publication decision, or SHIP authorization.
- No `suno_render` copy was edited; the coordinator must rebuild it from the repaired canonical source.

## 2. QA finding answered

The old hinge and tail narrated their own arrangement. The repair removes every cited line—`My bass won't shut up`, `The horn leaves one bright note`, `I sing it and grin` / `Hah—I sing it and grin`, `Bass stays hot`, and `Bass says done`—plus the adjacent mechanism line `The low drum steps in`. `Come occupy this` was also replaced because it asked for an action after the song needed to have completed it.

The new causal chain is entirely human and physical:

`cup slides left → chosen adult steps in → red straw points at concession gap → chosen adult grins → shoulder fills the space → she chooses him now → he answers AGAIN`.

The low-dembow seam and optional horn handoff remain production grammar, but neither carries meaning. If the renderer smooths the missing note, the event still happens.

## 3. Exact old/new sung lines

These seven one-for-one substitutions were made in V04 of Steps 09 and 10:

| # | Old | New |
|---:|---|---|
| 1 | `My bass won't shut up` | `Red straw; your smile tips up` |
| 2 | `The low drum steps in` | `You slide one step in` |
| 3 | `The horn leaves one bright note` | `Red straw points at the gap` |
| 4 | `I sing it and grin` | `You grin at the gap` |
| 5 | `Come occupy this` | `Your shoulder fills this space` |
| 6 | `Bass stays hot` | `Your smile stays hot` |
| 7 | `Bass says done` | `Your warm hand meets mine` |

Step 11 uses the same substitutions, with its functional body-noise cell preserved on line 4:

| Old | New |
|---|---|
| `Hah—I sing it and grin` | `Hah—you grin at the gap` |

Dedication correction in Step 11:

| Old | New |
|---|---|
| `This song was generated with AI; full credits above.` | `This song was generated with AI; full credits below.` |

### Sunna R4 HUM-test repair — attempt 1

The fresh recheck found a separate pre-existing omission: the defining instrumental figure appeared before the first vocal but was not requested after the last. The repair leaves every sung line untouched and changes the final SFX command in Steps 09, 10, and 11 as follows:

| Old | New |
|---|---|
| `*cup lid pops*` | `*after the final vocal, the defining two-note horn riff returns; cup lid pops*` |

The V04 music prompts now end: `Moved cup and occupied gap seal the hinge; the defining horn riff returns after final vocal, then lid pops.` The prompt trims `transparent`, `small`, and `crisp` from non-load-bearing production adjectives to make room without changing tuning, organ, claps, spatial staging, or the human/object hinge. The return happens after `Again` in Steps 09–10 and after `Ah—again` in Step 11; it is an instrumental command, not sung craft narration, a third outburst, or an encore.

## 4. Soundcraft measurement

Canonical functions: `scripts/measure_soundcraft.py → sung() → profile()` on the isolated final Step-11 V04 package.

| Metric | Before final-QA repair | After | Delta | Floor/ceiling | Verdict |
|---|---:|---:|---:|---:|---|
| Strict end-rhyme return | 0.342857 | 0.371429 | +0.028572 | >=0.30; repair margin >=0.32 | PASS |
| Exact line return | 0.242857 | 0.242857 | 0 | >=0.20; repair margin >=0.22 | PASS |
| Mean words per sung line | 4.014286 | 4.100000 | +0.085714 | <=7.5 | PASS |
| Alliteration per 100 words | 13.523132 | 15.331010 | +1.807878 | >=11; repair margin >=12 | PASS |
| Sung lines | 70 | 70 | 0 | >=70 for this package | PASS |

Step 09 and Step 10 independently measure `end rhyme 0.371429 / line return 0.257143 / words-per-line 4.057143 / alliteration 15.492958 / 70 lines`.

Final generator-field counts after the HUM-test repair: music prompt `984`, exclude `702`, lyrics `3044`, dedication `320`.

## 5. Canonical selected-block hashes

Hash boundary is not reconstructed locally. Every row below calls `output/daily/2026-08-24-sunna/music/build_suno_render_menu.py::selected_block` and SHA-256-hashes the returned bytes. This first table records the earlier final-QA repair history; the second table records the latest HUM-test repair.

| Artifact | Before bytes / SHA-256 | After bytes / SHA-256 |
|---|---|---|
| Step 09 | `6616` / `0d1084ad01734d62563091eb430fd56434351e939a14195ca720a138a5a0881c` | `6631` / `d2b3a4285a2012ffd31a6de6a8c2ceae838dcfc2d3769b3c5716f7ab8595dac5` |
| Step 10 | `10262` / `2eb776da8571cb1628c883619ab46633ad6831adec83370f29afa90ca3d3346e` | `10277` / `8773c68abd6fcb845a26deaff8c6f5b93473c7072453c5e4ffae0c4686948ad2` |
| Step 11 canonical current source | `10108` / `03aac3ed6d0e370adcfd8af4688dc72ec720f25a635abc670995dd462822ede8` | `10225` / `3c7f4e03f37b6fc144163dc08440e046987ab20f3feb218a1f520de5b6240319` |

### Latest post-vocal horn-return hashes

| Artifact | Before HUM repair bytes / SHA-256 | After HUM repair bytes / SHA-256 |
|---|---|---|
| Step 09 | `6631` / `d2b3a4285a2012ffd31a6de6a8c2ceae838dcfc2d3769b3c5716f7ab8595dac5` | `6690` / `e0a1d54d945a7028c47200776664bcaf67c9ae933a36b8854dab95b87bff35dd` |
| Step 10 | `10277` / `8773c68abd6fcb845a26deaff8c6f5b93473c7072453c5e4ffae0c4686948ad2` | `10399` / `89b83cf13e3bcfa7abbcf1b133c2311dad41b55f0f348d478f1a6952614a1381` |
| Step 11 canonical current source | `10225` / `3c7f4e03f37b6fc144163dc08440e046987ab20f3feb218a1f520de5b6240319` | `10383` / `f974cd211a483040326a287f251a5450a3c4e26f6bd717ebda5226d5490d4ff1` |

### QA/manifest hash reconciliation

The QA and old manifest name `c744e67db93bc0f5d5851e468e6ad0b740d0445e70456d729e0a1efaa1052a73`. Applying the exact `selected_block` boundary to the pre-repair current block and normalizing CRLF to canonical LF reproduces that hash exactly. The prior raw on-disk block had CRLF serialization and therefore hashes to `03aac3ed…`; this is newline representation, not a different package field. The LF-normalized block after the first final-QA repair was `6067b2d704813439a86a13c4f94142a8ce73bd40944d3233f556979ae4711569`; the current post-HUM-repair LF hash is `ca5c8b5d3353eded81dbe13043d762b83c69c1ec066b839088abe5ba0ae3571d`. Root’s rebuild must use the builder’s returned current raw hash `f974cd211a483040326a287f251a5450a3c4e26f6bd717ebda5226d5490d4ff1`, not stale `2f76a91d…`, `c744e67d…`, or `3c7f4e03…`.

## 6. Byte-identity proof for V01/V02/V03

The following `selected_block` raw hashes are identical before and after repair:

| Artifact | Variation | Before SHA-256 | After SHA-256 | Verdict |
|---|---:|---|---|---|
| Step 09 | V01 | `42eb98f9b0dbad9c9414280d8b76872bc3d1fe4106fdcd33daf1b31ce897a809` | `42eb98f9b0dbad9c9414280d8b76872bc3d1fe4106fdcd33daf1b31ce897a809` | IDENTICAL |
| Step 09 | V02 | `3d23340fa49b6590f10d3689eb8aa83fc5c55e7ef67fd2cdfab3fa7d17473f14` | `3d23340fa49b6590f10d3689eb8aa83fc5c55e7ef67fd2cdfab3fa7d17473f14` | IDENTICAL |
| Step 09 | V03 | `5de439e80e80e21739a1a11e82c6dbc5090bbb707cb8c60d0d4971095bca3924` | `5de439e80e80e21739a1a11e82c6dbc5090bbb707cb8c60d0d4971095bca3924` | IDENTICAL |
| Step 10 | V01 | `f7cafacd3a7d356416a35011df05bc59c772f0a912253b1337278eaecc6535da` | `f7cafacd3a7d356416a35011df05bc59c772f0a912253b1337278eaecc6535da` | IDENTICAL |
| Step 10 | V02 | `0a05384a5417d88167d83503e719ec0983724d30ed5b707b07ad738f208ec504` | `0a05384a5417d88167d83503e719ec0983724d30ed5b707b07ad738f208ec504` | IDENTICAL |
| Step 10 | V03 | `bd3fdefc6c945036af277b9addf2ef41e378881647f4939a89431eeefe443a3c` | `bd3fdefc6c945036af277b9addf2ef41e378881647f4939a89431eeefe443a3c` | IDENTICAL |
| Step 11 | V01 | `a0df292d77c39f604f6691200f01ac5e2803cce4024abec8fc04adbd619d2bf5` | `a0df292d77c39f604f6691200f01ac5e2803cce4024abec8fc04adbd619d2bf5` | IDENTICAL |
| Step 11 | V02 | `f426c849e276f36fd25fe050416cb7b8038eccbf25ec8381a1b89517717d8381` | `f426c849e276f36fd25fe050416cb7b8038eccbf25ec8381a1b89517717d8381` | IDENTICAL |
| Step 11 | V03 | `e3b4ee0c71162648a5fc189b48a1171cc02e87bd1a8a717487fe945014c82f7d` | `e3b4ee0c71162648a5fc189b48a1171cc02e87bd1a8a717487fe945014c82f7d` | IDENTICAL |

## 7. Preservation checks

- Title `Again Before Credits`: preserved.
- Direct adult `you` and chosen adult’s `AGAIN`: preserved and made more causal.
- Soda-machine glass, cold cup, red straw, concession gap and occupied space: preserved; straw/gap/shoulder now carry the hinge.
- Exactly two outbursts: preserved.
- Outburst 2: remains a 14-cell syllable-compressed return.
- Short tail: remains four sung cells plus one combined post-vocal horn/lid SFX, no fade and no third outburst.
- Dominant brisk ska world and sole low-dembow hinge: preserved in prompt, Disc_Channel and section header.
- NEWS/AWE source alloy: green visibility, two outbursts and short tail remain; no astronomy lecture appears.
- Temporary-brightness contraband: `Don't promise an encore`, credits pressure, present choice and `AGAIN` remain action rather than doctrine.
- Start With One dedication priority: Lloyd Brevett remains first; `full credits below` is now truthful; dedication remains `320` characters.

## 8. Validator outputs

### Stage validators

```text
STEP 09 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step09_artist_refined.md
VALIDATION PASS on attempt 1/3

STEP 10 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step10_revision_synthesis.md
VALIDATION PASS on attempt 1/3

STEP 11 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step10_final_package_enhanced.md
VALIDATION PASS on attempt 1/3
```

### Pair validator

```text
PAIR VALIDATION START pair_04 in output\daily\2026-08-24-sunna\music
STEP 06 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step06_facets.md
VALIDATION PASS on attempt 1/3
STEP 07 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step07_song_guides.md
VALIDATION PASS on attempt 1/3
STEP 08 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step08_generation.md
VALIDATION PASS on attempt 1/3
STEP 09 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step09_artist_refined.md
VALIDATION PASS on attempt 1/3
STEP 10 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step10_revision_synthesis.md
VALIDATION PASS on attempt 1/3

PAIR VALIDATION PASS
```

### Package validator

```text
PASS output\daily\2026-08-24-sunna\music\pair_04_step10_final_package_enhanced.md
```

### Soundcraft gate

```text
file                                            endRhy  lineRet  w/line   allit  lines
P04V04_FINAL_QA_REPAIR_REPORT.md                 0.371    0.243    4.10   15.33     70
```

### Portfolio gates

```text
extracted 6 pair files: pair_01_step09_artist_refined.md (14178 chars), pair_02_step09_artist_refined.md (16145 chars), pair_03_step09_artist_refined.md (14641 chars), pair_04_step09_artist_refined.md (14050 chars), pair_05_step09_artist_refined.md (12926 chars), pair_06_step09_artist_refined.md (13800 chars)
STEP 09 DISTINCTIVENESS PASS

extracted 24 finalist packages from 6 pair files
PORTFOLIO DISTINCTIVENESS PASS

extracted 24 finalist packages from 6 pair files
PORTFOLIO DISTINCTIVENESS PASS
```

The two portfolio PASS lines are respectively Step 10 and Step 11 with `--expected-total 24`.

## 9. Selected package LYRICS

```text
[Theme: green soda-machine glass and a sweating cup frame an adult concession-counter ask before the movie credits]
[SONG FORM: Two-outburst credits cut — hook-at-zero flare → 14-cell verse → 16-cell outburst one → 10-cell telescope verse → 8-cell low-dembow hinge → 14-cell syllable-compressed outburst two → 4-cell short tail]

[Disc_Rhythm: brisk_ska_pop_offbeat | tight_acoustic_drums | low_dembow_hinge | Center_Mono]
[Disc_Vocal: close_mic_adult_female_high_mezzo | bright_chest_mix | glossy_held_vowels | Center_Front]
[Disc_Sub: melodic_electric_bass | stable_low_pulse | felt_in_chest_not_ears | Mono_Sub_Lock]
[Disc_Pad: short_glossy_plate | low_concession_room | narrow_diamond_doubles | Stereo_Width_Mid]
[Disc_Texture: dry_rising_three_note_horns | biting_guitar_left | small_organ_right | Narrow_Center]

[Hook at Zero - EMO:Eagerness + Marvel - Adult Female Lead - Two horn notes, title at zero]
*horn flare rises*
Mm—too bright to keep (AGAIN)
Again before credits
Green on my mouth
You, take this space
[Verse 1 - EMO:Lust + Amazement - Adult Female Lead - Brisk offbeats, bright chest voice]
Cold cup by my gloss
Your elbow this close
Green glass on my gloss
I pose, then unpose
You order one drink
I steal your red straw
You sip from my drink
I smile over your straw
Screen-light skips on skin
Counter glass glows green
Pretty, I point at the gap
You know this scene
Come close, cross the gap
Come sweeten this scene
[Outburst 1 - EMO:Euphoria + Wonder - Adult Female Lead and Chosen Answer - Seventh lift, full horn flare]
Too bright to keep (AGAIN)
Again before credits
Come take this opening
I made it, so get it
Too bright to keep (AGAIN)
Green mouth, cold cup
Your shoulder slides up
Red straw; your smile tips up
Right here
Ooh-ah, good timing
Screen slips suddenly dark
My lacquered lips keep shining
Again before credits
Come spend the good bit
Too bright to keep (AGAIN)
You fit where I slid it
[Verse 2 Telescope - EMO:Apprehension + Desire - Adult Female Lead - Shorter phrases, guitar thins]
Credits creep close
The cup sweats fast
I almost pretend
Good nights must last
That line is too big
My mini handbag stays small
Want that next nice kiss
Keep your cute calendar
Ask me
I clear the counter for you
[Low-Dembow Hinge - EMO:Defiance + Liberation - Adult Female Lead - Stable bass, sung missing horn pitch]
I slide the cup left
You slide one step in
Red straw points at the gap
Hah—you grin at the gap
Don't promise an encore
Your shoulder fills this space
I choose you right now
You answer, AGAIN
[Outburst 2 - EMO:Ecstasy + Dignity - Adult Female Lead and Chosen Answer - Sung cells contract]
Too bright to keep (AGAIN)
Green on my mouth
Move that cup
You, take this space
Credits roll
Your smile stays hot
Come close, cross the gap
Ask me
I do
You too
Too bright
To keep
Right here
Again (AGAIN)
[Short Tail - EMO:Fulfillment - Adult Female Lead - Bass stop, no fade]
Cup goes left
You step in
Your warm hand meets mine
Ah—again
*after the final vocal, the defining two-note horn riff returns; cup lid pops*
```

## 10. Stop-line compliance

No render, listening verdict, `suno_render` edit, upload, publication, spend, sibling edit, commit, or push occurred.
