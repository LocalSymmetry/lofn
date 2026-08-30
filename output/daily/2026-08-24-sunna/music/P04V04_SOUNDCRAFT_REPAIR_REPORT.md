# P04V04 Soundcraft Repair Report — Again Before Credits

## 1. Scope and provenance

- Binding brief: `output/daily/2026-08-24-sunna/music/SOUNDCRAFT_REPAIR_BRIEFS.md`, P04V04 only.
- Threshold source: `vault/gates.yaml`, return-floor section.
- Execution provenance: `session_key: /root/sunna_p04`; `spawned_by_parent: true`.
- Modified source artifacts: `pair_04_step09_artist_refined.md`, `pair_04_step10_revision_synthesis.md`, and `pair_04_step10_final_package_enhanced.md`.
- Selected variation only: P04V04 `Again Before Credits`.
- No sibling artifact was opened for creative use. Cross-pair validators performed their required read-only portfolio scans.

## 2. Binding metrics

Canonical functions: `scripts/measure_soundcraft.py → sung() → profile()`. The final enhanced V04 lyrics field was isolated before measurement.

| Metric | Before | After | Delta | Repair target | Verdict |
|---|---:|---:|---:|---:|---|
| Strict end-rhyme return | 0.228571 | 0.342857 | +0.114286 | >=0.32 | PASS |
| Exact line return | 0.128571 | 0.242857 | +0.114286 | >=0.22 | PASS |
| Mean words per sung line | 3.971429 | 4.014286 | +0.042857 | <=7.5 | PASS |
| Alliteration per 100 words | 6.115108 | 13.523132 | +7.408024 | >=12.0 | PASS |
| Sung lines | 70 | 70 | 0 | 70–120 target | PASS |

The same repaired lyric in Step 09 and Step 10 measures `0.342857 / 0.257143 / 3.971429 / 13.669065`; Step 11’s functional `Mm—` pickup makes its first title cell intentionally non-exact, yielding the still-passing `0.242857` exact-return score.

Canonical CLI reproduction against this report’s isolated selected-package fixture:

```text
file                                            endRhy  lineRet  w/line   allit  lines
P04V04_SOUNDCRAFT_REPAIR_REPORT.md               0.343    0.243    4.01   13.52     70
```

## 3. Repair mechanics

- **End-sound braids:** local `gloss/gloss`, `drink/drink`, `straw/straw`, `gap/gap`, and `scene/scene` families belong to the cup-and-concession encounter.
- **Listener-owned exact returns:** `Right here`, `Ask me`, `You, take this space`, and `Come close, cross the gap` recur as adoptable actions rather than metric padding.
- **Speakable consonant play:** `cold cup`, `screen-light skips on skin`, `counter glass glows green`, `credits creep close`, and `mini handbag stays small` use room objects, mouth-feel, and flirt action.
- **Contraband protection:** no consent, boundary, ownership, finitude, value, policy, moral-summary, or sung craft-explanation vocabulary was introduced.

## 4. Exact changed sung lines

The following 22 substitutions were applied identically to V04 in Steps 09, 10, and 11:

| # | Before | After |
|---:|---|---|
| 1 | `Cup cold at my palm` | `Cold cup by my gloss` |
| 2 | `I steal one red straw` | `I steal your red straw` |
| 3 | `You watch how I sip` | `You sip from my drink` |
| 4 | `I like what you saw` | `I smile over your straw` |
| 5 | `The screen throws light` | `Screen-light skips on skin` |
| 6 | `The counter stays green` | `Counter glass glows green` |
| 7 | `I point at the gap` | `Pretty, I point at the gap` |
| 8 | `You know what I mean` | `You know this scene` |
| 9 | `Come stand by me` | `Come close, cross the gap` |
| 10 | `Improve this scene` | `Come sweeten this scene` |
| 11 | `Your shoulder comes up` | `Your shoulder slides up` |
| 12 | `Ooh-ah, right here` | `Right here` |
| 13 | `The screen may go dark` | `Screen slips suddenly dark` |
| 14 | `My lips keep on shining` | `My lacquered lips keep shining` |
| 15 | `Credits get near` | `Credits creep close` |
| 16 | `My handbag is small` | `My mini handbag stays small` |
| 17 | `I want your next kiss` | `Want that next nice kiss` |
| 18 | `Keep your whole calendar` | `Keep your cute calendar` |
| 19 | `You wait for my cue` | `Ask me` |
| 20 | `I clear it for you` | `I clear the counter for you` |
| 21 | `Take this spot` | `You, take this space` |
| 22 | `Want me?` | `Come close, cross the gap` |

## 5. Selected-block hashes

Hash definition: SHA-256 of the UTF-8 bytes from the `### Variation 4` heading through the byte immediately before the next global section.

| Artifact | Before bytes / SHA-256 | After bytes / SHA-256 |
|---|---|---|
| Step 09 | `4652` / `83acd9436412bfb86c3bef61846e6982047887a53d33ecc3cbf9816b7f7d0fb2` | `4722` / `add254d45d79bb90d3a063d837ac66fdb0905c4972c964d54ac3f2bc5de7f54c` |
| Step 10 | `4672` / `640aa64c8b2b13006cbd9807aa826923d92993ff1a9e9075520cc00fc2fa4750` | `4742` / `ce7d468186a8129523e6be9cc4397c62976de3b1077505c8c9d8a69cdacdc9a9` |
| Step 11 | `5959` / `afa0d2c213d654dfb91000e2cda1ae09e752893c4f4719c1b7b29a77c7798b82` | `6029` / `2f76a91d14cfa5a7c7d8e64148b7610fa5df6c5afc0b1616a0a7208eee181974` |

## 6. Byte-identity proof for untouched variations

Every V01/V02/V03 hash is identical before and after repair:

| Artifact | Variation | Before SHA-256 | After SHA-256 | Verdict |
|---|---:|---|---|---|
| Step 09 | V01 | `afccb4de5a3e1e3ac04dfe1cdcfeb14c63a717d1b7e86e063dfb1412072796d9` | `afccb4de5a3e1e3ac04dfe1cdcfeb14c63a717d1b7e86e063dfb1412072796d9` | IDENTICAL |
| Step 09 | V02 | `a492c9700cb55b771a23aee5340c6fd763170b367c4b11a2ec79809ced9c9894` | `a492c9700cb55b771a23aee5340c6fd763170b367c4b11a2ec79809ced9c9894` | IDENTICAL |
| Step 09 | V03 | `a06564a8f1dd4b69bb0b3f76d18613e453a1fbc691f3cbea9163b20f440b6729` | `a06564a8f1dd4b69bb0b3f76d18613e453a1fbc691f3cbea9163b20f440b6729` | IDENTICAL |
| Step 10 | V01 | `849c1426f3d84d22f1c4220ac730503725ddf19bb15b8f2674256edbca028792` | `849c1426f3d84d22f1c4220ac730503725ddf19bb15b8f2674256edbca028792` | IDENTICAL |
| Step 10 | V02 | `0173ad3c70380aac05ed4b555bb8458ebbf77f452ad87e0a48436b92ee6cfbe5` | `0173ad3c70380aac05ed4b555bb8458ebbf77f452ad87e0a48436b92ee6cfbe5` | IDENTICAL |
| Step 10 | V03 | `0058a1a9729548ba0aedc355a17525a83f5cd0e1813778347bded66bf2bfd8e9` | `0058a1a9729548ba0aedc355a17525a83f5cd0e1813778347bded66bf2bfd8e9` | IDENTICAL |
| Step 11 | V01 | `e3619afe247e2e1f5d0f2c753f510bf33de6ba63ce05db54b652a4732bdb5e18` | `e3619afe247e2e1f5d0f2c753f510bf33de6ba63ce05db54b652a4732bdb5e18` | IDENTICAL |
| Step 11 | V02 | `b2d705fbf44ae90afe62dc06ec938eca5c9085db602efcae5e37168a09695743` | `b2d705fbf44ae90afe62dc06ec938eca5c9085db602efcae5e37168a09695743` | IDENTICAL |
| Step 11 | V03 | `93f9848abcfee7e6d088ddd0043f13d9345db64af29fed4a91d07deb25dba85d` | `93f9848abcfee7e6d088ddd0043f13d9345db64af29fed4a91d07deb25dba85d` | IDENTICAL |

## 7. Preservation checks

- Soda-machine glass / soda glass: preserved in Theme and material field; `Green glass on my gloss` remains sung.
- Red straw: preserved and made relational in `I steal your red straw` / `I smile over your straw`.
- Concession gap: preserved and strengthened through `Pretty, I point at the gap` and exact `Come close, cross the gap` returns.
- `AGAIN`: preserved as one-word answer, hook response, hinge answer, compressed-return landing and tail release.
- Form: exactly two outburst headers, 16 sung cells in Outburst 1, 14 compressed sung cells in Outburst 2, and a four-cell short tail.
- Temporary-brightness contraband: preserved in the unchanged telescope verse, `Don't promise an encore`, present-tense choice, credits pressure and no-fade completion.
- Music/exclude prompts, title, genre world, low-dembow seam, source fact, lineage and dedication: unchanged.
- Dedication: `320` characters, below the `500`-character ceiling.

## 8. Validator outputs

### Stage validators

```text
STEP 09 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step09_artist_refined.md
VALIDATION PASS on attempt 2/3

STEP 10 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step10_revision_synthesis.md
VALIDATION PASS on attempt 2/3

STEP 11 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step10_final_package_enhanced.md
VALIDATION PASS on attempt 2/3
```

### Pair validator

```text
PAIR VALIDATION START pair_04 in output\daily\2026-08-24-sunna\music
STEP 06 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step06_facets.md
VALIDATION PASS on attempt 2/3
STEP 07 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step07_song_guides.md
VALIDATION PASS on attempt 2/3
STEP 08 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step08_generation.md
VALIDATION PASS on attempt 2/3
STEP 09 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step09_artist_refined.md
VALIDATION PASS on attempt 2/3
STEP 10 PASSED: output\daily\2026-08-24-sunna\music\pair_04_step10_revision_synthesis.md
VALIDATION PASS on attempt 2/3

PAIR VALIDATION PASS
```

### Portfolio distinctiveness

```text
extracted 6 pair files: pair_01_step09_artist_refined.md (14178 chars), pair_02_step09_artist_refined.md (16145 chars), pair_03_step09_artist_refined.md (14641 chars), pair_04_step09_artist_refined.md (14015 chars), pair_05_step09_artist_refined.md (12959 chars), pair_06_step09_artist_refined.md (13752 chars)
STEP 09 DISTINCTIVENESS PASS

extracted 24 finalist packages from 6 pair files
PORTFOLIO DISTINCTIVENESS PASS

extracted 24 finalist packages from 6 pair files
PORTFOLIO DISTINCTIVENESS PASS
```

The two portfolio PASS lines above are respectively Step 10 and Step 11 with `--expected-total 24`.

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
My bass won't shut up
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
The low drum steps in
The horn leaves one bright note
Hah—I sing it and grin
Don't promise an encore
Come occupy this
I choose you right now
You answer, AGAIN
[Outburst 2 - EMO:Ecstasy + Dignity - Adult Female Lead and Chosen Answer - Sung cells contract]
Too bright to keep (AGAIN)
Green on my mouth
Move that cup
You, take this space
Credits roll
Bass stays hot
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
Bass says done
Ah—again
*cup lid pops*
```

## 10. Stop-line compliance

No render, listening verdict, upload, publication, spend, sibling edit, commit, or push occurred.
