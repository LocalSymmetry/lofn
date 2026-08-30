# P05V04 Soundcraft Repair Report — `SCAN ME TWICE`

## Scope and provenance

- Binding brief: `output/daily/2026-08-24-sunna/music/SOUNDCRAFT_REPAIR_BRIEFS.md`
- Threshold source: `vault/gates.yaml`, soundcraft return floors.
- Session: `session_key: /root/sunna_p05`; `spawned_by_parent: true`.
- Selected final variation: P05V04, Backstage Merch Tunnel. In Step 09 this material is source `Variation 2`; in Step 10 it is `Synthesized Prompt 2`; in Step 11 it is final `VARIATION 4`.
- Modified artifacts: Pair 05 Step 09, Step 10, and Step 11 only, plus this report.
- Preserved anchors: tangerine wristband, wrong scan, road-case door, shared password `Wrong door, good timing`, and the `Can you` / `You can` question-permission mirror.
- No prompt, exclude field, title, genre world, seam, lineage, dedication, other variation, render, publication, commit, or push was changed.

## Canonical soundcraft measurement

Measured from the selected lyrics block with the canonical functions in `scripts/measure_soundcraft.py` (`sung()` → `profile()`), using the repository-defined ±4-line window. The selected block was also passed directly through the script's CLI in a transient isolated file; the file was deleted immediately after measurement.

| Metric | Before | After | Delta | Binding target | Verdict |
|---|---:|---:|---:|---:|---|
| Strict end-rhyme return | 0.178082 | 0.506849 | +0.328767 | >= 0.32 | PASS |
| Exact line return | 0.150685 | 0.273973 | +0.123288 | >= 0.22 | PASS |
| Mean words per sung line | 4.287671 | 4.219178 | -0.068493 | <= 7.5 | PASS |
| Alliteration per 100 words | 11.182109 | 12.337662 | +1.155553 | >= 12.0 | PASS |
| Sung lines | 73 | 73 | 0 | preserve full song | PASS |

```text
file                                            endRhy  lineRet  w/line   allit  lines
tmpic6jmkvz.md                                   0.507    0.274    4.22   12.34     73
```

The rhyme return comes from local room-specific families (`door/door`, `cart/heart`, `time/time`, `away/away`, `paper/paper`, `wrong/wrong`, `hand/stand`, `door/floor`, `yet/yet`, `nice/twice`, `timing/missing/swinging`). Exact return comes from listener-owned phrases and existing song objects: `Meet me there`, `You can wait`, `Tangerine`, and the shared password. No generic chant or explanatory thesis was added.

## Exact changed lyric lines

These same lyric edits were propagated into the selected block in Steps 09, 10, and 11.

| # | Before | After | Craft job |
|---:|---|---|---|
| 1 | `Mm-mm, strong arms` | `Mm-mm, hold that door` | Road-case call/answer and local `door` return. |
| 2 | `Your shoulder blocks the cart` | `Your shoulder stalls the cart` | Speakable shoulder/stalls consonance. |
| 3 | `Encore tangerine, it says` | `The wristband whispers tangerine` | Keeps the wristband/color anchor and adds wristband/whispers consonance. |
| 4 | `That shade is very loud` | `That loud shade starts my heart` | Creates the bodily `cart/heart` local family. |
| 5 | `I wear it better laughing` | `I wear the loud; you laugh on time` | Keeps appetite and sets up `time/time`. |
| 6 | `Can you hold a stare?` | `Can you hold a hard stare?` | Preserves the question mirror with hold/hard mouth-play. |
| 7 | `Can you miss a painted line?` | `Can you miss my painted line?` | Preserves the question while adding miss/my consonance. |
| 8 | `Answer with your feet` | `Answer with both bright feet` | Makes the physical answer more rhythmic through both/bright. |
| 9 | `The second scan goes sideways` | `The second scan slides away` | Preserves the wrong scan and links `away/away`. |
| 10 | `Orange loses interest` | `Orange slips off paper` | Completes the local `paper/paper` braid without explaining the refusal. |
| 11 | `(Wrong together)` | `(Still wrong)` | Returns the sung `wrong` ending and keeps the comic error live. |
| 12 | `You make mistakes look useful` | `You make mistakes look nice` | Flirtier surface; `nice/twice` return replaces utility language. |
| 13 | `I turn us from the reader` | `I turn us past the stand` | Preserves the bodily turn and links `hand/stand`. |
| 14 | `Good view from here` | `Meet me there` | First exact listener-owned return. |
| 15 | `You can cross the tape with me` | `You can cross the floor` | Preserves the permission mirror and links `door/floor`. |
| 16 | `You can wait for what comes next` | `You can wait, but not yet` | Keeps appetite suspended and links `yet/yet`. |
| 17 | `Your mouth knows when to wait` | `You can wait` | Exact permission return, still addressed to the adult `you`. |
| 18 | `Tangerine can keep scrolling` | `Tangerine` | Lets the absurd result return as a compact sung object. |
| 19 | `Come close when I say closer` | `Meet me there` | Returns her chosen invitation in the final detonation. |
| 20 | `Stay close when I say stay` | `Stay close; play nice` | Keeps playful terms while linking `nice/twice`. |
| 21 | `Wrong door` | `Wrong door, good timing` | Returns the shared password intact rather than splitting it into fragments. |

## Selected-block hash proof

SHA-256 covers the complete selected variation block in each artifact, not lyrics alone.

| Artifact / selected block | Before | After |
|---|---|---|
| Step 09 / source Variation 2 | `17f75f90fd99a6980045732e92a5991d882bbacd35515320342d8851dec75859` | `66f88ef445699a947d5f82a2ad08262580cf516b550617679748cc5c016e56e9` |
| Step 10 / Synthesized Prompt 2 | `9324562837cebda7ecb2259a8c9edaad413a7700d3096003c32726abac127e78` | `444f9ae485a6319cc306034d1994812f04e6c268ba5395594708a5d6a443ec7a` |
| Step 11 / final V04 | `0e43002bab6133403f6f97cee1779cf250f1ee69145d98b67f4391d09f703cbe` | `be0563368962c5b0fda0ec7ff511c33f501cfa9ad415d523ed898501324aa53b` |

The Step 11 after-hash also covers the corrected lyrics-field count and explicit soundcraft measurement in V04's Required Checks.

## Byte-identity proof for untouched final V01/V02/V03

The non-selected complete variation blocks were hashed before and after repair. Each equality below is exact.

| Artifact | Final variation / source mapping | Before SHA-256 | After SHA-256 | Verdict |
|---|---|---|---|---|
| Step 09 | final V01 / source V4 showroom | `0808e3708b191316dd72359c5c3d5b75018531dac98adb050335b3aa49048df6` | `0808e3708b191316dd72359c5c3d5b75018531dac98adb050335b3aa49048df6` | IDENTICAL |
| Step 09 | final V02 / source V1 trade booth | `86413e17d3e55146cc3c5e88e8e28e223cbfa57dfff6840f5ef00e3a31156ad0` | `86413e17d3e55146cc3c5e88e8e28e223cbfa57dfff6840f5ef00e3a31156ad0` | IDENTICAL |
| Step 09 | final V03 / source V3 portrait studio | `086a6bce685ed7806e5de78da3df561c31e93cfe2f9e44616e78502ee9710a28` | `086a6bce685ed7806e5de78da3df561c31e93cfe2f9e44616e78502ee9710a28` | IDENTICAL |
| Step 10 | final V01 | `66852909a20e6c9f65f69347c12c6c1c5db070f5122ddf994fca37dbcf83f8ff` | `66852909a20e6c9f65f69347c12c6c1c5db070f5122ddf994fca37dbcf83f8ff` | IDENTICAL |
| Step 10 | final V02 | `1e0220cf176645df82467992ea4e5372071d56b6732f1352ff79b8634fb79e87` | `1e0220cf176645df82467992ea4e5372071d56b6732f1352ff79b8634fb79e87` | IDENTICAL |
| Step 10 | final V03 | `3dbc297081485be24bbcae9ca9e948f8c305c06ccda33111f835f86bce7ae00d` | `3dbc297081485be24bbcae9ca9e948f8c305c06ccda33111f835f86bce7ae00d` | IDENTICAL |
| Step 11 | final V01 | `ef9b68cb2e9fdb1611f9f84b96fbe679868e09a9545f0d7cb205472cfb245740` | `ef9b68cb2e9fdb1611f9f84b96fbe679868e09a9545f0d7cb205472cfb245740` | IDENTICAL |
| Step 11 | final V02 | `8070c497bac66d99cbae59d3e0b31f1fadf47ed00770973aa5b76dbf4bd5771b` | `8070c497bac66d99cbae59d3e0b31f1fadf47ed00770973aa5b76dbf4bd5771b` | IDENTICAL |
| Step 11 | final V03 | `3f960e56fc01cd83ddb529eb891faf7a79d96031a7cf6a519addcef3f762a82f` | `3f960e56fc01cd83ddb529eb891faf7a79d96031a7cf6a519addcef3f762a82f` | IDENTICAL |

## Package measurements

- Step 09 selected lyrics field: `3,016` characters.
- Step 10 selected lyrics field: `2,985` characters.
- Step 11 selected lyrics field: `3,512` characters, below the `<5000` hard cap.
- Step 11 dedication: `335` characters, below the `<=500` requirement.
- Step 11 prompt/exclude: unchanged at `932` / `618` characters.
- SFX: unchanged at `1`; sung lines unchanged at `73`.

## Validator transcript

### Step validators and pair audit

```text
STEP 09 PASSED: output\daily\2026-08-24-sunna\music\pair_05_step09_artist_refined.md
VALIDATION PASS on attempt 1/3
STEP 10 PASSED: output\daily\2026-08-24-sunna\music\pair_05_step10_revision_synthesis.md
VALIDATION PASS on attempt 1/3
STEP 11 PASSED: output\daily\2026-08-24-sunna\music\pair_05_step10_final_package_enhanced.md
VALIDATION PASS on attempt 1/3
PAIR VALIDATION PASS
```

The pair audit also revalidated Steps 06, 07, 08, 09, and 10 as `PASSED` on attempt 1/3.

### Portfolio distinctiveness

```text
STEP 09 DISTINCTIVENESS PASS
PORTFOLIO DISTINCTIVENESS PASS   # --step 10, 24 finalist packages
PORTFOLIO DISTINCTIVENESS PASS   # --step 11, 24 finalist packages
```

## Final verdict

`PASS — BINDING SOUNDCRAFT REPAIR COMPLETE.` The selected song clears every repair target with margin, retains its funny/appetitive refusal and shared-joke action, and leaves P05 V01/V02/V03 byte-identical.
