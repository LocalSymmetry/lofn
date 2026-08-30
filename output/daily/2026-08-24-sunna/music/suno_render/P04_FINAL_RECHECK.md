# P04 Final Recheck — Again Before Credits

**Audit date:** 2026-08-24  
**Candidate:** P04V04  
**Scope:** repaired P04V04 blocks in Steps 09, 10 and 11 plus `P04V04_FINAL_QA_REPAIR_REPORT.md`; text only. No render, listening claim, publication, commit, push, or spend.

## Final verdicts

- **Original craft-narration blocker:** **FULLY REPAIRED.** The sung bass/horn/procedure lines remain absent and the hinge is carried by cup, straw, gap, shoulder, choice and answer.
- **Sunna R4 HUM-test blocker:** **FULLY REPAIRED.** The defining horn riff is now explicitly instructed to return after the final vocal in Steps 09, 10 and 11.
- **Pipeline Integrity:** **PASS within this focused recheck scope.** Fresh Step 09/10/11 validators pass and the reported selected-block hashes reproduce exactly.
- **Package:** **PASS.** Prompt, exclude, lyrics, soundcraft, SFX, title, dedication and lineage remain valid.
- **Human Subject:** **CLEAR.** The singer and addressed adult remain anonymous/invented and consent-legible; real people occur only in credit/source text.
- **Overall:** **SHIP — repaired canonical text source only.** No text blocker remains. This is not a render or listening verdict.

## Derived-menu status

The canonical repaired Step-11 source ships. The current `suno_render/04_again_before_credits.md`, `build_manifest.json`, and the frozen P04 hash in `build_suno_render_menu.py` still identify the pre-HUM-repair source (`6067b2d7…`). The coordinator must update the frozen hash to the current canonical LF hash `ca5c8b5d…`, rebuild the menu, and run the package validator on the rebuilt file before that derived menu is used. This is expected propagation work, not a creative repair or a reason to reopen P04.

## HUM-test proof

The final-vocal ordering is unambiguous in all three current blocks:

- **Step 09 MUSIC PROMPT:** `the defining horn riff returns after final vocal, then lid pops.`
- **Step 10 MUSIC PROMPT:** `the defining horn riff returns after final vocal, then lid pops.`
- **Step 11 MUSIC PROMPT:** `the defining horn riff returns after final vocal, then lid pops.`
- **Steps 09/10 lyric tail:** final sung `Again`, then `*after the final vocal, the defining two-note horn riff returns; cup lid pops*`.
- **Step 11 lyric tail:** final sung `Ah—again`, then the same post-vocal instrumental command.

This closes Sunna R4's hard HUM TEST: the two-note instrumental figure appears before the first vocal, is transformed through the missing third pitch at the hinge, and returns after the last vocal. The return is a production/SFX instruction, not something the human singer narrates. It creates neither a third outburst nor an encore.

The terminal instruction is longer than the house's short-SFX example, but it is not sung procedure debris and does not violate the package contract: the lyric still contains the short standalone `*horn flare rises*` cue that satisfies the <=5-word SFX requirement, while the longer final command exists to make the after-vocal ordering generator-legible.

## Preservation of the earlier repair

The human causal chain remains intact:

`cup slides left -> chosen adult steps in -> straw points at gap -> chosen adult grins -> shoulder fills space -> she chooses now -> he answers AGAIN -> warm hands meet`.

None of the previously blocked lines returns: `My bass won't shut up`, `The low drum steps in`, `The horn leaves one bright note`, `I/Hah—I sing it and grin`, `Come occupy this`, `Bass stays hot`, and `Bass says done` are absent from the current sung body. If Suno smooths the missing hinge pitch, the relational event still completes.

## Fresh deterministic evidence

Fresh results:

```text
STEP 09 PASSED: pair_04_step09_artist_refined.md
STEP 10 PASSED: pair_04_step10_revision_synthesis.md
STEP 11 PASSED: pair_04_step10_final_package_enhanced.md
PASS pair_04_step10_final_package_enhanced.md
```

Fresh soundcraft measurement of the isolated selected lyric in the repair report:

```text
endRhy 0.371 · lineRet 0.243 · words/line 4.10 · alliteration/100w 15.33 · sung lines 70
```

| Artifact | Current selected-block bytes | Current raw SHA-256 | Report match |
|---|---:|---|---|
| Step 09 P04V04 | 6690 | `e0a1d54d945a7028c47200776664bcaf67c9ae933a36b8854dab95b87bff35dd` | MATCH |
| Step 10 P04V04 | 10399 | `89b83cf13e3bcfa7abbcf1b133c2311dad41b55f0f348d478f1a6952614a1381` | MATCH |
| Step 11 P04V04 | 10383 | `f974cd211a483040326a287f251a5450a3c4e26f6bd717ebda5226d5490d4ff1` | MATCH |
| Step 11, CRLF normalized to LF | 10252 | `ca5c8b5d3353eded81dbe13043d762b83c69c1ec066b839088abe5ba0ae3571d` | MATCH |

| Metric | Current measurement | Gate | Result |
|---|---:|---:|---|
| MUSIC PROMPT | 984 chars | 850–1000; boundary FLAG at >=985 | PASS, no prompt-boundary FLAG |
| EXCLUDE PROMPT | 702 chars | <=1000 | PASS |
| Lyrics field | 3044 chars after newline normalization | <5000; target <=4800 | PASS |
| Sung lines | 70 | 70–120 | PASS + floor FLAG only |
| Dedication | 320 chars | <=500 | PASS |
| Strict end-rhyme return | 0.371 | >=0.30 | PASS |
| Exact line return | 0.243 | >=0.20 | PASS |
| Mean words/line | 4.10 | <=7.5 | PASS |
| Alliteration/100 words | 15.33 | >=11 | PASS |

The prompt stays a complete dense paragraph with terminal punctuation and no real artist name. The lyrics retain `[Theme:]`, `[SONG FORM:]`, seven precise taxonomy EMO headers, two SFX commands, 70 sung lines, and no sung digits or arrangement narration.

## Dedication and lineage

The dedication remains 320 characters and truthfully ends `full credits below.` Lloyd Brevett remains first; Jamaican ska and the Jamaican-diaspora/Panamanian/Puerto Rican dembow route remain credited; the Comet 220P observation debt remains named. The Lineage & Credit block distinguishes borrowed grammar/fact from generated sound and explicitly disclaims samples, interpolation, imitation, scene-origin claims, and real private biography. No named-break/player debt was introduced by the horn-return repair.

## Somatic close

- **Singer-Surface Skeptic:** YES — immediate adult flirt, body/place/object, six-syllable return.
- **Remark-Test Skeptic:** YES — the singer never explains the horn mechanics; the human event owns the hinge and tail.
- **Generator-Survival Skeptic:** YES — both positive prompt and terminal command specify the post-vocal return, while the song's meaning survives if the hinge handoff is smoothed.

**Bloc: 3 YES / 0 NO — PASS.** No named corpse, repetition collapse, new procedural leak, dedication defect, or additional text blocker was found.

## Final recommendation

**SHIP the repaired P04V04 canonical text source.** Coordinator propagation into the derived Suno menu remains required; after rebuild, this text package is ready for The Scientist's render authorization and eventual ear gate. No audio has been rendered or heard.
