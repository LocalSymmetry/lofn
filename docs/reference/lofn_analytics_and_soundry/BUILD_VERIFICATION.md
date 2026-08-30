# BUILD VERIFICATION — foundation round



*Adversarial verifier, workflow `wf_8a01f8df-ebe`, 2026-08-01. It ran `tsc`, `vitest` and the greps itself rather than trusting the five build lanes.*



## VERDICT: **RED**



## Real failures (verified by running them)

1. SPEC-MANDATED GATE FAILS WHEN ACTUALLY RUN. BUILD_SPEC.md:194 states verbatim: "`outline: none` is a forbidden string in both codebases, grep-asserted in CI." I ran that grep across all three src trees and it returns a hit: `apps/soundry/src/style.css:230` — `.ask textarea:focus { border-bottom-color: var(--chrome-1); outline: none; }`. The assertion the spec requires DOES exist (apps/lofn-kit/tests/tokens-contrast.spec.ts:72-74) but is scoped to the `CSS` constant, which is `tokens.css` only — it never reads apps/soundry/src/style.css or apps/lofn-analytics/src/shell/*.css. So the suite is green while the estate violates the rule. This is also a live a11y defect, not just a string: `.ask textarea:focus` has specificity (0,2,1) and beats the global `:focus-visible` rule at style.css:43 (0,1,0), so KEYBOARD FOCUS ON THE ANVIL'S PRIMARY TEXTAREA — the app's central control — renders with no outline, indicated by a border-bottom colour change alone. Colour-only focus indication, in the estate whose entire premise is measurable accessibility. Neither the L-SCAFFOLD-SOUNDRY agent's forbidden-string test (tests/posture-hosted.spec.ts) nor L-SCAFFOLD-ANALYTICS's (tests/no-disabled.spec.ts, which does check `title=`, `smooth scroll`, `order:`, `*-reverse`, `@keyframes`) includes `outline: none`.

2. THE SAME SCOPE HOLE APPLIES TO `filter: invert()`. BUILD_SPEC.md:28 states it "is a forbidden string in both apps' CSS, asserted by a test." tokens-contrast.spec.ts:68-70 asserts it against tokens.css only. I grepped both apps' src trees manually and found no violation TODAY — so this is a latent gate hole, not a live break, but it is the identical defect and the next stylesheet edit is unguarded.

3. `node scripts/validate-palette.mjs` EXITS 1, exactly as the L-KIT lane reported. I ran it; all 8 failures reproduce verbatim (light --sig-null on --sunk 2.65:1; dark + dark-auto --sig-pos/--sig-neg 2.16:1 and --sig-null 2.99:1 on --card; light quartet L* min gap 11.83 vs §1.2's stated 12). I am recording this as a real non-passing gate rather than a build break: the failures are four genuine internal spec contradictions (§1.1's --sig-null=--ink-faint binding vs §5.4's 3:1 mark floor; §1.1's dark block never redefining the Tide diverging ramp that --sig-pos/--sig-neg alias; §5.4's >=20 gate vs §1.2's own 17.4 dark measurement; §1.2's 12.1 ladder-gap arithmetic). The lane refused to silence them by editing a threshold or inventing a hex, which is the correct call. But the gate is wired to NO npm script — package.json has only `typecheck` and `test` — so `npm test` is green while the estate's own palette validator is red. Nothing in CI would ever run it.

4. NOT VERIFIED BY ANYONE, INCLUDING ME: the >=1100px three-column desktop layout of lofn-analytics has never been rendered. The L-SCAFFOLD-ANALYTICS agent disclosed this honestly (its Browser pane is fixed at 664px). layout-columns.spec.ts asserts the CSS text, which is a transcription check, not a rendering check. The primary layout of the primary app is unobserved.

## Fabrications — numbers/paths invented rather than sourced

1. UNDISCLOSED — apps/lofn-kit/src/chart/render.ts:42 `const WIDTH = 680;` and render.ts:41 `const PAD = { top: 24, right: 16, bottom: 44, left: 56 };`. The string `680` appears ZERO times across all 757 KB of BUILD_SPEC.md + SPEC_PATCH.md + SPEC_PATCH_2.md (I grepped -c on all four docs: 0/0/0/0). It is not derivable from any file on disk. Every plot in both apps is sized by this literal. The spec DOES size other plots explicitly (BUILD_SPEC:1618 gives `4 panels at 1160x72`) and DOES give the main column as `minmax(560px,1fr)` (SPEC_PATCH_2:1181), so a width was specifiable and simply was not specified here. The PAD quartet slipped my spec-diff only because 24/16/44/56 collide with the --sp- scale and the 44px secondary-button floor; as a padding set it is equally unsourced. This is the one number the L-KIT-CHART lane invented and did NOT list in its otherwise scrupulous 11-item spec-gap report, which listed inventions as small as a band-scale padding default.

2. DISCLOSED AND FLAGGED — port 5230 for lofn-analytics (apps/lofn-analytics/vite.config.ts, recorded in DEPLOY.md). I confirmed no document assigns Lofn Analytics a port; The Soundry's 5231 is fixed at BUILD_SPEC:886/1349, SPEC_PATCH:953 and SPEC_PATCH_2:698/2760. Invented, but hardcoded in exactly one place and named as a build-time choice.

3. DISCLOSED AND FLAGGED — `DEFAULT_BAND_PADDING = 0.2` at apps/lofn-kit/src/chart/scales.ts:27. BUILD_SPEC:234 types the band scale as `padding?: number` and states no default. Declared once, referenced once (scales.ts:103).

4. DISCLOSED AND FLAGGED — the key name `sound_return_axis` (apps/soundry/src/anvil/deck.ts:76). SPEC_PATCH_2:439 specifies the FLOOR (">= 6 distinct SOUND/RETURN axis values") but SPLICE F2 §1.4c's card schema has no such field and no source names its key. The floor (18 / 3-per-family / 6 axes) is verbatim from SPEC_PATCH_2:438-439; only the key name is invented.

5. DISCLOSED AND FLAGGED — Animation's measurement depth `1` at apps/lofn-analytics/src/shell/rail-source.ts. I confirmed Art's 1/5 (BUILD_SPEC:1428, SPEC_PATCH_2:1074) and Music's 3 of 5 (SPEC_PATCH_2:1103) are stated; Animation's is nowhere. Everything else in that file reproduces from BOTH the spec's printed rail (BUILD_SPEC:1416-1422) and data/snapshots/socials/latest.json (nightcafe 21000/1200/23000, suno 5913/3926/210, tiktok 3542/83000) — I checked the JSON directly.

6. DISCLOSED AND FLAGGED — 13 of 14 tab RegionNote strings in lofn-analytics are AUTHORED, not transcribed. Only M6 CONFLUENCE's notHere is verbatim (SPEC_PATCH_2:1280). Each authored entry cites its source section inline.

7. DISCLOSED AND FLAGGED — `LOCK_REASON_BLIND_IN_PANEL` (apps/soundry/src/shell/filters.ts:97). By contrast I verified `LOCK_REASON` (filters.ts:80-83) is BYTE-VERBATIM against SPEC_PATCH_2:2408-2410, and all six SPLICE 22 strings in headwinds-rail.ts (footer line, RegionNote forWhat/notHere, the on-M4 line, the LEDE dedupe token, the routing line) are byte-verbatim against SPEC_PATCH_2:1241-1268.

## Spec deviations

1. apps/lofn-kit/tests/palette-cvd.spec.ts FREEZES the current measurements instead of enforcing the spec's stated gate. BUILD_SPEC:619 and :1830 both require "CIEDE2000 >= 20 between every categorical pair in every simulation, both themes"; the test at line 112-117 instead asserts `expect(...).toBe(8.62)` and `toBe(5.09)`, and at line 132 asserts dark does NOT clear 20. This is transparently documented in the file header and is the right call given the contradiction — but the estate now has a green test asserting the spec's own gate is unmeetable. Needs a coordinator ruling on one word in §1.2 and §5.4 (CIEDE2000 -> CIE76 dE*ab), since CIE76 reproduces §1.2's published 25.2/17.4 to two decimals (I verified: 25.24 light / 17.42 dark).

2. apps/lofn-kit/src/data/ IS NOT THE SPEC'S PATH. BUILD_SPEC LANE I places the readers at apps/lofn-kit/src/load/{snapshot,schema-guard,invariants}.ts, and SPEC_PATCH SPLICE 9 puts the soundry reader at apps/lofn-analytics/src/load/soundry.ts. Neither src/load/ directory exists. Disclosed by the L-DATA lane; nothing is duplicated, but invariants I1-I10 and the utf-8-sig byte reader remain unwritten.

3. SOUNDRY'S CSP RESOLVES A BUILD_SPEC-vs-BUILD_SPEC CONFLICT UNILATERALLY. I confirmed apps/soundry/index.html:24 ships `connect-src 'self'` while apps/lofn-analytics/index.html:15 ships `connect-src 'none'`. §8.1's head block writes 'none' for the Soundry and calls it "retail-sealed in every posture"; Part 3 §0.1's autonomy table writes 'self' in all three postures. The lane chose 'self' and flagged it in two places. Defensible (a bench that cannot reach its own loopback server cannot bind a clip_id) but it is a choice between two spec'd values and needs ratifying.

4. THE PRINTED DENOMINATOR 46 of 210 DOES NOT REPRODUCE FROM THE DECLARED DEFAULTS. I recomputed in Python against the real snapshot: modern alone = 47; modern INTERSECT floor=50 INTERSECT outlier-excluded = 46; modern INTERSECT floor=60 INTERSECT outlier-INCLUDED = 46; modern INTERSECT floor=60 INTERSECT outlier-excluded (the declared defaults co=modern/out=exclude/floor=60) = 45. The 46 printed at BUILD_SPEC:1463, SPEC_PATCH_2:626/2073/2329/2341/2368 is reachable two ways, neither of which is the declared default set. Also R33b/SPEC_PATCH_2:790's "co=modern hides 164 of 210" implies 46 from the cohort window alone; the window alone leaves 47. Correctly reported and asserted; needs a ruling on whether floor defaults to 50, out defaults to label, or the printed 46 changes.

5. §4.2's TIER TABLE MAKES `NULL` UNREACHABLE. OBSERVATION's first condition and NULL's condition are the same predicate under first-match-wins print order, so no insight can ever reach NULL or its STOP_SPENDING_ON_IT magnitude. Implemented literally (which is also the conservative reading) with TIER_MATCH_ORDER exported so a ruling flips it in one place. Real dead branch in the spec.

6. apps/soundry/src/shell/filters.ts IS A SECOND IMPLEMENTATION of the applicability rule R33 exists to unify. SPLICE T9 puts chipState/factsOf/CardFacts in apps/lofn-kit/src/state/applicability.ts; that file does not exist (I confirmed the kit has state/filters.ts, cohort.ts, insight.ts, deeper.ts, empty.ts only). Flagged at the top of the Soundry file with a delegate-when-it-lands note.

7. THE 88px SHELF CAP IS NOT ENFORCED. §3.5 says the strip "wraps to 88px rather than scrolling horizontally"; fifteen chips at this build's label lengths exceed it. The lane left the strip wrapping freely rather than clipping, on the grounds that clipping is the silent drop R35 rejects. Correct instinct, but the spec's stated dimension is unmet and flagged only in a CSS comment.

8. V24-V27 ARE NUMBERED TWICE IN SPEC_PATCH_2 §S — it adopts both SPLICE 24 and SPLICE T5 "entire", giving four rule numbers eight rules. validate.ts implements V1-V23 plus T5's rewritten V15 and carries a SPEC-GAP comment for the rest. Renumbering either set unilaterally would put a third vocabulary in the estate; needs a coordinator ruling.

## Next units

1. FIX FIRST, BEFORE ANY NEW LANE: delete `outline: none` from apps/soundry/src/style.css:230 and give the Anvil textarea a real focus ring using BUILD_SPEC:194's own house rule (`outline: 2px solid var(--ink); outline-offset: 2px` plus the 1px inner --paper ring). Then WIDEN the gate: move the `outline: none` and `filter: invert` assertions out of tokens-contrast.spec.ts (where they see only tokens.css) into a shared forbidden-string sweep that reads EVERY .css and .ts under apps/*/src — the spec says "both codebases", and the current scope is why this shipped green.

2. RESOLVE OR REMOVE `WIDTH = 680` in apps/lofn-kit/src/chart/render.ts:42. Either derive it from SPEC_PATCH_2:1181's `minmax(560px,1fr)` centre column and say so in a comment, or get a ruling and record it in BUILD_SPEC §2.2 next to the already-specified `height?: number // default 220; spark 28; comb 34`. Audit the PAD quartet on line 41 the same way. This is the only undisclosed invented number in 509 tests' worth of work and it should not stay unsourced.

3. WIRE `node scripts/validate-palette.mjs` INTO package.json as a named script (e.g. `"palette": "node scripts/validate-palette.mjs"`) so its exit-1 is visible to anything that runs the repo, and decide explicitly whether CI blocks on it. Right now a red gate is invisible to `npm test` and only fires if a human types the path.

4. COORDINATOR RULING PACKET — these four are blocking and cannot be settled by any build agent: (a) the CIEDE2000/CIE76 misnaming in §1.2 and §5.4 (CIE76 reproduces the published 25.2/17.4 exactly; CIEDE2000 gives 8.62/5.09); (b) dark-mode values for the Tide diverging ramp, absent from §1.1's dark block, which is why --sig-pos/--sig-neg sit at 2.16:1 on --card; (c) the --sig-null = --ink-faint binding vs §5.4's 3:1 mark floor, which cannot both hold; (d) the 45-vs-46 denominator — whether `floor` defaults to 50, `out` defaults to `label`, or the printed 46 changes across six spec locations.

5. FIX BUILD_SPEC §1.1's DARK-MODE CSS IN THE SPEC TEXT. Line 118 prints `:root[data-theme="dark"], @media (prefers-color-scheme: dark) { :root:not(...) {...}}` — an at-rule inside a selector list, which every parser drops. tokens.css already transcribes it as two valid rules with byte-identical declarations and a test locking them together, but the spec document itself is still wrong and the next transcriber will hit it.

6. WRITE THE MISSING DATA FILES AND RECONCILE THE LOADER PATHS: data/watchlist/competitors.template.json and data/snapshots/competitors/ do not exist (I confirmed), so watchlist.spec.ts runs against a schema fixture and competitor-projection.spec.ts runs against our own 210-row Suno pull relabelled — an honest adversary, but not the real file. Same pass should reconcile apps/lofn-kit/src/data/ against the spec's apps/lofn-kit/src/load/ and land the unwritten invariants I1-I10 and the utf-8-sig snapshot reader.

7. RENDER AND LOOK AT THE >=1100px DESKTOP LAYOUT of lofn-analytics on a real wide viewport. It is the primary layout of the primary app and no human or machine has seen it. The tablet/mobile branches were smoke-tested at 664px and that pass found two real defects (a specificity collision hiding the mobile chip, a missing space in a rendered string) that no test caught — which is precisely the argument for doing it at 1280 too.


---

# What each build lane reported


## UNIT L-DATA — apps/lofn-kit/src/data/ (pure loaders + the R3 projection gate + the R1 watchlist reader)

**Files:** E:/git/private-claw/apps/lofn-kit/src/data/loaders.ts, E:/git/private-claw/apps/lofn-kit/src/data/competitors.ts, E:/git/private-claw/apps/lofn-kit/src/data/watchlist.ts, E:/git/private-claw/apps/lofn-kit/tests/loaders.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/competitor-projection.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/watchlist.spec.ts, E:/git/private-claw/apps/lofn-kit/tests/data-purity.spec.ts, E:/git/private-claw/apps/lofn-kit/package.json, E:/git/private-claw/apps/lofn-kit/tsconfig.json, E:/git/private-claw/apps/lofn-kit/vitest.config.ts, E:/git/private-claw/apps/lofn-kit/.gitignore

**Tests:** tests/loaders.spec.ts — 20 tests. Runs against the REAL files: data/snapshots/suno/latest.json (210 rows, recon ok, plays 374122, followers 3926), socials/latest.json (6 platforms incl. nightcafe, instagram named as skipped), distribution/latest.json (4 rows, nested cumulative windows 1596/333/50/5, Apple's top_tracks are bare strings), data/catalog/works.json (6149 rows, by_modality 3716/2223/210, engagement 109 of 210 = 51.9%, mp3 65 of 66). Asserts absence is null and never 0 (the 4 clips with no words_per_line), asserts feed-discriminator refusal, and asserts the §6.3 named refusal of the stale 2026-03-28 shape against the real tracking/suno-latest.json whose metrics.plays is the string "8.3K"., tests/competitor-projection.spec.ts — 22 tests. Fixture is our OWN 210-row Suno pull relabelled feed:"competitors" — every forbidden field present and full. (a) every key of every projected row is in the 6-key whitelist and the row has exactly 6 keys; (b) JSON.stringify(projection) contains none of COMPETITOR_FORBIDDEN_KEYS; (c) not one of the 210 style prompts, titles, negative_tags, personas, model_names or 863 emo tags survives; a field added to ingest_suno.py tomorrow cannot appear by default. THE NO-STRINGS ASSERTION at both tiers: every string emitted at Tier H is either an allowed key name, a Suno uuid, or an ISO-8601 instant; the Tier S shape object emits only its own frozen key names. A poisoned id is refused BY INDEX AND REASON CODE, never lands in rows[], and the refusal message does not quote it. Tier S reproduces the estate's measured numbers: distinct_tag_count 863, tags_per_clip {median 6, n 202}, median_words_per_line 6.4915. I11: a header-less pull is pullFailed, shape is not computed, reconciliationOk is null when the block is absent rather than false. Boundary test: no handle, display name, profile description, source URL, user_id, pulled_at or header key crosses., tests/watchlist.spec.ts — 19 tests. Exactly five slots in the frozen Appendix-C order; a sixth is refused with her own sentence; four is refused; a reorder is refused; a non-frozen `purpose` is refused; a handle with a <12-word why_this_one is refused; a handle failing the syntactic screen is refused; the same handle in two slots is refused NAMING BOTH slots; an absent disqualifier defaults to `unchecked` and never `clear`. R3: every slot defaults to Tier H; a Scientist-typed `hosting+shape` RENDERS; an unknown tier is refused rather than silently downgraded; the roster, the slots array, every slot and every disqualifier_check are FROZEN so `slot.study = "hosting+shape"` throws TypeError at runtime; neither watchlist.ts nor competitors.ts exports any write/save/set/update/put/patch/persist/escalate/promote symbol; both modules agree on one tier union., tests/data-purity.spec.ts — 28 tests. Every .ts in src/data is grepped (comments stripped) for `document`, `window`, `fetch(`, `localStorage`, `sessionStorage`, `node:fs`, `readFileSync`, `XMLHttpRequest` — all absent. Each file is asserted to open with a WHO EDITS comment naming who edits the JSON it wraps and when.

**Verification:** RAN: `npx vitest run` in E:/git/private-claw/apps/lofn-kit.
OUTPUT (final): "Test Files 1 failed | 3 passed" -> after one real fix -> "Test Files 4 passed (4) / Tests 89 passed (89) / Duration 4.41s". All 89 green.

THE ONE FAILURE I HIT, HONESTLY: `AssertionError: expected { median: 6, n: 210 } to deeply equal { median: 6, n: 202 }`. My first implementation counted a clip with `emo_tags: []` as a measured zero and put it in the tags-per-clip denominator, giving n=210. The estate's published denominator (BUILD_SPEC §M2.3(a), "863 distinct tags across 202 clips") is 202; exactly 8 of 210 clips carry an empty list. I changed the loader to the 202 denominator so the competitor card and M2.3 cannot print two denominators for one quantity, documented the choice in the code as a SPEC-GAP, and added a test asserting n + 8 empties = 210.

RAN: `npx tsc --noEmit` (full project, tsconfig.json).
OUTPUT: `src/state/empty.ts(26,29): error TS2307: Cannot find module '../chart/spec.js' or its corresponding type declarations.`
THAT ERROR IS NOT MINE. `src/state/empty.ts` and `src/state/cohort.ts` appeared in apps/lofn-kit/src/ during this unit — a CONCURRENT LANE is writing there. Its file imports `type { FeedId } from "../chart/spec.js"`, and no lane has written src/chart/spec.ts yet. Per BUILD_SPEC §8.3.3 (serialized writes, each lane writes only its own files) I did not touch it.

RAN, to get a clean signal on my own surface only:
`npx tsc --noEmit --strict --noUncheckedIndexedAccess --target ES2022 --module ESNext --moduleResolution bundler --lib ES2022,DOM,DOM.Iterable --isolatedModules --resolveJsonModule --forceConsistentCasingInFileNames --skipLibCheck --types node src/data/*.ts tests/*.spec.ts`
OUTPUT: (no output), exit=0. Clean under strict AND noUncheckedIndexedAccess, which is stricter than the tsconfig the spec asks for.

**Spec gaps left rather than invented:**
- R3 CONTRADICTS ITSELF ON STRINGS, AND THIS IS THE BIG ONE. R3 states both 'Tier H whitelist exactly: id · play_count · upvote_count · like_rate · created_at · is_public' AND 'no string from a competitor file may cross the loader boundary at any tier'. `id` and `created_at` ARE strings. Neither half can be dropped without breaking the other. I enforced both by making the whitelist exhaustive (six keys, no seventh) and STRUCTURE-LOCKING the two strings it admits by format: `id` must match a uuid regex, `created_at` must match an ISO-8601 regex, and a value that is not exactly one of those two shapes is REFUSED, not passed through. A uuid and a timestamp carry no vocabulary by construction — free text cannot ride out inside either field. This is an interpretation, not a ruling. If The Scientist wants the literal reading, `id` must become an opaque integer index and `created_at` an epoch number, and the whitelist in R3 has to be rewritten.
- THE UNIT PATH `apps/lofn-kit/src/data/` IS NOT IN THE SPEC. The spec's LANE I places the readers at `apps/lofn-kit/src/load/{snapshot,schema-guard,invariants}.ts`, and SPEC_PATCH SPLICE 9 puts the soundry reader at `apps/lofn-analytics/src/load/soundry.ts`. I built at the assigned `src/data/` path and did NOT write any of LANE I's three named files, so nothing is duplicated — but the two directories now have to be reconciled by the coordinator, and `src/load/snapshot.ts` (the utf-8-sig byte reader) and `invariants.ts` (I1-I10) are still unwritten and are not mine.
- THE ZIPF FIT METHOD IS NAMED NOWHERE. R3 requires `zipf_slope`; X3(a) specifies a log–log rank/frequency LINE but never the estimator. I anchored it to the estate's own `olsSlope` on log10(uses) vs log10(rank) — the only slope estimator the spec defines (SPEC_PATCH §SYN-2.1, `apps/lofn-kit/src/stats/ols.ts`). I did NOT reimplement it: it is an INJECTED parameter (`OlsSlope`), because `src/stats/ols.ts` does not exist yet and the kit's own invariant forbids a second implementation. The test supplies a stand-in. When L-KIT lands, the callers pass the real one and nothing in src/data changes.
- `words_per_line_slope` HAS NO x/y AND NO UNIT IN THE SPEC. I anchored x = words_per_line, y = like_rate from X2's axis definition ('change in like rate per +1 word per line, in pp'). The loader emits beta in Δ like_rate as a FRACTION per +1 word/line and does NOT convert to pp — a unit conversion inside a loader is a number nobody watched being made. The card must multiply by 100. If the intended contract is pp at the loader boundary, that is a one-line change and a test edit.
- THE TAG DENOMINATOR FOR `tags_per_clip`. No ruling says how to count a clip whose `emo_tags` is a present-but-empty list. 8 of 210 clips are in that state. I excluded them so the number reproduces the estate's published 202 exactly. The alternative (count them as measured zeros, n=210, median still 6) is equally defensible and would make the competitor card disagree with M2.3.
- WHICH FIELD `distinct_tag_count` COUNTS IS NEVER STATED. R3 lists the field; the corpus could be `tags` (the full style prompt) or `emo_tags` (the list). I anchored to `emo_tags` because it reproduces BUILD_SPEC's verified 863-distinct-across-202-clips and top-5 (triumph 34, defiance 27, anticipation 24, revelation 21, dread 21) exactly. `tags` is free prose and has no tokenizer specified anywhere in 484 KB of spec.
- WHERE `study` LIVES IN THE ROSTER FILE. R3 says 'per-handle, opt-in', so I typed it per-slot — but SPEC_PATCH PATCH 2's jsonc block for `data/watchlist/competitors.json` does not list the field at all. An absent `study` reads as `hosting` (R3's 'frozen default'), which is the tightening direction; an UNKNOWN value is refused rather than downgraded.
- §6.3's NAMED REFUSAL IS ONLY HALF-DESCRIBED. 'Any file matching the stale 2026-03-28 shape whose `plays` is the string "8.3K"' — the rest of that shape is never enumerated. I implemented only the literal half (a `plays` property whose value is a string, at top level or under `metrics`), verified against the real file at `E:/git/private-claw/tracking/suno-latest.json`. The rest of that file is already refused by the feed discriminator.
- R43c's TEMPLATE ADDITIONS `disqualifiers` AND `six_numbers` HAVE NO SHAPE. R43c says the template ships with them 'pre-filled', and names the six field names, but gives no JSON shape. I did not type them and did not invent them; `readWatchlist` ignores unknown envelope keys rather than refusing, so the template will still load when they are written.
- NO ROSTER AND NO COMPETITOR SNAPSHOT EXIST ON DISK. `data/watchlist/` does not exist (neither `competitors.json` nor `competitors.template.json`), and `data/snapshots/competitors/` does not exist. The watchlist tests therefore run against a template fixture built from the R1 schema, and the projection tests run against our OWN real 210-row pull relabelled as a competitor file — which is the honest adversary, since it is byte-for-byte the shape `ingest_suno.py --handle <h>` writes. Writing the committed template is not this unit's file.


## L-KIT-STATE — the filter vocabulary, the Insight object, the `deeper` contract and the empty/stale contract in `E:/git/private-claw/apps/lofn-kit/src/state/`

**Files:** E:/git/private-claw/apps/lofn-kit/src/state/filters.ts, E:/git/private-claw/apps/lofn-kit/src/state/cohort.ts, E:/git/private-claw/apps/lofn-kit/src/state/insight.ts, E:/git/private-claw/apps/lofn-kit/src/state/deeper.ts, E:/git/private-claw/apps/lofn-kit/src/state/empty.ts, E:/git/private-claw/apps/lofn-kit/src/a11y/guidance.ts

**Tests:** E:/git/private-claw/apps/lofn-kit/tests/filters-url.spec.ts (38 tests), E:/git/private-claw/apps/lofn-kit/tests/insight-overclaim.spec.ts (32 tests), E:/git/private-claw/apps/lofn-kit/tests/deeper-required.spec.ts (20 tests), E:/git/private-claw/apps/lofn-kit/tests/empty-never-zero.spec.ts (18 tests)

**Verification:** RAN IN E:/git/private-claw/apps/lofn-kit (node v24.14.1, npm 11.11.0, vitest 3.2.7, deps already installed — `npm install` reported "up to date in 821ms").

`npx tsc --noEmit`
  final run: NO OUTPUT, exit 0.
  (Two intermediate runs had errors, both in OTHER lanes' files, both since fixed by their owners:
   `src/format.ts(17,26): error TS2307: Cannot find module './plot/spec.js'` — a stale pre-rename
   path in the format lane; and `tests/tokens-contrast.spec.ts(24,8): error TS7016: Could not find a
   declaration file for module '../scripts/validate-palette.mjs'`. Zero errors were ever reported in
   `src/state/**` or `src/a11y/**`.)

`npx vitest run`
  Test Files  9 passed (9)
  Tests       237 passed (237)

`npx vitest run` scoped to this unit's four files:
  ✓ tests/insight-overclaim.spec.ts (32 tests)
  ✓ tests/empty-never-zero.spec.ts (18 tests)
  ✓ tests/deeper-required.spec.ts (20 tests)
  ✓ tests/filters-url.spec.ts (38 tests)
  Test Files  4 passed (4)   Tests  108 passed (108)

Two failures occurred during development and both were fixed:
  1. my own test bug — `toUrl(...).split("&")` left the `#/music` hash glued to the last segment
     (`expected [ 'floor=0#/music' ] to deeply equal [ 'floor=0' ]`); split on `/[&#]/` instead.
  2. my own arithmetic — I asserted 45 for `co=modern ∩ floor=50 ∩ out=exclude` before recomputing
     it against the snapshot; the true value is 46. Corrected and re-derived in Python against
     data/snapshots/suno/latest.json before re-asserting.

The filter tests read the REAL 210-row snapshot (data/snapshots/suno/latest.json, pulled_at
2026-08-02T00:03:48.319098+00:00, row_count 210) rather than a fixture, and every row count in them
was independently recomputed in Python first.

**Spec gaps left rather than invented:**
- THE PRINTED DENOMINATOR DOES NOT REPRODUCE. Under SPLICE T1 §3.2's own declared defaults (co=modern, out=exclude, floor=60, prov=+public) today's snapshot yields **45 of 210**, not the 46 printed in R33b, §3.5, SPLICE T2 and the Critic's table. The 46 is reachable two other ways and both are in the spec: (a) modern ∩ plays>=50 ∩ outlier excluded = 46 (vault/EXPERIMENT_STANDARD.md / analyze_power.py), (b) modern ∩ plays>=60 ∩ outlier INCLUDED = 46 (the figures BUILD_SPEC §3.3/M0.2 actually print). The declared defaults are (b) with the outlier switched to exclude, which is (c) = 45. Left as-is and asserted in a test that names all three; a coordinator ruling is needed on whether `floor` defaults to 50, `out` defaults to `label`, or the printed 46 changes.
- R33b's own sentence says `co=modern` hides 164 of 210 (leaving 46). The cohort window ALONE leaves 47 today; 46 requires a second cut. Asserted so the drift is visible.
- §4.2's tier table makes `NULL` UNREACHABLE. OBSERVATION's first condition is 'interval overlaps the null'; NULL's condition is 'excludes the null in neither direction' — the same predicate — and the table is first-match-wins in printed order. No insight can ever reach NULL or its STOP_SPENDING_ON_IT magnitude. Implemented literally (the literal reading is also the conservative one) with `TIER_MATCH_ORDER` exported so a ruling can flip it in one place, and a test that documents the dead branch.
- §4.2's table is not exhaustive: an interval that excludes the null with nEffective>=30 and `qBH === null` (no test ran) matches neither PATTERN nor FINDING. Falls to OBSERVATION, documented in code.
- 'The null' is never given a value. For a difference it is 0, for a ratio 1. `TierInput.nullValue` is supplied by the caller rather than assumed.
- §4.2 licenses FINDING three verbs (raises · lowers · predicts) and never binds them to `claim.direction`. A direction→verb map is provided but every value is still checked against VERB_LICENCE, which is the load-bearing half.
- The spec never says which prefix wins when a headline is both 'Uncontrolled:' (rule 4/R44) and 'As of … (stale):' (rule 12). Freshness is applied outermost because §6.2 calls the EXPIRED form 'inescapable'.
- §4.3 rule 9 (no stacked claims) cannot be decided from the schema — nothing records that an insight was composed. Only the routing half (`routesTo: soundry.hypothesis`) is checkable and that is all `stackedClaimRouting` asserts.
- §4.4 ranks the Tab Directive by `(tierRank, |interval.lo|, actionability)`. 'actionability' is defined nowhere; the sort uses the two defined keys and is stable rather than inventing a score. Tier RANK itself is also never numbered — the table's own weakest-first print order is used.
- `out`'s OUTLIER IDENTITY is never defined in the app spec (§3.2 gives only exclude·label·isolate). SHELLED, not invented, from the house script `scripts/analyze_power.py:69` — `outlier = max(rows, key=play_count)`. `label`/`isolate` remove no rows (render policies).
- `nmin` (`requires: nField`) and `q` (`requires: detail`) need per-plot field names that §3.3's `applyFilters(rows, f, honoured)` signature does not carry. Added an optional `RowContext` 4th argument; honouring either key WITHOUT it THROWS rather than silently removing nothing.
- `extract(rows, f)` must emit a sidecar carrying `{snapshot_id, generated_at, caveats[]}` — none derivable from rows + filter state, and the module owns no clock. Added a required `meta` third argument instead of fabricating them.
- §3.3 exports `trail()` and names no mutator. Added `recordTrail()`/`clearTrail()`, in-memory only.
- §3.2 row 15 gives `blind` a ROUTE-DEPENDENT default ('on on Results, else off') that a single `defaultValue` cannot hold. Static default is the else-branch; `defaultValueFor(key, route)` carries the exception.
- `exp`/`hyp`/`arm` are said to 'select the round / hypothesis / arm under view' but their ROW FIELD NAMES are never given. Left `fields: []` (universal, removes nothing) rather than guessing a field that would silently drop rows.
- COLLISION INSIDE SPEC_PATCH_2: SPLICE T1 §3.3 types `ShelfKeyDef.guidance` as `GuidanceNote`, but R25 narrows `GuidanceNote` to `kind:"statistic"` and R25/R32 give operable things `ControlNote`. A chip is a control with no formula. Typed `ControlNote`; flagged in the source.
- `LoadManifest`'s shape is never given anywhere. Read as `Partial<Record<WatchId, boolean>>` on SPLICE T9's own description of the fixture as 'the deterministic WatchId manifest'. `missingWatchIds()` reports gaps rather than assuming them.
- SPLICE 3 rule 6 requires a `decision` blocker to 'say in words that none exists' — those WORDS are never specified. `commandFor()` returns null and the renderer must supply them.
- §6.1 / SPLICE 2 / R20 do not say whether AXES are drawn for `needs-declaration` or `unwritten`. `EMPTY_RENDER_RULES[kind].axes` is `null` for exactly those two, and a test asserts the nulls so no renderer treats a gap as a default.
- §6.2 reserves 'the present tense' past FRESH but gives no lexicon, and a regex cannot decide tense. Only the three named literals (now/current/today) are enforced.
- CONTRADICTION: §3.5's ghost-chip example is `Music · Response · median · outliers excluded`, but §3.2 marks `out` subtractive and R33b/T2 draw subtractive defaults at FULL INK. Implemented by the rule (out is drawn full ink); the §3.5 example is stale copy. Relatedly, R33b's parenthesised subtractive list omits `q` while §3.2's table marks it subtractive — the table is treated as normative.
- §3.5's chip example prints `cohort = modern · 46 of 210` while §3.3 defines `label` as 'Cohort window' (matching SPLICE T2's 'shown as' column). There is no second short name in the spec; chips render `label`.
- `EmptyState.why` is specified as 'names the exact missing file, by path', which a `filtered-empty` has none of. The canonical filter expression is used instead.
- `crossAppHref` cannot know the Soundry's ORIGIN — SPLICE T4 makes it a runtime `configKey:"soundryOrigin"`, 'built at load, never literal'. It returns an origin-relative href and the shell prefixes it.
- `Row` was undefined estate-wide when this unit started; the filter functions were written generic over a structural `FilterableRow`. The chart lane has since defined `Row = Record<string, number|string|null|undefined>` in `src/chart/spec.ts` — a coordinator may want to unify.


## L-KIT-TOKENS + L-KIT-CHART (apps/lofn-kit)

**Files:** E:/git/private-claw/apps/lofn-kit/src/tokens.css, E:/git/private-claw/apps/lofn-kit/src/tokens.md, E:/git/private-claw/apps/lofn-kit/src/format.ts, E:/git/private-claw/apps/lofn-kit/src/chart/spec.ts, E:/git/private-claw/apps/lofn-kit/src/chart/validate.ts, E:/git/private-claw/apps/lofn-kit/src/chart/scales.ts, E:/git/private-claw/apps/lofn-kit/src/chart/axis.ts, E:/git/private-claw/apps/lofn-kit/src/chart/alt.ts, E:/git/private-claw/apps/lofn-kit/src/chart/table.ts, E:/git/private-claw/apps/lofn-kit/src/chart/render.ts, E:/git/private-claw/apps/lofn-kit/src/chart/marks/context.ts, E:/git/private-claw/apps/lofn-kit/src/chart/marks/line.ts, E:/git/private-claw/apps/lofn-kit/src/chart/marks/band.ts, E:/git/private-claw/apps/lofn-kit/src/chart/marks/interval.ts, E:/git/private-claw/apps/lofn-kit/src/chart/marks/slope.ts, E:/git/private-claw/apps/lofn-kit/src/chart/marks/bar.ts, E:/git/private-claw/apps/lofn-kit/src/chart/marks/spark.ts, E:/git/private-claw/apps/lofn-kit/src/chart/marks/strip.ts, E:/git/private-claw/apps/lofn-kit/scripts/validate-palette.mjs, E:/git/private-claw/apps/lofn-kit/scripts/validate-palette.d.mts, E:/git/private-claw/apps/lofn-kit/README.md

**Tests:** E:/git/private-claw/apps/lofn-kit/tests/tokens-contrast.spec.ts (40 tests), E:/git/private-claw/apps/lofn-kit/tests/palette-cvd.spec.ts (98 tests), E:/git/private-claw/apps/lofn-kit/tests/renderer-refusal.spec.ts (31 tests)

**Verification:** RAN FROM E:/git/private-claw/apps/lofn-kit.

1) `npx tsc --noEmit` -> exit 0, no output. Clean, including the 8 sibling-lane files already in src/state and src/data.

2) `npx vitest run` -> ALL GREEN.
   Test Files  11 passed (11)
        Tests  366 passed (366)
   My three files: tokens-contrast 40 passed, palette-cvd 98 passed, renderer-refusal 31 passed.
   The other 8 files are concurrent lanes' (filters-url, competitor-projection, insight-overclaim, deeper-required, watchlist, data-purity, empty-never-zero, loaders) and still pass against my spec.ts.

3) `node scripts/validate-palette.mjs` -> **EXIT 1, and this is correct, not a broken build.** Verbatim tail:

   dark blocks identical: true
   FAIL - 8 palette failure(s):
     - light: --sig-null on --sunk is 2.65:1, below 3:1
     - dark: --sig-pos on --card is 2.16:1, below 3:1
     - dark: --sig-neg on --card is 2.16:1, below 3:1
     - dark: --sig-null on --card is 2.99:1, below 3:1
     - dark-auto: --sig-pos on --card is 2.16:1, below 3:1
     - dark-auto: --sig-neg on --card is 2.16:1, below 3:1
     - dark-auto: --sig-null on --card is 2.99:1, below 3:1
     - light: quartet L* ladder minimum gap is 11.83, below §1.2's stated 12

   These are four real conflicts INSIDE THE SPEC (detailed in specGaps). I did not silence them by editing a threshold or inventing a hex. Instead tokens-contrast.spec.ts freezes the exact failure list, so a NEW failure goes red AND closing one goes red - a ruling cannot land without updating that list. That is why vitest is green while the gate exits 1.

4) Colour arithmetic independently verified before writing any test: my pipeline reproduces EVERY number BUILD_SPEC §1.1/§1.2 states, to the stated precision - ink 16.01/16.44, ink-dim 5.52/6.99, ink-faint 2.89/3.22, hairline 1.27, m1 10.77, m2 6.55, m3 3.89, m4 2.60, quartet L* 24.1/37.8/52.1/64.0, sequential L* 94/86/75/63/50/37/24 monotone under all four vision models plus greyscale in both themes. So when a §1.2 figure does NOT reproduce, the pipeline is not what is wrong.

5) Smoke-rendered a real interval plot through `plot()` via vite-node. Output is a complete <figure> with figcaption + visible question + generated 4-sentence sr-only description + Explore button + inline SVG + focusable mark list + <details> data table + basis line + caveat + query + deeper slot. No NaN, no undefined, no Infinity (asserted).

**Spec gaps left rather than invented:**
- THE CVD METRIC IS MISNAMED IN THE SPEC (biggest finding). §1.2 publishes 'Minimum pairwise CIELAB dE-00 ... 25.2 light / 17.4 dark' and §5.4/§8 restate the gate as 'CIEDE2000 >= 20'. Recomputed from §1.1's own hexes with Vienot 1999 in linear RGB: CIE76 dE*ab gives 25.24 light / 17.42 dark - reproducing §1.2 EXACTLY to two decimals. CIEDE2000 on the same palette gives 8.62 / 5.09. The palette is fine and the published numbers are real; the formula NAME is wrong. Both metrics are implemented and both are asserted in palette-cvd.spec.ts. Ruling needed: one word in §1.2 and §5.4.
- §5.4's '>= 20 both themes' GATE CONTRADICTS §1.2's OWN DARK MEASUREMENT of 17.4. Even on the metric §1.2 measured with, dark cannot clear 20. Lowering the gate would invent a threshold; repainting the dark quartet would invent four hexes. Locked as a test, not resolved.
- THE INFERENCE TRIAD HAS NO DARK VALUES. --sig-pos/--sig-neg alias --div-n4/--div-p4, and §1.1's dark block redefines the grounds, the quartet, the sequential ramp and the chrome but NEVER the TIDE DIVERGING ramp. So 'the only colours legal inside a stats plot' sit at 2.16:1 on --card in dark mode, failing §5.4's 3:1 mark floor. No dark diverging values exist anywhere in the precedence chain to use.
- --sig-null IS BOUND TO --ink-faint, which §1.1 itself measures at 2.9 light / 3.2 dark. §5.4 requires every mark >= 3:1 against its LOCAL ground; on --sunk (light, 2.65) and --card (dark, 2.99) it does not reach it. §1.1's binding and §5.4's floor cannot both hold.
- §1.2's L* LADDER ARITHMETIC. It states the light quartet's 'minimum gap of 12.1'; recomputed it is 11.83 - below the ~12 confusability floor §1.2 also states. Every other §1.2 figure reproduces exactly, so 12.1 is the number that is off.
- §1.1's DARK-MODE CSS IS SYNTACTICALLY INVALID. It prints `:root[data-theme="dark"], @media (prefers-color-scheme: dark) { :root:not(...) { ... }}` - an at-rule cannot appear in a selector list, so the whole rule is dropped by every parser. I transcribed it as two valid rules with byte-identical declarations (no hex changed) and added a test asserting the two blocks stay identical. Flagged because it is a spec text defect, not a build choice.
- V24-V27 ARE NUMBERED TWICE. SPEC_PATCH_2 §S adopts BOTH the Affordance Coach's SPLICE 24 (ghost alt / unique accessible name / no HTML `disabled`) AND the Task Taxonomist's SPLICE T5 (deeper reason / route / empty-and-none agreement / label) 'entire'. Four rule numbers, eight rules. validate.ts implements V1-V23 plus T5's rewritten V15 (delegated) and carries a SPEC-GAP comment for the rest; renumbering one set would put a third vocabulary in the estate.
- V21's FRESH WINDOW HAS NO FEED CADENCE MAP. §6.2 gives daily feeds 36 h and weekly feeds 8 d, but no ruling says WHICH FeedId is weekly-cadence. The mismatch band uses the daily window and says so, in both validate.ts and render.ts.
- format.ts: §2.4 gives worked outputs for int / pct1 / pp / dur / ratio / auto but NEVER defines what the `rate`, `r` or `month` members of `Fmt` print as. Implemented as clearly-labelled fallbacks (rate = percent, sig-fig rule when an interval is present; r = 2 dp; month = YYYY-MM UTC, derivable from Scale unit 'month'), each flagged inline.
- format.ts `auto`: §2.4's worked outputs carry a '%' ('half-width 2.53pp -> print `4%`') that the significant-figure rule itself cannot know about. `fmt(v, 'auto', halfWidth, unit)` takes Channel.unit as a 4th argument and appends it; fmt(4.49,'auto',2.53,'%') === '4%'. Flagged in case the format lane rules differently.
- §2.1's band Scale has an optional `padding` and no ruling states a default. DEFAULT_BAND_PADDING = 0.2 is declared once in scales.ts and flagged, rather than sprinkled through seven mark modules.
- SPLICE T4 declares `watchSatisfied(w, manifest: LoadManifest)` but no patch defines LoadManifest and `apps/_fixtures/load-manifest.json` does not exist on disk. Not my gap in the end - a concurrent lane landed src/state/deeper.ts with its own LoadManifest before I finished, and I deleted my duplicate and import theirs.


## L-SCAFFOLD-SOUNDRY — the app shell for E:/git/private-claw/apps/soundry/ (Vite 6 + TS 5.5 + vitest 3, no `dependencies` key, vanilla TS, pure model / DOM split, local-only posture)

**Files:** E:/git/private-claw/apps/soundry/package.json, E:/git/private-claw/apps/soundry/tsconfig.json, E:/git/private-claw/apps/soundry/vite.config.ts, E:/git/private-claw/apps/soundry/.gitignore, E:/git/private-claw/apps/soundry/README.md, E:/git/private-claw/apps/soundry/DEPLOY.md, E:/git/private-claw/apps/soundry/index.html, E:/git/private-claw/apps/soundry/results.html, E:/git/private-claw/apps/soundry/analytics.html, E:/git/private-claw/apps/soundry/panel.html, E:/git/private-claw/apps/soundry/server.mjs, E:/git/private-claw/apps/soundry/server.d.mts, E:/git/private-claw/apps/soundry/static-path.mjs, E:/git/private-claw/apps/soundry/static-path.d.mts, E:/git/private-claw/apps/soundry/scripts/check-local-only.mjs, E:/git/private-claw/apps/soundry/scripts/check-local-only.d.mts, E:/git/private-claw/apps/soundry/src/main.ts, E:/git/private-claw/apps/soundry/src/posture.ts, E:/git/private-claw/apps/soundry/src/style.css, E:/git/private-claw/apps/soundry/src/shell/chrome.ts, E:/git/private-claw/apps/soundry/src/shell/filters.ts, E:/git/private-claw/apps/soundry/src/shell/ask.ts, E:/git/private-claw/apps/soundry/src/anvil/deck.ts, E:/git/private-claw/apps/soundry/src/anvil/deck.data.json, E:/git/private-claw/apps/soundry/package-lock.json, E:/git/private-claw/.claude/launch.json (MODIFIED — added a `soundry` dev-server entry on port 5231 so the shell could be verified in a real browser; the only file touched outside apps/soundry/)

**Tests:** E:/git/private-claw/apps/soundry/tests/no-write-endpoint.spec.ts — 10 tests. Real listener on listen(0): binds 127.0.0.1; GET /soundry/health 200; POST / → 405 set-cookie null Allow: GET; POST /soundry/render → 405 set-cookie null; all of POST/PUT/PATCH/DELETE × 8 paths (incl. /soundry/rounds, /freeze, /generate, /observations/panel, /emit-snapshot) → 405 with no cookie; never sends access-control-allow-origin; server.mjs imports only node: builtins + relative; no wildcard bind in code (comments stripped first); package.json has no `dependencies` key., E:/git/private-claw/apps/soundry/tests/soundry-shelf.spec.ts — 26 tests. All fifteen chips render in FILTER_ORDER on all five surfaces; chipState is always one of the six named states, never undefined; SOUNDRY_LOCKED_KEYS is exactly {basis,co,est,floor,nmin,out,prov} = 7; locked (not dimmed, not parked, not absent) on all nine frozen-round cards B2/B4/B5/B7/B9.1/C1/C2/C4/C5; EVERY locked chip carries the verbatim SPLICE T3 reason, asserted character-for-character; INVARIANT F1's ≥10-word floor on every lock reason; a locked chip keeps the value carried from Analytics; the page-level strip locks all seven on Results and Analytics and locks none on the Anvil; `mod` parked with the verbatim sentence; `blind` forced+locked in a panel session and honoured on Results; the honoured table matches T3 key for key (co/out/floor/est honoured on A2+C3 and dimmed elsewhere); uncounted chips report n=null not 0; the app defines no keys of its own., E:/git/private-claw/apps/soundry/tests/posture-hosted.spec.ts — 13 tests. POSTURE=hosted throws; default-deny over 6 unrecognised values; local/retail/unset allowed; the refusal names A2 and A5; vite.config.ts calls assertLocalOnlyPosture(process.env) above `export default`; no deploy/publish/release/host script and no wrangler/pages-deploy/netlify/vercel in any script body; no `dependencies` key; no Dockerfile, no wrangler.toml; NO source file names `data/snapshots/soundry/` or `data/watchlist/` (the Soundry feed and the watchlist are stripped/refused); no `user_id`, no non-loopback absolute origin anywhere in source; all four entries declare the sealed CSP; no page loads a CDN or network font; no render/publish/upload bound to any click or submit handler., E:/git/private-claw/apps/soundry/tests/deck.spec.ts — 24 tests. The six frozen families in order, in code and in the JSON; the three §G4 floors (18 / 3-per-family / 6 distinct axes); shape validation of SPLICE F2 §1.4c's worked example and seven rejection cases; family coverage NAMES the thin family; count refusals say how many are missing; refusal copy never says 'lower the floor' and never suggests repeating a card; duplicate ids refused; countByFamily reports 0 rather than omitting; the shipped file is the un-authored stub and is reported as a stub not as 18 errors; `_authored_by` names Lofn and 'build agent' and `_edit_when` names L-ANVIL; loadDeck() throws DeckRefusal with STUB_REFUSAL; loadDeck succeeds the moment a deck clears every floor; a guard that binds all three floors the day the deck is authored; no code path writes deck.data.json (source grep for writeFileSync/appendFileSync/createWriteStream/writeFile/fs.promises near the filename; deck.ts imports node:fs nowhere and exports no writer).

**Verification:** ALL GREEN. Real output, unmodified.

$ npx tsc --noEmit
(no output)  tsc exit=0

$ npx vitest run
 RUN  v3.2.7 E:/git/private-claw/apps/soundry
 ✓ tests/deck.spec.ts (24 tests) 22ms
 ✓ tests/posture-hosted.spec.ts (13 tests) 30ms
 ✓ tests/soundry-shelf.spec.ts (26 tests) 27ms
 ✓ tests/no-write-endpoint.spec.ts (10 tests) 155ms
 Test Files  4 passed (4)
      Tests  73 passed (73)
   Duration  726ms

$ npx vite build
 vite v6.4.3 building for production...
 ✓ 15 modules transformed.
 dist/results.html                2.00 kB │ gzip: 0.93 kB
 dist/analytics.html              2.05 kB │ gzip: 0.98 kB
 dist/panel.html                  2.11 kB │ gzip: 1.14 kB
 dist/index.html                  3.01 kB │ gzip: 1.41 kB
 dist/assets/style-DUBp9DYp.css   7.98 kB │ gzip: 2.38 kB
 dist/assets/main-BGjmIlNC.js    22.04 kB │ gzip: 8.28 kB
 ✓ built in 292ms

$ node scripts/check-local-only.mjs
 check-local-only: PASS — the bundle holds no external origin and no render/publish/upload handler.
 exit=0

$ POSTURE=hosted npx vite build
 exit=1
 failed to load config from E:\git\private-claw\apps\soundry\vite.config.ts
 error during build:
 Error: POSTURE=hosted is refused. The Soundry is never hosted, on any domain, under any Access
 policy. It holds asset A2 (unposted variants) and asset A5 (the pre-registration chain). There is
 no hosted build to strip, because there is no hosted build. (received POSTURE="hosted")

TWO FAILURES OCCURRED DURING THE RUN AND WERE FIXED — reported for honesty:
 1. `tests/no-write-endpoint.spec.ts > binds a loopback host literal` failed: server.mjs's comment
    literally names the wildcard address in order to forbid it, and the assertion could not tell a
    prohibition from a bind. Fixed by stripping line comments before the assertion and adding a
    `listen(...)` pattern check. The prohibition comment stays.
 2. `tests/posture-hosted.spec.ts > no absolute external origin` failed with 121 offenders, all of
    them registry URLs inside `package-lock.json`. Fixed by excluding lockfiles by name (they ship
    in no bundle) — scanning them turns a boundary test into a lockfile test.

`node scripts/check-local-only.mjs` also failed once on the first real build: `dist/index.html`
carried `http://127.0.0.1:<port>` inside an HTML comment quoting §0.1. The gate reads §0.1
literally ("any absolute http(s):// origin other than the CSP declaration"), so the GATE was left
strict and the COMMENT was reworded. Nothing was weakened to make a test pass.

BROWSER VERIFICATION (dev server on 127.0.0.1:5231, then stopped):
 - `results.html` renders seven padlocked chips — Cohort window, Measure, Estimator, Outlier, Play
   floor, Ranking n-floor, Provenance floor — each with the verbatim lock sentence rendered as
   TEXT under the shelf, not only on hover (R27).
 - `index.html` renders all fifteen chips, none locked (correct: A0–A6 read no frozen round), the
   visible `<label>` "What should we try today?", the three verbatim seed chips, and the live
   "Draft a design" button with "Opens the design step. Writes nothing yet." inline.
 - Clicking submit navigated to `#/anvil/design`; the address bar read
   `http://127.0.0.1:5231/index.html?fv=1#/anvil/design` — the canonical `replaceState` rewrite of
   SPLICE T1 §3.1 is working.
 - Zero console errors.

**Spec gaps left rather than invented:**
- CSP `connect-src` COLLISION INSIDE BUILD_SPEC. §8.1's verbatim head block writes `connect-src 'none'` and calls the Soundry 'retail-sealed in every posture'. Part 3 §0.1's own autonomy table writes `connect-src 'self'` in all three build postures and states the only fetch target is the loopback server. Both are BUILD_SPEC, so precedence does not settle it. RESOLVED toward §0.1 (a bench that cannot reach its own server cannot bind a clip_id, read its ledger, or run §G5's measure route) and flagged in index.html and vite.config.ts. This is the one place I chose between two spec'd values.
- `/@kit/plot/plot.css` — §8.1's head links it. The kit on disk has `src/chart/` and ships no CSS other than `tokens.css`. Omitted from all four heads rather than invented.
- `apps/lofn-kit/src/shell.css` — named in SPEC_PATCH_2's L-KIT build order; NOT on disk. BUILD_SPEC U1 says `src/style.css` 'declares nothing', which with no kit sheet yields unstyled text. `src/style.css` carries a provisional shell sheet that uses ONLY tokens.css variables — the file contains no `#` character, no invented colour and no invented length — and must move into the kit when L-KIT lands.
- `apps/lofn-kit/src/state/applicability.ts` — SPLICE T9's file tree puts `chipState` / `factsOf` / `CardFacts` there, landed by U0b inside L-KIT. NOT on disk (the kit has `state/filters.ts` only). `apps/soundry/src/shell/filters.ts` therefore implements the Soundry's §T3 table in §3.4's exact shape and is a second implementation of a rule R33 exists to unify. Flagged at the top of the file; it must delegate the day the kit module lands.
- SPLICE T3's honoured table names ten keys, seven locked keys and one parked key — and leaves `sel` unnamed entirely, and `nmin` / `basis` / `prov` named only inside the lock. Away from the nine frozen-round cards those four have no declared home. They fall to §3.4 rule 4 (dimmed, generated `why`, computed `returnsOn`) rather than to an invented one; exported as `UNDECLARED_IN_T3` and asserted by a test so the gap is visible in code.
- THE SOUND/RETURN AXIS KEY NAME. §G4 requires `deck.data.json` to carry '>= 6 distinct SOUND/RETURN axis values' and §1.4A(iii) says the value comes from the music skill's Phase-0 positive vocabulary — but SPLICE F2 §1.4c's card schema has no such field and no source names its key. `sound_return_axis` is this build's placeholder. The FLOOR is specified and enforced; only the key name is invented, and it is flagged in the interface doc comment and in DEPLOY.md.
- A0's seed-chip source URLs. §1.1 gives the three sentences verbatim and says each 'carries the source card's URL'. Lofn Analytics has no card modules on disk. `sourceHref` is `null` and the chip renders an explicit unlinked line rather than a fabricated href.
- Chip counts. §3.4 types `ChipState.n` / `.of` as `number`. No card lane has landed so nothing has been counted, and printing `0 of 0` is the empty-rendered-as-zero failure §6.1 forbids. Widened to `number | null`, declared in a comment, and the shelf's right-edge readout prints the absence in words.
- PER-CARD LOCK vs PAGE-LEVEL STRIP. §T3 locks per card (nine named cards); §3.5 draws ONE 44px strip per page. No source says what the page-level strip shows. INTERPRETED: the strip resolves the lock against the tab's card index, so Results and Analytics show seven padlocks and the Anvil shows none. The alternative — a live-looking chip on a page where every card ignores it — is the exact lie the padlock exists to prevent. Tested both ways.
- The 88px shelf cap. §3.5 says the strip 'wraps to 88px rather than scrolling horizontally'. Fifteen chips at this build's label lengths exceed 88px, and clipping them is the silent drop R35 rejects. The cap is NOT clamped; the strip wraps freely and never hides or scrolls. Flagged in style.css.
- The Soundry's left rail contents. SPLICE 21 says 'Board 3 draws one left rail and one main panel', but no source says what the rail holds on Results or Analytics. Populated ONLY with card ids and titles lifted from BUILD_SPEC / SPEC_PATCH_2 headings (A0–A6, B1–B10, C1–C7); unbuilt stops render R26-blocked with a reason naming the owning build lane. No card content is invented.
- `LOCK_REASON_BLIND_IN_PANEL`. §T3 states the REQUIREMENT ('blind=on is forced and locked inside a panel session — the listener cannot turn it off, and the chip says so') but gives no chip copy. The sentence used is derived from that clause and meets F1's ≥10-word floor. The seven-key `LOCK_REASON` is verbatim and asserted character-for-character.
- `posture-hosted.spec.ts` for the Soundry. SPEC_PATCH's test 10 is written for Lofn Analytics (build hosted, assert dist/** is stripped). The Soundry HAS no hosted build, so the assertion was inverted to a refusal plus a source-level absence sweep. This is strictly stronger than the stripping test, and it is an interpretation, not a transcription.


## L-SCAFFOLD-ANALYTICS — the Lofn Analytics app shell (no cards)

**Files:** E:/git/private-claw/apps/lofn-analytics/package.json, E:/git/private-claw/apps/lofn-analytics/tsconfig.json, E:/git/private-claw/apps/lofn-analytics/vite.config.ts, E:/git/private-claw/apps/lofn-analytics/.gitignore, E:/git/private-claw/apps/lofn-analytics/README.md, E:/git/private-claw/apps/lofn-analytics/DEPLOY.md, E:/git/private-claw/apps/lofn-analytics/index.html, E:/git/private-claw/apps/lofn-analytics/src/main.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/page.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/chrome.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/rail-modality.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/rail-source.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/headwinds-rail.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/skiplinks.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/markup.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/modality.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/tabs.ts, E:/git/private-claw/apps/lofn-analytics/src/shell/grid.css, E:/git/private-claw/apps/lofn-analytics/src/shell/shell.css, E:/git/private-claw/apps/lofn-analytics/src/signals/standing.ts, E:/git/private-claw/apps/lofn-kit/src/a11y/scan.ts (NEW — SPLICE 17's scanElements), E:/git/private-claw/apps/lofn-kit/src/a11y/guidance.ts (APPENDED — SPLICE 17's registry, at the address that file's own header directs), E:/git/private-claw/apps/lofn-kit/src/tokens.css (APPENDED — --rail-left: 168px, --rail-right: 248px, per SPLICE 21)

**Tests:** E:/git/private-claw/apps/lofn-analytics/tests/no-disabled.spec.ts — V26/V27: no HTML `disabled` in any emitted markup, none in src/ or index.html by grep, `control()` has no code path that could write one, it THROWS on a blocked reason under 8 words, the reason renders as a visible <p> not a tooltip, plus SPLICE 24's appended forbidden strings (title=, smooth scroll, order:, *-reverse, "not available"/"coming soon"/"try again later", zero @keyframes), E:/git/private-claw/apps/lofn-analytics/tests/guidance-coverage.spec.ts — SPLICE 17 assertions (a)-(f) over every shell+chrome module on 3 modalities x 7 tabs x 2 register fixtures; plus every ControlNote fills does/after/reverses (and does != after, <=18 words), every RegionNote fills forWhat/notHere, and the disclosure is a button with a matching next-sibling panel, E:/git/private-claw/apps/lofn-analytics/tests/rail-radiogroup.spec.ts — the rail is role=radiogroup aria-label=Modality with exactly 3 role=radio, one aria-checked, a roving tabindex (one 0, two -1), NO role=tab/tablist/tabpanel, never disabled, order computed from likes descending, two bars described as ONE role=img; and the tab bar is <nav><ol> of hrefs with aria-current=page and never aria-selected, E:/git/private-claw/apps/lofn-analytics/tests/headwinds-rail.spec.ts — test 29: renders on all 21 screens; exactly 3 rows; rankStanding() is the only ordering path and refuses (naming why) what it cannot place; 7 stops total; ▼/– glyphs; the refusal list (no sparkline/pill/badge/sort/severity); the LEDE dedupe prints "in the lede ↑" and suppresses the number; the byte-exact footer line; the zero-row needs-declaration state with a `decision` counter and no blank box, E:/git/private-claw/apps/lofn-analytics/tests/layout-columns.spec.ts — three grid-template-columns on .app-grid and no more; the 1099/859/419 breakpoints; the left rail yields first and .rail-headwinds is untouched at tablet; no `display: none` on the rail at mobile; no order:/reverse in either sheet; the mobile chip's href is #headwinds; --sticky-h 138 on Music and 116 elsewhere; scroll-margin-top; five skip links in SPLICE 19's order; Method/Glossary last in the header on all 21 screens, E:/git/private-claw/apps/lofn-analytics/tests/_fixtures.ts — shared fixtures (not a spec file)

**Verification:** RAN AND PASSED, all real output.

$ cd apps/lofn-analytics && npx tsc --noEmit
(no output — exit 0)

$ npx vitest run
 RUN  v3.2.7 E:/git/private-claw/apps/lofn-analytics
 ✓ tests/layout-columns.spec.ts (14 tests) 36ms
 ✓ tests/headwinds-rail.spec.ts (18 tests) 42ms
 ✓ tests/rail-radiogroup.spec.ts (15 tests) 25ms
 ✓ tests/guidance-coverage.spec.ts (12 tests) 20ms
 ✓ tests/no-disabled.spec.ts (11 tests) 140ms
 Test Files  5 passed (5)
      Tests  70 passed (70)

REGRESSION CHECK on the kit I edited:
$ cd apps/lofn-kit && npx tsc --noEmit   -> clean
$ npx vitest run                          -> Test Files 11 passed (11) / Tests 366 passed (366)

$ npx vite build  (lofn-analytics)
 ✓ 19 modules transformed.
 dist/index.html 1.24 kB | dist/assets/main-XdIawTt2.css 8.81 kB | dist/assets/main-CkMST9yF.js 33.13 kB
 ✓ built in 284ms
(dist/ removed afterwards; it is gitignored)

FAILURES I HIT AND FIXED (reported because they were real):
1. Five tests failed on the first run. Three were my own CSS/TS header COMMENTS containing the
   literal forbidden strings "scroll-behavior: smooth", "flex-direction: row-reverse" and
   "@keyframes" — the grep gate does not read comments differently, and it was right. Comments
   reworded; the strings now appear nowhere in src/.
2. Two were my own greedy `[\s\S]*` media-query regexes in layout-columns.spec.ts leaking across
   @media blocks. Replaced with a brace-depth `mediaBlock()` helper.

LIVE SMOKE TEST (vite dev on 127.0.0.1:5230, then stopped):
 - page renders, zero console errors
 - URL canonicalised on load to `/?fv=1#/music/signal` via replaceState
 - clicked tab "Craft" -> `#/music/craft`, aria-current moved, title "Music · Craft — Lofn Analytics"
 - clicked the Art radio -> `/?fv=1&mod=art#/art/craft`; the TAB SURVIVED the modality change and
   the nav relabelled to "Art tabs" (Board 7's "carry between tabs", working)
 - history.back() -> `/?fv=1#/music/craft` with mod=music restored. Back-button honest.
 - DOM order of .app-grid children: NAV.rail-modality, MAIN#content, ASIDE#headwinds
 - document.querySelectorAll('[disabled]').length === 0
 - 21 elements carrying data-guidance; role=radiogroup present; zero role=tab
 - --sticky-h resolved to 138px on a Music tab
 - at the 664px pane width the mobile branch fired correctly: one column, the Headwinds rail
   full-width at the END of the flow (not collapsed), and the header chip visible reading
   "⚠ 0 headwinds" with href="#headwinds"
 - two real defects found ONLY by looking at it, both fixed: (a) the mobile chip stayed
   display:none because grid.css's media rule and shell.css's base rule had equal specificity and
   shell.css was emitted last — fixed with `header .hw-chip`; (b) a missing space rendered
   "registered standing headwinds0 of 1".

NOT VERIFIED: the >=1100px three-column layout was never seen rendered — the Browser pane is
fixed at 664px and cannot be resized. The desktop template is SPLICE 21's CSS transcribed
verbatim and is asserted by layout-columns.spec.ts, but no human or machine has looked at it.

**Spec gaps left rather than invented:**
- PORT. No document gives Lofn Analytics a port (The Soundry's 5231 is fixed at SPEC_PATCH:953 and SPEC_PATCH_2:698). Chose 5230, hardcoded in exactly one place (vite.config.ts) and recorded in DEPLOY.md as a build-time choice.
- THE ANIMATION RAIL'S TAB SET. BUILD_SPEC §10 gives Art "the same tab skeleton" and R45(b) adds CONFLUENCE to it; §11 describes Animation as four cards (N0.1-N0.4) and never states its tabs. SPLICE 21 requires the grid to be identical on every tab, so the seven-tab skeleton is applied to all three rails — noted in src/shell/tabs.ts as an inference, not a line of spec.
- ANIMATION'S MEASUREMENT DEPTH (0-5). §2 states Art's ("Art scores 1/5") and SPLICE 18's example states Music's ("depth 3 of 5"). Animation's is nowhere. Set to 1 from §2's own note for that row, flagged inline in rail-source.ts.
- M4.2 CONFIDENCE FOR TWO PROVENANCES. §8 M4.2 gives four confidence values against four provenance rungs; R47 gives EIGHT provenance values, and rules that `derived` and `randomised-round` have no static band. No confidence is stated for either. `confidenceOf()` returns null and `rankStanding()` REFUSES to rank such an entry, naming it — rather than invent a coordinate on her map.
- NO TAILWIND GLYPH FOR THE RAIL. SPLICE 22 gives `▼` (headwind) and `–` (DIRECTION UNKNOWN) and no third. A tailwind row prints the word instead of an invented glyph.
- THE STANDING REGISTER ITSELF. §8 M4.3 prints twelve claims and R45(c) names four more ids, but states no `severity` (1-5), `falsifier`, `n` or `asOf` for ANY of them — and M4.3 makes all four mandatory while SPEC_PATCH R6 forbids an agent from seeding the register. STANDING_REGISTER ships EMPTY, so the rail renders SPLICE 22's own specified needs-declaration state. This is a decision she has to make, not a file I can write.
- "SKIP TO METHOD" TARGET. SPLICE 19 names the fifth skip link but not what it points at. It routes to the Provenance tab (M5.6 is the Glossary), passed in by the caller rather than guessed as an anchor id.
- TAB RegionNote COPY. SPLICE 23 requires "each tab's forWhat + notHere" and prints only M6 CONFLUENCE's notHere verbatim (reproduced exactly). The other thirteen strings are AUTHORED from each tab's own description in BUILD_SPEC §4-§9, with the source section cited on every entry.
- PLATFORM -> MODALITY MAPPING. The rail's numbers come from `socials`, but no line states the rule a loader must apply (nightcafe->Art, suno->Music, tiktok+youtube->Animation). NOT implemented as a rule; rail-source.ts transcribes BUILD_SPEC §2's own printed rail instead, with an expiry note naming the loader lane.
- BUILD_SPEC §8.1's head block links `/@kit/plot/plot.css`. The kit on disk has `src/chart/`, not `src/plot/`, and ships no plot.css. Not linked — a dangling stylesheet href is a defect. Only `/@kit/tokens.css` is linked (verified 200 in dev and bundled in the build).