# RATIFICATIONS — Lofn Analytics & The Soundry

*Coordinator rulings on collisions the build lanes could not settle. Opened 2026-08-02 in response
to `BUILD_VERIFICATION.md`'s **RED** verdict, spec deviations 3 and 4.*

**Precedence this file operates under:** `SPEC_PATCH_2.md` > `SPEC_PATCH.md` > `BUILD_SPEC.md`.
A ratification here is **normative over all three** for the one line it names, and nothing else.
Every number below was recomputed from a file on disk in this session, and the recompute is printed
with it.

**Numbering:** `RAT-n`. Rulings are `RATIFIED` (binding, code and tests must agree) or `OPEN`
(named, measured, and explicitly not decided — a build agent may not close one).

---

## RAT-1 · THE SOUNDRY'S `connect-src` IS `'self'` — RATIFIED

### The collision

| source | text | scope |
|---|---|---|
| `BUILD_SPEC.md:832` (§8.1) | `connect-src 'none'` inside "every page head, verbatim shape" | Part 2, **shared** foundation, both apps |
| `BUILD_SPEC.md:837` (§8.1) | "leaves `'none'` in retail. **The Soundry is retail-sealed in every posture**" | Part 2, shared |
| `BUILD_SPEC.md:885` (Part 3 §0.1) | "**CSP** \| `connect-src 'self'` in all three build postures. The only fetch target in the bundle is `http://127.0.0.1:<port>`." | Part 3, **Soundry-specific** |

The lane shipped `'self'` and flagged it in two places (`apps/soundry/index.html`,
`apps/soundry/vite.config.ts`) rather than choose silently. That was the right procedure. This is
the ruling it asked for.

### RULING

**`connect-src 'self'`. All four Soundry entry heads. All three postures. `'none'` is struck for
The Soundry.**

### The argument, in the order the arguments actually decide it

**1 — Precedence already settles it; the lane was too modest.** The lane wrote "Both are BUILD_SPEC,
so precedence does not settle it." That is true only if the question stops at BUILD_SPEC. It does
not. `SPEC_PATCH.md` and `SPEC_PATCH_2.md` both outrank BUILD_SPEC, and between them they restate
`connect-src 'self'` **seven times as settled fact and never once restate `'none'`**:

```
SPEC_PATCH.md:255    "stands: the Soundry renders nothing, holds no audio, and CSP is `connect-src 'self'`"
SPEC_PATCH.md:607    "CSP is `connect-src 'self'`. RECOMMENDED DEFAULT: you drop the twelve mp3s into ..."
SPEC_PATCH.md:635    "...forbids a `dependencies` key, and pins connect-src 'self'"
SPEC_PATCH.md:637    "...the Soundry renders nothing, holds no audio, and CSP is connect-src 'self'"
SPEC_PATCH_2.md:781  "preserves §0.1 exactly: no generation API, no `dependencies` key, `connect-src 'self'` untouched"
SPEC_PATCH_2.md:1453 "§0.1 pins `connect-src 'self'`, forbids a `dependencies` key, forbids any ..."
SPEC_PATCH_2.md:2011 "no generation API, no `dependencies` key, `connect-src 'self'` untouched"
```

`'self'` is not one of two live readings. It is the reading both patch rounds **built on**, load-
bearing in four separate rulings (the panel-audio gap, the mp3 recommendation, the Anvil-generator
ruling R13, and the Forge Master's mechanics section). Overturning it to `'none'` would silently
invalidate arguments in the two documents that outrank BUILD_SPEC.

**2 — Specific beats general, and Part 3 says so in its own opening sentence.** Part 3 opens:
*"The shared foundation … is specified elsewhere and is imported, not restated. This part specifies
only what is Soundry's own."* §8.1's head block is the shared template for **both** apps; §0.1's
autonomy table is the Soundry's own, and it is the table the app's entire posture story is written
against (Writes / Reads / Never / CSP / Bind / Posture, five rows of which the build already
enforces byte-for-byte). A generic template yields to the specific table that exists precisely to
override it.

**3 — `'none'` makes §0.1's own next sentence false.** The same table cell reads: *"The only fetch
target in the bundle is `http://127.0.0.1:<port>`."* A cell cannot both name a fetch target and
forbid all fetching. Under `'none'` the Soundry cannot reach its own `server.mjs`: no
`GET /soundry/health`, no §0.2 `clip_id` binding, no §3.1 ledger read, no §G5 measure route. The
bench becomes four static pages. **A local bench that cannot reach its own loopback server cannot
bind a clip_id** — and the clip_id join is §0.2, the single mechanism the whole instrument rests on.

**4 — "Retail-sealed" is a posture claim, and it is not `connect-src`'s job.** §8.1's sentence is
true and stays true under this ruling. The seal is enforced by §0.1's own **Posture** and **Bind**
rows and by machinery already shipped and passing:

- `POSTURE=hosted` refuses the build (`vite.config.ts` asserts above `export default`; verified
  exit 1 with the §0.1 refusal text).
- `server.mjs` binds `127.0.0.1` only; no `access-control-allow-origin`.
- `scripts/check-local-only.mjs` walks `dist/**` and fails on **any** absolute `http(s)://` origin
  outside the CSP declaration, or on `render`/`publish`/`upload` bound to a handler. It passes.
- `tests/no-write-endpoint.spec.ts` (10 tests): every write verb on eight paths returns 405.
- `tests/posture-hosted.spec.ts` (13 tests): no deploy/publish/host script, no CDN, no network font.

`connect-src 'self'` permits exactly one thing those five do not already permit: the app talking to
`http://127.0.0.1:<its own port>`, which is the same machine, the same process tree, the same
operator. It widens the attack surface by zero origins. The seal is intact; it was never made of
`connect-src`.

### Consequences — what must be true in the tree

1. All four Soundry heads (`index.html`, `results.html`, `analytics.html`, `panel.html`) carry
   `connect-src 'self'`. **Already true.**
2. The string `connect-src 'none'` appears in **no** Soundry `.html`, `.ts` or `.mjs`. Asserted by
   `tests/posture-hosted.spec.ts` as of this ruling.
3. The two flagged comments (`index.html`, `vite.config.ts`) stop describing an unresolved collision
   and cite RAT-1 instead. A resolved question must not read as an open one.
4. The `csp-per-mode` plugin's only widening remains the dev HMR socket.

### RAT-1a · LOFN ANALYTICS KEEPS `'none'` FOR NOW — **OPEN**, deliberately

`apps/lofn-analytics/index.html:15` ships `connect-src 'none'`, while `BUILD_SPEC.md:1817` states
for that app: *"CSP is `connect-src 'self'`; everything else is `'self'`."* That is a plain
single-source instruction with no competing text, so on paper the app is out of compliance.

**It is left at `'none'` and recorded here rather than widened, because the argument that carries
RAT-1 does not carry here.** RAT-1 rests on a live need: the Soundry fetches its own loopback
server today and is inert without it. Lofn Analytics fetches **nothing** today — no loader lane has
landed, `src/load/soundry.ts` does not exist, and `rail-source.ts` transcribes the spec's printed
rail rather than reading a file. Widening a CSP for a capability nothing uses is a permission
granted in advance, which is the direction this estate does not move in.

**This closes the day the analytics loader lane lands**, at which point BUILD_SPEC:1817 applies
unchanged and the head becomes `'self'`. Until then the discrepancy is named, not hidden.

---

## RAT-2 · THE PRINTED `46 of 210` STANDS; THE `floor` DEFAULT BECOMES `50` — RATIFIED

### The collision

`BUILD_VERIFICATION.md` deviation 4: the printed denominator **46 of 210** does not reproduce from
the app's declared filter defaults. Recomputed independently this session, in Python, against
`data/snapshots/suno/latest.json` (`row_count` 210, `pulled_at` `2026-08-02T00:03:48.319098+00:00`):

```
total rows: 210
outlier: Triple Arch Over Me  349736 plays  id ef858626-2bf0-4aac-b331-c33fa2487e9d

modern window ALONE (2026-02 onward, no floor, outlier included) ......... 47
modern + floor >= 50 + outlier excluded  ................................. 46   <- EXPERIMENT_STANDARD
modern + floor >= 50 + outlier INCLUDED  ................................. 47
modern + floor >= 60 + outlier excluded  ................................. 45   <- app's declared defaults
modern + floor >= 60 + outlier INCLUDED  ................................. 46   <- BUILD_SPEC §3.3 / M0.2
modern rows with plays < 50 .............................................. 0
modern rows with 50 <= plays < 60 ........................ 1  ("Good At This Now", 56 plays)
```

Cross-checked against the house script, unmodified:

```
$ python scripts/analyze_power.py --json
 "cohort": { "from": "2026-02", "min_plays": 50, "n": 46 },
 "baseline_like_rate_pct": 4.489,
 "observed_sd_pp": 1.802,
 "median_plays": 258.0
```

### RULING

**The printed `46` is correct and is not touched anywhere. `SPEC_PATCH_2 §3.2` row 7 — `floor`,
"Play-maturity floor" — has its default changed from `60` to `50`.**

### The argument

**1 — 46 is the house number, and it reproduces exactly.** `vault/EXPERIMENT_STANDARD.md:22` reads
*"Modern era (2026-02 onward), songs with ≥50 plays, excluding the Triple Arch outlier — n = 46"*,
and `scripts/analyze_power.py` recomputes it to the row. Every downstream constant the estate
quotes — the **4.49 %** baseline, **1.80 pp** observed sd, **1.17 pp** true between-song sd, the
**58 %** binomial share, **258** median plays, the whole power table, the *"+15 % needs ~19 days"*
answer, and the ±2.53 pp sentence that opens Part 3 §0 — is computed on **that** cohort. The 46 is
not a caption. It is the denominator the two apps exist to defend.

**2 — `floor=60` is the one number in the chain nobody derived.** No source states where `60` came
from. It appears as a default in the §3.2 table and inside BUILD_SPEC's basis lines
(`:564`, `:1489`, `:650`), and nowhere is it argued. `50` is argued, in the file that measures it:
*"Below ~50 plays a like rate is almost pure noise and destabilises the mean"*
(`scripts/analyze_power.py:41`).

**3 — The whole 46-vs-45 gap is one song.** `floor=60` drops exactly one clip that `floor=50`
keeps: **"Good At This Now", 56 plays**. Zero modern clips sit below 50. So at `floor=50` the play
floor is *non-binding on today's snapshot* and the 46 comes from `co=modern ∩ out=exclude` — which
is exactly what R33b's own sentence describes when it says the cohort leaves 46. Under the declared
`floor=60` the app's default cohort is **n=45, a cohort no house document has ever analysed**, with
its own baseline (mean 4.351 %, median 4.421 %) that matches neither the 4.49 % Part 1 prints nor
the 4.37 % Part 2 §3.3 prints. Shipping it would put a third baseline in an estate that already has
two.

**4 — The alternative readings both cost more.** Changing `out`'s default to `label` would reach 46
via BUILD_SPEC §3.3's cohort — but `out=exclude` is `EXPERIMENT_STANDARD` §5.6 (*"The outlier is
quarantined by default"*), and unquarantining a song that is 93.5 % of all plays to fix a caption is
the trade nobody would take. Changing the printed 46 across six spec locations would mean printing
45 and abandoning the vault's published baseline. **Moving one unargued default to the number the
measurement script already uses is the smallest change that makes screen and vault agree.**

### Consequences — what must be true in the tree

1. `apps/lofn-kit/src/state/filters.ts` `SHELF.floor.defaultValue` = `["50"]`, and the chip guidance
   default string reads `50`. **Changed by this ruling.**
2. `applyFilters(ROWS, {}, ["co","out","floor","prov"])` returns **46** rows. Asserted.
3. Any card printing a basis line on the default cohort prints `n = 46 of 210`.

### RAT-2a · THE BASIS LINES THAT SAY "≥ 60 plays" — **OPEN**

Three BUILD_SPEC strings describe the default cohort as `≥ 60 plays`, and one prints statistics
computed on the `floor=60`, **outlier-included** variant:

| line | text | status under RAT-2 |
|---|---|---|
| `BUILD_SPEC:564` | "n = 46 clips created 2026-02-01 or later with ≥ 60 plays" | `46` right, `60` now wrong — should read `≥ 50 plays` |
| `BUILD_SPEC:1489` | "median of 46 clips with ≥60 plays … 4.37% … mean 4.29% … pooled 1.57%" | see below |
| `BUILD_SPEC:650` | "Widest filter: cohort ≥ 60 plays. [Relax to ≥ 50 → n 58]" | an illustrative empty-state string, not a default |

`:1489`'s three statistics are the `floor=60` **outlier-INCLUDED** cohort. Recomputed:

```
floor>=50 out=exclude  (RAT-2's default): n=46 median=4.4425% mean=4.4890% pooled=4.7279% median_plays=258
floor>=60 out=label    (as printed):      n=46 median=4.3695% mean=4.2875% pooled=1.5683% median_plays=267
```

The `pooled 1.57%` is the tell: pooling **with** a 349,736-play song at a 1.44 % like rate drags the
pooled rate to a third of every other estimator on the same card. Under RAT-2's default the card
would print `median 4.44% · mean 4.49% · pooled 4.73%`.

**Not ruled on here.** No card lane has landed, so nothing on disk prints these numbers yet, and
rewriting four BUILD_SPEC statistics is a spec edit, not a build fix. Recorded so the card lane
inherits the recompute instead of transcribing a stale trio. **The card lane must not transcribe
`BUILD_SPEC:1489`.**

---

## RAT-3 · THE ≥1100px DESKTOP LAYOUT, RENDERED — three columns CONFIRMED, two new defects OPEN

`BUILD_VERIFICATION.md` real failure 4 and next-unit 7: the primary layout of the primary app had
never been rendered. It has now been, on a real Chromium viewport at 1100 × 860 and 1280 × 860,
dev server `http://localhost:5230`.

### CONFIRMED — the three-column grid is exactly as specified

Measured from the live DOM, not from the stylesheet text:

```
innerWidth 1280 -> grid-template-columns: 168px 768px 248px
                   NAV.rail-modality  x=24   w=168
                   MAIN.col-centre    x=216  w=768
                   ASIDE.rail-headwinds x=1008 w=248

innerWidth 1100 -> grid-template-columns: 168px 588px 248px
                   NAV.rail-modality  x=24   w=168
                   MAIN.col-centre    x=216  w=588   (>= SPEC_PATCH_2:1181's minmax(560px,1fr))
                   ASIDE.rail-headwinds x=828 w=248
                   .hw-chip display: none   (correctly desktop-hidden)
```

DOM order NAV → MAIN → ASIDE, no `order:`, no reverse. `--rail-left: 168px` / `--rail-right: 248px`
resolve. Visually confirmed in a screenshot at both widths: modality rail left, `Music · Signal`
centre, Headwinds panel right. **layout-columns.spec.ts was telling the truth.**

### RAT-3a · THE STICKY TAB BAR IS CLIPPED IN HALF BY THE HEADER — **OPEN**, live defect

Visible in a scrolled screenshot at both 1100 and 1280, and reproduced numerically:

```
scrolled to y=600, innerWidth 1100:
  header      position: sticky  top: 0px   height: 97px   z-index: 40
  nav.tab-bar position: sticky  top: 72px  height: 115px  z-index: 30
```

`.tab-bar { inset-block-start: 72px }` transcribes SPLICE 19's declared header height of **72px**
(`--sticky-h` = "header 72 + shelf 44 + comb 22" = 138). **The built header renders at 97px.** The
header wins on `z-index` (40 vs 30), so the top **25px of the tab bar sits underneath it** whenever
the page is scrolled — the tab labels are visibly sliced through the middle. This is width-
independent; it is not a desktop-only bug, it was simply never seen because nobody scrolled.

Arithmetic of the 25px: the header's controls are `min-block-size: 48px` (`.icon-btn`,
`.method-link`, WCAG 2.5.8) inside a header whose block padding brings it to 97. 48 + 2×12 = 72
would satisfy the spec's number.

**Not fixed here.** Choosing between "shrink the header padding to reach 72" and "raise
`inset-block-start` to the header's real height" is a choice between a spec'd dimension and a built
one — the same class of question as RAT-1, and it needs the same kind of ruling rather than a build
agent's preference. `// SPEC-GAP: header renders 97px against SPLICE 19's declared 72px` belongs at
`apps/lofn-analytics/src/shell/shell.css` `.tab-bar`.

### RAT-3b · THE SEVEN-TAB BAR WRAPS FROM 1100px TO 1147px — **OPEN**

At the spec's own minimum desktop width the tab bar does not fit on one line:

```
innerWidth 1100, centre column 588px
  tab widths  Signal 60 · Response 83 · Craft 50 · Reach 58 · Headwinds 94 · Confluence 95 · Provenance 98
  measured requirement 635.16px (sum of li widths + six 16px gaps), 588px available
  -> Provenance wraps to row 2 (y=162 vs y=97); nav height 115px instead of 50px
```

Break-even walked on the live viewport, not calculated:

```
innerWidth 1146 -> centre 634px -> 2 rows, nav 115px
innerWidth 1147 -> centre 635px -> 2 rows, nav 115px
innerWidth 1148 -> centre 636px -> 1 row,  nav  50px
innerWidth 1280 -> centre 768px -> 1 row,  nav  50px
```

So **every width from 1100 to 1147 inclusive renders a two-row tab bar with `Provenance` alone on
the second row**, below the other six and above the tab-list's hairline rule.

Consequence, and it is the reason this is not cosmetic: `--sticky-h` is a **constant** 138px
(`page.ts:88`, inline on `main#content`), feeding
`scroll-margin-top: calc(var(--sticky-h, 116px) + 16px)` = 154px on every focusable control
(`shell.css:43`, commented "WCAG 2.4.11 — focus is never obscured by the sticky chrome"). At
1100px the sticky chrome actually obscures down to **y=187**. A control scrolled into view by
keyboard lands 33px **behind** the tab bar. The rule that names WCAG 2.4.11 fails 2.4.11 in a 46px
band of viewport widths.

**Not fixed here.** No source states whether the tab bar may wrap, and both plausible fixes invent
something the spec does not give: a scroll/overflow behaviour (which R35 rejects as a silent drop),
or a `--sticky-h` that measures itself at runtime (a mechanism, not a number). Named, measured,
open.

---

*RAT-1 and RAT-2 are binding as of 2026-08-02 and the tree agrees with both. RAT-1a, RAT-2a, RAT-3a
and RAT-3b are open and may not be closed by a build agent.*
