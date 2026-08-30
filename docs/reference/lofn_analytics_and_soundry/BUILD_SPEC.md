# BUILD SPEC — Lofn Analytics & The Soundry



*Produced by design fleet `wf_096531f8-d02` on 2026-08-01: 5 recon agents, a 7-seat data-viz/UX/statistics panel (after Tufte, Shneiderman, Amanda Cox, Heer, Gelman, Norman, Lupi), a 3-part moderated synthesis, and 2 adversarial critics. 17 agents, 0 errors.*



> The single-document synthesis blew the 64k output cap on the first attempt. It was split into three focused moderators (foundation / analytics / soundry) and resumed from cache.



## PART 1 — THE SHARED FOUNDATION (`apps/lofn-kit/`)

Everything in this part is built **once** and imported by both `apps/lofn-analytics/` and `apps/soundry/`. One token block, one chart renderer, one filter serializer, one insight object, one guidance contract, one empty/stale contract. If either app ever ships a second implementation of any of these, that is the defect — regardless of how the screen looks.

Two invariants govern the whole kit:

1. **The renderer refuses.** Honesty is a `validate()` pass that returns errors, not a rule somebody remembers. A chart that cannot state its `n`, its `as_of`, and its provenance **fails to render** and prints why, in ink, on the page.
2. **Absent is drawn.** Nothing in this kit is permitted to render a missing measurement as a zero, a flat line, a dash that reads as zero, or a collapsed card. `UNKNOWN` is a hatch with a word next to it.

House arithmetic that every number in both apps inherits (from the live pull, 2026-08-01): 210 clips · 374,122 plays · 5,913 likes · 3,926 followers · modern-cohort baseline like rate **4.49% (n=46)** · **58% of apparent between-song variance is binomial noise** · median song 258 plays = **±2.53pp at 95%**. Per-song verdicts are noise, and the kit is built so that a per-song verdict is hard to draw and easy to disbelieve.

---

# §1 · DESIGN TOKENS — `apps/lofn-kit/src/tokens.css`

One `:root` block, one file. Every other stylesheet in both apps **consumes and declares nothing**. The one legal exception is a value TypeScript must compute, written as an inline `style="--x: N"` and read by exactly one selector.

This is a deliberate, documented **extension of the Forge house pattern**, not a copy of it. Forge is dark-only by construction (`color-scheme: dark`, no `prefers-color-scheme` block anywhere, hardcoded derived grounds). The Scientist wrote "White background (or light/dark)". Light ships as the default and dark is a **separately specified set of values**, not an inversion. `filter: invert()` is a forbidden string in both apps' CSS, asserted by a test.

### 1.1 The block

```css
/* apps/lofn-kit/src/tokens.css
   THE ONLY :root DECLARATION IN EITHER APP.
   Semantic bindings are FROZEN — see tokens.md. tests/tokens-contrast.spec.ts
   and tests/palette-cvd.spec.ts fail the build on any change to a hex here. */

:root {
  color-scheme: light dark;

  /* ── GROUNDS (light, designed first) ───────────────────────────── */
  --paper:      #FAF7F2;   /* page */
  --card:       #FFFFFF;
  --sunk:       #F2EDE4;   /* rails, wells, table heads */
  --hairline:   #E3DCD1;   /* gridlines only  (1.27:1) */
  --rule:       #CFC5B7;   /* axes, dividers */
  --ink:        #1F1B17;   /* 16.0:1 */
  --ink-dim:    #6B635A;   /*  5.5:1 — smallest permitted text colour */
  --ink-faint:  #9C9186;   /*  2.9:1 — RULES AND HATCH ONLY, NEVER TEXT */
  --absent:     #B9AFA3;   /* always hatched, never a solid fill */

  /* ── CATEGORICAL QUARTET — four slots, verified, frozen ────────── */
  --m1: #302595;   /* night-indigo  L* 24   10.8:1 */
  --m2: #952A8E;   /* plum          L* 38    6.6:1 */
  --m3: #358982;   /* sea-teal      L* 52    3.9:1 */
  --m4: #EF7959;   /* ember         L* 64    2.6:1 → outline rule applies */
  --m1-edge: #241C70; --m2-edge: #6E1F69; --m3-edge: #256560; --m4-edge: #C1462A;
  --graphite: #706C68;   /* overflow, always labelled "other (n)" */

  /* ── SUNSET SEQUENTIAL — lightness is altitude in the sky ───────── */
  --seq-0: #F3EDE4; --seq-1: #ECD4B3; --seq-2: #F0AA6B; --seq-3: #F37337;
  --seq-4: #E22E34; --seq-5: #972977; --seq-6: #302595;
  /* L* 94 / 86 / 75 / 63 / 50 / 37 / 24 — strictly monotone */

  /* ── TIDE DIVERGING — encodes SIGN, never valence ──────────────── */
  --div-n4: #1A5954; --div-n3: #2E7872; --div-n2: #559790; --div-n1: #89BAB4;
  --div-0:  #EFE9E0;
  --div-p1: #D6A697; --div-p2: #DF6848; --div-p3: #CC3228; --div-p4: #9D1A20;
  /* L* 34/46/58/72/90/72/58/46/34 — symmetric, warm neutral midpoint */

  /* ── INFERENCE TRIAD — the only colours legal inside a stats plot ─ */
  --sig-pos:  var(--div-n4);   /* interval excludes the null, positive sign */
  --sig-neg:  var(--div-p4);   /* interval excludes the null, negative sign */
  --sig-null: var(--ink-faint);/* interval crosses the null — desaturated, always */

  /* ── CHROME-ONLY SUNSET (never inside a plot) ──────────────────── */
  --chrome-1: #E4762F;  /* wordmark */
  --chrome-2: #D94F6A;  /* focus ring, active tab underline */
  --chrome-3: #7B4B8A;  /* section accent */

  /* ── TYPE ───────────────────────────────────────────────────────── */
  --voice:   "Fraunces", "Iowan Old Style", Georgia, serif;
  --read:    "Atkinson Hyperlegible Next", ui-sans-serif, system-ui, sans-serif;
  --measure: "IBM Plex Mono", ui-monospace, "SF Mono", monospace;

  --t-micro:   10px;  /* axis micro, eyebrow */
  --t-tick:    12px;  /* ticks, chips, footnotes, tooltip body */
  --t-body:    14px;  /* body, table cells */
  --t-lede:    16px;
  --t-card:    19px;  /* card title, Fraunces 600 */
  --t-section: 24px;
  --t-tab:     34px;  /* Fraunces 600, WONK 1 */
  --t-hero:    56px;  /* Plex Mono 500, tabular */
  --t-page:    clamp(28px, 3.2vw, 44px);   /* the only clamp() in the system */

  /* ── SPACING (4-based) ──────────────────────────────────────────── */
  --sp-1: 4px;  --sp-2: 8px;  --sp-3: 12px; --sp-4: 16px; --sp-5: 20px;
  --sp-6: 24px; --sp-7: 32px; --sp-8: 40px; --sp-9: 56px; --sp-10: 72px;

  /* ── RADII / BORDERS ────────────────────────────────────────────── */
  --radius-chip: 2px; --radius-ui: 6px; --radius-card: 10px; --radius-pill: 999px;
  --radius-plot: 0;            /* a chart frame is a hairline, never a rounded box */
  --border-hair: 1px;
  --border-emph: 1.5px;        /* zero rules, reference lines */

  /* ── MOTION (two durations, full stop) ──────────────────────────── */
  --dur-state:  120ms;         /* hover, chip toggle */
  --dur-layout: 240ms;         /* tab change, filter re-render */
  --ease: cubic-bezier(0.2, 0, 0, 1);

  /* ── STATISTICAL CONSTANTS (mirrored in TS; one source) ─────────── */
  --ci-level: 0.95;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { /* values below */ }
}
:root[data-theme="dark"], @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #14110E; --card: #1D1916; --sunk: #0E0B09;
  --hairline: #332C25; --rule: #4A4038;
  --ink: #F4EFE7;      /* 16.4:1 */
  --ink-dim: #A79C90;  /*  7.0:1 */
  --ink-faint: #6E635A;/*  3.2:1 — rules only */
  --absent: #6E635A;

  --m1: #7E63DB; --m2: #D45CCA; --m3: #4BB0A7; --m4: #F1A38C;
  --m1-edge: #5D45AC; --m2-edge: #A6449E; --m3-edge: #368B84; --m4-edge: #B0674F;
  --graphite: #8C8177;

  --seq-0: #3A1F4A; --seq-1: #7A2F5E; --seq-2: #C2554F; --seq-3: #E9955A;
  --seq-4: #FBD79C; --seq-5: #FDEBCB; --seq-6: #FEF7EC;   /* order reversed: dark = low */

  --chrome-1: #F09A5B; --chrome-2: #EE7C93; --chrome-3: #A87BB8;
}}
```

`color-scheme: light dark` at `:root`, plus `color-scheme: dark` locally on `input[type=date]` in dark mode so native pickers match. The viewer's toggle writes `data-theme` on the root element and **must win in both directions** — the `:root[data-theme="light"]` guard in the media query is what makes that true.

### 1.2 Why the quartet is four, and why it is these four

Minimum pairwise CIELAB ΔE₀₀ across normal + protanope + deuteranope + tritanope simulation (Viénot 1999): **25.2 light / 17.4 dark**. Confusability floor is ~12. L* ladder 24 / 38 / 52 / 64 with a minimum gap of 12.1, so the scale also survives greyscale printing.

**Four, not five, not eight.** A six-slot all-sunset qualitative scale was attempted and fails: three warms collapse to ΔE 8.4 under deuteranopia and 4.3 under tritanopia. There is no honest six-warm qualitative scale. A fifth series becomes `--graphite` labelled `other (n)`. If a plot genuinely needs more than four identities, **it needs a facet, not a fifth hue** — `validate()` rule V8 enforces this.

Platform identity in particular is **never** hue-encoded: platform is a row label or a facet, and hue on Reach charts carries *provenance*, which has exactly four levels (authenticated-api / authenticated-page-read / public-page-read / search-snippet-provisional) plus never-pulled, which is hatch, not a hue.

### 1.3 Frozen semantic bindings (`apps/lofn-kit/src/tokens.md`, asserted by test)

| token | means, always, in both apps |
|---|---|
| `--m1` | our own work · baseline · control arm |
| `--m2` | external · competitor · another artist's surface |
| `--m3` | confirmed · authenticated · reconciled · passing |
| `--m4` | **attention** — recomputed-today, exploratory, needs a look. **Never "bad".** |
| `--graphite` | other (n) |
| `--sig-pos / --sig-neg / --sig-null` | interval position relative to the null. Sign, not valence. |
| `--absent` + hatch | never measured |

Two categorical dimensions may never share the quartet on one screen. **Modality is chrome, not colour**: shape glyphs `▤` Music · `■` Art · `▸` Animation. **Era is ordinal**: it takes four sampled stops of the sunset sequential ramp.

### 1.4 Ruling: "sunset colours" and "daily palette"

**Sunset is kept and given the two places where a warm hue sequence is perceptually legal:** the sequential ramp (where lightness carries the order — the pale end is the horizon, the dark end is the zenith) and the chrome (wordmark, active-tab underline, focus ring). It is **banned from categorical encoding** and from the inference triad.

**A rotating daily palette is a defect, not a style, and it is refused.** A scale that changes daily destroys the comparison of any screenshot, export, printout, or memory against any other — which is the entire purpose of an app you open every day. It is also unverifiable: one quartet can be checked against four vision models; 365 cannot.

**What the Scientist actually asked for is delivered instead, and it is better: THE DAY MARK** (§7.3). A 160 px almanac glyph in both app headers, redrawn every day from the day's real data state, using **zero new colours**, archived to `data/daymarks/YYYY-MM-DD.svg`. The app is different every day because the day is different. Nothing about it drifts, and its alt text is generated from the same tuple that draws it.

### 1.5 AWE / INDIGNATION — encoded without burning a colour channel

The duality is nominal, binary, and currently at **13.6% coverage (18 of 132 songs)** with **zero of those 18 joining the Suno snapshot by normalised title** — so its engagement n is 0.

- It **may not appear on any engagement axis** until a clip-id join exists. The card that would show it renders the join-spine diagram instead.
- Where it is legal (corpus composition, daily-run balance, Soundry arm labels) it uses **three redundant non-colour channels in a single ink**: shape (AWE hollow, 1.25 px stroke · INDIGNATION solid), fill, and a 6 px hairline riser (AWE above the mark, INDIGNATION below). Survives greyscale, all three CVD types, and a 96-dpi print.
- For quantitative comparison, use **position**: a paired dumbbell with AWE and INDIGNATION as two x-positions on a shared y, so the comparison reads as distance and hue stays free for era.
- **Neither pole may ever receive `--m3` or `--m4`.** If a reader concludes one is the pass state and the other the error state, the encoding has failed.

### 1.6 Type, self-hosted

Three families, one job each. **Fraunces** (variable, `opsz 144, wght 600, SOFT 0, WONK 1`) is the voice — wordmark, tab titles, card titles ≥19 px. **Atkinson Hyperlegible Next** (400/600) is the reading — chosen because both apps are required to carry tooltips, alt-text and guidance on every element, and this face was built for character disambiguation at small sizes. **IBM Plex Mono** (400/500, tabular figures) is the measurement: every number in a table, axis tick, `n =`, CI, `r`, id, or hex. A number inside a sentence is Atkinson; a number in a cell is Plex. Never mixed inside one cell.

Self-hosted woff2 in each app's `public/fonts/`, `font-display: swap`, latin subset, **total budget ≤ 200 KB**, `font-src 'self'`. No CDN — the CSP is sealed. Weights: 400 read · 500 measure · 600 voice · 700 eyebrow.

`.num { font-variant-numeric: tabular-nums lining-nums; }` on **every** element containing a digit, without exception. A metric that changes width when it updates cannot be compared to itself.

### 1.7 Chart-ink rules baked into `plot.css`

No 3-D, no pie, no donut, no gauge, no radial progress, no dual-y, no area chart of a rate, no gradient fill on a data mark, no drop shadow anywhere (elevation is paper tone; there is one `0 1px 0 var(--hairline)` under the sticky header at scroll and no other shadow token exists). Horizontal gridlines only, 1 px `--hairline`, behind marks; no vertical grid, no frame box, no axis line where gridlines already serve. Bar baselines are always zero. Any fill lighter than L\* 60 (i.e. `--m4`) carries a 1 px outline in its `-edge` token, and adjacent categorical fills get a 1 px `--paper` gap — the quartet members contrast only ~1.5:1 against *each other*, and that is how WCAG 1.4.11's 3:1 boundary requirement gets paid. Direct-label series; a legend is a failure state, permitted only above 3 series and then inline above the plot as coloured word-labels. Broken axes carry the **tear glyph** (two 6 px diagonals at 30°, 3 px apart, in `--rule`) — never an implied gap.

### 1.8 Accessibility floors (inherited from Forge, extended)

`button { min-height: 48px; min-width: 48px; }`. Secondary/link-buttons 44 px. Legend swatches are 9 px visually with a 44 px padded hit area (`padding: 10px 8px; margin: -10px -8px`) — applying the 48 px floor to a legend row destroys it. Focus: `outline: 2px solid var(--ink); outline-offset: 2px;` plus a 1 px inner `--paper` ring so it reads on any ground, and the house double ring (`outline: 3px solid var(--paper); box-shadow: 0 0 0 6px var(--chrome-2)`) on primary controls. `outline: none` is a forbidden string in both codebases, grep-asserted in CI. Text ≥ 4.5:1, chart marks ≥ 3:1 against their local ground. Breakpoints 1100 / 860 / 640 / 420, plus a `(pointer: coarse)` block. Every wide plot sits in an `overflow-x: auto` wrapper with `min-width` on the inner SVG; the page body never scrolls sideways.

---

# §2 · THE CHART GRAMMAR — `apps/lofn-kit/src/plot/`

No chart library, no CDN. The alternative to a library is not hand-written SVG per card — it is a small internal grammar: **a spec is data, a renderer is a pure function.** This buys three things this build needs:

1. **Charts become testable under `environment: "node"`.** `plot(spec) → string`. `expect(plot(spec)).not.toContain("NaN")` is a node test. The Forge DOM layer is structurally untestable; this sidesteps that entirely.
2. **Honesty becomes a validator.** `evidence` is required. An invented chart fails to compile.
3. **Cross-tab filtering falls out for free** — a plot declares which fields it honours and which it emits.

**Review rule: a card containing hand-written SVG is rejected on sight.**

### 2.1 `spec.ts` — types only, zero imports, zero DOM

```ts
export type Provenance =
  | "api-public"          // scripts/ingest_suno.py — public endpoint, reconciled
  | "authenticated"       // read from a logged-in page by The Scientist
  | "public-page"         // hand-read public profile counter
  | "search-snippet"      // provisional. permanent chip, dashed stroke.
  | "prose-archive"       // quoted from a markdown analysis; NOT re-derivable
  | "derived";            // computed here from one of the above

export interface Evidence {
  n: number;              // rows behind this mark set
  nEffective: number;     // THE SMALLEST CELL IN THE COMPARISON. This is what prints.
  asOf: string;           // the snapshot's pulled_at, ISO-8601 with offset
  provenance: Provenance;
  source: string[];       // repo-relative paths actually read
  snapshotIds: string[];  // pulled_at stamps of the exact files consumed
  reconciled: boolean;    // envelope reconciliation.ok
  caveat: string | null;  // the ONE thing most likely to mislead. Rendered visibly.
  query: string;          // re-runnable cohort expression, shown verbatim under the plot
}

export type Scale =
  | { type: "linear"; domain: [number, number]; zero?: boolean; clamp?: boolean }
  | { type: "log";    domain: [number, number] }             // rejects domain[0] <= 0
  | { type: "band";   domain: string[]; padding?: number }
  | { type: "time";   domain: [number, number]; unit: "month" | "day" }
  | { type: "ordinal";domain: string[]; range: string[] };   // colour only, max 4 levels

export type Fmt = "int" | "pct1" | "pct2" | "pp" | "ratio" | "dur" | "month"
                | "rate" | "r" | "raw" | "auto";  // "auto" = the significant-figure rule

export interface Channel { field: string; scale: Scale; title: string; unit: string; format?: Fmt; }

export interface Ref  { value: number; axis: "x" | "y";
                        kind: "zero" | "median" | "baseline" | "gate" | "mde" | "noise-floor";
                        label: string; }
export interface Note { x: number | string; y?: number; text: string;
                        kind: "outlier" | "note" | "bracket" | "tear"; }

export type EmptyKind = "never-pulled" | "measured-zero" | "filtered-empty"
                      | "not-applicable" | "below-floor" | "no-history";

export interface EmptyState {
  kind: EmptyKind;
  feed?: FeedId;
  headline: string;   // "No competitor scan has ever been run."
  why: string;        // names the exact missing file, by path
  how: string;        // the exact command, copyable, or the human decision required
  unlockAt?: string;  // "2 snapshots" | "5 named handles" — drives the ghost counter
  verbatim: boolean;  // true => contract-mandated wording, must not be paraphrased
}

export type Mark = "line" | "band" | "interval" | "slope" | "bar" | "spark" | "strip";

export interface PlotSpec {
  id: string;                 // stable; used for URL state and aria ids
  title: string;              // a SENTENCE stating the finding, never a noun phrase
  question: string;           // REQUIRED, distinct from title. Rendered as visible subtitle.
  mark: Mark;
  data: Row[];
  x: Channel;
  y: Channel;

  /* mark-specific */
  y2Field?: string;                        // band: the upper edge
  xLoField?: string; xHiField?: string;    // interval, variant "ci"
  xAField?: string;  xBField?: string;     // interval, variant "dumbbell"
  intervalVariant?: "ci" | "dumbbell";
  intervalStyle?: "line" | "grain";        // "grain" = THE GRAIN WHISKER (§7.2)
  stripPosture?: "rug" | "cloud" | "heat" | "bee";
  barOrient?: "v" | "h";
  stacked?: boolean;                       // gated by V10

  color?: { field: string; scale: Extract<Scale, {type:"ordinal"}>; title: string };
  facet?: { field: string; columns: number;
            scales: "shared" | "independent-x" | "independent-y";
            reason?: string };             // REQUIRED when scales !== "shared"; rendered
  detail?: string;                         // row identity for tooltip / aria / table
  nField?: string;                         // per-row n. interval & bar REQUIRE it.
  refs?: Ref[];
  annotate?: Note[];
  denominatorPlot?: string;                // REQUIRED when y is a rate or a median
  evidence: Evidence;
  deeper?: { href: string; label: string } | null;   // null is legal ONLY with a reason
  deeperAbsentReason?: string;
  filters?: FilterKey[];                   // FilterState keys this plot honours
  emits?: FilterKey | null;                // FilterState key this plot brushes
  height?: number;                         // default 220; spark 28; comb 34
  empty?: EmptyState;                      // REQUIRED whenever data may be absent
}
```

### 2.2 The seven marks

| mark | SVG | required channels | posture |
|---|---|---|---|
| **`line`** | `<polyline>` + `<circle r=3>` per datum | x linear\|log\|time\|band · y linear\|log | Trend across ordered positions. **No curve interpolation, ever** — with 14 monthly buckets at n = 2…43, a spline invents the values between measured points, and that invention is the lie. Points are kept: they are the accurate read and the hit target. |
| **`band`** | one `<path>`: upper edge forward, lower reversed, `Z`; fill `currentColor` @ 18%, no stroke | + `y2Field` | Uncertainty ribbon. Always drawn before its `line` in the same `<g>`. When n falls the band widens and the weakness is seen without a caption. |
| **`interval`** | `variant:"ci"` → whisker + 2× 6 px caps + `<circle r=4.5>`; `variant:"dumbbell"` → two dots + `--hairline` connector | x linear · y band · (`xLo`/`xHi`) or (`xA`/`xB`) · **`nField` mandatory** | **The forest plot and the drift dumbbell.** Horizontal only. `n` renders in a fixed right gutter, always. `n < 10` → hollow dot + dashed whisker + n in `--m4`. **Interval crossing the null → the whole row desaturates to `--sig-null` regardless of the point estimate.** `intervalStyle:"grain"` swaps the whisker for the Grain Whisker (§7.2). |
| **`slope`** | one `<polyline>` per series + end labels with collision nudge | x band (≥2 ordered positions) · y **must be a rate, share, or index** | Change across ordered positions. **V7 blocks raw counts** — a slope of raw counts across unequal windows plots the window, not the trend. This is the bump-chart slot; it never encodes rank. |
| **`bar`** | `<rect>`, 4 px rounded data-end, 2 px `--paper` gap between fills | band × linear-with-zero | Counts and shares. Horizontal when x is the measure. Stacked only under V10 (parts of one whole, sum 1.00 ± 0.01). Lollipop = bar with `width: 1px` + terminal dot. |
| **`spark`** | axis-free line or strip, ≤28 px, inline in a card header | x time · **≥3 points** | **Currently unrenderable — zero legal sparks exist.** The mark ships so the day the third snapshot lands the cards fill without a redesign, and so the refusal belongs to the grammar rather than to a developer's memory. Until then the slot renders the **series-depth glyph**: a dot strip of how many snapshots exist, captioned `1 snapshot · a trend needs 3`. |
| **`strip`** | `<circle r=2.5>` or 1 px `<rect>` per row, 0.55α so overplot reads as density | x required · y optional · colour optional | Four postures, one implementation: `rug` (1-D distribution — this is THE COMB), `cloud` (2-D scatter), `heat` (categorical census, colour = sequential ramp), `bee` (beeswarm, for a distribution behind a mean). |

Every mark emits `data-tip`, `data-key`, and a generated `aria-label`.

### 2.3 `validate.ts` — the honesty gate

`plot()` calls `validate()` first. On any error it renders a visible `<figure class="plot-invalid">` listing the errors in `--m4` ink. **It never renders a blank and never throws in production** — a malformed chart must be loud, because a silent one is indistinguishable from a true empty result.

| # | rule | why |
|---|---|---|
| V1 | `evidence` present; `n ≥ 0`; `nEffective ≤ n`; `asOf` non-empty; `provenance` in enum; `source.length ≥ 1` | no unsourced ink |
| V2 | `question` non-empty **and** `!== title`; `title` is a sentence (contains a verb-form token, ends without a bare noun phrase) | a chart with no question is decoration |
| V3 | `mark === "bar"` ⇒ measure scale has `zero: true` | truncated bar baselines misencode length |
| V4 | `mark === "spark"` ⇒ `x.scale.type === "time"` **and** `data.length ≥ 3` | **no time series exists on disk today; every spark renders its empty state** |
| V5 | `mark === "interval"` ⇒ a `kind:"zero"` or `kind:"baseline"` ref exists **and** `nField` is set | an effect size without a null line and an n is a rumour |
| V6 | `mark === "band"` ⇒ `y2Field` set and every row satisfies `y2 ≥ y` | |
| V7 | `mark === "slope"` ⇒ `y.format ∈ {rate, pct1, pct2, ratio}` | a slope of raw counts over unequal windows plots the window |
| V8 | `color.scale.domain.length ≤ 4` | past four categorical levels, facet |
| V9 | `y.field` matches `/_rate$|_median$|^median_|_share$/` ⇒ `denominatorPlot` set | a rate is never shown without its n |
| V10 | `stacked` ⇒ per-category shares sum to 1.00 ± 0.01 | the only legal stack is parts of one whole |
| V11 | `log` scale ⇒ `domain[0] > 0` and no datum ≤ 0 (zeros render as a 1 px `--ink` tick on the axis floor with a `0` label — a log axis must say what it did with them) | |
| V12 | `facet.scales !== "shared"` ⇒ `facet.reason` non-empty, **and the reason is rendered in the facet header** | independent scales must be argued for, in public |
| V13 | `data.length === 0` ⇒ `empty` present | a hole must be shaped |
| V14 | band-scale labels overflow their slot ⇒ error advising the horizontal orientation | **rotated axis labels are forbidden; a rotated label means the orientation is wrong** |
| V15 | `deeper === null` ⇒ `deeperAbsentReason` non-empty | the Scientist wrote "link to deeper results **if they exist**"; that conditional is how dead links ship. **Overruled to: every insight links, or states in words why it cannot.** |
| V16 | any `data` value used in an `interval` lacks a CI field ⇒ error | no point estimate is drawn anywhere in either app without its interval in the same block, at the same moment. Not a tooltip. |
| V17 | `evidence.provenance === "prose-archive"` ⇒ a `driftPair` note exists showing the prose value **and** today's recomputed value with both dates | the flagship cohort has already moved n=48 → 46 |
| V18 | multi-panel small multiple with `scales:"shared"` ⇒ every panel's computed domain is byte-identical | shared axes or it is not a small multiple |

### 2.4 Shared machinery

**`scale.ts`** (pure) — `makeScale(scale, [r0, r1]) => (v) => number` · `ticks(scale, target = 5)` · `fmt(v, format)`. Nice-number ticks 1/2/5 × 10ᵏ targeting 4–6; log ticks are decade powers, subdivided at 2 and 5 only when the domain spans under three decades (the plays axis spans 7 → 349,736, so it does not subdivide).

**`format.ts`** — the only place a number becomes a string, and the home of **THE SIGNIFICANT-FIGURE RULE**: with `format: "auto"`, round the estimate to the largest place value smaller than half the interval half-width. Half-width 2.53pp → print `4%`. Half-width 0.4pp → `4.4%`. Half-width 0.04pp → `4.49%`. **A number may never carry a digit its interval does not pay for.** Enforced in the formatter so an author in a hurry cannot bypass it. `pct1 → 4.5%` · `pp → +1.8 pp` (sign always) · `dur → 3:09` · `int → 374,122` · `ratio → 1.23×` (never a bare `123%`, which is ambiguous between +23% and 2.23×).

**`stats.ts`** — the four interval methods, and no fifth without a decision:
- `wilson(k, n, level)` — proportions (like rate). **The house default.**
- `bootstrapMedian(xs, level, iters = 10_000)` — cohort medians.
- `fisherZ(r, n, level)` — correlations.
- `logLinkGLM(y, treat, offset?)` → `{ beta, se, lift: Math.exp(beta), lo, hi }` — **the Scientist's specified lift method**: regress the metric on the binary treatment indicator with a log link, report `exp(β)` as the lift ratio and Wald SEs for the interval. General over mean / variance / proportion metrics. This is the DECIBAL estimator and it lives in the kit, not in either app.
- Plus `typeM(effect, se)` (expected exaggeration | significant) and `typeS(effect, se)` (probability of the wrong sign) and `mde80(n, baseline)`. These are not decoration: at the day-one Soundry design the exaggeration factor is ~4× and the sign-error probability ~5.5%, and the insight tier reads them.

**`CI_LEVEL = 0.95`** is a single exported constant, mirrored by `--ci-level`. Every interval in both apps is 95%, stated once in each app's Method tab and never re-litigated per card. The Soundry may additionally draw a 50% inner band on forest rows; it may not change the outer level.

**`axis.ts`** — `axisX`, `axisY`. Gridlines on the measure axis only. **The axis is drawn once per table or per small-multiple group, never per row and never per panel.** Fourteen rows share two axis rules.

**`legend.ts`** — `legend(colorChannel)` derived from the ordinal scale, never hand-authored (a legend that can disagree with its chart is a defect). Swatches carry shape as well as hue. Each sets `data-emit="<field>:<value>"`.

**`tooltip.ts`** — `bindPlots(root): () => void`. **One delegated listener for the entire page**, not one per mark. `pointerover` / `focusin` on `[data-tip]`, one `#plot-tip` node positioned via `getBoundingClientRect` and clamped to the viewport, dismissed on `pointerleave`, `focusout`, `Escape`, or the next tap. Touch: `click` on `[data-tip]`, persisting until dismissed. Full WCAG 1.4.13 (hoverable, persistent, dismissible). **Never `title=`.**

**`plot.css`** — consumes tokens, declares zero custom properties.

---

# §3 · CROSS-TAB FILTER STATE — `apps/lofn-kit/src/state/`

The Scientist's requirement is "all segmentations & filters carry between tabs when applicable, with clear controls." That is satisfied by having exactly **one place they live**, and by making non-application **visible** rather than silent.

### 3.1 Two scopes, two storage locations

- **GLOBAL** — the analysis. Stored in the **query string**, fixed key order so two URLs are diffable. Survives tab change, drill-down, reload, and share.
- **LOCAL** — the furniture (sort key/direction, expanded row, bin width, chart-type toggle, table page, scroll). Stored in the **hash fragment**, namespaced per tab: `#craft:sort=effect,dir=desc,exp=piano`. Resets on tab change, restored on return within the session, **excluded from a shared link** unless the user clicks *Copy link with view state*.

Rationale: sharing an analysis should not impose your scroll position on a colleague.

### 3.2 The global key table (identical vocabulary in both apps)

| key | control | values | default |
|---|---|---|---|
| `mod` | Modality | `music` · `art` · `anim` | `music` |
| `snap` | Snapshot | `latest` \| explicit `pulled_at` | `latest` |
| `co` | Cohort window (on `created_at`) | `all` · `modern` (2026-02→) · `YYYY-MM..YYYY-MM` | `modern` |
| `basis` | Measure | `response` (like rate) · `reach` (plays) | `response` |
| `out` | Outlier policy | `exclude` · `label` · `isolate` | `exclude` |
| `floor` | Play-maturity floor | integer 0–1000 | `60` |
| `nmin` | Ranking n-floor | integer 1–50 | `10` |
| `prov` | Provenance floor | `auth` · `+public` · `+snippet` | `+public` |
| `sel` | Cross-highlight | a facet value (style term, platform, arm) | none |
| `q` | Free-text search | string | empty |
| Soundry adds | `exp` round id · `hyp` hypothesis id · `arm` · `blind` (default `on` on Results) | | |

`floor` and `nmin` are different instruments and both ship: `floor` decides whether a *song's* rate is readable at all; `nmin` decides whether a *cell* may be ranked.

### 3.3 API

```ts
export type FilterKey = "mod"|"snap"|"co"|"basis"|"out"|"floor"|"nmin"|"prov"|"sel"|"q"
                      | "exp"|"hyp"|"arm"|"blind";
export type FilterState = Partial<Record<FilterKey, string[]>>;  // OR within, AND across
export function fromQuery(qs: string): FilterState;
export function toQuery(f: FilterState): string;                  // fixed key order
export function applyFilters(rows: Row[], f: FilterState, honoured: FilterKey[]):
  { rows: Row[]; dropped: number; byKey: Record<FilterKey, number> };
export function relaxations(rows: Row[], f: FilterState, honoured: FilterKey[]):
  Array<{ key: FilterKey; label: string; resultingN: number }>;   // precomputed, top 3
```

### 3.4 Display, applicability, reset

- **The chip strip is the only control surface.** One 32 px sticky row under the tab bar. Each chip names field, value, and resulting n: `cohort = modern · n 46`. Non-default chips carry a 2 px `--chrome-2` left edge. Default state shows one ghost chip: `Music · Response · outliers excluded`.
- **A global param that does not apply to the current tab is DIMMED, not dropped.** Tooltip: *"Cohort is not used on Reach — it applies again on Craft."* On return it is exactly as it was. **Silent dropping is the failure mode she is actually worried about, and it is invisible by construction unless the dimmed chip is drawn.**
- **A plot that honours none of the active filters renders a `not filtered` chip.** Nobody should have to wonder.
- **Per-chip `×` returns that one param to its default, not to empty.** Emptiness is a state with no data in it and must not be reachable by accident. `Reset to defaults` is disabled when already there.
- **Zero-result contract:** a filter state yielding 0 rows never blanks a panel. The panel ghosts and the precomputed `relaxations()` list appears — up to 3 one-click chip removals, each with its resulting n already computed. Every filter state has a next click.
- **Denominators, always.** Every card footer prints `n = <filtered> of <unfiltered>`. Above 50% removal the footer accent turns `--m4`.
- **Latency budget:** 210 rows / 367 KB. All aggregates precomputed at load; every repaint under 100 ms with **no spinner, ever**. A control that cannot meet that becomes an explicit `Apply` button. Slider drags render live and commit to history on release; `history.pushState` debounced 400 ms so a drag does not manufacture sixty entries. `popstate` re-reads.
- **Session trail:** last 10 global states, each restorable in one click, labelled by what changed (`cohort: modern → all`).
- **Cross-app:** the serializer is shared, so a headwind link in Lofn Analytics opens a Soundry Sonic Anvil seed with its filter intact. `basis=response` means the same thing in both apps.

---

# §4 · THE INSIGHT OBJECT — `apps/lofn-kit/src/insight/`

Every card in both apps renders one. There is no card without one, **including empty ones**.

### 4.1 Schema

```ts
export type Tier = "FINDING" | "PATTERN" | "OBSERVATION" | "NULL" | "ABSENT";
export type Magnitude = "MAKE_IT" | "TRY_A_PAIR" | "RUN_AN_EXPERIMENT"
                      | "STOP_SPENDING_ON_IT" | "PULL_THE_FEED" | "HOLD";

export interface Insight {
  id: string;                          // "music.form.words_per_line" — stable across pulls
  scope: { app: "analytics" | "soundry"; tab: string; card: string };

  headline: string;                    // GENERATED from a tier-locked template. <= 90 chars.
  subhead: string;                     // GENERATED. ALWAYS contains the interval and nEffective.
  tier: Tier;                          // COMPUTED, never chosen by an author.

  claim: {
    subject: string;
    measure: "like_rate"|"plays"|"upvotes"|"followers"|"count"|"r"|"lift";
    comparison: { baselineId: string; baselineValue: number; baselineAsOf: string };
    direction: "up"|"down"|"flat"|"unknown";
    pointEstimate: number;
    unit: "pct"|"pp"|"ratio"|"count"|"r"|"percentile";
    interval: { lo: number; hi: number; level: 0.95;
                method: "wilson"|"bootstrap-median"|"fisher-z"|"log-link-glm"|"none";
                iterations?: number };
    n: number;
    nEffective: number;                // the smallest cell in the comparison. THIS prints.
    kComparisons: number;              // family size
    qBH: number | null;                // Benjamini-Hochberg adjusted; null if no test ran
    mde80: number | null;              // smallest effect detectable at 80% power, this n
    typeM: number | null;              // expected exaggeration factor if "significant"
    typeS: number | null;              // probability of the wrong sign
    denominatorNote: string;           // "at a 258-play median, 4.49% is ~12 likes"
  };

  evidence: Evidence;                  // same type the PlotSpec carries
  quotedDrift?: { docValue: number; docN: number; docDate: string;
                  todayValue: number; todayN: number };

  controls: {
    eraControlled: boolean;            // REQUIRED true for any style/tag comparison
    outlierPolicy: "excluded"|"labelled"|"isolated"|"n/a";
    confoundsNamed: string[];          // non-empty for tier >= PATTERN
    randomized: boolean;               // true ONLY for a pre-registered Soundry round
    blockedOn: string[];               // e.g. ["day"] — the within-day randomisation
  };

  flags: Array<"NO_HISTORY"|"SINGLE_SNAPSHOT"|"WEAK_N"|"WEAK_R"|"NOISE_FLOOR"
             |"SUBSTRING_MATCH"|"ONE_PLATFORM"|"PROVISIONAL"|"UNCONTROLLED"
             |"PLACEMENT_ARTIFACT"|"SCHEMA_DRIFT"|"STALE"|"UNRECONCILED"|"POST_HOC_BAND">;

  nextMove: {
    directive: string;                 // <= 100 chars, imperative, names a Phase-0 lever
    lever: "form"|"style"|"duration"|"platform"|"cadence"|"ops"|"none";
    magnitude: Magnitude;              // BOUND TO TIER — not a judgement call
    routesTo: "lofn.phase0.axis"|"soundry.hypothesis"|"gates.proposal"|"ops.runbook"|"nothing";
    forbidden: string;                 // what this insight may NOT be used to justify. ALWAYS filled.
  };

  drill: { href: string; label: string } | null;
  drillAbsentReason?: string;
  altText: string;                     // GENERATED. A sentence with the finding, interval, n.
  generatedAt: string; generatorVersion: string;
}
```

### 4.2 Tier is computed, and the verb vocabulary is locked to it

The headline is **not written**. It is a template chosen by tier and filled from `claim`. **You cannot reach a stronger verb without reaching a stronger tier.** First match wins:

| Tier | Condition | Verb licence | Template |
|---|---|---|---|
| `ABSENT` | feed never pulled, or `data.length === 0` | none | *"No {feed} data has ever been pulled. This panel is empty because the snapshot does not exist — not because the numbers are zero."* |
| `OBSERVATION` | interval overlaps the null · **or** `method === "none"` · **or** style claim with `eraControlled === false` · **or** `reconciled === false` · **or** `typeM > 1.5` | *look like · show no separation · cannot be told apart* | *"{Subject} and {baseline} cannot be told apart at this n — {est}{unit}, 95% CI {lo} to {hi}, n={nEff}."* |
| `NULL` | interval excludes the null in **neither** direction **and** is narrow enough to exclude a practically meaningful effect on both arms (`max(|lo|,|hi|) < mde80`) | *does nothing · is not a lever* | *"{Subject} does nothing measurable — the effect is inside ±{mde}{unit} at 95% (n={nEff})."* |
| `PATTERN` | interval excludes the null **but** (`nEffective < 30` · or `|r| < 0.35` · or `qBH > 0.10` · or snapshots < 2 on a temporal claim) | *goes with · tends to · is higher in · is associated with* | *"{Subject} goes with a {direction} {measure} — {est}{unit} (95% CI {lo} to {hi}), n={nEff}."* |
| `FINDING` | interval excludes the null **and** `nEffective ≥ 30` **and** `qBH ≤ 0.10` **and** `eraControlled` **and** `reconciled` **and** (temporal ⇒ snapshots ≥ 2) | *raises · lowers · predicts* | *"{Subject} {verb} {measure} by {est}{unit} (95% CI {lo} to {hi}, n={nEff})."* |

**Causal verbs — `causes`, `drives`, `boosts`, `kills`, `makes` — are locked behind `controls.randomized === true`.** Nothing in Lofn Analytics is randomized, therefore **Lofn Analytics may never say "causes."** Only a pre-registered Soundry round unlocks that vocabulary. Both apps print this in the footer: *"Lofn Analytics observes. The Soundry tests. Only tested claims may say 'causes'."*

`magnitude` is bound to tier, so the size of the bet is not an opinion: `ABSENT → PULL_THE_FEED` · `OBSERVATION → RUN_AN_EXPERIMENT` · `NULL → STOP_SPENDING_ON_IT` · `PATTERN → TRY_A_PAIR` · `FINDING → MAKE_IT`.

### 4.3 The rules that stop over-claiming (each has a test)

1. **No point estimate on screen without its interval on screen**, same block, same moment. Not a tooltip.
2. **`nEffective`, never `n`.** "ambient · n=8", not "n=46".
3. **Denominator sentence is mandatory** when the cohort's median plays < 500: *"at a 258-play median, 4.49% is about 12 likes; the 95% interval is ±2.53pp."* This one sentence kills more false confidence than any error bar.
4. **Era gate.** Any style/tag comparison with `eraControlled === false` caps at `OBSERVATION` and its headline is prefixed **"Uncontrolled:"**. The dominant effect in this catalogue is time — engagement peaked 2026-05 at **5.10%** and fell to **3.38%** (06) and **3.56%** (07); month drift of 1–2pp exceeds most of the effects anyone wants to claim.
5. **Multiplicity is on the chart, not in a footnote.** A tab running k comparisons prints k and uses BH-adjusted q for tier promotion.
6. **Outlier gate.** Default measure everywhere is like rate. A `plays` card whose cohort contains Triple Arch renders the `PLACEMENT_ARTIFACT` banner and caps at `OBSERVATION`.
7. **Delta gate.** With one snapshot every delta is an em-dash + hatch + the word `UNKNOWN`. **Never `0`, never `0.0%`, never a flat sparkline.**
8. **Substring gate.** Every style-term insight carries *"matched as a substring of the style prompt; not proof it was audible in the render"*, with `drill` pointing at the render-audit corpus that proves that failure class.
9. **No stacked claims.** An insight may not compose two others ("ambient AND long lines = winner") unless the interaction was itself estimated. Interaction claims route to `soundry.hypothesis`.
10. **Type-M suppression.** When `typeM > 1.5` the point estimate is **replaced** by the interval and the MDE. Not annotated — replaced.
11. **Recompute-or-badge.** Every number is either recomputed live from a snapshot (carrying that `pulled_at`) or quoted from prose (carrying a **DRIFT** badge with both values and both dates). Never silently either.
12. **Freshness gate.** `asOf` older than 2× the feed's cadence prefixes the headline **"As of {date} (stale):"** (see §6.2).
13. **Degraded mode.** Any load invariant failure caps every tier at `OBSERVATION`, suppresses every absolute total, and pins a global banner.
14. **`forbidden` is always filled and always rendered** as the last line of a directive block — e.g. *"May not become a gate in `vault/gates.yaml` without a Soundry round."* Advisory means advisory; a metric never becomes a creative gate, never enters the ICB, and never argues the publish bar down.

### 4.4 Tab Directive

Each tab renders exactly one **Tab Directive** as its H1 subtitle at 28 px: the `nextMove.directive` of the argmax insight on that tab, ranked by `(tierRank, |interval.lo|, actionability)`, followed by the tier chip, the interval, `nEffective`, `[why this →]`, and the `forbidden` line. **A tab without a directive does not ship; a headwind without a next move is anxiety, not information.**

---

# §5 · GUIDANCE CONTRACT — `apps/lofn-kit/src/a11y/` + `plot/alt.ts`

### 5.1 The tooltip content object

Nothing in either app gets an info affordance unless it can fill this.

```ts
export interface GuidanceNote {
  term: string;          // the on-screen label, VERBATIM — never a paraphrase
  definition: string;    // <= 25 words, plain language. Never defines jargon with jargon.
  computed: string;      // the literal formula OR function name + file path. Required.
  basis: { n: number | null; nEffective: number | null; denominator: string;
           window: string; feed: FeedId; provenance: Provenance };
  caveat: string | null; // the ONE thing most likely to mislead.
                         // null is ILLEGAL for any derived statistic.
  deeper: { href: string; label: string } | null;
}
```

**Rules.**
1. **No decorative tooltips.** If `computed` and `basis` are both empty it is a label, not a tooltip — delete the affordance. An info icon that teaches nothing trains her to stop reading info icons, and then the one that mattered goes unread.
2. **Never `title=`** — invisible to most screen readers, unavailable on touch, unstylable, cannot hold a link. The trigger is `<button type="button" aria-expanded aria-controls>`.
3. **Three doors, one content:** hover, keyboard focus, and the **Glossary** in each app's Method tab — every `GuidanceNote` in the app on one printable page. If information only exists during a hover, it does not exist.
4. **WCAG 1.4.13 in full** — hoverable, persistent, dismissible with `Escape` without moving the pointer. 300 ms in / 100 ms out on hover; **zero delay on focus**.
5. **Tooltips are the second channel, never the first.** Anything with a non-null `caveat` also carries a persistent visible signifier: a superscript `†`, an inline `n=8` badge, a provenance glyph. A caveat reachable only by hover is a caveat designed not to be read.
6. **Never truncate.** Overflow opens a focusable popover containing the `deeper` link. Tooltips contain no interactive content.
7. **Every tooltip's last line is the provenance glyph and the as-of date.** No exceptions.

Shipping example:

> **Like rate** — upvotes divided by plays, shown as a percent.
> **Computed:** per clip at ingest (`scripts/ingest_suno.py`, field `like_rate`, 5 dp). The cohort figure is the **median of per-clip rates**, not the pooled ratio — the pooled one would be dominated by one clip.
> **Basis:** n = 46 clips created 2026-02-01 or later with ≥ 60 plays · snapshot pulled 2026-08-01 17:03 UTC · `data/snapshots/suno/latest.json` · api-public.
> **Caveat:** the median song has 258 plays. At that volume the 95% interval on a single song's rate is ±2.53 pp, and 58% of the apparent spread between songs is binomial noise. Per-song verdicts are noise.
> → *How this is computed*

### 5.2 Charts: the SVG is decorative, the data is in the DOM

```html
<figure role="group" aria-labelledby="p7-t" aria-describedby="p7-d">
  <figcaption id="p7-t"><!-- THE HEADLINE SENTENCE, not a noun phrase --></figcaption>
  <p class="plot-question"><!-- the visible question subtitle --></p>
  <p id="p7-d" class="sr-only"><!-- the four-sentence generated description --></p>
  <button class="plot-enter" aria-controls="p7-marks">Explore 24 data points</button>
  <svg aria-hidden="true">…</svg>
  <ol id="p7-marks" role="list" class="sr-anchored"><!-- one focusable <li> per mark --></ol>
  <details><summary>Data table (24 rows)</summary><table>…</table></details>
  <p class="plot-basis">n=46 (n-eff 8) · median like rate · 2026-02 onward · Triple Arch excluded
     · api-public · snapshot 2026-08-01 17:03 UTC</p>
  <p class="plot-caveat">†  …the caveat, visible, not hovered…</p>
</figure>
```

**Alt text is GENERATED from the spec, never hand-written** (hand-written alt drifts from its chart the first time the data changes). Four sentences, always in this order:

1. *Form and encoding.* "Horizontal dot plot with 95% confidence whiskers; 24 style terms down the vertical axis, like rate 0 to 7 percent across the horizontal."
2. *The reading.* "It asks whether any term in our own prompt language is associated with a higher like rate than the cohort baseline."
3. *Extremes and the middle, with numbers.* "Highest point estimate ambient at 5.55 percent, n equals 8. Lowest phonk at 3.18 percent, n equals 6. Cohort baseline 4.49 percent."
4. *The honest limit.* "No term separates from the baseline at this sample size; the widest interval spans 3.4 points. Full values in the data table below."

A plot whose spec cannot produce that string does not render.

**Data table is mandatory, not optional.** Same rounding as the chart, plus the columns the chart drops (`n`, CI bounds, as-of). `<details>` collapsed; `<summary>` states the row count. Sorting via a `<button>` inside `<th aria-sort>`, announced `aria-live="polite"`.

### 5.3 One keyboard model, learned once, identical on every chart in both apps

| Key | Action |
|---|---|
| `Tab` | one stop for the whole figure — lands on "Explore N data points" |
| `Enter` / `Space` | enter the mark layer (roving tabindex over the mark list) |
| `←` `→` | previous / next point within a series |
| `↑` `↓` | previous / next series, or previous / next facet |
| `Home` / `End` | first / last point |
| `Esc` | leave the mark layer, focus returns to the enter button — **never trapped** |
| `T` | toggle the data table |
| `?` | open this element's GuidanceNote |

On focus a mark gets the double focus ring plus a crosshair to both axes, and announces via `aria-live="polite"`: *"ambient. 5.55 percent. Interval 3.9 to 7.6. n equals 8. Below the ranking threshold. Point 3 of 24."*

**Selection by mark count** (chosen in `validate()`): ≤ 20 marks → each mark `tabindex="0" role="img"` with a generated label. > 20 marks → marks `aria-hidden`, the `<figure>` carries `role="img"` with the generated summary, and the data table **is** the accessible equivalent, not a description of one.

Per-tab focus order: skip-links (`Skip to filters` · `Skip to data tables` · `Skip to method`) → provenance bar → filter chips → cards in visual order. Focus never moves on its own except after an explicit user action, and never into a newly-rendered region without an `aria-live` announcement of what changed.

### 5.4 Colour-vision safety, as a CI gate

`apps/lofn-kit/scripts/validate-palette.mjs` renders the token palette, pushes it through deuteranopia / protanopia / tritanopia simulation matrices (Viénot 1999) and greyscale, then asserts:

- CIEDE2000 ≥ 20 between every categorical pair in every simulation, both themes;
- sequential ramp L\* strictly monotone in every simulation;
- text ≥ 4.5:1, UI components and graphical objects ≥ 3:1;
- every mark ≥ 3:1 against its local ground, **including the hollow unpowered outline**, which is the mark most likely to fail and whose visibility carries the most meaning.

A palette change that fails the test **fails the build**. She will trust a test; she should not have to trust anyone's eye. A **Simulate colour vision** control (all three simulations + greyscale) ships in each app's Method tab as a self-audit affordance.

Colour is never the sole encoding: every categorical hue is doubled by shape (● ■ ▲ ◆) and every line by linetype (solid / dashed 6-3 / dotted 2-3 / dash-dot); direction is doubled by glyph (▲ ● ▼); provenance is doubled by stroke (solid / 1 px outline / 2 px dash / em-dash-no-mark).

### 5.5 Motion

Two durations: `--dur-state 120ms` (hover, chip toggle) and `--dur-layout 240ms` (tab change, filter re-render), both `--ease`.

**Data marks never animate on load or on a data change.** An entrance animation on a bar chart tells a lie for 400 ms, and with this dataset the first thing a viewer would perceive is growth that does not exist. The only animated data transition permitted is a re-sort or re-scale of the **same rows** — position and opacity only, never a morph between mark types, never a tween of an interval's width (a widening CI reads as the data changing when it is only the page loading).

Banned outright: autoplay, carousels, looping, parallax, scroll-jacking, counting-up numbers, staggered reveals, skeleton shimmer shaped like a chart (a loading state must never look like data). Loading is a 1 px `--rule` progress hairline under the header, with a determinate `<progress>` when a count is known (`page 4 of 11`) and text via `aria-live`.

`@media (prefers-reduced-motion: reduce)` clamps everything to 1 ms, plus a **visible in-app toggle** writing `.reduce-motion` — the app must not require you to know where the OS setting lives. **No motion is ever load-bearing**; if an animation is the only way to perceive a change, a static before/after ships too.

---

# §6 · EMPTY-STATE AND STALE-DATA CONTRACTS — `apps/lofn-kit/src/contracts/`

### 6.1 Empty must never look like zero

**When there is no data the mark layer is not drawn at all** — only the frame and the hatch. No zero-height bar, no flat line at y=0, no `0%`, no plotted origin point, no needle at rest, no collapsed card.

| kind | meaning | visual | copy pattern |
|---|---|---|---|
| `never-pulled` | no snapshot has ever existed | dashed border, 45° hatch at 4 px pitch, **axes not drawn** | *"No {Platform} data has ever been pulled."* + why + the exact command + the file path it waits on + what will appear |
| `measured-zero` | the pull succeeded and the true value is 0 | **real axes, real zero line**, a labelled zero tick in solid `--ink` | *"0 units. This is a measured zero, not a missing reading — DistroKid reports no iTunes/Amazon sales in any window, as of 2026-08-02."* |
| `filtered-empty` | data exists; these filters match nothing | real axes, ghosted, active chips inline | *"0 of 210 works match these filters. Widest filter: cohort ≥ 60 plays. [Relax to ≥ 50 → n 58] [Reset]"* |
| `not-applicable` | the metric cannot exist for this slice | greyed with a lock glyph, no axes | *"Like rate does not exist for Animation. No feed measures engagement on video works."* |
| `below-floor` | rows exist but under the n-floor | drawn at reduced ink + banner | *"3 of 24 cells are below the n=10 ranking floor. They are shown, unranked, with their n."* |
| `no-history` | the chart needs a series and there is one point | **the axes ARE drawn, the future shape is visible**, a dotted horizon to a hollow terminal ring labelled `UNKNOWN`, plus the unlock counter | *"1 snapshot. A delta needs two; a trend needs three."* |

`tests/empty-vs-zero.spec.ts` asserts the absent-hatch fill and the measured-zero tick **never share a fill string**, in either theme.

**Every empty state carries a verb** — the exact next action, as a copyable command or a button, plus the file path. An empty state that only apologises is a design failure.

**The verbatim texts already written in the ingest contract for `suno`, `competitors`, `market`, `soundry`, and each social sub-feed ship as-is** (`empty.verbatim = true`, and a test asserts the string is not paraphrased). They are better than anything a redesign would produce, and they name the command that fills the panel.

### 6.2 Stale data

Freshness is a first-class property of every number, rendered next to it, never in a footer. **Age chip:** relative *and* absolute, tabular-nums — `12h ago · 2026-08-01 17:03 UTC`. Relative-only is an evaluation-gulf failure across timezones and long gaps.

| tier (daily feed: <36 h / 36 h–7 d / 7–30 d / >30 d) | treatment |
|---|---|
| **FRESH** | normal ink; age chip in `--ink-dim` |
| **AGING** | age chip promoted to `--chrome-1`; no banner |
| **STALE** | persistent card banner: *"This is a 9-day-old reading (pulled 2026-07-23 14:10 UTC). Nothing that happened after that date is in it. [Refresh]"* — **the number renders at full ink.** Stale is old, not wrong; greying an accurate historical figure makes it read as broken and teaches distrust of correct numbers. Any delta across the gap is labelled with the gap length. |
| **EXPIRED (>30 d)** | inescapable `as of` prefix: *"as of 2026-03-28 (126 days) — 8.3K plays"*. Every trend, delta, sparkline, and present-tense framing is **suppressed, not zeroed**. |

Weekly-cadence feeds use 8 d / 21 d / 60 d.

Additional hard rules:

- **"Now", "current", "today", and the present tense are reserved words**, unavailable to any number past its FRESH threshold. Copy switches to "as of {date}". Enforced by a copy lint over the generated headline corpus.
- **As-of mismatch band.** If two feeds share an axis and their ages differ by more than the fresher one's FRESH window, the chart draws a labelled mismatch band and both dates appear. This is what stops the six-week-old `tracking/creator-platforms-latest.json` (which has no timestamp field of its own) from silently sharing an axis with today's Suno pull.
- **Never carry forward, never interpolate.** A missing point is a **break in the line**, annotated with the gap length. Not a repeated value, not a straight segment.
- **Refresh states its consequence before acting:** *"Pull Suno — about 11 public HTTP requests to studio-api.prod.suno.com over ~4 s. No credentials sent. Writes one new file under data/snapshots/suno/. Does not publish, render, or spend."* Disabled buttons always say why.
- **UNRECONCILED outranks stale.** If `reconciliation.ok === false`, a global banner pins to every tab, every absolute total is replaced by the word **SUPPRESSED** with a link (never a number, never a zero), and per-row values survive with a per-row warning. This gate exists in the UI because the ingest script writes `latest.json` even on a failed reconciliation.
- **Read-only is permanent chrome, in both apps:** *"This app reads. It never posts, renders, or spends."* That is `vault/AUTONOMY.md` made visible instead of buried. Any render/publish/spend control that exists at all exists **disabled, with its reason inline** — it never silently isn't there and it never acts.

### 6.3 Load-time invariants and the error contract — `apps/lofn-kit/src/load/`

`loadSnapshot(path)` opens with `utf-8-sig` / BOM strip (a sibling artifact in this repo carries a BOM and hard-crashes a naive parse), applies a **schema discriminator** before anything else (`feed` + `handle` + presence of `rows[]` + `stats`), and **refuses** any file matching the stale 2026-03-28 shape whose `plays` is the string `"8.3K"`. Refusals are logged to a visible Schema Guard panel — a silently-eaten file is impossible.

Invariants I1–I10 run at load. Each failure has its **own** screen naming: what happened · what the system did with the data (kept / discarded) · what state you are in now · the one next action. No generic "failed to load".

**I8 renders, verbatim, and the string "songs deleted" is a forbidden literal asserted by test:** *"10 clip ids present in the 2026-07-27 analysis are absent from today's snapshot. Cause UNKNOWN — could be trashed, unpublished, or a header undercount. Plays went up over the same span. [Show the 10 ids]"*

---

# §7 · SIGNATURE DEVICES

Two signatures, one per app, plus one shared chrome device that answers the daily-palette mandate. Both signatures are **mark variants inside the grammar**, so they live in the kit and cannot drift from the renderer.

### 7.1 THE COMB — Lofn Analytics · `signature/comb.ts`

210 hairlines, 1 px, `--ink` at 70% α. **34 px tall in the Catalog hero, 22 px in a persistent strip under the filter bar on every tab.**

- `x` = log₁₀(`play_count`), domain 1 → 1,000, ticks at 10 / 100 / 1000 in Plex Mono 10 · baseline 1 px `--rule`.
- Then the **tear glyph**, then a 44 px gutter holding the single outlier line at full height in `--absent` with an `--m4` label. **One clip never sets a scale.**
- Collisions stack with 6 px jitter. Selected clips full ink; unselected at 12% α.
- Brushing sets the global `co` / `floor` filters; the brush handle is a 2 px `--m4` rule.
- Caption, Plex Mono 10: `210 clips · 1 off-scale · median 258 plays`.

*Why it is the signature:* it draws the entire population as ink rather than the top ten as bars, and it renders the app's central argument as a physical object you cannot un-see — a dense wall of songs under a thousand plays with one clip literally off the paper. It doubles as the cross-tab filter indicator, satisfying the carry-through requirement with ink instead of a chip. No template dashboard has ever drawn this. Implemented as `mark: "strip", stripPosture: "rug"` with `annotate: [{kind:"tear"}]`.

### 7.2 THE GRAIN WHISKER — The Soundry · `signature/grain.ts`

Every interval in the Soundry is a **10 px-tall rectangle**, not a line, filled with a 45° hatch whose **pitch = `clamp(2 + 26/√n, 2, 14)` px**, stroked 1 px in `currentColor`, no solid fill behind it.

`n=400 → 3.3 px` (reads nearly solid) · `n=100 → 4.6 px` · `n=25 → 7.2 px` (visibly open) · `n=9 → 10.7 px` · `n=4 → 14 px` (two lonely strokes).

The point estimate is a **9 px vertical ink rule** (not a dot — a dot competes with the hatch) plus a 3.5 px disc in the Tide colour for sign. The **Power Notch** is a 1 px `--m4` tick on the axis at the pre-registered MDE; an interval wider than 2× MDE labels its row `UNDERPOWERED — cannot resolve the effect you asked for`. Zero line: 1.5 px solid `--ink`, full height, at the same x on every row. **It never animates.**

*Why it is the signature:* uncertainty is normally the thinnest ink on a statistics page. Here it is the loudest texture, and you feel the sample size in your hand before you read the `n`. Given that 58% of apparent between-song variance in this catalogue is binomial noise, texture-as-sample-size is not decoration — it is the finding. Implemented as `intervalStyle: "grain"`.

### 7.3 THE DAY MARK — shared chrome · `signature/daymark.ts`

160 × 160, in both app headers, redrawn every day from the day's real data state, using **zero new colours**.

- **Outer ring:** 24 ticks, one per hour. The hour of the most recent successful pull is inked `--ink`; every other hour is `--hairline`.
- **Inner arc:** one segment per known feed. `--m3` filled if `latest.json` exists and reconciled · `--m4` outlined if it exists and failed reconciliation · `--absent` + hatch if never pulled.
- **Core:** a disc, `r = log₁₀(rows in the newest snapshot) × 8 px`, filled at the sunset-sequential stop for the cohort median like rate.
- One `--m4` hairline struck across the whole mark if any invariant failed.
- Caption, Plex Mono 10: `2026-08-01 · 3/9 feeds · 210 rows · recon OK`.
- Archived to `data/daymarks/YYYY-MM-DD.svg`. A year is a wall of 365 postcards.
- Alt text generated from the same tuple that draws it.

This is the daily change she asked for, delivered as data instead of as a rotating scale.

---

# §8 · FILE LAYOUT AND BUILD WIRING

The kit is a third app-shaped directory — plain TypeScript source, devDependencies only, no publish step, no workspace tooling. Both apps import it through a Vite alias. **No `dependencies` key. Ever.**

```
E:/git/private-claw/apps/lofn-kit/
  package.json                 # name lofn-kit, private, type module, devDeps only:
                               #   @types/node ^24.13.3 · typescript ^5.5.0 · vitest ^3.0.0
  tsconfig.json                # target ES2022, moduleResolution bundler, strict, noEmit,
                               #   isolatedModules, resolveJsonModule, lib ES2022+DOM
  vitest.config.ts             # environment: "node", include tests/**/*.spec.ts
  README.md                    # what it is, who edits it, what a new mark costs
  .gitignore                   # node_modules/  dist/  runs/

  src/
    tokens.css                 # §1 — THE ONLY :root BLOCK IN EITHER APP
    tokens.md                  # the frozen semantic bindings; the contrast/CVD measurements
    theme.ts                   # readTheme() / setTheme() / prefers-* wiring; no hex literals
    format.ts                  # fmt(), the significant-figure rule, tabular helpers
    plot/
      spec.ts                  # types only. zero imports.
      validate.ts              # V1–V18
      scale.ts                 # makeScale / ticks
      axis.ts                  # axisX / axisY / tearGlyph
      marks/line.ts  band.ts  interval.ts  slope.ts  bar.ts  spark.ts  strip.ts
      legend.ts
      alt.ts                   # the four-sentence generator
      table.ts                 # the <details> data-table generator
      plot.ts                  # validate() then render; pure PlotSpec -> string
      tooltip.ts               # ONE delegated listener per page
      keys.ts                  # the keyboard model
      plot.css                 # consumes tokens, declares none
    state/
      filters.ts               # FilterKey, FilterState, applyFilters, relaxations
      url.ts                   # fromQuery / toQuery / hash namespace / pushState debounce
      chips.ts                 # chip strip rendering, dimmed-not-dropped applicability
    insight/
      insight.ts               # the Insight type
      tier.ts                  # computeTier() — the table in §4.2
      headline.ts              # tier-locked templates; the causal-verb lock
      stats.ts                 # wilson / bootstrapMedian / fisherZ / logLinkGLM
                               #   / typeM / typeS / mde80 ; CI_LEVEL = 0.95
    contracts/
      empty.ts                 # EmptyState renderers, one per kind
      empty-text.ts            # the VERBATIM ingest-contract strings, frozen
      stale.ts                 # tiers, age chips, reserved-word lint helper
      provenance.ts            # glyph + chip + ordering
    load/
      snapshot.ts              # utf-8-sig read, envelope parse
      schema-guard.ts          # discriminator; the refusal log
      invariants.ts            # I1-I10, each with its own failure screen text
    signature/
      comb.ts  grain.ts  daymark.ts
    a11y/
      sr.ts                    # sr-only, aria-live announcer
      focus.ts                 # roving tabindex, escape-never-trapped

  tests/                       # environment: "node"; import { describe, expect, it } only.
    tokens-contrast.spec.ts    # every text token >= 4.5:1, every mark >= 3:1, both themes
    palette-cvd.spec.ts        # CIEDE2000 >= 20 all pairs, all 3 simulations, both themes
    palette-frozen.spec.ts     # restates every hex literally — a change is a two-file edit
    plot-validate.spec.ts      # one it() per rule V1-V18, named as a sentence
    plot-render.spec.ts        # every mark x {0,1,n} rows: has <figure, has aria-label,
                               #   contains no NaN / undefined / Infinity
    scale.spec.ts              # log rejects <=0; band rejects dupes; ticks() 4-6 over 40 domains
    format-sigfig.spec.ts      # 4.49 +/- 2.53 -> "4%"; 4.49 +/- 0.4 -> "4.5%"
    filters-url.spec.ts        # round-trip, fixed key order, per-chip reset to DEFAULT not empty
    insight-tier.spec.ts       # the tier table, incl. typeM>1.5 demotion
    headline-vocab.spec.ts     # NO causal verb reachable with randomized:false
    empty-vs-zero.spec.ts      # absent-hatch fill !== measured-zero fill, both themes
    empty-verbatim.spec.ts     # the contract strings are byte-identical
    stale.spec.ts              # reserved words unavailable past FRESH
    invariants.spec.ts         # I1-I10 against the real data/snapshots/suno/latest.json
    alt-text.spec.ts           # generated alt contains mark, both axes, n, and one number
    no-forbidden.spec.ts       # greps src+consumer CSS for "outline: none", "filter: invert",
                               #   "songs deleted", "vs avg artist", raw hex in .ts
    signature.spec.ts          # comb tear glyph present; grain pitch(n) monotone decreasing

  scripts/
    validate-palette.mjs       # the CI colour gate; exit 1 on any failure
```

### 8.1 Consumer wiring (identical in both apps)

```ts
// apps/<app>/vite.config.ts  — additions only
import { defineConfig } from "vitest/config";   // NOT "vite"
resolve: { alias: { "@kit": join(HERE, "..", "lofn-kit", "src") } },
server:  { fs: { allow: [HERE, join(HERE, "..", "lofn-kit")] } },
```
```jsonc
// apps/<app>/tsconfig.json — additions only
"paths": { "@kit/*": ["../lofn-kit/src/*"] }
```
```html
<!-- every page head, verbatim shape -->
<meta http-equiv="Content-Security-Policy"
  content="default-src 'self'; connect-src 'none'; img-src 'self' data:; media-src 'self';
           font-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'" />
<link rel="stylesheet" href="/@kit/tokens.css" />
<link rel="stylesheet" href="/@kit/plot/plot.css" />
```
The `csp-per-mode` Vite plugin widens `connect-src` to `'self'` + the HMR socket in dev, `'self'` when hosted, and leaves `'none'` in retail. **The Soundry is retail-sealed in every posture** — it is local-only, Scientist + invited collaborators.

### 8.2 CI (`.github/workflows/lofn-apps-ci.yml`), paths-filtered to `apps/**`

```
lofn-kit:      npm ci → typecheck → test → node scripts/validate-palette.mjs
lofn-analytics: npm ci → typecheck → test → build (retail, sealed)
soundry:        npm ci → typecheck → test → build (retail, sealed)
```

### 8.3 Repo preconditions (blocking, before the first file)

1. **`.gitignore` currently has `data/catalog/` and `data/archive/*` but NOT `data/snapshots/`.** The snapshot embeds the Suno `user_id`, follower counts, and the full style prompt of all 210 clips — our complete unreleased style-prompt corpus. Add `data/snapshots/` and `data/daymarks/` (or all of `data/`) **before** anything commits. See open question 1.
2. **Build-fleet interlock.** `python3 scripts/run_lock.py acquire apps/lofn-kit --run-slug <slug>` as the first action of this part, heartbeat per lane wave, release after the build report. The lock file goes in a **sidecar** `apps/<slug>/.lofn/.RUN_LOCK`, not in the build tree, so Vite/eslint/git never glob it. Exit 3 means STOP.
3. **Serialized writes.** Each build lane writes only its own files. `package.json`, `tsconfig.json`, the vite configs, and any index are written by the coordinator alone, single-threaded, after a lane wave lands. Six concurrent appenders to one file is a corruption the harness must make impossible by construction.
4. **CANONICAL PATHS keys** `APPS`, `APP_KIT`, `SNAPSHOTS`, `APP_LOCK` must be added to `.claude/skills/lofn/SKILL.md` **and** the `.agents/` twin before either app names its own storage — an undeclared path is a repair blocker under `EXECUTION.md`.

.

# THE SOUNDRY — `apps/soundry` (soundry.localhost)

**Part 3 of 3. The shared foundation — the `PlotSpec` grammar and its `validate()` rules, the `Insight` object and its tier ladder, the token block, the empty-state and stale-data contracts, `input-cleaning.mjs`, the four-state absence rules — is specified elsewhere and is imported, not restated. This part specifies only what is Soundry's own.**

Three tabs, as she drew them: **Sonic Anvil · Experiment Results · Analytics**. No fourth tab.

One sentence of permanent chrome, present on every tab, at 11px under the header:

> *This bench writes prompts and plans experiments. It renders nothing, posts nothing, spends nothing. The bench proposes; the pipeline makes.*

---

## 0 · What the Soundry actually is, and the one thing it must not become

The Soundry is a **measurement instrument pointed at our own catalogue**. Its output is not songs — the `lofn` pipeline makes songs. Its output is **a ledger of pre-registered contrasts with intervals attached**, and a promotion route back into Phase 0.

Everything below is governed by one measured fact, and I want it stated before any UI:

> At our median of **258 plays per song**, a single song's like rate carries a **±2.53 pp 95% interval** on a **4.49 pp** baseline. **58% of the apparent between-song spread is binomial noise.** A per-song verdict is a coin flip with a title.

So: the Soundry never ranks the twelve, never declares a daily winner, and never prints a point estimate it cannot pay for. Its unit of knowledge is the **arm**, not the song, and its unit of time is the **round**, not the day.

### 0.1 The AUTONOMY boundary, mechanically

| | |
|---|---|
| **Writes** | `data/soundry/rounds/<round-id>/**` (the ledger) · `output/soundry/<round-id>/*.md` (paste-ready prompt packets) · `data/snapshots/soundry/<stamp>.json` + `latest.json` (the feed Lofn Analytics reads) |
| **Reads** | `data/snapshots/suno/*.json` (its own outcome instrument) · `vault/gates.yaml` (advisory display only) · its own ledger |
| **Never** | calls Suno, calls any generation or render API, posts, uploads, DMs, or spends. Reads **no** API key. There is no `POST /soundry/render` route and no Render control anywhere in the UI — see §1.8. |
| **CSP** | `connect-src 'self'` in all three build postures. The only fetch target in the bundle is `http://127.0.0.1:<port>`. No CDN, no network font, no chart library. |
| **Bind** | `server.mjs` binds `127.0.0.1` only. No `access-control-allow-origin: *`. `soundry.localhost` resolves via the operator's hosts file; the app works identically at `http://127.0.0.1:5231/`. |
| **Posture** | **Local-only.** There is no hosted build. `scripts/check-local-only.mjs` walks `dist/**` and fails the build if any `.html`/`.js` contains an absolute `http(s)://` origin other than the CSP declaration, or the strings `render`, `publish`, or `upload` bound to a button handler. This is the Forge `hosted:check` inverted: the Forge proves the founder page is *absent* from a hosted bundle; the Soundry proves the *whole app* never leaves the machine. |

### 0.2 The join key the repo does not have, created here

Recon is unambiguous: **nothing on disk maps a Suno clip uuid to a run artifact.** Structured title matching tops out at 40.9%. The Soundry cannot analyse anything without that join, so it **creates it at the one moment the information exists** — when the Scientist posts a variant.

Each variant card has a **`clip_id` binding field**. She pastes the Suno clip URL or uuid back into the card after posting; the app extracts the uuid, records `posted_at`, and writes it to the variant record. **A variant with no bound `clip_id` can never enter an analysis.** It renders in the maturity board as an open ring labelled `UNBOUND — no outcome can be attributed to this card`.

This is one paste per song, twelve pastes a day, and it is the entire reason the Soundry can answer questions the rest of the estate cannot. Make the field the largest interactive target on the card.

---

## 1 · TAB 1 — SONIC ANVIL

Route ladder, all URL-backed: `#/anvil` → `#/anvil/design` → `#/anvil/prereg` → `#/anvil/cards`. You cannot skip forward; the next step's chip is disabled with a stated reason until its precondition holds.

### 1.1 A0 — The ask

Her big input, unchanged. A visible `<label>` (never a placeholder — placeholders vanish on focus), Fraunces 32/600 over a 2px `--rule` underline that goes ember on focus.

Below it, **seed chips derived live from Lofn Analytics headwinds**, each carrying the source card's URL so the *motivation* for a round is provenanced, not remembered:
- *"The audience we can see is 45+ by 663 to 84. Test a register that is not youth-coded."*
- *"Median words-per-line 6.3, r = −0.30 and weak. Test it properly."*
- *"Engagement peaked 2026-05 at 5.10% and has fallen twice. Test whether anything we changed caused that."*

The button reads **"Draft a design"** and states its consequence inline: *"Opens the design step. Writes nothing yet."*

### 1.2 A1 — Factor declaration, and the two legal round modes

Free text is parsed into **candidate two-level factors** — `words_per_line {≤5 | ≥7}`, `chorus {exact repeat | varied}`, `duration {~200s | ~255s}`, `emo_tag_count {≤6 | ≥12}`, `register {plain | ornate}` — each editable, each requiring both level definitions in words before it can be selected.

Then the round mode. **Exactly two, and the difference is not cosmetic:**

| | **CONFIRM** (default) | **SCREEN** |
|---|---|---|
| Factors | **1** | 2–11 |
| Design | 2 arms × 6 replicates, **randomised within day, blocked on day** | 12-run Plackett–Burman, blocked on day |
| Resolution | Full — the contrast is clean | **III — main effects are aliased with two-factor interactions** |
| Multiplicity family | k = 1 | k = number of factors, BH-adjusted |
| Highest reachable Insight tier | **FINDING** | **PATTERN**, permanently capped |
| What it can output | a confirmed lift with an interval | a ranked shortlist of factors worth a CONFIRM round |
| Banner it always carries | — | *"This screens. It does not confirm. No SCREEN result may be adopted, promoted, or written into a gate."* |

**Why CONFIRM is the default, against the panel's PB-12 enthusiasm:** PB-12 genuinely buys eleven answers for twelve songs, and that is real. But it buys them at Resolution III, and with k=11 the BH-adjusted n pushes a +15% detection from ~19 days to ~37. With **zero** maturity curve on disk and **one** Suno snapshot, the first rounds must be the ones whose answer is unambiguous. SCREEN unlocks after the maturity curve exists (§2.3) — the mode chip is disabled before then with exactly that reason.

### 1.3 A2 — The power card, computed before a single prompt is written

This is the whiteboard's "Include POWER ANALYSIS", taken literally enough to change the design while it is still editable.

**The model the app implements** (constants frozen in `src/design.data.json`, each with a provenance comment, each re-estimated from the accumulating Soundry corpus once ≥2 rounds close — the card draws both the seeded and the re-estimated curve):

```
p̄        = 0.0449          baseline like rate, modern cohort n=46, snapshot 2026-08-01
σ²_true  = 1.204e-4        true between-song variance (σ_true = 1.097 pp)
σ²_tot(m) = σ²_true + p̄(1−p̄)/m         m = plays per song at readout
SE_log(n,m) = sqrt( 2·σ²_tot(m) / n ) / p̄        n = songs PER ARM
MDE_ratio(n,m) = exp( 2.802 · SE_log )           2.802 = z_.975 + z_.80
days = ceil( 2n / cards_per_day )
```

These are not invented: at m=258 the binomial term is 1.662e-4, which is **58% of σ²_tot** and gives a per-song 95% interval of ±2.53 pp — both measured facts, reproduced. And the curve lands exactly on the two timelines already established: **n=114/arm at m=258 → MDE +15.0% → 19 days at 12 cards/day**; **n=48/arm at mature plays → +15.0% → 8 days**. The app's power card is therefore verifiable against the record, and a test asserts those two points to 3 dp.

- **PLOT A2.1 — "When can this round answer its own question?"**
  `mark` line · `x` days of running this design, linear 1–60 · `y` MDE as a **lift ratio on a log axis**, 1.00–2.00 · `color` ordinal 3 levels = assumed plays per song at readout {90 · 258 · mature} · `facet` none · `refs` a horizontal `kind:"threshold"` rule at the effect she typed she cares about; a vertical `kind:"threshold"` rule at the planned duration · `nField` n per arm on the x gutter · `evidence` derived from `data/snapshots/suno/latest.json`.
  **Question:** *how many days until this design can see the effect I am looking for?*

- **The sentence, rendered at 18px above the plot, generated not authored:**
  > *"At 12 cards/day, 2 arms, and 258 plays per song, this design detects a **+15.0%** lift in 19 days. It cannot see +5% inside 60 days. The largest era-controlled style effect ever measured here is about +25%."*
  And when the design is underpowered against its own declared MDE, that sentence says so first, in the same slot, and the freeze step's chip turns ember with `UNDERPOWERED AS DESIGNED — freeze anyway?`. **It does not block.** The cost is made visible before it is paid; the decision stays hers.

- **The blocking argument, printed once as a caption, because it is worth seven days:**
  > *"Month-to-month baseline drift is 1–2 pp — larger than the effects being tested. Unblocked, that drift enters the error term and pushes a +15% round from 19 days to ~26. Randomising within day and blocking on day removes it and costs nothing, because within-day randomisation makes treatment orthogonal to day."*

### 1.4 A3 — How the twelve are diversified

This is the question the whiteboard leaves open and the one that decides whether the round works.

**The factor is held constant within an arm. Everything else is BALANCED ACROSS ARMS — not randomised independently.**

Five **nuisance dimensions** are sampled from `src/sampler.data.json`, stratified so every level appears an equal number of times in arm A and arm B:

| dimension | pool | balance rule |
|---|---|---|
| seed theme | rotating pool, no repeat within a round | 3 themes × 2 arms × 2 |
| persona | the 5 Suno personas + `none` | each persona ≤2 per arm, equal across arms |
| key / mode | 12 × {major, minor} | equal count of major/minor per arm |
| tempo band | {≤95, 96–125, 126+} | equal per arm |
| lyric domain | {EXISTENCE, NEWS, BODY, WORK} | equal per arm |

Independent randomisation of five nuisance dimensions at n=6 per arm produces visible imbalance roughly half the time, and an imbalance the eye can see is an imbalance a reader will (correctly) not trust. Balancing is cheap and it is the difference between twelve variations and twelve confounds.

**Slot order is randomised** (the operator renders in card order; Suno queue position and time-of-day are plausible nuisances) and **`slot` is recorded as a covariate**, available as a sensitivity analysis, absent from the primary model.

The RNG is seeded and the seed is written to `design.json`. The assignment is reproducible from the seed alone; a test asserts it.

- **PLOT A3.1 — Design matrix.** `mark` strip (heat-strip) · `x` factor (1 column in CONFIRM, up to 11 in SCREEN) · `y` run 1–12 · `color` ordinal binary {−, +} · `facet` none. **Question:** *is this a design, or twelve vibes?*
- **PLOT A3.2 — Nuisance balance.** `mark` bar · `x` count, `zero: true` · `y` nuisance level · `color` ordinal 2 = arm · `facet` dimension, 5 panels, `scales: "shared"`. Any level where |A − B| ≥ 2 renders ember with `IMBALANCED — reshuffle`. **Question:** *do the two arms differ in anything except the factor?*

### 1.5 A4 — Pre-registration, and the gate it holds

**Generation is disabled until the pre-registration is frozen.** `POST /soundry/rounds/:id/generate` returns **409** with `preregistration not frozen` if it is not. This is the multiplicity control, and it is a server rule, not a UI convention — an analysis plan written after the numbers is not an analysis plan.

The frozen record (`preregistration.json`) carries:

```
hypothesis            free text, verbatim, immutable after freeze
mode                  CONFIRM | SCREEN
factors[]             { name, level_minus, level_plus }
control_definition    which level is control, stated in words
surface               suno | nightcafe | panel        (nightcafe: [NEEDS FEED])
instrument            platform | panel                (never both — see §3.3)
primary_outcome       { field, family: count|proportion|mean|variance, readout_age_days }
secondary_outcomes[]  each permanently flagged exploratory
mde                   declared as a LIFT RATIO (e.g. 1.15), not a percentage-point
alpha                 0.05
power_target          0.80
planned_n_per_arm     integer
planned_days          integer
analysis              { link: "log", family, block: "day", se: "HC1-sandwich",
                        quantile: "t" | "z", exclusions[] }
stopping_rule         "fixed-n" | "fixed-days"        (no sequential rule without a boundary)
multiplicity          { family: "round", k, method: "BH", q: 0.10 }
frozen_at, frozen_by, spec_hash
```

`spec_hash` = SHA-256 of the canonically-serialised record (sorted keys, no whitespace). **The hash is stamped into every variant file and into every analysis result.** If a computed analysis's spec differs from the frozen hash, the Analytics tab renders `ANALYSIS DEVIATION — the analysis that ran is not the analysis that was registered` in place of the estimate, and the round is excluded from the multiplicity denominator with a visible note. Amendments append with their own timestamp and reason; they never overwrite.

**One round open at a time.** `POST /soundry/rounds` returns **423 Locked** while another round's `status` is `open`, naming the open round. Two concurrent rounds on one profile share days, share listeners, and make both day-blocks unidentifiable.

### 1.6 A5 — The twelve cards

3 × 4 grid, 12px gutter. Each card, exactly as the whiteboard specifies plus what the design requires:

- **Title** (Fraunces 19/600)
- **Style Prompt** · **Exclude Prompt** · **Lyrics** · **Caption** — four collapsible wells, monospace only for tag strings
- **Arm ribbon**: a 4px rule down the left edge; `CONTROL` cards carry a visually distinct frame, not merely a word
- **Factor fingerprint**: a 5-segment micro-bar, filled ember where a dimension varies, graphite where held. **Two cards with identical fingerprints raise `DUPLICATE CELL — this card buys nothing.`**
- **Derived-measure strip, computed before render** from the drafted lyric: `sung_lines · words_per_line · repeated_line_ratio · emo_tag_count · target duration`, each checked against the live `vault/gates.yaml` band. Out-of-band renders a hairline and the reason. A card that violates a **declared hold-constant** cannot be marked ready.
- **Per-field copy button**, with a 2s visible confirmation *and* an `aria-live` announcement. Twelve cards × five long fields is a sixty-field manual task; without per-field affordances and completion feedback it is a slip factory.
- **`clip_id` binding field** (§0.2) — the largest target on the card.
- **State chip**: `DRAFT → EXPORTED → POSTED → ACCRUING → READ → CLOSED`. Colour + glyph + word. **Nothing auto-advances.** `POSTED` requires a bound clip id; `READ` requires `age ≥ readout_age_days`.
- **`not_posted_reason`** — a required field if a card is closed unposted. If ≥2 unposted cards come from one arm and any lacks a reason, `GET /soundry/analysis/...` returns the contrast **suppressed** with `DIFFERENTIAL ATTRITION — arm B lost 3 cards, 2 without a stated reason.` Selective non-posting is the most likely way this instrument breaks, and it breaks silently.
- **Grid header**: progress ring, `4 of 12 exported · 0 posted`, and the realised allocation `A 6 / B 6`.

### 1.7 A6 — Export

Writes `output/soundry/<round-id>/<variant-id>.md` (paste-ready, one file per card) and `data/soundry/rounds/<round-id>/variants/<variant-id>.json`. The button states exactly what it wrote, with the paths, after it writes. It fetches nothing.

### 1.8 The render control that does not exist

The Affordance Coach's seat argued for a disabled Render button with its reason stated inline — a legitimate position, and I am overruling it. **There is no render affordance in the Soundry, disabled or otherwise.** A disabled button is a promise that the feature exists and is merely gated; under `vault/AUTONOMY.md` it does not exist and must not appear to. In its place: the permanent chrome line (§0), and one line in the Anvil footer — *"Rendering and posting are yours. This bench has never had a button for them."* `tests/server-soundry.spec.ts` asserts `POST /soundry/render` → **405**, `set-cookie` null — a test whose whole job is to prove a route does not exist.

---

## 2 · TAB 2 — EXPERIMENT RESULTS

Routes: `#/results` (round list) · `#/results/<round>` · `#/results/<round>/variant/<id>` · `#/results/ledger`.

### 2.1 B1 — Round header

The frozen hypothesis, **verbatim and non-editable**, with `frozen_at`, the truncated `spec_hash`, the declared MDE, planned n/days, and:

- **n so far / n needed / days to decision**, as three tabular figures and a 90px progress rule.
- **Peek count.** Every load of this route before `planned_days` elapses appends a `peek` event and increments a visible counter: `opened 6 times before the decision date`. Not a scold — a number, next to the decision. With a fixed-n stopping rule and daily generation, optional stopping is the largest forking-path generator in the build, and the only cheap defence is making it countable.

### 2.2 B2 — Maturity board

`mark` strip (2-D) · `x` age in days since `posted_at`, linear 0–30 · `y` variant, grouped by arm · `color` ordinal 2 = arm, plus **hollow** for unbound/unposted · `facet` none · `refs` a vertical rule at `readout_age_days` (default 14, labelled *"readout"*) and a second at the 258-play median crossing, labelled *"below here, one song's rate is mostly dice"* · `evidence` from the latest suno snapshot.

**Question:** *which of these are readable yet?* Unbound cards sit at x=0 as open rings with `UNBOUND`.

### 2.3 B3 — Accumulation curves — the missing instrument, named

`mark` line · `x` age in days · `y` cumulative plays, log · `color` ordinal = arm · `facet` none.

**`[NEEDS FEED: suno — ≥14 consecutive daily snapshots]`.** Verbatim empty state:

> *"One Suno snapshot exists. Nothing on disk describes how fast a song accrues plays, so the readout age of 14 days is an assumption, not an observation, and so is every duration quoted in this app. Fill this by running `python3 scripts/ingest_suno.py` daily; the first curve appears on day 14. Until then the power card's `plays per song` slider is a hypothesis with a handle on it."*

**This card unlocks SCREEN mode** (§1.2) and re-estimates `m` in the power model. It is the highest-value unbuilt thing in the Soundry and the empty state should say so in those words.

### 2.4 B4 — Arm contrast — the one result plot

`mark` **interval, rendered as the Grain Whisker** (the Soundry's signature from the shared visual language: a 10px hatched rectangle whose hatch pitch = `clamp(2 + 26/√n, 2, 14)` px, so sample size is felt as texture before the `n` is read) · `x` **lift ratio on a log axis, symmetric about 1.00**, shared across every row and every round in the app · `y` arm, **control pinned first and anchoring the reference** · `color` interval-position status — entirely above 1.0 teal, entirely below ember, **straddling 1.0 desaturated to `--ink-faint` regardless of the point estimate** · `nField` n per arm, fixed right gutter · `refs` a 1.5px `kind:"zero"` rule at ratio 1.00 spanning full height, plus two ember **Power Notches** at the pre-registered MDE and its reciprocal · `evidence` n, `pulled_at` of the snapshot used, provenance `derived`, caveat naming the readout age.

**Question:** *did the treatment arm beat the concurrent control by more than the noise?*

Guards, all rendered as banners rather than caveats:
- Interval wider than 2 × MDE → `UNDERPOWERED — this round cannot resolve the effect it was designed to find.`
- Achieved n below planned n → the point estimate is **suppressed** and only the whisker renders (§4.4).
- Type-M exaggeration ratio > 1.5 → point estimate suppressed, whisker only.

### 2.5 B5 — Per-variant cloud (and what it is not)

`mark` strip (1-D) · `x` like rate at readout age, shared with B4's underlying scale · `y` none · `facet` arm, 2 panels, `scales: "shared"` · `color` single ink.

**Question:** *is the arm's difference carried by the arm, or by one song?* Card subtitle, permanent: *"This is a distribution, not a ranking. At 258 plays a song's rate carries ±2.53 pp; the visible spread here is 58% dice."*

### 2.6 B6 — The refusal card

Where a "winner of the day" would go, in the same slot, at the same size:

> **This day has no winner, and this round has no best song.**
> Ranking twelve songs by observed like rate at ~258 plays selects on noise: 58% of the spread you can see is binomial. Picking the top card and building a full song on it is a coin flip with a leaderboard skin.
> Winners here are **arms**, declared after the pre-registered n — currently **day 6 of 19, 36 of 114 per arm.**

### 2.7 B7 — Listener feedback

Verbatim notes as text cards, each carrying its cleaning flags, plus `mark` bar / `x` mention count / `y` coded theme / `color` valence, 3 levels.

Notes failing `cleanFeedbackText().safeForAnalysis` are **stored and displayed**, drawn at reduced ink with an `excluded from analysis` chip. Never silently dropped, never silently counted. **Feedback is texture and provocation. It is never averaged into a metric and never pooled with platform response** (§3.3).

### 2.8 B8 — Promotion and benching

Two exits, both write a file, no third state, no silent drop. An `undecided: 4` counter persists until every variant has one.

- **Promote** → writes a Phase-0 brief to `output/soundry/<round-id>/promotions/<variant-id>_phase0.md`. The button reads **"Open as a Phase-0 brief in `lofn`."** It does not say Ship. The Soundry has no publish surface, and the promoted design re-enters through the front door and is gated by `lofn-qa`.
- **Bench** → writes the evidence with the decision, so the same idea is not re-proposed blind.

### 2.9 B9 — The Ledger (`#/results/ledger`)

**Decision: the Ledger is a route under Experiment Results, not a fourth tab.** Three seats wanted it promoted; her three-tab IA is right and the Ledger is a *record*, not a *workspace*. It is reachable from every hypothesis row in Analytics and from every round header, which is where anyone actually wants it.

- **PLOT B9.1 — Pre-registration timeline.** `mark` bar (timeline) · `x` date, time scale · `y` hypothesis · `color` ordinal 4 = {pre-registered · running · resolved · abandoned}, with `ANALYSIS DEVIATION` rows in ember · `facet` none.
  **Question:** *are we learning, or are we curating?* **Abandoned hypotheses are never hidden.** The count in this ledger is the multiplicity denominator; concealing the abandoned ones makes every interval in the app too narrow.
- **B9.2 Decision log** — one line per promotion or benching with the evidence that provoked it.
- **B9.3 Deviation register** — every amendment, its timestamp, its reason, and the hash before and after.

---

## 3 · THE EXPERIMENT LEDGER — data model

Two locations, deliberately separate:

- **`data/soundry/rounds/<round-id>/`** — the working store. Round id `R-YYYY-MM-DD-<slug>`.
- **`data/snapshots/soundry/<ISO8601-basic>.json` + `latest.json`** — the **feed** Lofn Analytics reads, emitted by `emit-snapshot.mjs`. It carries the ingest contract's envelope (`feed: "soundry"`, `pulled_at`, `source`, `header`, `row_count`, `rows`) and **only closed rounds with a frozen pre-registration**. An open round is never exported.

### 3.1 Files in a round

```
preregistration.json     frozen, hashed, immutable; amendments append to amendments[]
design.json              rng_seed, assignment matrix, nuisance balance table, block=day, slot order
variants/<vid>.json      the prompt quad + factor levels + nuisance levels + slot
                         + spec_hash + state + clip_id + posted_at + not_posted_reason
observations/suno.jsonl  append-only; one row per (variant, ingest pull)
observations/panel.jsonl append-only; one row per (listener, variant)
analysis/<contrast>.json cached GLM output keyed by input hash; recomputed when the hash changes
events.jsonl             append-only audit: created · frozen · generated · exported
                         · clip_bound · peek · closed · promoted · benched · amended
```

Every file is UTF-8 **without BOM**; every reader opens with `utf-8-sig` regardless, because a sibling artefact in this repo carries one and hard-crashes a naive parse.

### 3.2 `observations/suno.jsonl` — the platform instrument

One row per variant per daily ingest:

```json
{"variant_id":"R-2026-08-04-wpl/v07","clip_id":"…uuid…","pulled_at":"2026-08-18T00:03:48+00:00",
 "age_days":14.0,"play_count":271,"upvote_count":12,"like_rate":0.04428,
 "snapshot_file":"data/snapshots/suno/2026-08-18T00-03-48.json","is_public":true}
```

**Analysis reads at a fixed AGE, not a fixed date.** For each variant the app selects the observation whose `age_days` is nearest `readout_age_days`, requires `|age_days − readout| ≤ 1.0`, and records the realised age. Reading at a fixed calendar date compares a 3-day-old song to a 17-day-old one and calls the difference a treatment effect. Consecutive rows also difference into per-day increments, which is what builds B3's maturity curve.

Eligibility for the primary analysis: `clip_id` bound **and** `is_public` **and** a readout observation within tolerance **and** the variant's `spec_hash` matches the frozen one. Anything else is listed, with its reason, in the analysis card's exclusions block.

### 3.3 `observations/panel.jsonl` — the listener instrument

A local, blind panel session served at `#/panel/<session-token>`, one listener at a time, access limited to the Scientist and invited collaborators, all traffic loopback.

- Titles, prompts, and arm labels **hidden**; presentation order randomised per listener and recorded.
- Captured per (listener, variant): a forced-choice or 0–10 rating, optional free text, elapsed seconds, and the presentation position.
- Listener identity is a **pseudonym hash only** — `anon-<hmac16>` over a per-session secret. No name, no email, no handle is ever written. Free text passes `cleanFeedbackText`; flagged notes are stored with `analysisEligible: false`.

**Platform response and panel response are different instruments with different populations and different noise structures, and they are never pooled into one number.** `preregistration.instrument` names exactly one as primary; the other, if collected, is permanently `exploratory`. The Analytics tab's instrument register (§4.7) makes any attempt to mix them visible.

### 3.4 Randomisation record

`design.json` carries `rng_seed`, the algorithm name, and the realised assignment. A test regenerates the assignment from the seed and asserts byte equality. Within-day randomisation with day as a fixed-effect block is the design's whole defence against the measured 1–2 pp month drift; it must be reproducible or it is a claim, not a method.

---

## 4 · TAB 3 — ANALYTICS

### 4.1 C1 — DECIBAL, the topline card

The Scientist's specified method, implemented exactly, in ~140 lines of dependency-free TypeScript (`apps/soundry/glm.mjs`, IRLS, pure, node-testable).

**The model.**

```
η_i = β₀ + β₁·T_i + Σ_{d=2..D} γ_d·1[day_i = d]   (+ offset o_i)
g(μ) = log(μ)                        log link, always
T_i ∈ {0,1}                          binary treatment indicator
day  = fixed-effect blocks           within-day randomisation makes T orthogonal to day,
                                     so blocking costs df and nothing else
lift = exp(β₁)                       the lift ratio
CI   = exp( β₁ ± q · SE(β₁) )        Wald, two-sided, α from the pre-registration
```

**Family by metric kind — the log link is what makes all four commensurable, which is why every effect in this app can share one axis:**

| metric kind | example | family | V(μ) | offset | `exp(β₁)` means |
|---|---|---|---|---|---|
| **count** | plays, upvotes at readout age | quasi-Poisson | μ | none | ratio of expected counts |
| **proportion** | like rate = upvotes/plays | quasi-Poisson on upvotes | μ | `log(plays_i)` | **rate ratio** |
| **mean** (strictly positive) | listener rating, dwell seconds | Gamma | μ² | none | ratio of means |
| **variance / dispersion** | spread of response within an arm | Gamma on per-(arm,day) sample variances, weights `df/2` | μ² | none | **variance ratio** |

The proportion case uses Poisson-with-offset rather than log-binomial deliberately: log-binomial fails to converge on a nontrivial fraction of real data, and the offset form gives the identical rate-ratio interpretation with a robust SE. The variance case exploits `s²·df/σ² ~ χ²_df`, i.e. `s² ~ Gamma(df/2, 2σ²/df)` — a Gamma log-link GLM on the cell variances is the right estimator for a ratio of variances, and it is the same three lines of code as the others.

**Estimation — IRLS, specified so it can be tested:**

```
η = Xβ + o ; μ = exp(η) ; dμ/dη = μ
W_ii = (dμ/dη)² / V(μ)          Poisson → μ ;  Gamma → 1
z_i  = η_i − o_i + (y_i − μ_i)/μ_i
β ← solve(XᵀWX, XᵀWz)           Cholesky, ridge 1e-10 fallback
stop when max|Δβ| < 1e-10, or 50 iterations → NON-CONVERGENT
```

**Standard errors — Wald, made honest under the overdispersion this catalogue actually has:**

```
bread   = (XᵀWX)⁻¹
φ̂       = Σ (y−μ)²/V(μ) / (n−p)                 Pearson dispersion
SE_model = sqrt( φ̂ · diag(bread) )
u_i      = x_i · (dμ_i/dη_i)(y_i−μ_i)/V(μ_i)     Poisson → x_i(y−μ) ; Gamma → x_i(y−μ)/μ
meat     = Σ u_i u_iᵀ
SE_HC1   = sqrt( diag( bread · meat · bread ) · n/(n−p) )
```

**`SE_HC1` is the reported SE.** Model-based and quasi-φ̂ SEs are shown in the drawer for comparison, never on the card. Clip-level overdispersion here is enormous; a model-based Wald interval would be visibly too narrow and the card would lie in the direction that flatters us.

**Quantile:** `t_{n−p}` when `n − p < 30`, `z` otherwise, and **the card says which**.

**Refusals, all hard:**

| condition | card renders |
|---|---|
| no control arm in the round | `NO COMPARATOR — this round has no concurrent control. DECIBAL is not defined.` |
| pre-registration not frozen | `UNREGISTERED — an analysis plan written after the numbers is not an analysis plan.` |
| either arm has 0 events | `ZERO CELL — the ratio is not estimable.` |
| a day block contains only one arm | that day is **dropped**, and the card names it: `day 2026-08-07 dropped — single-arm block.` |
| IRLS non-convergent, or `cond(XᵀWX) > 1e12` | `NOT ESTIMABLE` + the diagnostic |
| achieved n < planned n | the **point estimate is suppressed**; only the Grain Whisker and the achieved MDE render |
| type-M > 1.5 | point estimate suppressed, whisker only, dial reads the exaggeration factor |
| `spec_hash` mismatch | `ANALYSIS DEVIATION` and exclusion from the multiplicity denominator |

**The card itself.** Hero numeral (IBM Plex Mono 56/500 tabular) reading the lift as a percentage — `+12.4%` — with the **ratio and the interval on the same line at 18px**: `×1.124 · 90% CI ×1.03 to ×1.23 · n 114/114 · 19 days · HC1 · t₍₁₀₉₎`. Beneath it, at 13px, the reference class, always, verbatim:

> **DECIBAL measures the lift of a treatment arm over the control arm that ran on the same days, in the same rounds, on the same surface. It is not a comparison to other artists.**
> A single profile's data cannot identify an average-artist baseline: no competitor row has ever been pulled, the five watched targets are unnamed anywhere on disk, and a comparison set chosen for being "growing/trending" is selected on the outcome — it would report the selection rule, not us. The comparator here is us, concurrent.

**Reference-class selector — exactly three, and only two are legal as a headline:**

1. **Round mode** (default while a round is open) — treatment arm vs concurrent control within the round, day-blocked. The cleanest thing this app can say.
2. **Estate mode** — all Soundry-era variants carrying ≥1 **adopted** factor level vs concurrent Soundry variants carrying none, day-blocked across the whole corpus. This is *"did the bench pay for itself"*, and it is the number worth putting biggest once two rounds have closed.
3. **Historical mode** — Soundry-era vs pre-Soundry catalogue. Offered, **permanently greyed, never the headline**, and carrying its own reason: *"Non-concurrent. Engagement peaked 2026-05 at 5.10% and fell to 3.38% and 3.56%; a before/after against that catalogue is measuring the calendar."*

**Empty state, verbatim:** *"No round has closed. There is no lift to report — not a lift of zero, no lift at all."* Never `0%`, never `100%`, never a needle at rest.

### 4.2 C2 — The hypothesis board

Her sketch, executed with one correction that the panel was unanimous on.

`mark` interval (Grain Whisker) · `x` **lift ratio, log axis, symmetric about 1.00, identical for every row** · `y` hypothesis, **sorted by pre-registration date, never by effect** (sorting a hypothesis board by result is a fishing expedition laundering itself into a finding; a `sort by |effect|` control exists, clearly labelled, and is not the default) · `color` ordinal 4 = decision state {pre-registered-not-run hollow outline · accumulating half-ink · concluded-supported teal · concluded-not-supported ink}, **doubled by marker shape** · `facet` **by outcome family only where the units genuinely differ** — count/proportion/mean all produce a *ratio* and therefore share one axis, which is a direct dividend of the log link; **variance ratios get their own facet** because a ratio of variances is not comparable to a ratio of means · `refs` the 1.00 rule spanning every row at full height, plus a per-row dashed **MDE notch** so an interval that excludes 1.00 while sitting wholly inside the MDE reads correctly as *real but too small to act on* · `nField` gutter carrying **n · #days · achieved MDE · BH q** — all four, always.

**She drew a separate axis per hypothesis (`0/+5/+10/+15/+20` on one, `−10/−5/0/+5/+10` on the next). Overruled to one shared symmetric log-ratio axis.** Two hypotheses on two scales cannot be compared, which is the only reason to put them on one page; and an axis starting at zero cannot draw a negative result, which is a significance filter built into the geometry. Her second sketched axis is already symmetric — the two only have to agree.

`[more details →]` is kept exactly as she drew it, opening a drawer with: the frozen hypothesis text, the full GLM output (β, SE model / quasi / HC1, φ̂, deviance, df, convergence iterations), the exclusion list with reasons, the day blocks used and dropped, the realised allocation, and the peek count.

**Rows that straddle 1.00 stay in date order, desaturated. They are not hidden and not sunk.** *"We tried this and it did nothing"* is a result, and a board that shows only survivors is a file drawer with a chart on it.

### 4.3 C3 — Power planner

`mark` line · `x` n per arm, log 6–1000 · `y` MDE lift ratio, log 1.00–2.00 · `color` ordinal 3 = assumed plays per song {90 · 258 · mature} · `refs` a vertical rule at today's achieved n; the declared MDE horizontally · direct end-of-line labels, no legend.

Same `power.ts` module as A2, so the design-time promise and the post-hoc reality cannot drift. Once ≥2 rounds close, `σ²_true` is **re-estimated from the Soundry corpus** and the card draws both curves with the seeded one dashed.

**Question:** *how many listeners, or how many days, before the next question is answerable?* This is what sets the card count on the Anvil.

### 4.4 C4 — Multiplicity ledger

`mark` line (step) · `x` rank of the p-value · `y` p-value, linear 0–1 · `color` ordinal 2 = {passes BH at q=0.10 · does not} · overlaid BH reference line `y = (rank/m)·q` in ember · `facet` none.

Header, in words: total pre-registered contrasts to date (including abandoned), and the expected number of false positives at α=0.05 for that count.

**Question:** *how many of our wins are the one-in-twenty?* A bench running twelve cards a day will manufacture a nominally significant result on a regular schedule by construction, and the app should say so on its own analytics page rather than wait to be asked.

### 4.5 C5 — Learning transfer

`mark` interval · `x` lift ratio, the shared axis · `y` adopted factor level · `color` state · `facet` none.

The estate-mode DECIBAL decomposed by adopted factor: *did each thing we learned actually move the catalogue?* `[NEEDS: ≥1 adopted factor and ≥2 closed rounds]`. If this card never fills, the Soundry is a hobby, and that sentence belongs in its empty state.

### 4.6 C6 — Promotion log

Table: experiment · promoted-on · destination run · outcome · `↗`. One line above it, permanent: *"The bench proposes; the pipeline makes. Nothing ships from the Soundry."*

### 4.7 C7 — Instrument register

One row per hypothesis naming its primary instrument (platform / panel), its population, and its known bias. Any hypothesis whose analysis touched both instruments renders ember with `INSTRUMENT MIXING`. Plus the standing sentence: *"Soundry response measures what a Suno browser, or a small panel, rewards. It is advisory. It never overrides the Somatic read, never becomes a gate in `vault/gates.yaml`, and never argues the publish bar down."*

---

## 5 · THE HYPOTHESIS CARD — what it shows before it has power

The individual card (the `[more details]` target and the Anvil's live preview) has three lives, and the transition between them is computed, never chosen.

### 5.1 Born — a width, not a point

Before the pre-registered n is reached, **there is no dot and no number.** The card is:

- a **Grain Whisker spanning ±MDE** around 1.00 — the region this design was built to resolve — drawn at the widest hatch pitch, so the eye reads "we know nothing yet" as texture;
- the words **`Not yet measurable`**;
- three tabular figures: **`n so far 36 / 114 needed`**, **`day 6 of 19`**, **`decision on 2026-08-23`** — the last computed from the realised cards-per-day, not the planned one, so a slow week moves the date honestly;
- the **achieved MDE at today's n**: `at n=36 this design can currently see ×1.28 or larger`;
- the frozen hypothesis text and the spec hash.

A prior density centred at 1.00 may be drawn behind the whisker; it is empirical (the estate's own between-style-effect spread) and it is labelled as such, never as a belief about this hypothesis.

### 5.2 Accumulating

Once n crosses the pre-registered threshold, the point estimate appears as a 9px ink rule with a 3.5px disc coloured by sign, the whisker's hatch tightens as √n, and the **type-M dial** appears: a small arc reading the current exaggeration factor, computed by retrodesign against the declared MDE.

```
δ = log(MDE)                     the effect the round was designed to find
s = SE_HC1(β₁)                   achieved
power  = Φ(δ/s − z_.975) + Φ(−δ/s − z_.975)
type-S = Φ(−δ/s − z_.975) / power
type-M = E[ |β̂| | |β̂| > z_.975·s ] / δ      truncated-normal, 10k fixed-seed draws
```

**`type-M > 1.5` re-suppresses the point estimate.** Not a warning next to it — the number goes away and the whisker stays. If the estimate is on average 1.5× too big conditional on being visible, displaying it is worse than displaying nothing, because a reader cannot mentally divide by an unknown factor.

### 5.3 Concluded

Point estimate, HC1 Wald interval, n, days, achieved power, BH q, the exclusion list, the peek count, and the verdict: `supported` / `not supported` / `underpowered` / `abandoned`. All four verdicts stay on the board permanently.

### 5.4 What the card refuses to display — the complete list

1. A point estimate before the pre-registered n. A width instead.
2. A point estimate when type-M > 1.5.
3. A **p-value, a significance asterisk, or a significant/not-significant chip.** The board sorts, shrinks, and shows intervals; it does not threshold.
4. Any interval computed under a spec that differs from the frozen hash.
5. Any estimate from a round with no concurrent control.
6. Any estimate from an unregistered round above `OBSERVATION` tier, ever.
7. A ranking of the twelve, a "winner of the day", or a per-song verdict.
8. A delta, sparkline, arrow, or growth rate computed from a single ingest pull. `UNKNOWN`, in the hatch, never `0`.
9. A lift against "the average artist", or against any comparator not present on disk.
10. A pooled figure mixing platform and panel response.
11. A secondary outcome promoted above the fold or above `OBSERVATION`.
12. A per-hypothesis axis, an asymmetric axis, or an axis that cannot draw a negative result.
13. A causal verb (`causes`, `drives`, `boosts`, `kills`) on any row whose round was not randomised — which in practice means SCREEN-mode aliased effects and any historical-mode comparison. Randomised CONFIRM rounds are the **only** place in the entire estate where that vocabulary is legal, and the footer of both apps should say so.
14. Any render, publish, upload, or spend affordance.

---

## 6 · Build posture

`apps/soundry/` follows the Forge house pattern: `"type": "module"`, `"private": true`, **no `dependencies` key**, devDeps `typescript ^5.5 · vite ^6 · vitest ^3 · @types/node ^24`. `defineConfig` from `vitest/config`. Three HTML entries (`index.html` Anvil · `results.html` · `analytics.html`) plus `panel.html`, all `base: "./"`, `server.port: 5231`. `server.mjs` has zero runtime dependencies — `node:crypto|fs|http|path|url` only.

Tests are `tests/**/*.spec.ts`, `environment: "node"`, imports always exactly `import { describe, expect, it } from "vitest";`. Banned: snapshots, jsdom, `vi.mock`, `vi.fn`, `beforeEach`. Required specs:

- **`glm.spec.ts`** — the GLM against hand-computed fixtures for all four families; the two frozen power points (n=114/m=258 → 1.150; n=48/mature → 1.150) to 3 dp; non-convergence returns `NOT ESTIMABLE`, never `NaN`.
- **`prereg-freeze.spec.ts`** — generate before freeze → 409; mutate after freeze → 403; the hash is stable under key reordering.
- **`design.spec.ts`** — the assignment regenerates byte-identically from `rng_seed`; every nuisance level is balanced across arms; slot order is recorded.
- **`refusals.spec.ts`** — the fourteen refusals of §5.4, each asserted as an absence: `expect(JSON.stringify(card)).not.toContain("p_value")`, `expect(card).not.toHaveProperty("estimate")` below n, `expect(rendered).not.toContain("winner")`.
- **`server-soundry.spec.ts`** — a real spawned server on a `node:net` port with `DATA_DIR` in `mkdtempSync`; happy path, each 403/409/423, and `POST /soundry/render` → **405 with `set-cookie` null**.
- **`autonomy.spec.ts`** — grep the built bundle: no external origin, no generation-API hostname, no render handler.

---

## 7 · Voice

This is the tool of a disappointed idealist, not a growth console. It has no confetti, no streaks, no "great job", no counting-up numbers, no green-means-good. Its warmest gesture is telling you the truth about how long something will take, and its signature — the Grain Whisker — exists so that the thing dashboards normally whisper is the loudest texture on the page.


> **Scope of this part:** `apps/lofn-analytics` only. The shared foundation — design tokens, the `PlotSpec` chart grammar + `validate()`, `FilterState`/URL serializer, the `Insight` object and its tier ladder, and the `GuidanceNote` tooltip contract — is assumed to exist in `apps/_shared/` and is referenced by name here, never re-specified. The Soundry, the DECIBAL lift card, and the log-link GLM live in the other part.

---

# LOFN ANALYTICS — `analytics.lofn.ai` (local-only today)

## 0 · What this app is, and what it refuses to be

Lofn Analytics **observes**. It reads dated snapshots off disk and renders them. It does not pull, post, render, publish, or spend — the single exception is one explicit **Refresh** control per feed that shells the feed's ingest script and states its consequence before running (§14.4). Per `vault/AUTONOMY.md` that is the whole autonomy surface.

Three consequences, enforced structurally:

1. **No causal verbs.** Nothing in this app is randomised, therefore no headline may say *causes / drives / boosts / kills*. The vocabulary is tier-locked by the shared `Insight` generator; the strongest verb available here is *goes with*. The footer of every page prints: **"Lofn Analytics observes. The Soundry tests. Only tested claims may say 'causes'."**
2. **No invented numbers.** A panel with no snapshot renders its feed's verbatim empty-state text and the exact command that would fill it. A delta with one observation renders the word **UNKNOWN** in absent-hatch, never `0`, never a flat line.
3. **Advisory, never a gate.** No figure in this app may be written into `vault/gates.yaml`, may enter an ICB, or may argue the publish bar down. Every card's `next_move.forbidden` field is filled and rendered.

**Posture:** retail/local build only. No hosted deployment until The Scientist says so — the snapshots embed the full style-prompt corpus of 210 unreleased-language clips plus a Suno `user_id`. See open question 1.

---

## 1 · The four source feeds — true status, 2026-08-01

The whiteboard names four feeds. `scripts/ingest_receiver.mjs:43-52` already fixes eight feed ids. **Three have files on disk. Five have never been pulled.** The app must show that ratio on its face, not bury it.

| # | Whiteboard feed | feed id(s) | On disk today | Method / confidence | What the app may draw from it |
|---|---|---|---|---|---|
| **1** | Lofn Socials | `suno` | ✅ `data/snapshots/suno/{2026-08-02T00-03-48.json, latest.json}` · 210 rows · `reconciliation.ok = true`, Δ0 plays, Δ0 upvotes, 11 pages | **api-public** (public endpoint, no cookie) | Everything on the Music rail. The only per-item feed we own. |
| | | `socials` | ✅ `data/snapshots/socials/latest.json` · `pulled_at 2026-08-02T00:15:00Z` · 6 rows (suno, tiktok, youtube, spotify, **nightcafe**, instagram) | **public-profile-read**, `confidence: observed` | Headline counters only. **No deltas — one point per platform.** Instagram row present, `status: not-read`, SKIPPED BY INSTRUCTION. |
| | | `distribution` | ✅ `data/snapshots/distribution/latest.json` · `pulled_at 2026-08-02T00:35:00Z` · 4 rows (spotify, apple_music, youtube_contentid, itunes_amazon) | **authenticated-page-read** (DistroKid), self-labelled *estimated* | The only genuine trend structure on disk today — four nested cumulative windows per platform, plus the one demographic table we possess. |
| | | `spotify` (S4A) | ❌ **ABSENT.** DistroKid's own note: *"Spotify for Artists is NOT connected to this DistroKid account"*; `artists.spotify.com/…` 404s for this session. | — | Nothing. Renders the never-pulled state with the connect step named. |
| | | `instagram` | ❌ **ABSENT by instruction**, not by failure. | — | A present row that says why it is empty. |
| **2** | Competitor Scans (5 watched targets) | `competitors` | ❌ **ABSENT.** No `data/snapshots/competitors/`. **The five targets are not named in any file in this repo.** | — | Nothing. Verbatim empty state + the two blockers: (a) The Scientist names 5 handles + platforms, (b) one pull. |
| **3** | Market Trends + Scan | `market` | ❌ **ABSENT.** No `data/snapshots/market/`. Source stack exists (`vault/DAILY_RESEARCH_SOURCES.md` §D, `vault/RESEARCH_SOURCES_200.md` §C) but nothing has been pulled through it. | — | Nothing. Verbatim empty state. |
| **4** | Soundry Artist Results + Feedback | `soundry` | ❌ **ABSENT.** The application does not exist; the only on-disk hit for the string is the `KNOWN_FEEDS` entry. | — | Nothing. Verbatim empty state. |
| — | *(legacy, quarantined)* | `tracking/*` | ⚠️ `tracking/creator-platforms-latest.json` (Tavily snippets, no `timestamp` field, as-of only inferable from prose: "As of 2026-06-06"), `tracking/suno-latest.json` (2026-03-28, `plays` is the **string** `"8.3K"`) | **search-snippet (provisional)** / stale manual | **Rejected at load by discriminator (I10).** Visible only in the Provenance tab's refusal log. Never plotted. |

**Feed-status is chrome, not a page.** The header carries an 8-segment ring (§3.2) reading **3 of 8 fed**, and every never-pulled feed's panels ship the ingest contract's verbatim text unchanged.

---

## 2 · The rail — Art / Music / Animation, and why Art goes first

**Five of seven seats voted to demote the rail to a filter chip.** I am overruling them, because the fact that arrived after they wrote is decisive: **`data/snapshots/socials/latest.json` carries a NightCafe row with 23,000 works, 1,200 followers and 21,000 likes — 3.55× Suno's 5,913 likes.** Art is not the empty room the panel assumed. It is our largest audience.

But their carry-through objection is correct and is honoured mechanically: **the rail is not navigation, it is the render of the `modality` URL parameter.** One state, one control, drawn where she drew it.

```
RAIL  (left, 168px, sticky, role="tablist", URL param ?m=)

  ▤  ART          21,000 likes · 1,200 followers · 23,000 works
      ██████████░  1 surface read · per-work engagement ABSENT

  ▦  MUSIC        5,913 likes · 3,926 followers · 210 works
      ██████████░  per-clip engagement · reconciled · no history

  ▸  ANIMATION    83,000 likes (TikTok, whole account) · 210 prompt records
      ██░░░░░░░░░  no per-work engagement on any surface
```

- **Order is by measured audience, descending, and the order is computed, not typed.** Art first is a finding, not a layout preference — the app is telling her something the moment it opens.
- **Default landing is MUSIC**, because it is the only modality with per-item data and therefore the only one where a directive can be earned. The rail says why: Art's coverage bar is long but its *depth* meter is one segment.
- **Two bars per rail item, never one.** Top bar = audience size (log, shared across the three). Bottom bar = **measurement depth** (0–5: profile counter · per-work counts · per-work engagement · demographics · time series). Art scores 1/5 on 21,000 likes. That gap *is* the Art rail's headline.
- Modality is encoded by **glyph** (`▤ ▦ ▸`), never by colour. Colour is reserved for the frozen quartet's semantic bindings.
- Rail items are never disabled. Every modality opens a real set of tabs; the ones without feeds render inventory and a named ingest gap.

---

## 3 · Global chrome (every tab, every modality)

### 3.1 Header (72px)
`LOFN ANALYTICS` wordmark (Fraunces 24/600, wonk) · **THE DAY MARK** at 96px (§3.2) · reconciliation pill · theme toggle · reduce-motion toggle · the permanent line **"This app reads. It never posts, renders, or spends."**

**Reconciliation pill:** `SUNO 210/210 · Δ0 plays · 11 pages · RECONCILED ✓`. If `reconciliation.ok === false` on any loaded snapshot, the pill turns ember, a full-width **UNRECONCILED** band pins to the top of every tab, **every absolute total in the app is replaced by an em-dash**, and every card's tier is capped at `OBSERVATION`. The ingest script writes `latest.json` even on failure (`scripts/ingest_suno.py:257-273`); this app is the only gate.

### 3.2 THE DAY MARK — what "daily palette" becomes
96px in the header, 160px on Provenance, archived to `data/daymarks/YYYY-MM-DD.svg`. Zero new colours; it is redrawn daily from the day's data state, not from a rotating palette:
- **Outer ring** — 24 hour ticks; the hour of the most recent successful pull inked, the rest hairline.
- **Inner arc** — 8 segments, one per `KNOWN_FEEDS` id. `M3` filled = exists + reconciled · `M4` outlined = exists + failed reconciliation · absent-hatch = never pulled. **Today: 3 of 8 filled.**
- **Core** — disc, `r = log₁₀(rows in newest snapshot) × 8px`, filled at the Sunset-ramp stop for the current cohort median like rate.
- One ember strike across the whole mark if any invariant I1–I10 failed.
- Caption (Plex Mono 10): `2026-08-01 · 3/8 feeds · 210 rows · recon OK`.
- Alt text generated from the same tuple that draws it.

A rotating palette would make yesterday's screenshot incomparable with today's. This gives her a mark that changes every day and is made of the day's data.

### 3.3 Filter shelf (44px, sticky, URL-backed, carries across every tab and both apps)
| control | values | default | key |
|---|---|---|---|
| Modality | art · music · animation | `music` | `m` |
| Cohort window | all · 2026-only · custom `YYYY-MM..YYYY-MM` | `all` | `win` |
| Play floor | 0–1000 slider | **60** | `pf` |
| Outlier | **exclude** · include+label · isolate | `exclude` | `out` |
| Estimator | median · mean · pooled | **median** | `est` |
| Provenance floor | api-public · +authenticated-page · +public-page · +search-snippet | `+public-page` | `prov` |

- **A control that does not apply to the current tab renders dimmed with a tooltip naming why, and is restored untouched on return.** Silent dropping is the failure mode the whiteboard's "carry between tabs when applicable" is guarding against, and it is invisible unless you draw the dimmed chip.
- Live count at the shelf's right edge: `n = 46 of 210` and the count of rows suppressed by the provenance floor.
- **Estimator is a first-class control** because on the identical n=46 modern cohort the same data yields median **4.37%**, mean **4.29%** and pooled **1.57%**. Every card prints the estimator name inline. (See open question 2.)
- **Zero-result contract:** a filter state matching nothing never blanks a panel. The panel ghosts and offers up to three one-click relaxations with their resulting n precomputed.

### 3.4 THE COMB (22px, under the shelf, every Music tab)
210 hairlines, 1px, ink @70%. `x = log₁₀(play_count)`, domain 1→1,000, then a **tear glyph** (two 6px diagonals at 30°) and a 44px gutter holding the single outlier at full height in absent-ink with an ember label. Collisions stack at 6px jitter. Selected clips full ink; unselected at 12%.

It is the signature device **and** the cross-tab filter indicator: she never has to ask what filter is on, she can see the ink. Brushing sets the cohort filter. Corner readout, Plex Mono 10: `210 clips · 1 off-scale · median 50 plays`.

### 3.5 Tab Directive
Every tab renders exactly one directive as its subtitle, at 28px: the `next_move.directive` of the argmax insight on that tab, ranked by `(tier_rank, |interval.lo|, actionability)`. Beneath it: tier chip · interval · `n_effective` · `[why this →]` · and the `forbidden` line. **A tab that cannot produce a directive does not ship.**

---

## 4 · MUSIC · Tab M0 — **SIGNAL** *(her "Top Trends & Insights")*

**Directive today:** *"Nothing on this tab is a trend yet. Run `python3 scripts/ingest_suno.py` daily; the second pull is what makes this page work."* — tier `ABSENT` / magnitude `PULL_THE_FEED` / routes `ops.runbook`.

### M0.1 · Can this app compute a trend yet? — the snapshot dial
- **mark** 28 tick-arcs on a 96px ring · **x** snapshot index (angular) · **y** — · **colour** present = ink, absent = absent-hatch · **facet** feed (3 rings: suno, socials, distribution) · **feed** all three
- **Q:** is there enough history for any delta on this page to be real?
- **Today:** 1 tick inked per ring. Copy, verbatim: *"One Suno snapshot exists (pulled 2026-08-02T00:03:48Z). A delta needs two. Nothing on this page is zero; everything not shown is UNKNOWN."*
- **Empty state:** n/a — this card is the empty state, promoted to a first-class instrument.

### M0.2 · The Response Index
- **mark** hero numeral (Plex Mono 56/500 tabular) + a 3px reference ruler beneath · **x** like rate 0–8% · **colour** Sunset-ramp stop for the value · **feed** `suno`
- **Today:** `4.37%` — sub-line, Atkinson 14: *"**median** of 46 clips with ≥60 plays created since 2026-02, as of 2026-08-01. Mean on the same cohort is 4.29%; pooled is 1.57%. Half the catalogue is under 50 plays; at 50 plays, 4.4% is two likes."*
- **Q:** how are people responding, before distribution is allowed into the room?

### M0.3 · Response over time — the era chart, and the most important plot in the app
- **mark** dot per clip (r 3.5, 35% fill, x-jittered) + a per-month **median rule** with a 90% bootstrap ribbon · **x** `created_at` month (14 buckets, 2025-06 → 2026-07) · **y** like rate, 0–13% · **colour** single ink; ribbon at 18α · **facet** none · **feed** `suno`
- Months with **n < 5 render dots only, no median rule**, with the n printed at the axis in ember. 2026-01 is n=2 and gets no rule, permanently.
- **Annotation bracket** over 2026-02 → 2026-07: *"engagement roughly tripled here — the strongest effect in this data is TIME, not style."* Second annotation on the peak: *"2026-05, 5.10% across 16 clips — and it has fallen twice since: 3.38% in June, 3.56% in July."*
- **Measured monthly medians, drawn live, not typed:** 2025-06 1.33 (n=36) · 07 1.74 (43) · 08 2.00 (18) · **09 0.00 (36)** · 10 1.55 (6) · 11 1.89 (16) · 12 2.41 (6) · 2026-01 3.68 (2) · 02 3.76 (3) · 03 3.17 (5) · 04 4.54 (9) · **05 5.10 (16)** · 06 3.38 (6) · 07 3.56 (8).
- **Q:** is our work landing better than it used to, and by how much do we actually know? The 2025-09 cohort's median is **0.00% across 36 clips** — a fact no headline number will ever surface.

### M0.4 · The Placement — permanent, never a footnote
- **mark** one 100%-width horizontal stacked bar, 2 segments, 2px gap · **x** share of total plays · **y** none · **colour** `M4` ember (Triple Arch) / graphite (the other 209) · **feed** `suno`
- Direct labels on both segments. Headline: *"One song is 93.5% of all plays and has the worst modern like rate we have — 1.44% against a 4.37% median. 349,736 plays is a placement event, not a result."*
- This card exists so no other card has to keep apologising, and so the `house_lexicon` gate built on that number is visibly resting on it.

### M0.5 · Standing findings — dated, not live
- **mark** dumbbell, one row per published finding · **x** value · **y** finding name · **colour** `M3` sea-teal = as published 2026-07-27, `M4` ember = recomputed on today's snapshot, connector hairline · **feed** `suno` + `prose-archive`
- Seeded with the two measured drifts: era-cohort median **4.51% (n=48) → 4.37% (n=46)**; all-time ≥50-plays median **2.35% → 2.46% (n=112)**. Catalogue **220 → 210** while plays went **up**.
- **Q:** does the thing we believe still reproduce?
- Any row whose ember end is more than 30 days old and un-recomputed renders hollow.

### M0.6 · Movers — `[NEEDS FEED: suno × 2 snapshots]`
- **mark** slope segment + terminal dot · **x** snapshot date (2 positions) · **y** like rate · **colour** Tide diverging on sign of Δ · **feed** `suno`
- **Empty state:** the axis is drawn, both date ticks are drawn, and between them sits a single hairline with a hollow terminal ring labelled `UNKNOWN`. The shape of the chart it will become, with no fake slope in it. Below: *"Run `python3 scripts/ingest_suno.py`. This chart draws itself on the second pull."*

---

## 5 · MUSIC · Tab M1 — **RESPONSE** *(her "Metric Leaders")*

**Directive today:** *"Stop ranking individual songs. At a 258-play median a like rate carries a ±2.53pp 95% interval — twelve of the top twenty overlap. Read cohorts, not songs."* — `FINDING` / `RUN_AN_EXPERIMENT` / `soundry.hypothesis`.

### M1.1 · Response leaders — the whiteboard's card, re-sorted
- **mark** dot + horizontal 95% **Wilson** interval whisker · **x** like rate, 0–15%, shared · **y** title, top 20 · **colour** era (4 sampled Sunset stops) · **facet** none · **feed** `suno`
- **Ranked by the Wilson lower bound, not the point estimate.** This is the one design decision that stops the leaderboard being a noise generator: it demotes a 3-play clip at 33% below a 400-play clip at 6%. The sort control offers point-estimate ranking, and selecting it fires an interstitial, not a tooltip.
- `n` (plays) in a fixed right gutter, always. Clips below the play floor render hollow with dashed whiskers.
- **Q:** which songs actually earned a response, ranked so a lucky small sample cannot win?
- **Interlock:** no single row on this card may generate a directive. `next_move.magnitude` is capped at `RUN_AN_EXPERIMENT` for every row.

### M1.2 · Reach leaders — the demoted twin
- Same mark, `x = play_count` log₁₀ with the **tear glyph** and the outlier gutter. Never summed, never headlined. Toggling to this view carries the M0.4 banner inline.

### M1.3 · The noise floor — the epistemics chart
- **mark** histogram bars + a heavy median rule + a shaded "unreliable" region · **x** `log₁₀(play_count)` · **y** clip count · **colour** single ink; bars left of the reliability threshold at 45% ink · **feed** `suno`
- **Annotation on the median rule:** *"median 50 plays — at 4.4%, that is 2 likes."* Second annotation at the modern-cohort median of 267 plays: *"±2.53pp at 95%. **58% of the apparent between-song variance in this catalogue is binomial noise.** Per-song verdicts are noise."*
- **Q:** how much can I trust any one song's number? This card is the evidentiary basis for M1.1's sort order and links to it both ways.

### M1.4 · Response vs reach
- **mark** circle r 3.5, 1px stroke, 18% fill · **x** `play_count` log₁₀ (with the tear) · **y** like rate 0–13% · **colour** era (4 Sunset stops) · **size** constant, deliberately · **feed** `suno`
- Reference: dashed horizontal at the cohort median; vertical at 50 plays. Triple Arch beyond the tear as a hollow square, ember-labelled.
- **Mark opacity is bound to `play_count`** on a documented ramp; sub-floor marks render at 40%. Uncertainty in the ink, not in a caption.
- **Q:** does more distribution buy more response? 349,736 plays at 1.44% against 4.37% — no.

### M1.5 · Model version — the confound nobody has controlled
- **mark** dot + 90% bootstrap whisker · **x** median like rate · **y** `major_model_version` — **v4.5 n=61 · v4.5+ n=67 · v5 n=42 · v5.5 n=40** · **colour** single ink · **feed** `suno`
- **Q:** is the "we got better" story actually "Suno got better"?
- **Caveat, mandatory and rendered on the card:** *"Model version is near-perfectly confounded with time — v4.5 is the old catalogue, v5.5 is the new one. This card cannot separate them. It exists so the era story is not read as a craft story without an argument."*
- Not on the whiteboard. Four cleanly time-ordered engine versions sit directly under the tripling claim and nothing on disk has ever mentioned them.

### M1.6 · The catalogue table (drill target of every plot on this rail)
Virtualised, 210 rows. Columns: title · created · plays · upvotes · **like rate with an in-cell Wilson bar on a shared 0–15% axis** · duration · sung_lines · words_per_line · repeated_line_ratio · emo_tag_count · persona · model · `tags` (truncated, expandable). The **exact cohort expression is printed above it** with a `copy query` button. Export writes CSV + a JSON sidecar `{snapshot_id, filter_state, generated_at, n, caveats[]}` — a number that leaves this workspace carries its provenance or it does not leave.

---

## 6 · MUSIC · Tab M2 — **CRAFT** *(the four named insights live here)*

**Directive today:** *"Write the next run to ≤6 words per line and 80–100 sung lines, and treat everything else on this tab as a Soundry question. `forbidden:` may not be written into `vault/gates.yaml` — the 2026-07-27 recommendations were never applied."* — `PATTERN` / `TRY_A_PAIR` / `lofn.phase0.axis`.

### M2.1 · Lyric length per line — **whiteboard insight #1**, and the strongest signal we have
- **mark** open circle + linear fit with a **grain-hatched** CI band · **x** `words_per_line`, 3–10 · **y** like rate · **colour** era in exactly **two** levels (pre-2025-12 graphite, 2026-onward ink) — two, not a ramp, because the era confound is the point and a gradient would hide it · **facet** none · **feed** `suno`
- Reference: dashed vertical at **7.5**, `kind:"gate"`, labelled *"live gate — `vault/gates.yaml:155 mean_words_per_line_ceiling`"*; the 2026-07-27 recommended band as a dashed outline beside it.
- **Mandatory caption, verbatim:** *"Recommended bands were never applied to `vault/gates.yaml`. The solid rule is live policy; the dashed rule is a proposal."*
- Named extremes plotted with leader lines: *Five wrong colors* 3.7 w/l (best) · *The Water Warms Silently* 8.8 w/l, zero repetition → 2.0% (worst in catalogue) · *The Architecture of a Sigh* 7.7 → 2.0%.
- Corner block, Plex Mono 12: `ρ = −0.33 · n = 46 · weak`. **The word "weak" is rendered, not implied.**
- **Q:** does writing shorter lines get more response? Probably, slightly, and it is the only craft measure whose interval does not comfortably contain zero.

### M2.2 · Form band ladder — three small multiples, shared x
- **mark** dot + 90% bootstrap whisker · **x** median like rate, **shared scale across all three panels** · **y** band label · **colour** ink; leading band's dot filled `M4` · **facet** measure — sung lines (`<60` / `60–80` / **`80–100`** / `100+`) · repeated-line ratio (`0–.15` / `.15–.25` / `.25–.35` / **`.35+`**) · duration (`157–205` / `208–239` / **`241–269`** / `272–327` s) · **feed** `suno`
- Band labels carry their n inline. Bands with n<10 render in grain hatch with the n in ember.
- **Banner, mandatory:** *"These bands were chosen after seeing this data. The 80–100 sung-line band and the 241–269s duration band are post-hoc splits of the same songs that suggested them. They cannot be confirmed by the data that generated them. Status: HYPOTHESIS — not tested on new data."* with `[Send to Soundry as a registered hypothesis →]`.
- Measured on the current modern cohort: `sung_lines` ρ = +0.23, `repeated_line_ratio` ρ = +0.13, `duration` ρ = +0.10. Three weak-to-nothing signals, and the ladder should look like three weak-to-nothing signals.

### M2.3 · Suno EMO usage — **whiteboard insight #2. The per-tag leaderboard is REFUSED, and the card says why.**

Measured on the snapshot: **863 distinct EMO tags across 202 clips.** The modal tag is used exactly once. The most-used tag is `triumph` at n=34; then `defiance` 27, `anticipation` 24, `revelation` 21, `dread` 21. (The parallel count on the `output/songs/*.md` corpus is 335 distinct across 43 songs — same verdict, different denominator.) **There is no per-tag performance table that is not noise**, and the app must not draw one.

What ships instead, three panels:
- **(a) Vocabulary census.** **mark** log-log rank/frequency (Zipf) · **x** tag rank · **y** uses · **colour** single ink · **feed** `suno`. **Q:** is this a vocabulary or a phrasebook? Annotation: *"863 tags, 202 clips, median 7 tags per clip. 1 tag would need ~30 clips before its interval was narrower than the whole catalogue's spread. Exactly one tag clears that."*
- **(b) Tag COUNT vs response** — the measurable part, 100% coverage. **mark** hex bin (14px) · **x** `emo_tag_count`, 0–20 · **y** like rate · **colour** Sunset sequential on bin count · **feed** `suno`. Measured ρ = **−0.07**, n=46. Generated headline, tier `NULL`: *"Emotion-tag count does nothing measurable — the effect is inside ±2pp with 90% confidence (n=46)."* That is a real answer and it should be read as one.
- **(c) The missing artifact, named.** *"A per-tag performance read needs a controlled EMO lexicon — a mapping from 863 free-text tags onto ~12 families. `vault/EMO_LEXICON.yaml` does not exist. Until it does, this panel refuses to rank tags, because at a median of one use per tag a ranking would be a list of coin flips sorted by outcome."*

### M2.4 · Genre / style performance — **whiteboard insight #3. Shipped as one screen object, never as a list of rows.** See §13 for the interlock.
- **mark** dot + horizontal 90% bootstrap CI whisker · **x** like-rate difference from the **era-matched** baseline, in **pp**, symmetric axis with a heavy zero rule · **y** style term, **sorted by n descending — never by effect** · **colour** interval position: entirely above `M3` sea-teal / entirely below `M4` ember / **crossing zero → desaturated to ink-faint regardless of the point estimate** · **facet** none · **feed** `suno`
- Every row prints `n` and BH-adjusted `q`. `k` is printed on the card header.
- **Era-control toggle, ON by default.** Switching it off shows the pooled estimate and the dots visibly move. That 180ms motion is the only animated data in the app and it earns its place — it demonstrates the confound instead of describing it.
- **Measured on the current modern cohort (n=46, baseline 4.37%), and every one of these is why the card must not be read as a ranking:** piano n=12 5.43 · tape n=6 5.11 · acoustic n=7 4.95 · punk n=9 4.95 · folk n=8 4.88 · ambient n=9 4.61 · sub n=34 4.44 · choir n=10 4.38 · strings n=5 4.32 · female n=39 4.32 · drone n=9 4.29 · glitch n=18 4.20 · crystalline n=13 4.11 · anthem n=6 3.94 · spoken n=10 3.94 · somatic n=4 3.43 · snarl n=9 3.38 · phonk n=6 2.97 · baile n=5 2.56 · distortion n=1 6.25.
- **Generated headline today, tier `NULL`/`OBSERVATION`:** *"No style term separates from the era-matched baseline once 20 comparisons are adjusted for. The style leaderboard is noise at these cell sizes (smallest n = 1)."* This is the most useful sentence on the tab and it is the one that stops her chasing "up: Baile Phonk."
- **Overlap disclosure, printed above the plot, not in a tooltip:** *"**45 of the 46 modern clips match two or more of these 20 terms.** These are overlapping sets, not a partition; the effects are not additive. Terms are case-insensitive substring matches against the free-text Suno style prompt — a term's presence is not proof it was audible in the render (`output/render-audits/2026-07-24-eight-song-deep-dive/RENDER_AUDIT.md` is the standing proof of that failure class)."*
- **The matcher is disclosed in-card**, in a `<details>`: the term list as editable chips, the case rule, the tokenizer, and the note that no de-overlap policy exists. Plus: *"Recomputed on the 210-row snapshot. The 2026-07-27 prose table used a different, unrecorded matcher over a different cohort; the two do not agree and cannot be reconciled. Example: this matcher gives `ambient` n=9 at 4.61%; the prose reports n=8 at 5.55%. This card never claims to reproduce that table."*

### M2.5 · Trend setters + breakers — **whiteboard insight #4. Partly `[NEEDS FEED: suno × 2]`, partly buildable today.**
Her fourth named insight needs a time derivative. Split honestly:
- **(a) Style trend, LOCKED.** **mark** slope segment · **x** two snapshot dates · **y** term's era-matched effect · **colour** Tide on sign. Renders as a ghost wireframe at 12% ink with a padlock and `unlocks at 2 snapshots · 1 of 2`. It auto-enables; nobody has to remember to build it.
- **(b) Platform trend, LIVE TODAY** — a cross-link, not a duplicate: the Reach tab's Window Ladder (§7.1) is the only genuine trend structure on disk. This panel embeds it at 30% height with `[open in Reach →]`.
- **(c) Setters vs breakers, defined.** A *setter* is a term whose effect rose while its usage rose; a *breaker* is one whose effect rose while its usage fell. Both are two-snapshot quantities. The definitions render now so the card is not re-litigated later.

### M2.6 · Render features vs response — the highest-value unbuilt card in the app
- **mark** scatter, 6 small multiples, shared y · **x** measured render feature — `crest_db`, `loudness_spread_db`, `opening_2s_db`, `stereo.correlation`, `width_dynamics.width_max`, `duration_s` · **y** like rate · **colour** `M4` ember = audited, absent-hatch = unmeasured · **facet** feature · **feed** `suno` ⋈ `output/render-audits/**/measurements.json` (read `utf-8-sig` — that file carries a **BOM** and crashes a naive parse)
- **Q:** does what the generator *actually produced* predict response? This is the only card that connects THE GRAIN LAW to a listener.
- **Coverage, drawn as a stacked bar above the panels:** 8 of 66 renders measured. **Build affordance, not a shrug:** *"58 of 66 renders are unmeasured. `scripts/measure_render.py` is local, offline, and its dependencies are already installed (numpy 2.4.6 / soundfile 0.14.0). One batch run takes this card from n=8 to n=60."*

### M2.7 · The broken rung — a diagram, not a chart
- **mark** 4-node chain with measured edge weights · `output/songs/*.md —2/105→ suno` · `songs.md —weak→ output/mp3` · `mp3 —60/61→ suno (title match)` · `mp3 —8/66→ render audit` · **colour** edge inked `M3` if ≥90%, absent-hatch if <50%
- **Q:** why can't this app tell me which pipeline change moved engagement? Because one rung is 2%, and now she can see which one. Named fix: a `clip_id ↔ run_artifact` map written once, at publish time.

---

## 7 · MUSIC · Tab M3 — **REACH** *(her "Platform Reach")*

**Directive today:** *"Every distributed channel is cooling at once and nothing new has shipped to them. Suno is the only surface where a work's response is visible; everything else is a follower count."* — `FINDING` / `MAKE_IT` / `ops.runbook`.

### M3.1 · Window Ladder — the trend that exists today
The DistroKid feed reports **nested cumulative** windows (`last_7 ⊂ last_30 ⊂ last_90 ⊂ last_365`). Four nested bars read as a cliff that is partly an artefact of the nesting. Difference them into **disjoint buckets** and normalise to a per-day rate:
```
B1 days   1–7   : W7            / 7
B2 days   8–30  : (W30  − W7)   / 23
B3 days  31–90  : (W90  − W30)  / 60
B4 days  91–365 : (W365 − W90)  / 275
Guard: any negative difference ⇒ panel renders UNRECONCILED, marks suppressed, four raw windows printed as text.
```
- **mark** step — a horizontal segment spanning each bucket's actual day range at that bucket's rate, so **segment width encodes duration** · **x** days-before-`pulled_at`, 365 → 0, linear, shared, one axis rule at the bottom of the stack only · **y** events per day, **log, shared**, zero-rate buckets as a 1px rule on the axis floor with a `0` label · **colour** single ink; the most recent bucket takes ember when below its predecessor, `M3` when above · **facet** platform, stacked vertically, 4 panels at 1160×72 · **feed** `distribution`
- Direct labels at each panel's right edge. No legend.
- **Computed live from the snapshot:** Spotify **4.59 → 4.72 → 1.96 → 0.71**/day · Apple **0.05 → 0.93 → 0.61 → 0.14** · YouTube ContentID **0.64 → 43.80 → 7.43 → 1.86** · iTunes/Amazon **0 → 0 → 0 → 0** (a *measured* zero, rendered with real axes and a labelled zero marker — not the never-pulled hatch).
- **Q:** is each channel accelerating or cooling, and by how much per day?
- **What the shape says, and it is the whole argument for this encoding:** YouTube ContentID's 2,987 views are a 60-day spike (43.8/day) that has collapsed 23× to 1.86/day — the same structural signature as Triple Arch on Suno. A big-number tile reading "2,987 YouTube views" reports the opposite of what happened.
- **Caveat in the panel header:** *"DistroKid labels all of this estimated. The four windows are nested and cumulative; each is divided by its own length. A toggle offers the as-reported cumulative view."*

### M3.2 · Platform roster
- **mark** lollipop (dot + rule to axis) · **x** the metric, log₁₀ (range 3,926 → 1) · **y** platform, sorted descending · **colour** **provenance** — api-public `M1` / authenticated-page `M3` / public-page graphite / search-snippet ink-faint + 45° dash / never-pulled absent-hatch · **facet** metric (followers · works · plays/streams · likes), **free x per facet, each labelled with its unit** · **feed** `socials` + `distribution`
- **Units row, mandatory, directly under the header.** Suno plays ≠ YouTube views ≠ Spotify streams ≠ iTunes units. **Cross-platform arithmetic is disabled in the UI; there is no "total reach" number anywhere in this app.**
- **Every delta cell reads `UNKNOWN` in absent-hatch.** Not zero, not blank. The socials feed's own note is the standard.
- Instagram's row is present, `status: not-read`, note column `SKIPPED BY INSTRUCTION`. A missing row invites someone to add it; a present row that says why it is empty does not.
- Nulls render as **"not exposed"**, never 0: TikTok `plays_total` is null; YouTube `likes_total` is null; TikTok's `83K` renders `83,000 ±500 (rounded by TikTok)`.

### M3.3 · Two YouTubes, never summed — a lockout card
Content ID (2,987 views/365d, 100+ countries, music detected across all of YouTube) and the `@lofnai` channel (17 subscribers, 37 videos, 6–168 views each, ~11 months dormant) render in **two blocks with a rule between them and a `surface` label on each**. The app **refuses to compute a combined total**; the sum control does not exist. The feed's own note forbids conflating them, and a lockout is how you honour that.

### M3.4 · Who is listening — the loudest card in the app
- **mark** horizontal bar, **x starts at zero** · **x** views · **y** age band `13-17 · 18-24 · 25-34 · 35-44 · 45-54 · 55-64 · 65+` = **32 · 52 · 244 · 317 · 218 · 235 · 210** · **colour** single ink · **facet** none, with a second small panel for gender (male 835 / female 474) · **feed** `distribution.youtube_contentid`
- **Annotation bracket:** 45+ combined **663** against under-25 **84**.
- **Q:** who is the audience we actually have, versus the one the style prompts are written for?
- **Adjacent text row, deliberately placed, not argued:** the catalogue's stated posture — *"Gen-Z Firestarter"*, *"TikTok-ready hook"*, *"bratty pop-punk snarl"* — with `[open the prompts that say this →]` filtering M1.6 to style terms matching `gen-z|tiktok|snarl|bratty`.
- **Caveat:** *"YouTube Content ID only. This is the one demographic table we possess; Suno, TikTok and NightCafe give us none."*

### M3.5 · Where they are
- **mark** ranked horizontal bar (**not** a choropleth — 30 countries with counts of 22–718 is a ranking, and a map encodes land area as importance) · **x** views from zero · **y** country, top 12 + "other" · **colour** single ink · **feed** `distribution`
- **US 718 · Hong Kong 321 · Taiwan 144 · Russia 134 · UK 130 · Germany 99 · India 77.** Annotation: **HK + TW = 465 against a US of 718, and beat the UK, Germany and Canada combined. Nothing in the pipeline has ever considered an East Asian audience.**

### M3.6 · Discovery path — are we found, or replayed?
- **mark** horizontal bar, **stacked to 100%** — the only stack in the app, gated by the grammar's sum-to-1.00±0.01 rule · **x** share · **y** platform · **colour** source, ordinal 5 levels, direct-labelled · **feed** `distribution`
- Spotify: album 51.7 · **collection 40.9** · others_playlist **5.4** · artist 2.1. Apple: **search 100**.
- **Q:** are people finding us, or replaying us? *"Distributed everywhere, discovered nowhere. 40.9% collection means the same few people replaying."*

### M3.7 · Slope chart — her sketch, staged
- **mark** slope segment + endpoint dots · **x** snapshot date · **y** value indexed to 100 at t₀ (raw in the tooltip) · **colour** platform (quartet + graphite for the 5th) · **facet** metric
- **Today:** each platform is a single dot at t₀ with a **dotted horizon running right to a hollow terminal ring labelled UNKNOWN**. Ghosted at 12% ink, padlocked, `unlocks at 2 snapshots`. Her design is right; it is just not yet renderable, and it will render itself.

---

## 8 · MUSIC · Tab M4 — **HEADWINDS**

Her right-rail boxes are **promoted to a tab and rebuilt as one sorted signal table** — a warning with no comparison class is unreadable; you cannot know whether *"down: Suno uploads"* is remarkable without seeing the ups on the same scale. The three highest-severity items also mirror into a persistent right rail on every other tab, because that is where she drew them.

### M4.1 · The signal table
One table, all signals. Columns: signal · current · **90px sparkline-or-step on a shared axis within a signal family** · slope (per week or per bucket) · n · as-of · provenance glyph · `↗`.
- **Sorted by slope descending.** Tailwinds at top in ink; a single 1px ash rule spanning the table marks zero; headwinds below it in ember.
- **No boxes, no badges, no severity pills.** Position below the rule and ember ink carry it.
- Rows with one observation render a single dot and the text `1 observation — no slope`. That is the honest state and it is also a to-do list.

### M4.2 · Severity × confidence map
- **mark** circle r 6 · **x** confidence (api-public 1.0 / authenticated-page 0.8 / public-page 0.6 / snippet 0.3) · **y** severity, ordinal 1–5 · **colour** category — data / reach / craft / doctrine (the quartet) · **grain-hatch density carries uncertainty; size is not a channel** · quadrant rules at 0.5 / 3
- **Q:** which warning deserves an hour today?

### M4.3 · The standing register — seeded from disk, not invented
Every row requires four fields or it cannot render: **claim · number with its n · provenance chip with as-of · what would falsify this.** A headwind with no falsifier is a mood.
1. **Concentration.** 93.5% of plays are one clip, and `house_lexicon` in `vault/gates.yaml` is built on it. → M0.4.
2. **209 of 210 works are under 1,000 plays.** Median 50.
3. **No history.** One Suno snapshot, one socials, one distribution. Every Suno delta is UNKNOWN.
4. **Every distributed channel is cooling simultaneously** — Spotify 333/90d → 5/7d, Apple 71 → 1, YouTube CID 2,812 → 13. Common cause named as a hypothesis: nothing new has been released to them.
5. **Distribution is not discovery.** Playlist 5.4%, Apple 100% search.
6. **Spotify for Artists is not connected.** The richest reach feed is unreachable; actionable today.
7. **The audience is 35–64; the prompts are written for Gen-Z.**
8. **The catalogue fell 220 → 210 while plays rose.** Cause UNKNOWN.
9. **Policy/evidence mismatch.** The 2026-07-27 recommendations were never applied to `vault/gates.yaml`.
10. **The ingest overwrites `latest.json` even when reconciliation fails.**
11. **Art is our largest audience and our least measured** — 21,000 NightCafe likes, zero attributable to a piece.
12. **NightCafe has an untended community layer** — challenges, chat rooms, and three unread badge counters (28 / 3 / 1) observed but not opened.

### M4.4 · The headwind I refuse to draw
She wrote *"Losing followers."* The snapshot says **3,926 and rising during the pull** (3,925 → 3,926, with 48+ new follows in the preceding five hours). A June search snippet says 1.6K. Those are different surfaces four months apart, not a decline. The card renders **DIRECTION UNKNOWN — two incompatible sources**, shows both, and subtracts neither.

---

## 9 · MUSIC · Tab M5 — **PROVENANCE**

Not an About page. A working table, and the terminus of every `↗` in the app.

- **M5.1 Feed ledger.** One row per `KNOWN_FEEDS` id: feed · `pulled_at` · method · confidence · rows · **I1–I10 invariant strip** (ten 8×8px cells, ink = pass, ember = fail, hollow = n/a; hover names the check; the same data in a visually-hidden table) · notes · `↗` raw file · **the exact refresh command**. Never-pulled feeds render the ingest contract's verbatim empty-state text **in full**, in the notes cell.
- **M5.2 Schema guard / refusal log.** Three incompatible Suno schemas named: `data/snapshots/suno/latest.json` (the one we read), `data/snapshots/socials/latest.json` (headline counters, different shape), `tracking/suno-latest.json` (**REJECTED** — `metrics.plays` is the string `"8.3K"`, 2026-03-28). A refused file is *shown as refused*, so a silently-eaten glob is impossible.
- **M5.3 Drift panel.** Every prose figure beside its live recompute, both dated: n=48/4.51% → n=46/4.37%; 2.35% → 2.46% (n=112); 373,487 → 374,122; 220 → 210. Every `prose-archive` badge in the app links here.
- **M5.4 Catalogue diff — the 220→210 card.** When two snapshots exist: rows added/removed by `id`, bucketed `is_trashed` / `is_public=false` / **cause UNKNOWN**. **Today it renders prose:** *"The 2026-07-27 analysis says 220 in its header and 224 in its own §0 — it contradicts itself. The snapshot says 210. Plays went UP over the same interval. Cause: UNKNOWN."* **The string `"songs deleted"` is in the test suite's forbidden list.**
- **M5.5 Reproduce it.** The pull recipe, copyable, with both gotchas rendered inline as warnings: both sort params mandatory (bare `?page=N` → HTTP 422); start at `page=1` (page 0 aliases page 1 and truncates the walk at 21 clips).
- **M5.6 Glossary.** Every `GuidanceNote` in the app on one printable page. Plus the **Simulate colour vision** control (deuteranopia / protanopia / tritanopia / greyscale) and the reduce-motion toggle.

---

## 10 · ART rail — `?m=art`

**Directive today:** *"Art is our largest audience and our least measured. 21,000 NightCafe likes, and not one of them is attributable to a piece. Pull per-creation engagement for the top 100 and this rail becomes the most useful surface we own."* — `FINDING` / `PULL_THE_FEED` / `ops.runbook`.

The Art rail gets **the same tab skeleton** — Signal / Response / Craft / Reach / Headwinds / Provenance — with the depth honestly marked. Symmetric shells with visible holes beat two missing sections.

### A0.1 · The NightCafe headline *(feed: `socials.nightcafe`)*
No plot. Four stat tiles with rounding stated: `23,000+ works` · `21,000+ likes` · `1,200+ followers` · `PRO tier, 306 credits`. Headline: *"MISSING FROM EVERY PRIOR ANALYSIS. 21,000 likes is 3.55× Suno's 5,913. No snapshot, ledger, or analysis had this surface until 2026-08-02."*
Provenance chip: **public-profile-read · observed · counters displayed rounded, precision low, order of magnitude not in doubt.**

### A0.2 · Audience by surface — the rail's own argument
- **mark** horizontal bar, log x · **x** likes · **y** surface (NightCafe 21,000 · TikTok 83,000\* · Suno 5,913) · **colour** provenance · **feed** `socials`
- \*TikTok's 83,000 is whole-account and spans music and animation; it renders in its own panel with a `not comparable` label rather than in the same bar group. **The app computes no cross-surface total.**

### A0.3 · Measurement depth — the honest census
- **mark** 100% stacked bar · **x** share of works · **y** surface · **colour** data state: engagement measured / work counted only / filename-only / nothing (absent-hatch) · **feed** `socials` + local corpora
- **Today:** NightCafe 23,000 works, **0 with per-work engagement**. `output/past_art/**` 4,690 files with **no sidecar JSON anywhere** — 65.9% parse the filename grammar, 34% do not. `output/images/**` 299 render receipts, the only corpus with **100% reproducible seeds**.

### A0.4 · Production volume *(local corpus)*
- **mark** bar · **x** month, 2025-08 → 2026-02 · **y** files rendered · **colour** single ink; the non-parsing residue as a ghosted segment above each bar · **feed** local `output/past_art/**`
- **Q:** how much are we making, and is the rate falling?

### A0.5 · Renderer mix *(local corpus)*
- **mark** horizontal bar · **x** file count · **y** model (Poe-GPT-5-mini 1,298 · gpt-5-mini 1,170 · gemini-3-flash-preview 242 · …) · **colour** single ink

### A0.6 · The community layer we are not tending — a headwind card
Verbatim from the snapshot: challenges (#ModifierMonday, Hype Crew, NO THEME, Galleria of Portraits), per-creation comment counts, **three unread badge counters (28 / 3 / 1)**, notifications route not yet located. Directive: `PULL_THE_FEED`. **`forbidden:` this card may not trigger any outbound action — replying is on the far side of `vault/AUTONOMY.md` and needs its own explicit approval in chat.**

### A0.7 · The Art ingest gap — the work order
No plot. What to pull (per-creation likes/comments for the top 100 NightCafe creations; challenge placements), the cost, and what it unlocks (a like-rate per piece, a style→response read on the largest surface we own, and the first Art directive this app can honestly issue). `[NEEDS FEED: nightcafe per-creation]`.

**Empty states on the Art rail's Craft and Response tabs** render the Music tab's chart frames ghosted at 12% ink with `[NEEDS FEED: nightcafe per-creation]` and the exact ingest command. Same skeleton, visibly unfilled, is the correct rendering — it makes the gap legible as a gap.

---

## 11 · ANIMATION rail — `?m=animation`

**Directive today:** *"Animation has no audience to learn from. TikTok has 3,542 followers and 83,000 likes with no per-post data; YouTube has 17 subscribers and has been dormant ~11 months. Pull TikTok per-post, or stop counting YouTube as reach."* — `FINDING` / `PULL_THE_FEED` / `ops.runbook`.

- **N0.1 · The dormancy headline.** No plot — a chart of 17 would be decoration. Stat tiles + sentence: *"17 subscribers, 37 videos, per-video views between 6 and 168, ~11 months since the last activity. This channel is not reaching anyone."* Provenance: public-page · observed once.
- **N0.2 · Where animation actually lives.** TikTok: 3,542 followers, 83,000 likes (`±500`, rounded), `works_published: null`, `plays_total: null`. `[NEEDS FEED: tiktok per-post]` — and the recorded failure is named: *"direct fetch blocked, 2026-03-28. Any figure from a snippet must carry `provenance: search-snippet (provisional)`."*
- **N0.3 · Prompt-shape inventory** *(local `output/past_video/*.json`, 210 records)*. **mark** horizontal stacked bar, 2px gaps · **x** share of 210 · **y** one row per dimension (`selected_personality`, `selected_panel`, `model`) · **colour** categorical capped at 5 + "Other". This is the one corpus where personality and panel genuinely vary (LOFN-Prime 153 / LLM Generated 48 / SATISFYDRIP 3 / SPATIALMUSE 3 / CINEMACORE 3). **Q:** what have we actually varied in animation?
- **N0.4 · The Content-ID trap, restated.** A permanent note: YouTube Content ID's 2,987 views are *music detected across all of YouTube*, not animation, and not this channel. They may not appear on this rail.

---

## 12 · The whiteboard's four named insights — dispositions in one table

| # | Her insight | Disposition | Card | Why |
|---|---|---|---|---|
| 1 | lyric length / line | **SHIPS, tier `PATTERN`** | M2.1 | ρ = −0.33, n=46. The strongest single craft signal, and it is still weak. Ships with the word "weak" rendered and the live-vs-proposed gate rules drawn. |
| 2 | Suno EMO usage | **SHIPS AS COUNT ONLY. Per-tag leaderboard REFUSED.** | M2.3 (a/b/c) | 863 distinct tags across 202 clips; modal tag used once; top tag n=34. Tag count vs response is ρ = −0.07, tier `NULL` — a real answer. The missing artifact (`vault/EMO_LEXICON.yaml`) is named on the card. |
| 3 | genre performance | **SHIPS AS ONE SCREEN OBJECT, never as rows.** | M2.4 + §13 | Cell n runs 1–39; 45 of 46 modern clips match ≥2 of 20 terms; k=20. Today's generated headline is a null, and that null is the finding. |
| 4 | trend setters + breakers | **SPLIT.** Style trend LOCKED at `1 of 2 snapshots`; platform trend LIVE via the Window Ladder. | M2.5 + M3.1 | A derivative needs two observations. The DistroKid nested windows are the only real one on disk, and they are genuinely four-resolution. |

---

## 13 · The multiplicity interlock — normative, tested, not a convention

The single largest way this app could lie is by letting one style-token row be read as a finding. Six mechanisms, all enforced in code:

1. **A token row is not an addressable object.** `style_term` rows have no permalink, no card of their own, no export. The addressable object is the board. A row's `Insight.drill` is the board, filtered — never a standalone insight page.
2. **Tier ceiling.** Any insight whose `claim.k_comparisons > 1` and whose `q_bh > 0.10` is capped at `OBSERVATION`. With k=20 and the current cell sizes, **every row is capped today**, and the board's own headline is generated at `NULL`.
3. **Verb lock.** `OBSERVATION` licenses only *cannot be told apart / show no separation*. There is no template at any tier in this app that can say *causes*.
4. **`n_effective`, not `n`.** The printed n is the smallest cell in the comparison — `ambient · n=9`, never `n=46`.
5. **Sort lock.** The board sorts by **n descending**. Sorting by effect is available, clearly labelled, and is not default: sorting a screen by outcome is a fishing expedition rendered as a UI affordance.
6. **Directive lock.** `next_move.magnitude` for every style-token row is hard-capped at `RUN_AN_EXPERIMENT`, `routes_to: soundry.hypothesis`, and `forbidden: "may not become a gate in vault/gates.yaml, may not enter an ICB, may not select a style for the next run on its own."` A style token can never produce a `MAKE_IT`.

Tests (`tests/multiplicity.spec.ts`): *"caps every style-term row at OBSERVATION while k exceeds 1 and q exceeds 0.10"* · *"refuses to emit a MAKE_IT directive from a style-term insight"* · *"prints n_effective, never the cohort n"* · *"defaults the board sort to n descending"*.

---

## 14 · Drill-down, guidance, and refresh

**14.1 Four levels.** L0 insight sentence → L1 evidence plot (expands in place) → L2 rows (M1.6 catalogue, filtered) → L3 work record.

**14.2 Every card declares a target.** In code: `drill: { to, scope }` **or** `drill: null, reason: "<sentence rendered to the user>"`. A card with neither **fails the build** (`tests/drill-registry.spec.ts`). Her sketch says "link to deeper results *if they exist*" — that conditional is exactly how dead links ship. Overruled to: **every insight links, or states in words why it cannot.**

**14.3 L3 work record.** Header (title, created, model, persona, duration) → Engagement (plays, upvotes, like rate + Wilson CI, percentile within era cohort) → Craft (each derived measure drawn as a tick on its own M2.2 band chart, so she sees where *this* song sits) → Prompt (full `tags`) → Lyrics `[JOIN: output/past_music, output/songs — 40.9% structured match; if unmatched, the card says "no local prompt record matched"]` → Render audit `[8/66 measured; if absent, names the local command]` → History `[NEEDS: ≥2 snapshots]`. **Any fuzzy title join is labelled low-confidence on the row itself.** **Reverse drill:** every L3 links back up to its aggregate with the isolating filter already applied.

**14.4 Refresh states its consequence before acting.** *"Pull Suno — about 11 public HTTP requests to `studio-api.prod.suno.com` over ~4s. No credentials sent. Writes one new file under `data/snapshots/suno/`. Does not publish, render, or spend."* Disabled buttons always say why. `competitors`, `market`, `soundry` have **no refresh button at all** — they have a blocker list, because a button that cannot work is worse than no button.

**14.5 Accessibility (inherits the shared contract).** Every plot ships a generated four-sentence description and a real `<details>` data table with the same rounding plus n and CI bounds. One keyboard model across every chart. Hover is never the only path to a value. `title=` appears nowhere.

---

## 15 · App skeleton, data access, build gates

**Pattern:** the Forge house pattern (`E:/git/saga/game-dev/apps/forge-arcade`) — Vite 6 + TS 5.5 + vitest 3, **no `dependencies` key**, `server.mjs` with zero runtime deps, `environment: "node"` tests. Diverges in exactly two documented places: (a) **light + dark are both first-class** (the Forge sheet is dark-only with no semantic pairs), (b) **a type scale is tokenised** (the Forge sheet has none).

```
apps/lofn-analytics/
  package.json  tsconfig.json  vite.config.ts  Dockerfile  .gitignore  README.md  DEPLOY.md
  index.html                      # the whole app; tabs are hash-routed
  server.mjs                      # static + /analytics/snapshots/<feed>/latest.json
  snapshot-route.mjs  .d.mts      # reads data/snapshots/ at request time; shared with the dev middleware
  static-path.mjs     .d.mts
  scripts/
    check-build.mjs               # asserts NO snapshot bytes in dist/, asserts the forbidden-string list
    card-registry-check.mjs       # every card has evidence, question, drill-or-reason  (CI gate)
  src/
    style.css                     # THE ONLY :root token block
    contract.ts                   # FeedId, Snapshot, Clip, Insight, CardSpec — frozen types
    registry.ts                   # import.meta.glob over src/cards/**/index.ts — never hand-edited
    load/  invariants.ts snapshot.ts discriminator.ts     # I1-I10, utf-8-sig, schema guard
    stats/ wilson.ts bootstrap.ts bh.ts era.ts estimator.ts window-ladder.ts
    plot/                         # consumes apps/_shared/plot; adds comb.ts daymark.ts dial.ts step.ts
    shell/ rail.ts filters.ts chrome.ts directive.ts
    cards/ m0-*/ m1-*/ m2-*/ m3-*/ m4-*/ m5-*/ art-*/ anim-*/
  public/
  tests/
```

**Data access.** The app never touches the network. `server.mjs` (and the identical dev middleware) serves `GET /analytics/snapshots/<feed>/latest.json` by reading `../../data/snapshots/<feed>/latest.json` at request time — no copy, no staleness, and **no snapshot bytes ever enter `dist/`** (`scripts/check-build.mjs` asserts it). CSP is `connect-src 'self'`; everything else is `'self'`. **Add `data/snapshots/` to `.gitignore` before the first commit** — it is absent from all 82 current lines and the tree carries follower counts and the full style-prompt corpus.

**Run lock.** First action of any build wave: `python3 scripts/run_lock.py acquire apps/lofn-analytics --run-slug <slug> --engine claude`; heartbeat per lane wave; release after the build report. **Exit 3 means STOP.** Lane discipline is inherited from `EXECUTION.md:44` — each lane writes only its own `src/cards/<id>/` namespace; `package.json`, `index.html`, `vite.config.ts`, `src/registry.ts` and `src/contract.ts` are **coordinator-only, single-threaded, after a wave lands.**

**Test contract** (`tests/**/*.spec.ts`, `environment: "node"`, imports always exactly `import { describe, expect, it } from "vitest"`; no snapshots, no jsdom, no `vi.mock`):
1. `invariants.spec.ts` — I1–I10 against the real file on disk, including the BOM read path and the `tracking/suno-latest.json` rejection.
2. `honesty.spec.ts` — the negative-space test. For every CardSpec the app builds: `evidence.asOf` non-empty; `question` non-empty and ≠ `title`; and `JSON.stringify(spec)` contains none of `4.51`, `373487`, `348663`, `"220"` unless `provenance === "prose-archive"`.
3. `no-fabrication.spec.ts` — with a one-snapshot fixture, no rendered card emits a delta, slope, sparkline, arrow, or growth rate; every delta slot serialises the literal `UNKNOWN`.
4. `multiplicity.spec.ts` — the four assertions in §13.
5. `forbidden-strings.spec.ts` — `"songs deleted"`, `"total reach"`, `"avg artist"`, `"causes"`, `"drives"` appear nowhere in any rendered output.
6. `drill-registry.spec.ts` — every card declares `drill` or `reason`.
7. `window-ladder.spec.ts` — the disjoint differencing, including the negative-difference guard.
8. `server.spec.ts` — spawn `server.mjs` on a `listen(0)` port, poll `/healthz`, assert the snapshot route serves and that `POST /` returns 405 with `set-cookie` null. **There is no write endpoint, and the test's job is to prove it.**
9. `palette.spec.ts` — the CVD validator: CIEDE2000 ≥ 20 for every categorical pair under normal + protan + deutan + tritan, in both themes; text ≥ 4.5:1; marks ≥ 3:1.

**CI** (`.github/workflows/lofn-analytics-ci.yml`, path-filtered): `npm ci → typecheck → test → card-registry-check → build → check-build`.


## Decisions — v2:05436371879d6bde8f9c15ca64431b4b9fa2c5213394e619844cf5bf87f29328
- LIGHT vs DARK — both ship, LIGHT is the default, and this is a documented DIVERGENCE from the Forge house pattern, not a copy of it. Ink Auditor, Task Taxonomist, Headline Editor, Uncertainty Hawk, Affordance Coach and Art Director all independently demanded a light path; Forge is dark-only by construction with hardcoded derived grounds and zero prefers-color-scheme rules. The Scientist wrote 'White background (or light/dark)' and light is genuinely better for dense scatter and small-multiple reading. Dark is a separately specified token set with its own verified quartet and its own measured contrast — filter: invert() is a forbidden string, grep-asserted.
- THE CATEGORICAL SCALE — the Art Director's FOUR-slot quartet wins over the Headline Editor's eight-slot LOFN DUSK and the Interaction Grammarian's five. Reason: it is the only one with published pairwise CIEDE2000 across all three CVD simulations in BOTH themes (25.2 light / 17.4 dark) plus a monotone L* ladder that survives greyscale, and the Headline Editor's own working notes record two of their sunset orderings FAILING (dark magenta-vs-aqua at 1.6, dark gold-vs-red at 13.0). Four is also what any honest plot in this data needs, because platform identity is demoted to row labels and provenance has exactly four levels.
- SUNSET — the Ink Auditor's total ban on sunset inside a plot is REDUCED, not accepted. Sunset owns the sequential ramp (where lightness carries the order and hue collapse costs nothing — the pale end is the horizon, the dark end the zenith) and the chrome. It stays out of categorical encoding and out of the inference triad. The Art Director's argument that lightness-is-altitude is the one perceptually legal home for a warm hue sequence beat the Ink Auditor's blanket exclusion.
- THE DAILY PALETTE — refused as an encoding, delivered as THE DAY MARK. A rotating scale destroys comparison of any screenshot, export or memory against any other, which is the entire purpose of an app opened every day, and 365 palettes cannot be CVD-verified. The Art Director's replacement — a 160px almanac glyph drawn from the day's real feed state in zero new colours, archived to data/daymarks/ — gives her the living-daily-object she wanted and makes it data. Promoted from an Analytics card to shared chrome so it answers the mandate in BOTH apps.
- TYPOGRAPHY — the Art Director's three self-hosted families (Fraunces / Atkinson Hyperlegible Next / IBM Plex Mono) beat the Interaction Grammarian's 'system-ui only, a webfont is 300KB for a dashboard'. This is exactly the clause where the art director wins on HOW it looks: Lofn is known for strong, unique, clean visuals and system-ui is nobody's signature. Cost is bounded to <=200KB latin-subset, self-hosted, font-src 'self' — no CDN, no network font, CSP unbroken. Atkinson also does real work: both apps are required to carry guidance on every element and it was built for small-size character disambiguation.
- CONFIDENCE LEVEL — 95%, one constant (CI_LEVEL in stats.ts, --ci-level in tokens.css), settled once. The Headline Editor and the Uncertainty Hawk both proposed 90% so that something could clear the bar; the measured arithmetic already on record uses 95% (median song 258 plays = +/-2.53pp at 95%), and the Ink Auditor specified Wilson 95. Uniformity across every card in both apps beats a per-seat preference, and 95% is the conservative side of the disagreement — which is the correct side for a tool whose named risk is manufactured confidence. The Soundry may add a 50% inner band; it may not change the outer level.
- THE INFERENCE TRIAD IS SEPARATED FROM THE CATEGORICAL QUARTET. The Ink Auditor wanted teal = interval-excludes-zero-positive and ember = negative; the Art Director froze --m4 ember as 'attention, never bad'. Both survive by splitting the channel: interval position uses the Tide diverging endpoints (--sig-pos / --sig-neg / --sig-null), the quartet keeps its identity bindings, and copy is forbidden from labelling a sign as good or bad. Tide encodes SIGN, not valence.
- MARK COUNT — seven, exactly as the brief scoped: line, band, interval, slope, bar, spark, strip. The Art Director's dumbbell (used for the prose-vs-recomputed drift register and the era-controlled style board) folds into interval as variant:'dumbbell' rather than becoming an eighth mark, because the geometry is identical: two ends, a connector, a gutter n. The Grain Whisker folds in as intervalStyle:'grain'. Scatter, rug, beeswarm and heat-strip fold into strip as postures. Fewer mark types means fewer untested render paths.
- THE ART RAIL IS NOT THE MINOR THIRD OF THE APP. Five of seven seats reasoned from 'Art has 4,690 files and zero engagement rows' and designed Art as an honest empty inventory. The measured facts say NightCafe is the LARGEST surface in the estate — 23K+ creations, 1.2K+ followers, 21K+ likes — larger than Suno by every count. Those seats were reasoning from what is on disk, not from what exists. The kit therefore treats modality as a first-class filter with a real Art feed slot, and the empty state for Art is 'never-pulled', not 'not-applicable'. NightCafe has no feed id in KNOWN_FEEDS — see open question 2.
- MODALITY IS A FILTER, NOT NAVIGATION — five seats converged here independently and I am ratifying it. As three peer rail sections, modality becomes the one dimension that CANNOT carry between tabs because it IS the tabs, which contradicts her own written requirement. As a persistent chip encoded by shape glyph, it carries everywhere, costs no colour channel, and shows its coverage badge before the click instead of after.
- EVERY CHART IS A SPEC, NOT DRAWING CODE — the Interaction Grammarian's PlotSpec/validate/render triple is adopted whole, and it is the load-bearing decision of this part. It makes charts testable under vitest's node environment (the Forge DOM layer is structurally untestable and the recon flags that as a real defect class), it turns every honesty rule from a memory into a build failure, and it makes cross-tab filtering a property of the grammar. Review rule: a card containing hand-written SVG is rejected on sight.
- TWO-SCOPE FILTER STORAGE — the Task Taxonomist's query-string-is-the-analysis / hash-is-the-furniture split is adopted over a single flat URL state. Sharing an analysis should not impose your scroll position on a colleague, and the dimmed-not-dropped applicability rule is the only one of the six competing designs that makes silent filter-dropping visible.
- ALT TEXT IS GENERATED, IN THE AFFORDANCE COACH'S FOUR-SENTENCE SHAPE. The Coach specified the best content (form, reading, extremes with numbers, honest limit); the Grammarian specified the right mechanism (generated from the spec, because hand-written alt drifts from its chart the first time the data changes). Both, merged: generated, four sentences, and a plot whose spec cannot produce the string does not render.
- THE SIGNIFICANT-FIGURE RULE LIVES IN THE FORMATTER, NOT IN AUTHORS' DISCIPLINE. The Uncertainty Hawk's rule — round to the largest place value smaller than half the interval half-width — is the single cheapest mechanical removal of false precision available, and putting it in fmt() means nobody in a hurry can bypass it. 4.49% +/- 2.53pp prints as '4%'.
- TYPE-M SUPPRESSION IS A TIER GATE, NOT A CAVEAT. Above an exaggeration factor of 1.5 the point estimate is REPLACED by the interval and the MDE, and the insight caps at OBSERVATION. At the day-one Soundry design the factor is ~4x with a ~5.5% sign-error probability; a caveat next to a 4x-inflated number is not a control, a refusal to print the number is.


## Overrules ACCEPTED — v2:05436371879d6bde8f9c15ca64431b4b9fa2c5213394e619844cf5bf87f29328
- Modality moves from a left-rail section to a persistent global filter chip encoded by shape (Task Taxonomist, Interaction Grammarian, Affordance Coach, Art Director, Ink Auditor). PRESERVES: modality stays first and always visible, and now genuinely carries between tabs as her requirements demand.
- The daily rotating palette is refused as an encoding and delivered as THE DAY MARK — a glyph drawn from the day's real feed state in zero new colours, archived per day (Art Director). PRESERVES: the app is visibly different every day, and it is different because the day is.
- Sunset is banned from categorical encoding and from the inference triad, and confined to the sequential ramp and the chrome (Ink Auditor, Art Director, reconciled). PRESERVES: the apps look unmistakably like Lofn; the sunset is in the ramp where hue order is perceptually legal.
- AWE / INDIGNATION is encoded by shape + fill + riser in one ink, never by colour, and may not touch an engagement axis while its join to the Suno snapshot is n=0 (Art Director). PRESERVES: the duality as the organising idea and as a first-class filter and arm label, without implying one pole is the error state.
- No sparkline, slope, delta, growth rate, trend arrow or the word 'trend' may render from a single snapshot; the mark ships and refuses, enforced by validate() rule V4, and the slot shows a series-depth glyph that self-heals (Ink Auditor, Uncertainty Hawk, Headline Editor, Affordance Coach, Art Director — unanimous). PRESERVES: 'show trends as well as now', by making the trend's absence the visible thing and by building the history that fills it.
- 'Every insight links to deeper results IF THEY EXIST' becomes 'every insight declares a drill target OR renders the sentence why it cannot', enforced at build time by validate() rule V15 (Task Taxonomist). PRESERVES: her intent exactly, made mechanical so it survives the second month.
- Guidance moves out of hover: tooltips are the second channel, every plot renders its question as a visible subtitle and its evidence (n, as-of, provenance, caveat) as a visible footnote, and title= is banned (Affordance Coach, Interaction Grammarian). PRESERVES: her 'tooltips, alt-text and informational guidance for ALL elements' — more present, not less, because it survives a screenshot, a touch device and a screen reader.
- Every leaderboard and every catalogue chart defaults to like rate; the placement outlier is quarantined by default and drawn beyond a tear glyph in its own gutter wherever plays appear (Ink Auditor, Task Taxonomist, Headline Editor, Art Director, Uncertainty Hawk). PRESERVES: Metric Leaders in her position, leading with the metric that is not one placement event.
- Per-hypothesis effect axes are replaced by ONE shared symmetric axis with a single full-height zero rule (Ink Auditor, Art Director, Headline Editor). PRESERVES: exactly the mark she drew — point estimate, CI whisker, n, #days, [more details] — on a scale where two hypotheses can actually be compared.
- 'Include POWER ANALYSIS' becomes an MDE-at-80%-power column plus an MDE-vs-n curve shown at DESIGN time, never post-hoc power beside a p-value (Uncertainty Hawk, Affordance Coach, Task Taxonomist, Headline Editor). PRESERVES: her explicit instruction, taken literally enough to change a decision instead of adding a footnote.
- Colour is never the sole encoding, and the palette is validated by a runnable CI gate (CIEDE2000 >= 20 across three CVD simulations plus greyscale, both themes) rather than by anyone's eye (Affordance Coach, Headline Editor, Art Director). PRESERVES: 'strong, unique + clean' — clean is a measurable property here.
- Data marks never animate on load or on a data change, and no interval ever tweens its width (Uncertainty Hawk, Ink Auditor, Interaction Grammarian, Art Director). PRESERVES: the app feeling responsive — motion moves to state feedback, where it helps.


## Overrules REJECTED (her intent wins) — v2:05436371879d6bde8f9c15ca64431b4b9fa2c5213394e619844cf5bf87f29328
- REJECTED — the Uncertainty Hawk's 'delete the topline index card entirely'. Her three objections to 'Total Lift vs Avg artist' are all correct and all preserved: no comparator population exists on disk, a set selected for being 'growing/trending' would set the sign of the answer, and our numerator is 93.5% one placement. But she has since specified the estimator she wants — a log-link GLM regressing the metric on the binary treatment indicator, exp(beta) for the lift ratio, Wald SEs for the interval — and named the card DECIBAL. That is a real, general, defensible method. So the CARD ships, named DECIBAL, with logLinkGLM() in the shared kit; only the COMPARATOR is overruled, from 'average artist' to our own dated pre-registered baseline. Her original intent — one top-line number that says whether this is working — wins, and it now measures something that exists.
- REJECTED — the Ink Auditor's blanket ban on sunset inside every plot. Sunset in a CATEGORICAL scale is genuinely indefensible (three warms collapse to CIEDE2000 8.4 under deuteranopia) and that half of the objection is accepted. But a warm hue sequence whose lightness is strictly monotone is a legal sequential scale under all three CVD types and in greyscale, and refusing it would delete the one place her palette can do real encoding work. Her intent — that the app look unmistakably like hers — wins for the sequential ramp; the Auditor's intent — that data ink stay near-monochrome — wins for everything else.
- REJECTED — the Headline Editor's eight-slot LOFN DUSK palette, in favour of the four-slot quartet. Their own validation notes record two prior orderings failing outright, and eight categorical slots is a licence to build charts with eight identities that nobody can read. Her intent (a rich, unmistakable colour world) is preserved in the seven-stop sunset sequential and the nine-stop Tide diverging, where lightness does the ordering work.
- REJECTED — the Interaction Grammarian's 'no webfont file; system-ui is the honest fallback'. The reasoning (CSP, weight budget) is sound and the conclusion is wrong for this product: she is known for strong, unique, clean visuals and system-ui is a default, not a signature. Self-hosted latin-subset woff2 under 200KB with font-src 'self' breaks no constraint. Her intent — the apps must look like Lofn — wins on how it looks, per the moderation rule.
- REJECTED — the Uncertainty Hawk's and the Headline Editor's 90% intervals. Both argued 90 so that findings could clear the bar; that is exactly the reason to keep 95, in a tool whose named failure mode is manufactured confidence. The measured arithmetic already on record (median song 258 plays = +/-2.53pp) is 95%. Their intent — that uncertainty be impossible to look past — is better served by the wider interval, and it is preserved intact in the Grain Whisker, the type-M suppression, and the significant-figure rule.
- REJECTED — the Ink Auditor's 'Headwinds is not a component; delete the boxed warning entirely'. The kit does not ship a boxed-alert component and the Auditor's substantive point (a warning with no comparison class is unreadable) is enforced: every headwind must carry its number, its n, its as-of, its provenance, a slope or an explicit '1 observation — no slope', and a falsifier. But her intent — that bad news is impossible to miss — is a real requirement, and dissolving Headwinds into the bottom of a sorted table is an app-level layout call, not a foundation-level ban. The foundation gives both apps the pieces and refuses to decide it here.
- REJECTED — the seats' shared premise that the Art rail is nearly empty. Five seats designed Art as an honest hole because output/past_art carries no engagement fields. The measured facts show NightCafe is the largest surface in the estate (23K+ creations, 1.2K+ followers, 21K+ likes). The empty state for Art is therefore 'never-pulled', not 'not-applicable', and the modality filter is built for a real Art feed rather than a permanent apology.


## Build order — v2:05436371879d6bde8f9c15ca64431b4b9fa2c5213394e619844cf5bf87f29328
- UNIT 0 (coordinator, single-threaded, BLOCKS everything) — repo preconditions: add `data/snapshots/` + `data/daymarks/` to `E:/git/private-claw/.gitignore` (it currently has only `data/catalog/` and `data/archive/*`); acquire `python3 scripts/run_lock.py acquire apps/lofn-kit --run-slug <slug>` writing to the sidecar `apps/lofn-kit/.lofn/.RUN_LOCK`; add CANONICAL PATHS rows APPS / APP_KIT / SNAPSHOTS / APP_LOCK to BOTH `.claude/skills/lofn/SKILL.md` and `.agents/skills/lofn/SKILL.md`; write `apps/lofn-kit/package.json`, `apps/lofn-kit/tsconfig.json`, `apps/lofn-kit/vitest.config.ts`, `apps/lofn-kit/.gitignore`, `apps/lofn-kit/README.md`.
- LANE A — TOKENS (no dependencies; unblocks every visual lane): `apps/lofn-kit/src/tokens.css`, `apps/lofn-kit/src/tokens.md`, `apps/lofn-kit/src/theme.ts`, `apps/lofn-kit/scripts/validate-palette.mjs`, `apps/lofn-kit/tests/tokens-contrast.spec.ts`, `apps/lofn-kit/tests/palette-cvd.spec.ts`, `apps/lofn-kit/tests/palette-frozen.spec.ts`. Gate: validate-palette.mjs exits 0 in both themes before any lane consumes a token.
- LANE B — TYPES + STATS + FORMAT (pure, no DOM, parallel with A): `apps/lofn-kit/src/plot/spec.ts`, `apps/lofn-kit/src/format.ts`, `apps/lofn-kit/src/insight/stats.ts` (wilson / bootstrapMedian / fisherZ / logLinkGLM / typeM / typeS / mde80 / CI_LEVEL), `apps/lofn-kit/tests/format-sigfig.spec.ts`. Everything downstream imports spec.ts as types only.
- LANE C — SCALES + AXES + VALIDATE (needs B/spec.ts): `apps/lofn-kit/src/plot/scale.ts`, `apps/lofn-kit/src/plot/axis.ts`, `apps/lofn-kit/src/plot/validate.ts`, `apps/lofn-kit/tests/scale.spec.ts`, `apps/lofn-kit/tests/plot-validate.spec.ts` (one it() per rule V1-V18).
- LANE D — MARKS (needs C; the seven files are independent and parallelisable across seven sub-lanes): `apps/lofn-kit/src/plot/marks/{line,band,interval,slope,bar,spark,strip}.ts`. Then the coordinator writes `apps/lofn-kit/src/plot/plot.ts` and `apps/lofn-kit/tests/plot-render.spec.ts` after the wave lands.
- LANE E — PLOT CHROME (needs C, parallel with D): `apps/lofn-kit/src/plot/legend.ts`, `apps/lofn-kit/src/plot/alt.ts`, `apps/lofn-kit/src/plot/table.ts`, `apps/lofn-kit/src/plot/plot.css`, `apps/lofn-kit/tests/alt-text.spec.ts`.
- LANE F — FILTER STATE (needs only B/spec.ts; fully parallel): `apps/lofn-kit/src/state/filters.ts`, `apps/lofn-kit/src/state/url.ts`, `apps/lofn-kit/src/state/chips.ts`, `apps/lofn-kit/tests/filters-url.spec.ts`.
- LANE G — INSIGHT (needs B/stats.ts; parallel with C-F): `apps/lofn-kit/src/insight/insight.ts`, `apps/lofn-kit/src/insight/tier.ts`, `apps/lofn-kit/src/insight/headline.ts`, `apps/lofn-kit/tests/insight-tier.spec.ts`, `apps/lofn-kit/tests/headline-vocab.spec.ts`.
- LANE H — CONTRACTS (needs A tokens + B spec; parallel): `apps/lofn-kit/src/contracts/empty.ts`, `apps/lofn-kit/src/contracts/empty-text.ts` (the verbatim ingest strings), `apps/lofn-kit/src/contracts/stale.ts`, `apps/lofn-kit/src/contracts/provenance.ts`, `apps/lofn-kit/tests/empty-vs-zero.spec.ts`, `apps/lofn-kit/tests/empty-verbatim.spec.ts`, `apps/lofn-kit/tests/stale.spec.ts`.
- LANE I — LOAD + INVARIANTS (fully independent; can start at t=0): `apps/lofn-kit/src/load/snapshot.ts` (utf-8-sig), `apps/lofn-kit/src/load/schema-guard.ts`, `apps/lofn-kit/src/load/invariants.ts`, `apps/lofn-kit/tests/invariants.spec.ts` run against the real `E:/git/private-claw/data/snapshots/suno/latest.json`.
- LANE J — A11Y + TOOLTIP (needs A tokens + E; the only DOM-touching lane): `apps/lofn-kit/src/a11y/sr.ts`, `apps/lofn-kit/src/a11y/focus.ts`, `apps/lofn-kit/src/plot/tooltip.ts`, `apps/lofn-kit/src/plot/keys.ts`.
- LANE K — SIGNATURES (needs A tokens + D/strip + D/interval): `apps/lofn-kit/src/signature/comb.ts`, `apps/lofn-kit/src/signature/grain.ts`, `apps/lofn-kit/src/signature/daymark.ts`, `apps/lofn-kit/tests/signature.spec.ts`.
- UNIT L (coordinator, after all lanes land) — cross-cutting gates and consumer wiring: `apps/lofn-kit/tests/no-forbidden.spec.ts` (greps for `outline: none`, `filter: invert`, the literal `songs deleted`, `vs avg artist`, and any raw hex in `.ts`); the `resolve.alias` + `server.fs.allow` + `tsconfig paths` blocks for `apps/lofn-analytics/` and `apps/soundry/`; `.github/workflows/lofn-apps-ci.yml`; then `python3 scripts/run_lock.py release apps/lofn-kit` and the ledger entry.


## Open questions for The Scientist — v2:05436371879d6bde8f9c15ca64431b4b9fa2c5213394e619844cf5bf87f29328
- PRIVACY GATE ON `data/` — `.gitignore` currently ignores `data/catalog/` and `data/archive/*` but NOT `data/snapshots/`, and the Suno snapshot embeds your profile uuid, follower counts, and the full style prompt of all 210 clips (our complete unreleased style-prompt corpus). Committing it publishes that corpus. RECOMMENDED DEFAULT: add `data/snapshots/` and `data/daymarks/` to `.gitignore` before the first commit, keep every snapshot local-only, and have the apps read from disk rather than from a bundled copy. Say the word if you want the snapshots tracked instead — that is your call, not mine, and it is the only thing in this part that could leave the workspace.
- NIGHTCAFE HAS NO INGEST PATH, AND IT IS THE LARGEST SURFACE — 23K+ creations, 1.2K+ followers, 21K+ likes, bigger than Suno on every count. `scripts/ingest_receiver.mjs` KNOWN_FEEDS has suno, youtube, tiktok, spotify, instagram, competitors, market, soundry — no `nightcafe`. Without it the Art rail ships as a permanent never-pulled panel while being the biggest thing we do. RECOMMENDED DEFAULT: add `nightcafe` to KNOWN_FEEDS now (a one-line change, and the kit is built with the slot), and capture the endpoint and payload shape from one logged-in tab when you are next in there — read-only, no posting, no replies to fans, which stays on the far side of the AUTONOMY line until you say otherwise in chat.


## Decisions — v2:f66200145924df103fec23595b91a14f8df59f55d7c8877bda9cf4779c78c03d
- THREE TABS, NOT FOUR. The Uncertainty Hawk, the Interaction Grammarian, and the Task Taxonomist each proposed a fourth tab (Ledger / pre-registration store). I kept her three-tab IA and made the Ledger a route under Experiment Results (`#/results/ledger`), reachable from every round header and every hypothesis row. Their argument — that a hypothesis board without a pre-registration store behind it is unfalsifiable — is entirely correct and is honoured by making the freeze a SERVER GATE (409 on generate-before-freeze), not by adding navigation. A record is not a workspace.
- CONFIRM (2 arms x 6, blocked on day) IS THE DEFAULT; PLACKETT-BURMAN IS A GATED SECOND MODE. The Uncertainty Hawk's PB-12 is genuinely elegant — twelve runs, eleven factors, each at 6-vs-6 precision — and it was the single strongest technical proposal at the table. I overruled it as the default on two grounds the Hawk's own arithmetic supplies: Resolution III aliases main effects with two-factor interactions, and k=11 under Bonferroni/BH pushes a +15% detection from ~19 days to ~37. With zero maturity curve on disk and one Suno snapshot, the first rounds must be the ones whose answer is unambiguous. SCREEN mode exists, is permanently capped at PATTERN tier, and unlocks only once the accumulation-curve card (B3) has >=14 daily ingests behind it.
- THE LOG LINK IS WHAT LETS THE HYPOTHESIS BOARD SHARE ONE AXIS. The Interaction Grammarian required `facet scales: independent-x` on the hypothesis board because hypotheses in different units share no axis; the Ink Auditor and the Headline Editor required ONE shared axis because two scales cannot be compared. The Scientist's own specified method resolves it: exp(beta) from a log-link GLM makes count, proportion, and mean outcomes all produce a RATIO, so all three share one symmetric log-ratio axis. Only variance ratios get a separate facet, because a ratio of variances is not commensurable with a ratio of means. Both seats get what they were protecting.
- NO RENDER BUTTON AT ALL, OVERRULING THE AFFORDANCE COACH. He argued for a disabled Render control stating its reason inline, so the absence is explained rather than inferred. I overruled it: a disabled button is a promise that the feature exists and is merely gated, and under vault/AUTONOMY.md it does not exist. Replaced by permanent chrome ('This bench writes prompts and plans experiments. It renders nothing, posts nothing, spends nothing.'), a footer line, and a test that asserts POST /soundry/render returns 405 — a test whose entire job is to prove a route does not exist.
- PLATFORM AND PANEL RESPONSE ARE NEVER POOLED, AND EXACTLY ONE IS PRIMARY PER HYPOTHESIS. The Hawk wanted the panel to replace Suno response (n per song becomes controllable rather than waited for); the Headline Editor and Grammarian wanted platform response as the outcome. Both instruments ship; `preregistration.instrument` names one; the other is permanently exploratory; an Instrument Register card (C7) flags any analysis that touched both. Different populations, different noise structures, one number would be meaningless.
- THE CLIP-ID BINDING FIELD IS THE APP'S LOAD-BEARING INVENTION. Recon proves no clip-id-to-artifact join exists anywhere on disk (40.9% structured title match, fuzzy matching adds zero). Rather than build the Soundry on fuzzy matching, the app CREATES the join at the one moment the information exists: the Scientist pastes the Suno clip id back into the card after posting. A variant with no bound clip_id can never enter an analysis. Twelve pastes a day is the price of every number in this app being attributable.
- READ AT A FIXED AGE, NOT A FIXED DATE. No seat raised this and it would have silently wrecked every round. Plays accumulate, so a calendar-date readout compares a 3-day-old song to a 17-day-old one and calls the difference a treatment effect. The analysis selects the observation nearest `readout_age_days` (default 14, tolerance +/-1 day) and records the realised age. Differencing consecutive observations is also what builds the maturity curve that the whole power model currently assumes.
- DIFFERENTIAL ATTRITION IS A HARD SUPPRESSION, NOT A CAVEAT. Also raised by no seat. The most likely way this instrument breaks is the Scientist quietly not posting the arm whose prompts she liked less. If >=2 unposted cards come from one arm and any lacks a recorded `not_posted_reason`, the contrast is suppressed with DIFFERENTIAL ATTRITION named on the card.
- NUISANCE DIMENSIONS ARE BALANCED ACROSS ARMS, NOT RANDOMISED INDEPENDENTLY. At n=6 per arm, independently randomising five nuisance dimensions produces visible imbalance roughly half the time, and visible imbalance is correctly distrusted. Balancing is cheap and it is the difference between twelve variations and twelve confounds. Slot order IS randomised and recorded as a covariate, available as a sensitivity analysis, absent from the primary model.
- HC1 SANDWICH SEs ARE WHAT THE CARD SHOWS, WITH MODEL-BASED AND QUASI-PHI IN THE DRAWER. The Scientist specified Wald SEs; I am implementing Wald with a robust variance estimator because clip-level overdispersion here is enormous (58% of spread is binomial at the median) and a model-based interval would be visibly too narrow — lying in the direction that flatters us. t-quantile when n-p < 30, z otherwise, and the card states which.
- PEEKING IS COUNTED, NOT FORBIDDEN. She will open the results tab daily; twelve cards a day guarantees it. Every load before the decision date appends a `peek` event and increments a visible counter on the round header. A fixed-n stopping rule plus a countable peek log is cheaper and more honest than an alpha-spending boundary nobody will maintain.
- THE POWER MODEL'S CONSTANTS ARE VERIFIED AGAINST THE MEASURED FACTS, NOT ASSERTED. sigma_true = 1.097pp and p-bar = 0.0449 reproduce all three given numbers exactly: 58% binomial share at m=258, a +/-2.53pp per-song 95% interval, +15% MDE at n=114/arm (19 days at 12/day) and at n=48/arm mature (8 days). A test asserts the last two to 3 dp, so the power card cannot silently drift from the record.


## Overrules ACCEPTED — v2:f66200145924df103fec23595b91a14f8df59f55d7c8877bda9cf4779c78c03d
- 'Total Lift vs Avg artist: 123%' -> DECIBAL against the CONCURRENT CONTROL ARM. Every seat that looked at it independently reached the same verdict: there is no average artist on disk, no competitor row has ever been pulled, the five watched targets are unnamed anywhere in the repo, and a comparison set chosen for 'growing/trending' is selected on the outcome. The reference-class overrule is accepted in full and the reason is printed on the card, permanently, in her sight. PRESERVES: one memorable top-line number in the exact position and at the exact size she drew it, that answers 'is the bench working' — against a comparator that exists and is measured on the same days.
- '123%' as a bare scalar -> the number now carries its ratio, its 90% interval, its n, its days, its SE method, and its quantile on the same line. Her own Soundry design already demands CI whiskers and a power analysis; a hero scalar with no uncertainty contradicted the rest of her own page. PRESERVES: the hero numeral as the largest type on the tab.
- 12 free-form cards -> a declared design with an explicit CONTROL arm, one varied factor per treatment card, balanced nuisance dimensions, and a factor fingerprint per card that flags duplicate cells. Twelve variations that differ on every axis at once produce no attributable result and can never populate the hypothesis board she drew. PRESERVES: 'GUIDED EXPERIMENTATION' — guided is the word she wrote, and a design is what makes it guided rather than merely prolific. All twelve cards keep the full Suno quad (Title / Style / Exclude / Lyrics / Caption) unchanged.
- Per-hypothesis effect axes (0/+5/+10/+15/+20 on one row, -10/-5/0/+5/+10 on the next) -> ONE shared symmetric log-ratio axis with a single zero rule spanning the full height of the table. Two scales cannot be compared, which is the only reason to stack them; and an axis starting at zero cannot draw a negative result, which is a significance filter built into the geometry — it would hide every finding that says an idea hurt us. PRESERVES: exactly the mark she drew (point estimate + CI whisker + n + #days + [more details]) — all four survive; only the ruler changes, and her second sketched axis is already symmetric.
- 'Include POWER ANALYSIS' as a note beside a result -> power moves to DESIGN time as an MDE-vs-days curve rendered before a single prompt is written, plus an achieved-MDE column on every hypothesis row. Post-hoc power is a deterministic restatement of the p-value; the minimum detectable effect at the achieved n is the quantity that changes a decision, and it can only change one while the design is still editable. PRESERVES: her instruction, taken literally enough to alter the experiment rather than annotate it.
- 'Take winners + make full songs' at the end of a day -> no daily winner, ever. 58% of the visible between-song spread is binomial noise at our median 258 plays; the top of twelve is a coin flip with a leaderboard skin. Winners are ARMS, declared after the pre-registered n. Where a winner would go, the app renders the refusal card with the arithmetic in it. PRESERVES: both halves of her stated goal — '(1) learn, (2) take winners + make full songs.' Winners still get promoted to full songs through the lofn front door; they are promoted on evidence that will survive.
- A point estimate from the moment a hypothesis has any data -> the hypothesis card is born as a WIDTH: a Grain Whisker spanning +/-MDE, the words 'Not yet measurable', n so far / n needed, day X of Y, the decision date computed from the realised cards-per-day, and the achieved MDE at today's n. No dot until the pre-registered n; the estimate is re-suppressed if the type-M exaggeration ratio exceeds 1.5. PRESERVES: her explicit demand for a power analysis, made into the thing that decides whether a number may be drawn at all.


## Overrules REJECTED (her intent wins) — v2:f66200145924df103fec23595b91a14f8df59f55d7c8877bda9cf4779c78c03d
- REJECTED (Uncertainty Hawk): route the 12 daily test songs off the main @localsymmetry profile, or cut the cadence. His interference and dilution arithmetic is sound — 12/day for 30 days is 360 songs against a 210-song catalogue built over 13 months, and the cards do compete for the same listeners. But the whole point of the platform instrument is Suno response on OUR profile; moving the tests elsewhere measures a different audience and the result does not transfer. And the app cannot enforce it in any case: under AUTONOMY the Scientist posts, not the app. Her intent — 12 song tests a day on the surface that matters — wins. What I accept from his seat instead: one round open at a time (423 Locked on a second), a recorded `concurrent_soundry_load` covariate, and the cadence field defaulting to 12 with the power card showing live what 6 buys. This is the one item I would put back on her desk (see open questions).
- REJECTED (Uncertainty Hawk, as default): Plackett-Burman 12-run screening of up to 11 factors as the standard round. Kept as a second, gated, tier-capped mode. Resolution III aliasing plus a k=11 multiplicity penalty (his own table: 21 days -> 37) makes it the wrong first instrument when zero maturity curve exists. Her intent — that a day of tests produce a usable answer — is better served by one clean contrast than eleven aliased ones.
- REJECTED (Task Taxonomist, Interaction Grammarian, Uncertainty Hawk): a fourth tab. She specified three tabs: Sonic Anvil / Experiment Results / Analytics. The Ledger is a route under Experiment Results and is linked from every round header and every hypothesis row; the pre-registration gate itself is enforced server-side. Her IA wins; the credibility organ is preserved without spending navigation on a record nobody browses.
- REJECTED (Affordance Coach): a disabled Render button stating its reason inline. A disabled control promises a gated feature; under vault/AUTONOMY.md rendering is not gated, it is absent. Replaced by permanent chrome, a footer line, and a test asserting the route 405s. His underlying intent — that the absence be explained rather than inferred — is fully preserved and arguably better served.
- REJECTED (Headline Editor, partially): tier-locking causal verbs out of the entire estate. Correct for Lofn Analytics, wrong here. A pre-registered CONFIRM round with within-day randomisation and a concurrent control is the one place in this workspace where a causal claim is licensed, and refusing it would waste the only randomisation we have. SCREEN-mode aliased effects and historical-mode comparisons stay locked out.
- REJECTED (Ink Auditor, partially): deleting the top-line index card and folding it into the forest plot as one more row. She wants a top-line index and she is right to — it is the card that says whether the bench justifies itself. DECIBAL keeps its slot, its size, and its prominence; it just names its reference class and carries its interval.


## Build order — v2:f66200145924df103fec23595b91a14f8df59f55d7c8877bda9cf4779c78c03d
- U1 · SCAFFOLD (no deps beyond the shared foundation) — apps/soundry/package.json · apps/soundry/tsconfig.json · apps/soundry/vite.config.ts (4 entries: index.html Anvil, results.html, analytics.html, panel.html; base './'; port 5231; test block) · apps/soundry/index.html · apps/soundry/results.html · apps/soundry/analytics.html · apps/soundry/panel.html · apps/soundry/.gitignore (node_modules/, dist/, runs/) · apps/soundry/README.md · apps/soundry/DEPLOY.md (local-only posture table) · apps/soundry/src/style.css (imports the shared token block, declares nothing) · apps/soundry/scripts/check-local-only.mjs
- U2 · STATS CORE (parallel with U3, U4; depends only on U1) — apps/soundry/glm.mjs + glm.d.mts (IRLS, log link, quasi-Poisson / Poisson-with-offset / Gamma / Gamma-on-variances, phi-hat, HC1 sandwich, Cholesky with ridge fallback, condition-number guard, NOT-ESTIMABLE returns) · apps/soundry/src/power.ts (SE_log, MDE_ratio, days) · apps/soundry/src/retrodesign.ts (power, type-S, type-M by 10k fixed-seed truncated-normal draws) · apps/soundry/src/design.data.json (p_bar 0.0449, sigma2_true 1.204e-4, z-sum 2.802, each with a provenance comment) · apps/soundry/src/design.ts wrapper header comment naming who edits the JSON · tests/glm.spec.ts · tests/power.spec.ts (asserts n=114/m=258 -> 1.150 and n=48/mature -> 1.150 to 3dp) · tests/retrodesign.spec.ts
- U3 · LEDGER STORE (parallel with U2, U4) — apps/soundry/round-store.mjs + round-store.d.mts (round CRUD, append-only jsonl writers, utf-8-sig readers) · apps/soundry/src/contract.ts (Preregistration, Design, Variant, SunoObservation, PanelObservation, AnalysisResult, RoundEvent) · apps/soundry/src/canonical-hash.ts (sorted-key canonical JSON -> SHA-256) · tests/round-store.spec.ts · tests/prereg-freeze.spec.ts (409 generate-before-freeze, 403 mutate-after-freeze, hash stable under key reordering)
- U4 · DESIGN + SAMPLER (parallel with U2, U3) — apps/soundry/src/assignment.ts (CONFIRM 2x6 within-day randomisation blocked on day; SCREEN Plackett-Burman 12-run matrix; seeded PRNG; slot permutation) · apps/soundry/src/sampler.ts + apps/soundry/src/sampler.data.json (5 nuisance dimensions, arm-balanced stratified draw) · tests/design.spec.ts (byte-identical regeneration from rng_seed; every nuisance level balanced across arms; PB matrix orthogonality)
- U5 · SERVER (after U3) — apps/soundry/server.mjs (127.0.0.1 bind only; GET /soundry/health · GET|POST /soundry/rounds · POST /soundry/rounds/:id/freeze · POST /soundry/rounds/:id/generate · POST /soundry/rounds/:id/variants/:vid/clip · POST /soundry/observations/panel · POST /soundry/events · GET /soundry/analysis/:round/:contrast · POST /soundry/emit-snapshot; 403 / 409 / 423 / 405; no CORS wildcard) · apps/soundry/static-path.mjs + .d.mts · tests/server-soundry.spec.ts (spawned server on a node:net port, DATA_DIR in mkdtempSync, every refusal, and POST /soundry/render -> 405 with set-cookie null) · tests/autonomy.spec.ts (bundle grep: no external origin, no generation-API hostname, no render handler)
- U6 · GRAIN WHISKER MARK (after U1; parallel with U5) — apps/soundry/src/plot-ext/grain-whisker.ts (extends the shared `interval` mark: 10px hatched rect, pitch = clamp(2 + 26/sqrt(n), 2, 14), 9px point rule + 3.5px sign disc, ember Power Notch at the MDE, no animation) · apps/soundry/src/plot-ext/grain-whisker.css · tests/grain-whisker.spec.ts (pitch at n = 4 / 9 / 25 / 100 / 400; refuses to render without nField)
- U7 · SONIC ANVIL UI (after U4, U5, U6) — apps/soundry/src/anvil/model.ts (PURE: factor parsing, mode rules, power curve spec, balance check, fingerprint diff) · apps/soundry/src/anvil/main.ts (all DOM/fetch/history; the four routes; per-field copy with aria-live; clip_id binding; state chips; no render affordance) · apps/soundry/src/anvil/anvil.css · tests/anvil-model.spec.ts
- U8 · EXPERIMENT RESULTS UI (after U5, U6; parallel with U7) — apps/soundry/src/results/model.ts (PURE: maturity board rows, fixed-age readout selection with +/-1d tolerance, eligibility + exclusion reasons, differential-attrition detector, peek counting) · apps/soundry/src/results/main.ts (round list, round detail, variant drawer, #/results/ledger route) · apps/soundry/src/results/results.css · tests/results-model.spec.ts (fixed-age selection; DIFFERENTIAL ATTRITION fires at 2 unposted from one arm without a reason)
- U9 · ANALYTICS UI + DECIBAL (after U2, U5, U6; parallel with U7, U8) — apps/soundry/src/decibal.ts (assembles the design matrix with day fixed effects, picks family from preregistration.primary_outcome.family, calls glm.mjs, returns lift ratio + Wald HC1 CI + quantile used + dropped-block list + refusal reason) · apps/soundry/src/analytics/model.ts (PURE: hypothesis board rows, BH q, shared log-ratio axis domain, tier gating, the fourteen refusals) · apps/soundry/src/analytics/main.ts · apps/soundry/src/analytics/analytics.css · tests/decibal.spec.ts · tests/refusals.spec.ts (each of the fourteen asserted as an absence in the serialised card)
- U10 · PANEL INSTRUMENT (after U3, U5; parallel with U7-U9) — apps/soundry/src/panel/main.ts + panel.css (blind session: titles and arm labels hidden, per-listener randomised order, rating + optional free text, elapsed seconds, presentation position) · apps/soundry/panel-identity.mjs + .d.mts (anon-<hmac16> pseudonym, per-session secret, no name ever written) · reuse of the shared input-cleaning.mjs for free text (analysisEligible flag) · tests/panel-identity.spec.ts (JSON.stringify(row) contains no listener name; flagged notes stored with analysisEligible false)
- U11 · OBSERVATION JOIN + SNAPSHOT EMITTER (after U3; parallel with U7-U10) — apps/soundry/observation-join.mjs + .d.mts (reads data/snapshots/suno/*.json, joins by bound clip_id, appends one row per variant per pull to observations/suno.jsonl, computes age_days and per-day increments) · apps/soundry/emit-snapshot.mjs + .d.mts (writes data/snapshots/soundry/<stamp>.json + latest.json in the ingest-contract envelope, closed+frozen rounds only, never an open round) · tests/observation-join.spec.ts · tests/emit-snapshot.spec.ts (an open round never appears in the emitted rows)


## Open questions for The Scientist — v2:f66200145924df103fec23595b91a14f8df59f55d7c8877bda9cf4779c78c03d
- CADENCE AND SURFACE — do the 12 daily test songs go to the main @localsymmetry profile? Twelve a day for thirty days is 360 songs against a 210-song catalogue built over thirteen months; the tests will compete with each other and with the catalogue for the same listeners, which is a real threat to the no-interference assumption every interval in this app rests on. It is also the one thing the app cannot decide, because you post, not it. RECOMMENDED DEFAULT: keep 12/day on the main profile exactly as drawn — that surface is the instrument and a side profile measures a different audience — with three mitigations shipped: one round open at a time (423 Locked), a recorded concurrent-load covariate, and the power card showing live what 6/day buys instead of 12.
- THE PANEL — does a listener panel exist, and who is on it? 'Get feedback from listeners' is on the whiteboard, but nothing on disk records a panel size, a recruitment route, or a single listener. A panel is statistically far better than waiting for plays (n per song becomes something you control), but if it does not exist the first rounds must run on Suno response alone and the timelines in this spec are the ones that apply. RECOMMENDED DEFAULT: ship the panel storage, the blind session page, and the pseudonymisation, but pre-register round 1 with instrument = platform and readout_age_days = 14, and let the accumulation-curve card revise that age once fourteen daily ingests exist.


## Decisions — v2:eb88413f8b4495226663676e3ddd62f85a7f49882afeb14bd5461c4ffc0188be
- THE RAIL STAYS A RAIL — I overruled five of seven seats. The Ink Auditor, Task Taxonomist, Interaction Grammarian, Art Director and Affordance Coach all independently voted to demote Art/Music/Animation from left-rail navigation to a filter chip, and every one of them argued it on the same premise: that Art has 4,690 files and zero engagement rows, so a rail section would be an empty room. That premise died with the socials pull. data/snapshots/socials/latest.json carries a NightCafe row — 23,000 works, 1,200 followers, 21,000 likes, PRO tier, status 'major-untracked'. Art is 3.55x Suno's likes. It is our LARGEST surface. So the rail survives as she drew it. But their carry-through objection is correct and is honoured mechanically: the rail is not navigation, it is the rendered control for the ?m= URL parameter, so modality carries across every tab by construction. Rail order is computed by measured audience descending (Art, Music, Animation) and each item carries TWO bars — audience size and measurement depth (0-5). Art scores 1/5 on 21,000 likes. That gap is the Art rail's headline, and putting Art first is a finding, not a layout preference.
- METRIC LEADERS: like rate wins over plays, unanimously across all seven seats, and the sort key is the WILSON LOWER BOUND (Interaction Grammarian) rather than the point estimate (implicit in the Ink Auditor and Headline Editor cards). The Grammarian wins: at a 258-play median the 95% interval is +/-2.53pp, so a point-estimate ranking of 210 songs is a ranking of dice. The Ink Auditor wanted the leaders card deleted entirely in favour of the cohort table; rejected — she asked for leaders, and leaders-by-lower-bound is a real answer to a real question. Both ship: M1.1 is the leaderboard, M0.3 is the cohort table.
- PLATFORM REACH: the Ink Auditor, Interaction Grammarian and Headline Editor all converged independently on the same replacement for the bump/slope chart — difference the DistroKid nested cumulative windows into four DISJOINT buckets and normalise to a per-day rate. They are right and it is the only genuine trend structure on disk. But the Task Taxonomist's framing (a 'window ladder' answering 'is this channel accelerating or decaying RIGHT NOW') is the better card title, and the Ink Auditor's step encoding — segment WIDTH encodes the bucket's day span — is the better mark, because it stops '7 days' and '275 days' reading as equally-weighted observations. Merged: Taxonomist's question, Ink Auditor's geometry. Her slope chart is ALSO built, ghosted at 12% ink with an unlock counter, because it is the right chart and it will draw itself on the second pull.
- COLOUR: the Art Director wins on HOW, the Ink Auditor wins on HOW MUCH. The Art Director's frozen four-slot categorical quartet (verified at min pairwise CIEDE2000 25.2 light / 17.4 dark across four vision models, with an L* ladder that survives greyscale) plus sunset moved into the sequential and diverging ramps is the correct resolution of 'sunset colors, daily palette' — it gives her the warmth in the one place a warm hue sequence is perceptually legal. The Ink Auditor's near-monochrome ban was rejected as a palette but SURVIVES AS A PER-PLOT CAP: at most three inks inside any single chart, and teal/ember encode interval-excludes-zero and nothing else. The 'daily palette' becomes the Day Mark — a 96px almanac glyph redrawn daily from the day's actual feed state, zero new colours, archived to data/daymarks/. A rotating encoding scale is a data defect: it makes yesterday's screenshot incomparable with today's, and I can verify one quartet across four vision models but not 365.
- EMO TAGS: I measured before deciding, and the measurement is worse than the brief's warning. 863 distinct tags across 202 clips in the snapshot (the brief's 335-across-43 is the parallel count on the output/songs/*.md corpus — same verdict, different denominator). The modal tag is used exactly ONCE. The top tag is triumph at n=34. There is no per-tag performance table that is not noise, so the app REFUSES to draw one and says why on the card. What ships is the part that is measurable: emo_tag_count vs like rate at 100% coverage, which comes out at rho = -0.07 and generates a NULL-tier headline — 'emotion-tag count does nothing measurable.' That is a real answer, not a missing one. The missing artifact is named on the card: vault/EMO_LEXICON.yaml does not exist.
- STYLE / GENRE PERFORMANCE: the Headline Editor's tier ladder and the Interaction Grammarian's validate() approach are fused into a six-point interlock (spec section 13) because neither alone is sufficient. The Editor's tier system stops the WORDS over-claiming; the Grammarian's registry stops the STRUCTURE over-claiming. The decisive number is one I measured: 45 of the 46 modern clips match two or more of the 20 style terms. These are overlapping sets, not a partition, the effects are not additive, and cell n runs from 1 to 39. So a token row is not an addressable object at all — no permalink, no standalone card, no export, sort locked to n descending, and next_move.magnitude hard-capped at RUN_AN_EXPERIMENT so a style token can never produce a MAKE_IT. Four vitest assertions enforce it.
- STALE DATA: the Affordance Coach and the Uncertainty Hawk disagree directly. The Hawk's ink-is-a-function-of-certainty rule would desaturate an old number; the Coach argues stale is OLD, not WRONG, and greying an accurate historical figure teaches her to distrust correct numbers. The Coach wins on stale (full ink, persistent age banner, present tense becomes a reserved word). The Hawk wins on UNPOWERED (hollow marks, no dot, interval only) — those are different failure modes and they get different ink.
- THE ART DIRECTOR'S AWE/INDIGNATION ENCODING is adopted wholesale and its own constraint is honoured: the duality never touches an engagement axis in this app, because zero of the 18 labelled songs join the Suno snapshot by normalized title. n = 0. It appears only as a corpus-composition dimension and as a Soundry arm label, encoded by shape plus riser, never by colour, and neither pole ever receives the pass colour or the attention colour.
- TWO CARDS THAT NOBODY ASKED FOR AND BOTH SHIP. M1.5 Model Version (the Interaction Grammarian's find): major_model_version splits 61/67/42/40 across v4.5 / v4.5+ / v5 / v5.5, near-perfectly confounded with time, sitting directly underneath the 'engagement tripled' claim, and nothing on disk has ever mentioned it. M3.4 Who Is Listening (the Grammarian and Taxonomist both flagged it): 45+ outnumbers under-25 by 663 to 84, and the card is placed deliberately adjacent to our own stated posture — 'Gen-Z Firestarter', 'TikTok-ready hook', 'bratty pop-punk snarl' — with a drill link to the prompts that say it. Putting two things next to each other so the contradiction is SEEN rather than argued is the most useful thing this app does.
- THE ESTIMATOR IS A CONTROL, NOT A CONSTANT. On the identical n=46 modern cohort the same rows yield median 4.37%, mean 4.29%, and pooled 1.57%. The brief's blessed baseline of 4.49% matches none of the three exactly. That is not a discrepancy to hide, it is an argument: every card prints the estimator name inline and the shelf carries an estimator control, because a like rate with no named estimator is not a number.


## Overrules ACCEPTED — v2:eb88413f8b4495226663676e3ddd62f85a7f49882afeb14bd5461c4ffc0188be
- Metric Leaders leads with LIKE RATE, not plays; plays are demoted to a labelled secondary card behind a tear glyph and an interstitial. PRESERVES her intent that the card tells her at a glance what is winning — it now tells her what is actually winning rather than what was once placed. One clip is 93.5% of all plays at the worst modern like rate in the catalogue (1.44% vs 4.37%), so a plays leaderboard is a leaderboard about one placement event.
- The sparkline squiggles beside Plays and Likes are replaced by a 28-tick snapshot dial reading '1 snapshot — a delta needs two.' It becomes a real sparkline automatically at >=3 snapshots with no redesign. PRESERVES 'show trends as well as now' by making the trend's ABSENCE the thing on screen, and by shipping the one real trend that does exist today (the Window Ladder) on the Reach tab.
- Platform Reach's bump/slope chart is replaced TODAY by a per-day-rate step chart over four disjoint buckets derived from the DistroKid nested windows; her slope chart is built, ghosted at 12% ink, padlocked with 'unlocks at 2 snapshots', and auto-enables. PRESERVES her exact chart, staged rather than faked — and delivers a genuine four-resolution trend on day one. A bump chart would also rank Suno plays against Spotify streams against YouTube views, which is a category error no legend can repair.
- 'Headwinds' as boxed warnings in a right rail becomes ONE sorted signal table shared with the tailwinds, split by a zero rule, headwinds in ember below it, plus a severity-x-confidence map. The three highest-severity items still mirror into a persistent right rail where she drew them. PRESERVES the urgency and the boxes; adds the comparison class that makes a warning legible — you cannot know whether 'down: Suno uploads' is remarkable without seeing the ups on the same scale.
- 'Losing followers' is REFUSED as a headwind and rendered as DIRECTION UNKNOWN with both incompatible sources shown. The snapshot says 3,926 and RISING during the pull (3,925 -> 3,926, 48+ new follows in the preceding five hours); the 1.6K figure is a June search snippet from a different surface. PRESERVES her intent that bad news is impossible to miss — by not spending that slot on a decline we did not measure.
- 'Every insight links to deeper results IF THEY EXIST' becomes 'every insight declares a drill target OR renders the sentence naming why it cannot', enforced by a build-time test over the card registry. PRESERVES her intent exactly, made mechanical so it survives the second month — 'if they exist' is evaluated by a developer at build time and never re-evaluated when a feed goes empty.
- Both light and dark ship as separately specified token sets with per-theme verified palettes, diverging from the Forge house pattern which is dark-only with hardcoded derived grounds and no prefers-color-scheme block anywhere. PRESERVES her literal 'White background (or light/dark)'. Flagged as a real cost, not a checkbox: every derived ground in the inherited sheet needs a semantic pair.
- Every tab ends with a directive answering 'so what do I make next?', generated from the argmax insight and bound to its tier (ABSENT -> PULL_THE_FEED, OBSERVATION -> RUN_AN_EXPERIMENT, NULL -> STOP_SPENDING_ON_IT, PATTERN -> TRY_A_PAIR, FINDING -> MAKE_IT), with a 'forbidden' line always filled. PRESERVES her requirement that all headliner insights guide the next creation — and stops a rounding error wearing a verb.
- A PROVENANCE tab is added, not on the whiteboard. PRESERVES nothing of hers directly and is justified on its own: with 3 of 8 feeds fed, three incompatible Suno schemas coexisting on disk, and a documented 220->210 discrepancy nobody can explain, the app's central risk is a beautiful lie. Every caveat elsewhere in the app links here.
- Causal verbs are locked out of the entire app. Nothing here is randomised, so no headline may say causes/drives/boosts/kills; the vocabulary unlocks only in a pre-registered Soundry round. The footer of every page states the split. PRESERVES her intent that these insights be actionable, by making the one place that can license action clearly distinguishable from the one that cannot.


## Overrules REJECTED (her intent wins) — v2:eb88413f8b4495226663676e3ddd62f85a7f49882afeb14bd5461c4ffc0188be
- REJECTED: demoting the Art / Music / Animation rail to a filter chip (Ink Auditor, Task Taxonomist, Interaction Grammarian, Art Director, Affordance Coach — five of seven seats). All five argued from the same dead premise: that Art has zero engagement data and would be an empty section. The socials pull that landed after they wrote carries a NightCafe row with 23,000 works, 1,200 followers and 21,000 likes — 3.55x Suno's likes. Art is our largest surface, not a minor third. Her original intent — that these three are how she thinks about the work, at the top level, always visible — wins, and the fact base now supports it better than it supported the panel. Their carry-through requirement is satisfied anyway by making the rail the rendered control for the ?m= URL param.
- REJECTED: the Ink Auditor's deletion of the Metric Leaders card in favour of the cohort table alone. She asked to see who is winning; that is a legitimate question with a legitimate answer (rank by the Wilson lower bound). The cohort table ships too, on the Signal tab. Both, not one.
- REJECTED: the Ink Auditor's near-monochrome data ink for the whole app. Lofn is known for strong, unique, clean visuals, and a warm-neutral page with one grey ink is neither unique nor Lofn. The Art Director's verified four-slot quartet plus sunset ramps wins on look; the Ink Auditor's discipline survives as a per-plot cap of three inks and as the rule that teal/ember encode interval-excludes-zero and nothing else.
- REJECTED: the Uncertainty Hawk's rule that a number may not render before it has power, applied to descriptive counts. 374,122 plays and 210 works are COUNTS, not estimates — they carry no sampling uncertainty and they render. The Hawk's rule is retained in full force for every RATE, EFFECT and DELTA, which is where it belongs.
- REJECTED: the Uncertainty Hawk's implication that the Art and Animation rails should stay dark until they have engagement data. They render inventory, production volume, renderer mix, the measurement-depth census, and a named work order. A visible hole with a command under it is a working panel; a dark rail is a rail she will stop clicking.
- REJECTED: the Art Director's Day Mark at 160px hero on every tab. Reduced to 96px in the header, full size only on Provenance. It is a genuinely good device and it answers 'daily palette' honestly, but at hero scale on six tabs it starts spending more pixels than the plots it sits above — which is the Ink Auditor's objection, and on this one he is right.
- REJECTED: the Uncertainty Hawk's desaturation of stale-but-accurate figures. The Affordance Coach wins: stale is old, not wrong, and greying a correct historical number teaches her to distrust correct numbers. Stale renders at full ink with a persistent age banner, and the present tense becomes a reserved word instead.


## Build order — v2:eb88413f8b4495226663676e3ddd62f85a7f49882afeb14bd5461c4ffc0188be
- W0 · COORDINATOR, SERIAL, BEFORE ANY LANE. `python3 scripts/run_lock.py acquire apps/lofn-analytics --run-slug <slug> --engine claude` (exit 3 = STOP). Add `data/snapshots/` to `E:/git/private-claw/.gitignore`. Scaffold coordinator-only files, which no lane may touch afterwards: apps/lofn-analytics/package.json · tsconfig.json · vite.config.ts · index.html · .gitignore · src/contract.ts (FeedId, Snapshot, Clip, CardSpec, Insight — frozen) · src/registry.ts (import.meta.glob over src/cards/**/index.ts, marked never-hand-edited).
- W1-L1 · LOADER + INVARIANTS (parallel with L2, L3). apps/lofn-analytics/src/load/snapshot.ts (utf-8-sig read, BOM strip) · src/load/invariants.ts (I1-I10, hard-fail table) · src/load/discriminator.ts (feed+handle check, rejects the tracking/suno-latest.json shape) · snapshot-route.mjs + snapshot-route.d.mts (reads ../../data/snapshots/<feed>/latest.json at request time; imported by BOTH server.mjs and the vite dev middleware) · tests/invariants.spec.ts.
- W1-L2 · STATS KERNEL (parallel with L1, L3). apps/lofn-analytics/src/stats/wilson.ts · bootstrap.ts (10k percentile) · bh.ts (Benjamini-Hochberg) · era.ts (within-month matching) · estimator.ts (median/mean/pooled, named) · window-ladder.ts (nested->disjoint differencing + the negative-difference guard) · matcher.ts (style-term substring matcher, term list as data, overlap census) · tests/window-ladder.spec.ts.
- W1-L3 · PLOT PRIMITIVES (parallel with L1, L2; consumes apps/_shared/plot, adds nothing generic). apps/lofn-analytics/src/plot/comb.ts (210 hairlines, log x, tear glyph, outlier gutter, brush) · src/plot/daymark.ts (24 hour ticks, 8 feed segments, log-r core, ember strike) · src/plot/dial.ts (28-tick snapshot ring) · src/plot/step.ts (disjoint-bucket step, width encodes day span) · src/plot/tear.ts · src/plot/hatch.ts (absent-hatch, grain hatch by n).
- W2-L4 · SHELL (needs L1+L3). apps/lofn-analytics/src/style.css (THE ONLY :root block; light+dark semantic pairs; tokenised type scale) · src/shell/rail.ts (computed order, two bars per item, glyph-encoded) · src/shell/filters.ts (10 URL params, dimmed-not-dropped applicability, zero-result relaxations) · src/shell/chrome.ts (header, reconciliation pill, UNRECONCILED global gate) · src/shell/directive.ts (tab directive from argmax insight) · src/main.ts (hash router, popstate).
- W3-L5 · MUSIC SIGNAL + RESPONSE (needs L4). src/cards/m0-dial/ · m0-response-index/ · m0-era/ · m0-placement/ · m0-standing-findings/ · m0-movers-locked/ · m1-response-leaders/ · m1-reach-leaders/ · m1-noise-floor/ · m1-response-vs-reach/ · m1-model-version/ · m1-catalogue-table/ (each dir: index.ts exporting one CardSpec, plus its own spec file).
- W3-L6 · MUSIC CRAFT (needs L4 + L2/matcher.ts; parallel with L5). src/cards/m2-words-per-line/ · m2-band-ladder/ · m2-emo-census/ · m2-emo-count/ · m2-emo-missing-lexicon/ · m2-style-board/ · m2-trend-setters-locked/ · m2-render-features/ · m2-broken-rung/ · tests/multiplicity.spec.ts (the four section-13 assertions).
- W3-L7 · REACH + HEADWINDS (needs L4 + L2/window-ladder.ts; parallel with L5, L6). src/cards/m3-window-ladder/ · m3-platform-roster/ · m3-two-youtubes/ · m3-who-is-listening/ · m3-geography/ · m3-discovery-path/ · m3-slope-locked/ · m4-signal-table/ · m4-severity-map/ · m4-register/ · m4-refused-followers/.
- W3-L8 · ART + ANIMATION RAILS (needs L4; parallel with L5-L7). src/cards/art-nightcafe-headline/ · art-audience-by-surface/ · art-measurement-depth/ · art-production-volume/ · art-renderer-mix/ · art-community-untended/ · art-work-order/ · anim-dormancy/ · anim-tiktok-gap/ · anim-prompt-inventory/ · anim-contentid-trap/ · plus the ghosted Music-frame empty states for the Art Response/Craft tabs.
- W3-L9 · PROVENANCE (needs L1; parallel with L5-L8). src/cards/m5-feed-ledger/ · m5-schema-guard/ · m5-drift/ · m5-catalogue-diff/ · m5-reproduce/ · m5-glossary/ (glossary is generated from the GuidanceNote registry, never hand-listed).
- W4 · COORDINATOR, SERIAL, AFTER LANES LAND. server.mjs (static + snapshot route + /healthz; zero runtime deps; 405 on any non-GET) · static-path.mjs + .d.mts · scripts/check-build.mjs (no snapshot bytes in dist/; forbidden-string sweep) · scripts/card-registry-check.mjs (every card: evidence, question, drill-or-reason) · tests/honesty.spec.ts · tests/no-fabrication.spec.ts · tests/forbidden-strings.spec.ts · tests/drill-registry.spec.ts · tests/server.spec.ts · tests/palette.spec.ts · README.md · DEPLOY.md (three-posture table) · .github/workflows/lofn-analytics-ci.yml.
- W5 · DOCTRINE LANDING, SERIAL. Add the APPS / SNAPSHOTS / APP_LOCK rows to the CANONICAL PATHS table in BOTH .claude/skills/lofn/SKILL.md and .agents/skills/lofn/SKILL.md · write .claude/skills/lofn-analytics/SKILL.md and hand-write its .agents/ twin in Codex voice (never sed) · add the roster rows to both skills/README.md files and backfill the two already missing (lofn-render-audit, lofn-step11-packager) · insert the identical bullet at CLAUDE.md:20->21 and AGENTS.md:20->21 · `PYTHONIOENCODING=utf-8 python scripts/check_skill_mirror.py` must print IN SYNC · manually diff CLAUDE.md against AGENTS.md (no script does this) · log to memory/YYYY-MM-DD.md · `python3 scripts/run_lock.py release apps/lofn-analytics`.


## Open questions for The Scientist — v2:eb88413f8b4495226663676e3ddd62f85a7f49882afeb14bd5461c4ffc0188be
- May this app ever be hosted at analytics.lofn.ai, and may `data/snapshots/` be committed? The snapshots embed your Suno user_id, follower counts, and the complete style-prompt corpus of all 210 clips — including unreleased language. `.gitignore` currently has no `data/` entry across all 82 lines, so the first `git add` publishes the lot. RECOMMENDED DEFAULT: local-only build, `data/snapshots/` added to `.gitignore` before the first commit, no hosted posture and no Dockerfile deploy until you say otherwise. The app is fully useful at `localhost` and the DNS can wait.
- Which estimator is THE baseline? On the identical n=46 modern cohort the same rows give median 4.37%, mean 4.29%, and pooled (sum upvotes / sum plays) 1.57%; the 4.49% figure in the brief reproduces none of the three exactly, so something upstream is using a fourth definition. This matters because it is the denominator of every effect in both apps and the anchor of the Soundry's lift. RECOMMENDED DEFAULT: median of per-clip like rates on the era cohort, printed with the estimator name inline on every card, with mean and pooled available as a shelf control — the pooled figure is dominated by Triple Arch and should never be the default.


# ===== CRITIC 1 — verdict: NEEDS-PATCH =====


## gaps
- COLLABORATOR ACCESS - DROPPED IN BOTH APPS, AND STRUCTURALLY FORECLOSED. Her App-2 line is "Access only to The Scientist + invited collaborators", and her App-1 tab goal ends "...analytics for Lofn (private-claw) and collaborators." No part of the spec defines a user model, an invite, an auth check, or a share flow. Worse, Part 3 section 0.1 pins server.mjs to 127.0.0.1 only, and Part 3 section 3.3 says panel access is "limited to the Scientist and invited collaborators, all traffic loopback" - loopback means a collaborator must be sitting at her keyboard. The session token in #/panel/<session-token> is in the hash fragment, which is never transmitted to a server, so it cannot gate anything. Not named as an overrule anywhere. This is the one whiteboard requirement that is both dropped and contradicted.
- "COMBINE ALL 4 SOURCES INTO USEFUL INSIGHTS" - NO FUSION SURFACE EXISTS. Verified on disk: only data/snapshots/suno/, data/snapshots/socials/ and data/snapshots/distribution/ have files; data/snapshots/competitors/ and data/snapshots/market/ do not exist, data/snapshots/soundry/ is an empty directory. Honest empty states are correct - but the spec designs no card at all for feeds 2, 3 and 4. Part 2 section 1 gives them the single word "Nothing" plus verbatim empty text. There is no plot spec, no card list, no "what appears on the second pull" wireframe for Competitor Scans or Market Trends, and no analytics card that consumes data/snapshots/soundry/latest.json once the Soundry emits it (Part 3 section 3 specifies the emitter; Part 2 specifies no reader). Contrast M2.5(a) and M3.7, which DO ship locked ghost wireframes that auto-enable. Three of her four named feeds get less than that, and no card anywhere joins two feeds into one insight - which is the literal ask.
- COMPETITOR FEED HAS NO PATH OUT OF ITS EMPTY STATE. Her feed #2 is "5 watched growing/trending targets". Part 2 section 1 correctly reports the five are unnamed anywhere on disk, then section 14.4 removes the refresh button entirely. But scripts/ingest_suno.py:264-265 already supports --feed competitors --handle <h> (it writes data/snapshots/competitors/<handle>/), so the pull exists - what is missing is any UI to record the five handles. The blocker list can therefore never be cleared from inside the app, and the spec does not name a file where the handles would live. A named on-disk roster (e.g. data/watchlist.yaml) plus a Provenance-tab editor is the minimum patch.
- "TOP TRENDS & INSIGHTS" - HER CENTRE PANEL IS SILENTLY RELOCATED. She drew a centre panel of ranked directional statements ("up: Baile Phonk", "down: Anthems"). Part 2 section 4 maps M0 SIGNAL to it, but M0's six cards contain no ranked up/down list at all (snapshot dial, response index, era chart, placement bar, standing-findings dumbbell, locked movers). The sorted tailwind/headwind list she asked for actually exists - as M4.1, on the HEADWINDS tab. Refusing per-style rankings IS explicitly overruled and well-argued at M2.4 and section 12, but moving the surviving up/down list off her centre panel onto a different tab is not named as an overrule anywhere.
- THE RIGHT RAIL IS ONE SENTENCE. Promoting Headwinds to a tab is explicitly overruled with a reason (section 8: "a warning with no comparison class is unreadable") and intent is preserved - good. But the compensating promise, "The three highest-severity items also mirror into a persistent right rail on every other tab", gets no card spec, no width, no position in the grid, no interaction, no empty state, and no mention in the section 15 file layout (src/shell/ lists rail.ts, filters.ts, chrome.ts, directive.ts - nothing for the headwind rail). The only literal element she drew on the right side of App 1 is specified in twenty words.
- "TOOLTIPS, ALT-TEXT, AND INFORMATIONAL GUIDANCE FOR ALL ELEMENTS" - NARROWED TO CHARTS WITHOUT SAYING SO. Part 1 section 5 is genuinely strong for plots (generated four-sentence alt, mandatory data table, one keyboard model, three doors). But section 5.1 rule 1 says an affordance that cannot fill `computed` and `basis` should be deleted - which converts "all elements" into "all elements that happen to be statistics". No GuidanceNote is specified for the non-chart controls she will touch most: filter chips, the estimator selector, sort buttons, rail items, the state chips, the clip_id field, the peek counter, the copy buttons. There is no registry of required notes and no test asserting coverage - the tests/ list in section 8 has alt-text.spec.ts (generated alt only) and nothing like guidance-coverage.spec.ts. Compare card-registry-check.mjs, which does enforce evidence/question/drill per card; guidance gets no equivalent gate.
- DRILL-DOWN IS SPECIFIED FOR ANALYTICS ONLY. Her requirement ("Every insight/plot should link to deeper results if they exist") is well handled in Part 1 V15 and Part 2 section 14.2 with an explicit, well-argued overrule ("every insight links, or states in words why it cannot"). Part 3 then declares roughly a dozen plots (A2.1, A3.1, A3.2, B2, B3, B4, B5, B7, B9.1, C1-C5) and gives deeper/deeperAbsentReason for none of them; the only drill affordance in the Soundry is [more details ->] on C2. Because deeper is optional in the type (deeper?: ... | null), an omitted field slips past V15 rather than failing loudly - so the Soundry ships the exact dead-link-by-omission the overrule was written to prevent.
- NO OWNER FOR THE SECOND SNAPSHOT. "Show trends as well as now" is the requirement most of Part 2 is currently locked against: M0.6, M2.5(a), M3.7, every spark, and the Soundry's B3 accumulation curve all unlock at snapshot 2 or 14. The spec's answer everywhere is copy telling her to run `python3 scripts/ingest_suno.py` daily. Nothing schedules it - there is no cron entry, no scheduled task, and no mention of one (.claude/ contains only launch.json). The single dependency the whole trend layer rests on is left as a habit.
- SOUNDRY HAS NO FILTER SHELF. Part 1 section 3.2 declares that "Soundry adds exp / hyp / arm / blind" to the shared global key table, and section 3.4 promises the chip strip is "the only control surface" with dimmed-not-dropped applicability. Part 3 never renders a chip strip, never mentions a filter shelf, dimmed chips, relaxations(), or the session trail, and never says which of its three tabs honour exp/hyp/arm/blind. Her "filters carry between tabs" requirement is therefore satisfied in one app of two.
- THE ANVIL'S PROMPT-WRITING VOICE IS UNSPECIFIED. Her card contract is five fields - Title, Style Prompt, Exclude Prompt, Lyrics, Caption - and all five are present in Part 3 section 1.6 as containers. What is absent is any statement of house style for their contents: no reference to the skills/lofn-music step contracts, no EMO-tag convention, no Suno v5.5 two-field rule, no lyric length target beyond the vault/gates.yaml band check. The cards are specified as slots, not as Lofn's prompts.
- "[MORE DETAILS]" IS KEPT, BUT THE HYPOTHESIS AXIS LABELS ARE SILENTLY UNIT-SWITCHED. Collapsing her two per-hypothesis axes onto one shared symmetric axis IS explicitly overruled with a good reason (Part 3 section 4.2). The un-named part: her axes were labelled in percentage points (0/+5/+10/+15/+20; -10/-5/0/+5/+10) and the replacement is a lift ratio on a log axis. Part 1 format.ts even forbids printing a bare percent for a ratio. Only the C1 hero numeral still speaks percent. That is probably right, but it changes the unit she wrote on every hypothesis row and is not called out.


## fabricationRisks
- THE HOUSE BASELINE IN PARTS 1 AND 3 DOES NOT REPRODUCE FROM THE SNAPSHOT. Part 1's "house arithmetic that every number in both apps inherits" states modern-cohort like rate 4.49% (n=46) and median song 258 plays; Part 3 section 1.3 freezes p-bar = 0.0449 and m = 258 into the power model. Recomputed from data/snapshots/suno/latest.json with the spec's own cohort (created_at >= 2026-02-01, play_count >= 60): n = 46 confirmed, median like rate = 4.3695%, median plays = 267. No floor/window combination tried (floors 0/30/50/60/100 x starts 2026-01/02/03) yields 4.49% or 258. Part 2 uses the correct 4.37%/267 throughout and even flags the 4.51%/n=48 prose drift at M0.5 - so the estate's shared constants are the stale prose figure that Part 2's own DRIFT panel exists to catch, imported without the DRIFT badge that Part 1 section 4.3 rule 11 makes mandatory. Every +/-2.53pp restatement and both frozen power points inherit it.
- "MATURE" PLAYS IS UNDEFINED, AND THE FROZEN TEST POINT ONLY PASSES AT INFINITY. Part 3 section 1.3 requires a test asserting n=114 @ m=258 -> 1.150 and n=48 @ "mature" -> 1.150 to 3 dp. The first checks out exactly (sigma2_tot = 1.204e-4 + 1.662e-4; SE_log = 0.049939; exp(2.802*SE) = 1.1502). The second only reaches 1.150 when the binomial term is ~zero - solving backwards, "mature" must be about 159,000 plays per song. On disk the median is 267 plays and exactly one clip of 210 exceeds 1,000 (Triple Arch Over Me, 349,736). So the third colour level on A2.1 and C3, and the "8 days" timeline it produces, is an unreachable asymptote presented as an assumption the user can select. Define mature numerically or delete the level.
- THE FEED RING WOULD OVERSTATE COVERAGE, AND THE TWO PARTS DISAGREE ON THE DENOMINATOR. Part 2 section 3.2 draws "8 segments, one per KNOWN_FEEDS id ... Today: 3 of 8 filled"; Part 1 section 7.3's Day Mark caption reads "3/9 feeds". scripts/ingest_receiver.mjs:43-52 fixes exactly eight ids: suno, youtube, tiktok, spotify, instagram, competitors, market, soundry. socials and distribution are NOT among them - yet those are two of the three directories that actually have files. Counted against KNOWN_FEEDS the honest reading today is 1 of 8. Either the ring is not keyed on KNOWN_FEEDS (then say what it is keyed on), or the receiver's set needs socials/distribution added; as written the app's most prominent honesty instrument prints a number it cannot derive.
- THE SOUNDRY'S SEED CHIPS ARE PROSE DRESSED AS COMPUTATION. Part 3 section 1.1 says the chips are "derived live from Lofn Analytics headwinds, each carrying the source card's URL", then hardcodes three strings. One reads "Median words-per-line 6.3, r = -0.30 and weak." Recomputed on the same modern cohort: median words_per_line = 5.83, Spearman rho = -0.329 - and Part 2 M2.1 itself prints rho = -0.33. So the chip contradicts both the disk and its sibling app while claiming provenance from it. Either generate the chips from the same code path as M2.1/M4.1 or label them as authored examples.
- 90% INTERVALS APPEAR IN AT LEAST SIX PLACES DESPITE A FROZEN 95% CONSTANT. Part 1 section 2.4 declares CI_LEVEL = 0.95 a single exported constant mirrored by --ci-level, states "Every interval in both apps is 95% ... never re-litigated per card", and permits only a 50% inner band in the Soundry. Part 2 then specifies 90% bootstrap intervals at M0.3, M1.5, M2.2 and M2.4, and Part 3's DECIBAL card prints "90% CI x1.03 to x1.23". The Insight schema types the field as the literal level: 0.95, so a 90% interval cannot even be represented. Whichever way this is resolved, today the spec would ship cards whose stated confidence level is not the level the kit computed.
- POSITIVE VERIFICATION, FOR CONTRAST: every empirical claim I could check in Part 2 reproduces exactly from data/snapshots/suno/latest.json - the 14 monthly medians (2025-06 1.33 n=36 through 2026-07 3.56 n=8, including the 2025-09 0.00 across 36), 863 distinct EMO tags across 202 clips with triumph n=34, model versions 61/67/42/40, all 20 style-term cells (piano n=12 5.43 ... distortion n=1 6.25), "45 of 46 modern clips match two or more", Triple Arch 349,736 plays / 1.44% / 93.48% share, 209 of 210 under 1,000 plays, and rho = -0.329 / +0.237 / +0.125 / +0.101. The BOM on output/render-audits/2026-07-24-eight-song-deep-dive/measurements.json is real, vault/gates.yaml:155 is mean_words_per_line_ceiling: 7.5, and the ingest_suno.py claims (writes latest.json on failed reconciliation at lines 260-273; both sort params required or HTTP 422 at line 43-44; page=0 aliases page=1 at lines 150-153) all hold. The fabrication risk is concentrated in Parts 1 and 3, not Part 2.


## buildBlockers
- TWO INCOMPATIBLE FILTER VOCABULARIES FOR THE SAME SHARED SERIALIZER - this breaks the whiteboard's carry-over requirement outright. Part 1 sections 3.2/3.3 define the frozen FilterKey union as mod|snap|co|basis|out|floor|nmin|prov|sel|q|exp|hyp|arm|blind with defaults mod=music, co=modern. Part 2 section 3.3's filter shelf uses m (modality), win (cohort, default all), pf (play floor), est (estimator), out, prov. Four of six keys are renamed, the cohort default is inverted, est does not exist in the union at all (so applyFilters cannot honour the estimator control Part 2 calls "first-class"), and basis/nmin/sel/q/snap have no control in Analytics. Part 1 section 3.4's promise - "the serializer is shared, so a headwind link in Lofn Analytics opens a Soundry Sonic Anvil seed with its filter intact" - cannot hold. Pick one table and delete the other.
- THE SONIC ANVIL HAS NO GENERATOR. This is her core App-2 flow: "What should we try today?" -> 12 cards -> each card = full Suno prompts (Title, Style Prompt, Exclude Prompt, Lyrics, Caption). Part 3 specifies the ask box (1.1), factor parsing (1.2), power (1.3), nuisance balancing (1.4), the freeze gate (1.5), the card chrome (1.6) and the export (1.7) - and never once says what writes the five fields. POST /soundry/rounds/:id/generate is named only by its 409 behaviour. Meanwhile section 0.1 forbids calling "any generation or render API", says the app "Reads no API key", section 6 forbids a dependencies key, and section 0.1's CSP pins connect-src 'self'. So twelve full lyric sheets must be produced by a dependency-free offline TypeScript server, by unspecified means. Either name the mechanism (hand-off packet to the lofn skill? deterministic templating over src/sampler.data.json? the Scientist writes them and the bench only assigns arms?) or the build agent invents the most important behaviour in the app.
- THE LISTENER PANEL HAS NO WAY TO PLAY AUDIO. Her requirement is "get feedback from listeners." Part 3 section 3.3 specifies the blind panel at #/panel/<session-token> - hidden titles, randomised order, ratings, free text - and never specifies an audio source. The Soundry renders nothing and holds no audio files; the only handle on a song is clip_id. Streaming it would mean an external origin, which section 0.1's CSP (connect-src 'self') forbids and scripts/check-local-only.mjs is explicitly written to fail the build over ("fails the build if any .html/.js contains an absolute http(s):// origin"). As specified the panel instrument cannot be listened to, which also means observations/panel.jsonl, B7, and the instrument: panel branch of the pre-registration can never be populated.
- THE KIT LIVES AT TWO ADDRESSES AND THE :ROOT BLOCK IS DECLARED TWICE. Part 1 builds the foundation at apps/lofn-kit/ and opens with "If either app ever ships a second implementation of any of these, that is the defect." Part 2's scope note says the foundation "is assumed to exist in apps/_shared/", and its section 15 file tree imports apps/_shared/plot. Part 1 section 1.1 heads the token file "THE ONLY :root DECLARATION IN EITHER APP"; Part 2 section 15 labels apps/lofn-analytics/src/style.css "THE ONLY :root token block". Part 1 section 8.1 wires the alias as @kit -> ../lofn-kit/src, which resolves to neither. Part 2 also re-homes comb.ts and daymark.ts into src/plot/ when Part 1 places them in apps/lofn-kit/src/signature/ - the duplicate implementation the kit's own invariant forbids.
- M3.2 FAILS THE KIT'S OWN VALIDATOR, AND THE INFERENCE PLOTS BREAK THE FROZEN TOKEN BINDINGS. Part 2 M3.2 colours the platform roster by provenance with five levels (api-public M1 / authenticated-page M3 / public-page graphite / search-snippet ink-faint+dash / never-pulled hatch); Part 1 V8 hard-errors on color.scale.domain.length > 4, so the card renders as plot-invalid. Separately, Part 1 section 1.1 states the inference triad --sig-pos/--sig-neg/--sig-null are "the only colours legal inside a stats plot" and section 1.3 freezes --m4 as "attention ... Never bad" - yet M2.4 colours CI position sea-teal/ember and Part 3 section 2.4 B4 does the same ("entirely above 1.0 teal, entirely below ember"), which is both an illegal palette inside a stats plot and the exact good/bad reading --m4 is frozen against. There are also three mutually inconsistent provenance vocabularies: Part 1 section 2.1 (six values), Part 1 section 1.2 ("exactly four levels", differently named), Part 1 section 3.2 prov (three), Part 2 section 3.3 (four).
- THE SPEC DEFERS DECISIONS TO AN OPEN-QUESTIONS LIST THAT IS NOT IN THE SPEC. "See open question 1" is cited three times (Part 1 section 8.3, Part 2 section 0, Part 2 section 15 - the hosting/gitignore decision) and "(See open question 2.)" once (Part 2 section 3.3 - the estimator). Neither question, nor any list containing them, appears in the document. Two of the three deployment-shaped decisions in the build are parked in a file the builder does not have.
- THE GITIGNORE PRECONDITION CONTRADICTS AN EXPLICIT WRITTEN DECISION IN THE REPO. Part 1 section 8.3.1 and Part 2 section 15 make it blocking to add data/snapshots/ to .gitignore "before anything commits", and Part 2 says it is "absent from all 82 current lines". .gitignore is 93 lines, and lines 78-82 already say, deliberately: "Snapshots and queues ARE tracked: they are the evidence behind every number the apps display, and their history is the trend." Both positions are defensible - the tree does carry the Suno user_id (header.user_id in data/snapshots/suno/latest.json) and 210 style prompts - but a build agent handed this spec would silently reverse a documented decision, and would un-track the very history the trend layer depends on. This needs The Scientist, not a lane.
- THE RUN-LOCK COMMAND CONTRADICTS ITS OWN SIDECAR RULE. Part 1 section 8.3.2 requires the lock in "a sidecar apps/<slug>/.lofn/.RUN_LOCK, not in the build tree, so Vite/eslint/git never glob it", then gives the command `python3 scripts/run_lock.py acquire apps/lofn-kit --run-slug <slug>` (Part 2 section 15 repeats it for apps/lofn-analytics). scripts/run_lock.py lines 54 and 142-145 write <run_dir>/.RUN_LOCK - so that command puts the lock exactly where the sentence forbids. The commands must read `acquire apps/lofn-kit/.lofn` and `acquire apps/lofn-analytics/.lofn`.
- PROMOTE/BENCH IS A PER-SONG VERDICT, WHICH B6 FORBIDS. Part 3 section 2.6 refuses a daily winner because "58% of the spread you can see is binomial" and declares "Winners here are arms". Part 3 section 2.8 then makes the two exits per-variant - promotions/<variant-id>_phase0.md, plus "an undecided: 4 counter persists until every variant has one" - i.e. the app requires exactly the twelve per-song judgements the refusal card exists to prevent. Her intent ("take winners + make full songs") does need an exit; specify whether promotion is arm-level with a representative variant attached, or add the guard that a variant may only be promoted once its arm's contrast has concluded.
- "THE ONLY STACK IN THE APP" IS ASSERTED FOUR TIMES OVER. Part 2 M3.6 calls itself "the only stack in the app, gated by the grammar's sum-to-1.00 +/- 0.01 rule", but M0.4 (2-segment 100% bar), A0.3 (100% stacked measurement-depth bar) and N0.3 (stacked prompt-shape inventory) are also stacks. Minor alone, but validate() V10 and any test written from that sentence will disagree with three shipped cards.


# ===== CRITIC 2 — verdict: NEEDS-PATCH =====


## gaps
- FEED IDs: `socials` and `distribution` — the two feeds carrying most of Part 2's content — are NOT in KNOWN_FEEDS. `scripts/ingest_receiver.mjs:43-52` fixes exactly {suno, youtube, tiktok, spotify, instagram, competitors, market, soundry}. Part 2 §1 lists `socials`/`distribution` as feed ids and cites those very lines as authority. Consequence: Part 2 §3.2's Day Mark 'inner arc — 8 segments, one per `KNOWN_FEEDS` id ... Today: 3 of 8 filled' is arithmetically impossible — only ONE of the 8 (`suno`) has a snapshot at `data/snapshots/suno/latest.json`. Part 1 §7.3 independently says the caption reads `3/9 feeds`. Three different denominators (9, 8, and the disk truth of 8-with-1-filled).
- NO INGEST SCRIPT for socials/distribution/nightcafe. Part 2 §0: 'the single exception is one explicit **Refresh** control per feed that shells the feed's ingest script.' `ls scripts/` shows no social/distribution/nightcafe/competitor/market ingest script — only `scripts/ingest_suno.py` and the generic `scripts/ingest_receiver.mjs`. `data/snapshots/socials/latest.json` self-describes as 'Manual read of PUBLIC profile pages ... Not an API pull'; `data/snapshots/distribution/latest.json` as 'DistroKid Stats ... authenticated browser session'. Both were hand-read by a human in a browser. There is no script to shell, and the spec never says what the Refresh button on the Art rail / Reach tab actually invokes.
- EMPTY-DIRECTORY STATE UNCOVERED: `data/snapshots/soundry/` exists on disk and is empty. Part 2 §1 asserts soundry is 'ABSENT ... the only on-disk hit for the string is the `KNOWN_FEEDS` entry' — false, the directory is also on disk. None of the six §6.1 `EmptyKind`s covers 'directory exists, no snapshot in it': `never-pulled` is defined as 'no snapshot has ever existed' and keys on a path that now resolves.
- N0.3 READS FIELDS THAT DO NOT EXIST AT THE NAMED PATH. Spec: '*(local `output/past_video/*.json`, 210 records)* ... one row per dimension (`selected_personality`, `selected_panel`, `model`)'. Top-level keys in all 210 files are: timestamp, concept, medium, prompts, input_text, competition, model, input_settings, panel_prompt, personality_prompt, meta_prompt. `selected_personality`/`selected_panel` are nested inside `input_settings`. The counts are correct only via that nested path (input_settings.selected_personality: LOFN-Prime 153, LLM Generated 48, SATISFYDRIP 3 — verified). A builder following the spec literally gets undefined x210 and, per the honesty gate, renders an empty state that misrepresents the corpus. Also `input_settings.selected_panel` is 'LLM Generated' for 193 of 210, so 'the one corpus where personality **and panel** genuinely vary' is half false.
- `input-cleaning.mjs` / `cleanFeedbackText()` DOES NOT EXIST IN THIS REPO. Part 3's scope note claims it 'is specified elsewhere and is imported, not restated', but Part 1 — the shared foundation — never mentions it and §8's file tree has no such file. The only implementation is in a DIFFERENT repository: `E:/git/saga/game-dev/apps/forge-arcade/input-cleaning.mjs`. Part 3 §2.7 (B7) and §3.3 both gate listener notes on `cleanFeedbackText().safeForAnalysis`.
- NO APPS EXIST. `apps/` contains only `nail-art-studio`. `apps/lofn-kit`, `apps/_shared`, `apps/lofn-analytics`, `apps/soundry`, `data/soundry/`, `output/soundry/`, `data/daymarks/` are all absent — correctly implied by the spec, but §8.3.4 only asks for `APPS`, `APP_KIT`, `SNAPSHOTS`, `APP_LOCK` to be added to CANONICAL PATHS (`.claude/skills/lofn/SKILL.md:151-178`). `data/daymarks/`, `data/soundry/`, `output/soundry/` are also undeclared, and under `EXECUTION.md §4` an undeclared path is a repair blocker.
- NULL HANDLING IS NEVER SPECIFIED. In `data/snapshots/suno/latest.json`: `words_per_line` and `repeated_line_ratio` are null on 4 of 210 rows; `persona` is null on 88 of 210. M1.6 tables all three, M2.1 scatters words_per_line, M2.2 bands repeated_line_ratio, L3 (§14.3) shows persona. Part 1 §6.1's six EmptyKinds are all PANEL-level; there is no cell-level null contract. §4.3 rule 7 ('Never `0`, never `0.0%`') is exactly the failure a missing null rule produces.
- M1.5 (Model version) CANNOT DRAW ITS OWN CELLS UNDER THE TAB'S DEFAULT FILTERS. Spec prints 'v4.5 n=61 · v4.5+ n=67 · v5 n=42 · v5.5 n=40' — verified, but that is the ALL-210 split. On the tab's declared defaults (Part 2 §3.3: play floor 60, outlier exclude, modern cohort) the split is v5 n=7, v5.5 n=39, and v4.5/v4.5+ are n=0. Either the card silently defeats the filter shelf or two of four rows are empty; the spec does not say which.
- M2.4 CELL SIZES CANNOT CARRY A BOOTSTRAP CI. The card draws a '90% bootstrap CI whisker' for 20 style terms with cells of n=1 (distortion), n=4 (somatic), n=5 (strings, baile). A bootstrap median CI at n=1 is a point and at n=4-5 is not informative. The spec sets no minimum n to DRAW a whisker — Part 1 §3.2's `nmin` default of 10 is explicitly a *ranking* floor, and Part 2 §3.3's filter shelf does not carry `nmin` at all.
- M0.3 DRAWS MEDIAN RULES AT n=5. 'Months with **n < 5** render dots only, no median rule' permits a bootstrap median rule + 90% ribbon at n=5 (2025-10) and n=6 (2025-12, 2026-06). The n<5 rule correctly kills 2026-01 (n=2) and 2026-02 (n=3), but n=5 is not a materially better basis and the spec offers no argument for the threshold.
- M2.2 RUNS THREE 4-BAND LADDERS ON 46 CLIPS (~11/cell) WITH POST-HOC EDGES. The banner honestly says so ('cannot be confirmed by the data that generated them ... Status: HYPOTHESIS'), but the card still renders a bootstrap whisker per band and a leading-band dot filled `M4`, which is the visual grammar of a result.
- EVERY DURATION FIGURE IN THE SOUNDRY RESTS ON AN ADMITTED ASSUMPTION. B3's own verbatim empty state: 'the readout age of 14 days is an assumption, not an observation, **and so is every duration quoted in this app**.' Yet A2 ('detects a +15.0% lift in **19 days**'), C1 ('19 days'), B6 ('day 6 of 19'), and §5.1 ('decision on 2026-08-23') all render those durations as tabular figures with no caveat attached. Zero maturity curve exists on disk — one Suno snapshot.
- `FeedId` IS USED BUT NEVER DEFINED. Part 1 §2.1 (`EmptyState.feed?: FeedId`), §5.1 (`basis.feed: FeedId`), Part 2 `src/contract.ts`. Given the KNOWN_FEEDS/snapshot-directory mismatch above, its membership is genuinely undecidable from disk.
- THE HONEST CASES ARE HANDLED WELL — no objection. `data/snapshots/competitors/` and `data/snapshots/market/` are genuinely absent; the five watched competitor handles are nowhere in the repo; `vault/EMO_LEXICON.yaml` does not exist; Spotify-for-Artists is confirmed unconnected by `data/snapshots/distribution/latest.json` notes ('Spotify for Artists is NOT connected to this DistroKid account'). Instagram's `status: not-read` / 'SKIPPED BY INSTRUCTION' row is real. All of these get honest empty states with named blockers, and §14.4's 'no refresh button at all' for competitors/market/soundry is the right call.
- VERIFIED-CORRECT DATA CLAIMS (for the record, so the patch does not churn them): 210 rows / 374,122 plays / 5,913 upvotes / 3,926 followers / reconciliation.ok true / 11 pages; Triple Arch 349,736 plays = 93.48% at 1.443% like rate; 209 of 210 under 1,000 plays; all-210 median 50 plays; all-time >=50-plays median 2.4585% at n=112; 863 distinct EMO tags across 202 clips with triumph 34 / defiance 27 / anticipation 24 / revelation 21 / dread 21; all 20 M2.4 style-term n and median pairs; 45 of 46 modern clips matching >=2 terms; every M2.1/M2.2/M2.3 rho to 2dp; the entire M3.1 Window Ladder arithmetic on all four platforms; the YouTube age/gender/country tables; NightCafe 21,000/1,200/23,000/306-credits/28-3-1 badges; 3.55x; measurements.json carries a BOM and has 8 entries; numpy 2.4.6 / soundfile 0.14.0; `vault/gates.yaml:155 mean_words_per_line_ceiling: 7.5`; `house_lexicon` contains 'triple arch'; the 2026-07-27 drift pairs (4.51%/n=48, 2.35%, 373,487, 220-vs-224, ambient n=8 at 5.55%); the ingest_suno 422 and page-0 gotchas (`scripts/ingest_suno.py:43,150-152`); AWE/INDIGNATION 18 of 132 songs = 13.6%.


## fabricationRisks
- THE HOUSE BASELINE IS WRONG AND THE TWO APPS DISAGREE. Part 1: 'House arithmetic that every number in both apps inherits (**from the live pull, 2026-08-01**) ... modern-cohort baseline like rate **4.49% (n=46)** ... median song **258 plays**'. It is not from the live pull and it is not a median: `scripts/analyze_power.py:77` computes `p = st.mean(lrs)` on the >=50-plays / 2026-02+ / outlier-EXCLUDED cohort. Recomputed from `data/snapshots/suno/latest.json`: that cohort is n=46, median 4.4425%, MEAN 4.4890%, median plays 258. Part 2 §3.3/M0.2 then prints median **4.37%** / mean 4.29% / pooled 1.57% for 'the identical n=46 modern cohort' — a DIFFERENT cohort (floor 60, outlier INCLUDED; verified: median 4.3695%, mean 4.2875%, pooled 1.5683%, median plays 267). Two apps, one label, two baselines. The 4.49%/258 pair is prose-archive (`vault/EXPERIMENT_STANDARD.md:26,28`; `memory/2026-08-01.md:36,40`), so Part 1's own V17 ('provenance === "prose-archive" => a `driftPair` note exists') is violated by the spec's own header block.
- PART 3'S FROZEN POWER CONSTANTS ARE NOT THE MEASURED ONES, AND THE TEST THAT 'VERIFIES' THEM IS SELF-CERTIFYING. §1.3 freezes `sigma2_true = 1.204e-4` and claims 'These are not invented: at m=258 the binomial term is 1.662e-4, which is 58% of sigma2_tot ... both measured facts, reproduced ... a test asserts those two points to 3 dp.' `scripts/analyze_power.py:84-85` computes the binomial term as the MEAN of per-song p(1-p)/n_i, not p-bar(1-p-bar)/258. Running it: observed sd 1.80pp (var 3.24e-4), binomial sd 1.37pp (var 1.877e-4), TRUE between-song sd 1.17pp (var **1.369e-4**). The spec substituted the median-song binomial term and back-solved sigma2_true; both decompositions happen to land on 58%, which hides the swap. With the on-disk 1.369e-4, MDE at n=114/m=258 is **x1.161**, not x1.150. `glm.spec.ts` asserting 1.150 to 3dp would freeze constants that contradict the script `vault/EXPERIMENT_STANDARD.md:15` says to re-run ('Re-run it; do not trust it from memory').
- M0.3'S MONTHLY MEDIANS IGNORE BOTH OF THE APP'S OWN DEFAULT FILTERS. The typed table ('2025-06 1.33 (n=36) ... **09 0.00 (36)** ... 05 5.10 (16) ... 07 3.56 (8)') reproduces EXACTLY and ONLY against all 210 clips with no play floor and the outlier included. Part 2 §3.3 sets play floor default 60 and outlier default exclude; `scripts/analyze_power.py` on the stated cohort gives 2025-09 = **0.29% at n=15**, 2025-07 = 1.54% at n=13, 2026-04 = 4.58% at n=8. The card's own headline — 'The 2025-09 cohort's median is **0.00% across 36 clips** — a fact no headline number will ever surface' — is true only with the filters off, and the spec says these are 'drawn live, not typed'.
- M0.3 SHIPS THE 'TRIPLING' HEADLINE OVER A BRACKET THAT CONTAINS A DECLINE. 'Annotation bracket over **2026-02 -> 2026-07**: engagement roughly tripled here'. Disk: 2026-02 = 3.76%, 2026-07 = 3.56%. The tripling is 2025-06 (1.33%) -> 2026-05 (5.10%). `vault/EXPERIMENT_STANDARD.md:80-83` says explicitly: 'Any dashboard inheriting the "engagement roughly tripled" headline without the June-July fall is showing a stale story.'
- M2.4 PLOTS ABSOLUTE MEDIANS ON AN AXIS DEFINED AS A DIFFERENCE. Spec: '**x** like-rate difference from the **era-matched** baseline, in **pp**, symmetric axis with a heavy zero rule'. The 20 listed values (piano 5.43 ... baile 2.56) are absolute per-term medians — verified exactly against the floor-60/outlier-INCLUDED cohort, not differences and not era-matched. 'Era-matched' is never defined (matched on calendar month? on the modern block? weighted how?) and no era-matching estimator exists in Part 1 §2.4's `stats.ts` (wilson / bootstrapMedian / fisherZ / logLinkGLM / typeM / typeS / mde80 only). The cohort also includes Triple Arch while the tab's outlier default is exclude.
- THE COMB CAPTION IS SPECIFIED TWICE WITH TWO DIFFERENT NUMBERS. Part 1 §7.1: `210 clips · 1 off-scale · median 258 plays`. Part 2 §3.4: `210 clips · 1 off-scale · median 50 plays`. Disk: median of all 210 = 50. Part 1 prints the modern-cohort median under an all-210 mark, on a device both parts call live-drawn.
- M2.7'S JOIN WEIGHTS DO NOT REPRODUCE AND USE TWO DENOMINATORS FOR ONE CORPUS. '`output/songs/*.md —2/105→ suno`': `output/songs/` holds **132** .md files (Part 1 §1.5 independently and correctly says 132), and normalised exact title match against the snapshot is **0/132** (4 loose). '`mp3 —60/61→ suno (title match)`' then '`mp3 —8/66→ render audit`' in the same bullet list: `output/mp3/` holds **66** files; exact normalised match is **60/66**. There is no 105 and no 61 anywhere on disk.
- THE '40.9% STRUCTURED MATCH' FIGURE IS NUMERICALLY IDENTICAL TO AN UNRELATED SPOTIFY STATISTIC. Part 2 §14.3 '[JOIN: output/past_music, output/songs — 40.9% structured match]' and Part 3 §0.2 'Structured title matching tops out at 40.9%.' `data/snapshots/distribution/latest.json` Spotify `discovery_sources.collection = 40.91`, which M3.6 also prints. Measured title-match rates: `output/songs` 0/132 exact, `output/past_music` 0/468 exact, `output/mp3` 60/66. Nothing on disk yields 40.9%. Part 3 §0.2 makes this figure the entire justification for the clip_id binding field.
- A0.5 RENDERER COUNTS DO NOT REPRODUCE AND THE PARSER IS UNSPECIFIED. Spec: 'Poe-GPT-5-mini 1,298 · gpt-5-mini 1,170 · gemini-3-flash-preview 242'. Counting those substrings across the 4,690 files in `output/past_art/**`: **1,352 / 1,258 / 447** — gemini is off by 85%. A0.3's 'the filename grammar' (which drives the '65.9% parse / 34% do not' split, itself not summing to 100) is named but never defined anywhere in the spec. A0.3's '`output/images/**` **299** render receipts' is also wrong: 364 .json files exist under `output/images/`.
- M2.3(a)'S 'median **7** tags per clip' IS 6 ON DISK. Everything else in the annotation verifies exactly (863 tags, 202 clips, one-tag-needs-~30-clips framing).
- M2.6 ARITHMETIC: '8 of 66 renders measured' then '**58** of 66 renders are unmeasured ... One batch run takes this card from n=8 to **n=60**.' 66 - 8 = 58 correct; 8 + 58 = 66, not 60.
- FOUR TAB DIRECTIVES CLAIM TIERS AND MAGNITUDES THE SPEC'S OWN LADDER FORBIDS. Part 1 §4.2 binds `FINDING -> MAKE_IT` and `ABSENT -> PULL_THE_FEED` ('magnitude is bound to tier, so the size of the bet is not an opinion'). Part 2 §10 declares '`FINDING` / `PULL_THE_FEED`' and §11 declares the same — an illegal pair in both. M1.1 declares tier `FINDING` for a statement about interval WIDTH with no interval, no test and therefore `qBH = null`; §4.2 is first-match-wins and `method === "none"` lands on OBSERVATION, so FINDING is unreachable. M3.1 declares `FINDING` off n=4 platforms with no interval. Each of these is a stronger verb than the tier ladder can license — the exact failure the ladder exists to prevent.
- M1.1'S DIRECTIVE QUOTES THE WRONG COHORT AND AN UNVERIFIABLE COUNT. 'At a **258-play median** a like rate carries a +/-2.53pp 95% interval — **twelve of the top twenty overlap**.' M1.1 ranks the whole catalogue, whose median is 50 plays (258 is the modern >=50-plays outlier-excluded cohort). The 'twelve of twenty' count appears nowhere on disk and is not derivable from the stated sort (Wilson lower bound over top-20 by an unspecified measure).
- PART 3 §1.1'S SEED CHIP: 'Median words-per-line **6.3**, r = -0.30 and weak.' Measured on the modern cohort: median words_per_line = **5.834**; Pearson r = -0.31 (Spearman -0.329). The r is right; the median is not, and the chip is the motivating text a round gets pre-registered against.
- M4.4 DATES A NUMBER ITS SOURCE DOES NOT DATE. 'A **June** search snippet says 1.6K.' The 1.6K string is at `tracking/creator-platforms-latest.json:80`; that file has NO `timestamp` field (the spec itself says so in §1). The only date in the file is 'As of 2026-06-06' at line 67, attached to a TikTok answer, not to the Suno row. Also §6.2 calls the file 'six-week-old' — 2026-06-06 to 2026-08-01 is eight weeks.


## buildBlockers
- THE SHARED FOUNDATION HAS TWO NAMES AND THE INSIGHT OBJECT HAS TWO INCOMPATIBLE SHAPES. Part 1 builds `apps/lofn-kit/`; Part 2's scope note says the foundation 'is assumed to exist in `apps/_shared/` and is referenced by name here, never re-specified.' Neither exists. Worse, Part 1 §4.1 specifies camelCase (`nextMove.directive`, `nEffective`, `claim.kComparisons`, `qBH`, `drill: {href,label}`, `deeperAbsentReason`) while Part 2 §4/§13/§14.2 uses snake_case with a different drill shape (`next_move.directive`, `n_effective`, `claim.k_comparisons`, `q_bh`, `routes_to`, `drill: { to, scope }` or `drill: null, reason`). One object, two schemas; whichever a lane picks breaks the other app's named tests (`insight-tier.spec.ts` vs `drill-registry.spec.ts`).
- THE 'IDENTICAL VOCABULARY IN BOTH APPS' FILTER SERIALIZER HAS TWO KEY SETS AND TWO VALUE VOCABULARIES. Part 1 §3.3 freezes `FilterKey = mod|snap|co|basis|out|floor|nmin|prov|sel|q|exp|hyp|arm|blind`, cohort default `modern`, modality values `music·art·anim`. Part 2 §3.3 uses `m|win|pf|out|est|prov`, cohort default `all`, modality values `art·music·animation`, and §11 routes `?m=animation`. Part 2's Estimator (`est`) — 'a first-class control' whose value every card prints — has NO key in Part 1's frozen union. Part 1 §3.4's promise that 'a headwind link in Lofn Analytics opens a Soundry Sonic Anvil seed with its filter intact' is unimplementable across two vocabularies. A build agent must invent the mapping.
- FOUR REQUIRED MARKS ARE NOT IN THE SEVEN-MARK GRAMMAR AND ONE IS EXPLICITLY BANNED. Part 1 §2.2 fixes `line|band|interval|slope|bar|spark|strip` and §2 states 'a card containing hand-written SVG is **rejected on sight**'. Part 2 requires: `step` (M3.1 Window Ladder; §15 even lists `plot/step.ts`), a radial tick-arc dial (M0.1; §15 `plot/dial.ts`) which Part 1 §1.7 bans outright ('no gauge, no radial progress'), a hex bin (M2.3(b) 'hex bin (14px)' — `strip` postures are only rug/cloud/heat/bee), a linear fit with a hatched CI band (M2.1 — no OLS/fit estimator exists in `stats.ts`), a hero numeral + reference ruler (M0.2), and a 4-node weighted chain diagram (M2.7). Part 3 A2.1/B4 use `refs: kind:"threshold"`, which is not in `Ref.kind = zero|median|baseline|gate|mde|noise-floor`. Every one of these fails typecheck or `validate()` as written.
- PROVENANCE IS SPECIFIED THREE INCOMPATIBLE WAYS AND V8 MAKES IT UNENCODABLE. Part 1 §1.2: provenance 'has exactly four levels (authenticated-api / authenticated-page-read / public-page-read / search-snippet-provisional)'. Part 1 §2.1's `Provenance` enum has SIX with different spellings (`api-public | authenticated | public-page | search-snippet | prose-archive | derived`). Part 2 M3.2 colours by FIVE (adding never-pulled). V8 hard-caps `color.scale.domain.length <= 4`, so M3.2 fails validation and §1.2's own claim that 'hue on Reach charts carries *provenance*' contradicts itself.
- THE REFRESH CONTROL AND THE 'NO WRITE ENDPOINT' TEST CANNOT BOTH SHIP. Part 2 §15 test 8: 'spawn `server.mjs` ... assert ... `POST /` returns 405 with `set-cookie` null. **There is no write endpoint, and the test's job is to prove it.**' Part 2 §14.4 requires a Refresh button that shells `python3 scripts/ingest_suno.py` — a server-side side-effecting action. The spec never names the transport. A builder must invent an endpoint and then decide which test to weaken. (Note the ingest gotchas the button's copy cites DO check out: `scripts/ingest_suno.py:43` '`?page=N` alone returns HTTP 422', `:150-152` page-0 aliasing truncating at 21 clips, and `pages_walked: 11`.)
- M2.2'S BAND EDGES HAVE GAPS AND DOUBLE-COUNTED BOUNDARIES WITH NO SOURCE. Duration bands `157–205 / 208–239 / 241–269 / 272–327 s`: on-disk modern-cohort durations run 156.92 -> 327.04, so 156.92 s falls below every band and a 206 s / 240 s / 270 s song falls in NO band. `sung_lines` `<60 / 60–80 / 80–100 / 100+` and `repeated_line_ratio` `0–.15 / .15–.25 / .25–.35 / .35+` put the shared edge in two bands each. The binning rule (inclusive/exclusive, gap policy) must be invented, and the derived band n's the spec quotes depend on it.
- THE SOUNDRY'S NUISANCE SAMPLER POOLS DO NOT EXIST. Part 3 §1.4 samples five dimensions from `src/sampler.data.json`: 'the **5 Suno personas** + `none`', a 'rotating pool' of seed themes, '12 x {major, minor}', '{<=95, 96–125, 126+}'. `data/snapshots/suno/latest.json` `header.personas` is the INTEGER 5, not a list; the rows carry EIGHT distinct persona strings (Lofn 100, LOFN-Prime 14, Solve et coagula 2, Alexis Dreams 2, Nia X 1, Viralata 1, Polaroid-Void 1, Loopbot 1) plus 88 nulls. Neither 'the 5 personas' nor the key/tempo/theme pools are anywhere on disk. `design.spec.ts` asserts 'every nuisance level is balanced across arms' against pools that must be invented first.
- THE RUN-LOCK INSTRUCTION CONTRADICTS ITSELF AND THE SCRIPT. §8.3.2: '`python3 scripts/run_lock.py acquire apps/lofn-kit --run-slug <slug>`' and, in the same bullet, 'The lock file goes in a **sidecar** `apps/<slug>/.lofn/.RUN_LOCK`, **not in the build tree**.' `scripts/run_lock.py:54,145` writes `<run_dir>/.RUN_LOCK`, so the given command writes `apps/lofn-kit/.RUN_LOCK` — in the build tree. CANONICAL PATHS (`.claude/skills/lofn/SKILL.md:177`) declares `RUN_LOCK` -> `output/<run-slug>/.RUN_LOCK`. Three answers; the doctrine (CLAUDE.md: 'Exit 3 means STOP', the 2026-07-24 eleven-hour loss) makes getting this wrong expensive.
- AUTONOMY / SECURITY — THE PANEL SESSION BREAKS THE LOOPBACK BIND. Part 3 §3.3: a panel served at `#/panel/<session-token>`, 'access limited to the Scientist **and invited collaborators**, all traffic loopback.' §0.1: '`server.mjs` binds `127.0.0.1` only' and '**Local-only.** There is no hosted build.' A collaborator on another machine cannot reach a loopback bind. A build agent resolving this by widening the bind puts a server holding the full unreleased style-prompt corpus of 210 clips and the Suno `user_id` on the network — the exact exposure Part 2 §0's 'Posture' paragraph and `check-local-only.mjs` exist to prevent. `vault/COLLABORATOR_STANDARD.md`, which `vault/AUTONOMY.md` names as owning PUBLISH/PRIVATE consent, is never invoked for panel listeners or their free-text notes.
- THE .gitignore PRECONDITION ASKS THE AGENT TO REVERSE AN EXPLICIT DOCUMENTED DECISION. §8.3.1 and Part 2 §15 both call it blocking: '**Add `data/snapshots/` to `.gitignore` before the first commit** — it is absent from all 82 current lines.' The current `.gitignore` carries a comment written for exactly this case: '*# --- Lofn Analytics / Soundry data layer --- / # Snapshots and queues ARE tracked: they are the evidence behind every number / # the apps display, and their history is the trend.*' The premise that this is an oversight is false; the change would orphan already-tracked evidence and delete the history the trend depends on. This is The Scientist's call (the spec elsewhere calls it 'open question 1'), not a build precondition an agent should execute.
- A REFRESH BUTTON FOR socials/distribution WOULD HAVE TO INVENT ITS BACKEND, AND THE ONLY GENERIC CANDIDATE IS A WILDCARD-CORS WRITE RECEIVER. `scripts/ingest_receiver.mjs` sets `access-control-allow-origin: *` and binds a port to accept POSTed snapshots. Wiring a Refresh control to it from inside an app whose permanent chrome reads 'This app reads. It never posts, renders, or spends.' would stand up a cross-origin write endpoint. The spec says 'one explicit Refresh control per feed' without naming the backend for the two feeds that have none.
- §8.3.4'S CANONICAL PATHS LIST IS CORRECT BUT INCOMPLETE. `APPS`, `APP_KIT`, `SNAPSHOTS`, `APP_LOCK` are indeed missing from `.claude/skills/lofn/SKILL.md:151-178` (verified) — but so are `data/daymarks/` (§7.3 archives an SVG per day), `data/soundry/rounds/` and `output/soundry/` (Part 3 §3, §1.7). `.agents/` must be mirrored and `python3 scripts/check_skill_mirror.py` must pass; the spec names the mirror requirement but not these three keys.
- THE DAY MARK IS SPECIFIED AT TWO SIZES WITH TWO FEED COUNTS AND TWO CAPTIONS. Part 1 §7.3: '160 x 160, **in both app headers**', caption `2026-08-01 · 3/9 feeds · 210 rows · recon OK`, 'one segment per known feed'. Part 2 §3.2: '96px in the header, 160px on Provenance', '8 segments, one per `KNOWN_FEEDS` id', caption `2026-08-01 · 3/8 feeds · 210 rows · recon OK`. Part 1 §1.3 also fixes modality glyphs as '`▤` Music · `■` Art · `▸` Animation' while Part 2 §2 draws '`▤` ART · `▦` MUSIC · `▸` ANIMATION' — Music and Art swap the same glyph.