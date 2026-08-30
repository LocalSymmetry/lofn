# PALETTE CONTRADICTIONS — the four rulings the colour gate is waiting on

*Written by the FIX-2 repair lane, 2026-08-02, in response to BUILD_VERIFICATION.md
"Real failures" #3 and "Spec deviations" #1. Every number in this file was measured by running
`node apps/lofn-kit/scripts/validate-palette.mjs`, not read off a page.*

## What this document is for

`apps/lofn-kit/scripts/validate-palette.mjs` is the CI colour gate BUILD_SPEC §5.4 specifies. It
reports **11 failing assertions**. They are not eleven bugs — they are **four internal
contradictions inside BUILD_SPEC §1.1 / §1.2 / §5.4**, each of which reports across several themes.

Precedence does not settle any of them. SPEC_PATCH_2 > SPEC_PATCH > BUILD_SPEC is the order, and
**neither patch document mentions the palette at all** (verified: `grep -n "div-n\|div-p\|div-0\|
prefers-color-scheme\|data-theme\|CIEDE\|validate-palette" SPEC_PATCH.md SPEC_PATCH_2.md` returns
nothing for every palette term). All four conflicts are BUILD_SPEC against BUILD_SPEC, so the higher
document cannot overrule the lower one — they are the same document.

No build agent may settle them, because every available resolution is either **a hex nobody
authored** or **a threshold nobody set**. So each is recorded here, and each of its assertions is
registered as **EXPECTED-FAIL** in `EXPECTED_FAIL` at the bottom of `validate-palette.mjs`, keyed by
the contradiction ids below.

**The gate is honest in both directions.** It exits 1 if a NEW failure appears, *and* it exits 1 if
a registered EXPECTED-FAIL assertion starts passing — because that means a ruling landed (or the
palette drifted) and this document is now stale. It exits 0 only while the estate is in exactly the
state described here. `--strict` makes every failure blocking, for the day the rulings land.

---

## C1 — The Tide diverging ramp has no dark-mode values, so the inference triad fails the mark floor in dark

**4 assertions:** `dark: --sig-pos on --card is 2.16:1, below 3:1` ·
`dark: --sig-neg on --card is 2.16:1, below 3:1` · and the same two under `dark-auto`.

### The two citations

- **BUILD_SPEC.md:67–71 (§1.1, light `:root`)** declares the Tide diverging ramp —
  `--div-n4: #1A5954 … --div-0: #EFE9E0 … --div-p4: #9D1A20`, commented
  `L* 34/46/58/72/90/72/58/46/34 — symmetric, warm neutral midpoint`.
- **BUILD_SPEC.md:73–76 (§1.1)** binds the inference triad to its endpoints —
  `--sig-pos: var(--div-n4)`, `--sig-neg: var(--div-p4)` — under the header
  `INFERENCE TRIAD — the only colours legal inside a stats plot`.
- **BUILD_SPEC.md:120–133 (§1.1, dark block)** redefines the grounds, the ink ladder, `--absent`,
  the quartet **and its four `-edge` outlines**, `--graphite`, the whole seven-step sequential ramp
  (explicitly noting `order reversed: dark = low`) and all three chrome hues — and **redefines not
  one `--div-*` token**. So `--sig-pos` / `--sig-neg` inherit their light values into dark.
- **BUILD_SPEC.md:622 (§5.4)** — *"every mark ≥ 3:1 against its local ground, **including the hollow
  unpowered outline**, which is the mark most likely to fail and whose visibility carries the most
  meaning."*

### The measured numbers

| theme | token | hex | on `--paper` | on `--card` | on `--sunk` |
|---|---|---|---|---|---|
| light | `--sig-pos` (`--div-n4`) | `#1A5954` | 7.56 | 8.07 | 6.92 |
| light | `--sig-neg` (`--div-p4`) | `#9D1A20` | 7.55 | 8.07 | 6.92 |
| dark  | `--sig-pos` (`--div-n4`) | `#1A5954` | **2.33** | **2.16** | **2.43** |
| dark  | `--sig-neg` (`--div-p4`) | `#9D1A20` | **2.33** | **2.16** | **2.43** |

Dark grounds: `--paper #14110E`, `--card #1D1916`, `--sunk #0E0B09`.

Note that the two light values are luminance-matched to within 0.01 (7.56 / 7.55) — the ramp was
designed with care, in one theme. Note also that dark fails on **all three grounds**, not only the
worst one: this is not a local-ground quibble, and no choice of card colour rescues it.

### The minimal ruling needed

**Nine dark hexes for `--div-n4 … --div-p4`** (or, at absolute minimum, the two endpoints
`--div-n4` / `--div-p4`), added to §1.1's dark block, each ≥ 3:1 on `#0E0B09` / `#14110E` /
`#1D1916`, still luminance-matched to each other, still symmetric about `--div-0`.

**Measured for the ruling, not proposed as one:** the dark block already contains two tokens that
would clear the floor if the triad were re-aliased instead of repainted — `--m3 #4BB0A7` at 6.71:1
and `--m4 #F1A38C` at 8.61:1 on `--card`. That is the cheapest possible ruling, and it costs
something specific: BUILD_SPEC.md:1842 **separates the triad from the quartet on purpose** ("the
Ink Auditor wanted teal = interval-excludes-zero-positive and ember = negative; the Art Director
froze `--m4` ember as 'attention, never bad'. Both survive by splitting the channel"), and §1.3
freezes `--m4` as *attention, never bad*. Aliasing the triad onto the quartet in dark re-merges the
channel the spec deliberately split, and makes ember mean *negative sign* on exactly one theme.
That trade is The Scientist's to make, not a build agent's.

---

## C2 — `--sig-null` is bound to `--ink-faint`, which §1.1 itself measures below §5.4's mark floor

**3 assertions:** `light: --sig-null on --sunk is 2.65:1, below 3:1` ·
`dark: --sig-null on --card is 2.99:1, below 3:1` · and the same under `dark-auto`.

### The two citations

- **BUILD_SPEC.md:76 (§1.1)** — `--sig-null: var(--ink-faint);/* interval crosses the null —
  desaturated, always */`. Restated as behaviour at **BUILD_SPEC.md:308** (`interval` mark: *"the
  whole row desaturates to `--sig-null` regardless of the point estimate"*) and at
  **BUILD_SPEC.md:1070** (*"straddling 1.0 desaturated to `--ink-faint` regardless of the point
  estimate"*), and at **SPEC_PATCH.md:1707 / :1835**, which put `--sig-null` on the Soundry's cards.
- **BUILD_SPEC.md:622 (§5.4)** — every mark ≥ 3:1 against its local ground.

### The measured numbers

| theme | `--ink-faint` | on `--paper` | on `--card` | on `--sunk` | §1.1's own comment |
|---|---|---|---|---|---|
| light | `#9C9186` | **2.89** | 3.08 | **2.65** | BUILD_SPEC.md:51 — `/* 2.9:1 */` |
| dark  | `#6E635A` | 3.22 | **2.99** | 3.36 | BUILD_SPEC.md:125 — `/* 3.2:1 */` |

**The decisive fact: §1.1's own published light figure is already below §5.4's floor.** The gate's
2.65 is the worst of three grounds, but even §1.1's best-case 2.89-on-`--paper` — the number
BUILD_SPEC itself prints — does not reach 3:1. The two clauses cannot both hold on any ground, in
any theme. The dark shortfall (2.99 vs 3.00) is a rounding hair; the light one (2.65, and 2.89 at
best) is not.

`--ink-faint` is also pinned by §1.1:51 to `RULES AND HATCH ONLY, NEVER TEXT` — a token designed to
recede — and §5.4:622 argues the exact opposite case for marks, that the *least* powered mark is the
one whose visibility carries the most meaning. Both sentences are principled. They point opposite
ways at the same hex.

### The minimal ruling needed

One of exactly three, and it must be named:

1. **Unbind.** `--sig-null` stops aliasing `--ink-faint` and gets its own hex in both themes, ≥ 3:1
   on `--sunk` (light) and `--card` (dark), still visibly recessive against the powered triad.
   *One new token, two hexes. Touches nothing else.*
2. **Exempt.** §5.4 carves out `--sig-null` explicitly: the mark whose meaning is *"the interval
   crosses the null, do not read me"* is the one mark permitted below the floor. *One clause in
   §5.4.* This is the reading that directly contradicts §5.4:622's own stated rationale, so if it is
   chosen, that sentence needs amending too.
3. **Repaint `--ink-faint`.** Darkens every gridline and every hatch in both apps, and invalidates
   §1.1's two published ratios. *Largest blast radius; listed for completeness, not recommended.*

---

## C3 — §5.4's `CIEDE2000 ≥ 20` gate names a metric that does not reproduce §1.2's own published figures, and sets a threshold §1.2's own dark measurement cannot meet

**4 assertions** (3 in the gate + 1 frozen in `palette-cvd.spec.ts`):
`light: quartet pair m1/m2 is CIEDE2000 8.62, below §5.4's stated 20` · `dark: … m1/m2 … 5.09 …` ·
the same under `dark-auto` · plus the standing test at `tests/palette-cvd.spec.ts:132` that asserts
dark **does not** clear 20.

### The two citations

- **BUILD_SPEC.md:143 (§1.2)** — *"Minimum pairwise CIELAB ΔE₀₀ across normal + protanope +
  deuteranope + tritanope simulation (Viénot 1999): **25.2 light / 17.4 dark**. Confusability floor
  is ~12."*
- **BUILD_SPEC.md:619 (§5.4)** — *"CIEDE2000 ≥ 20 between every categorical pair in every
  simulation, both themes"*. Restated verbatim at **:794**, **:1830**, **:1864**, and as a
  justification for choosing the quartet at **:1837** and **:1956** (both of which reprint
  "25.2 light / 17.4 dark" *and* call the metric CIEDE2000).

### The measured numbers — verified independently by this lane

Minimum pairwise distance across the four vision models §1.2 names (Viénot 1999 applied in **linear**
RGB), computed from the hexes §1.1 declares:

```
CIE76 dE*ab      light  min = 25.24  (protan, m1/m2)
CIE76 dE*ab      dark   min = 17.42  (protan, m3/m4)
CIEDE2000 dE00   light  min =  8.62  (protan, m1/m2)
CIEDE2000 dE00   dark   min =  5.09  (protan, m1/m2)

§1.2 publishes: 25.2 light / 17.4 dark
```

**CIE76 ΔE\*ab reproduces §1.2's published pair to two decimal places, in both themes. CIEDE2000
misses it by a factor of roughly three.** Two independent figures matching to 2 dp under one metric
and neither matching under the other is not a coincidence; §1.2's numbers are real and were measured
with CIE76.

### The minimal ruling needed — TWO separate rulings, and the first one is one word

**C3a — the metric is misnamed.** Change `ΔE₀₀` at §1.2 (BUILD_SPEC.md:143) and `CIEDE2000` at §5.4
(BUILD_SPEC.md:619, and the restatements at :794, :1830, :1864, :1837, :1956) to **CIE76 ΔE\*ab**.
This is a one-word correction to a document, not a repalette. **No hex changes. No number in §1.2
changes.** It is the single cheapest ruling in this file and it should be taken first, because C3b
cannot be argued sensibly until the estate agrees which ruler it is holding.

**C3b — the threshold still fails dark, even after C3a.** §1.2's own dark measurement is 17.4.
§5.4 demands ≥ 20 in **both** themes. So §5.4 contradicts §1.2 by 2.6 ΔE regardless of which metric
is named. Options, in ascending cost:

1. **The gate is `≥ 12`, the confusability floor §1.2 itself states** ("Confusability floor is ~12").
   Light 25.24 and dark 17.42 both clear it comfortably. This is the only option whose number is
   already written in the spec. *This is what the validator enforces today* (`CONFUSABILITY_FLOOR`),
   which is why C3 currently produces no failures from the confusability check.
2. **The gate is per-theme: `≥ 20` light, `≥ 17` dark** — i.e. §5.4 records what §1.2 measured.
   Two numbers, one clause.
3. **Repaint the dark quartet** to clear 20 under protanopia while keeping the L* ladder monotone.
   Four new hexes plus four new `-edge` hexes; would also need re-verifying against C1's dark Tide
   values, whichever they turn out to be.

### One honesty note about the gate itself

Until this repair, `validate-palette.mjs` **did not measure §5.4's first bullet at all**. It
exported `STATED_CVD_GATE = 20` and enforced `CONFUSABILITY_FLOOR = 12` instead — a defensible
reading (option 1 above), but the file's own header claims it runs "every assertion §5.4 lists", and
it did not. §5.4's stated gate is now measured for real and registered EXPECTED-FAIL under C3, so
the contradiction is visible in the gate's output rather than only in a comment. **Both** thresholds
are now checked: the confusability floor passes, the stated gate fails, and the gap between them is
exactly what C3b is a ruling about.

---

## C4 — §1.2's stated L\* ladder gap of 12.1 does not reproduce from §1.1's own hexes

**1 assertion:** `light: quartet L* ladder minimum gap is 11.83, below §1.2's stated 12`.

### The two citations

- **BUILD_SPEC.md:55–58 (§1.1)** declares the light quartet: `--m1: #302595 /* L* 24 */`,
  `--m2: #952A8E /* L* 38 */`, `--m3: #358982 /* L* 52 */`, `--m4: #EF7959 /* L* 64 */`.
- **BUILD_SPEC.md:143 (§1.2)** — *"L\* ladder 24 / 38 / 52 / 64 with a minimum gap of **12.1**, so
  the scale also survives greyscale printing"*, in the same sentence as *"Confusability floor is ~12."*

### The measured numbers

```
light quartet hex : #302595 #952A8E #358982 #EF7959
light quartet L*  : 24.0669 / 37.8422 / 52.1281 / 63.9595
gaps              : 13.7753 / 14.2859 / 11.8314
```

The **rungs** reproduce exactly — 24.07 / 37.84 / 52.13 / 63.96 round to §1.2's printed
24 / 38 / 52 / 64. Only the **gap** does not: the true minimum is **11.83**, on the `m3 → m4` rung.

12.1 reproduces from neither the exact L\* values (11.83) nor the printed rounded ladder
(64 − 52 = 12). Every other figure in §1.1 and §1.2 reproduces to the stated precision, so the hexes
are right and the 12.1 is the arithmetic that slipped.

### The minimal ruling needed

**One number in §1.2: `12.1` → `11.8`.** Note that 11.83 is *within* §1.2's own stated confusability
floor of **"~12"** — the tilde is doing real work here, and 0.17 ΔL\* below a tilde'd floor is not a
perceptual defect. So the smallest honest ruling is a typo correction, and the validator's
`LADDER_GAP_FLOOR = 12` should follow §1.2 down to the corrected figure rather than the reverse.

The alternative — repainting `--m3` or `--m4` to open the third rung to a true 12.1 — moves a
frozen categorical hex, re-opens all of C3's CVD arithmetic, and buys 0.27 ΔL\*. Recorded so it is
on the record as rejected on cost, not overlooked.

---

## Summary — what a ruling packet has to answer

| id | the contradiction | assertions | the smallest thing that closes it |
|---|---|---|---|
| **C1** | §1.1's dark block never redefines the Tide ramp that `--sig-pos`/`--sig-neg` alias, vs §5.4's 3:1 mark floor | 4 | 2–9 dark hexes for `--div-*`, **or** a ruling re-aliasing the triad in dark (and amending §1.3 / :1842) |
| **C2** | §1.1's `--sig-null = --ink-faint` binding vs §5.4's 3:1 mark floor | 3 | unbind `--sig-null` (2 hexes), **or** exempt it in §5.4 (1 clause) |
| **C3a** | §1.2/§5.4 name CIEDE2000; the published 25.2 / 17.4 are CIE76 | — | **one word**, in six places |
| **C3b** | §5.4's `≥ 20 both themes` vs §1.2's own dark measurement of 17.4 | 4 | adopt §1.2's own `~12` confusability floor, **or** state `≥ 20 light / ≥ 17 dark` |
| **C4** | §1.2's stated ladder gap 12.1 vs 11.83 measured from §1.1's hexes | 1 | **one number**: 12.1 → 11.8 |

Three of the five are corrections to spec *text* that change no pixel (C3a, C3b, C4). Two require
paint or an exemption (C1, C2). None of the five can be closed by a build agent without inventing
something.

## Reproducing every number in this file

```
cd apps/lofn-kit
npm run palette          # honest mode: EXPECTED-FAIL reported, exit 0 only if the list matches exactly
npm run palette:strict   # every failure blocks; exit 1 while any contradiction is open
npm test                 # runs the palette gate, then vitest
```

`tests/tokens-contrast.spec.ts` imports `EXPECTED_FAIL` from the gate rather than restating it, so
the list in this document, the list the gate prints and the list the suite freezes are one list.
