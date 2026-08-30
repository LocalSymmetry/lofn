# Sunna R4.1 — Deterministic QA Evidence

This is measured evidence for the independent `lofn-qa` judge. It is not a creative verdict.

## Scope

- Run: `2026-08-24-sunna-r4-1`
- Modality: music only
- Cardinality: 6 pairs × 4 Step-10 packages = 24; selected set = 3 ACCESSIBLE + 3 AMBITIOUS, 2 NEWS + 4 EXISTENCE
- Render status: no audio rendered, uploaded, published, or paid for
- Run-local personality line band: 36–56 sung lines, per `gates_sunna_r41.yaml`

## Continuity tripwire

- Canonical ICB: `CREATIVE_CONTEXT.md`
- ICB bytes: `142674`
- ICB SHA-256: `1ce0b17997c26edf5291ce01bb2ffdc9544c7ae147b9bffdcf8b197114c206c9`
- Full ICB appears as one exact unbroken substring in all six `music/dispatch/pair_0N_dispatch.md` files: `6/6`
- `(after ` speaker-tag count in every dispatch: `18`
- ICB begins at byte/character offset `2781` in each dispatch, after the common dispatch-control header; no edits occur inside the embedded ICB

## Final validator transcript

```text
STEP 11 PASSED: pair_01_step11_enhancement.md
STEP 11 PASSED: pair_02_step11_enhancement.md
STEP 11 PASSED: pair_03_step11_enhancement.md
STEP 11 PASSED: pair_04_step11_enhancement.md
STEP 11 PASSED: pair_05_step11_enhancement.md
STEP 11 PASSED: pair_06_step11_enhancement.md

Suno package validator (explicit run-local 36–56 line band): PASS 6/6
Sunna R4.1 anti-abstraction surface audit: PASS 6/6
Canonical Step-11 file and standalone render file: byte-identical 6/6
```

## Final package measurements

| # | Title | Style chars | Exclude chars | Lyrics chars | Sung lines | Dedication chars | End rhyme | Exact-line return | Mean words/line | Alliteration | Canonical/render SHA-256 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | Friday Looks Good on Me | 956 | 578 | 2844 | 44 | 368 | .659 | .227 | 6.30 | 11.91 | `c7b6e5e04b6e956d0a216535c0f3ceb122ea20d3649c36d21d5ccc73d84e2481` |
| 02 | Make It Nine | 957 | 659 | 2392 | 40 | 388 | .300 | .300 | 4.70 | 12.23 | `a82ad855883a40b06bf8f6c4d84ece833a5f29786aa14a2cb79bc0bfce6295ef` |
| 03 | Zip Me Into Saturday | 956 | 393 | 2753 | 42 | 399 | .333 | .310 | 5.29 | 11.71 | `43b6325d78520baa92078ba2c24c46eff3e2bba7e8d250e05465125e3e8785aa` |
| 04 | One Lime, Red Chair | 946 | 514 | 2767 | 41 | 413 | .415 | .390 | 5.85 | 11.67 | `45b1f427e9729a4e33f718244f4e87b159be04d7cbb432243ab9a79dd71358b8` |
| 05 | Pretty Busy Tomorrow | 945 | 454 | 2578 | 40 | 437 | .325 | .325 | 4.78 | 14.66 | `946c2f3367692b9e9867c39a7c035ebb66851b2935e973d70611937a520ee05e` |
| 06 | Hot Girl Future Tense | 935 | 610 | 2639 | 42 | 367 | .357 | .238 | 5.83 | 11.43 | `e0502259bb9c20c48031844d775ad36c9fd12e346637f148907d929cf868cdc8` |

All six style fields clear 850–1000 and the preferred 870–960 band. All lyrics are under 5000 characters and all dedications are measured under 500 characters. Soundcraft floors used for this final pass: end rhyme ≥.30; exact-line return ≥.20; mean words/line ≤7.5; alliteration ≥11.

## Blind calibration packet

Six independent three-member sets exist at `music/qa_blind/set_01/` through `set_06/`. Each contains one selected candidate, one permitted judge-side Golden package, and one archived known-mediocre decoy, stripped to bare prompt/exclude/lyrics payload and shuffled. Every member re-stats at ≥1500 bytes. The six private coordinator keys live outside those directories and are forbidden to the independent judge until its Stage-1 rankings are sealed.
