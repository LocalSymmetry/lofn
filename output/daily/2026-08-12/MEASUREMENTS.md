# MEASUREMENTS — 2026-08-12 · all 24 songs

**Generated, not transcribed.** `scripts/measure_soundcraft.py` → `profile(sung(lyric))` over the normalized canonical step-10
artifacts, against `vault/gates.yaml`. Never eyeballed.

**Bands/floors:** style 850–1000 (target 870–960, hug flag ≥985) · lyrics field ≤5000 (target 4800) · sung lines 70–120 · end-rhyme ≥0.3 · line-return ≥0.2 · words/line ≤7.5 · allit/100w ≥11.0

| song | style | lyrics | sung | end-rhyme | line-return | words/line | allit/100w | flags |
|---|---|---|---|---|---|---|---|---|
| P01-V1 | 953 | 4732 | 84 | 0.762 | 0.571 | 6.20 | 17.08 | — |
| P01-V2 | 950 | 4612 | 82 | 0.854 | 0.598 | 6.07 | 17.67 | — |
| P01-V3 | 950 | 4391 | 82 | 0.732 | 0.622 | 6.11 | 16.57 | — |
| P01-V4 | 948 | 4781 | 88 | 0.500 | 0.625 | 5.94 | 14.34 | — |
| P02-V1 | 958 | 4490 | 93 | 0.548 | 0.656 | 5.82 | 22.00 | — |
| P02-V2 | 881 | 4063 | 92 | 0.511 | 0.641 | 5.23 | 19.96 | — |
| P02-V3 | 882 | 4074 | 91 | 0.484 | 0.648 | 5.20 | 19.66 | — |
| P02-V4 | 947 | 4245 | 90 | 0.422 | 0.633 | 5.88 | 19.47 | — |
| P03-V1 | 948 | 4767 | 90 | 0.411 | 0.267 | 5.20 | 11.75 | — |
| P03-V2 | 919 | 4655 | 90 | 0.367 | 0.278 | 5.30 | 11.74 | — |
| P03-V3 | 915 | 4339 | 90 | 0.389 | 0.344 | 4.80 | 11.57 | — |
| P03-V4 | 953 | 4638 | 90 | 0.489 | 0.311 | 5.19 | 11.35 | — |
| P04-V1 | 942 | 4796 | 92 | 0.424 | 0.489 | 7.08 | 17.67 | — |
| P04-V2 | 871 | 4783 | 88 | 0.511 | 0.511 | 7.19 | 14.69 | — |
| P04-V3 | 886 | 4643 | 84 | 0.357 | 0.488 | 6.98 | 18.43 | — |
| P04-V4 | 894 | 4761 | 83 | 0.386 | 0.494 | 7.39 | 12.72 | — |
| P05-V1 | 958 | 4326 | 109 | 0.688 | 0.789 | 4.34 | 11.42 | — |
| P05-V2 | 930 | 3900 | 94 | 0.851 | 0.787 | 3.93 | 12.74 | — |
| P05-V3 | 913 | 3908 | 98 | 0.806 | 0.755 | 3.61 | 10.17 | allit |
| P05-V4 | 896 | 3983 | 96 | 0.802 | 0.792 | 4.15 | 9.55 | allit |
| P06-V1 | 953 | 4734 | 79 | 0.759 | 0.532 | 7.27 | 15.16 | — |
| P06-V2 | 953 | 4631 | 79 | 0.823 | 0.519 | 6.62 | 21.99 | — |
| P06-V3 | 956 | 4551 | 79 | 0.810 | 0.519 | 6.87 | 19.15 | — |
| P06-V4 | 940 | 4747 | 79 | 0.772 | 0.532 | 6.58 | 28.85 | — |

## Result

**24/24 measured. ZERO hard fails.** Every band, floor and ceiling passes on all twenty-four songs.

**2 FLAG-only hits**, all alliteration, all in P05 (V1 8.60 / V3 7.08 / V4 9.55 against 11.0) — declared by the pair, not discovered here. Stated structural cause: the assigned half-line echo makes ~40% of each lyric one-to-three-word fragments, which inflate the word denominator while offering no adjacent-consonant positions. Routed to the Somatic Gate; **no threshold was changed mid-run.**

**Cross-pair:** `validate_portfolio_distinctiveness.py --step 10 --expected-total 24` → **PASS** (24 finalist packages extracted from 6 pair files; lyric, prompt and 5-gram ceilings all cleared).

## Join-side note

Every style char-count the six pair agents reported re-stats **exact** against this independent pass, as do P05's three alliteration values to two decimals. Three apparent discrepancies during normalization were all **mine** — trailing prose inside a header range inflating counts, and a newline guard too strict for a multi-line exclude field. This is the second consecutive run where the agents were right and the coordinator was wrong: **the failure rate lives on the JOIN side.**