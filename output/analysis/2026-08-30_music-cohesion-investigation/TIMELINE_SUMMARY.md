# Timeline Summary — deterministic lyric proxies

This is a descriptive control, not an aesthetic verdict. Values are medians across the six declared selections in each completed run. Lyrics were stripped by `analyze_lyrics.py`; the extraction corpus hash is recorded in `CORPUS.json`.

| Run | Sung lines | Words / line | Ultra-short | Fragment proxy | List-run proxy | Adjacent carry | Exact repeat | Abstract / 100w | Action / 100w | Sections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-08-21 | 84.0 | 7.147 | .031 | .170 | .018 | .569 | .215 | .806 | 5.986 | 10.0 |
| 2026-08-22 | 80.0 | 8.025 | .025 | .222 | .031 | .710 | .265 | .721 | 2.915 | 10.0 |
| 2026-08-24 | 93.5 | 4.510 | .259 | .449 | .169 | .469 | .210 | 1.290 | 9.655 | 9.0 |
| 2026-08-25 | 95.5 | 4.110 | .329 | .393 | .168 | .413 | .173 | 2.439 | 11.339 | 12.0 |
| 2026-08-26 | 75.0 | 7.585 | .000 | .220 | .021 | .554 | .374 | 2.534 | 3.808 | 12.5 |
| 2026-08-27 | 84.0 | 6.175 | .063 | .228 | .068 | .584 | .289 | 1.719 | 5.006 | 12.5 |
| 2026-08-28 | 87.5 | 5.827 | .123 | .177 | .063 | .649 | .291 | 2.565 | 8.014 | 13.5 |
| 2026-08-29 | 75.5 | 4.820 | .078 | .248 | .052 | .481 | .183 | 2.352 | 12.061 | 11.0 |

## What these proxies permit us to say

1. The post-Wedding period is **not a monotone decline in local linkage**. The carry proxy falls on 08-24/25, recovers through 08-28, then falls again on 08-29.
2. The 08-29 batch is **not unusually fragmentary**. Its ultra-short and fragment rates are below 08-24/25, and its adjacent-carry median is almost identical to the user-named high-bar cohort (.481 vs .482).
3. Abstract vocabulary does show a persistent level shift: every run from 08-25 onward has a higher median than 08-21/22. This proxy is narrow and cannot distinguish a necessary abstract word from an abstract proposition.
4. More sections do not explain the defect. The 08-28 run has the most sections and also the strongest post-08-22 carry proxy.
5. The 08-29 batch combines comparatively short lines, elevated abstract vocabulary, the highest action-word proxy in the timeline, and weak exact repetition. That is consistent with **many active statements serving different propositions**, not with inert description or missing verbs.

## Required cautions

- `adjacent_carry_proxy` counts lexical/connective carry; it does not know whether the same human problem is advancing.
- The engine-token proxy is excluded here because it undercounts devices that are not named in the lyric and therefore is not comparable across formats.
- These are text-only measures. They cannot diagnose generator phrasing, melody, edit timing, or vocal rendering.
