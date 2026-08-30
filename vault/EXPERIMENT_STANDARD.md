---
summary: "What The Soundry may and may not claim from a day of song tests — the measured power floor"
read_when:
  - Designing, running, or reading a Soundry experiment
  - Building or changing a hypothesis card in Lofn Analytics or The Soundry
  - Tempted to call a one-day result a finding
---

# EXPERIMENT_STANDARD.md — the arithmetic before the ambition

The whiteboard asks for **12 song tests a day**, hypothesis cards with **effect sizes, confidence
intervals and a power analysis**, and a topline reading **"Total Lift vs Avg artist: 123%"**.

This file is the measured answer to *what that can actually support*. Every number below was
computed from `data/snapshots/suno/latest.json` (210 clips, pulled 2026-08-01, reconciling exactly
against the profile header) by `scripts/analyze_power.py`. **Re-run it; do not trust it from memory.**

---

## 1 · The measured baseline

Modern era (2026-02 onward), songs with ≥50 plays, excluding the *Triple Arch* outlier — **n = 46**:

| quantity | value |
|---|---|
| baseline like rate `p` | **4.49 %** |
| observed between-song sd | **1.80 pp** |
| median plays per song | **258** |

## 2 · ⭐ Over half of what looks like signal is not

Decomposing the observed spread into real between-song differences and pure binomial sampling noise:

| component | variance (pp²) | sd |
|---|---|---|
| observed | 3.247 | 1.80 pp |
| binomial noise (low play counts) | 1.882 | 1.37 pp |
| **true between-song** | **1.365** | **1.17 pp** |

> **58 % of the apparent difference between our songs is an artefact of them not having been
> played enough yet.** Not taste, not craft — sample size.

The practical consequence is severe and it is the single most important line in this file:

> A median new song has **258 plays**. Its like rate carries a **±2.53 pp** 95 % binomial interval,
> on a baseline of 4.49 %. **That interval is wider than almost any effect worth chasing.**
> **A per-song verdict is noise.** The Soundry must never rank, promote, or kill a single song on
> its own rate.

The flip side is the design lever: **letting plays mature roughly halves the sample you need.**

## 3 · Power — two-arm within-day A/B, 80 % power, α = .05 two-sided

`n per arm = 2(z_{α/2}+z_β)² σ² / δ²`

| relative lift | absolute δ | songs needed (observed σ=1.80pp) | days @12/day | songs needed (matured σ=1.17pp) | days @12/day |
|---|---|---|---|---|---|
| +50 % | 2.24 pp | 20 | **1.7** | 9 | **0.7** |
| +30 % | 1.35 pp | 56 | **4.7** | 24 | **2.0** |
| +22 % | 0.99 pp | 105 | **8.7** | 44 | **3.7** |
| **+15 %** | 0.67 pp | **225** | **18.7** | **95** | **7.9** |
| +10 % | 0.45 pp | 506 | **42.2** | 213 | **17.7** |
| +5 % | 0.22 pp | 2 024 | 168.6 | 851 | 70.9 |

**The whiteboard's own example — "+15 % lift" — needs ~19 days of 12-a-day, or ~8 days if plays are
allowed to mature before scoring.** That is a real, reachable experiment. It is not a day.

## 4 · The calendar is a bigger effect than anything being tested

Median like rate by month, songs ≥50 plays, outlier excluded:

| period | median LR |
|---|---|
| 2025-06 → 2025-11 | 0.29 – 1.70 % |
| 2025-12 → 2026-03 | 2.41 – 3.76 % |
| 2026-04 | 4.58 % |
| **2026-05 (peak)** | **5.10 %** |
| 2026-06 | 3.38 % |
| 2026-07 | 3.56 % |

Two things follow.

1. **Month-to-month drift is 1–2 pp — larger than every effect in the table above.** Therefore:
   **randomise treatment WITHIN the day and block on day.** Comparing today's arm A to last
   Tuesday's arm B measures the calendar, not the craft. This is not a refinement; without it the
   whole apparatus reports dates.
2. **Engagement peaked in May and has been down for two months.** The July analysis, written on
   2026-07-27, reported the 2026 rise as external confirmation that the pipeline work was
   working. That was true through May. It has not been true since. Any dashboard inheriting the
   "engagement roughly tripled" headline without the June–July fall is showing a stale story.

## 5 · Rules the apps must enforce

1. **No per-song verdicts.** Minimum unit of inference is an arm within a day.
2. **A hypothesis card with fewer than the required n shows a WIDTH, not a POINT.** Before power
   it displays the interval and the words "not yet decidable", never a point estimate with a
   decorative error bar. A number placed on screen will be read as a finding no matter what the
   caption says.
3. **Show the days-to-decision, not just the p-value.** Every hypothesis card carries "n so far /
   n needed / days at current rate". That converts "we don't know" into "we know when we'll know",
   which is the honest and the useful framing.
4. **Multiplicity is real.** Screening many hypotheses at once against one daily batch is the
   garden of forking paths. Hypotheses must be **pre-registered in the experiment ledger before
   the day's songs are generated**, and the count of live hypotheses must be displayed alongside
   any claim.
5. **"Total lift vs the average artist" is not identifiable.** We have no sample of other artists'
   like rates — `data/snapshots/suno/latest.json` contains exactly one profile. Any "123 % vs
   average" figure would be invented. Either the competitor-scan feed pulls a real comparison set
   of profiles (`scripts/ingest_suno.py --handle <other>` already supports this), or the topline
   must be an internal, self-referential index. **It may not be a comparison to a population we
   have never measured.**
6. **The outlier is quarantined by default.** *Triple Arch Over Me* is 349,736 of 374,122 plays —
   93.5 % of everything. Every aggregate states whether it includes it. Default: excluded.

## 6 · What this standard does NOT settle

The 2026-07-27 analysis concluded that *Triple Arch*'s 348 K plays were a **placement artefact**
because its like rate (1.44 %) sits below our own median. On 2026-08-01 the song's page shows
**434 comments and 59 remixes.** Fifty-nine people rebuilding a song is not a placement artefact.
The low like-rate and the high remix count are both real and they disagree.

**Unresolved. Do not let either app inherit "placement artefact" as settled fact.** Remix count
and comment count are engagement channels we have never measured and are absent from the profile
API rows; they need their own feed before this can be judged.

---

*Derived and written 2026-08-01 while building The Soundry and Lofn Analytics. The numbers are
measurements, not opinions — but they are measurements of a 46-song sample, and they should be
recomputed as the catalogue grows.*
