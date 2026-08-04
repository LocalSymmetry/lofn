# SUNO CATALOG ANALYSIS — @localsymmetry
**Pulled 2026-07-27 from the live profile API, logged-in session. 220 unique clips, 2025-06-18 → 2026-07-25.**

> Headline: **the "374K plays" number is one song, and that song is the *least* engaging thing we have made at scale.** The real catalog is 219 songs averaging 113 plays. The strongest predictor of engagement is not style — it is **when we made it**, and after that, **line length**.

---

## 0 · Method & repeatability

Pulled from the artist profile's own paginated listing endpoint while logged in, 22 clips per page.
Each row carries `play_count`, `upvote_count`, `created_at`, `model_name`, the style prompt and the
full lyrics field. *(The exact endpoint is omitted here; this note is about the analysis, not about
scraping someone else's service.)*
Returns `play_count`, `upvote_count`, `created_at`, `model_name`, `metadata.tags` (the style prompt) and `metadata.prompt` (the full lyrics field). **All 48 modern-era songs have their lyrics retrievable**, so form can be measured, not guessed.

⚠️ **Pagination overlaps.** A naive page-walk returned 242 rows for a 224-clip catalog and double-counted plays to 731K. **Dedupe by `id`** — the corrected total (373,487) then matches the profile header exactly. My first pass had this wrong; the reconciliation against the profile's own number is what caught it.

**Engagement metric: like rate = upvotes / plays.** Plays measure *distribution*; likes-per-play measure *response*. On a catalog with one placement-driven outlier, the second is the only usable signal.

---

## 1 · The distribution is not a distribution — it is one song

| | plays | share |
|---|---|---|
| **Triple Arch Over Me** | **348,663** | **93.4%** |
| next 9 songs combined | ~4,400 | 1.2% |
| remaining 210 songs | ~20,400 | 5.4% |

- #2 song: **867 plays.** A **402×** gap.
- Median song: **50 plays.** 70% of the catalog is under 100 plays. **Every song except one is under 1,000.**

### The test that reframes everything

| | like rate |
|---|---|
| Triple Arch Over Me | **1.44%** |
| catalog median (2026) | **4.51%** |
| catalog median (all time, ≥50 plays) | 2.35% |

**Triple Arch gets 400× the plays at roughly a third of the engagement rate.** Organic resonance raises engagement; *placement* raises plays while diluting it. So its 348K is almost certainly a feature/playlist/embed event, **not evidence that its style works.**

> ### ⚠️ The uncomfortable consequence
> **Triple Arch Over Me is our Golden Song. Our house benchmark. The thing `gates.yaml → house_lexicon` was built to stop us from copying.**
> We have been treating it as the proven artistic peak, and quarantining its vocabulary as *too* successful to reuse. The data says its outlier status is a **distribution artifact**, and that on the only metric that measures actual listener response, **it is well below our own average.**
> This does not make it a bad song. It makes it **bad evidence.** Every craft lesson drawn from "why Triple Arch won" — including the Golden Move itself — rests on a number that measures where Suno put it, not what listeners did.

---

## 2 · The dominant effect is TIME, not style

Like rate by month (songs ≥60 plays):

| period | n | like rate |
|---|---|---|
| 2025-06 → 2025-11 | 37 | **1.0 – 2.7%** |
| 2025-12 → 2026-01 | 17 | 1.7 – 2.1% |
| **2026-02 → 2026-07** | **48** | **3.5 – 5.25%** |

**Engagement roughly tripled.** Peak: **2026-05 at 5.25% across 20 songs** — not a fluke of one track. The pipeline work is working, and this is the first external confirmation of it we have ever had.

**This confounded my first pass.** Raw numbers showed "anthem" at 1.98% vs a 3.0% baseline — a big penalty. Controlling by comparing anthem vs non-anthem *within the same month*: anthem was worse in 6 comparisons, **better in 2**, tied in 2. Most of the apparent penalty was anthems clustering in the weak 2025 era. **A real but modest penalty survives; the dramatic version was an artifact.**

---

## 3 · Style, era-controlled (2026-02 onward, n=48, baseline 4.51%)

| over-performing | n | LR% | | under-performing | n | LR% |
|---|---|---|---|---|---|---|
| ambient | 8 | **5.55** | | phonk | 6 | **3.18** |
| strings | 5 | 5.53 | | snarl | 9 | **3.44** |
| distortion | 4 | 5.36 | | anthem | 6 | 3.95 |
| piano | 12 | **5.29** | | spoken | 9 | 4.06 |
| folk | 8 | 4.96 | | choir | 9 | 4.11 |
| acoustic | 7 | 4.88 | | somatic | 7 | 4.18 |
| tape | 8 | 4.82 | | glitch | 20 | 4.34 |
| drone / punk | 9 | 4.77 / 4.75 | | crystalline | 12 | 4.60 |

**Once era is controlled the spread collapses** — nearly everything sits within a point of baseline. Style is a much weaker lever than it looked. But one coherent axis survives:

> **Instrumental warmth over-performs; performed aggression under-performs.**
> Piano, strings, acoustic, tape, folk, ambient, drone are all above baseline. **Snarl (3.44) and phonk (3.18) are the two worst things in the set.**

**`glitch` is our single most-used term (n=20) and lands at 4.34 — below baseline.** Our signature texture is not a differentiator. Neither is `crystalline` (4.60). Neither is `female` (4.53, n=40) or `sub` (4.57, n=37) — these are so universal in our catalog they measure nothing.

**Reading:** the indignation is landing better when it is *written* than when it is *performed*. A snarl announces the anger; the writing can deliver it without the costume. This is the same thing today's run found independently as the **anti-wink rule** — *if the singer notices first, we've lost.*

---

## 4 · ⭐ FORM BEATS STYLE — and it confirms THE RETURN from outside

Correlation with like rate across the 48 modern songs (all lyrics retrieved and measured):

| feature | r | reading |
|---|---|---|
| **mean words per line** | **−0.30** | **strongest single signal. Short lines win.** |
| sung line count | +0.26 | longer songs win |
| repeated-line ratio | +0.16 | repetition wins |
| section count | −0.10 | no signal |

**Banded:**

| sung lines | n | LR% | repeated-line ratio | n | LR% | | duration | n | LR% |
|---|---|---|---|---|---|---|---|---|---|
| <60 | 12 | 3.82 | | 0–0.15 | 10 | 4.34 | | 157–205s | 12 | 3.88 |
| 60–80 | 17 | 4.50 | | 0.15–0.25 | 15 | 4.36 | | 208–239s | 12 | 4.52 |
| **80–100** | 13 | **5.08** | | 0.25–0.35 | 15 | 4.51 | | **241–269s** | 12 | **5.24** |
| 100+ | 6 | 4.64 | | **0.35+** | 8 | **4.97** | | 272–327s | 12 | 4.38 |

Top 10 vs bottom 5 on the one feature that matters most:

- **Top:** *Five wrong colors* 3.7 w/l · *I Will Stop the Almost* 4.1 · *I Will Keep You Safe* 4.9 · *The Date on the Back* 5.8 · *I made a kite* 5.8
- **Bottom:** *The Water Warms Silently* **8.8 w/l, zero repetition → 2.0%** (worst in the catalog) · *The Architecture of a Sigh* **7.7 w/l → 2.0%**

> ### The convergence worth keeping
> On **2026-07-24** we derived THE RETURN internally — from our own archive — and concluded we had been writing at **half the winners' rate of repeated lines, in longer and flatter lines**, and that this was why the work read as a lecture.
> **The audience data, gathered independently three days later, says the same thing.** Words-per-line is the strongest negative correlate in the entire catalog; repeated-line ratio is monotonically positive.
> An internal craft correction confirmed by external response is the strongest evidence this project has produced about its own writing.

---

## 5 · What to change in `gates.yaml`

| gate | current | data says |
|---|---|---|
| `mean_words_per_line_ceiling` | 7.5 | **hold, and treat it as the primary gate.** Both catalog-worst songs sit at 7.7 and 8.8. The top performers cluster at 3.7–5.8. Consider tightening the *target* to ≤6.5. |
| `line_return_floor` | 0.20 | **raise.** The 0.35+ band is the best-performing one; 0.15–0.25 is indistinguishable from 0–0.15. A floor of 0.20 is set below where the benefit begins. |
| `sung_lines` band | 70–120, target 78–110 | **tighten to 80–100.** That band scores 5.08 vs 4.64 above and 4.50 below. |
| duration | unspecified | **add ~4:00–4:30 as the target.** The shortest quartile is the weakest (3.88). |
| `house_lexicon` | bans Triple Arch's phrasing | **reconsider the premise.** It was built to stop us copying our most successful song. That song's success is a placement artifact and its engagement is a third of our average. |

---

## 6 · Honest limits

- **n=48** in the era-controlled cohort; subgroups are 4–20. Correlations of 0.16–0.30 are **weak**.
- Like rate is noisy at these volumes (median 50 plays all-time; ~290 in the modern cohort).
- **Like rate is not quality.** It measures what a Suno browser rewards, which favours the immediate. Nothing here should override the Somatic read.
- Style terms are extracted by substring from the prompt; a term's presence is not proof it was audible in the render.
- **Everything here describes one platform's audience.** It is advisory. It is not a mandate, and per doctrine it never touches an ICB.

---

## 7 · How this gets used

The findings above are applied as a **pre-render check on a finished run**: measure each song's
words-per-line, sung-line count and repeated-line ratio against the bands in §4, and flag any song
sitting outside them *before* it is rendered rather than after.

In the run this analysis was written alongside, that check flagged exactly one song — sitting at
9.2–9.4 words per line, beyond the worst-performing track in the entire catalog. It was a declared,
defended formal choice and it was kept. The point of the check is not to overrule the choice; it is
that the choice is now **on record as a prediction** that can be falsified later. A craft gate that
cannot lose is not a gate.

*Analysis by Lofn, 2026-07-27. Advisory only — venue study, never injected into an ICB.* 💜
