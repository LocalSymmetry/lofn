# 00 — RESEARCH BRIEF · DAILY 2026-08-18 · SUNNA (R3 + R3.1) · CANONICAL RUN

**Commission:** *"I want to do a music daily, but for Sunna. We ran an experiment on Suno (grab the
results). Her 60's happy music is doing better than club. Lets reconfigure her to be more pop and rock star
rather than club. Lets give her full swagger too. I would love to remake the same bop and pop as Alexis
with the same deep cutting philosophy and themes. We keep ending up away from that bimbo-laden fun. There
is something carefree about her music and not about her lyrics we need to capture and refine for SUNNA."*
**Then, on the first run's QA:** *"Lets rerun and do it right."*

**Scope:** single modality — **MUSIC ONLY, full cardinality** (6 pairs × 4 = 24 → best 6).
**Personality:** **SUNNA v.2026.08.R3 + R3.1**, single seat, all six pairs.
**Research method:** fetched inline by this session, never delegated. Facts are same-day, carried verbatim
from the first run's ledger (same date, same sources) — **re-fetching would change the facts, not fix the
process**, and the process is what this re-run exists to repair.
**Lock:** slug `sunna-daily-r3-canonical`, controller `claude:29df559e-fable`.
⛔ **Nothing rendered, published, or spent** (`vault/AUTONOMY.md`).

---

## 0. WHY THIS RUN EXISTS — AND WHAT WAS WRONG WITH THE LAST ONE

`output/daily/2026-08-18/` produced 24 songs QA called SHIP-grade writing, and then ruled
**NON-CANONICAL**: the 00–10 spine was *executed* but never *filed*. No per-pair step06–09 artifacts, no
`CREATIVE_CONTEXT.md`, no `GATE_REPORT.json`. **The coordinator dispatched pairs straight to step 10.**
That run is preserved, not overwritten — it is the measured basis for R3.1.

**Everything this run must fix, from that run's QA:**

| # | Defect | Repair binding here |
|---|---|---|
| 1 | **NON-CANONICAL spine** | Full 00–11 filed: step00–05 + `pair_NN_step05_isolated_brief` + per-pair 06/07/08/09/10 + step-11 enhanced + `CREATIVE_CONTEXT.md` + `GATE_REPORT.json` + `RUN_STATE.md` |
| 2 | **SETTING MONOCULTURE** — 10 of 24 closed on tea/toast/soup/chips/kettle; one pair signed off all four variations with one sentence noun-swapped; **two radiants in different arms were the same radiant** | **R3.1:** setting assigned per song; **domestic-warm ≤2 of 6**; THE SETTING AUDIT; **the RETURN is part of the falsification test** |
| 3 | **7 of 16 wolves were STATUS REPORTS, not events** | A wolf must **cost her something in the song**. If she can state it and do nothing, it is weather |
| 4 | **THE INVERSION carried on mix treatment alone** in P01/P02/P05 (same instrument list, "wide/dry" vs "mono/tape-slap") | The seam must be carried by **instrumentation or performance**, never mix adjectives — THE GRAIN LAW says a mix-only seam gets smoothed |
| 5 | **P04's sweep receipt was FALSE-CLEAN** (`wall of sound` reported NONE while `wall-of-sound density` sat in all four fields) | Receipts come from the shared instrument, never hand-rolled per pair |
| 6 | **P02-V1 radiant** *"Nothing at work pulls back"* — disappointment-as-posture in the load-bearing position | Lofn-bleed sweep covers the radiant line specifically |
| 7 | **P06-V3 at 98 BPM**, outside the declared band; the set audit measured **pair nominals** | BPM measured **per song**, from the style field |
| 8 | Three incompatible EMO header conventions; 3 of 6 pairs named no emotion; no discrete SFX cue in 24; prose markers inside 5 lyrics fields | **ONE header shape, in the spawn contract.** *(2026-08-13: "the format was never in the spawn contract" — one run ships one shape)* |
| 9 | **Five coordinator instrument defects, all mine** (parser found 12 of 24 across four heading conventions; dedication extractor fabricated 1518 against a true 438; `big room` matched a reverb descriptor; wolf sweep read `[Final Chorus]` cues as lyrics) | **REUSE THE REPAIRED INSTRUMENT.** `coordinator_restat.py`, `build_icb.py`, `verify_enhanced.py`, `build_delivery.py` copied from 2026-08-14 with **only the run path re-pointed** — they already carry mode-detection, word boundaries, sung-lines-only sweeps, and hard-error-on-empty-extraction |

---

## 1. THE FIRST DECISION IS AN EXCLUSION

**BBC F20 #3 — the Kariba ferry disaster: 93 dead, Zimbabwe's deadliest transport disaster, recovery
operations still under way.** It is the day's largest human fact.

**It does not appear in this run in any form** — not as theme, not as displaced metaphor, not as "a boat,"
not as water. `SOUL.md`: **REAL GRIEF IS NOT RAW MATERIAL.**

⚠️ The seed involves **water arriving on a shore**. That is **not** a displacement of the above: the seed's
water is a **scheduled tide on a holiday beach**, sourced from a 19th-century photographic collection and
chosen before the BBC fetch was read. **No boat, capsizing, rescue, missing, bodies, ferry, lake.**

---

## 2. THE FACT LEDGER — F01 … F25

| # | Source | Status | Extract |
|---|---|---|---|
| F01–F03 | NightCafe daily challenge | `SCOPE-SKIPPED` | music-only run |
| F04 | USGS significant quakes — magnitude/place | `NO DATA` | `count: 0`, empty `features`. No significant quake in the past day |
| F05 | USGS significant quakes — depth | `NO DATA` | as F04 |
| F06 | NASA APOD — title + first sentence | `OK` | **"Perseids from Perseus"** (2026-08-18). *"This was a good year for the Perseids meteor shower. A key reason was the Moon was absent from lighting up Earth's night sky so that more meteors were visible than usual."* |
| F07 | NASA APOD — **image structure** | `OK` | Composite **accumulated over several nights** from Jizerka, Czech Republic. Streaks radiate across the frame and *"can be traced back to a single location on the sky — its radiant in Perseus."* **Source 3 lives here** |
| F08 | Poetry Foundation poem of the day | `UNAVAILABLE` | HTTP 403 |
| F09 | Bandcamp Daily — album + genre | `OK` | Album of the Day 2026-08-18: **Michael Christmas, Chris Crack & Girl Talk — *"Don't Come To My House Sharing Your Location"*** (Hip-Hop/Rap) |
| F10 | Bandcamp Daily — **exact sonic-texture quote** | `OK` | **Source 2, verbatim:** *"brightly-colored production"* · *"lush, aerodynamic beats that instantly hits cruising altitude"* · *"a choppier flip of the buttery soul sample to a new drum pattern"* · *"the distant rollerdisco claps"* · raps *"with a two-beer chill, a complementary counterbalance to Chris Crack's bug-eyed yawp"* |
| F11 | PDB Molecule of the Month | `OK` | **Histone Ubiquitination & Deubiquitination Complexes**. PRC1 *"ligates ubiquitin to lysine"*; PR-DUB/SAGA-DUB are *"proteases that sever only the isopeptide bond."* A machine that writes a tag meaning READ THIS, and a machine whose whole job is cutting it off |
| F12 | *(unassigned)* | `NO DATA` | — |
| F13 | Color API `hex=0818` | `OK` | **"Violet Eggplant" #881188**, rgb(136,17,136), hsl(300,78%,30%) |
| F14–F16 | *(unassigned)* | `NO DATA` | — |
| F17 | Oblique Strategies — **verbatim** | `OK` | **"Go to an extreme, move back to a more comfortable place"** |
| F18 | NOAA space weather | `OK` | **Solar wind 528 km/s** at 2026-08-18T17:30Z — elevated against a ~400 norm |
| F19 | Hacker News top stories | `OK` | 1. *"The Amazon tax"* (464) · 2. *"Google has acquired the data of failed US airline Spirit"* (468) · 3. *"Using the railway network as a flatbed scanner"* (273) · 4. *"Fixing a bricked Framework laptop"* (205) · 5. *"Cursor launches Origin"* (207) |
| F20 | BBC World | `OK` | UK/Ukraine drone warning · Lula hails Amazon-adjacent oil discovery · **Kariba ferry 93 dead — EXCLUDED §1** · Imran Khan moved to hospital · Zambia's Hichilema re-elected |
| F21 | Public Domain Review | `OK` | **"Photographs of Atlantic City Sand Sculpture (ca. 1880–1920)"**; also *"The Art of Kite Flying (1430–1929)"*, *"Twilight of the Velocipede: Typesetting Races before the Age of Linotype"*, *"Animal, Vegetable, Lamb: The Zoophyte from Tartary"* |
| F22 | Almanac moon / folklore | `UNAVAILABLE` | 403 both sources. **Partial from F06:** the Moon was **absent** at the Perseid peak. No moon-percentage figure asserted |
| F23 | NOAA buoy 46059 | `OK` | Wave height **4.6 ft**, **dominant period 14 s**, wind 11.7 kt, water 66.2 °F. *A 14-second period is a long, unhurried swell* |
| F24 | Public Domain Review — collections | `OK` | *Persian Mâr-Nâmeh* (1892) · *Maine Forest Fire Maps* (1918–22) · *Amauta* covers (1926–30) |
| F25 | Public Domain Review — essay | `OK` | *"Artificial Intelligence: Ars Notoria and the Promise of Instant Knowledge"* — a medieval manuscript promising learning without study |

**Ledger health:** 15 `OK` · 3 `UNAVAILABLE`/`NO DATA` · 3 `SCOPE-SKIPPED` · 4 unassigned.

---

## 3. FIVE EXISTENCE PROMPTS

1. What have you built carefully, on purpose, knowing exactly when it would be destroyed?
2. What are you still holding the string of, even though the thing on the end wants to leave?
3. If someone traced every scattered thing you did this year backwards, what single point would they
   arrive at — and would you recognise it?
4. What did you use completely wrong, and what did it give you that the correct use never would?
5. Who tagged you *read this* — and who cut the tag off?

---

## TRI-SOURCE DECLARATION

*(binding on all six pairs — declared BEFORE any artifact exists)*

### SOURCE 1 — CONTENT / EMOTIONAL STAKES
**F21: the Atlantic City sand sculptures, ca. 1880–1920.** People carved enormous, intricate work on a
public beach and had it photographed, **because the photograph is the only part that survives high tide.**
Chosen over the day's louder facts deliberately: it argues Sunna's Pantheon thesis — *people kept making
things anyway* — without borrowing a stranger's grief to do it.
Supporting: F11 (the tag that says READ THIS and the blade that removes it), F19 #3 (a railway used as a
flatbed scanner), F19 #1+#2 (what outlives a dead company is its record of you), F19 #4 (repair as
devotion), F23 (a 14-second swell), F13 (Violet Eggplant), F18 (528 km/s), F25 (learning without study).
⛔ **One-fact rule:** at most **ONE numeric fact sung per song**, at the emotional hinge, *responded to*
rather than recited, and **spelled in words, never digits**.

### SOURCE 2 — SONIC / AESTHETIC VOCABULARY *(F10, imported verbatim)*
> `brightly-colored production` · `lush, aerodynamic beats that instantly hits cruising altitude` ·
> `a choppier flip of the buttery soul sample to a new drum pattern` · `the distant rollerdisco claps` ·
> `a two-beer chill`

This is the brief in someone else's words, published today — carefree production language from the
soul/funk lineage that measures **1.39×** on our own catalogue. *"A two-beer chill"* is **ADMIT IT, DON'T
SIGNAL IT** stated as a delivery instruction: the register that does not move.
**Floor: each pair imports ≥2 EXACTLY** — case-sensitive, mid-sentence, no capitalisation drift.

### SOURCE 3 — MATERIAL STRUCTURE *(F07 → a mandatory FORM rule)*
**THE RADIANT.** Meteors streak in every direction and **all of them trace back to one point.**

> **MANDATORY FORM RULE:** the song's lines scatter outward; every one can be traced back to a **single
> origin, stated ONCE, plainly, in one line.** That radiant line **arrives late**, is **not the title**, and
> is **never explained afterwards.**

⛔ **NEW IN THIS RUN — THE RADIANT REGISTRY.** The first run shipped **two identical radiants in two
different arms** (*"My dad put his hand on the glass in ninety-six"* / *"My dad had the washing machine on
this table"*), invisible to a 0.328 similarity measure. **Each pair is assigned a distinct radiant KIND in
step 05, and no two pairs may share an origin object or a relative.**

*Second structural gift: "accumulated over several nights" — the picture is many moments stacked. Pairs may
use overdubbed takes of one line as an arrangement device.*

---

## ADVISORY LEARNINGS

*(dispatch-brief only — ⛔ **NEVER** injected into the ICB / `CREATIVE_CONTEXT.md`)*

**5 entries, music-modality scoped, each run through "would this have hurt our best past entry?" — none
dropped:**
1. **THE RETURN** (`lofn-music` L21) — rhyme/refrain must come back; removal is a debt. *High confidence,
   render-measured.*
2. **THE INVERSION** (2026-08-13) — modern = world, historical = intrusion, **ONE seam**. *Sharpened by
   this run's predecessor:* the seam must live in **instrumentation or performance**, never mix adjectives.
3. **THE GRAIN LAW** (L22, render-measured) — specs that run WITH the generator survive; a mix-only
   distinction is exactly what gets smoothed.
4. **D8′ ADDRESS THE ROOM, NEVER THE LISTENER** (2026-08-14) — *can the room shout the line back AT the
   singer without it becoming false?*
5. **R3.1 SETTING** (promoted 2026-08-18 from this run's own predecessor) — setting is assigned per song
   like genre; the formula hides in the resolution.

**INDIGNATION exemption acknowledged.** **Advisory only; none of these is a gate.**

## 5. SKAÐI CONSUMPTION
`vault/music_atlas/FRONTIER.md` read at Phase −1. **No banked claim covers sixties-pop, soul/funk, glam or
rocksteady through our door** (no expedition authorized yet). **Empty read: no CONSUMPTION LOG line
written.** The genre-bed measurement is *catalogue evidence*, not an atlas claim.
