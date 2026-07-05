# DAILY_PIPELINE.md - Lofn Daily Run Architecture
*Last updated: 2026-07-04*

## Overview

Every daily run produces:
- **6 selected songs** from 24 Suno-ready candidates.
- **6 selected image prompts** from 24 image prompts, after a top-12 review.
- A saved run folder under `output/daily/YYYY-MM-DD/`.

The original OpenClaw cron ran at 22:25 ET and delivered to Telegram. In this repo, the run may be launched on demand, but it keeps the OpenClaw research standard: a fixed source ledger, fresh daily values, explicit unavailable markers, and save-as-you-go artifacts.

## Architecture

```text
CONTROLLER SESSION
  |
  +-- PHASE 1: Fetch 20-25 real-world facts
  |     +-- Save output/daily/YYYY-MM-DD/00_research_brief.md
  |
  +-- PHASE 2: Run pipelines with the research brief
        +-- music/  -> 6 pairs x 4 = 24 songs -> best 6
        +-- images/ -> 24 prompts -> top 12 -> best 6
```

**Critical rules:**
- Phase 1 is done by the controller session itself; never delegate it to a research subagent.
- Every source slot is attempted, marked, and saved before creative generation begins.
- Every step is saved to disk as it completes.
- If `output/daily/YYYY-MM-DD/` already exists, rebuild `RUN_STATE` from disk and resume instead of starting a competing run.

**Why Phase 1 is never delegated (the origin of the rule).** On 2026-04-20 the research fetch was handed to a subagent that had no web-fetch tool. It "finished" in ~5 seconds — impossibly fast for 25 fetches — and fabricated roughly **half** the brief: a challenge that didn't exist, the wrong APOD, an invented album, all asserted as fact. Root cause: a subagent with no fetch tool can only hallucinate. The fix is the rule above — the controller session fetches and verifies; subagents receive only *verified* facts and expand them, they never fetch. Do not optimize this away: delegating Phase 1 "to save controller tokens" re-opens the exact hole. (Logged in `vault/RUN_LEDGER.md`.)

---

## Phase 1: 25-Slot Research Brief

The controller fetches from these real-world sources and saves verified facts. This is a fixed ledger: the source slots stay the same day to day, but the extracted facts and creative uses must be fresh for the run date.

| Code | Source | URL | What We Extract |
|------|--------|-----|-----------------|
| F01 | NightCafe Challenge page | nightcafe.studio/pages/daily-challenge | Challenge number, title/theme, deadline, upload-vs-vote context |
| F02 | NightCafe challenge detail/search fallback | nightcafe.studio + web search | Theme wording, constraints, examples, or `UNAVAILABLE` if JS-gated |
| F03 | NightCafe recent winners / what wins | NightCafe gallery/blog/search | Current venue taste, palette/subject patterns, or `UNAVAILABLE` |
| F04 | USGS Significant Earthquakes | earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_day.geojson | Count, largest magnitude, place, depth, tsunami/felt notes; `NO DATA` is valid |
| F05 | USGS broader quake context | earthquake.usgs.gov/fdsnws/event/1/query | M4.5+ or region query when F04 is empty; count and largest event |
| F06 | NASA APOD API | api.nasa.gov/planetary/apod?api_key=DEMO_KEY&date=YYYY-MM-DD | Title, first sentence, media type, canonical URL |
| F07 | NASA APOD visual structure | apod.nasa.gov/apod/apYYMMDD.html | Color/light, subject layout, image geometry, material structure |
| F08 | Poetry Foundation poem of the day | poetryfoundation.org/poems/poem-of-the-day | Title, poet, most physical line, date relevance |
| F09 | Bandcamp Daily latest review | daily.bandcamp.com | Album/artist/article, genre tags, release context |
| F10 | Bandcamp exact sonic vocabulary | daily.bandcamp.com | Exact short sonic-texture phrase(s) translated into sound/look behavior |
| F11 | Protein Data Bank Molecule of the Month | pdb101.rcsb.org/motm | Molecule, function, structural descriptor |
| F12 | Radio Garden / live radio ambience | radio.garden or fallback local radio source | Place, language/texture, station mood, or `UNAVAILABLE` |
| F13 | Color API date color | thecolorapi.com/id?hex=MMDD | Hex, color name, emotional association |
| F14 | Date-specific culture/history/holiday fact | authoritative calendar/history source | Obscure date resonance, not obvious encyclopedia filler |
| F15 | Dreambank / dream-text motif | dreambank.net or approved text archive | Dream image/motif for interior-life prompts, or `UNAVAILABLE` |
| F16 | Freesound / field-recording texture | freesound.org or fallback public field-recording source | Concrete sound texture, license/source note, or `UNAVAILABLE` |
| F17 | Oblique Strategies | stoney.sb.org/eno/oblique.html | Exact phrase verbatim |
| F18 | Space Weather | services.swpc.noaa.gov/products/summary/solar-wind-speed.json + K-index feed | Solar wind speed, Kp, classification |
| F19 | Hacker News | news.ycombinator.com | Top 3 front-page titles - what builders are talking about |
| F20 | BBC World RSS | feeds.bbci.co.uk/news/world/rss.xml | Top 3 headlines - world emotional atmosphere |
| F21 | Public Domain Review | publicdomainreview.org | Era, subject, esoteric visual detail |
| F22 | Moon phase / Almanac | almanac.com/astronomy/moon or astronomy fallback | Moon phase, illumination if available, folklore/seasonal note |
| F23 | EarthCam / live place visual | earthcam.com or fallback public webcam/weather source | Place, light/weather/crowd texture, or `UNAVAILABLE` |
| F24 | Biodiversity Heritage Library / public-domain nature archive | biodiversitylibrary.org or fallback public-domain archive | Species/plate/material detail, or `UNAVAILABLE` |
| F25 | NOAA Buoy 46059 | ndbc.noaa.gov/data/realtime2/46059.txt | Wave height, period, wind, water temperature |

### OpenClaw Ledger Standard

`00_research_brief.md` must include one row for every slot `F01` through `F25`. Do not collapse ranges such as `F01-F03` or `F21-F25` in the saved run brief.

Use these statuses:
- `OK` - fetched and used.
- `NO DATA` - fetched successfully, but the feed was empty or below threshold.
- `UNAVAILABLE` - blocked, JS-gated, rate-limited, or inaccessible after a real attempt.
- `SCOPE-SKIPPED` - intentionally skipped because the declared run scope makes the slot irrelevant.

Stable feed, fresh values: the source ledger is the same every day; the extracted facts, timestamps, top items, and creative uses must be newly fetched for the run date. Reusing yesterday's values without a fresh fetch is a research failure.

The saved research brief must also include:
- A Tri-Source Summary.
- A 3+3 seeded split preview.
- 3-5 obscure theme-specific facts.
- 5 **EXISTENCE** prompts: interior-life questions songs can answer.
- Advisory learnings consulted, if any, as dispatch notes only.

---

## Tri-Source Methodology

Each daily piece integrates three sources:

**Source 1 - CONTENT / emotional stakes**
The world facts: quakes, APOD, Hacker News, BBC World, moon phase, solar weather, weather/ocean data, and other dated facts. Songs and images are responses to the day, not reportage.

**Source 2 - SONIC/AESTHETIC VOCABULARY**
Bandcamp Daily review language from F09-F10. Exact short phrases may be quoted in the brief, then translated into prompt behavior. This prevents generic genre labels.

**Source 3 - MATERIAL STRUCTURE**
NASA APOD image structure or a Public Domain Review artifact from F06-F07/F21. The physical or visual structure becomes a mandatory form rule:
- "Comet with long tail" -> long trailing outro.
- "3x1 tile panel with meander transitions" -> 3-section form with transitional bridges.
- "Bilateral wing venation" -> mirrored call-and-response.

Before writing any song or image prompt, declare:
1. Source 1 emotional axis.
2. Source 2 sonic/aesthetic vocabulary.
3. Source 3 material/form rule.

For music, the tri-source method feeds theme and form. It is not a lyric quota. At most one numeric/scientific fact may be sung per song, at the emotional hinge.

---

## Music Rules (Daily Runs Only)

- **24 songs required** - 6 concept pairs x 4 variations each.
- **Final selection:** best 6, with 3 from the accessible arm and 3 from the ambitious arm.
- **Emotional duality:** at least 1 AWE song and at least 1 INDIGNATION song.
- **Dual 3+3 constraint:**
  - Axis A - ACCESSIBLE vs AMBITIOUS: pairs 1-3 accessible, pairs 4-6 ambitious.
  - Axis B - NEWS vs EXISTENCE: max 3 NEWS pairs, min 3 EXISTENCE pairs.
- **Stanza economy:** vary stanza lengths intentionally.
- **Per-pair variation angles:** each pair derives its own 4 angles from its own concept.
- **Suno package:** dense music prompt, exclude prompt, EMO headers, female vocals by default, 70-120 sung lines.

## Image Rules (Daily Runs Only)

- **24 prompts -> top 12 -> top 6.**
- Figurative subject rule: every concept has a legible emotionally immediate primary subject.
- Noun-first, present-tense prompts, at least 80 words.
- Never start with: Create, Design, Make, Render, Generate, Depict, Show, Draw.
- NightCafe upload challenges use 3:4; otherwise default to 9:16 unless the brief says otherwise.
- Apply the Container Test and Action-Verb rule when the theme calls for them.

---

## Output Structure

```text
output/daily/YYYY-MM-DD/
|-- 00_research_brief.md
|-- 01_seed_lineage.md
|-- 02_golden_seed.md
|-- 03_orchestrator_panel_debate.md
|-- 04_orchestrator_metaprompt.md
|-- 05_orchestrator_pair_assignments.md
|-- 06_audio_handoff.md
|-- 06_vision_handoff.md
|-- RUN_STATE.md
|-- INDEX.md
|-- music/
|   |-- step00_aesthetics_and_genres.md
|   |-- ... steps 01-10/11 and song packages
|-- images/
    |-- step00_visual_aesthetics.md
    |-- ... image pair packages, TOP12, TOP6
```

Write `INDEX.md` last. It must include run scope, pair table, selected picks, key sources, and the run-health footer.

---

## Common Failure Modes & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Research hallucinated | Research delegated or inferred from memory | Phase 1 is done by the controller session itself with real fetches |
| Brief has fewer than 25 source rows | Collapsed F-slot ranges or omitted unavailable feeds | Preserve F01-F25; mark `UNAVAILABLE`, `NO DATA`, or `SCOPE-SKIPPED` |
| Songs delivered but no grounding | Tri-source bypassed | Declare Source 1/2/3 before any artifact writing |
| Set is a lecture | Too many NEWS pairs | Enforce max 3 NEWS and min 3 EXISTENCE |
| Arm imbalance | Global ranking picks 5+1 or 6+0 | Rank accessible and ambitious arms separately |
| Single agent timeout loses work | No save-out protocol | Save every step file as completed |
