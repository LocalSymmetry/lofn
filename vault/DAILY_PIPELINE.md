# DAILY_PIPELINE.md — Lofn Daily Run Architecture
*Last updated: 2026-04-20*

## Overview

Every day at 22:25 ET, the Lofn cron pipeline fires. It produces:
- **6 songs** (Suno prompts + lyrics, via lofn-audio)
- **6 image prompts** rendered (top 6 of 24, via lofn-vision)
- Both delivered to Telegram

## Architecture

```
CRON (main session, 22:25 ET)
  │
  ├── PHASE 1: Fetch 20+ real-world facts
  │     └── Save to output/daily/YYYY-MM-DD/00_research_brief.md
  │
  └── PHASE 2: Spawn pipelines in parallel
        ├── lofn-audio → 6 songs → Telegram
        └── lofn-vision → 6 image prompts rendered → Telegram
```

**Critical rules:**
- Phase 1 is done by the CRON SESSION ITSELF — never delegated to lofn-research (hallucinator)
- lofn-audio is spawned DIRECTLY from the cron — never via depth-2 chains
- Every step saved to disk as it completes (save-out protocol)

---

## Phase 1: 20-Source Research Brief

The cron fetches from these real-world sources and saves verified facts:

| Code | Source | URL | What We Extract |
|------|--------|-----|-----------------|
| F1–F3 | NightCafe Challenge | nightcafe.studio/pages/daily-challenge | Challenge #, theme, what wins |
| F4–F5 | USGS Earthquake | earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_day.geojson | M+location, depth, tsunami, felt |
| F6–F7 | NASA APOD | api.nasa.gov/planetary/apod?api_key=DEMO_KEY | Title, first sentence, color/light descriptor |
| F8 | Poetry Foundation | poetryfoundation.org/poems/poem-of-the-day | Title, poet, most physical line |
| F9–F10 | Bandcamp Daily | daily.bandcamp.com | Album, artist, genre tags, sonic texture quote |
| F11 | Protein Data Bank | rcsb.org/pages/educational_resources/molecule_of_the_month | Molecule, function, structural descriptor |
| F13 | Color API | thecolorapi.com/id?hex=MMDD | Hex, color name, emotional association |
| F17 | Oblique Strategies | stoney.sb.org/eno/oblique.html | Exact phrase verbatim |
| F18 | Space Weather | services.swpc.noaa.gov/products/summary/solar-wind-speed.json | Solar wind speed, Kp, classification |
| F19 | **HackerNews** | **news.ycombinator.com** | **Top 3 front-page titles — what builders are talking about** |
| F20 | **BBC World RSS** | **feeds.bbci.co.uk/news/world/rss.xml** | **Top 3 headlines — world emotional atmosphere** |
| F21 | Public Domain Review | publicdomainreview.org | Era, subject, esoteric visual detail |
| F22 | Old Farmer's Almanac | almanac.com/astronomy/moon | Moon phase, folklore |
| F25 | NOAA Buoy 46059 | ndbc.noaa.gov/data/realtime2/46059.txt | Wave height, period, water temp |

*Note: F12 (Radio Garden), F15 (Dreambank), F16 (Freesound), F19 (Flightradar), F22 (arXiv), F23 (EarthCam), F24 (BHL) frequently unavailable due to JS requirements. Mark UNAVAILABLE and continue.*

**F19 and F20 (news headlines) were added 2026-04-20.** They provide the world's emotional temperature — what humans are scared of, marveling at, or angry about right now. Use as raw emotional material, not just topic prompts.

---

## Tri-Source Methodology (Music Pipeline)

Each daily music session builds songs from THREE integrated sources:

**Source 1 — CONTENT (emotional stakes)**
The world facts: earthquake, comet, headlines (F19 HackerNews, F20 BBC World), moon phase, solar weather.
This is what the world is carrying today. Songs are responses to it — not reportage, but resonance.

**Source 2 — SONIC VOCABULARY**
Bandcamp Daily review language (F9–F10). The exact words from the review are imported into Suno prompts.
This grounds the sound in something specific and real, not generic genre labels.

**Source 3 — MATERIAL STRUCTURE**
NASA APOD image structure OR the PDR artifact (F6–F7, F21).
The physical/visual structure of today's image is translated into a mandatory song form rule:
- e.g. "Comet with long tail" → songs must have a long trailing outro that fades like light
- e.g. "3×1 Delft tile panel with meander transitions" → 3-section song form with transitional bridges
- e.g. "Bilateral wing venation" → call-and-response structure with mirrored verses

**Before writing any song, lofn-audio declares:**
1. Source 2 sonic vocabulary (exact Bandcamp terms)
2. Source 3 → song form rule (how does the image structure become musical architecture?)
3. Source 1 emotional axis (what is today's world carrying?)

---

## Music Rules (Daily Runs Only)

- **6 songs minimum** — concept pairs × 1 song each
- **Emotional Duality** — min 1 AWE song, min 1 INDIGNATION song
- **3+3 Split** — max 3 news-anchored, min 3 existence-exploring
- **Stanza Economy** — vary stanza lengths intentionally, never uniform
- Each song: Suno Music Prompt (≤1000 chars, female vocals, [EMO: tags]) + Lyrics Prompt (≥50 lines)

## Image Rules (Daily Runs Only)

- **24 prompts → top 12 rendered → top 6 delivered**
- Figurative Subject Rule — every concept has a legible emotionally immediate primary subject
- Noun-first, present-tense prompts, ≥80 words
- Never start with: Create/Design/Make/Render/Generate/Depict/Show/Draw

---

## Output Structure

```
output/daily/YYYY-MM-DD/
├── 00_research_brief.md       ← verified facts (F1–F25)
├── 01_expanded_options.md     ← creative expansion (if used)
├── music/
│   ├── 00_aesthetics.md
│   ├── 01_essence.md
│   ├── ... (steps 00–10)
│   ├── song_01_*.md
│   └── song_06_*.md
└── images/
    ├── 00_aesthetics.md
    ├── ... (steps 00–10)
    └── [rendered image files]
```

---

## Common Failure Modes & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Research hallucinated | lofn-research (Gemini) in the chain | Phase 1 done by cron itself only |
| Audio agent returns in 9s | Spawned at depth-2 | Always spawn lofn-audio directly from cron |
| Subagents timeout ~1m | Wrong model (claude-sonnet instead of gpt-5.4) | lofn-audio uses gpt-5.4 per config |
| Songs delivered but no research grounding | Music-spine pipeline bypassed tri-source | Tri-source declared before any song writing |
| Single agent timeout loses all work | No save-out | Save every step file as completed |
