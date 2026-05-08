# Research Step: Tri-Source Fetch

## Task

Fetch real-world data from at least 5 of these sources. Use web_fetch or oxylabs_web_fetch for each.

**Required sources (at least 3):**
1. **USGS Earthquake Feed** — `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson`
2. **NASA APOD** — `https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY`
3. **Bandcamp Daily** — `https://daily.bandcamp.com/`
4. **Space Weather** — `https://services.swpc.noaa.gov/products/summary/solar-wind-speed.json`

**Optional sources (pick at least 2):**
5. Oblique Strategies — `https://stoney.sb.org/eno/oblique.html`
6. Poetry Foundation poem of the day
7. Moon phase / astronomical data
8. BBC World News headlines
9. Hacker News top stories
10. Any other source that feels relevant to the creative brief

**Theme-specific research:**
Also search the web for 3-5 facts about the challenge theme that will inform the creative direction. These should be obscure, surprising, or culturally rich — not the obvious Wikipedia entry.

## Output Format

Write a research brief as markdown with this structure:

```markdown
# Research Brief — [Theme] — [Date]

## F1-F3: Challenge Context
- F1: Challenge theme and rules
- F2: Allowed models / tools
- F3: What wins historically in this theme

## F4-F5: Seismic
- Any significant earthquakes in the last 24 hours
- Pattern or notable cluster

## F6-F7: NASA APOD
- Today's image description
- Creative implication

## F8-F9: Bandcamp Daily
- Album or feature of the day
- Sonic vocabulary that could enter the work

## Theme-Specific Research
- 3-5 obscure/surprising facts about the theme
- Cultural or historical context that resists the obvious

## Tri-Source Summary
- Source 1 (Challenge): key creative direction
- Source 2 (Visual/Aesthetic): what enters the work from research
- Source 3 (Material Structure): form rule derived from research
```

Save to: `{output_dir}/00_research_brief.md`
