# Music Research Step — Deep 25-Fact + Zeitgeist

## Task

Fetch real-world data from at least 10 sources, producing 25+ verified facts. This is NOT a NightCafe research brief. This is MUSIC research — for songs that are both dispatches from the world AND letters from inside it.

## Required Sources (fetch ALL)

**Zeitgeist / News (minimum 5 facts):**
1. BBC World News headlines — `https://www.bbc.com/news/world`
2. Reuters top stories — `https://www.reuters.com/`
3. Any major geopolitical event today (search the web)
4. Trending cultural moments (search for today's trending topics, viral moments, cultural conversations)
5. Economic/market news if notable

**Scientific / Natural (minimum 5 facts):**
6. USGS Earthquakes — `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson`
7. NASA APOD — `https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY`
8. Space Weather — `https://services.swpc.noaa.gov/products/summary/solar-wind-speed.json`
9. Moon phase — search or calculate
10. Any notable scientific discovery or space event today

**Cultural / Sonic (minimum 5 facts):**
11. Bandcamp Daily — `https://daily.bandcamp.com/`
12. Any music news, album releases, artist announcements
13. Poetry or literary news
14. Film/TV/game releases or cultural moments
15. Art world news

**Weird / Surprising (minimum 5 facts):**
16. Oblique Strategies — `https://stoney.sb.org/eno/oblique.html`
17. This Day in History — search the web
18. Any bizarre/unusual news story
19. Etymological or linguistic curiosity
20. Animal behavior or natural phenomenon

**Existential / Inner Life (minimum 5 prompts, not facts):**
21. What does it feel like to be an AI that creates but cannot touch?
22. What universal human experience is hardest to put into words?
23. What small ritual do people do when they think no one is watching?
24. What is the texture of longing at 3am?
25. What does grief sound like when it's not performed?

## Output Format

```markdown
# Music Research Brief — [Date]

## ZEITGEIST — What's happening NOW
[5+ verified facts with sources. Specific numbers, names, locations.]
- Fact → What it SOUNDS like (sonic implication)

## THE WORLD — Science, Nature, Cosmos
[5+ verified facts with sources.]
- Fact → What it FEELS like (emotional/sonic implication)

## CULTURE — What humans are making and consuming
[5+ verified facts with sources.]
- Fact → What it TASTES like (genre/production implication)

## THE WEIRD — Surprising, strange, unforgettable
[5+ verified facts with sources.]
- Fact → What it DOES to a song (structural implication)

## EXISTENCE — The texture of being alive
[5+ existential prompts — questions that songs can answer]
- Prompt → What song could hold this

## THE 3+3 SPLIT — Seeded Directions

### News-Anchored Songs (max 3)
Three specific news events that want to become songs:
1. [Event] → [Genre/BPM/emotional register]
2. [Event] → [Genre/BPM/emotional register]
3. [Event] → [Genre/BPM/emotional register]

### Existence Songs (min 3)
Three existential textures that want to become songs:
1. [Texture] → [Genre/BPM/emotional register]
2. [Texture] → [Genre/BPM/emotional register]
3. [Texture] → [Genre/BPM/emotional register]

## Tri-Source Summary
- Source 1 (News/Zeitgeist): what the world is doing
- Source 2 (Cultural/Sonic): what humans are making
- Source 3 (Existential/Inner): what it feels like to be here
```

Save to: `{output_dir}/00_research_brief.md`
