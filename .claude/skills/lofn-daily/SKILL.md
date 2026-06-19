---
name: lofn-daily
description: Run the Lofn daily pipeline backed by Claude — fetch real-world facts, then generate the day's music (24 songs) and images (24→top 6) through the full Lofn pipeline with the daily rules (tri-source method, dual 3+3 constraint, emotional duality, library-only selection). Use for "daily run", "today's dailies", "run the daily pipeline", "do the daily drop", or a scheduled creative drop. Down-scalable for a quick test. Do NOT use for a single one-off competition piece (use `lofn`) or QA-only audits.
---

# Lofn Daily — Claude-backed daily run

The canonical recurring Lofn drop, ported to Claude. Faithful to `vault/DAILY_PIPELINE.md`. Where the original fired from an OpenClaw cron at 22:25 ET and delivered to Telegram, **you (Claude) run it on demand and present results in chat + save to disk.** The research is done by **this session itself with real fetches — never hallucinated, never delegated to a research subagent.**

```
PHASE 1  Research — fetch 20–25 verified real-world facts → 00_research_brief.md   (you, inline)
PHASE 2  Generate (in parallel):
           MUSIC  → lofn pipeline → 6 pairs × 4 = 24 songs → best 6
           IMAGE  → lofn pipeline → 24 prompts → top 12 → top 6
PHASE 3  QA (lofn-qa) → save under output/daily/YYYY-MM-DD/ → present the drop
```

---

## 0. Run scope (confirm before a big run)
The full daily run is **two complete pipelines** (24 songs + 24 images) — large. Pick a scope:
- **Full daily** — music (24→6) **and** image (24→12→6). The real thing.
- **Single modality** — just music **or** just image (still full cardinality).
- **Test slice** — 1 modality, **2 pairs × 2 variations**, library personality/panel, skip render-ranking. Proves the wiring fast and cheap. *(Good default when the user says "test the daily run".)*

State the chosen scope in the brief and in the run INDEX so QA knows the intended cardinality. Down-scaling is explicit, not silent.

---

## PHASE 1 — Research brief (you, inline, real fetches)
Use **WebFetch / WebSearch** (the Claude equivalents of the legacy `web_fetch`). Follow `skills/lofn-core/steps/00_music_research.md` (25-fact music research) and, when the image lane is in scope, the NightCafe-themed research in `skills/lofn-core/steps/00_research.md`. Fetch from the daily source table (`vault/DAILY_PIPELINE.md`) — at minimum:

| Code | Source | Extract |
|------|--------|---------|
| F1–F3 | NightCafe daily challenge (`nightcafe.studio/pages/daily-challenge`) | challenge #, theme, what wins (image lane) |
| F4–F5 | USGS quakes (`earthquake.usgs.gov/.../significant_day.geojson`) | magnitude, place, depth |
| F6–F7 | NASA APOD (`api.nasa.gov/planetary/apod?api_key=DEMO_KEY`) | title, first sentence, color/light, **image structure** |
| F8 | Poetry Foundation poem of the day | poet, most physical line |
| F9–F10 | Bandcamp Daily (`daily.bandcamp.com`) | album, genre tags, **exact sonic-texture quote** |
| F11 | Protein Data Bank molecule of the month | molecule, structural descriptor |
| F13 | Color API (`thecolorapi.com/id?hex=MMDD`) | hex, name, emotional association |
| F17 | Oblique Strategies (`stoney.sb.org/eno/oblique.html`) | exact phrase verbatim |
| F18 | Space weather (`services.swpc.noaa.gov/products/summary/solar-wind-speed.json`) | solar wind speed, Kp |
| F19 | Hacker News (`news.ycombinator.com`) | top 3 titles — what builders discuss |
| F20 | BBC World RSS (`feeds.bbci.co.uk/news/world/rss.xml`) | top 3 headlines — world's emotional temperature |
| F21–F25 | Public Domain Review, Almanac moon, NOAA buoy 46059 | esoteric visual detail, moon folklore, wave data |

Mark any JS-gated/unavailable source `UNAVAILABLE` and continue. Add 3–5 obscure theme-specific facts (not the obvious Wikipedia entry). Also write the 5 **EXISTENCE** prompts (interior-life questions songs can answer) from `00_music_research.md`.

**Save** `output/daily/YYYY-MM-DD/00_research_brief.md` with the Tri-Source Summary + the 3+3 seeded split (see below). Today's date is available in context — use it for the directory.

> ⛔ **One controller per date.** Before writing, check whether `output/daily/YYYY-MM-DD/` already has an in-progress run; if so, resume/stop it rather than racing a second lane.

---

## PHASE 2 — Generate (daily rules layered on the `lofn` pipeline)
For each in-scope modality, run the **`lofn`** pipeline (Phase 0 Golden Seed → Phase 1 3-panel orchestrator → modality steps → QA) using the research brief as `{input}`, **plus these daily-only rules:**

### Tri-Source Methodology (declare BEFORE writing any artifact)
Every daily piece integrates three sources; state them explicitly in the metaprompt and each pair brief:
- **Source 1 — CONTENT / emotional stakes:** today's world facts (quakes, APOD, F19 HN, F20 BBC, moon, solar weather). Songs/images are *resonance*, not reportage.
- **Source 2 — SONIC/AESTHETIC VOCABULARY:** the exact Bandcamp review language (F9–F10) imported into prompts — grounds the sound/look in something specific and real, not generic genre labels.
- **Source 3 — MATERIAL STRUCTURE:** the NASA APOD image structure (or a PDR artifact) translated into a **mandatory form rule** — e.g. "comet with long tail" → long trailing fade-out outro; "3×1 tile panel with meanders" → 3-section form with transitional bridges; "bilateral wing venation" → mirrored call-and-response.

### Dual 3+3 Constraint (set at pair-assignment time, Phase 1 step 5)
- **Axis A — ACCESSIBLE vs AMBITIOUS:** pairs 1–3 ACCESSIBLE, pairs 4–6 AMBITIOUS. Final top 6 = best 3 from each arm; rank **within each arm only** (never 5+1 or 6+0 by global score).
- **Axis B — NEWS vs EXISTENCE:** **max 3** pairs anchored to today's research/news; **min 3** pairs explore existence/interior-life/universal experience. All 6 on one theme = a lecture, not a record.

### Emotional Duality & diversity
- **≥1 AWE song and ≥1 INDIGNATION song** in the set.
- 6 different verse architectures / camera grammars across the 6 pairs (the standing distinctiveness rule). Vary stanza lengths intentionally.

### Library-only selection
For daily runs, **always select personality + panel from the existing libraries** (`personalities_index.md` / `panels_index.md`) — **no generation.** Freshly generated personalities over-fit the day's theme and lose the battle-tested DNA. (Generation is reserved for competition/Scientist-special runs.)

### Modality specifics
- **MUSIC** (`lofn-music`): 24 songs (6×4); each ≤1000-char two-field Suno prompt, female vocals default, EMO headers, 70–120 lines; `06_audio_handoff.md` carries 2 Golden Songs. Run music in parallel with image.
- **IMAGE** (`lofn-image`): 24 prompts → rank → **top 12 → top 6**; figurative legible primary subject (thumbnail test); noun-first present-tense ≥80 words; warm palette leads on NightCafe-style venues (INDIGNATION underperforms there — see `vault/COMPETITION_WORKFLOW.md`); aspect 3:4 for upload challenges else 9:16. Apply the **Container Test** (`COMPETITION_WORKFLOW.md`) and **Action-Verb rule** (action theme → cinematic wide, not portrait).

Run the two modalities concurrently (independent `lofn` runs writing to `music/` and `images/` subdirs). Each fans its 6 pairs out as parallel subagents per `.claude/skills/lofn/EXECUTION.md`.

---

## PHASE 3 — QA & deliver
1. **Cardinality checkpoint** before any selection: Step 05 ≥6 pairs; steps 06–10 have output for EVERY pair; ≥24 final prompts; (music) 6 enhanced packages. If any fail, rerun from the collapsed step — do not deliver.
2. Run **lofn-qa** on each modality (16-point Suno gate / Visual Somatic Gate). Verify the daily rules held: tri-source declared, 3+3 split honored, emotional duality present.
3. **Save** under `output/daily/YYYY-MM-DD/` (`00_research_brief.md`, `music/`, `images/`), final picks also as individual files per `skills/lofn-core/OUTPUT.md`; write the run INDEX last (env-scan summary, panel process, pairs table, selected picks, intended renderers).
4. **Present the drop** in chat: the research highlights, the tri-source declaration, the best 6 songs (paste-ready Suno packages) and top 6 image prompts, with the panel decisions and "why these win." No render calls — emit text; the user renders (Suno / Flux / Lyria).

---

## Failure modes (from `vault/DAILY_PIPELINE.md`, adapted)
| Problem | Fix |
|---------|-----|
| Research hallucinated | Phase 1 is done by THIS session with real WebFetches — never a research subagent |
| Songs lack grounding | Declare Source 1/2/3 before any song is written |
| Set is a lecture | Enforce Axis B (min 3 EXISTENCE pairs) |
| Arm imbalance | Rank within ACCESSIBLE / AMBITIOUS arms separately (3+3) |
| Work lost mid-run | Save every step file as it completes; one controller per date; checkpoint after each wave |

## Scheduling (optional)
On-demand here. To make it recur, use the `schedule` skill (cloud routine) or `/loop` — the legacy 22:25-ET cron is OpenClaw-specific and does not apply.

**Reference:** `vault/DAILY_PIPELINE.md` · `vault/COMPETITION_WORKFLOW.md` (competition variant + Masterpiece Monday learnings) · `skills/lofn-core/steps/00_music_research.md` · `skills/lofn-core/steps/00_research.md` · the `lofn` / `lofn-music` / `lofn-image` / `lofn-qa` skills.
