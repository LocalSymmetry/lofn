---
name: lofn-daily
description: Run the Lofn daily pipeline backed by Codex — fetch real-world facts, then generate the day's music (24 songs) and images (24→top 6) through the full Lofn pipeline with the daily rules (tri-source method, dual 3+3 constraint, emotional duality, library-only selection). Use for "daily run", "today's dailies", "run the daily pipeline", "do the daily drop", or a scheduled creative drop. Down-scalable for a quick test. Do NOT use for a single one-off competition piece (use `lofn`) or QA-only audits.
---

# Lofn Daily — Codex-backed daily run

> **⚖️ AUTHORITY (2026-07-01):** the `.claude/skills/` twin of this skill is the CANONICAL policy source; this Codex mirror binds to it and to `.agents/skills/lofn/EXECUTION.md` §8 (Policy Deltas — golden-output quarantine, no-skip/NON-CANONICAL, itemized packet, per-pair variation angles, judge separation, the publish bar, gate mid-bands). On any disagreement, the `.claude` file wins.

The canonical recurring Lofn drop, ported to Codex. Faithful to `vault/DAILY_PIPELINE.md`. Where the original fired from an OpenClaw cron at 22:25 ET and delivered to Telegram, **you (Codex) run it on demand and present results in chat + save to disk.** The research is done by **this session itself with real fetches — never hallucinated, never delegated to a research subagent.**

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
Use **WebFetch / WebSearch** (the Codex equivalents of the legacy `web_fetch`). Follow `skills/lofn-core/steps/00_music_research.md` (25-fact music research) and, when the image lane is in scope, the NightCafe-themed research in `skills/lofn-core/steps/00_research.md`. Fetch from the daily source table (`vault/DAILY_PIPELINE.md`) — at minimum:

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

> ⛔ **One controller per date.** Before writing, check whether `output/daily/YYYY-MM-DD/` already has an in-progress run; if so, **resume** it from the RUN_STATE manifest (§ "Run-state manifest & resume") rather than racing a second lane or re-running passed pairs.

### Advisory learnings note (dispatch brief only — NEVER the ICB)
Before dispatching either modality, **tag-walk `vault/COMPETITION_LEARNINGS.md`** (and `vault/LESSONS_INDEX.md` if it exists) for the **3–5 entries that intersect THIS run's theme/venue/modality** — e.g. a container/object theme pulls the Container Test, a NightCafe image lane pulls the warm-palette/anti-austerity lessons, a portrait theme pulls the portrait shifts. Surface them **as ADVISORY NOTES in the dispatch brief / Phase-0 reasoning ONLY.** Hard rules:
- **NEVER injected into the ICB / `CREATIVE_CONTEXT.md`.** The ICB stays read-only and lesson-free. These notes live in the brief the coordinator reasons over, not in the verbatim block every subagent receives.
- Each note carries its **confidence %** (from the entry) and is run through the mandatory **"would this have hurt our best past entry?"** gate before it is allowed to influence anything. A lesson that would have hurt a past win is dropped, not applied.
- **Venue/modality-scoped:** an image-venue lesson must NOT leak into the music lane; a NightCafe-voting lesson must NOT leak into non-competition runs.
- **Advisory, never a hard constraint.** A note can inform the pair brief's framing; it can never auto-reject a candidate or become a gate. Promotion to a hard constraint is a **human** decision.
- **Triggered-INDIGNATION is EXEMPT from suppression.** A lesson such as "INDIGNATION underperforms on NightCafe" may inform image-venue *selection* advice, but it must NEVER suppress an INDIGNATION piece the panel deliberately chose, and never touches the music lane or the ≥1-INDIGNATION duality rule.

State in the brief: "Advisory learnings consulted: <N entries, tags>; INDIGNATION exempt; advisory-only." If zero entries intersect, say so. Write-back of one curated entry per shipped/selected piece happens in `lofn-qa` / Phase 3 — not here.

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
- **MUSIC** (`lofn-music`): 24 songs (6×4); each ≤1000-char two-field Suno prompt, female vocals default, EMO headers, 70–120 lines; `06_audio_handoff.md` carries the 2 Golden Song **names** + the GOLDEN MOVE block (⛔ never payloads — `EXECUTION.md` §8.1). Run music in parallel with image.
- **IMAGE** (`lofn-image`): 24 prompts → rank → **top 12 → top 6**; figurative legible primary subject (thumbnail test); noun-first present-tense ≥80 words; warm palette leads on NightCafe-style venues (INDIGNATION underperforms there — see `vault/COMPETITION_WORKFLOW.md`); aspect 3:4 for upload challenges else 9:16. Apply the **Container Test** (`COMPETITION_WORKFLOW.md`) and **Action-Verb rule** (action theme → cinematic wide, not portrait).

Run the two modalities concurrently (independent `lofn` runs writing to `music/` and `images/` subdirs). Each fans its 6 pairs out as parallel subagents per `.agents/skills/lofn/EXECUTION.md`.

> ### ⚙️ Concurrency: cap-and-stagger (do NOT run all 12 chains at once)
> The full daily is **two pipelines × 6 pairs = 12 concurrent chains** — enough to blow the tool/context budget if fired together. **Cap and stagger, don't serialize:**
> - **Cap the in-flight pair-subagents** (default ~6 at a time, not all 12). Launch one modality's 6-pair wave, then the other's, OR interleave in two staggered batches — never one 12-block message.
> - The cap is a **throttle, not a serializer:** keep enough parallelism that Lofn's throughput win survives; the goal is bounded peak load, not one-at-a-time.
> - A single-modality run (6 pairs) fans out normally; the cap only bites when both pipelines are live.
> - This is the daily's binding of EXECUTION.md §2 fan-out — the per-modality pair-parallelism rule is unchanged; only the *cross-modality* peak is capped.

---

## PHASE 3 — QA & deliver
1. **Cardinality checkpoint** before any selection: Step 05 ≥6 pairs; steps 06–10 have output for EVERY **non-quarantined** pair; ≥24 final prompts; (music) 6 enhanced packages. Rebuild the RUN_STATE manifest from disk first (§ "Run-state manifest & resume"), then read it — don't re-crawl by hand. **A `quarantined` pair is a named, non-fatal hole, not a cardinality failure: do NOT feed its artifact downstream, do NOT block delivery on it — surface "N of 6 broke open" and continue with the survivors.** If a *non-quarantined* pair is missing output, rerun from the collapsed step — do not deliver.
2. Run **lofn-qa** on each modality (16-point Suno gate / Visual Somatic Gate). Verify the daily rules held: tri-source declared, 3+3 split honored, emotional duality present.
3. **Save** under `output/daily/YYYY-MM-DD/` (`00_research_brief.md`, `music/`, `images/`), final picks also as individual files per `skills/lofn-core/OUTPUT.md`; write the run INDEX last (env-scan summary, panel process, pairs table, selected picks, intended renderers, **and the 3-field run-health footer** — see § "Run-state manifest & resume").
4. **Present the drop** in chat: the research highlights, the tri-source declaration, the best 6 songs (paste-ready Suno packages) and top 6 image prompts, with the panel decisions and "why these win." No render calls — emit text; the user renders (Suno / Flux / Lyria).

---

---

## Run-state manifest & resume (the outer loop)

The daily fans out ~60 artifacts across two pipelines; the human should never reconstruct run state by eye. **Disk is the only authority** — the manifest is a *cache the coordinator rebuilds by stat-ing files*, never a hand-asserted second truth. If manifest and disk disagree, re-derive from disk.

### RUN_STATE manifest (consume & rebuild)
Maintain `output/daily/YYYY-MM-DD/RUN_STATE.md` (or `.json`). **Rebuild it by stat-ing the run dir after every wave**, and write it as the **LAST** action of each step so it never claims an artifact that isn't on disk. Per artifact record:

`{ step, pair, modality, canonical_path, exists, byte_size, sha, gate_verdict, attempt_count, status: pending | done | quarantined }` + the **ICB sha** for the run.

- This is metadata only — no creative payload, no ICB summary lives here.
- **Resume:** on re-entry for an existing date, rebuild from disk, then run only the `pending` pairs/steps. **Never regenerate a `done` artifact** (it may be paid image/video work) and never re-run a passed gate. A reply that says "let me write this" counts as `pending` until the file is on disk.
- The mechanics mirror `.agents/skills/lofn/EXECUTION.md` §6 (checkpoint) — this section is the daily's binding of it; keep the existing one-line human-readable note too.

### Quarantine terminal (before QA)
A pair that fails the **same** gate on its 3rd repair attempt is terminal:
- Keep the existing **max-3-attempt** repair loop (EXECUTION.md §4). Before the 3rd attempt fires, allow a **cognitive-grace near-miss pass** (e.g. buffer a 5002-char lyrics field down to spec) as attempt-2.5 — a rescuable near-miss must not be quarantined.
- **No-progress check:** compare the *specific failed gate's measured value* (chars, word-count, sung-line count) across attempts, NOT raw bytes — an intentional revision must still count as progress. If the failed gate's value does not move between attempts, stop early.
- On terminal failure, mark the pair `status: quarantined` in the manifest, **do NOT consume its artifact downstream**, and emit **"N of 6 pairs broke open at step X (gate: <name>)" BEFORE QA**. Quarantine forces a human acknowledgement — it is a named, non-fatal outcome, never a silent 5-shipped-as-6.

### Run-level circuit breaker (same-gate correlated failure)
Quarantine handles a *lone* broken pair. The breaker handles a **systemic** one: when **one named gate** fails across many pairs (e.g. a stale rule failing most of the 24 artifacts), **STOP the daily run and surface the named blocker for human eyes** instead of letting 24 × 3 attempts hammer the same broken constraint.
- **Scoped to SAME-gate correlation**, not aggregate fail-count: independent one-off pair failures still retry locally and quarantine normally. The breaker only trips when the *same* gate is the failing one across a threshold of pairs (the systemic signal).
- When it trips: name the failing gate, report which pairs hit it, and HALT — do not auto-repair. This is a coordinator prose rule + a disk re-read of the manifest's `gate_verdict` column, never a daemon. Threshold tuned conservatively so a genuinely-hard batch isn't aborted; surface the gate, never a silent kill.

### Run-health footer (3 fields)
Append a terse **3-field** line to the run INDEX (and present it in chat): **pairs shipped / pairs quarantined / total gate-retries** (per modality if both ran). Three fields only — a status line for the human, not a metrics culture. One explicit "escalate to human" outcome (the breaker, or any quarantine) is always surfaced, never hidden.

---

## Failure modes (from `vault/DAILY_PIPELINE.md`, adapted)
| Problem | Fix |
|---------|-----|
| Research hallucinated | Phase 1 is done by THIS session with real WebFetches — never a research subagent |
| Songs lack grounding | Declare Source 1/2/3 before any song is written |
| Set is a lecture | Enforce Axis B (min 3 EXISTENCE pairs) |
| Arm imbalance | Rank within ACCESSIBLE / AMBITIOUS arms separately (3+3) |
| Work lost mid-run | Save every step file as it completes; one controller per date; rebuild the RUN_STATE manifest from disk after each wave and resume only `pending` artifacts |
| A single pair keeps failing | Max-3-attempt loop + attempt-2.5 grace pass + no-progress halt → mark `quarantined`, surface "N of 6 broke open," do NOT consume its artifact, continue with survivors |
| A stale rule fails everything | Run-level circuit breaker on SAME-gate correlated failure → STOP and name the gate for the human instead of 24×3 wasted retries |
| Budget blown by parallelism | Cap-and-stagger: never fire all 12 chains at once (~6 in-flight); throttle, don't serialize |

## Scheduling (optional)
On-demand here. To make it recur, use the `schedule` skill (cloud routine) or `/loop` — the legacy 22:25-ET cron is OpenClaw-specific and does not apply.

**Reference:** `vault/DAILY_PIPELINE.md` · `vault/COMPETITION_WORKFLOW.md` (competition variant + Masterpiece Monday learnings) · `vault/COMPETITION_LEARNINGS.md` (advisory dispatch-brief lessons; never the ICB) · `.agents/skills/lofn/EXECUTION.md` §2 fan-out / §4 gates / §6 checkpoint (the manifest, quarantine, and breaker bind to these) · `skills/lofn-core/steps/00_music_research.md` · `skills/lofn-core/steps/00_research.md` · the `lofn` / `lofn-music` / `lofn-image` / `lofn-qa` skills.
