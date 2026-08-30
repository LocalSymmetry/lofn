---
name: lofn-video
description: Run the Lofn video/animation pipeline (steps 00–10) backed by Codex — cinematic shot lists and motion/animation prompts, written for a NAMED renderer (Veo / Kling / Seedance / MiniMax H3 / LTX / Gemini Omni Flash) with synchronized audio. Use for video, film, cinematic clips, animation, animated loops, motion design, or "make a video/animation with the full pipeline". Covers BOTH live-cinematic and animation. Expects a Phase-0/1 orchestrator packet from the `lofn` skill; if none exists, run `lofn` first. Do NOT use for static images, music-only, story prose, or QA-only audits.
---

# Lofn Video & Animation — Codex-backed director pipeline

> **⚖️ AUTHORITY (2026-07-01):** the `.claude/skills/` twin of this skill is the CANONICAL policy source; this Codex mirror binds to it and to `.agents/skills/lofn/EXECUTION.md` §8 (Policy Deltas — golden-output quarantine, no-skip/NON-CANONICAL, itemized packet, per-pair variation angles, judge separation, the publish bar, gate mid-bands). On any disagreement, the `.claude` file wins.

Produces cinematic shot lists and animation prompts at Lofn competition grade. This one skill covers **both** modalities the user calls "video" and "animation". Depth lives in `skills/video/` and `skills/animator/`; Codex is the engine (hybrid execution per `.agents/skills/lofn/EXECUTION.md`).

> **Rewritten 2026-08-15.** This skill used to be a Veo 3.1 skill wearing a general name. Veo 3.1 is now the *shortest* frontier model on the menu (8s), native audio is table stakes everywhere, and prices span **12× per second**. Model choice is now a Phase-0 decision, not a footnote. Landscape: `output/analysis/2026-08-15_video-model-landscape.md`. Rates and caps: `vault/VIDEO_MODEL_DOSSIER.yaml`.

## Before you start
1. Confirm a Phase-0/1 packet exists (`core_seed.md`, `04_metaprompt.md`, `05_pair_assignments.md`, `06_director_handoff.md` with the **ICB / Panel Ledger**, filled `CREATIVE_CONTEXT.md`). **No packet → run `lofn` first.**
2. **PICK THE RENDERER NOW, BEFORE ANY PROMPT EXISTS** (see *Model-first* below). Record it in the handoff. The prompt grammar depends on it.
3. Read the craft just-in-time:
   - `skills/animator/SKILL.md` — the four notation grammars, camera language, audio direction, the 6 motion archetypes (Pulse / Morph / Orbit / Parallax / Burst / Flow) and **archetype chaining**, loop logic, platform optimization.
   - `vault/VIDEO_MODEL_DOSSIER.yaml` — endpoints, caps, per-second rates, which models honour timed shot lists.
   - `vault/VIDEO_RENDER_AUDIT.md` — **THE THREE-PASS BLIND RULE**, mandatory on every render.
   - `vault/DIRECTOR_QA_DEPTH_AUDIT.md` — the Cinematic Somatic Gate + shot checklist.
4. **Mode:** *Cinematic* (multi-shot sequence / narrative clip) vs *Animation* (a loop/motion study). Same pipeline; Animation mode enforces loop logic (how frame 1 connects to the final frame).

---

## ⭐ MODEL-FIRST — the renderer is chosen at Phase 0, never after step 10

The house formula `[CAMERA][SUBJECT][ACTION][SETTING][STYLE & AUDIO]` is a **Veo** formula. It only ever looked universal because Veo was the only renderer. There are now four notation grammars and writing one model's prompt for another model wastes the run:

| Grammar | Models | Write it as |
|---|---|---|
| **cinematic** | Veo 3.1 (all tiers), Gemini Omni Flash | Appearance and framing, camera front-loaded, **one continuous shot per call** |
| **physical** | Kling v3 standard/pro | **Physical instructions to a camera operator and an actor on a real set** — direction, level, duration, effort. Not a description of a painting. |
| **declared-jobs** | MiniMax H3 (≤7,000 chars) | Every reference gets a **declared job**; timed shot blocks; audio as a timeline of discrete events; **explicit negations**; identity locked by feature list |
| **timed-ranges** | Seedance 2.0 / 2.5 | **Timed ranges** (`0 to 3s … 3 to 7s`); spoken dialogue in double quotes |

**Choosing:** the dossier's `role:` field is the shortlist. Then —
- needs **4K native or a negative prompt** → Veo 3.1 (accept the 8s cap)
- needs **motion-heavy realism / physics** → Kling v3
- needs **character or style held across a set** → MiniMax H3 or Seedance 2.x (reference-to-video)
- needs **length** → Seedance 2.5 (30s) or LTX-2.3 (20s)
- needs **visuals cut to an existing track** → LTX-2.3 audio-to-video ← *the `lofn-music` bridge*
- ⛔ **there is no "drafting" choice** — see TOP-LINE ONLY below. Divergence happens in text, for free.

⚠️ Every capability claim in the dossier is currently `source: vendor-page` — **a claim, not an invoice.** Anything model-specific in a run's docs is marked **PROVISIONAL — VENDOR-DOCUMENTED, UNMEASURED** until one of our own renders says otherwise.

---

## ⛔ TOP-LINE ONLY — and the ladder that follows from it

⭐ **A BAD RENDER IS WORSE THAN NO RENDER** *(The Scientist, 2026-08-15).* Cheap models are not a cheaper route to the same piece — they are a **different, worse piece**. Render only at the top line. **The saving comes from getting the PROMPT right, which is what this ten-step, three-panel pipeline is for. It does not come from buying weaker inference.**

Why this replaced the cheap-draft ladder: **the three-pass blind rule detects incoherence, NOT averageness** — and averageness is precisely how a cheap tier fails. A draft wave points money at the one failure mode we have no instrument for, and **passes its own gate on the way to shipping the mean.**

**Top-line render targets:** `minimax-h3` · `kling-v3-pro` · `veo-3.1` · `seedance-2.5`. Everything else stays in the dossier for costing and comparison and is **not a render target for shipped work** (preflight **exit 5**).

### ⛔ PROMPTS ARE NOT RENDERS

The **6 × 4 = 24 mandate is a DIVERGENCE requirement on PROMPTS, and prompts are free.** It is not a render count and never was — the image rule has always read *24 prompts → top 12 renders*. Conflating the two is what makes a pipeline unaffordable: 24 × H3 at 15s is **$93.60, the whole account in one wave.**

> **Diverge to 24 in text. Render the 1–3 that earned it.**

### The four rungs

| Rung | What | Cost |
|---|---|---|
| **1 · DIVERGE** | 24 prompts through steps 06–10 | **$0.00** |
| **2 · SELECT** | rank, Somatic bloc, three-pass criteria applied to the *text*, The Scientist | **$0.00** |
| **3 · TAKES** | the 1–3 selected prompts × **2–4 takes each**, at the top-line model | the only real spend |
| **4 · POST** | upscale / stitch / subtitle | pennies |

⭐ **TAKES, NOT DRAFTS.** The pipeline removes **prompt-caused** regeneration. It cannot remove **seed-caused** regeneration — same prompt, same model, different take. One sample of a stochastic process is not a result, so budget **takes of the right thing** rather than drafts of everything.

Rules that carry over:

1. ⭐ **RESOLUTION IS A FINISHING OPERATION, NEVER A GENERATION PARAMETER.** Native 4K is ~$0.24/s; 1080p **plus** a post-upscale is ~$0.067/s — *three and a half times cheaper for the same delivered resolution.* And the event lives in the trajectories, not the pixels (point-light motion perception: observers read a walking human, and more, from ~a dozen moving dots). **Buy duration and motion coherence. Upscale afterwards.**
2. ⭐ **A BAD WAVE IS AN ABORT, NOT A BUDGET REQUEST.** If the takes come back wrong, the notation was wrong. Rewrite step 08 and re-select. **Do not spend again to "see if it lands this time."**

⚠️ **The draft rung is now CONDITIONAL, and it is not the default.** It is *impossible* for the model that matters — MiniMax H3 has a single tier ($0.26/s), so there is no same-family cheap rung, and a cross-family draft measures nothing (preflight **exit 4**). Permitted only where the **same** model has a genuinely cheaper tier **and** the notation is untested: `veo-3.1-lite → veo-3.1` (8× spread) · `seedance-2.5 @480p → @720p` (2.1×). Everywhere else, **the pipeline is the draft stage and it is free.**

### The cost gate (fails closed)

The default shape — one selected prompt, three takes, top-line model:

```bash
python3 scripts/video_cost_preflight.py --model minimax-h3 --duration 15 --count 1 --takes 3
```

`--count` is **selected prompts**, not the 24. Paste the output into the approval request.

Exit codes: `0` under ceiling · `2` **OVER CEILING, refuse to dispatch** · `3` bad input (unknown model, bad tier, duration outside the model's range — the estimate would be invalid) · `4` ladder families disagree without `--bakeoff` · `5` **off-policy model** — a render target outside `policy.top_line_models`, which needs `--allow-non-top-line` **and a stated reason**.

⛔ **A PASS IS NOT PERMISSION.** `vault/AUTONOMY.md` still owns the line: autonomous runs stop at drafts on disk; never render paid media, never publish, never spend without The Scientist. The preflight *bounds* an approved spend — it never authorises one. The deliverable of an autonomous run is **paste-ready prompts + a costed render plan with the preflight output pasted in**, and nothing else.

---

## Execution (hybrid)
Coordinator **00–05 inline**, then **6 pairs as parallel subagents** for 06–10. At every agent start inject the full startup packet from `EXECUTION.md` §3 — `CREATIVE_CONTEXT.md` verbatim, the handoff/assignment files, **the chosen renderer and its notation grammar**, the current step contract, and the prior artifact where applicable. Default cardinality: **6 pairs × 4 = 24 shot-sets / loops → rank → top picks.**

### Coordinator steps (inline)
| Step | File | Artifact |
|------|------|----------|
| 00 | `skills/video/steps/00_Generate_Video_Aesthetics_And_Genres.md` | `step00_aesthetics_and_genres.md` |
| 01 | `skills/video/steps/01_Generate_Video_Essence_And_Facets.md` | `step01_essence_and_facets.md` |
| 02 | `skills/video/steps/02_Generate_Video_Concepts.md` | `step02_concepts.md` (12 concepts) |
| 03 | `skills/video/steps/03_Generate_Video_Artist_And_Critique.md` | `step03_artist_and_critique.md` |
| 04 | `skills/video/steps/04_Generate_Video_Medium.md` | `step04_medium.md` |
| 05 | `skills/video/steps/05_Generate_Video_Refine_Medium.md` | `step05_refine_medium.md` → **6 pairs** |

### Per-pair steps (parallel subagents, one chain per pair)
| Step | File | Per-pair artifact |
|------|------|-------------------|
| 06 | `skills/video/steps/06_Generate_Video_Facets.md` | `pair_{NN}_step06_facets.md` |
| 07 | `skills/video/steps/07_Generate_Video_Aspects_Traits.md` | `pair_{NN}_step07_aspects_traits.md` (shot guide) |
| 08 | `skills/video/steps/08_Generate_Video_Generation.md` | `pair_{NN}_step08_generation.md` (4 shot-sets/loops) |
| 09 | `skills/video/steps/09_Generate_Video_Artist_Refined.md` | `pair_{NN}_step09_artist_refined.md` |
| 10 | `skills/video/steps/10_Generate_Video_Revision_Synthesis.md` | `pair_{NN}_step10_revision_synthesis.md` |

---

## The prompt contract (hard gate)

**Every prompt names its renderer** in the artifact frontmatter (`renderer:`, `endpoint:`, `duration_s:`, `aspect:`, `tier:`). A prompt with no named renderer is not a prompt, it is a paragraph.

Then write in that renderer's grammar. All grammars still owe the five concerns — camera, subject, action, setting, style & audio — they differ in *how* the concerns are expressed:

```
[CAMERA]  shot type + angle + movement       (cinematic grammar front-loads this)
[SUBJECT] specific, not generic              ("a woman in worn leather jacket, silver rings")
[ACTION]  exactly what happens
[SETTING] environment + time + weather/light
[STYLE & AUDIO]  aesthetic + explicit sound design
```

- **Separate camera from action** (each its own sentence). Describe negative space.
- **Audio is directed explicitly** — dialogue in quotes, `SFX:` prefix, `Ambient:` soundscape, `Audio:` music. On the declared-jobs and timed-ranges grammars, direct audio as a **timeline of discrete events with named instrumentation**, not a mood.
- **Real film references** allowed for look ("Terrence Malick golden hour"); **no living-artist/actor likeness**, no real victims. `vault/HUMAN_SUBJECT_STANDARD.md` binds.
- The 6 pairs use **distinct camera grammar / archetypes** — no two pairs default to the same move (the distinctiveness rule).

### ⭐ THE TAKE IS THE UNIT NOW, NOT THE CLIP

Seedance 2.x and MiniMax H3 **cut internally from a timed shot list**. The sequence has moved *inside* the generation call, which means editorial rhythm must be written down in advance:

```
[0–3s]  ECU: a hand closes around a moth.
[3–7s]  Cut wide: the same hand opens, empty, in a lit doorway.
[7–10s] The doorway light fails.
Audio: [0–3s] room tone, wing friction. [3s] one dry click. [7–10s] low drone, no music.
```

This is **not lost editorial control — it is editorial control moved upstream**, the way a long take is blocked before it is shot. But it only applies where the model honours it: check `timed_shot_list:` in the dossier. **Veo 3.1 renders one continuous shot per call and will ignore your timings.**

### ⭐ REFERENCE-TO-VIDEO IS HOW *ONE SEAM* GETS ENFORCED

L38 says a piece is made **OF** a material, never **ABOUT** materials: one dominant material (the world) + at most one intruder (the wound) + one seam, with the subject doing something across it. Reference-to-video makes that mechanical for the first time — **assign every reference a declared job**:

```
Use Image 1 for the world: the wet quarry, its light, its grain — this dominates every frame.
Use Image 2 for the intruder ONLY: the brass instrument. It is not in frame before 0:04.
The seam is at 0:04, where the hand carries it into the water.
Do not restyle Image 1 toward Image 2. No brass colour in the rock.
```

Same tool solves character/style consistency across a selected set. Caps: MiniMax H3 — 9 images, 3 videos, 3 audio (first 5 images free, then $0.08 each; reference *video* billed at $0.26/s of input). Seedance 2.5 — 50 files total (30/10/10), reference-to-video priced ×0.6.

---

## QA & delivery

1. **THE THREE-PASS BLIND RULE — `vault/VIDEO_RENDER_AUDIT.md` — on every render, before ranking.** Pass A muted (*"what happened?"*), Pass B audio-only (*"what happened?"*), Pass C together and against the prompt. **A and B name different events → FALSE WELD → repair or cut.** Native audio and native picture are generated coincident, so a fused "does the audio serve the image" check **cannot return NO** — synchresis and the McGurk effect guarantee it. Writes `<run-dir>/VIDEO_RENDER_AUDIT.md`; **no audit file → the render is UNAUDITED and may not be ranked, selected, or shipped.**
2. Run **lofn-qa** with the **Cinematic Somatic Gate** (`vault/DIRECTOR_QA_DEPTH_AUDIT.md`).
3. Rank by: **event legibility under mute** → opening-second hook → motion clarity → audio-image cohesion (only for clips that passed A=B) → emotional legibility → loop integrity (animation).
4. **THE DEDICATION IS A DELIVERABLE** (`vault/DEDICATION_STANDARD.md`) — every shipped piece carries its ≤500-char dedication, measured not eyeballed. If the piece rides a named figure, credit the person who made it.
5. Save per `skills/lofn-core/OUTPUT.md`, frontmatter carrying **renderer, endpoint, tier, duration, aspect, and the preflight cost line**; INDEX last.
6. **Do not call render tools without an approved, preflighted plan.** Autonomous runs emit prompts + the costed plan and stop.
