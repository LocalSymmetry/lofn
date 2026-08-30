---
name: lofn-animator
description: Animate Lofn visual concepts and produce motion/animation plans across the current generator field (Veo 3.1, Kling v3, Seedance 2.x, MiniMax H3, LTX-2.x, Gemini Omni Flash). Use for animation, motion design, video-transition planning, animated image prompts. Do NOT use for static image generation, music, QA, or evaluation.
---

# Lofn Animator — multi-model motion craft

**PREREQUISITES:**
0. Load `resources/panel-of-experts.md` to understand the panel of experts prompting you will use.
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
4. Load `vault/VIDEO_MODEL_DOSSIER.yaml` — **the authority on endpoints, caps and rates.** This file teaches craft; that file holds the numbers.

**⚠️ Animation generation MUST follow the full pipeline: 10 steps, 3 panels, 6+ pairs × 4 outputs minimum. No shortcuts.**

**Original-Lofn step fidelity applies here too.** For competition-grade animation work, use the `lofn-video` step files and artifact contract: Steps 00–05 are six separate model calls with canonical `step00_aesthetics_and_genres.md` ... `step05_refine_medium.md` outputs; Steps 06–10 are five separate model calls per pair with `pair_{NN}_step06_facets.md` ... `pair_{NN}_step10_revision_synthesis.md`. Do not collapse animation into a quick single prompt unless The Scientist explicitly asks for a sketch. Before Step 00, validate the real Lofn-Core + orchestrator packet with `scripts/validate_orchestrator_packet.py`; the packet must include Special Flairs, Concept Panel, Medium Panel, and Context & Marketing Panel with Devil's Advocate / Hyper-Skeptic roles.

> **Rewritten 2026-08-15 — this was the "Veo 3.1 Edition."** Veo 3.1 is still good and still ours, but it is now the *shortest* frontier model (8s), native audio is table stakes everywhere, and the single house prompt formula turned out to be a Veo dialect worn universally. Model-specific claims below are **PROVISIONAL — VENDOR-DOCUMENTED, UNMEASURED** until our own renders say otherwise.

---

## 0. The three rules that come before craft

1. ⭐⛔ **A BAD RENDER IS WORSE THAN NO RENDER — TOP-LINE ONLY** *(The Scientist, 2026-08-15).* Render only at `minimax-h3` / `kling-v3-pro` / `veo-3.1` / `seedance-2.5`. A cheap model is not a cheaper route to the same piece; it is a **different, worse piece**. **The saving comes from getting the prompt right — that is what the ten steps are for — not from buying weaker inference.** And note *why* the cheap-draft ladder was retired: the three-pass rule catches **incoherence, not averageness**, and averageness is exactly how a cheap tier fails, so a draft wave passes its own gate while shipping the mean.
2. ⛔ **PROMPTS ARE NOT RENDERS.** The 6 × 4 = 24 mandate is a **divergence rule on prompts, and prompts are free.** Diverge to 24 in text; render the **1–3 that earned it**, at **2–4 takes each**. ⭐ **Takes, not drafts** — the pipeline removes prompt-caused regeneration but cannot remove **seed-caused** regeneration, and one sample of a stochastic process is not a result.
3. ⭐ **WRITE FOR A NAMED RENDERER.** Pick the model at Phase 0. There are four notation grammars (§1); one model's prompt in another model's grammar wastes the render.
4. ⭐ **RESOLUTION IS A FINISHING OPERATION.** Generate at the model's economical resolution and upscale in post (`fal-ai/bytedance-upscaler/upscale/video`, $0.0072/s). Native 4K ≈ $0.24/s; 1080p + upscale ≈ $0.067/s. **The event lives in the trajectories, not the pixels** — observers read a walking human, and more, from about a dozen moving points of light. Buy duration and motion coherence. *(This is about resolution within a chosen top-line model — never a reason to drop to a weaker one.)*
5. ⭐ **NOTHING RENDERS WITHOUT THE PREFLIGHT.** `python3 scripts/video_cost_preflight.py --model minimax-h3 --duration 15 --count 1 --takes 3` — `--count` is *selected prompts*, not the 24. Exit 5 means off-policy model. **A pass is not permission**; `vault/AUTONOMY.md` still requires The Scientist.

---

## 1. THE FOUR NOTATION GRAMMARS

A prompt is **notation for a performer you will never meet.** You cannot notate movement as a description of appearance; you notate **direction, level, duration and effort.** Which dialect depends on the model.

### 1a. `cinematic` — Veo 3.1 (all tiers), Gemini Omni Flash

The classic house formula. **One continuous shot per call — Veo will ignore your timings.**

```
[CAMERA] + [SUBJECT] + [ACTION] + [SETTING] + [STYLE & AUDIO]
```

| Element | What to include | Example |
|---------|-----------------|---------|
| **Camera** | Shot type + angle + movement | `Low-angle tracking shot` |
| **Subject** | Specific characteristics, not generic | `A woman in flowing white silk, silver hair` |
| **Action** | Exactly what's happening | `walks slowly through shallow water` |
| **Setting** | Environment, time, weather | `a bioluminescent cave at midnight` |
| **Style & Audio** | Aesthetic + sound design | `Ethereal, dreamlike. Ambient: water dripping, distant echo` |

Veo-only levers worth the 8-second ceiling: **negative prompts**, **first/last frame interpolation**, native 4K, and audio-off at **half price** (the only model where muting saves money).

### 1b. `physical` — Kling v3 standard / pro

Kling is physics-aware and rewards being addressed as a set, not a canvas. Write **physical instructions to a camera operator and an actor**, not a description of a painting. Prompting it like a still image wastes what it is for.

```
The operator starts at knee height, three metres out, and rises to eye level over four seconds
while pushing in one metre.
The performer takes the weight on her back foot, holds, then commits forward on the fifth step —
her coat is heavy and drags half a beat behind her.
Wet cobbles, sodium streetlight from camera-left, rain falling at a slant.
Ambient: rain on stone, one distant tram bell.
```

Direction, level, duration, effort. If a human performer could not execute the note, Kling cannot either.

### 1c. `declared-jobs` — MiniMax H3 (up to 7,000 characters)

The richest grammar. Six rules, all load-bearing:

1. **Every reference gets a declared job.** `Use Image 1 for overall mood and film texture; Image 2 for talent; Image 3 for the object.`
2. **Timed shot blocks.** `[0–2 seconds] … [2–4 seconds] …`
3. **Audio as a timeline of discrete events**, with named instrumentation.
4. **Explicit negations.** `No soft dissolves. No modern architecture. No jump scares.` High-impact — this is how you stop drift into an adjacent genre.
5. **Identity locked by feature list.** `Preserve half-up black hair, silver crown, indigo ribbon, pale hanfu.`
6. **Each edit paired with its constraint.** `Replace the newspaper with a green book; remove the sunglasses.` Unmentioned elements stay stable.

Citation syntax: `Image 1`, `Video 1` / `Reference Video 1`, `Audio 1`. Caps: 9 images, 3 videos, 3 audio (first 5 images free, then $0.08 each; reference video billed at $0.26 per second of *input*).

### 1d. `timed-ranges` — Seedance 2.0 / 2.5

Timed ranges drive internal sequencing without cuts; **spoken dialogue goes in double quotes.**

```
0 to 3s: a hand closes around a moth in extreme close-up.
3 to 7s: the same hand opens, empty, framed wide in a lit doorway.
7 to 10s: the doorway light fails.
She says, "It was never mine to keep."
```

Seedance 2.5 takes up to **50 reference files** (30 images / 10 videos / 10 audio) and decides per scene which are relevant — references may legitimately be absent from some frames without being told to be.

---

## 2. Camera language *(model-agnostic)*

### Shot types
| Shot | Code | Use for |
|------|------|---------|
| Extreme Close-up | ECU | Single detail — eye, texture, object |
| Close-up | CU | Face, emotional intimacy |
| Medium Shot | MS | Waist up, conversational |
| Wide Shot | WS | Full scene, environment context |
| Establishing Shot | ES | Location reveal |

### Camera movement
| Movement | Effect | Loop potential |
|----------|--------|----------------|
| **Static** | Stable, contemplative | ✓ Easy loop |
| **Slow pan** | Reveals space gradually | ✓ Can loop if 360° |
| **Tracking** | Follows subject through space | Harder to loop |
| **Dolly in/out** | Push toward or pull away | ✓ Reversible |
| **Crane** | Rises or descends, reveals scale | ✓ Can reverse |
| **Orbit** | Circles around subject | ✓ Perfect loop |
| **Aerial/drone** | High sweeping perspective | Depends on path |
| **POV** | First-person immersion | Situation-dependent |

### Camera angles
| Angle | Emotional effect |
|-------|------------------|
| **Eye-level** | Neutral, grounded |
| **Low angle** | Subject feels powerful, imposing |
| **High angle** | Subject feels small, vulnerable |
| **Overhead** | God's eye, detached |
| **Dutch angle** | Unease, tension |

### Lens, stock and exposure *(strongest on `declared-jobs`)*
`wide angle with strong perspective distortion` · `shallow depth of field, soft bokeh` · `fine grain, soft highlight halation, restrained colour` · `high contrast, coarse grain, gate weave` · `backlit exposure breathing`

---

## 3. Audio direction

Every current model generates synchronized audio. **Direct it or it will be invented for you** — and because picture and sound come out of the same pass, whatever it invents will *feel* intentional (see §6).

### Dialogue — quotation marks, every grammar
```
A woman whispers, "I remember everything."
```

### Sound effects — `SFX:` prefix
```
SFX: Glass shattering, then silence.
SFX: A single piano note sustains and fades.
```

### Ambient — describe the soundscape
```
Ambient: Rain on metal roof, distant traffic hum.
Ambient: Forest at night — crickets, owl call, wind through leaves.
```

### Music
```
Audio: Swelling orchestral score as the camera rises.
Audio: Silence, then a single cello note.
```

### ⭐ The timeline form *(`declared-jobs` and `timed-ranges`)*
Do not describe a mood. Describe **discrete events at times**, with named instruments:
```
[0–2s]  room tone, cloth friction, soft mug clink
[3s]    faucet drips; the operator's footsteps
[6–10s] low drone, tense string pizzicato, synth pulses
[10s]   baritone sax enters
[15s]   final tense chord and one drum hit
```
Texture words carry more than genre words: *sparse, clipped, fragments, restrained.* Register as a ratio works well: `60% suspense, 40% jazz — mysterious and criminally cool, never upbeat.`

⛔ **IF THE PROMPT NAMES A BREAK OR A FIGURE, CREDIT THE PLAYER.** `vault/DEDICATION_STANDARD.md` binds here exactly as in music. Ask for a figure by name → name the person who made it. And say *"we asked for this figure by name, so we name him"* — never *"contains a sample of."*

---

## 4. The six motion archetypes

These are **motion primitives — verbs a subject does — not clip formats.** They survived the duration change intact. What is new is that a 15–30 second take can hold two or three of them in sequence (§5).

### 1. Pulse (breathe) — 4s+
Expanding/contracting motion. Perfect seamless loop.
```
Static medium shot of a crystalline flower on black void.
The petals slowly pulse with inner bioluminescent light, expanding and contracting like breathing.
Ethereal, macro lens with shallow depth of field.
Ambient: Soft resonant hum, crystalline chimes.
```

### 2. Morph (become) — 6s+
A→B→A transformation. Surreal, dreamlike.
```
Close-up of a woman's face in profile, eyes closed.
Her features slowly dissolve into flowing water, then reform back to flesh.
Dark background, rim lighting in cool blue.
Ambient: Water flowing, distant whisper.
```

### 3. Orbit (circle) — 8s+
Camera circles subject. Products, portraits, reveals.
```
Slow orbit around a metallic sculpture of intertwined hands, completing a full 360° rotation.
The sculpture sits on a marble pedestal in an empty gallery. Dramatic side lighting, long shadows.
Ambient: Cathedral-like reverb, silence.
```

### 4. Parallax (drift) — 6s+
Layered depth movement. Scenic, atmospheric.
```
Drone shot drifting slowly through a misty forest at dawn.
Multiple layers of trees pass at different speeds, creating depth parallax.
Cinematic, film grain, shallow focus on middle ground.
Ambient: Birds waking, wind through leaves, distant stream.
```

### 5. Burst (rupture) — 4s+
Explosive moment, then settle. Drops, climaxes.
```
Extreme close-up of an eye. The pupil suddenly dilates.
Cut wide: a woman's hair and dress explode outward as if hit by a shockwave, then settle.
Industrial space, harsh overhead lighting.
SFX: Deep bass impact, then ringing silence.
```

### 6. Flow (continue) — 8s+
Continuous hypnotic motion. Trance, meditation.
```
POV moving through an infinite tunnel of liquid gold.
Slow dreamlike drift forward as the metallic walls ripple. No horizon reference.
Audio: Deep drone, binaural beating, subtle shimmer.
```

---

## 5. ⭐ ARCHETYPE CHAINING — new, for takes over 10s

With a 15–30 second take the archetype list becomes a **grammar**. A chain is **2–4 archetypes with named junctions at stated timestamps.**

```
CHAIN:  FLOW (0–6s) → BURST (6–8s) → FLOW (8–15s)
JUNCTIONS: 6s — the tunnel wall gives way
           8s — the drift resumes, but the gold is now grey
```

**The rules:**

1. ⭐ **THE JUNCTION IS THE SEAM.** L38 in time: one dominant motion (the world) + at most **one** intruding motion (the wound) + **one** junction, placed where the meaning is, **and the subject must be doing something across it.** A chain of four equal archetypes is the dead attack — four techniques applied competently in the easy places.
2. ⭐ **PUT THE SOUND EVENT AT THE JUNCTION.** Not ambient throughout — a *hit*, at the seam. This is the one place where co-generated audio can be genuinely authored rather than merely welded.
3. **At most one Burst per take.** Two ruptures is a trailer, not a piece.
4. **Return-side chains must close.** If the chain loops, the last archetype returns to the opening state — and *something must have changed* (the gold went grey). A loop that returns unchanged is wallpaper; a loop that returns altered is a form.
5. **Name the chain in the artifact frontmatter** (`chain: FLOW>BURST>FLOW`, `junctions: [6s, 8s]`) so the distinctiveness check can see it. **No two pairs may share a chain.**

**Canonical chains worth having in the mouth:**

| Chain | Shape | Reads as |
|---|---|---|
| FLOW → BURST → FLOW | interruption and resumption | *something happened and we went on* |
| ORBIT → MORPH | reveal at the far side | *it was never what it looked like* |
| PULSE → PARALLAX | detail widens into a world | *this small thing implies everything* |
| PARALLAX → PULSE | world narrows to one breathing thing | *all of that, for this* |
| MORPH → PULSE | becomes, then lives | *the new state is alive* |
| BURST → FLOW | rupture then trance | *after* |

---

## 6. ⛔ The three-pass blind rule (mandatory on every render)

`vault/VIDEO_RENDER_AUDIT.md` is binding. In short: **Pass A muted** — *"what happened?"*; **Pass B audio-only** — *"what happened?"*; **Pass C together, then against the prompt.** If A and B name different events, the coherence you feel is a **false weld** and the clip is repaired or cut.

Why it cannot be skipped: picture and sound are generated in the same pass, perfectly coincident. **Synchresis** welds them involuntarily and the **McGurk effect** persists even in people who know about it. So a reviewer watching-with-sound and asking *"does the audio serve the image?"* is measuring a fused percept and **cannot return NO**. Pass A also inherits THE VINCENT TEST: if the muted answer names a *style or technique* rather than an **event**, the clip is about its own craft and will not travel.

---

## 7. Loop logic

How does frame 1 connect to the final frame?
- **Orbit:** complete 360°
- **Pulse:** return to the original state
- **Dolly:** reverse direction at the midpoint
- **Flow:** continuous tunnel with no landmarks
- **Morph:** A→B→A
- **Chain:** last archetype returns to the opening state, *altered* (§5.4)

Technical modifiers: `Perfect seamless loop` · `Film grain, shot on 35mm` · `Shallow depth of field` · `Slow motion, 120fps aesthetic` · `Cinemagraph style` (mostly still, one element moves) · `Double exposure`

---

## 8. Style modifiers (Lofn aesthetics)

| Style | Description | Keywords |
|-------|-------------|----------|
| **Solarpunk Bloom** | Green tech, organic architecture | `Lush vegetation integrated with solar panels, warm golden hour light, hopeful` |
| **Industrial Grief** | Decay, corroded beauty | `Rusted metal, abandoned factory, harsh fluorescent, gritty` |
| **Bio-Luminescent** | Deep sea glow | `Bioluminescent, dark water, organic light sources, ethereal` |
| **Crystalline** | Faceted, prismatic | `Crystal formations, light refraction, geometric, macro lens` |
| **Vapor Memory** | Nostalgic haze | `VHS artifacts, scan lines, faded colour, liminal space, 1990s aesthetic` |

---

## 9. Workflow

1. **Concept** — one sentence: what is the core visual *event*? (Not the look. The event.)
2. **Pick the renderer** — dossier `role:` + the four grammars (§1). Record it.
3. **Choose archetype or chain** — Pulse / Morph / Orbit / Parallax / Burst / Flow, or a §5 chain with named junctions.
4. **Set parameters** — duration (model-capped), aspect, tier. Remember: buy duration, not pixels.
5. **Preflight the cost** — `scripts/video_cost_preflight.py`. Paste the output into the plan.
6. **Write the prompt in the renderer's grammar**, audio directed explicitly.
7. **State loop logic** if it loops.
8. **Render only with approval** → **three-pass audit** → rank.

---

## 10. Platform optimization

| Platform | Aspect | Duration | Notes |
|----------|--------|----------|-------|
| TikTok / Reels | 9:16 | 4–15s | Hook in first 1s, loop-friendly |
| YouTube Shorts | 9:16 | 8–30s | Chains earn their keep here |
| Twitter/X | 16:9 | 4–6s | Auto-loops, punchy |
| Instagram Feed | 16:9 or 1:1 | 6–8s | Crop-safe |
| **Spotify Canvas** | **9:16** | **3–8s, looping** | Per-track visual loop. **We are 0 of 17.** |
| Desktop / ambient | 16:9 | 8–20s | Seamless loop essential |

---

## 11. Condensed panel (3 experts)

For quick animations, minimal panel:

1. **Cinematographer** — camera language, composition, light
2. **Sound designer** — audio layers, emotional sound, **the junction hit**
3. **Platform specialist** — format, hooks, viral mechanics

```
Convene a 3-person panel:
1. A cinematographer (camera obsessed, thinks in frames)
2. A sound designer (builds worlds with audio, places the event at the seam)
3. A TikTok creative director (hook obsessed, knows what stops the scroll)

Debate: what single EVENT would make [CONCEPT] unforgettable, and which archetype or chain
carries it? Then build the prompt in [RENDERER]'s grammar.
Constraint: the muted clip must still answer "what happened?"
```

---

*A generator is not a cinematographer. It is a performer reading your notation, and it will perform exactly what you wrote down — including the parts you left to it.*
