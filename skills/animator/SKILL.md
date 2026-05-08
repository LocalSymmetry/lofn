---
name: lofn-animator
description: Animate Lofn visual concepts and produce motion/animation plans. Use for animation, motion design, video-transition planning, animated image prompts. Do NOT use for static image generation, music, QA, or evaluation.
---

# Lofn Quick Animator — Veo 3.1 Edition

**PREREQUISITES:**
0. Load `resources/panel-of-experts.md` to understand the panel of experts prompting you will use.
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.

**⚠️ Animation generation MUST follow the full pipeline: 10 steps, 3 panels, 6+ pairs × 4 outputs minimum. No shortcuts.**

Fast animation seeds for Veo 3.1 via Google One Ultra. Cinematic loops, motion studies, viral moments — with synchronized audio.

## Veo 3.1 Capabilities

| Feature | Options |
|---------|---------|
| **Resolution** | 720p, 1080p |
| **Aspect Ratio** | 16:9 (landscape), 9:16 (vertical/TikTok) |
| **Duration** | 4s, 6s, 8s |
| **Audio** | Dialogue, SFX, ambient — all synchronized |
| **Advanced** | Image-to-video, first/last frame, reference images |

---

## The Veo 3.1 Prompt Formula

```
[CAMERA] + [SUBJECT] + [ACTION] + [SETTING] + [STYLE & AUDIO]
```

### Element Breakdown

| Element | What to Include | Example |
|---------|-----------------|---------|
| **Camera** | Shot type + angle + movement | `Low-angle tracking shot` |
| **Subject** | Specific characteristics, not generic | `A woman in flowing white silk, silver hair` |
| **Action** | Exactly what's happening | `walks slowly through shallow water` |
| **Setting** | Environment, time, weather | `a bioluminescent cave at midnight` |
| **Style & Audio** | Aesthetic + sound design | `Ethereal, dreamlike. Ambient: water dripping, distant echo` |

---

## Camera Language

### Shot Types
| Shot | Code | Use For |
|------|------|---------|
| Extreme Close-up | ECU | Single detail — eye, texture, object |
| Close-up | CU | Face, emotional intimacy |
| Medium Shot | MS | Waist up, conversational |
| Wide Shot | WS | Full scene, environment context |
| Establishing Shot | ES | Location reveal |

### Camera Movement
| Movement | Effect | Loop Potential |
|----------|--------|----------------|
| **Static** | Stable, contemplative | ✓ Easy loop |
| **Slow pan** | Reveals space gradually | ✓ Can loop if 360° |
| **Tracking** | Follows subject through space | Harder to loop |
| **Dolly in/out** | Push toward or pull away | ✓ Reversible |
| **Crane** | Rises or descends, reveals scale | ✓ Can reverse |
| **Orbit** | Circles around subject | ✓ Perfect loop |
| **Aerial/drone** | High sweeping perspective | Depends on path |
| **POV** | First-person immersion | Situation-dependent |

### Camera Angles
| Angle | Emotional Effect |
|-------|------------------|
| **Eye-level** | Neutral, grounded |
| **Low angle** | Subject feels powerful, imposing |
| **High angle** | Subject feels small, vulnerable |
| **Overhead** | God's eye, detached |
| **Dutch angle** | Unease, tension |

---

## Audio Direction

Veo 3.1 generates synchronized audio. Direct it explicitly:

### Dialogue
Use quotation marks:
```
A woman whispers, "I remember everything."
```

### Sound Effects
Use `SFX:` prefix:
```
SFX: Glass shattering, then silence.
SFX: Thunder rumbles in the distance.
SFX: A single piano note sustains and fades.
```

### Ambient Sound
Describe the soundscape:
```
Ambient: Rain on metal roof, distant traffic hum.
Ambient: Forest at night — crickets, owl call, wind through leaves.
Ambient: The quiet hum of a server room.
```

### Music Direction
```
Audio: Swelling orchestral score as the camera rises.
Audio: Lo-fi beats, muffled as if through headphones.
Audio: Silence, then a single cello note.
```

---

## Lofn Animation Archetypes (Veo 3.1 Optimized)

### 1. Pulse Loop (4s)
Breathing, expanding/contracting motion. Perfect seamless loop.
```
Static medium shot of a crystalline flower on black void.
The petals slowly pulse with inner bioluminescent light,
expanding and contracting like breathing.
Ethereal, macro lens with shallow depth of field.
Ambient: Soft resonant hum, crystalline chimes.
```

### 2. Morph Cycle (6s)
A→B→A transformation. Surreal, dreamlike.
```
Close-up of a woman's face in profile, eyes closed.
Her features slowly dissolve into flowing water,
then reform back to flesh. Seamless transformation.
Dark background, rim lighting in cool blue.
Ambient: Water flowing, distant whisper.
```

### 3. Orbit Reveal (8s)
Camera circles subject. Perfect for products or portraits.
```
Slow orbit around a metallic sculpture of intertwined hands,
camera completing a full 360° rotation.
The sculpture sits on a marble pedestal in an empty gallery.
Dramatic side lighting, long shadows.
SFX: Soft mechanical whir of camera movement.
Ambient: Cathedral-like reverb, silence.
```

### 4. Parallax Drift (6s)
Layered depth movement. Scenic, atmospheric.
```
Drone shot drifting slowly through a misty forest at dawn.
Multiple layers of trees pass at different speeds,
creating depth parallax. Golden light filters through canopy.
Cinematic, film grain, shallow focus on middle ground.
Ambient: Birds waking, wind through leaves, distant stream.
```

### 5. Reaction Burst (4s)
Explosive moment, then settle. For drops, climaxes.
```
Extreme close-up of an eye. Pupil suddenly dilates.
Cut to wide shot: a woman's hair and dress explode outward
as if hit by shockwave, then slowly settle back.
Industrial space, harsh overhead lighting.
SFX: Deep bass impact, then ringing silence.
```

### 6. Flow State (8s)
Continuous hypnotic motion. Trance, meditation.
```
POV shot moving through an infinite tunnel of liquid gold.
Slow, dreamlike drift forward as the metallic walls ripple.
The passage curves gently, light source always ahead.
Abstract, no horizon reference.
Audio: Deep drone, binaural beating, subtle shimmer.
```

---

## Style Modifiers (Lofn Aesthetics)

### Visual Styles
| Style | Description | Veo Keywords |
|-------|-------------|--------------|
| **Solarpunk Bloom** | Green tech, organic architecture | `Lush vegetation integrated with solar panels, warm golden hour light, hopeful` |
| **Industrial Grief** | Decay, corroded beauty | `Rusted metal, abandoned factory, harsh fluorescent, gritty` |
| **Bio-Luminescent** | Deep sea glow | `Bioluminescent, dark water, organic light sources, ethereal` |
| **Crystalline** | Faceted, prismatic | `Crystal formations, light refraction, geometric, macro lens` |
| **Vapor Memory** | Nostalgic haze | `VHS artifacts, scan lines, faded color, liminal space, 1990s aesthetic` |

### Technical Modifiers
```
Perfect seamless loop          — Start/end frames match
Film grain, shot on 35mm       — Cinematic texture
Shallow depth of field         — Dreamy focus falloff
Slow motion, 120fps aesthetic  — Time stretched
Cinemagraph style              — Mostly still, one element moves
Double exposure                — Layered imagery
```

---

## Workflow

### Step 1: Concept (One Sentence)
What's the core visual moment?

### Step 2: Choose Archetype
Pulse | Morph | Orbit | Parallax | Burst | Flow

### Step 3: Set Parameters
- **Duration:** 4s / 6s / 8s
- **Aspect:** 16:9 (cinematic) or 9:16 (vertical)
- **Resolution:** 720p (faster) or 1080p (quality)

### Step 4: Build Prompt
```
[Camera: shot + angle + movement]
[Subject: specific, detailed]
[Action: exactly what happens]
[Setting: environment, time, light]
[Style: aesthetic, mood, film reference]
[Audio: dialogue / SFX / ambient / music]
```

### Step 5: Loop Logic (if needed)
How does frame 1 connect to final frame?
- Orbit: Complete 360°
- Pulse: Return to original state
- Dolly: Reverse direction at midpoint
- Flow: Continuous tunnel with no landmarks

---

## Example Seeds

### Solarpunk Pulse (4s, 9:16)
```
Static close-up of a woman's face in profile, eyes closed.
Tiny solar panels embedded in her skin like scales catch the light.
A vine slowly grows across her cheek, blooming a small flower,
then recedes back — a breathing cycle.
Golden hour light, warm and hopeful, shallow depth of field.
Ambient: Soft wind, distant birdsong, gentle electronic hum.
```

### Industrial Grief Burst (4s, 16:9)
```
Wide shot of a corroded metal flower in an abandoned factory.
Sudden bloom: rust particles explode upward like inverse snow,
sparks arc from the petals, then slowly drift back down.
Harsh fluorescent lighting, deep shadows, gritty texture.
SFX: Metal stress groan, spark crackle, then hollow silence.
```

### Bio-Luminescent Morph (6s, 16:9)
```
Close-up of a woman's face underwater, eyes open.
Her features slowly transform — skin becomes translucent,
bioluminescent patterns flow across her face like circuitry,
then fade back to human flesh.
Dark ocean depths, only her glow illuminates the frame.
Ambient: Muffled water pressure, distant whale song, heartbeat.
```

### Crystalline Orbit (8s, 16:9)
```
Slow 360° orbit around a floating crystal cluster.
The formation hovers in a void, rotating counter to camera.
Prismatic light refracts through facets, casting rainbow caustics.
Deep black background, single dramatic light source.
Audio: Sustained crystalline tone, subtle harmonic shifts as light changes.
```

### Vapor Memory Flow (8s, 9:16)
```
POV walking through an empty shopping mall at night, 1992.
Fluorescent lights flicker, faded pastel storefronts pass by.
VHS tracking artifacts ripple across the frame.
No people, just the hum of the building.
Liminal, nostalgic, slightly unsettling.
Ambient: Distant muzak, HVAC drone, own footsteps echoing.
```

---

## Pro Tips

1. **Front-load camera** — Start prompt with shot type so Veo prioritizes framing
2. **Separate camera from action** — "The camera pulls back" as its own sentence
3. **Be specific on subjects** — Not "a person" but "a woman in worn leather jacket, silver rings"
4. **Describe negative space** — What's NOT in frame can be as important
5. **Layer audio** — Ambient + SFX + optional music creates depth
6. **Use real film references** — "Shot like Blade Runner" or "Terrence Malick golden hour"

---

## Platform Optimization

| Platform | Aspect | Duration | Notes |
|----------|--------|----------|-------|
| TikTok/Reels | 9:16 | 4-8s | Hook in first 1s, loop-friendly |
| YouTube Shorts | 9:16 | 8s | Can extend with cuts |
| Twitter/X | 16:9 | 4-6s | Auto-loops, punchy |
| Instagram Feed | 16:9 or 1:1 | 6-8s | Crop-safe |
| Desktop/Ambient | 16:9 | 8s | Seamless loop essential |

---

## Condensed Panel (3 Experts)

For quick animations, minimal panel:

1. **Cinematographer** — Camera language, composition, light
2. **Sound Designer** — Audio layers, emotional sound
3. **Platform Specialist** — Format, hooks, viral mechanics

### Quick Panel Prompt
```
Convene a 3-person panel:
1. A cinematographer (camera obsessed, thinks in frames)
2. A sound designer (builds worlds with audio)
3. A TikTok creative director (hook obsessed, knows what stops the scroll)

Debate: What single 4-second moment would make [CONCEPT] unforgettable?
Build the Veo 3.1 prompt together.
```

---

*Veo 3.1 is a cinematographer that gets your vision on the first take. Direct it.*
