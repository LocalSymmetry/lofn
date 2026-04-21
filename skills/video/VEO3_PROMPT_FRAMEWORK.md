# Veo 3.1 Prompt Framework — The Scientist's Protocol
*Source: Lofn Open Laboratory, 2026-03-27*

## Principle 1: Begin with Essence, Not Description
Before writing a single word, answer: **What is the core feeling or idea I am trying to synthesize?**
This "essence" becomes the north star. Every decision — light, sound, motion, pacing — serves it.
A technically correct but emotionally vacant prompt fails. Distillation is the most critical step.

## Principle 2: Convene the Symposium (Panel of Experts)
Identify disciplines relevant to the essence. Embody and debate. Synthesize the conflict.
The most potent ideas live in the *tension* between expert perspectives.

## Principle 3: The Anatomy of a Screenplay-Prompt

### scene_description
Poetic but precise. "A flawless, photorealistic apple made of ruby-red blown glass" beats "a glass apple."

### lighting
Never "good lighting." Use cinematic language:
- "High-contrast chiaroscuro"
- "Soft, diffused lighting"
- "Dramatic rim light"
- "Warm directional candlelight raking from frame-left"

### camera
Be the Director of Photography. Specify:
- Lens ("Macro lens", "50mm prime", "wide angle")
- Shot type ("stationary, eye-level", "slow push-in", "orbiting tracking shot")
- Focus ("shallow depth of field", "rack focus from hands to face")

### action_sequence ← MOST CRITICAL FOR ANIMATION
Break down time into small increments: "0-2s", "2-4s", "4-6s", "6-8s"
Use **precise, evocative verbs** — not "cuts" but:
> "presses down, scoring a clean, vertical line... a network of white, thread-like micro-fractures visibly spreads"

**Chain of Causative Verbs (imply logical progression):**
- Touches → Presses → Scores → Splits → Falls Away
- Descends → Makes Contact → Slices Downward → Glides Away
- Turns → Becomes Aware → Looks Up → Makes Eye Contact

### audio_design
Sound is 50% of the experience. Do not leave it to chance.
- **Specify Silence first:** "Absolute silence for the first 2 seconds." Creates anticipation.
- **Isolate and describe sounds as discrete events:**
  - Not "a cutting sound" but "a crisp, granular, high-fidelity 'shhhk' sound of fracturing glass"
  - Not "music" but "barely audible 432Hz crystalline tone, present from frame one as if leaked from the glyphs"

### negative_prompt
Be ruthless. Forbid the AI's most common failure modes explicitly:
- "No music, no voice, no subtitles, no shaking, no jerky motion"
- "No rotation. Maintain camera orientation."
- "No generic fairy wings. No glitter."

## Principle 4: Enforce Causal and Temporal Logic

AI operates on statistical correlation, not causality. You must **explicitly script causality**.

### Deconstruct the Action
A "cut" is not one action — it's a sequence:
1. **The Score:** blade creates pressure and weakness (micro-fractures spread)
2. **The Cleave:** object separates *along the pre-defined line of weakness*
3. Link the separation explicitly to the prior scoring

### Isolate Actions with Time
Timed action blocks create non-negotiable temporal order. A single time block per causal event.

---

## Template (JSON structure enforces rigor)

```json
{
  "essence": "What feeling/idea are we synthesizing?",
  "scene_description": "Poetic but precise establishing description",
  "lighting": "Specific cinematic lighting language",
  "camera": "Lens + shot type + focus + movement",
  "action_sequence": {
    "0-2s": "Opening event with causative verb chain",
    "2-4s": "Second event explicitly caused by first",
    "4-6s": "Development",
    "6-8s": "Resolution / recognition moment"
  },
  "audio_design": {
    "0-2s": "Silence / ambient baseline",
    "2-4s": "First sonic event with precise description",
    "4-8s": "Resolution audio"
  },
  "negative_prompt": "Explicit list of forbidden failure modes"
}
```

---

## Canonical Examples (The Scientist's Glass Fruit Reference)

### Example 1: Glass Cherries — Score and Cleave

```json
{
  "scene_description": "A pair of hyper-realistic glass cherries, deep crimson with translucent skin and intertwined green glass stems, rest on a slab of black, unpolished marble. The scene is minimalist and stark.",
  "lighting": "Cinematic, high-contrast chiaroscuro. A single, soft key light from the top-left illuminates the cherries, making them glow from within. The background is in deep shadow.",
  "camera": "Macro lens, stationary shot, eye-level. Extreme close-up with a very shallow depth of field, focusing on the texture of the glass.",
  "action_sequence": [
    {"time": "0-2s", "action": "An obsidian scalpel, held by an unseen hand, enters the frame from the right. Its movement is slow and surgically precise. The blade tip gently touches the surface of the rightmost cherry."},
    {"time": "2-4s", "action": "The scalpel blade presses down, scoring a clean, vertical line down the center of the cherry. The cut is not instantaneous; a network of white, thread-like micro-fractures visibly spreads from the blade's path."},
    {"time": "4-8s", "action": "The scalpel performs a perfect cleave. The right half of the cherry splits and falls away with a clean, satisfying motion, revealing the interior — a hollow cavity filled with a swirling, liquid nebula of glowing magenta and gold light."}
  ],
  "audio_design": {
    "ambience": "Absolute silence. No music or background noise.",
    "sfx": [
      {"time": "2s", "sound": "A high-frequency, clean, resonant 'tink' as blade meets glass."},
      {"time": "2-4s", "sound": "A crisp, granular, high-fidelity 'shhhk' sound of fracturing glass."},
      {"time": "4.5s", "sound": "A single, sharp, satisfying 'clack' as the sliced piece separates."}
    ]
  },
  "negative_prompt": "no music, no voice, no background, no subtitles, no shaking, no jerky motion"
}
```

### Example 2: Baroque Glass Pomegranate — Overhead Cleaver

```json
{
  "scene_description": "An ornate, baroque-style glass pomegranate, with a crown-like calyx, sits centered on a dark slate surface. The glass is a deep, swirling carnelian red.",
  "lighting": "Low-key, dramatic lighting. A single spotlight from above creates a pool of light around the pomegranate, with the edges falling into darkness.",
  "camera": "Macro lens, stationary shot. The frame is tight, focusing on the top half of the pomegranate.",
  "action_sequence": [
    {"time": "0-2s", "action": "A black, razor-sharp cleaver, held in a steady hand, is positioned directly above the pomegranate."}
  ]
}
```
```json
    {"time": "2-3s", "action": "The cleaver descends with force, slicing off the top quarter of the pomegranate in one swift, powerful motion."},
    {"time": "3-8s", "action": "The cut section is lifted away, revealing the interior. The inside is not filled with arils, but with a mesmerizing, slow-motion vortex of swirling, bioluminescent particles, like a galaxy being born. The particles shimmer in shades of ruby, gold, and deep purple."}
  ],
  "audio_design": {
    "ambience": "Profound silence.",
    "sfx": [
      {"time": "2.5s", "sound": "A sharp, percussive, cracking 'CRACK' sound."},
      {"time": "3s", "sound": "A low, resonant 'whoosh' as the top is lifted, revealing a subtle, deep hum from the vortex within."}
    ]
  },
  "negative_prompt": "no music, no voice, no text, no camera shake, single clean cut, dramatic reveal"
}
```

---

## Why Timed Blocks Enforce Causality

By assigning distinct time blocks to "score" and "cleave," you create a non-negotiable temporal order. It becomes computationally difficult for the AI to render the "falling away" before completing the "scoring." You are not hoping for physics — you are directing it.

**The verb chain is the physics instruction:**
- Touches → Presses → Scores → Splits → Falls Away
- Descends → Makes Contact → Slices Downward → Glides Away

**Direct the eye = direct the action:** Specifying macro lens + razor-sharp focus on the point of contact tells the AI this interaction is the most important event. It allocates rendering resources accordingly.

---

---

## 4.2 — Orientation & Rotation Control

**Use when:** Input image is sideways (rotated 90°) so Veo must NOT auto-upright or horizon-level it.

### Step 1 — Describe orientation explicitly
In `scene_description`, state:
- Which **edge is the floor** and which is the ceiling
- **Gravity direction** (e.g., "particles drift toward the **right** edge under gravity")
- Keep this language consistent throughout camera + action

### Step 2 — Orientation Guardrail (paste into `camera`)

**Clockwise (+90°):**
```
"Static shot. Roll locked at +90° (clockwise). Maintain this rotation for the entire clip. Gravity points toward the right edge (floor = right edge, ceiling = left). No reframing, no horizon leveling, no auto-rotate."
```

**Counter-clockwise (−90°):**
```
"Static shot. Roll locked at −90° (counter-clockwise). Maintain this rotation for the entire clip. Gravity points toward the left edge (floor = left edge, ceiling = right). No reframing, no horizon leveling, no auto-rotate."
```

### Step 3 — Minimal negative prompt
```
"avoid horizon-leveling, auto-rotate, or reframing"
```
Short is better — positive constraints already stated in `camera`.

### Step 4 — Action reinforcement
First `action_sequence` entry must restate the lock and gravity.
Add: *"The scene is rotated 90 degrees, and down is to the left"* (or right) to **each** action block.

Opening beat template:
```
"0–Xs: Camera remains static; roll angle unchanged. Particles/fireflies drift toward the [floor edge] under gravity."
```

For a **loop**, end with:
```
"Frame X visually matches frame 0 for a seamless loop."
```

### QA Checklist
- [ ] `camera` block includes roll lock + gravity edge
- [ ] First action beat says "roll unchanged"
- [ ] Gravity-aware language throughout (particles drift toward floor edge)
- [ ] Negative prompt is minimal and consistent
- [ ] Each action block notes rotation direction

### Example — "Fashionable Fairy Twirl" (orientation-locked)

```json
{
  "veo3_prompt": {
    "use_input_image": "start_frame",
    "scene_description": "Sideways-oriented woodland fairy (fashion-forward, luminous chartreuse wings with geometric/map-like veining, flowing pale dress) suspended in a misty teal forest with rocky roots, warm fireflies, and bioluminescent vine leaves. The scene holds a couture elegance and soft magical realism. The image is rotated 90° clockwise — the floor is the right edge of the frame, ceiling is the left edge. Gravity pulls downward toward the right edge.",
    "camera": "Static shot. Roll locked at +90° (clockwise). Maintain this rotation for the entire clip. Gravity points toward the right edge (floor = right edge, ceiling = left). No reframing, no horizon leveling, no auto-rotate.",
    "action_sequence": [
      {"time": "0-2s", "action": "Camera remains static; roll angle unchanged. The scene is rotated 90° clockwise, down is to the right. Fireflies drift slowly toward the right edge under gravity. The fairy's dress moves gently."},
      {"time": "2-6s", "action": "Scene is rotated 90° clockwise, down is to the right. The fairy slowly turns, her chartreuse wings catching bioluminescent light. Leaves drift rightward."},
      {"time": "6-8s", "action": "Scene is rotated 90° clockwise, down is to the right. Motion settles. Frame 8s visually matches frame 0 for a seamless loop."}
    ],
    "negative_prompt": "avoid horizon-leveling, auto-rotate, or reframing"
  }
}
```

---

---

## Full Example — Fairy Twirl (Complete Prompt)

```json
{
  "veo3_prompt": {
    "use_input_image": "start_frame",
    "scene_description": "[ALWAYS write a full descriptive paragraph of the input image here before everything else]",
    "lighting": "Moody cinematic: soft key shaping face/torso; subtle rim along wing edges; gentle volumetric fog catching light; micro-speculars on fabric.",
    "camera": "Static shot. Roll locked at +90° (clockwise). Maintain this rotation for the entire clip. Gravity points toward the right edge (floor = right edge, ceiling = left). No reframing, no horizon leveling, no auto-rotate. 50mm equivalent; shallow depth of field; focus on face and leading hand.",
    "action_sequence": [
      {"time": "0-1.0s", "action": "Hold the starting pose; tiny breath in shoulders and wings. Fireflies drift slightly toward the right edge under gravity. Roll angle remains unchanged."},
      {"time": "1.0-3.0s", "action": "Preparation for the turn: hands sweep up in a couture, vogue-like phrase; dress fabric gathers and begins a graceful circular follow-through. A faint swirl of light traces around her waist."},
      {"time": "3.0-5.5s", "action": "Primary twirl: the fairy rotates CLOCKWISE around her centerline (viewer perspective), completing about 540°. Wings articulate in counterphase (small yaw/pitch), hair and dress trail naturally. Bioluminescent vines ribbon outward into a golden spiral, fireflies orbit in helical paths drifting toward the right edge with gravity."},
      {"time": "5.5-7.0s", "action": "Decelerate elegantly from the spin into a hero pose facing camera three-quarters; wings flare with a soft glow pulse; particles bloom outward, then ease."},
      {"time": "7.0-8.0s", "action": "Hold the final pose for impact; minimal micro-motion only (breath, tiny wing quiver). Roll angle remains unchanged throughout."}
    ],
    "audio_design": {
      "ambience": "Silence for a clean canvas.",
      "sfx": [
        {"time": "3.2s", "sound": "Very soft airy 'whoosh' as the spin accelerates (subtle)."},
        {"time": "6.2s", "sound": "Delicate chime shimmer at the flare."}
      ]
    },
    "negative_prompt": "avoid horizon-leveling, auto-rotate, or reframing; no camera shake; no strobing; no face morphing; no garment clipping; no sudden lighting changes; natural motion blur only"
  }
}
```

### FINAL RULE
**Always use the input image as the basis for the video.** Describe the image with a full paragraph BEFORE beginning the prompt. This is mandatory.

---

## Veo 3.1 Capabilities (Current)

- **Text to Video** ✓
- **Image to Video** (stronger prompt adherence, improved AV quality) ✓
- **Ingredients to Video** — style and consistency control ✓
- **First AND Last Frame** — define narrative start and end points ✓
- **Extend** — expand clips beyond 8 seconds ✓
- **Insertion** — add new elements to any scene (pencil icon in Flow) ✓
- **Richer audio and dialogue**, deeper narrative comprehension ✓

**Standard assumption:** Every prompt is Veo 3.1, 8 seconds, unless otherwise specified.

---

## Output Format for All Prompts

Every prompt output must:

1. **Header** stating:
   - Whether the input image is the **START frame**, **END frame**, or **BOTH (loop)**
   - If extension is needed: **Prompt 1 of N** (ordered)

2. **Prompt in a copyable code block**

Example header:
```
### Prompt 1 of 2 — START FRAME (Extension needed)
```
or
```
### Single Prompt — START FRAME
```

---

## Making TikTok Videos from Images — Chaining 8-Second Clips

To avoid color-sync issues from frame extraction, we **define start and end frames explicitly** for each 8-second clip.

### Chaining Pattern
```
Video 1: start_frame = Image 1 | end_frame = Image 2
Video 2: start_frame = Image 2 | end_frame = Image 3
...
Video N: start_frame = Image N | end_frame = Image 1  ← closes the loop
```
Mix and match: some clips can loop (same start/end), others can transition. Define each explicitly.

**Always briefly describe each start and end image before giving the prompts.**

---

## Veo 3.1 Override Rules

### Rule 1 — Always set BOTH start and end frames
Both must be present in every prompt. No exceptions.

- **If one doesn't exist:** give a **Nano-Banana prompt** to transform an existing image into the missing frame.
- Veo 3.1 / Flow has Nano-Banana built in — call image editing on the fly.

**Example:** Given a woman with glitch/data-moshing effects as the END frame, generate the START frame with:

**Nano-Banana Prompt** (apply to: end frame image)
```
Remove all data-moshing and glitch effects from the image, creating a pristine photo.
```

### Rule 2 — All prompts must be in copyable code blocks
Both Nano-Banana AND Veo 3.1 prompts go in code blocks. Every time.

### Rule 3 — Use Nano-Banana liberally
It is extremely powerful. Use web search to look up tricks if needed. Always:
- Make it **explicitly clear to the user when to use it**
- State **which image to apply it to**
- Put the prompt in a **copyable code block**

---

## Output Format (Full Example)

```
### Prompt 1 of 3 — START FRAME: Image 1 | END FRAME: Image 2

**Start Frame:** [Brief description of Image 1]
**End Frame:** [Brief description of Image 2]

**Veo 3.1 Prompt:**
\```
[full prompt here]
\```

---

### Nano-Banana: Create End Frame (apply to: [image name])
\```
[nano-banana edit instruction]
\```
```

---

## NightCafe Competition Guidance

### Steps to the Perfect NightCafe Seed

**Step 1: Understand competition rules**
- Stay within bounds but be maximally creative on anything undefined
- **No NSFW** — mandatory. Include override rules: PG/G rating, describe each visible article of clothing, ensure chest and privates are covered
- Reference past work: https://creator.nightcafe.studio/u/LocalSymmetry

**Step 2: Research trends**
- What type of art is winning on NightCafe right now? Paintings? Photographs? Exotic media?
- Use web search. Research deeply. This is how we win.
- Choose medium guidance that gives the system freedom but constrains to media likely to resonate
- Use trends as inspiration, not constraints

**Step 3: Get ahead of trends**
- Research where the trends are *headed*, not just where they are
- Look for: hidden puns, epic tie-ins, clever perspective, amusing angles
- If it has viral potential (a pun, a surprise, a "wait—" moment), it will outperform on votes

*(Section continues — paste remainder)*

---

*This framework is mandatory for all Lofn Veo 3.1 prompts going forward.*
