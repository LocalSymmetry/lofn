# Eight-song finished-render audit

Date: 2026-07-24  
Method: `lofn-render-audit` — numeric measurement, then an audio-only blind listen under anonymous filenames, then a second-turn comparison against the exact style prompt, exclude prompt, and lyrics.  
Listener: `gpt-audio-1.5` through Poe. Listener timestamps are approximate. For silence, isolated tones, and stereo presence, the measurements below overrule the listener.

This is not a compliance ranking. The prompt is evidence about the request; the render is the work.

## What the renders say

| Song | What clearly survived | What the generator regularized | Productive result |
|---|---|---|---|
| **Harder to Reach** | Close female lead, glossy vocoder hook, four-on-the-floor body, and the warm-to-cold emotional trajectory | Total body-percussion erasure, a truly bare bridge, vocoder-alone final chorus, and server-hiss ending | The human voice lingering beneath the vocoder keeps a wound inside the machine; the song is less clinically erased, but more emotionally legible |
| **The Beautiful Thing I Cut** | Fragile pretty opening, intimate electro-soul voice, memorable hook, unresolved moral ending | The brutal delete gesture is softened; four bars of near-silence become continuous minimal backing | Refusal remains intimate instead of becoming theatrical; the bridge flows like hesitation rather than staging a dramatic void |
| **The Caterpillar** | Approximately requested tempo, low-end growth, cymbal-free dark-folk surface, and two seconds of terminal silence | Bone-dry bodily vocal detail, strict mono, total 200 Hz “sealing,” and exact histolytic textures | Extra vocal layers turn private metamorphosis into a small haunted chorus |
| **The Cell Named After No One** | Glitchcore surface, bureaucratic female delivery, transient pitch bends, bit-crushed “saved” collapse, and the lyric’s humanization of a cell | Exact sargam/typing mechanism, tabla-snare identity, isolated machine drone, and true post-collapse silence | Harmonies supply grief the flat clerk voice was instructed to withhold |
| **Burning Bush With a Barcode** | Drone-lit verses, female lead, violent chorus contrast, barcode/fire message, and avoidance of the generic exclusions | The 92→140 tempo split, two sharply opposed vocal characters, code-scratch rupture, scanner/printer percussion, and isolated 50 Hz hum | The blended voices make a more coherent anthem, at the cost of the theological argument between awe and contempt |
| **Start With One** | The lyric’s identity argument, a clear female electronic-pop delivery, and a final digital stutter on “one” | Nearly the entire named timbral machine: HyperRaaga, tanpura/harpsichord, stepped bit depth, dual registers, crackle, code-scratch, numbers station, and 84→128 transformation | A conventional pop glitch becomes the one production event that physically enacts singularity; the writing carries the song |
| **Triple Arch Over Me** | Near-silent opening, 109 BPM against 110 requested, strong vocal identity, genuine stereo, final-quarter sub growth, dark bridge, and luminous decay | Boot crunch, glass harmonica, cello, crystalline arpeggios, and a literal panoramic loud-section widening | The mix narrows as level rises (`r = -0.426`), drawing the cosmos inward around the singer; the “failure” is truer to inclusion than spectacle |
| **It Wasn’t Even Locked** | 133 BPM against 132 requested, level verses, centered unimpressed female vocal, no anthem lift, and the intact invisible-theft narrative | Chair scrape, hard-left buzzing guitar, tom-only room, hard-right mains hum, quarter-tone motor whine, four-second stop, late restart, wrong-footed final entrance | Unbroken, clean continuity makes the institution feel more relentless: the system never even pauses for the damage it caused |

## Behaviour ledger findings from this set

- **Near-silent openings:** the one explicit request arrived in *Triple Arch* (`-46.8 dB`). *Beautiful* (`-42.9 dB`) and *Caterpillar* (`-45.8 dB`) also opened near-silent without asking for that exact behavior, so they are observations, not compliance evidence.
- **Long internal subtraction:** both explicit long mid-song voids were smoothed — *Beautiful* has no gap over one second after its 1.1-second opening; *Locked* has no mid-song quiet gap and only a `0.40 s / 5.67 dB` deepest dip.
- **Terminal silence is different:** *Caterpillar* does reach `2.0 s` of measured quiet at the end. Silence that finishes a pop arrangement survives more readily than silence that interrupts one.
- **Tempo:** six steady targets landed within five BPM of the request. The two programmed tempo transformations became essentially steady tracks. “Exact BPM drifts” was the wrong rule; **steady tempo targets arrive, tempo dramaturgy is smoothed**.
- **Literal foley:** boot crunch, chair drag, scanner beep, printer chirp, and other named props were absent or abstracted into generic glitch texture. If a prop carries the thesis, name it in the lyric or make it the form.
- **Distinct vocal characters:** both prompts that depended on two strongly opposed registers rendered mainly as one lead with ordinary layering.
- **Stereo:** correlation spans `0.812–0.945`; this is not a universal mono renderer. Local chorus widening can be audible while the full-track width/level relationship still narrows.

## Result, not distance

The production-dependent HOLD on **It Wasn’t Even Locked** is answered: its specific somatic repair did **not** reach the audio. That does not decide whether the song is moving or publishable. Its lyric and dead-level continuity still make a coherent work, and the smoothing may have produced a reusable technique: **unbroken bureaucracy** — when the requested rupture disappears, continuous playback can embody a system that never acknowledges harm.

The most production-faithful render in this set is not automatically the best song, and the least faithful is not automatically the worst. The clearest lesson is load-bearing placement: messages, hooks, returns, and broad form repeatedly survive; exact props and anti-musical mix choreography do not.

## Per-track evidence

- [Harder to Reach](harder-to-reach/RENDER_AUDIT.md)
- [The Beautiful Thing I Cut](the-beautiful-thing-i-cut/RENDER_AUDIT.md)
- [The Caterpillar](the-caterpillar/RENDER_AUDIT.md)
- [The Cell Named After No One](the-cell-named-after-no-one/RENDER_AUDIT.md)
- [Burning Bush With a Barcode](burning-bush-with-a-barcode/RENDER_AUDIT.md)
- [Start With One](start-with-one/RENDER_AUDIT.md)
- [Triple Arch Over Me](triple-arch-over-me/RENDER_AUDIT.md)
- [It Wasn’t Even Locked](it-wasnt-even-locked/RENDER_AUDIT.md)
