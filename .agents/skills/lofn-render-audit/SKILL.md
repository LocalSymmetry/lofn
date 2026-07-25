---
name: lofn-render-audit
description: Audit a FINISHED audio render against the prompt that asked for it — numeric measurement plus a listening-model pass — to find out which intents survived the generator, which were smoothed away, and which deviations are better than what was specified. Use after rendering Suno output, when asked "did it survive my intents", to decide a HOLD that depends on production, or to accumulate the Suno behaviour ledger. Do NOT use for text-only QA (that is lofn-qa) or for generating anything.
---

# Lofn Render Audit — what the generator actually did

> **⚖️ AUTHORITY (2026-07-24):** the `.claude/skills/lofn-render-audit/` twin is the CANONICAL policy source; this Codex mirror binds to it. On any disagreement, the `.claude` file wins.

> Every gate in this pipeline reads **text**. Until 2026-07-24 nothing had ever measured a finished **render**, so a whole class of failure was invisible by construction: a song can pass all 16 points of the Suno gate and arrive with its defining gesture gone.

This skill is **judge-side**. Golden song payloads are permitted here. It generates nothing.

---

## ⚖️ THE GRAIN LAW — the finding this skill exists to apply

**The founding comparison** — the first three tracks measured, 2026-07-24 (two new, one Staff Pick). It is kept because it is the case that produced the law; the current evidence base is the **n=8 ledger** below, which supersedes any number here.

| specified | *It Wasn't Even Locked* | *Start With One* | *Triple Arch Over Me* |
|---|---|---|---|
| L-R correlation | 0.945 (mono) | 0.908 (mono) | **0.812 (real stereo)** |
| near-silent opening | — | — | **survived, −41 dB** |
| "more sub at the end" | — | — | **survived, +8.4 → +11.4 dB** |
| ~4 s full stop | **0.40 s / 5.7 dB** | — | — |
| quarter-tone drone pair | **zero tonal components** | — | — |

**The rule is not "production never survives."** My first draft of this finding said that, from a sample of two, and the benchmark disproved it.

> **Specs that run WITH the generator's grain survive. Specs that run AGAINST it get smoothed.**
>
> **With the grain** — things a pop arrangement wants to do anyway: open quiet, build sub, thicken toward the last chorus, widen a pad, add air. These render, often beautifully.
> **Against the grain** — anti-musical instructions: stop the take dead for four seconds, hold two untuned drones a quarter-tone apart, hard-pan a mains hum, refuse to correct a limp. These get smoothed, because smoothing is what the model does.

**Consequences that bind the writing tiers:**
1. A **Somatic-Gate or distinctiveness objection answered in the production spec is not answered.** It must be answered in the **lyric** or the **form**, which survive. (On 2026-07-24 a `REPAIR — 2/3` was closed with a mix decision that never reached audio.)
2. A hollow centre whose mechanism is *"the mix collapses to mono"* is **unfalsifiable** when the render is already mono. Pick a mechanism the render can carry — or accept that it lives on the page.
3. Spatial language is **cheap to write and usually free of consequence.** Do not spend the character budget on it or let a song depend on it.

---

## 🎨 THE SCIENTIST'S LAW — judge the result, not the distance from intent

*Stated 2026-07-24 and load-bearing for this whole skill:*

> *"The generators are imperfect and rarely do your actual prompts justice. I judge the final result, not the distance from intent. As long as the message survives and the song sounds amazing, it is worth publishing. For art, just having a moving piece, even if unintended, is a worthy discovery. Finding out and using the generator's flaws as techniques — having it fail in just the right way — can create new sonic experiences."*

**This skill must never be used to compute a compliance score.** An intent-vs-render diff is **raw material**, not a grade. Three standing rules:

- **A deviation is a finding, not a fault.** Report it neutrally; let the human's ear rule.
- **Hunt for the productive failure.** On the benchmark, width correlates **−0.43** with level: it *narrows* as it gets loud, the exact inverse of the specified "chorus widens like a panorama." For a song whose thesis is *"I am not the center, I am included,"* narrowing into the climax is arguably truer than widening. **That inversion is a technique now.** Log deviations that improved the piece into `vault/COMPETITION_LEARNINGS.md` as reusable moves.
- **A song that lost every production gesture and still moves the listener is evidence about where the load belongs** — the writing — not evidence of failure.

---

## RUN IT

### Pass 1 — numeric (always; no network, no key)
```bash
pip install numpy soundfile          # libsndfile >= 1.1 decodes MP3
python3 scripts/measure_render.py <track.mp3> [...]      # JSON to stdout
```
Reports: duration · peak/crest · loudness spread · opening level · sustained quiet gaps · **deepest mid-third dip (depth AND width)** · tempo candidates · per-quarter band energy · stereo correlation & side/mid · **width-vs-level correlation** · isolated sustained tones · interval pairs.

⚠️ **Trust its guards, and keep them.** An earlier version reported six quarter-tone drone pairs that were adjacent FFT bins in one low-frequency cluster — at 60 Hz the neighbouring bin *is* ~3 % away, so the result was arithmetic. The prominence floor and the 50-cent separation exist because of that. **A validator confident enough to fail an artifact deserves at least as much suspicion as the artifact.**

### Pass 2 — listening model (when a key and egress exist)

**Backend:** GPT-5.6 (audio-capable) via the Poe OpenAI-compatible endpoint `https://api.poe.com/v1` with `POE_API_KEY` (see `TOOLS.md`), or any audio-capable endpoint available in the environment. *Neither key nor egress is available in the sandboxed web session — run this pass locally, or pass the audio through whatever channel the session does have.*

**🔒 THE BLIND RULE — the whole validity of this pass depends on it.**
**Send the audio ALONE first. Do not send the prompt, the title, the lyrics, the intent, or this skill's expectations.** A model shown the intent will confirm the intent. This is the same clean-context discipline QA runs under, for the same reason.

**Prompt for the blind pass:**
> Listen to this track and describe only what you actually hear. Do not guess at intent.
> 1. Structure with timestamps — where sections begin and end.
> 2. Any silence or near-silence longer than one second — where, and how long.
> 3. Sustained non-musical sound (hum, drone, whine, motor, machine noise) — where, and roughly what pitch.
> 4. Stereo image — what sits left, right, centre; does the width change, and where?
> 5. The vocal — register, distance from the mic, dry or reverberant, doubled or single.
> 6. The single most distinctive sonic event in the track, and its timestamp.
> 7. Anything that sounds like an artifact, a glitch, or a mistake — and whether it works.
> 8. In one sentence: what is this song about, from the audio alone?

**Then, and only then**, a second turn with the style prompt, exclude prompt and lyrics attached:
> Here is what was asked for. For each specified element, say: ARRIVED / PARTIAL / ABSENT / INVERTED, with the timestamp that justifies it. Flag anything the render does that was **not** asked for and is **better** than what was.

**Cross-check the listener against Pass 1.** Item 2 must agree with `quiet_gaps` and `deepest_mid_dip`; item 3 with `sustained_tones_hz`; item 4 with `stereo` and `width_dynamics`. **Where they disagree, the numbers win on presence/absence and the listener wins on quality and meaning.** A listener claiming a four-second silence the envelope does not show is confabulating; an envelope showing a dip the listener never noticed is a dip that does not matter.

### Pass 3 — write it down
- Per-track audit → `output/<run>/RENDER_AUDIT.md`: the intent table (ARRIVED/PARTIAL/ABSENT/INVERTED), the numbers, the listener's blind read, and **the productive deviations**.
- **Any deviation worth reusing** → `vault/COMPETITION_LEARNINGS.md` as a named technique.
- **Any generator behaviour that recurs** → the Suno behaviour ledger below.
- Operational failures → `vault/RUN_LEDGER.md`.

---

## 📓 THE SUNO BEHAVIOUR LEDGER — n=8, first full deep dive (2026-07-24)

Eight finished renders, numeric + blind listen. **This table replaces the n=1/n=2 guesses that preceded it; two of those were wrong.**

| spec class | behaviour | n | confidence |
|---|---|---|---|
| **steady tempo target** | **ARRIVES** — 6 of 6 landed within 5 BPM (110→109, 132→133) | 6 | **medium-high** |
| **tempo dramaturgy** (programmed 92→140, 84→128) | **SMOOTHED to a steady track** | 2 | medium |
| near-silent opening | arrives (−46.8 dB when asked; also appears unrequested) | 3 | medium |
| **terminal** silence (ends the arrangement) | **ARRIVES** — 2.0 s measured | 1 | low-medium |
| **interrupting** silence (long mid-song void) | **SMOOTHED** — both explicit voids vanished | 2 | medium |
| progressive sub build to final chorus | arrives (+8.4 → +11.4 dB) | 1 | low-medium |
| literal foley props (boot crunch, chair drag, scanner beep, printer chirp) | **ABSENT or abstracted to generic glitch** | ≥4 | medium |
| two strongly opposed vocal registers | **collapses to one lead + ordinary layering** | 2 | medium |
| named exotic timbres (HyperRaaga, tanpura, glass harmonica, cello, crystalline arps) | largely **absent**; atmosphere substituted | ≥3 | medium |
| sustained untuned drone / mains hum | absent | 2 | low-medium |
| hard-panned non-musical element | absent | 2 | low-medium |
| stereo image | **spans 0.812–0.945 correlation — NOT a universal mono renderer** | 8 | medium-high |
| "chorus widens" | inverted globally (r = −0.43) while local layer-spread is audible | 1 | low |

**Corrections this set forced on my earlier claims:**
- *"Exact BPM drifts"* was **wrong** — an artifact of a 50 ms analysis hop (a beat at 110 BPM is 11 frames). Steady tempos arrive; only tempo *dramaturgy* is smoothed. `measure_render.py` now runs tempo on a 10 ms hop.
- *"Suno renders mono"* was **wrong** — two samples that both fought the grain.
- **Silence is not one behaviour.** Silence that *finishes* an arrangement survives; silence that *interrupts* one gets filled. That distinction is the single most useful row here.

**Method:** keep `n` visible; nothing rises above `medium` under 4 tracks. Two rules already died of a sample of one or two.

---

## 🔧 PRODUCTIVE DEVIATIONS — the flaws worth reusing

*The Scientist's award-winning thought, made concrete. These are techniques now, harvested from failures.*

- **UNBROKEN BUREAUCRACY** — a requested rupture disappears, and continuous playback becomes a system that never pauses for the damage it caused. (*It Wasn't Even Locked*: the four-second stop vanished; the dead-level continuity is more relentless than the break would have been.)
- **INWARD COSMOS** — the mix narrows as level rises instead of widening, drawing the scale in around the singer. Truer to inclusion than spectacle. (*Triple Arch Over Me*, r = −0.426.)
- **THE WOUND INSIDE THE MACHINE** — a human voice left audible under a vocoder that was supposed to erase it; less clinical, more legible. (*Harder to Reach*.)
- **GRIEF THE CLERK WITHHELD** — harmonies supply the feeling a flat bureaucratic delivery was instructed to refuse. (*The Cell Named After No One*.)
- **THE HAUNTED CHORUS** — unrequested vocal layers turn a private metamorphosis into a small crowd. (*The Caterpillar*.)
- **SINGULARITY BY GLITCH** — an ordinary pop stutter becomes the one production event that physically enacts the word "one". (*Start With One*.)

**How to use these:** they are not instructions to Suno — asking for them would put them back on the wrong side of the grain. They are **things to recognise and keep** when a render hands them to you, and reasons to listen before repairing.

## WHEN NOT TO USE THIS
- Text-only QA → `lofn-qa`.
- Generating or repairing lyrics/prompts → `lofn-music`.
- **Never** to compute a compliance percentage, and never to overrule a human ear that likes the track.

*The render is the work. The prompt was only ever the request.* 💜
