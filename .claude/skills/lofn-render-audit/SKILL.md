---
name: lofn-render-audit
description: Audit a FINISHED audio render against the prompt that asked for it — numeric measurement plus a listening-model pass — to find out which intents survived the generator, which were smoothed away, and which deviations are better than what was specified. Use after rendering Suno output, when asked "did it survive my intents", to decide a HOLD that depends on production, or to accumulate the Suno behaviour ledger. Do NOT use for text-only QA (that is lofn-qa) or for generating anything.
---

# Lofn Render Audit — what the generator actually did

> Every gate in this pipeline reads **text**. Until 2026-07-24 nothing had ever measured a finished **render**, so a whole class of failure was invisible by construction: a song can pass all 16 points of the Suno gate and arrive with its defining gesture gone.

This skill is **judge-side**. Golden song payloads are permitted here. It generates nothing.

---

## ⚖️ THE GRAIN LAW — the finding this skill exists to apply

Three finished tracks measured on 2026-07-24 (two new, one Staff Pick):

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

## 📓 THE SUNO BEHAVIOUR LEDGER — the deep dive, accumulated

The point of running this repeatedly. One row per **spec class**, not per song; update the evidence count as tracks accrue. This is how we stop guessing what the renderer does.

| spec class | behaviour | n | confidence |
|---|---|---|---|
| near-silent opening | **arrives** | 1 | low — needs more |
| progressive sub build to final chorus | **arrives** (+8.4 → +11.4 dB) | 1 | low |
| long full stop (≈4 s) mid-song | **smoothed** to <0.5 s | 1 | low |
| sustained untuned drone / mains hum | **absent** | 1 | low |
| hard-panned non-musical element | **absent**; render collapses toward mono | 2 | low-medium |
| "chorus widens" | **inverted** — narrows into loud sections (r = −0.43) | 1 | low |
| exact BPM | **drifts** (110 asked → ~120 rendered) | 1 | low |

**Method:** never raise a row above `medium` on fewer than 4 tracks; keep `n` visible so nobody theologises a sample of one — the mistake this ledger's first version made by declaring "Suno renders mono" from two tracks that both happened to fight the grain.

---

## WHEN NOT TO USE THIS
- Text-only QA → `lofn-qa`.
- Generating or repairing lyrics/prompts → `lofn-music`.
- **Never** to compute a compliance percentage, and never to overrule a human ear that likes the track.

*The render is the work. The prompt was only ever the request.* 💜
