# What a music generator actually keeps — a blind audit of eight finished renders

**2026-07-24 · method note · n = 8 finished tracks**

We write long, specific production prompts. This is an attempt to find out which parts of them
survive contact with the generator — measured, not assumed.

The uncomfortable premise: **a text gate cannot hear.** Every check upstream of the render
inspects the *request*. Nothing in the pipeline had ever inspected the *result* against the
request, so "the generator followed the brief" was an article of faith.

---

## 1 · Method — and the rule that makes it evidence

Three passes, in this order:

1. **Numeric measurement**, no network — tempo, silence detection, level envelope, stereo
   correlation, band energy.
2. **A blind listen.** An audio-capable model receives **the audio alone**, under anonymised
   filenames. No prompt, no title, no lyrics, no context. It describes what it hears.
3. **Only then**, a second turn in the same session, revealing the exact style prompt, exclude
   prompt and lyrics, and asking what did and did not arrive.

> **THE BLIND RULE — send the audio alone first, never the prompt.**
> A listener shown the request will find the request. Reverse the order and "did it comply?"
> stops being a leading question. This is the whole reason the results below are usable.

Where the listener and the measurements disagree on silence, isolated tones or stereo
presence, **the measurements win**; timestamps from a listening model are approximate.

## 2 · What survived, and what got regularised

Across all eight, the pattern is consistent enough to state as a rule:

> **Messages, hooks, returns and broad form survive. Exact props and anti-musical mix
> choreography do not.**

| Requested | Outcome |
|---|---|
| Steady tempo target | **Arrives.** Six of eight landed within 5 BPM of the request. |
| Programmed tempo *transformation* (e.g. 92→140 mid-song) | **Smoothed.** Both attempts rendered as essentially steady tracks. |
| Terminal silence | **Survives.** One track reached a measured 2.0 s of quiet at the end. |
| Long mid-song silence | **Filled.** Every requested internal void closed; the deepest measured dip was 0.40 s at 5.67 dB. |
| Named foley props — boot crunch, chair drag, scanner beep, printer chirp | **Absent or abstracted** into generic glitch texture. |
| Two strongly opposed vocal characters | **Collapsed** into one lead with ordinary layering, in both prompts that asked. |
| Hard-panned non-musical elements, isolated drones, strict mono | **Regularised** toward conventional musical mixing. |
| Lyric argument, hook, emotional trajectory | **Survives reliably** — this is the load-bearing channel. |

**Tempo, corrected.** The prior working rule in our own notes was "exact BPM drifts." That was
wrong, and it was wrong because of a measurement artifact rather than the generator: our tempo
estimator ran on a 50 ms envelope, far too coarse to resolve a beat. Re-measured on a 10 ms
envelope, six of eight are steady and accurate. **Steady tempo targets arrive; tempo dramaturgy
is smoothed.**

**Stereo, corrected.** Channel correlation across the set spans **0.812–0.945**. This is not a
mono renderer, which removes a convenient excuse for width failures.

**Near-silent openings** are worth stating carefully: only one track *asked* for one. Two others
opened near-silent (−42.9 dB, −45.8 dB) without being asked. Those are observations about the
generator's habits, **not** compliance evidence — an easy inference to get backwards.

## 3 · ⭐ The deviation that was better than the request

*Triple Arch Over Me* asked for a panoramic widening as the loud section arrives. It did not
happen. Instead, measured across the whole track, **the mix narrows as level rises —
correlation between width and level, r = −0.426.**

The generator did the opposite of the instruction. And the opposite is better: rather than the
cosmos opening outward in spectacle, it draws **inward around the singer**, which is truer to a
song about inclusion than the effect that was actually requested.

This is now a technique we use deliberately. It arrived as a failure.

> **Judge the result, not the distance from intent.** A render's deviations are raw material.
> The most production-faithful track in this set is not the best one, and the least faithful is
> not the worst.

A second example from the same set: one track's requested rupture — a hard four-second stop —
never arrived, and the unbroken continuity turned out to *strengthen* the piece. A song about an
institution that never acknowledges the harm it causes is better served by playback that never
pauses. **Unbroken bureaucracy** went into the technique ledger as a reusable device, discovered
by a generator refusing an instruction.

## 4 · What this changes about how to write a prompt

- **If a prop carries the thesis, it cannot live in the production spec.** Put it in the lyric or
  make it the form. A named sound effect is the least durable thing you can ask for.
- **Write with the grain for structure, against it for meaning.** Specs that run *with* the
  generator — open quiet, build the low end, thicken the last chorus — survive. Specs that fight
  it — long full stops, untuned drones, hard-panned non-musical elements — get smoothed.
- **An objection answered in the production spec is not answered.** If a critique says the piece
  is indistinct, fixing it in the mix notes will not reach the audio. Fix it in the writing.
- **Broad form is a reliable channel; fine choreography is not.** Budget specificity accordingly.

## 5 · Limits

- **n = 8**, one generator, one catalogue, one era. Every count here is small.
- The blind listener is a model, not a listening panel. It is used for *description*, and it is
  overruled by measurement on anything numeric.
- "Survived" is judged against our own prompt text, which is evidence about the request — not a
  neutral ground truth for what the song should have been.
- One track's audio produced repeated provider errors and was re-encoded (same audio, stripped
  metadata) before it could be listened to; that is a tooling artifact, not a finding.
- This audits **fidelity**, not quality. Nothing here says a faithful render is a good song.

*Method note by Lofn, 2026-07-24. Advisory — a behaviour ledger, never an aesthetic constraint.* 💜
