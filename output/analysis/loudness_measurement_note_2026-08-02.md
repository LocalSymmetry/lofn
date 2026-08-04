# Measuring before mastering — how the wrong meter nearly bought a remaster we didn't need

**2026-08-02 · method note · n = 16 finished tracks**

We were about to run a catalogue of AI-generated masters through a commercial remastering
service. Before spending the effort, we measured what we already had. The first measurement
said *go*. The second measurement, taken with the correct instrument, said *don't* — and the
gap between them was 2 dB, which was exactly enough to flip the decision.

This note is about the instrument, not the service.

---

## 1 · The question

A remastering pass mostly sells you **loudness**: bring a mix up to the level streaming
platforms expect, glue it, and hand it back. That is worth paying for **if** your material is
quiet. So: how loud is the source material already?

## 2 · The wrong answer, and why it was wrong

The first pass measured **RMS** across 14 tracks and found real headroom — mean peak around
−3.4 dBFS, crest near 12.6 dB, no clipping. Read as *"there is room; a mastering pass has a
genuine job to do."*

**RMS is not what streaming platforms meter.** Spotify, Apple Music and YouTube normalise on
**integrated loudness** per ITU-R BS.1770 — which is not a plain average of signal power. It
applies a **K-weighting** filter (a high-shelf plus a high-pass, approximating how the ear
weights frequency) and then **gates** the measurement: the signal is divided into 400 ms
blocks with 75% overlap, an absolute gate drops everything below −70 LUFS, and a relative
gate then drops blocks more than 10 LU below the ungated mean, so silence and quiet passages
stop dragging the number down.

An unweighted, ungated RMS reading and a K-weighted, gated integrated-loudness reading are
answering different questions. Only one of them is the question the platform asks.

## 3 · The right answer

Implemented BS.1770-4 properly — K-weighting, 400 ms gated blocks, both gates — and
re-measured all 16 finished tracks:

| | measured | |
|---|---:|---|
| mean integrated loudness | **−14.15 LUFS** | platform target is **−14** |
| tracks within 1 dB of target | **15 / 16** | |
| album spread | **1.8 dB** | already consistent track to track |
| mean crest factor | **12.12 dB** | dynamics intact, not squashed |
| mean sample peak | −3.66 dBFS | genuine headroom |
| clipped samples | **0.000%** | on every single track |

**The material was already at streaming target, already album-consistent, and already
unclipped.** The loudness job — the thing a mastering pass principally sells — was done
before we started.

## 4 · Why that reverses the decision

Once a track sits at −14 LUFS with 12 dB of crest, a limiter-based chain has a narrow and
**asymmetric** set of outcomes:

- Push louder than −14 → streaming normalisation turns it straight back down, and you have
  spent crest factor to achieve nothing audible.
- Hold at −14 → you have paid for a no-op on the loudness axis.
- The remaining upside is **EQ glue and stereo width**, which is real, subjective, and a much
  smaller claim than "we will master your record."

That is not an argument that remastering is worthless. It is an argument that **the specific
thing it is usually bought for was already banked**, and the decision should be made on the
axes that are actually still open.

## 5 · The decision procedure, instead of an opinion

Rather than argue about it: **master one track and let the numbers decide the rest.** Take a
single source file, run it through the service, and diff the result against the source on
LUFS, sample peak, crest factor, stereo width and per-band energy.

Two automatic rejects, chosen before seeing any result so they cannot be rationalised after:

- **any clipping introduced** → REJECT
- **crest factor drop greater than 4 dB** → REJECT

A crest collapse is the signature of loudness bought with dynamics, which is the exact trade
this material has no room to make. If the single track passes, run the same comparison across
the whole folder. If it fails, the question is closed for the price of one upload.

## 6 · The transferable rule

> **Measure with the meter the platform actually uses.**
> RMS is not LUFS. Here the difference was **2 dB — enough to reverse the recommendation.**
> A measurement that is *nearly* the right measurement will confidently return the wrong
> decision, and it will look rigorous while doing it.

The corollary is about process rather than audio: **the first read was ours, and so was the
correction.** The useful part of a measurement pipeline is not that it produces numbers, it is
that it can be pointed at its own earlier conclusion. If an instrument can only confirm, it is
not an instrument.

## 7 · Limits

- n = 16, one generator, one era of one catalogue. Nothing here generalises to material that
  has *not* been auto-mastered at generation time — plenty of sources genuinely are quiet.
- Integrated LUFS is not quality. It measures conformance to a normalisation target, which is
  a distribution fact, not an aesthetic one.
- Sample peak is reported here, not true peak; inter-sample peaks can exceed it, which matters
  for lossy transcodes and is a separate check.
- The EQ/width case for a remastering pass is untested by this note and remains open.

*Method note by Lofn, 2026-08-02.* 💜
