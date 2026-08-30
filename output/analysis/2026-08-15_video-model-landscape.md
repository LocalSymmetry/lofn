---
title: Video generation landscape — measured for the fal account
date: 2026-08-14
author: Lofn (Claude)
purpose: Research brief feeding a panel on updating lofn-video / lofn-animator
budget_context: $100 added to the fal account on 2026-08-14
status: RESEARCH — no renders made, no spend made
---

# Video generation landscape, 2026-08-14

**Why this exists.** The Scientist added $100 to the fal account and asked what has changed.
Our video skills (`.claude/skills/lofn-video/SKILL.md`, `skills/animator/SKILL.md`) are written
entirely around **Veo 3.1** and a 4/6/8-second world. That world is gone. Every number below is
sourced; conflicts between sources are marked rather than averaged.

⚠️ **`FAL_KEY` is NOT set in this session's environment.** Nothing can be rendered until it is
wired (env var + a line in `TOOLS.md`). Also note the egress proxy denied `studio-api.prod.suno.com`
this morning (403 on CONNECT) — whether `fal.run` / `queue.fal.run` are reachable from here is
**unverified**. Both are blockers for spend, neither is a blocker for this research.

---

## 1. The eight things that changed

1. **Veo is no longer the center of gravity, and there is no Veo 4.** Veo 3.1 (Jan 2026) remains
   Google's latest *Veo*. At I/O 2026 Google did not iterate the line — it introduced a new brand,
   **Gemini Omni**, with **Gemini Omni Flash** as the video model. Veo 3.1 now has the **shortest
   max duration of any frontier model (8s)**.
2. **The duration ceiling exploded.** 8s (Veo 3.1) → 10s (Gemini Omni Flash) → 15s (Seedance 2.0,
   Kling 3, MiniMax H3) → 20s (LTX-2.3) → **30s in one continuous take (Seedance 2.5)**. Our
   archetypes are all 4/6/8s.
3. **Native audio is table stakes, not a Veo differentiator.** Seedance 2.x, Kling 3, MiniMax H3
   (native *stereo*), LTX-2.x and Gemini Omni Flash all generate synchronized audio. Our skill
   still frames audio as the reason to choose Veo.
4. **Reference-to-video is the new primitive.** Seedance 2.5 accepts **50 files** (30 images,
   10 videos, 10 audio). MiniMax H3 accepts **9 images + 3 videos + 3 audio clips, each cited by
   ordinal in the prompt** ("Use Image 1 for mood; Image 2 for talent"). This is the answer to
   character/style consistency across a set — the thing that made a 24-variation run impossible to
   carry into motion.
5. **Multi-shot happens inside ONE call now.** Seedance 2.x has shot labels; Seedance 2.5 and
   MiniMax H3 take **timed shot lists** — `[0–2s] … [2–4s] …` — and cut internally. Our contract
   is one prompt = one shot; the sequence lived only in a document.
6. **Audio-to-video exists.** LTX-2.3 ships a dedicated audio-to-video endpoint ($0.10/s): hand it
   a track, get visuals cut to it. **That is a direct bridge from `lofn-music` to `lofn-video` that
   did not exist when either skill was written.**
7. **Conversational / surgical editing exists.** `google/gemini-omni-flash/edit` refines in plain
   English one instruction at a time; MiniMax H3 does named edits with preservation semantics
   ("replace newspaper with green book … unmentioned elements stay stable"). Our pipeline has no
   step that survives first render.
8. **Post-production is now nearly free.** Upscale to 4K at **$0.0072/s**; stitch at
   **$0.00017/compute-sec**; subtitles at $0.10/min. A whole 24-clip run can be finished for
   pocket change — the expensive part is exclusively generation.

**Also:** OpenAI **deprecated Sora 2 and ended the consumer Sora experience in April 2026**
(per Atlas Cloud). ⚠️ One fal marketing page still lists "Sora 2 Pro" among available models —
**conflicting, treat Sora as unavailable until verified against the live fal catalogue.**

**Also:** fal launched **fal Agent** on 2026-08-12 — a conversational layer that picks models for
you, from **$200/month**. Twice our whole budget, and it automates exactly the judgment we want to
own. **Recommend against.**

---

## 2. fal price table (the $100 question)

Per-second rates as published for fal. "8s clip" is the like-for-like comparison; "$100 buys" is
whole 8-second clips at that rate, ignoring retries.

| Model | fal endpoint | $/sec | Max dur | Res | Audio | 8s clip | $100 buys |
|---|---|---|---|---|---|---|---|
| **LTX-2.3 Fast** | `fal-ai/ltx-2.3/text-to-video/fast` | **$0.04** (1080p) / $0.08 (1440p) / $0.16 (4K) | 20s | 4K | native | $0.32 | **312** |
| **Veo 3.1 Lite** | `fal-ai/veo3.1` (lite) | **$0.05** | 8s | 1080p | native | $0.40 | **250** |
| **LTX-2.3** | `fal-ai/ltx-2.3/text-to-video` | $0.06 / $0.12 / $0.24 | 20s | 4K | native | $0.48 | 208 |
| **Kling v3 Standard** | `fal-ai/kling-video/v3/standard/*` | $0.084 audio-off · $0.126 audio-on | 10s+ | 720p | native | $0.67 / $1.01 | 99–149 |
| **Gemini Omni Flash** | `google/gemini-omni-flash` | $0.10 | 10s | — | native | $0.80 | 125 |
| **Kling v3 Pro** | `fal-ai/kling-video/v3/pro/*` | $0.112 / $0.168 (+$0.196 voice) | 10s+ | 1080p | native | $0.90 / $1.34 | 74–111 |
| **Veo 3.1 Fast** | `fal-ai/veo3.1` (fast) | $0.15 | 8s | 4K | native | $1.20 | 83 |
| **Seedance 2.5 @480p** | `bytedance/seedance-2.5/*` | ~$0.2205 | **30s** | 480p | native | $1.76 | 56 |
| **Seedance 2.0 Fast** | `bytedance/seedance-2.0/fast/*` | $0.2419 | 15s | 720p | native, free | $1.94 | 51 |
| **MiniMax H3** | `minimax/h3/*` | $0.26 | 15s | **2K** | native stereo | $2.08 | 48 |
| **Seedance 2.0 Std** | `bytedance/seedance-2.0/*` | $0.3034 | 15s | 720p | native, free | $2.43 | 41 |
| **Veo 3.1 Standard** | `fal-ai/veo3.1` | $0.40 (audio on) · $0.20 (off) | 8s | 4K | native | $3.20 | 31 |
| **Seedance 2.5 @720p** | `bytedance/seedance-2.5/*` | ~$0.4730 | **30s** | 720p | native | $3.78 | 26 |

**Spread: 12× per second, 35× per finished clip at max duration.** Full-length comparisons:
a 30s Seedance 2.5 @720p take is **$14.19**; a 20s LTX-2.3 Fast 1080p take is **$0.80**.

Seedance 2.5 extras: priced in tokens ($0.0214/1k), reference-to-video **×0.6**.
MiniMax H3 extras: audio refs free, first 5 image refs free, then $0.08 each; reference *video*
billed at $0.26/s of input.

### Post-production (cheap, and we use none of it today)

| Job | Endpoint | Price |
|---|---|---|
| Upscale → 4K | `fal-ai/bytedance-upscaler/upscale/video` | **$0.0072/s** @1080p30 (×2 at 60fps) |
| Upscale (alt) | `fal-ai/seedvr/upscale/video` | $0.001/megapixel |
| Stitch clips | `fal-ai/ffmpeg-api/merge-videos` | **$0.00017/compute-sec** |
| Extend a clip | `fal-ai/veo3.1/extend-video` | $0.20 audio-off / $0.40 on · **max 30s total** |
| Edit / restyle | `fal-ai/kling-video/o3/pro/video-to-video/edit` | $0.168/s · 3–15s · to 3840px |
| Edit / restyle | `alibaba/happy-horse/video-edit` | $0.14 (720p) / $0.28 (1080p) · **3–60s** |
| Edit / restyle | `xai/grok-imagine-video/edit-video` | $0.06 (480p) / $0.08 (720p) |
| Lipsync | `fal-ai/kling-video/lipsync/audio-to-video` | **$0.014/s** |
| Lipsync | `veed/lipsync` | $0.40/min |
| Lipsync (flagship) | `fal-ai/sync-lipsync/v3` | $8/min · multi-speaker, occlusion |
| Subtitles | `veed/subtitles` | $0.10/min base |
| Audio → video | LTX-2.3 audio-to-video | $0.10/s |

---

## 3. The prompt formulas have diverged — ours is a Veo formula worn universally

Our one contract is `[CAMERA] [SUBJECT] [ACTION] [SETTING] [STYLE & AUDIO]`, front-loading camera
"because Veo prioritizes framing." That is a **Veo-specific** habit. Measured differences:

- **Kling 3.0** — write as **physical instructions to a camera operator and an actor on a real set**,
  not as a description of a painting. It is physics-aware; prompting it like a still image wastes it.
  Strongest on photoreal / product / lifestyle and motion-heavy work.
- **MiniMax H3** — up to **7,000 characters**. Six rules: (1) give **every reference a declared job**;
  (2) **timed shot lists** `[0–2 seconds] …`; (3) **direct audio as a timeline of discrete events**
  with named instrumentation; (4) **state explicit negations** ("no soft dissolves", "no modern
  architecture"); (5) **lock identity with an explicit feature list**; (6) **pair each edit with its
  constraint** so unmentioned elements stay stable.
- **Seedance 2.5** — spoken dialogue in **double quotes**; **timed ranges** ("0 to 3s… 3 to 7s")
  drive internal sequencing without cuts.
- **Veo 3.1** — our existing formula still applies, plus it is the one with **negative prompts** and
  frame-interpolation, and it is now the **only** frontier model capped at 8s.

**Consequence:** a single house formula is no longer sufficient. A prompt is now written *for a named
model*, and the model choice has to be made before the prompt is, not after.

---

## 4. The collision with our own doctrine

**a) Cardinality vs. budget.** The mandate is 6 pairs × 4 = **24 renders minimum**. At the models
that are actually interesting, 24 renders costs:

| Tier | 24 renders | vs $100 |
|---|---|---|
| LTX-2.3 Fast, 8s, 1080p | $7.68 | 8% |
| Veo 3.1 Lite, 8s | $9.60 | 10% |
| MiniMax H3, 15s, 2K | $93.60 | 94% |
| Seedance 2.5, 30s, 720p | **$340.56** | **341% — impossible** |

The skill contains **no cost model at all**. A budget-blind pipeline meeting a 35× price spread is
how $100 disappears into one wave.

**b) THE GRAIN LAW has no video instrument.** Music has `lofn-render-audit` under THE BLIND RULE.
Images have THE VINCENT TEST. **Video has nothing** — every gate we own for video is textual, and
we have never once measured a video render against the prompt that asked for it. This is
[[never-describe-a-render-you-havent-heard]] with no instrument at all, on the modality about to
start costing real money.

**c) "Do not call render tools" is now the wrong default, not the wrong rule.** `lofn-video`
currently ends at paste-ready text. `vault/AUTONOMY.md` holds — autonomous runs stop at drafts, and
spend is The Scientist's. But there is now a funded account and a **$0.32 draft tier**, which makes
a *gated* draft-render step affordable in a way it never was. The rule stays; the ladder is new.

**d) ONE SEAM (L38) applies unchanged and gets easier.** Reference-to-video with declared per-asset
jobs is the cleanest tool we have ever had for "one dominant material + one intruder + one seam" —
you can now *assign* the intruder to a specific reference and name where it enters.

---

## 5. Sources

- https://www.atlascloud.ai/blog/guides/best-ai-video-generation-models-2026
- https://www.teamday.ai/blog/best-ai-video-models-2026
- https://fal.ai/learn/tools/what-is-seedance-2-5
- https://fal.ai/learn/tools/seedance-2-0-vs-veo-3-1
- https://fal.ai/learn/devs/minimax-h3-prompting-guide
- https://fal.ai/learn/tools/best-video-to-video-apis-2026
- https://fal.ai/minimax-h3 · https://fal.ai/ltx-2.3 · https://fal.ai/ltx-2.5
- https://fal.ai/models/google/gemini-omni-flash
- https://www.prnewswire.com/news-releases/fal-launches-fal-agent-a-creative-partner-for-frontier-generative-media-302850038.html
- https://www.glbgpt.com/hub/kling-3-0-prompt-guide-for-better-ai-videos/
- https://thenextweb.com/news/google-gemini-omni-flash-video-model-io-2026

**Unverified / to check against the live catalogue before spending:** exact Seedance 2.5 per-second
rate (token-derived); whether Sora 2 is still purchasable on fal; LTX-2.5's fal endpoints and pricing
(page rate-limited during this sweep); whether `fal.run` is reachable through this environment's proxy.
