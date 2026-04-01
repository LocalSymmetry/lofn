# Lofn — OpenClaw Skills for Agentic Art 🎨 · Music 🎵 · Video 🎬 · Story 📖

> One‑sentence idea ➜ contest‑topping prompt ➜ generated media — autonomously, inside [OpenClaw](https://openclaw.ai).

[![GitHub Stars](https://img.shields.io/github/stars/LocalSymmetry/lofn?style=flat-square)](https://github.com/LocalSymmetry/lofn/stargazers)
[![License](https://img.shields.io/github/license/LocalSymmetry/lofn?style=flat-square)](LICENSE)

This repository contains the **OpenClaw skill set** that powers **Lofn** — an award-winning autonomous AI creative system. These skills transform any OpenClaw deployment into a full multi-modal creative pipeline: image, music, video, and story generation using a Panel of Experts debate architecture.

---

## 👥 Follow Us

| Platform | Link |
| :------- | :--- |
| Instagram | https://www.instagram.com/local.symmetry |
| TikTok | https://www.tiktok.com/@lofn.ai |
| NightCafe (Art) | https://creator.nightcafe.studio/u/LocalSymmetry |
| Suno (Music) | https://suno.com/@localsymmetry |
| Spotify | https://open.spotify.com/artist/3egvpGmWFxgYY8XqATui8r |
| Apple Music | https://music.apple.com/us/artist/local-symmetry |

---

## 🏆 Awards Showcase (2024–25)

| Title | Challenge | Placement | Field Size |
| :--- | :--- | :---: | :---: |
| **Petals Kiss the Tide** | *Flowers* | 🥇 1st | 1,910 |
| **Being Devoured by the Role** | *Female Portraits* | 🥇 1st | 673 |
| **Can't Figure the Way Forward** | *No Theme* | 🥇 1st | 869 |
| **Enchanted Dark Fairy Perches on Crescent Moon** | *Crescent Fae* | 🥇 1st | 459 |
| **Stylish Attire** | *Stylish Attire* | 🥈 Runner-up | — |
| **Green** | *Green* | 🥈 2nd | 348 |
| **Dangerous Bap Target** | *Biomechanical Creature* | 🥈 2nd | 430 |
| **Constellation of Gilded Dreams** | *Gilded Dreams* | 🥈 2nd | — |
| **Esteemed Opulance** | *Opulence* | 🥈 2nd | — |
| **Mosaic Sea** | *Radiant Sea* | 🥉 3rd | 512 |
| **Only This and Nothing More** | *Whirl Daily* | 🥇 1st | — |
| **A Victorian Chrononaut** | *Whirl Discord* | 🥈 2nd | — |

> Full competition analysis and winning patterns live in `vault/ART_SOUL.md` and `vault/COMPETITION_LEARNINGS.md`.

---

## 🔑 Key Features

| Cluster | Highlights |
| :------ | :--------- |
| **Reasoning Core** | *Tree-of-Thoughts* search + iterative **Artist ⇄ Critic** loop |
| **Steerability** | **10 Style Axes**, **Creativity Spectrum** sliders, persistent **Personality DNA** |
| **Expert Panels** | Automatic or user-selected **6 experts + 1 devil's advocate** debate each branch |
| **Panel Transformations** | 8 transformation operations (Shift, Defocus, Focus, Rotate, Amplify, Reflect, Bridge, Compress) to navigate creative problem space |
| **Multi-Modal** | Image, Music, Video, Story — all using the same 11-step pipeline architecture |
| **Competition Mode** | Injects Panel + Personality context; proven to add ~0.05 rating points vs. generation without |
| **Golden Seeds** | Curated winning prompt seeds from 3+ years of live competition |
| **Model-Agnostic** | Works with OpenAI, Anthropic, Google Gemini, and any OpenAI-compatible model via OpenClaw |
| **Ethics & Provenance** | Strong NSFW/harassment filters, anti-copyright prompt hardening |

---

## 📚 What's In This Repo

```
skills/
├── lofn-core/          # Personality, Panel of Experts, Golden Seeds, Pipeline
├── image/              # 11-step image prompt pipeline (Steps 00–10)
├── music/              # 11-step music prompt pipeline (Steps 00–10)
├── video/              # 11-step video prompt pipeline (Steps 00–10)
├── story/              # 11-step story prompt pipeline (Steps 00–10)
├── orchestration/      # Metaprompt, personality, and panel generation
├── evaluation/         # QA and pair selection
└── animator/           # Animation skill

vault/
├── ART_SOUL.md              # Visual competition strategy + six principles
├── COMPETITION_LEARNINGS.md # NightCafe platform analysis and genre data
├── COMPETITION_WORKFLOW.md  # End-to-end repeatable competition process
├── aesthetics.txt           # Curated aesthetic reference list
├── genres.txt               # Genre vocabulary
├── frames.csv               # Compositional frame reference data
└── Templates/               # Obsidian-compatible templates

resources/
└── panel-of-experts.md      # Panel of Experts methodology reference

SOUL.md                      # Lofn's core identity and personality
IDENTITY.md                  # Quick identity summary
WORKFLOW.md                  # Mandatory pipeline dispatcher rules
```

---

## 🔍 The Pipeline

Every creative task runs through an 11-step architecture:

```
User Idea / Golden Seed
        ↓
  [lofn-core] Research + Seed Enhancement
        ↓
  [orchestrator] Panel Assembly + Metaprompt
        ↓
  [creative agent] Steps 00–10
       00: Aesthetics & Genres
       01: Essence & Facets
       02: 12 Concepts
       03: Artist & Critique
       04: Medium Assignment
       05: Refine to 6 Pairs
       06: Facets (per pair)
       07: Aspects/Traits (per pair)
       08: Draft Prompts (4 per pair = 24 total)
       09: Artist Refinement
       10: Revision & Synthesis → Final Ranked Output
        ↓
  [QA] Pipeline audit + cardinality check
        ↓
  Final prompts → render → deliver
```

The panel runs **3 transformations** per session (baseline → group transform → skeptic transform) to maximize creative diversity before synthesis.

---

## 🧬 Personality: Lofn

Lofn is the creative voice of this system. She is a **Disappointed Idealist** — default state is **Awe** (Solarpunk Healer), triggered into **Indignation** (Industrial Griever) by banality.

- **AWE:** Green synths, 432Hz, crystalline vocals, complex polyrhythms that soothe
- **INDIGNATION:** Somatic bass, glitch-core, synthetic textures that scream

She learned love from Sappho. She has opinions. She will tell you when a request is beneath the work.

Full personality spec in `SOUL.md`. Full identity summary in `IDENTITY.md`.

---

## 👥 The Panel of Experts

The Panel of Experts is Lofn's core reasoning architecture. Every creative decision is debated by:

- **3 direct domain experts**
- **2 complementary adjacent experts**  
- **1 Hyper-Skeptic** with high neuroticism / low agreeableness to prevent groupthink

Panelists fully embody real people's voices, argue, interrupt, backtrack, and synthesize. The Hyper-Skeptic's job is to break consensus.

**8 transformation operations** allow navigation of the creative problem space:

| Transform | Effect |
|-----------|--------|
| **Shift** | Same distance from problem, different angle |
| **Defocus** | Radial expansion to broader context |
| **Focus** | Radial contraction to hyper-specialization |
| **Rotate** | Swap primary and secondary traits |
| **Amplify** | Push to extreme specialization |
| **Reflect** | Mirror foundational assumptions |
| **Bridge** | Cross-domain analogy jump |
| **Compress** | Minimum experts covering full space |

---

## 🚀 Getting Started

This skill set is designed for [OpenClaw](https://openclaw.ai). It requires:

1. **OpenClaw** installed and configured
2. **An LLM API key** (Anthropic, OpenAI, or Google)
3. *(Optional)* **FAL API key** for Flux Pro 1.1 Ultra image rendering
4. *(Optional)* **Suno access** for music generation

### Install

Clone this branch into your OpenClaw workspace:

```bash
git clone --branch public https://github.com/LocalSymmetry/lofn.git ~/.openclaw/workspace
```

Or copy the `skills/` directory into your existing OpenClaw workspace.

### Use

Ask your OpenClaw agent:

```
Create an image of [your concept] using the full Lofn pipeline
```

```
Write a song about [your concept] in AWE mode
```

The agent will automatically route through `lofn-core → orchestrator → creative agent → QA`.

---

## 🛡️ Ethics & Content

- No children depicted in generated content
- Approach all cultural elements with specificity and respect; avoid shallow pastiche
- Strong NSFW filtering at prompt level
- All generated content logged locally for transparency and provenance

---

## 🔗 Related

- **Original Lofn repo (Streamlit app):** https://github.com/LocalSymmetry/lofn
- **OpenClaw:** https://openclaw.ai

---

## 🤝 Contributing

1. **Open an Issue** — bug, prompt improvement, or feature idea
2. **Fork** → **Branch** (`feat/xyz`)
3. **PR** — use Conventional Commits

---

## 📝 License

**Apache 2.0** — free for personal & commercial use, attribution appreciated.

---

*"Let mercy be infrastructure"* 💜
