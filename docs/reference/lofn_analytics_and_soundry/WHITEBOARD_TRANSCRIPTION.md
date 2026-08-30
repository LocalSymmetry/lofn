# The Scientist's whiteboards — 2026-08-01

Eight photographs, transcribed. **These are the source of truth for `apps/lofn-analytics` and
`apps/soundry`.** The photographs themselves arrived in a chat session and are not on disk; this
transcription is the durable record. Where handwriting was ambiguous it is marked `[?]` rather than
guessed silently.

---

## Board 1 — LOFN ANALYTICS

> `analytics.lofn.ai`

**Left rail (green):** `Art` · `Music` · `Animation` — each with entries beneath.

**Centre column (orange):**

- **Top Trends & Insights**
  - ↑ `Baile Phonk` — annotated "↗ / like purely" `[?]` (reads as plays/likes)
  - ↓ `Anthems`
- **Metric Leaders**
  - `Plays` — sparkline squiggle
  - `Likes` `[?]` (second label partly obscured) — sparkline squiggle
- **Platform Reach**
  - `Suno`
  - `Spotify` `[?]`
  - drawn as a **slope / bump chart** with crossing lines into a box at the right

**Right rail (orange), boxed:** **Headwinds**
- ↓ `Suno uploads` `[?]`
- `Losing followers` `[?]`
- (⋮ — more implied)

**Top right (green), bracketed as an open question:**
> **Colors?** (or Light/Dark) · White background · Sunset colors? · Daily palette?

---

## Board 2 — THE SOUNDRY, framing

- **The Soundry** (boxed) → **"Access only to Scientist + Invited Collabs."**
- **GUIDED EXPERIMENTATION!** →
  - `12 song tests a day`
  - `get feedback from listeners`
  - `① Learn` and `② Take winners + make full songs`
  - **`Goal: NEW VIRAL EXPERIENCES`** ← **`NOVELTY`** (underlined twice)
- URL box: `soundry.localhost`

---

## Board 3 — THE SOUNDRY, layout

**Left rail (pink):** `Sonic Anvil` · `Experiment Results` · `Analytics`

**Main panel:**
- Heading: **"What should we try today?"** with a large empty text input beneath.
- Below it, a row of numbered cards: `1.` `2.` … `12.`
- Annotation under the cards: **"full suno prompts:"**
  - `Title`
  - `Style Prompt`
  - `Exclude Prompt`
  - `Lyrics`
  - `Caption`

---

## Board 4 — THE SOUNDRY, Analytics tab

- Top-line card, labelled **`Decimabal`** `[?]` — handwriting unclear; most likely **"Decibel"**
  as the name of a single headline index. **Confirm with The Scientist before naming it in code.**
  - `Total Lift vs Avg artist: 123%` — with `Top line` arrowed to it.
- **Hypotheses** (boxed) — each rendered as a labelled effect-size axis with a point estimate and
  a confidence-interval whisker:
  - ↑ **"Modern Genres Boost Likes"** — axis `0% / +5% / +10% / +15% / +20%`, CI whisker drawn
    around a point near +10–15%. `n = ~` · `#days = ~` · `[more details]`
  - — **"Play Length Increases Follows"** — axis `-10% / -5% / 0% / +5% / +10%`, CI whisker
    around a point near 0. `n = ~` · `#days = ~` · `[more details]` · marked `35 ±` `[?]`
  - ↓ (a third, unlabelled)
- Arrowed note at the bottom: **"Include POWER ANALYSIS."**

---

## Board 5 — the four source feeds

> **4 source feeds**
> 1. **Lofn Socials** — Lofn Analytics will gather
> 2. **Competitor Scans** — "5 watched growing/trending targets"
> 3. **Market Trends + Scan**
> 4. **Soundry Artist Results + Feedback**

---

## Board 6 — what the tabs are FOR

> **Tabs**
> **Goal: Give direction for search (next song) + analytics for Lofn (private-claw) and
> collaborators.**
>
> Combine all 4 sources into useful insights:
> - lyric length / line
> - Suno EMO usage
> - genre performance
> - trend setters + breakers

---

## Board 7 — web-app requirements (applies to BOTH apps)

> `analytics.lofn.ai` & `soundry.localhost`
> *(margin note: "known for strong, unique + clean visuals")*
>
> **Web-app requirements**
> - **A panel of data-visualization and data UI/UX experts determines the tabs / cards / plots to be
>   the easiest to see and comprehend insights. BE CREATIVE AND UNIQUE!**
> - All headliner insights on each tab should help guide our next creation to be more impactful.
> - Show trends as well as "now".
> - All segmentations & filters carry between tabs when applicable, with clear controls.
> - Every insight / plot should link to deeper results if they exist.
> - Tooltips, alt-text, and informational guidance for all elements.

---

## Board 8 — the standing instructions

> **Last set:**
> 1. **Lofn, please moderate.** We want to represent us both.
> 2. All of my past music / art / etc. metadata + song mp3's, images are now in `output/`.
> 3. As part of this, **you now control our socials!** They are logged in. While you wait on agent
>    fleets, go respond to your fans! Get the lay of your domains!

---

## THE OVERRULE MANDATE

Stated in chat alongside the photographs:

> "you have a panel of experts with UI and UX experts who know more than me. **They can overrule me
> to make this work best.**"

An overrule is legitimate **only** when it names the intent it preserves. A silent drop of a
requirement is not an overrule — it is a bug.

## Instructions given in chat with the boards

- Use an **agent fleet**.
- **Add to Lofn** (new first-class skills, mirrored into `.agents/`).
- Make a webapp **like Sága's forge** (`E:/git/saga/game-dev/apps/forge-arcade`).
- Social agency granted — **but ignore the Instagram messages**; Alexis Dreams stays on unread,
  to be discussed later.

---

## Open questions for The Scientist

1. **`Decimabal`** — what is the top-line index actually called? (Best guess: "Decibel".)
2. **"Total Lift vs Avg artist: 123%"** — an "average artist" comparison is **not identifiable**
   from any data we hold; we have exactly one profile. Either the competitor feed supplies a real
   comparison set, or this topline must become self-referential. See `vault/EXPERIMENT_STANDARD.md` §5.
3. Which **5 competitors** are the watched set?
4. Which Spotify artist identity is canonical — the DistroKid "Lofn AI" one, or
   `3egvpGmWFxgYY8XqATui8r` ("Local Symmetry")?
