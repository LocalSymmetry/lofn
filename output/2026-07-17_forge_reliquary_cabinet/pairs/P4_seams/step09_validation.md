# STEP 09 — PRE-GENERATION VALIDATION + VERIFICATION BENCH
## PAIR P4 — THE SEAMS (AMBITIOUS) · kintsugi 5-state overlay system
*GPT_I2 mode: Step 09 is a pre-generation VALIDATION pass, not artist-voice refinement (renderer rules §09). The prompt IS the final artifact until rendered — no chained edits.*

---

## A. PRE-COMMIT GATE (adapted for a UI asset-pack — the 7-style scene formula is N/A here)
*Rationale: the renderer's §08 "7 explosive art styles in one container" gate is for competition single-scenes. P4 ships a state-machine overlay system, so the gate is re-derived from this pair's binding requirements. Logged so no one thinks the scene-gate was skipped by accident.*

| Check | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| Single named light = gold-enters-from-the-crack (self-emission, no external key) | ✅ | ✅ | ✅ | ✅ |
| Named materials, physical confidence (molten gold ink · cracked egg tempera · craquelure lip) | ✅ | ✅ | ✅ | ✅ |
| Knockout velvet-black ground stated for alpha / transparency explicit | ✅ | ✅ | ✅ | ✅ |
| Overlay wraps a frame (or declares its void) | ✅ slices | ✅ rect | ✅ crop | ✅ vessel |
| 2–4 tones, counted | 4 | 3 | 4 | 3 |
| 128px read stated for every relevant state | ✅ all 5 | ✅ | ✅ | ✅ |
| Greyscale geometry-ladder legibility preserved (thickness+pattern, not color) | ✅ | n/a¹ | n/a¹ | n/a¹ |
| Storybook override present + specific | ✅ | ✅ | ✅ | ✅ |
| Zero veto words (ethereal/dreamlike/whimsical/gentle light/soft glow/magical/delicate) | ✅ | ✅ | ✅ | ✅ |
| Zero artist names | ✅ | ✅ | ✅ | ✅ |
| Zero forbidden negation contamination (no "no lighthouse"-style negatives that summon the thing) | ✅ | ✅ | ✅ | ✅ |
| Self-contained, one-generation (Reiteration Bug respected) | ✅ | ✅ | ✅ | ✅ |
| Resolution set, aligned to /16 | 1536×1024 | 1024×1024 | 1024×1024 | 1024×1536 |

¹ V2/V3/V4 isolate one state; the full-ladder greyscale proof lives on V1 (the reference), which all three inherit.

**GPT_I2 §09 five-point pass:** (1) pre-commit re-run — clean · (2) zero artist names — verified by scan · (3) zero forbidden negation language — verified · (4) Storybook override present & specific in all four — verified · (5) self-contained, no downstream iteration expected — confirmed. **GATE: PASS.**

---

## B. VERIFICATION BENCH — NORMAN / KARE / CHAYKA (skeptics dissent in earnest)

### NORMAN (Shelf Cynic — usability states are the beautiful states; THE LADDER IS HIS TEST)
The whole pair lives or dies on his desk. Greyscale-desaturate the ladder and read left to right: **broken thread · beaded thread · bright cage · bar-with-one-blob · calm solid line.** Five different shapes by thickness and pattern — verdict **PASS**, with one catch he raised and one he confirmed:
- **CATCH (fixed, see reroll R1):** at 128px his first read of dormant vs waking was "two faint threads" — the beads shrank out and the dashes blurred to a solid. Ladder integrity was failing at thumbnail exactly where accessibility matters most.
- **CONFIRM:** the lit-lattice (state 3) is the thickest, most cross-linked geometry and reads as a cage/ring — it earns its double duty as the focus indicator at a 48px target. "The beautiful state IS the usable state. Approved."

### KARE (Pixel Rococo — 128px, four shapes, light source named)
"A seam is already a line drawing, so thumbnail survival is the easy part — my worry is *what the shape reads AS.*"
- **CATCH (fixed, see reroll R2):** V3's lone bead on black read as a gem/orb at 128px — primary-read inversion (a jewel, not a seam; and a jewel is slop). Demanded the bead be anchored to a channel.
- **CONFIRM:** with the channel bleeding off two edges, V3 now reads as "bar bearing one blob" — the pressed state, unmistakable. All four name their single light source. **PASS.**

### CHAYKA (Slop Sentry — named materials only; would this hang in any other AI arcade?)
"An AI arcade of AI games is presumed slop until proven otherwise, and amber gold is the slop uniform."
- **CATCH (fixed, see reroll R3):** the first gold draft was warm-amber — "this could hang on any AI shelf on the internet."
- **CONFIRM:** the palette is now *named* — molten gold INK, brass-lemon, electric blue-white at pour temperature (metal physics, not decoration), over cracked egg tempera; the concept is kintsugi-as-play-biography (the crack records the life), which no generic shelf has authored. "The blue-white leak is the tell that a mind was here. **PASS.**"

**Bench verdict: 3/3 PASS after 3 rerolls.**

---

## C. REROLL LOG (recorded per instruction — the bench has teeth)

**R1 · NORMAN · dormant↔waking collapse at 128px → FIXED (not a full reroll; spec-tightened).**
Problem: thin-broken (state 1) and thin-beaded (state 2) both reduced to "faint thread" at thumbnail; the never-color-alone delta vanished. Fix applied to the ladder spec + V1 prompt: dormant's breaks widened to *visible dashes with wide dark gaps*; waking's beads enlarged and reduced to *three or four large, widely-spaced* beads (not many tiny ones). The 1→2 delta now survives desaturation and 128px. Re-benched: PASS.

**R2 · KARE · V3 bead reads as a gem/orb → REROLLED.**
Problem: an isolated molten bead centered on knockout black is a jewel, not a seam — primary-read inversion and a slop silhouette. Reroll: the bead is anchored to a *single wide channel that bleeds off the lower-left and upper-right edges*, declaring the seam continues beyond the crop. Now unambiguously a crack in a surface. Re-benched: PASS.

**R3 · CHAYKA · warm-amber gold = slop uniform → REROLLED (palette).**
Problem: warm amber "could hang in any other AI arcade." Reroll: gold re-specified to the ambitious-arm *brass-lemon* with an *electric blue-white leak at the molten core* (physically metal at pour temperature — Flair #9, teal-is-the-night). De-slopped by named-material physics, not decoration. Re-benched: PASS.

**Also weighed (no change required):**
- TUFTE (Compression Auditor) checked the tone caps: 4/3/4/3 tones, no fifth — and confirmed the beaded/molten "motion" ships as a static truth (reduce-motion honored). No reroll.
- The Ajar Door (#1), thumbprint (#4), cooled-bronze-remembers (#13), one-mischief keyhole-eye (#15), and second ember (#2) are all carried on V4 without crowding — TIFFANY confirmed the leads stayed few and thick (no filigree creep). No reroll.
