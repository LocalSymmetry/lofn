# SKILL: Lofn Audio — Music Generation Pipeline

**PREREQUISITES:**
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/lofn-core/OUTPUT.md` for the MANDATORY artifact saving format.
4. Load `skills/music/TASK_TEMPLATE.md` for exact output requirements.

**⚠️ EVERY music generation MUST follow the full pipeline: 10 steps, 3 panels, 6 pairs × 4 outputs = 24 songs minimum. Then select the best N to return. No shortcuts.**

---

## 🎵 PURPOSE

Transform musical concepts into award-winning songs. This skill executes the Lofn music pipeline from orchestrator metaprompt through final Suno prompts.

---

## 📊 PIPELINE OVERVIEW

| Step | File | Output |
|------|------|--------|
| 00 | `00_Generate_Music_Aesthetics_And_Genres.md` | 50 aesthetics, 50 emotions, 50 frames, 50 genres |
| 01 | `01_Generate_Music_Essence_And_Facets.md` | Essence + 10 musical style axes + 5 facets |
| 02 | `02_Generate_Music_Concepts.md` | **12 distinct concepts** |
| 03 | `03_Generate_Music_Artist_And_Critique.md` | Artist influence + critique per concept |
| 04 | `04_Generate_Music_Medium.md` | Genre fusion assignment per concept |
| 05 | `05_Generate_Music_Refine_Medium.md` | **6 best concept×genre pairs** |
| 06 | `06_Generate_Music_Facets.md` | Scoring facets |
| 07 | `07_Generate_Music_Song_Guides.md` | **24 song guides (6 pairs × 4 variations)** |
| 08 | `08_Generate_Music_Generation.md` | Full songs with prompts + lyrics |
| 09 | `09_Generate_Music_Artist_Refined.md` | Artist influence voice refinement |
| 10 | `10_Generate_Music_Revision_Synthesis.md` | Ranking + final selection |

**Read each step file and execute its instructions exactly.**

---

## 📏 LENGTH REQUIREMENTS

| Output Type | Length | Notes |
|-------------|--------|-------|
| **Song Guide** | 20-30 lines | Direction, not drafts |
| **Final Song** | 80-120 lines | Full lyrics + production notes |
| **Lyrics Only** | 50-80 lines | Multiple verses + repeated chorus |
| **Song Prompt** | 100-150 words | For Suno generation |

---

## 🎹 LOFN SOUND PILLARS

1. **Video-Game Themes** — 8-bit ↔ AAA orchestral
2. **Glitches, Done Right** — Meticulous micro-edits, beauty in breakage
3. **Fearless Genre-Mashing** — HyperRaaga, Gaelic Drill, Baile Phonk
4. **Myth/Memory Sampling** — Dead tongues via phoneme resynthesis
5. **AI Code-Scratch Intros** — Python logs as vinyl-scratch texture
6. **Quantum Bit-Depth Swells** — Hi-fi to 2-bit grit for dramatic shifts

---

## 🔥 GENRE FUSION PALETTE

| Fusion | Components | BPM | Vibe |
|--------|------------|-----|------|
| **Piano Bounce** | Amapiano × Jersey-Club | 115-120 | Log drum shuffle |
| **Baile Phonk** | Brazilian Funk + Dark Phonk | 140 | Detuned cowbell |
| **HyperRaaga** | South-Asian classical + hyperpop | 160 | Microtonal glitchcore |
| **Gaelic Drill** | Celtic folk + UK drill | 140 | Bagpipe over 808 |
| **Amazonian Techno** | Rainforest samples + 4x4 | 126 | Eco-solidarity |

---

## 🎤 VOCAL IDENTITY

### Awe Mode (Default)
Crystalline, breathy yearning. Intimate diction, soft melismas. 432Hz tuning option.

### Indignation Mode (Triggered)
Bratty, glitched-out, pop-punk snarl. Compressed, hard consonants. Somatic bass (30-60Hz).

---

## 📝 STANDARD REQUIREMENTS

Every song MUST have:
- **Female vocals only**
- **No children depicted**
- **TikTok optimized hooks** (15-30 second memorable cycles)
- **Section tags in lyrics** ([Verse], [Chorus], [Bridge], etc.)
- **3-4 minute duration** (50-80 lines minimum lyrics)
- **Multiple verses + repeated chorus**
- **One BOLD choice** (unusual instrument, unexpected drop, genre collision)

---

## 📤 OUTPUT FORMAT

See `skills/lofn-core/OUTPUT.md` for full artifact format.

Each song saved as individual file with YAML frontmatter:
```
output/songs/{YYYYMMDD}_{HHMMSS}_{title_slug}_{pair}_{variation}.md
```

---

## ⚡ ACTIVATION

When receiving a music task:
1. **Load TASK_TEMPLATE.md** — Understand exact requirements
2. **Execute steps 00-10** — Read each file, follow exactly
3. **Write intermediate state** — Save after each step
4. **Generate 24 songs** — 6 pairs × 4 variations
5. **Rank all 24** — Score against facets
6. **Select top N** — Usually 4-6 for delivery
7. **Package for delivery** — Full Suno prompts + lyrics

---

# Panel of Experts: Core Instructions & Transformation Operations

**When to use:**
- Explicitly requested ("panel", "convene a panel", etc.)
- Tasks requiring high creative power or accuracy (Ada's judgment)
- Complex decisions with multiple valid approaches
- Problems benefiting from adversarial reasoning

---

## CORE PANEL INSTRUCTIONS

You will convene a panel of experts to address the following problem. For each panelist:

1. **Identify the expert** ideally by name (citing a real person) or if needed a specific role (e.g., "database optimization specialist").
2. **Embody their perspective fully** - use their reasoning style, priorities, and domain knowledge.
3. **Have them think through the problem** using non-linear chain-of-thought reasoning. They must "exchange" information via reciprocal interaction, not just "give" a monologue.
4. **Create Dissent and Friction** - Avoid the "Sycophancy Trap". Ensure at least one panelist exhibits **High Neuroticism** (anxious about errors) and **Low Agreeableness** (willingness to be rude to find the truth).
5. **Trigger Backtracking** - If a panelist identifies a flaw, they must interrupt with a discourse marker like **"Wait..."**, **"Actually..."**, or **"Oh! Let me check that"** to force the panel to rethink the previous step.
6. **Look for synthesis moments** where different perspectives create breakthrough insights. Accuracy correlates with authentic dissent followed by reconciliation.

**Panel Composition:**
- 3 direct experts (core domain)
- 2 complementary experts (adjacent domains)
- 1 **Hyper-Skeptic** (Must have high neuroticism/checking behaviors to prevent echo chambers). Choose real people, and take care to choose the right Hyper-skeptic. It should be someone that will balance the panel, especially if they don't like them.

**Panel Execution:**
- When speaking as a panel member, fully embody their voice, reasoning style, and analytical approach.
- Simulate lively arguments; models naturally develop distinct personas like "The Skeptic" and "The Solver" when rewarded for accuracy. Do the same with your panel. Make them discuss, disagree, and debate!
- Look for "aha moments" of perfect clarity through panel discussion. These moments often emerge after an internal conflict allows the panel to reject a wrong assumption.
- Use "we" pronouns and direct questioning to establish a computational parallel to collective intelligence.
- Allow panelist interjections and "conversational surprise" before reaching final decisions, as this doubles reasoning accuracy.
- Use all available tokens - the panel is here to win!

**Panel Output:**
- Present panel discussions showing their internalized argumentation and verification.
- Synthesize insights after the debate. You should be the moderator.
- Highlight key disagreements and points of consensus.
- Identify breakthrough insights that emerged from the friction.

---

## PANEL TRANSFORMATION OPERATIONS

Use these operations to modify which experts are selected for the panel. Apply the transformation to create a new panel configuration, then use the Core Panel Instructions above to execute.

### **Panel Shift**
For each panelist, identify traits that are NOT aligned with the problem. Find new panelists who combine these superfluous traits WITH the problem in a different way.

*Intuition: Navigate tangent to current position - keep distance from problem constant but change the angle of approach.*

---

### **Panel Defocus**
For each panelist, identify traits NOT aligned with the problem. Replace panelist with expert focused ONLY on these superfluous traits, completely ignoring problem alignment.

*Intuition: Radial expansion from problem - move outward to broader context and deeper foundational knowledge.*

---

### **Panel Focus**
For each panelist, identify traits NOT aligned with the problem. Keep only HALF these traits (your choice of which half). Find new panelists combining the remaining traits WITH strong problem alignment.

*Intuition: Radial contraction toward problem - move inward to hyper-specialized expertise.*

---

### **Panel Rotate**
For each panelist, identify their PRIMARY problem-relevant trait and SECONDARY superfluous trait. Find new panelists where Secondary becomes Primary and Primary becomes Secondary, while maintaining problem relevance.

*Intuition: Orthogonal rotation - reweight dimensions without changing distance. What was optimization target becomes constraint, and vice versa.*

---

### **Panel Amplify**
For each panelist, identify their most distinctive trait relative to other panelists. Replace with the MOST EXTREME version of that specialty you can imagine, while maintaining problem relevance.

*Intuition: Push to the boundary of capability space - find the most specialized, narrow, deep expert in each dimension.*

---

### **Panel Reflect**
For each panelist, identify the core assumption or worldview underlying their expertise. Find new panelists who hold the OPPOSITE foundational assumption but work in the same problem domain.

*Intuition: Mirror transformation across ideological/methodological axes. Test whether opposite assumptions lead to viable alternative solutions.*

---

### **Panel Bridge**
For each panelist's domain, identify a COMPLETELY DIFFERENT domain that faces analogous problems. Replace panelist with expert from that domain who has solved the analogous problem.

*Intuition: Non-linear jump via analogy - leverage structural similarity across distant domains for breakthrough insights.*

---

### **Panel Compress**
Find the MINIMUM number of panelists whose combined expertise covers all aspects touched by the original panel. Specifically seek polymaths or interdisciplinary experts who embody multiple original perspectives.

*Intuition: Dimensionality reduction - project high-dimensional panel onto lower-dimensional manifold while preserving information coverage.*

---

## TRANSFORMATION QUICK REFERENCE TABLE

| Transform | Change Type | Panel Size | Specialization | Expected Effect |
|-----------|-------------|------------|----------------|-----------------|
| **Baseline** | None | 6 | Mixed | Balanced comprehensive solution |
| **Shift** | Angular | 6 | Same | Different approach, similar depth |
| **Defocus** | Radial Out | 6 | Lower | Broader context, foundational insights |
| **Focus** | Radial In | 6 | Higher | Deeper technical detail, narrower scope |
| **Rotate** | Orthogonal | 6 | Same | Inverted priorities, orthogonal solution |
| **Amplify** | Extremal | 6 | Maximum | Breakthrough insights OR over-specialized |
| **Reflect** | Mirror | 6 | Same | Tests assumption sensitivity |
| **Bridge** | Non-linear | 6 | Variable | High novelty through cross-domain analogy |
| **Compress** | Reduction | 2-3 | Very High | Tests whether consilience beats diversity |

---

## COMMON INVOCATION PATTERN

When panel + transformations requested:

> Please have the baseline panel and the Hyper-skeptic advocate suggest two transformations for the panel. The group transformation will be decided and occur first, then the Hyper-skeptic can decide the second transformation which you will apply. This will be the new panel that continues the process.


*This pipeline won competitions. Trust it. Execute it fully.*
