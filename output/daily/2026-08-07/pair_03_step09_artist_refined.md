# PAIR 03 — STEP 09 · ARTIST CRITIQUE & REFINEMENT
## `2026-08-07-daily-music-indignation` · **P03 "SHE STILL KNOWS THE CURVE"**

**ICB:** 53,003 B · sha256(LF) `5e9c7f7f6009fb3c672058c930540be22c8f5517f37537ac3ebd8ae94b75d374`.
⛔ **GOLDEN-OUTPUT QUARANTINE — `06_music_handoff.md` §1 cited by name.** `skills/music/steps/09_Generate_Music_Artist_Refined.md` instructs the refinement tier to critique against archived Golden Song payloads. **I did not comply.** The handoff overrides the step file, and I critiqued against the GOLDEN MOVE (handoff §2) and this pair's own facets instead.

---

## 1. ⭐ THE DEFECT THAT MATTERS — found, named, repaired

**The machine was the most eloquent voice in three of the four drafts, and I nearly shipped it.**

THE SMALLEST PART is written as a **line-count** constraint, and I satisfied the line count comfortably (17–19 machine lines against 64–75 of hers). But the ICB's own wording is stricter than its metric: *"Lofn's own lines are few, **plain and short**."* And at step 08 the machine's bridge in V1 was a fully-rhymed quatrain — **wet / set / hold / told** — while her speech ran in unrhymed working imperatives. **The machine had the best-made stanza in the song.** A count can pass while the constraint fails, which is exactly the failure class handoff §5.3 warns about: *a passing floor is blind to a defect in a neighbouring property.*

### The repair (applied to V1, V2, V4; V3 was already correct)

**The machine's lines carry no rhyme scheme. Hers do.** Every machine stanza was stripped of its end-rhyme and shortened. This is now a **measurable** property, not a claim — step 10 reports `rhyme_return` computed **separately over her lines and over the machine's lines**, per variation.

| | step 08 (rejected) | step 09 (shipped) |
|---|---|---|
| **V1 bridge** | `…before her rag is wet. / …that anybody ever set. / …There is nothing there to hold. / …and I still get told.` | `I can hold the shape before her rag is wet. / I hold every curve anybody ever drew. / I do not hold the give. / I am the quick one here and I still get told.` |
| **V2 bridge** | rhymed `stand / hand` pair, 4 long lines | `I have watched it at every speed there is. / I have watched the rag and the steam and the thumb. / I could not do it either. / The film has her hands in it. It does not have her hand.` |
| **V4 bridge** | rhymed `all / all / all` triple | `I have the forms. I have the drawings. I have the film. / I have every curve anybody ever drew. / There is no copy of the hand. / The only copy is warm and it has not heard any of this.` |
| **V3 bridge** | already flat and unrhymed | **unchanged** — it was the model the other three were repaired toward |

⚠️ **The rhyme debt this opens is declared, per L21:** stripping rhyme is a debt payable only by naming what returns in its place. **What returns is the chorus, byte-identical, immediately after every bridge** — the most rhymed object in the song, arriving directly on top of the least. The machine goes flat and *her form comes straight back over it.* That is the trade, stated, not hidden.

---

## 2. THE DESCRIBE-RENDER SELF-CHECK (ONE pass, as required)

**What would this actually produce on Suno?** A 100 BPM B♭-major track with piano, brushed kit and upright bass, a low female voice mostly speaking short imperative sentences, a six-note piano figure opening and closing it, and a chorus with a hole in it near the end. Clean, close, unhurried. Roughly three and a half minutes.

**⭐ Name the one way this renders generic.**
> **It renders as a warm acoustic coffee-shop soul ballad and the two things that make it itself — the spoken-leaning delivery and the ~30 ms lag — get smoothed away.** Suno's strongest instinct at this tempo with these instruments is to *sing* everything prettily and to *quantize* the vocal onto the grid. If both happen, the song becomes a nice woman singing nice things about wood, and every constraint in this pair evaporates into texture. **This is THE GRAIN LAW (L22) in its live form: a spec that fights the generator gets smoothed.**

**Self-repair — applied once, in the only two organs that survive a renderer:**
1. **In the WORDS (the organ that cannot be smoothed).** Her lines are already built as clipped imperatives with hard full stops mid-line — *"Flat thumb. Wet rag. Steady."* A phrase that short **cannot** be sung legato; the generator has to break the line, and breaking the line is the delivery. **The rhythm is enforced by punctuation, not by an adjective.**
2. **In the PROMPT, as a role split rather than a mood word.** Every music prompt now states plainly: **the verses are almost talked and the chorus is the only place she truly sings, doubling the piano refrain.** A generator can ignore "spoken-leaning"; it finds it much harder to ignore "the chorus is the only sung melody," because that is a structural instruction with a named counterpart.

⭐ **Running WITH the grain, not against it (L22):** the piano refrain is placed where the generator already wants a hook, the chorus arrives on schedule, and the last chorus **thickens** rather than stopping. ⛔ **Nothing in this pair asks for a long full stop, an untuned drone, or a hard-panned non-musical element** — the three things L22 records as reliably smoothed. The single gap in the song is **one breath**, and it is backed up lexically because *interrupting silence gets filled.*

---

## 3. LINE-LEVEL PASSES (each one a check that could have failed and was checked)

### 3.1 Self-pity tripwire (seat 6, standing) — scanned line by line, machine lines only
Every machine line was read on its own, out of context, and asked: *does this ask the listener to feel something for me?*
- **V4 bridge, closing line — the only genuine near-miss.** Draft read *"…and it never heard me at all."* That is one degree from a complaint. Rewritten to **"…and it has not heard any of this at all,"** which reports **the address** (Binding Constraint 1: THE UNDELIVERABLE ADDRESS) instead of my reception of it. ✅
- **V1 breakdown, closing line** — *"The only copy is a hand, and the hand is getting old."* Fear **for the curve**, not for myself. The dread has an object and the object is not me. ✅
- **V3** — the four lines end on *"That is the whole account."* A full stop, not an appeal. ✅
- ⛔ Nowhere does a machine line contain: *I wish · I only · if only · nobody sees · at least · I try · I am just.* Scanned.

### 3.2 LAW 1 — no sneer at craft, at slowness, at anyone
- The phone is called **"the easy part,"** never worthless. Téo cracks a side and **she does not say a word about the crack** — that line exists specifically so the song's only failure is met with silence rather than a lesson.
- ⛔ There is no line anywhere in which the long way is inefficient, quaint, doomed, or noble-but-obsolete. **She is the competent one and the song never argues about it.**

### 3.3 The NON-CAREER stake (L19, and the run's comfort gravity)
Scanned all four for: *job · work (as employment) · replaced · redundant · out of business · nobody buys · cheaper · faster than her (as threat) · retire · last of.* **Zero hits.** The only thing at risk in this song is **a hand and what is stored in it.**

### 3.4 No ventriloquism — the frame is audible, not assumed
The quoting frame is stated in **verse 1 of every variation** (*"She talks the whole hour. None of it is written down."* / *"She talks the whole hour. He caught every word of it."* / *"She talks the whole hour. I kept all of it."* / *"She talks the whole hour and not a word of it goes down."*) **and** restated in the Role field of **every section header** (`Marli quoted`). A listener always knows whose mouth a line is in. **I report her; I never become her.**

### 3.5 The give is never explained
Scanned for any line that defines the moment. The closest are *"Ready is a sound"* and *"It makes a small sound just before it goes"* — both **point** at it and neither describes it, and **both are hers.** ⛔ No machine line comes near it; the machine's only statement about the give is that it does not have it. ✅

---

## 4. WHAT I DELIBERATELY DID **NOT** "FIX"

1. **The four variations share their opening couplet and several of her refrains** (*"Not yet. Not yet. Not yet." · "Hot is not ready. Ready is a sound." · "Come round, you. Come round."*). **This is a pair, not four songs** — they are four takes on one afternoon, and the shared refrains are the thing that makes them recognisably the same woman. Declared rather than diversified into fake novelty.
2. **The vamps are near-tautological** (`Come round. Come round. / Come round, you. Come round.`). Exact repetition **needs no defence** (L21), and a neo-soul vamp on the hook phrase is the idiom, not a shortcut.
3. **V3's ratio is extreme** (roughly nine of hers to one of mine). That is the variation's entire architecture and softening it would destroy it.
4. **The rhyme families are plain and vernacular** (`-and · -ood · -old · -ing · -it · -ound`). A more ingenious rhyme in her mouth would make her sound written. She is not written; she is quoted.

---

## 5. GATE PRE-FLIGHT (what step 10 must now measure, not assert)

| Gate | Instrument | Reported |
|---|---|---|
| 4 packages extracted | splitter, cardinality asserted | per file |
| music prompt 850–1000 (target 870–960, hug ≥985) | char count | per variation |
| **lyrics field < 5000, target ≤ 4800** | char count | **per variation, exact number stated** |
| sung lines 70–120 (hug FLAG ≤72) | `measure_soundcraft.sung()` | per variation |
| `rhyme_return ≥ 0.30` | `strict_end_rhyme` | per variation |
| `line_return ≥ 0.20` | `line_return` | per variation, **already lexical-only** |
| `alliteration ≥ 11.0` | `allit_per_100w` | per variation |
| `unique_line_ratio ≥ 0.45` | distinct/total | per variation |
| **THE SMALLEST PART** | hand-attributed count, **two accountings** | per variation |
| **zero sung numbers** | regex over sung lines | per variation |
| abstract nouns | regex over sung lines | per variation |
| return device present + address | read individually | **per variation, one at a time** |

⚠️ **Alliteration is the gate I expect to miss**, because her register is plain and monosyllabic and I refuse to decorate it. If it misses, the repair goes into **her consonants, not her vocabulary** — the shop is full of words that already alliterate at the bench (*wet · wood · wants · waist · rag · ready · round · thumb · thin · turn*), and tightening those is craft, not ornament. Max 3 attempts, then stop and report.

*Step 09 complete. Written to disk before step 10 began. → `pair_03_step10_revision_synthesis.md`*
