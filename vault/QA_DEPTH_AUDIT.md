# QA Depth Audit — Mandatory Gate
*Added: 2026-04-29*
*Lesson: QA previously passed audio pipeline (63/70) while Steps 03-05 were 7, 5, and 3 line stubs and pair files were 46 lines each.*

---

## The Rule

**QA must audit LINE COUNTS and CONTENT DEPTH, not just file presence.** A file existing with 3 lines of filler is a FAIL, not a PASS.

---

## Per-Step Minimums

| Step | File | Minimum Lines | Must Contain |
|------|------|--------------|--------------|
| 00 | 00_audio_aesthetics.md / 00_aesthetics.md | **100** | Aesthetic framework, sonic palette, emotional register, reference points |
| 01 | 01_audio_essence.md / 01_essence.md | **100** | Core creative DNA, unique differentiator, what this run is ABOUT |
| 02 | 02_song_concepts_12.md / 02_concepts_12.md | **150** | 12 distinct concepts, each with emotional core + genre direction + hook |
| 03 | 03_artist_critic.md | **150** | Artist + Critic perspectives on ALL 12 concepts. Scores (1-10). At least 4 disagreements. Revision directions. |
| 04 | 04_medium_facets.md | **100** | Per-pair: genre braid specifics, production palette, arrangement architecture, sonic references, distinctness proof |
| 05 | 05_pair_refinement.md | **120** | Per-pair: A-angle + B-angle, pair-specific constraints, lyric/instrumental direction |
| 06-10 | pair_XX_steps_06_10.md | **120** | Step 06 expansion, 07 variations, 08 full prompts, 09 critic/scores, 10 final selection |
| Pairs | pair_XX_songs_4.md | **80** | 4 complete song guides with full lyrics, production notes, Suno prompts, scoring |
| Final | final_24_song_guides.md | **250** | All 24 songs consolidated, no truncation |
| Top | top_6_song_guides.md | **80** | 6 ranked selections with full rationale |

---

## AUTO-FAIL Triggers

Any of these = **immediate FAIL, do not pass go:**

1. Any step file under 10 lines (stub/filler)
2. Step 03-05 under half the minimum line count
3. Pair files under 40 lines (means no full song guides)
4. Final assembly file missing or under 50 lines
5. Lyrics truncated to 1-2 lines when the pair source file has more (check source vs. assembly)
6. Same structural text repeated across all pairs without material variation (templating)

---

## QA Report Format

Every QA report must include:

```
## Depth Audit
| Step | File | Lines | Min Req | Status |
|------|------|-------|---------|--------|
| 00   | ...  | 128   | 100     | PASS   |
| 03   | ...  | 7     | 150     | ❌ FAIL |

## Content Audit
- [ ] 03_artist_critic has scores for all 12 concepts: YES/NO
- [ ] 04_medium_facets has per-pair analysis: YES/NO
- [ ] 05_pair_refinement has A/B angles per pair: YES/NO
- [ ] Pair files have full lyrics (not truncated): YES/NO
- [ ] Artist/Critic disagrees on ≥4 concepts: YES/NO

## Pair Depth Audit
| Pair | Lines | Songs | Full Lyrics | Status |
|------|-------|-------|-------------|--------|
| 01   | 46    | 4     | ⚠️ sparse  | ⚠️     |

## Verdict
PASS / FAIL / PASS WITH RERUN
```

---

## Integration

This file is referenced in the main session's task prompts. Every QA agent task MUST include:

```
## QA DEPTH AUDIT (mandatory)
Read /data/.openclaw/workspace/vault/QA_DEPTH_AUDIT.md
Apply ALL auto-fail triggers. Report line counts per step. 
DO NOT pass a pipeline with stub files.
```
