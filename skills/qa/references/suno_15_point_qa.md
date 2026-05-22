# Suno 13-Point QA Gate
*Established: 2026-05-09, Condensed: 2026-05-17*
*Source lineage: Triple Arch / Suno staff-pick follow-up analysis, repaired final QA, eligibility framework*

## Purpose

The 7 eligibility properties are necessary but not sufficient. They predict whether a song can be broadly felt. They do not fully verify whether a final Suno package is paste-ready, hook-survivable, Lofn-specific, and protected from prompt slop.

Every music QA pass must run this 15-point gate on each final selected song.

## Scoring

Each point is PASS / PARTIAL / FAIL.

For an ACCESSIBLE / release-targeted song:
- **PASS:** 11-13 pass, no blocking fails
- **REPAIR:** 8-10 pass or any non-blocking fail
- **FAIL:** <8 pass or any blocking fail

For an AMBITIOUS song:
- Eligibility points may intentionally fail, but delivery/package points remain mandatory.
- QA must classify failures as intentional ambition vs accidental inaccessibility.

## The 13 Points

### Eligibility Core — 7 points

1. **Body in the song** - first 30 seconds create felt physical location: temperature, texture, room, weather, surface, breath, object, or body.
2. **Adoptable hook** - hook is prayer/invocation/vow/address, not thesis, accusation, defense, or explainer.
3. **Vast emotional TAM** - emotional field is broadly humanly felt: awe, grief, longing, protection, wonder, tenderness, fear of loss, becoming.
4. **Specificity paradox** - one precise fact/detail/number/place/date/material unlocks the universal claim.
5. **Cognitive ease** - structure is legible on first listen; accessible songs favor verse/chorus, major/Mixolydian or emotionally legible harmony, roughly 95-120 BPM unless justified.
6. **Vocal co-discovery** - singer discovers the truth while singing; not a report, manifesto, or already-processed lecture.
7. **Sonic threshold** - opening gives the nervous system a doorway: room tone, environmental sound, sparse pulse, breath, silence-with-noise-floor, or other calm before demand.

### Suno / Lofn Delivery Core — 6 points

8. **Standalone Suno style prompt present** - `## 1. MUSIC PROMPT` or `SUNO STYLE PROMPT` exists, is copy-paste-ready, and is not replaced by scattered metadata.
9. **Prompt density and restraint** - prompt is producer-grade and bounded: target 850-1000 chars (or justified sparse exception), no artist names, no bloated tag soup, no contradictory instructions. It must lead with genre/style + tempo/energy, then vocalist, instrumentation/sound palette/mix, musical arrangement arc, signature sonic device, and short concrete avoidances. Banned openings include “Begin in/by/with…”, “Use…”, “Build the track from…”, “Chronology:”, and “For an adult human singer…” as the first clause.
10. **Performance-ready lyric syntax** - lyrics use Suno meta-tags: mandatory `[Theme: <specific scene-pressure / emotional operating system>]` as the first lyric line immediately followed by mandatory `[SONG FORM: <named musical form and sequence>]`, then EMO section headers (format: `[Section - EMO:<emotion(s)> - Vocalist - cues]` where the emotion is drawn from the full emotional taxonomy at `skills/lofn-core/refs/EMOTION_TAXONOMY.md` - e.g. `nostalgia + yearning`, `righteous fury`, `tender grief`). Theme is not a generic topic; Song Form is not merely “pop” or “dance.” The bare Lofn architectural states (`AWE:`, `INDIGNATION:`, `SYNTHESIS:`) are NOT valid emotion labels for section headers. Vocalist/mix cues, SFX cues, and at least one non-lexical hook where appropriate. **Call-and-response lyrics must use Suno-native `Call (Response)` format on a single line, never separate `Lead:` / `Choir:` / `Doubles:` lines.**
11. **15-30 second hook survivability** - one section can stand alone as a short clip and still carry the song's emotional thesis without sounding like a slogan.
12. **Active personality fidelity** - song sounds like the personality selected for this run. QA must name the active personality/persona from the orchestrator or seed and verify the output matches that personality's voice, values, and sonic palette. Personality mismatch is the fail condition.
13. **Package readiness** - final artifact contains title, music prompt, negative/avoid prompt, full lyrics, production notes, hook note, and any special events (blackout drop, room tone, artifact cue, tempo shift). The Scientist can paste it without reconstructing missing pieces.

## Blocking Fails

Any of these blocks PASS until repaired:

- Missing standalone Suno style/music prompt.
- Missing full lyrics.
- Bare `[Verse]` / `[Chorus]` tags only in final lyrics.
- Missing or generic `[Theme: ...]` / `[SONG FORM: ...]` opening, or Song Form not immediately after Theme.
- Music prompt opens with procedural/narrative amateur phrasing instead of producer-grade genre/sound/vocal/instrumentation.
- Real artist names in final Suno prompt.
- No hook or no short-clip extractable section for accessible/release-targeted songs.
- Song is an abstract lecture with no body anchor unless explicitly routed AMBITIOUS and labeled as such.
- Sung lyric count <60 lines unless the song is routed AMBITIOUS with an explicit runtime/brevity justification. Target 70-120 for 3:00-4:00 runtime.
- Section headers using bare Lofn architectural states (`AWE:`, `INDIGNATION:`, `SYNTHESIS:`) as emotion labels instead of drawing from the full emotional taxonomy with `EMO:<emotion(s)>` syntax.
- QA cannot identify the active run personality or the song's voice/sonic decisions do not match that personality.

## Required QA Report Section

```markdown
## Suno 13-Point QA Gate
| # | Check | PASS/PARTIAL/FAIL | Evidence | Repair if needed |
|---|-------|-------------------|----------|------------------|
| 1 | Body in the song | | | |
...
| 13 | Package readiness | | | |

Verdict: PASS / REPAIR / FAIL
Blocking failures: yes/no
Accessible classification: ACCESSIBLE / AMBITIOUS
```

## Relationship to the 7-Property Eligibility Framework

The first 7 points are the eligibility framework. Points 8–12 are the operational Suno/package/Lofn survival checks. Point 13 is package readiness.

Do not collapse the 15-point gate back into the 7-point framework. The distinction matters.
