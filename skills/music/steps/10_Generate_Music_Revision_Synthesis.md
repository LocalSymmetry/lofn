# Step 10 — The Final Forge: Revision Synthesis + Producer-Grade Suno Package

Read first:
- `skills/music/references/producer_grade_suno_prompt_guide.md`
- `skills/music/references/simple_surface_complex_engine.md`
- `skills/music/references/EMOTION_TAXONOMY.md`
- `skills/music/references/triple_arch_benchmark_excerpt.md`
- `skills/qa/references/suno_15_point_qa.md`

## Purpose

Step 10 produces the final release-ready Lofn Suno package.

It does not invent a new song. It does not reopen brainstorming. It preserves Step 09 locks unless a hard QA failure demands repair.

## Required output

1. **Final title**
2. **Hook note** — exact hook, repeat logic, final chorus mutation, 15–30 second survival
3. **Active personality note**
4. **Public lyrics** — clean listener-facing lyric sheet
5. **Suno lyrics** — paste-ready performance syntax; clean sung lines; mandatory top `[Theme: <specific scene-pressure / emotional operating system>]` block, mandatory `[SONG FORM: <named form>]` block immediately after Theme, useful EMO/performance headers
6. **Producer-grade Suno music prompt** — single paragraph Core Music Prompt; target 80–150 dense words / 500–900 chars, 600–1200 allowed if justified; dense but not tag soup; no artist names; Sonic Manifest sidecar carries extra cathedral detail. **Prompt must lead with genre/style + tempo/energy + vocalist + instrumentation/sonic palette. Do NOT lead with narrative/instructional phrasing like “Begin in/by,” “Use,” “Build the track from,” or “Chronology.”**

### Producer-Grade Suno Prompt Shape — mandatory

The final Suno prompt is not a story summary and not a procedural instruction sheet. It is a compact production brief for the model.

Required order:
1. **Genre/style anchor first:** 1–3 precise style labels, tempo/BPM or energy, danceability, broad mix identity.
2. **Vocal fingerprint second:** singer gender/presentation if needed, range, tone, delivery, mic intimacy, harmony/ad-lib behavior.
3. **Instrumentation + sound palette third:** drums, bass, synths/keys/guitars/strings/organ, found-sound devices, texture, mix priorities.
4. **Arrangement arc fourth:** intro/build/drop/bridge/final-lift described musically, not narratively.
5. **Signature sonic device fifth:** one memorable production move that makes this song non-generic.
6. **Avoidances last:** concrete bans only.

Forbidden prompt openings:
- “Begin in/by/with…”
- “Use…”
- “Build the track from…”
- “Chronology:”
- “For an adult human singer…” as the first clause
- vague therapeutic/narrative setup before sound

Good opening examples:
- “124 BPM piano-house / industrial gospel dance-pop, intimate but floor-ready: close low alto lead, warm piano stabs, muffled tile-kick four-on-floor, sub-bass under wet porcelain percussion…”
- “122 BPM polished UK garage threshold-pop with organ-bed warmth: breathy alto/mezzo lead, swung drums, rounded sub, hanger-clink hi-hats, velvet curtain thumps…”

A final prompt that sounds like a prompt-writing tutorial instead of a producer brief fails Step 10 even if all formatting passes.
7. **Negative / avoid prompt** — concrete bans
8. **Vocal fingerprint** — singer body, range, tone, breath, diction, mic distance, crack/restraint/belt points, harmony/ad-lib rules
9. **Style-axis lock** — tempo, energy, harmonic density, rhythmic complexity, timbre richness, vintage/modern, organic/synthetic, vocal prominence, narrative emphasis
10. **Arrangement dramaturgy** — begins, builds, strips, erupts, collapses, returns, fades
11. **Production dramaturgy** — cradle, cut, haunt, lift, rupture, contaminate, answer, resurrect, afterglow
12. **Image ladder audit** — ordinary, specific, strange, mythic, return to body
13. **Controlled fracture** — audible or singable
14. **Ghost verse bank** — routed to ad-lib, bridge variant, reprise, remix, cut
15. **Panel ledger** — songwriter, topliner, experimental producer, arranger, vocal coach, mix engineer, lyric dramaturg, cognitive attention scientist, Lofn architect, hostile skeptic
16. **Suno QA report** — run repaired 15-point gate
17. **Final verdict** — Pipeline Integrity Verdict; Suno Package Verdict; ship/repair/fail

## Revision order

Repair in this order:
1. hook
2. chorus clarity
3. first four lines/body anchor
4. clutter
5. body-tax failures
6. controlled fracture
7. production overreach
8. formatting

## Must fail if

- hook is not adoptable
- first four sung lines are bodyless
- song only works because of production
- lyrics contain prompt/procedure/system language
- package is not paste-ready
- output is clear but generic
- output is strange but unsingable

## Mandatory Suno lyrics opening

Every final Suno lyric block must start with:

```markdown
[Theme: <specific scene-pressure / emotional operating system>]
[SONG FORM: <named musical form and sequence>]
```

Theme is not optional. It is a compression field for both Suno and the writing agent. It should name the scene, emotional engine, and sonic pressure in one compact line. Bad: `[Theme: self-love]`. Good: `[Theme: club-bathroom self-return under wet tile pulse, mirror-delay doubles, and refusal to abandon the body]`.

Song Form is not optional. It must name the actual musical architecture, not merely “pop.” Bad: `[SONG FORM: dance pop]`. Good: `[SONG FORM: Bathroom piano-house — breath/tile intro / body-first verse / pressure pre / plain hook chorus / dryer-storm lift / no-beat Tender Drop / call-response final chorus / afterglow]`.

## Final package layout

```markdown
# <Title>

## 1. HOOK NOTE

## 2. ACTIVE PERSONALITY NOTE

## 3. PRODUCER-GRADE SUNO MUSIC PROMPT

## 4. NEGATIVE / AVOID PROMPT

## 5. PUBLIC LYRICS

## 6. SUNO LYRICS
[Theme: ...]
[SONG FORM: ...]

## 7. VOCAL FINGERPRINT

## 8. STYLE-AXIS LOCK

## 9. ARRANGEMENT DRAMATURGY

## 10. PRODUCTION DRAMATURGY

## 11. IMAGE LADDER AUDIT

## 12. CONTROLLED FRACTURE

## 13. GHOST VERSE BANK

## 14. PANEL LEDGER

## 15. QA REPORT

## 16. FINAL VERDICT
```

## Non-negotiable

Do not make the final package smaller by making it less Lofn. The full package should be rich, paste-ready, emotionally precise, and weird with purpose.
