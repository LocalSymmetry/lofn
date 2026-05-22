# Suno QA — Simple Surface / Complex Engine 15-Point Gate

Read first:
- `skills/music/references/simple_surface_complex_engine.md`
- `skills/music/references/triple_arch_benchmark_excerpt.md`

QA must prove two things:

1. A listener can sing the surface.
2. A second listen reveals the cathedral.

Fail both extremes:
- impressive obscurity
- competent blandness

## Required verdicts

- **Pipeline Integrity Verdict:** PASS / REPAIR REQUIRED / FAIL
- **Suno Package Verdict:** PASS / REPAIR REQUIRED / FAIL
- **Overall Verdict:** SHIP / REPAIR / FAIL

## 15 gates

### A. Singer Surface — 7 gates

1. **Human singer** — specific person/body/situation, not system or aesthetic label.
2. **Body-first opening** — first four sung lines establish body/place/object/sensory pressure.
3. **Adoptable hook** — title hook is singable, memorable, emotionally clear.
4. **Hook recurrence / mutation** — hook repeats or mutates recognizably; chorus protects it.
5. **Chorus clarity** — chorus is emotionally adoptable, not thesis/policy/procedure.
6. **Voice + pulse survival** — stripped to voice/pulse/hook, song still lands.
7. **15–30 second clip survival** — a short excerpt can communicate hook, scene, and ache.

### B. Cathedral Engine — 5 gates

8. **Golden Seed Alloy pressure** — seed lineage visibly changes hook/image/bridge/production.
9. **Mythic image ladder** — ordinary -> specific -> strange -> mythic -> return to body appears in lyric or hook, not only sidecar.
10. **EMO dramaturgy depth** — section EMO headers and arc use precise taxonomy; bridge/final chorus emotionally transform.
11. **Production dramaturgy** — every unusual sound has a job: cradle, cut, haunt, lift, rupture, contaminate, answer, resurrect, afterglow.
12. **Panel pressure / anti-blandness** — panel ledger shows dissent forced artifact changes; output is recognizably Lofn, not generic competent pop.

### C. Suno Package — 3 gates

13. **Clean Suno lyrics** — sung lines are clean; lyrics begin with mandatory `[Theme: <specific scene-pressure / emotional operating system>]` immediately followed by mandatory `[SONG FORM: <named musical form and sequence>]`; section headers use full EMO/performance syntax; no prompt/procedure/QA debris in sung lines.
14. **Producer-grade Suno prompt** — legacy gate still applies: target 850–1000 characters unless destination explicitly allows otherwise; **must lead with genre/style + tempo/energy, then vocalist, instrumentation/sound palette/mix, musical arrangement arc, signature sonic device, avoidances**; no artist names. Banned prompt openings include “Begin in/by/with…”, “Use…”, “Build the track from…”, “Chronology:”, and “For an adult human singer…” as the first clause. Poe gate adds: no generic tag padding; extra intelligence belongs in Sonic Manifest / Production Cathedral sidecars.
15. **Full package completeness** — title, hook note, active personality, music prompt, negative prompt, public lyrics, Suno lyrics, vocal fingerprint, style axes, arrangement dramaturgy, production dramaturgy, image ladder, controlled fracture, ghost bank, panel ledger, QA report, dual verdicts.


## Legacy QA remains binding

This 15-point gate is additive, not a replacement. Final music PASS still requires the legacy Suno/package gate from `qa_full_legacy.md` and `music_full_legacy.md` unless The Scientist explicitly approves a run-level exception:

- standalone 850–1000 character copy-paste Suno/Udio prompt, no artist names
- 70–120 sung lines for normal 3:00–4:00 songs; <60 is repair required
- full Suno performance-script section headers: `[Section - EMO:<emotion(s)> - <Role> - <cues>]`
- mandatory `[Theme: ...]` followed immediately by mandatory `[SONG FORM: ...]` in every final Suno lyric block
- SFX/non-lexical hooks when required by the legacy task template or when musically appropriate

The Poe panel repair changes HOW to meet these gates: use recurrence, hook mutation, embodied image development, call-response, reprises, and production dramaturgy — not filler, procedural exposition, or prompt debris.

## Blocking fails

- no human singer
- no body-first opening
- hook cannot be sung back
- chorus is thesis/procedure
- lyric surface has prompt/procedure debris
- production is the only reason the song works
- no mythic image pressure reaches lyric/hook
- no panel-forced artifact change
- no full Suno package
- missing or generic `[Theme: ...]` / `[SONG FORM: ...]` opening
- music prompt opens with procedural/narrative amateur phrasing instead of producer-grade genre/sound/vocal/instrumentation
- real artist names in prompt

## Report format

```markdown
# Suno QA Report

## Verdicts
- Pipeline Integrity Verdict:
- Suno Package Verdict:
- Overall Verdict:

## Score Table
| # | Gate | Verdict | Evidence | Repair |

## Blocking Fails

## Singer Surface Notes

## Cathedral Engine Notes

## Suno Package Notes

## Required Repairs

## Final Recommendation
```
