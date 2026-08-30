# SUNNA R4 Top Six — Soundcraft Repair Briefs

## Gate result

`scripts/measure_soundcraft.py` exposed a selected-set failure class that the structural Step validators do not measure. The binding floors in `vault/gates.yaml` are:

- strict end-rhyme return: **>= 0.30**
- exact line return: **>= 0.20**
- mean words per sung line: **<= 7.5**
- alliteration: **>= 11.0 per 100 words**

Repair targets should clear with margin: end rhyme **>= 0.32**, line return **>= 0.22**, alliteration **>= 12.0**. Do not pad metrics with meaningless chant; the return must belong to the song's event.

| Pair / selected | Current end rhyme | Current line return | Words/line | Current alliteration | Repair need |
|---|---:|---:|---:|---:|---|
| P01V01 `LOOK WITH YOUR EYES` | 0.250 | 0.461 | 4.26 | 9.57 | local end-sound braids + consonant play only |
| P02V03 `Leave It Crooked` | 0.230 | 0.338 | 4.73 | 8.00 | local end-sound braids + consonant play only |
| P03V01 `PRETTY ON PURPOSE` | 0.189 | 0.081 | 4.66 | 13.62 | end-sound braids + exact listener-owned returns |
| P04V04 `Again Before Credits` | 0.229 | 0.129 | 3.97 | 6.12 | end-sound braids + exact returns + consonant play |
| P05V04 `SCAN ME TWICE` | 0.178 | 0.151 | 4.29 | 11.18 | end-sound braids + exact returns |
| P06V04 `High Tide, High Gloss` | 0.111 | 0.111 | 5.17 | 11.02 | end-sound braids + exact returns |

## Shared repair contract

1. Modify only the selected variation in its Step 09, Step 10, and Step 11 source artifacts; all three non-selected variations stay byte-identical.
2. Preserve title, event, direct adult `you`, genre world, one seam, facts, prompt/exclude architecture, lineage, dedication, and the QA-accepted causal action.
3. Preserve existing exact choruses where the song has them. Add return by bringing back listener-owned lines, short answers, or transformed surrounding actions—not by replacing every section with the chorus.
4. Raise end rhyme through natural local families and mouth-feel. Do not force nursery couplets or generic `night/light/fire/desire` filler.
5. Raise alliteration through speakable consonant clusters tied to the room's objects and verbs; never insert production commentary into sung lines.
6. Keep the hidden message contraband. No policy, consent, boundary, ownership, finitude, value, or moral-summary vocabulary may be introduced merely to explain the engine.
7. Re-run Step 09, Step 10, Step 11, pair validation, `measure_soundcraft.py` on the selected package, and Step 09/10/11 portfolio distinctiveness. Re-measure dedication <=500 characters.
8. Record old/new selected-block hash, exact changed lyric lines, metric deltas, validator outputs, and byte-identity proof for the three untouched variations in `PxxVxx_SOUNDCRAFT_REPAIR_REPORT.md`.

## Pair-specific anchors to preserve

- **P01V01:** the newly accepted stripe-hold -> shoulder cue -> clipped-bell deadline -> inside-line crossing; both full choruses remain byte-identical.
- **P02V03:** damp claim ticket, washer glass, blue square removal, retained crescent mark, `fold it` -> `keep it`, missing-center brass return.
- **P03V01:** compact click, ticket stripe, enormous sequined shark, impractical-prize choice, stopped joke; increase return without making the lyric defend glamour in prose.
- **P04V04:** soda glass, red straw, concession gap, `AGAIN`, exactly two outbursts, compressed second return, short tail; avoid sung craft explanation.
- **P05V04:** tangerine wristband, wrong scan, road-case door, shared password, question/permission mirror; keep refusal funny and appetitive.
- **P06V04:** ferry rail, salt/ticket residue, two knocks, pocketed phone, gangway-first exit, exact title alignment; do not turn impermanence into a farewell lesson.

## Stop line

These are selected-block repairs only. No render, listening verdict, upload, publication, spend, commit, or push is authorized.
