# QA Repair — P04 V3, P05 V1, P06 V4

Date: 2026-08-26  
Scope: bounded Step-11 Gate-13 repair of the three selected variations only. Golden/reference outputs, blind packets/maps, unrelated packages, P01–P03, and standalone selected files were not opened or edited.

## P04 V3 — Answer the Held Note

Frozen-field correction:

- `Future-garage organ club, 127 BPM, ...` → `Future-garage organ club, 126 BPM, ...`

Exact sung-line substitutions:

| Before | After |
|---|---|
| `The kick arrives like scissors` | `My heel lands sharp as scissors` |
| `The low cell does not blink` | `The brass rail does not blink` |
| `The organ threads the cuts` | `My fingers thread the slips` |
| `This velvet atrium preset is the wrong bright wrapper` | `This velvet atrium wraps my voice in borrowed brightness` |
| `My signal is an offer` | `My held breath is an offer` |
| `The upper air turns clear` | `My shoulders leave the wall` |
| `The low cell keeps its path` | `My heel stays on the tile` |
| `The organ stays unbroken` | `My open mouth stays steady` |
| `The kick still cuts the tile` | `I cross one silver tile` |

Major Deviations now records the 127→126 BPM QA correction and the bounded Gate-13 cleanup instead of claiming no deviation. The sidechain consequence, organ/low-cell invariants, announcement-vowel return, prompt, Disc channels, and dedication remain where production mechanism belongs.

## P05 V1 — My Last Bar Is Mine

Exact sung-line substitutions:

| Before | After |
|---|---|
| `The low line walks through` (Moving-Bass Blackout) | `My left hand reaches through` |
| `My pulse keeps its route` | `My breath stays warm and true` |
| `That little final t` | `My fingertip taps free` |
| `Kicks every cell for me` | `The room leans close to me` |
| `The low line walks through` (Room Return) | `My red nail follows through` |

Major Deviations records the bounded Gate-13 cleanup. The fixed bass contour and final-`t` percussion restart remain in the style prompt and Disc channels.

## P06 V4 — Leave the Ruby Line

Frozen-field correction:

- `High-shine digital power-pop/electroclash, 118 BPM, ...` → `High-shine digital power-pop/electroclash, 122 BPM, ...`

Exact sung-line substitutions:

| Before | After |
|---|---|
| `The squeak turns clear and singing` | `My lipstick catches morning` |
| `Only at my stop` | `Only where I stop` |
| `Low road keeps on walking` | `My bare feet cross the room` |
| `Claps keep landing dry` | `Dawn warms my raised hand` |
| `I give the gleam a height` | `I want the next word bright` |

Major Deviations now records the 118→122 BPM QA correction and the bounded Gate-13 cleanup instead of claiming none. The squeak-to-crown consequence and invariant cadence remain in the style prompt and Disc channels.

## Preservation checks

- Every lyric replacement is one line for one line. Selected sung-line counts remain P04 V3 `76`, P05 V1 `80`, and P06 V4 `72`.
- Every new sung line occurs exactly once in its canonical file; no replacement duplicates an existing lyric line.
- Titles, hooks, forms, actions, emotional arcs, rhyme/return systems, consequences, current chassis, excludes, Disc channels, and dedications are preserved. Positive prompts are unchanged except for the two required tempo digits.
- All non-selected variations were left untouched.
- Complete ICB byte-prefixes remain exact:

| Canonical file | Prefix bytes | SHA-256 before/after | Exact |
|---|---:|---|---|
| `pair_04_step10_final_package_enhanced.md` | 112161 | `b7f59689af182c6b8566a13ac14c137fc41049aa5d719762cbc3be6a7c2f96c4` | YES |
| `pair_05_step10_final_package_enhanced.md` | 112162 | `4d4689af085237f7481015f46b6fb9449335b1f6b9c4e70d7728ef9c1d3b1460` | YES |
| `pair_06_step10_final_package_enhanced.md` | 112161 | `b7f59689af182c6b8566a13ac14c137fc41049aa5d719762cbc3be6a7c2f96c4` | YES |

## Validator results

Canonical command form:

`python skills/music/scripts/validate_suno_packages.py <canonical-file> --min-sung-lines 70 --max-sung-lines 120`

| Canonical file | Exit | Result |
|---|---:|---|
| `pair_04_step10_final_package_enhanced.md` | 0 | PASS |
| `pair_05_step10_final_package_enhanced.md` | 0 | PASS |
| `pair_06_step10_final_package_enhanced.md` | 0 | PASS |

Text packages only. No audio was rendered or assessed.
