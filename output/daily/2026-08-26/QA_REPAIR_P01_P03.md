# Gate-13 Repair — P01–P03

Scope was limited to the selected variation in each canonical cumulative artifact: P01 V1, P02 V3, and P03 V3. No unselected variation, P04–P06 artifact, standalone selected file, prompt, exclude, Disc channel, dedication, title, hook, form, action, consequence specification, or ICB byte-prefix was edited.

## P01 V1 — Boop the Future Anyway

Sung-line substitutions:

- `Beat one holds the floor` → `Both feet hold the floor`
- `Ooh—oop above the beat` → `Ooh—oop, red climbs my wrist`
- `Oop before the word` → `Oop—I lean toward it`

The three substitutions keep the floor image, the returned `oop`, the physical reach, the line count, and the existing meter footprint while removing beat-placement and word-placement reportage.

## P02 V3 — I Start Unfinished

Sung-line substitutions:

- `Another bright voice grows` → `Our crooked picture glows`
- `The highest note keeps opening` → `The cold blue flash keeps finding us`
- `The low two notes stay sure` → `We lean back in for more`

The substitutions retain the `grows/glows` sound return, the `more` rhyme, the shared booth action, and the line count while leaving the fourth–seventh–octave consequence and invariant two-note walk in the prompt and Disc channels.

## P03 V3 — Roll Me Into Weather

EMO-header normalizations, preserving the same emotional sequence:

- `EMO:Cold-Gate Vigilance Opens` → `EMO:Curiosity, Apprehension`
- `EMO:Filtered-Air Unease Tightens` → `EMO:Apprehension, Dread`
- `EMO:Maybe-Rain Fascination Leans` → `EMO:Wonder, Longing`
- `EMO:High-Answer Curiosity Circles` → `EMO:Curiosity, Wonder`
- `EMO:Four-Voice Alarm Flashes` → `EMO:Shock, Apprehension`
- `EMO:Windowless Dread Keeps Moving` → `EMO:Dread, Resolve`
- `EMO:Hard-Consonant Resolve Pushes` → `EMO:Resolve, Defiance`
- `EMO:Open-Air Resolve Brightens` → `EMO:Resolve, Hope`
- `EMO:Three-Register Liberation Gleams` → `EMO:Empowerment, Wonder`
- `EMO:Future-Rain Appetite Laughs` → `EMO:Excitement, Hope`
- `EMO:Dry-Stop Acceptance Settles` → `EMO:Relief, Hope`

Sung-line substitutions:

- `My song sits ready` → `The case waits by me`
- `Into no refrain` → `Into unnamed rain`
- `The high voice circles` → `My hand turns the wheel`
- `The low stays bare` → `Cold air meets me there`
- `Air rhymes with dare` → `I breathe one dare`
- `Then leaves it there` → `And leave my fear there`
- `One at the handle` → `One hand grips the handle`
- `Two at the rail` → `One shoe waits by the rail`
- `Four in the doorway` → `I face the open doorway`
- `None at the center` → `I lean into the cart`
- `Air through the vowels` → `Cold slips under my collar`
- `Ticks through the tail` → `Wheels click past blue tape`
- `My song goes wider` → `The case rolls toward weather`
- `Before it goes out` → `I follow it out`
- `The fact is a door` → `The cold door gives an inch`
- `My answer is push` → `Both my hands say push`
- `My title goes first` → `The case rolls out first`
- `Not guaranteed use` → `With no promised room`
- `I push through the seam` → `I push past the seal`
- `Put cool rain in my sound` → `Put cool rain on my tongue`
- `Low voice stays rooted` → `My hands leave the handle`
- `High voices wear air` → `I lift my face to air`
- `My song wants skin` → `My skin wants rain`
- `No cleaner take` → `My hands still shake`

These one-for-one substitutions retain the case, cart, wheel, rail, blue tape, threshold, push, weather, rain, `air/dare/there` return, bodily risk, and future appetite while removing voice-count/register, production, rhyme-procedure, consequence-ledger, seam, title, song, and take self-reporting. The low-cell and vocal-position mechanism remains unchanged in the style prompt, performance headers, and Disc channels.

## Validation

Command:

```text
python skills/music/scripts/validate_suno_packages.py output/daily/2026-08-26/pair_01_step10_final_package_enhanced.md output/daily/2026-08-26/pair_02_step10_final_package_enhanced.md output/daily/2026-08-26/pair_03_step10_final_package_enhanced.md --min-sung-lines 70 --max-sung-lines 120
```

Result:

```text
PASS output\daily\2026-08-26\pair_01_step10_final_package_enhanced.md
PASS output\daily\2026-08-26\pair_02_step10_final_package_enhanced.md
PASS output\daily\2026-08-26\pair_03_step10_final_package_enhanced.md
```

Sung-line counts remain exactly unchanged: P01 V1 `79`, P02 V3 `82`, P03 V3 `88`.

The ICB source is `112160` bytes with SHA-256 `64E46DADF745D3BB5C92EC7EE6523AE4C5B644CF3AED8F31EFF6612F434D0958`. The first `112160` bytes of all three repaired canonical artifacts match that source byte-for-byte and have the same prefix hash.
