# Top 6 Refinement QA

Run: `2026-07-05_hard_fought_hope`  
Refinement set: `refined_top6/`  
Status: PASS

## Refined Files

| Rank | Title | File | Agent | Validator |
| --- | --- | --- | --- | --- |
| 1 | Bring The Chairs Up | [01_bring_the_chairs_up.md](refined_top6/01_bring_the_chairs_up.md) | Noether | PASS |
| 2 | Answer The Siren | [02_answer_the_siren.md](refined_top6/02_answer_the_siren.md) | Euler | PASS |
| 3 | Not Fixed, Still Rolling | [03_not_fixed_still_rolling.md](refined_top6/03_not_fixed_still_rolling.md) | Volta | PASS |
| 4 | Keep Tomorrow Open | [04_keep_tomorrow_open.md](refined_top6/04_keep_tomorrow_open.md) | Archimedes | PASS |
| 5 | Two Windows Catch | [05_two_windows_catch.md](refined_top6/05_two_windows_catch.md) | Mill | PASS |
| 6 | One More Morning Clicked On | [06_one_more_morning_clicked_on.md](refined_top6/06_one_more_morning_clicked_on.md) | Kierkegaard | PASS |

Validation command:

```powershell
$files = Get-ChildItem output/music/2026-07-05_hard_fought_hope/refined_top6/*.md | Sort-Object Name | ForEach-Object { $_.FullName }
python skills/music/scripts/validate_suno_packages.py @files
```

Result: all six refined packages returned `PASS`.

## Leakage Scan

Artist/influence proper-name scan returned no matches in `refined_top6/`.

## Similarity Check

The top-six refinement lowered lexical sameness across the finalist set.

| Set | Music Prompt Avg | Lyrics Avg | Combined Avg |
| --- | ---: | ---: | ---: |
| Original top 6 | 0.185 | 0.112 | 0.136 |
| Refined top 6 | 0.171 | 0.086 | 0.111 |

## Shared Motif Reduction

| Motif | Original Count / Files | Refined Count / Files | Result |
| --- | ---: | ---: | --- |
| `not fine` | 4 / 3 | 0 / 0 | removed |
| `still` | 26 / 6 | 11 / 1 | isolated to bus title/hook lane |
| `two bright` | 1 / 1 | 0 / 0 | removed |
| `dark corner` | 3 / 3 | 0 / 0 | removed |
| `answer` | 26 / 6 | 19 / 1 | isolated to siren title/hook lane |
| `answered` | 5 / 2 | 0 / 0 | removed |
| `teeth` | 9 / 6 | 6 / 1 | isolated to bus lane |
| `morning` | 14 / 3 | 12 / 1 | isolated to clinic title/hook lane |
| `open` | 16 / 5 | 8 / 1 | isolated to repair-cafe title/hook lane |
| `room` | 24 / 6 | 5 / 1 | mostly removed |

## Distinctive Tokens After Refinement

- `01_bring_the_chairs_up.md`: bring, chair, chairs, phone, buzz, scrape, glass, cracked, screen, split.
- `02_answer_the_siren.md`: siren, kettle, lid, table, street, pass, steam, tape.
- `03_not_fixed_still_rolling.md`: fixed, rolling, gang, rail, teeth, brake, jaw, route, air.
- `04_keep_tomorrow_open.md`: fan, tomorrow, open, wire, radio, bench, flux, bead, solder, knock.
- `05_two_windows_catch.md`: windows, catch, pane, tar, late, roof, cable, boot, rail.
- `06_one_more_morning_clicked_on.md`: clicked, relay, seal, tray, cold, bite, chain, label.

## Verdict

PASS. The six refinements preserve the top-six hooks while reducing pair bleed. The refined set should be preferred for first audio renders over the original `songs/` top six.
