---
type: pipeline_index
run_name: "nightcafe-select-unbracing"
created: 2026-07-12T00:06:08-07:00
modality: image
pairs_generated: 6
variations_per_pair: 4
total_generated: 24
total_selected: 8
personality: "LOFN-ArtCore"
panel_transforms: ["REFLECT", "BRIDGE"]
status: "SHIP_FOR_RENDERING"
---

# Pipeline Run: UNBRACING

## Outcome

The complete Lofn image pipeline selected **P06 — Failure Becomes Mesh** as the strongest eight-frame story engine for NightCafe Select, *The Architecture of Feeling*. The final prompt-only package has passed clean-context adversarial QA and is ready for the three-frame render pilot. No image has been rendered or submitted.

## Environmental scan

The visible field favors family/life montage, fairy romance, cute fantasy quests, autobiography, topical anthropomorphic video, and conventional apocalypse. Those entries often have strong recurring characters and quick premises, but architecture usually behaves as scenery. This run occupies the white space: relief after prolonged vigilance made visible as a causal change in load, support, posture, and affordance.

## Panel process

Three six-seat panels, each containing a Hyper-Skeptic, transformed the Golden Seed **“A body learns that an opening is not the same thing as a collapse.”** The decisive insight was that the feeling should occur when the load path changes, while the body remains uncertain that it is no longer needed as structure. Six distinct concept-medium chains produced 24 final prompts; a separate evaluator ranked all 24 and then judged eight-frame expandability independently from single-image strength.

## Pairs generated

| Pair | Engine | Arm | Best single | Global rank | Engine result |
|---:|---|---|---|---:|---|
| 01 | Gilded Compression | ACCESSIBLE | V1 — The Seam She Polishes | 6 | cut |
| 02 | Brocade Load Ledger | ACCESSIBLE | V2 — The Net Remembers | 9 | cut |
| 03 | Missing Tessera Plan | ACCESSIBLE | V1 — The Tile Refuses | 11 | cut |
| 04 | Negative Cast | AMBITIOUS | V1 — The Cast Stands | 1 | render-evidence fallback |
| 05 | Counterweight Shadow | AMBITIOUS | V1 — No Body Under It | 3 | cut |
| 06 | Permeable Wire Room | AMBITIOUS | V1 — Failure Becomes Mesh | 2 | **selected engine** |

P04V1 was the top standalone image at 9.26/10. P06 won the separate engine adjudication at 48.5/50 because it offers eight materially distinct causal states and the strongest transformed motif.

## Final eight-frame Body of Work

1. [The Bearing Line](../images/2026-07-11_nightcafe_select_unbracing/20260711_235401_the_bearing_line_P06_F01.md) — body and brace share the load.
2. [The Work That Multiplies](../images/2026-07-11_nightcafe_select_unbracing/20260711_235402_the_work_that_multiplies_P06_F02.md) — competent repair accumulates.
3. [A Room Shaped Like Vigilance](../images/2026-07-11_nightcafe_select_unbracing/20260711_235403_room_shaped_like_vigilance_P06_F03.md) — repair becomes enclosure.
4. [The Last Turn](../images/2026-07-11_nightcafe_select_unbracing/20260711_235404_the_last_turn_P06_F04.md) — she refuses one more repair.
5. [Failure Becomes Mesh](../images/2026-07-11_nightcafe_select_unbracing/20260711_235405_failure_becomes_mesh_P06_F05.md) — the brace breaks and four routes catch.
6. [Where the Weight Went](../images/2026-07-11_nightcafe_select_unbracing/20260711_235406_where_the_weight_went_P06_F06.md) — redistribution becomes visible.
7. [The Room Holds](../images/2026-07-11_nightcafe_select_unbracing/20260711_235407_the_room_holds_P06_F07.md) — architecture carries itself.
8. [Ordinary Hands](../images/2026-07-11_nightcafe_select_unbracing/20260711_235408_ordinary_hands_P06_F08.md) — one ordinary task and one residual check.

## Submission package

- [Final entry package](FINAL_ENTRY_PACKAGE.md)
- [Character and topology bible](CHARACTER_MATERIAL_BIBLE.md)
- [Artist statement](ARTIST_STATEMENT.md)
- [NightCafe render protocol](NIGHTCAFE_RENDER_PROTOCOL.md)
- [Release/falsification record](release_record.json)
- [24-candidate evaluation](eval_ranking.md)

## Quality gates

- [Initial adversarial QA](QA_REPORT.md): REPAIR — provenance manifest only; no creative rewrite.
- [Manifest repair delta](QA_REPAIR_DELTA.md): SHIP to packaging.
- [Final package QA](QA_FINAL_REPORT.md): **SHIP for rendering and mandatory post-render QA**.
- Final deterministic gate: eight prompts, 142–150 words, exact identity lock 8/8, noun-first 8/8, banned renderer terms 0, artist statement 131 words, release record valid.

## Immediate render decision

Render Frames **1, 5, and 8** first. Frame 5 must show one adult, one broken black member, at least three receiving endpoints, a ceiling response, and lagging body posture at phone-grid size and in grayscale. One controlled simplification is allowed. If the gate still fails—or two pilot frames lose the same identity/topology requirement—switch the whole engine to P04.

## Run health

- `pairs_shipped`: 6
- `pairs_quarantined`: 0
- `total_gate_retries`: 13
- `qa_repairs_issued`: 1
