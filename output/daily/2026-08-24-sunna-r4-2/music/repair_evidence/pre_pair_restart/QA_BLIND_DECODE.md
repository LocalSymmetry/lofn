# Sunna R4.2 — Blind QA Decode

Identity keys were opened only after `QA_BLIND_STAGE1.md` was written. This file records every decode, including the unfavorable and uncalibrated orders.

## Original six-set plate

| Set | Blind order | Decode | Candidate vs decoy | Golden vs decoy |
|---:|---|---|---|---|
| 01 | B > A > C | B candidate · A golden · C decoy | WIN | WIN |
| 02 | B > C > A | B candidate · C golden · A decoy | WIN | WIN |
| 03 | B > A > C | B candidate · A golden · C decoy | WIN | WIN |
| 04 | C > B > A | C golden · B candidate · A decoy | WIN | WIN |
| 05 | A > B > C | A candidate · B golden · C decoy | WIN | WIN |
| 06 | C > A > B | C candidate · A golden · B decoy | WIN | WIN |

**Original calibration:** healthy. Candidate beats decoy 6/6; golden beats decoy 6/6. Five candidates place first; original P04 places second between golden and decoy.

## P04 routed repair

The original blind critique said P04's accumulated `sweet/sugar/lime` alliteration sounded written to a device. Root removed surplus pet names and modifier-alliteration while preserving one adult `sweetie` call, the physical salt/shaker consonance, every action/form obligation, and all soundcraft floors. The current P04 package is therefore not byte-identical to original set 04 and required a fresh blind check.

| Reblind | Blind order | Decode | Candidate placement | Control order |
|---|---|---|---:|---|
| repair 1 | B > A > C | B candidate · A decoy · C `Triple Arch` golden | 1 | decoy > golden |
| repair 2 | B > A > C | B candidate · A decoy · C `Five Wrong Colors` golden | 1 | decoy > golden |
| repair 3 | C > A > B | C candidate · A decoy · B `Blue Screen` golden | 1 | decoy > golden |

**Repair result:** the current P04 candidate placed first before three independent clean-context judges and beat both comparison members 3/3. However, every repair mini-set inverted the golden/decoy order. Those mini-sets are therefore **not used as calibration evidence**. The variance is disclosed rather than sampled away. The original six-set plate supplies the healthy calibration; the three repair sets supply only a consistent comparative read of the repaired candidate.

## Evidence

- Blind judgments: `QA_BLIND_STAGE1.md`, `QA_BLIND_P04_REPAIR.md`, `QA_BLIND_P04_REPAIR2.md`, `QA_BLIND_P04_REPAIR3.md`
- Sealed identity maps: `qa_keys/`
- Anonymous inputs: `qa_blind/`, `qa_blind_repair/`, `qa_blind_repair2/`, `qa_blind_repair3/`
