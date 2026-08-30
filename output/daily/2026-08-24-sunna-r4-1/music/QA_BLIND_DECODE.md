# Sunna R4.1 — Blind QA Decode

The independent judge sealed `QA_BLIND_STAGE1.md` before reading any identified candidate or any private key. The coordinator opened the six withheld keys only after the judge completed the audit.

| Set | Sealed order | Decoded order | Candidate vs decoy | Golden vs decoy | Judge health |
|---:|---|---|---|---|---|
| 01 | B > C > A | candidate > golden > decoy | PASS | PASS | healthy |
| 02 | B > C > A | golden > candidate > decoy | PASS | PASS | healthy |
| 03 | A > B > C | candidate > golden > decoy | PASS | PASS | healthy |
| 04 | C > B > A | candidate > golden > decoy | PASS | PASS | healthy |
| 05 | C > A > B | golden > candidate > decoy | PASS | PASS | healthy |
| 06 | B > C > A | candidate > golden > decoy | PASS | PASS | healthy |

**Calibration:** PASS. Golden beats decoy 6/6; candidate beats decoy 6/6. The judge is not broken, and no blind-ranking repair is added beyond the independent report's already named Phase-1, set-uniqueness, Gate-9, vocalist-fingerprint, and Somatic failures.

This decode does not promote the run. `music/suno_render/QA_REPORT.md` remains controlling: the R4.1 attempt is a text-package **FAIL** and may not render or ship.
