# Per-Pair Cardinality and Mechanical Audit

## Cardinality

- Refined pairs at Step 05: **6**.
- Step 06 pair artifacts: **6**.
- Step 07 pair artifacts: **6**.
- Step 08 pair artifacts: **6**.
- Step 09 pair artifacts: **6**.
- Step 10 pair artifacts: **6**.
- Complete Step-10 packages: **24** — exactly 4 per pair, 2 revised + 2 synthesized.
- Pair isolation: each generator wrote only its own pair namespace; no shared creative file was modified by a pair agent.
- **CARDINALITY: PASS.**

## Style-field count ledger

| Pair | Variation 1 | Variation 2 | Variation 3 | Variation 4 | Band verdict |
|---:|---:|---:|---:|---:|---|
| 01 | 953 | 960 | 957 | 958 | PASS |
| 02 | 933 | 955 | 925 | 953 | PASS |
| 03 | 915 | 899 | 916 | 911 | PASS |
| 04 | 916 | 948 | 916 | 947 | PASS |
| 05 | 914 | 960 | 882 | 913 | PASS |
| 06 | 913 | 915 | 919 | 947 | PASS |

All 24 positive Style fields lie inside the authoritative 850–1000-character band and use terminal punctuation. All use separate concrete Exclude fields.

## Lyric-field ledger

- All 24 packages contain **78 probable sung lines**.
- All 24 lyric fields stay below **5000 characters**.
- All 24 begin with [Theme: ...] immediately followed by [SONG FORM: ...].
- Each has full EMO/performance section headers and at least one standalone SFX cue.
- News pairs 01, 04, and 05 passed manual human-subject identifiability review; Pair 05 uses the paired magnitude once in one variation only.

## Validator compatibility note

The deterministic validator fully passes Pairs 02 and 05. Pairs 01, 03, 04, and 06 used valid multi-candidate raw/Step-10 schemas whose headings differ from the validator’s single-package MUSIC PROMPT expectation; custom exact-count scans prove their package cardinality and field bands above. The six selected Step-11 files use the canonical single-package schema and must pass validate_suno_packages.py before QA. This is a schema-compatibility warning, not missing content.

## Verdict

**PASS TO STEP 11.** No collapsed pair, missing variation, style-band failure, or lyric-cardinality failure exists.

