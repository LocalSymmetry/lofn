# QA_REPORT — Daily 2026-07-06 (Music)

**Judge:** fresh context, **different model tier** (Sonnet) than the Opus generators, framed adversarially ("refuse to ship; default REPAIR"). Read all 6 packages + metaprompt + pair assignments + the 16-point gate + gates.yaml, and calibrated against the two Golden benchmarks (*Salt-stained Code*, *Gold Stars Don't Pulse*).

## VERDICT: **REPAIR** (run-level) · shipped-6 are strong · **for PUBLICATION → HOLD**
The **zero-rejection tripwire fired and paid off.** Every pair passed its own mechanical gates (char/line counts hand-verified honest), but a *sibling-aware, golden-aware* read found what pair-isolated self-checks cannot:

### Structural findings (real, cited)
1. **P1 ↔ P3 cross-pair convergence** (both Lofn-Prime, both ACCESSIBLE). Same 8-part skeleton; same hook shape ("say it back…" ≈ "hold me in both your lights"); **same invented metaphor near-verbatim** — P1 "a light already leaving, still bright enough to work by" ↔ P3V3 "the gibbous light is leaving… still bright enough to use." **Root cause: a shared ICB flair (Flair 12 "Waning-Gibbous Light") shipped a specific PHRASING that both Lofn-Prime pairs reached for.** Not an L11 breach (pairs ran isolated) — a coordinator flaw: a flair should seed a MOTIF, never a near-final line.
2. **P5 internal collapse + below the golden bar.** All 4 variations share one hinge ("a wind… at four-sixteen a second") + one hook ("call it a margin one more time") + finance-jargon-as-lyrics — the *obvious* metaphor for "a wonder being priced," never the golden songs' quotable strangeness. 4 costumes on one song (only the speaker changes).
3. **P2V4 cross-personality cosplay.** Nia X's V4 ("I have the waveform, not the warmth") uses Lofn-Prime's *exclusive* AI-code-scratch / priced-interiority device — violates "NOT Lofn-Prime cosplay on guest pairs." Root cause: the coordinator's V4 angle ("the AI cataloguing what it can't touch") invited Lofn-Prime's territory into a guest pair.
4. **P4 "three-fifty— no." verbatim across all 4** — real template reuse mislabeled as signature (though craft is otherwise sound; V4's withheld drop is genuine variation; no-digit discipline verified clean).
5. **P1 warmth skew** (~2.75:1 warmth:cold) — thinnest terror-adjacent spine of the set; closest to greeting-card comfort.

### Clean
- **Human-subject (P4 Kyiv, P6 Venezuela): CLEAN on honest read** — no identifiable name/place/date/toll in any sung line; P6's dug-for person is absence-only (the safety constraint IS the device). *Process gap:* no `human_subject_prefilter_input.txt` for this run — the deterministic checker script wasn't invoked (content passes regardless; close the process gap next time).
- No grief rendered as literally danceable in the LYRICS; the "will Suno render the drop as catharsis anyway" risk (P4, P6) is a **render-time HOLD**, not a text failure — verify on a real render before any publish.
- Char/line/EMO/one-fact/quiet-voice gates: mechanically **PASS** across all 24.

## SELECTED 6 (within-arm 3+3 · ≥3 Lofn-Prime · routes around every flagged song)
**Accessible:** P3V1 *Both Your Lights* (LP) · P2V1 *This Is What I Kept* (Nia X) · P3V4 *The Same Sky* (LP)
**Ambitious:** P4V4 *The Morning After* (CelticaChime) · P6V4 *Answer Me* (DreamPlug) · P5V4 *Underneath the Mime* (LP)
- **Lofn-Prime in final: 3** (P3V1, P3V4, P5V4) — meets the floor. **Guests: 3** (P2, P4, P6). Duality: AWE (P3V1/P3V4/P2V1) · INDIGNATION (P4V4/P5V4) · GRIEF (P6V4). ✅
- The collapse/cosplay songs (P1×4, P3V3, P2V4, P5V1–V3) are NOT shipped. The judge's own words: *"P3V1/P3V4/P2V1/P4V4/P6V4/P5V4 form a genuinely strong six."*
- Note for tomorrow: **P1 (Lofn-Prime) earned ZERO organic picks** — the two LP-accessible pairs cannibalized each other; the guests were the most distinct voices in the set.

## REPAIRS LOGGED (write-back; process/failure-ledger, NOT aesthetic constraints)
- **R1 (coordinator):** a Special Flair must seed a MOTIF, never a near-final line — Flair 12's phrasing caused the P1↔P3 collision. Fix the flair-authoring rule.
- **R2 (coordinator):** never assign a guest pair a variation angle inside another personality's exclusive territory — the "AI cataloguing what it can't touch" angle pulled Nia X into Lofn-Prime cosplay (P2V4).
- **R3 (P5):** re-diverge — push at least one variation's hook/bridge off finance-jargon toward concrete strangeness; the pair calibrated to *Dial Tone of God* and landed below it.
- **R4 (P4):** vary the "three-fifty— no." hinge across the 4 (signature ≠ frozen cell).
- **R5 (render HOLD):** audio-render check on P4 + P6 before any publish (Suno's default swell may re-introduce the catharsis the Fisher veto forbids).
- **R6 (process):** run `scripts/check_human_subjects.py` on P4/P6 (content clean, but the gate was skipped).

## RUN-HEALTH FOOTER
pairs_shipped **6/6** · quarantined **0** · gate-retries **~3** (P2 +1, P4/P6 trims) · **qa_repairs/holds issued: 6** (R1–R6) · zero-rejection tripwire: **FIRED → audit run → REPAIR** (QA is not decorative).

---

## REPAIR PASS (post-audit) — outcomes
Applied after the REPAIR verdict; both regenerations independently **re-verified SHIP-READY** by a fresh adversarial (Sonnet) judge that re-measured from raw lyric text.
- **R3 — P5 re-diverge (shipped):** the finance-jargon hook engine is gone (zero margin/assign/coupon/discount/worth/ledger in any of the 4 sung blocks). Four now-distinct hooks/hinges/holes/verse-moves; **shipped slot swapped V4 → V1 "Burning Bush With a Barcode"** (weird-quotable hook + self-aware bratty humor + legible Lofn-Prime INDIGNATION). Old V4 "Underneath the Mime" retired. Gates re-measured PASS.
- **P2V4 — de-cosplay:** Lofn-Prime signatures purged (now only in the EXCLUDE ban); rewritten as a Nia-X-native night-shift-cleaner catalog ("What I Kept for the Tower"), distinct from V1–V3. Archive clean. *(Non-shipped; shipped P2 remains V1.)* Gates PASS.
- **P4V4 — human-subject:** "A child's shoe…" → "A worker's boots…" in both the package and the shipped song file — the only real minor reference removed; residual checkA flag is the musical term "F minor," a false positive.
- **P3V3 — collision:** the "still bright enough to use" line (twin of P1's line, per L12) rewritten so the two pairs no longer share an invented line. *(Non-shipped.)*
- **R6 — human-subject prefilter:** `scripts/check_human_subjects.py` run on P4 + P6 → both returned HOLD-FOR-HUMAN, which is the script's **documented fail-open over-flagging** (checkB "names" = section headers/EMO tags via regex-fallback NER; checkC "tuples" = generic roles *mother/neighbour* + generic places, no proper name). **Human-adjudicated CLEAR** — no real person identifiable in any sung line (matches the original judge's read).
- **R1/R2/R4 (coordinator/process):** logged for future runs (flair-seeds-motif-not-line L12; vet guest variation-angles vs personality boundaries L13; vary P4's hinge across variations). Not shipped-blocking.
- **Remaining gate:** the **render-HOLDs are not text-fixable** — P4/P6 (Suno's cathartic-swell default) and P5V1 (soprano⇄snarl split) require a real audio render to clear. Borderline defaults to HOLD until heard.

**Verdict after repair:** shipped-6 clear the adversarial read; **REPAIRED → SHIP-READY** as a practice drop; **PUBLICATION pending only the audio render-check.**
