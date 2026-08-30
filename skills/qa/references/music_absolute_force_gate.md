# Music Absolute Force Gate — judge-side only

> ⛔ **GOLDEN-OUTPUT QUARANTINE.** This reference and every control it names belong only in clean QA, render audit, and private refinement review. A generating agent receives Golden Moves, never these payloads.

Established at The Scientist's instruction on 2026-08-30 after the preregistered investigation at `output/analysis/2026-08-30_music-cohesion-investigation/`. That study found that the criticized cohort had fewer fragments and blind hard cuts than the named high bars, while a five-part causal “compulsion” checklist rated 5/6 criticized songs highly and only 2/5 high bars highly. The checklist measured completed plot/form machinery, not impact. Fresh QA reversed the old all-SHIP verdict to Overall REPAIR.

This is a **QA correction, not a creative restriction**. Cleverness, fragments, repetition, contemplation, complex form, visible labor, and delayed wants remain fully available. Do not restore the Remark Test under another name.

## The four evidence lanes never substitute for one another

1. **Compliance evidence:** ICB continuity, cardinality, counts, required fields, lineage, and package validity.
2. **Plain-lyric evidence:** what the sung words communicate before prompts, annotations, or explanations help.
3. **Production-intent evidence:** what a MUSIC PROMPT asks a future renderer to do. This is a plan, not an achieved event.
4. **Render evidence:** what the finished audio actually does, established only by `lofn-render-audit` under THE BLIND RULE.

A voice moving across stereo, a missing kick, a vowel arriving on time, or a bass reversal **cannot clear a lyric or Somatic objection before audio exists**. Conversely, a strong plain lyric does not prove the render survived.

## Stage A1 — freeze the plain-lyric read first

For every music finalist, the coordinator gives the fresh judge only shuffled **plain sung words**: no title identity, MUSIC PROMPT, EXCLUDE, `[Theme:]`, `[SONG FORM:]`, `Disc_` lines, EMO/performance headers, SFX directions, production dramaturgy, self-check, score, pair/run identity, or intended interpretation. Use `scripts/build_blind_set.py --mode lyrics` where possible. The key stays outside the blind directory.

The judge writes and saves the Stage-A receipt before seeing the full package or ICB. It answers with exact lyric evidence:

1. **Recognizable pressure:** what contradiction, debt, fear, appetite, relation, or condition could belong to a listener? A floor plan or plot summary is not an answer.
2. **First necessity:** which line first makes the next line matter? If none does, say so.
3. **First-cycle surface:** before the first major return finishes, is a human pressure already present, even if the singer's explicit want is deliberately withheld?
4. **Apparatus removal:** remove the name/explanation of the clever device. Does the pressure remain, or was the scene built to prove the device?
5. **Section continuity:** do later sections deepen one pressure, or repeatedly change jobs so the listener must rebuild the song's governing question?
6. **Personal recognition:** name the exact line a listener might borrow as her own, not merely understand.

Do **not** reduce this to “clear want + failure cost + irreversible act + chorus consequence + changed return.” Those are useful observations, not a compulsion score. `Spare Hair Tie`, `Triple Arch Over Me`, and `The Wedding Takes It Louder` prove that contemplation, unresolved contradiction, and continuous world-pressure can compel without a conventional plot turn.

### The first-cycle anti-rescue rule

A later bridge may complete the story; it cannot retroactively make the opening publish-ready. If the first cycle is principally thesis, policy, procedure, device setup, or abstract demonstration and the affected person/pressure arrives only afterward, Stage A is **REPAIR — LATE PERSON**. This does not require an early declared want: an already-costly pressure may be unmistakable before the singer admits what she wants.

## Stage A2 — the actual absolute controls

Every finalist in a high-bar Lofn run is compared as **plain lyrics** against all three core controls:

1. `Triple Arch Over Me` — `skills/music/references/suno_format_example_triple_arch.md`
2. `Spare Hair Tie` — `skills/qa/references/absolute_force_spare_hair_tie.md`
3. `Burning Bush With a Barcode` — `output/songs/2026-07-06_P05V1_burning-bush-with-a-barcode.md`

Use one known-mediocre decoy to test the judge. `ASK FOR IT BY NAME` and `The Wedding Takes It Louder` are challenge controls when the run foregrounds cleverness, formal machinery, constraint density, moral debt, or continuous craft-world pressure. The coordinator supplies their plain lyric payloads from the canonical catalogue; a missing challenge control is reported, never silently replaced with an easier song.

Readings are strict:

- **Decoy above any core control:** judge calibration is broken; halt and re-run with a clean judge.
- **Candidate above decoy:** proves only “not known filler.” It provides **zero positive evidence for SHIP**.
- **Candidate below all three core controls:** `REPAIR — BELOW ABSOLUTE BAR`.
- **Mixed candidate/control result:** at most `HOLD`; name the exact dimension it wins and the exact pressure it loses.
- **Candidate above all three with concrete lyric evidence:** eligible to continue through the other gates, never an automatic SHIP.

The Scientist's ear remains the release authority. A model bracket cannot waive her finding that the song missed the bar.

## Stage B1 — proxy-capture and production-dependence checks

Before the Somatic votes, write two sentences:

- **Pressure sentence:** “The song matters because …” using only sung evidence.
- **Mechanism sentence:** “The song's formal device does …”

If the mechanism sentence is clearer or more consequential than the pressure sentence, flag **PROXY CAPTURE**. If removing an unrendered production instruction removes the decisive human consequence, flag **PRODUCTION-DEPENDENT TEXT**. Either flag is at least HOLD; if two Skeptics agree, REPAIR.

Concrete detail does not clear this gate by presence. A chair, card, train, wrapper, blue door, hand, or body noise may be an occupiable prop and still exist mainly to demonstrate a formal operation. Ask what pressure it carries.

## Stage B2 — independent Somatic votes

The three Skeptics freeze votes independently before discussion on:

1. **Recognition:** is there a listener-ownable pressure before decoding?
2. **Necessity:** does the lyric create a need for the next line/return, not merely a successful rekey?
3. **Evidence boundary:** does the lyric work without imagined production?

Every NO cites exact words and a counter-move. One supported NO on Recognition or Evidence Boundary makes the overall verdict **at most HOLD** until the report records direct lyric-surface counterevidence; scores, majority enthusiasm, compliance, and promised production cannot erase it. Two of three NO votes on any axis block the piece and route it to REPAIR.

## Report receipt

Every music `QA_REPORT.md` adds:

- `## Plain-Lyric First Read` — frozen Stage-A answers;
- `## Absolute High-Bar Bracket` — candidate, all three core controls, decoy, rank, and evidence;
- `## Evidence Boundary` — which claims are text, intent, or rendered fact;
- `## Proxy-Capture Check` — pressure sentence, mechanism sentence, verdict;
- `Absolute Force Verdict: CLEAR / HOLD / REPAIR` beside Pipeline, Package, Human-Subject, and Overall.

No receipt means the music Overall verdict cannot be SHIP.
