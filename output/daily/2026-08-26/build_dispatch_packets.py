from __future__ import annotations

import json
from pathlib import Path


RUN = Path(__file__).resolve().parent
ICB = (RUN / "CREATIVE_CONTEXT.md").read_bytes().replace(b"\r\n", b"\n")
PAIRS = json.loads((RUN / "concept_medium_pairs.json").read_text(encoding="utf-8"))

contract = """

# PAIR-CHAIN EXECUTION CONTRACT

This packet is the complete generator context. Read it from first byte to last before drafting. Golden-song payloads are quarantined and absent. Use only the Golden Move instructions and the two reference names in the handoff.

Run five genuinely separate passes and write five canonical artifacts in the run root:

1. Step 06 → `pair_{pair_id}_step06_facets.md`: exactly five pair-specific scoring facets; run the supplied Concept, Medium, and Context-panel pressure; do not invent another panel.
2. Step 07 → `pair_{pair_id}_step07_song_guides.md`: four genuinely distinct variation guides. Open each guide with explicit answers to **Where is the body standing?** and **What could hurt it here?** Name separate title, action/seam, section architecture, rhyme/return system, bold sonic device, current chassis, changed layer, invariant layer, one-fact decision, and predicted generic-render failure.
3. Step 08 → `pair_{pair_id}_step08_generation.md`: four complete songs, not sketches. Each candidate gets title, dense positive style paragraph, separate exclude field, Theme, SONG FORM, full taxonomy-valid EMO section headers, standalone SFX, and full lyrics. No real artist/group names in style, exclude, or lyrics.
4. Step 09 → `pair_{pair_id}_step09_artist_refined.md`: refine all four without any artist imitation or name in generator fields; preserve pair isolation and repair the most consequential mismatch only.
5. Step 10 → `pair_{pair_id}_step10_revision_synthesis.md`: four complete final candidate packages plus within-pair ranking. Every candidate must carry a ≤500-character measured dedication and an exact self-check. Predict the literal first five seconds, vocal placement, groove, bold-device landing, and the one way it would render generic; self-repair once if needed.

Each of the five written files must begin byte-for-byte with the complete ICB that begins this packet; append the step body after the ICB. Never edit `CREATIVE_CONTEXT.md`, shared coordinator files, another pair namespace, `RUN_STATE.md`, or `INDEX.md`.

Hard package contract for every Step-10 candidate:

- positive `## 1. MUSIC PROMPT`: one 850–1000-character dense prose paragraph, target 870–960, terminal punctuation; genre/tempo/energy → female vocalist core → signature device → physical palette → vocal staging → arrangement arc; vivid first-five-seconds world, spatial language, kinetic defect, explicit acoustic status; no avoidances in this paragraph;
- `## 1B. SUNO EXCLUDE PROMPT`: 400–900 characters, hard max 1000, concrete negative tokens/failure classes;
- lyrics field opens `[Theme: ...]`, then `[SONG FORM: ...]`; every section header uses `[Section - EMO:Taxonomy Emotion(s) - role - cue]`; at least one standalone `*SFX*`; call/response uses `Call (Response)` on one line; 70–120 target sung lines, hard floor 60, exact field under 5000 characters, digits in sung lyrics spelled out;
- song is Sunna's own first-person want, with a concrete place, risk, action, midpoint turn, and a return structure; at most one scientific/numeric fact, emotionally answered;
- current chassis load-bearing in first two seconds, one inherited body relation, changed and invariant layers honored, no period ensemble; P04 alone may use a full club bed;
- no Golden Song content, no real artist/group names in generator-facing fields, no children, no body-inventory filler, no production or QA debris in sung lines;
- dedication is public text: no claims about living people, no implication that generated audio contains a sample, credit any named sound figure/tradition honestly, measure ≤500 characters.

Daily isolation: the four variations may share the pair seed but must vary their whole answer/seam blocks, section maps, hook grammars, and survivor-fragment jobs. They may not become one lyric with nouns swapped.
"""

for pair in PAIRS:
    pid = pair["pair_id"]
    brief = (RUN / f"pair_{pid}_step05_isolated_brief.md").read_text(encoding="utf-8")
    assignment = json.dumps(pair, ensure_ascii=False, indent=2)
    tail = (
        contract.format(pair_id=pid)
        + "\n\n# NORMALIZED PAIR RECORD\n\n```json\n"
        + assignment
        + "\n```\n\n# PAIR-ISOLATION BRIEF\n\n"
        + brief
    ).replace("\r\n", "\n").encode("utf-8")
    target = RUN / f"pair_{pid}_dispatch_packet.md"
    target.write_bytes(ICB + tail)
    print(f"{target.name}: {target.stat().st_size} bytes")

