from __future__ import annotations

from pathlib import Path
import sys


RUN_DIR = Path(__file__).resolve().parent
ICB = (RUN_DIR / "CREATIVE_CONTEXT.md").read_text(encoding="utf-8")

CONTRACT = r"""

# STEP 11 CLEAN-ROOM REFINEMENT CONTRACT

You are a new refinement intelligence. The Step 06–10 author is not in this
context. Read the canonical Step 11 instruction separately and refine every
one of the four Step-10 variations below. This packet is the only creative
input; Golden outputs, the Golden index, examples, prior runs, and other pairs
are forbidden generator context.

## Output

- Write one canonical file named `pair_{PAIR}_step10_final_package_enhanced.md`.
- Begin it byte-for-byte with the frozen ICB at the top of this packet.
- Preserve four independent complete packages; do not collapse the pair.
- Use recognized headings `# V1 — Title` through `# V4 — Title`.
- For every package use, in order:
  - `## SUNO STYLE PROMPT`
  - `## SUNO EXCLUDE PROMPT`
  - `## SUNO ENHANCED LYRICS`
  - `## 3. TITLE`
  - `## DISC CHANNELS (5)`
  - `## DEDICATION`
  - `## Major Deviations`
- End with a within-pair ranking and one recommendation.

## Hard gates per package

- Style prompt: one dense paragraph, 850–1000 characters, target 870–960,
  terminal punctuation; no real artist names and no negative/exclude prose.
- Exclude prompt: concrete comma-separated failure classes, 400–900
  characters, always under 1000.
- Lyrics field: under 5000 characters, target 70–120 probable sung lines and
  hard floor 60; Theme, SONG FORM, full section-specific EMO performance
  script, and at least one meaningful short standalone SFX cue.
- First-person singer remains the subject. Hope is appetite pointed at a
  specific future. Frivolity may be load-bearing and has no ceiling.
- Current electronic chassis is load-bearing from the first two seconds; keep
  the one inherited bodily intelligence, never rebuild a complete period
  ensemble. Only P04 may use the run's full club consequence.
- Preserve the pair's invariant and changed layer. The final return must pay
  for any removals with a named returning element.
- At most one researched fact in a song; spell digits in sung text; no real
  people, children, or procedure debris in generator fields.
- Dedication is measured at 500 characters or fewer and names what was
  borrowed and who it is owed to without claiming generated audio is a sample.
- The Disc block has exactly five render-useful channels, including a
  describe-render prediction and a declared survival risk.
- `## Major Deviations` must explicitly say `None` when there are none.

Run the canonical Suno package validator and pair-level checks available for
Step 11. Repair only this pair until deterministic gates pass. Text packages
are not renders: identify production claims that remain unverified.
"""


def tail_after_icb(text: str, source: Path) -> str:
    if not text.startswith(ICB):
        raise SystemExit(f"ICB prefix mismatch: {source}")
    return text[len(ICB):].lstrip("\n")


selected_pairs = [int(value) for value in sys.argv[1:]] if len(sys.argv) > 1 else list(range(1, 7))
if any(pair < 1 or pair > 6 for pair in selected_pairs):
    raise SystemExit("pair arguments must be integers from 1 through 6")

for pair in selected_pairs:
    nn = f"{pair:02d}"
    source = RUN_DIR / f"pair_{nn}_step10_revision_synthesis.md"
    if not source.exists():
        raise SystemExit(f"missing Step 10 input: {source}")
    step10 = source.read_text(encoding="utf-8")
    packet = (
        ICB.rstrip()
        + "\n"
        + CONTRACT.replace("{PAIR}", nn).rstrip()
        + f"\n\n# PAIR {nn} STEP-10 INPUT\n\n"
        + tail_after_icb(step10, source).rstrip()
        + "\n"
    )
    target = RUN_DIR / f"pair_{nn}_step11_dispatch_packet.md"
    target.write_text(packet, encoding="utf-8", newline="\n")
    print(f"{target.name}\t{len(packet.encode('utf-8'))} bytes")
