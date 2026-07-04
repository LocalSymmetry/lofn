# KNOB CENSUS — every magic number in the Lofn pipeline, with file:line provenance

**Status:** WS0 deliverable (Lofn Creative Studio plan §3.1, §11). This is the grounded,
file:line-cited census that turns each canon magic number into a studio **knob**. Every
row below was located by grepping `skills/**`, `.claude/skills/**`, `vault/gates.yaml`,
and the modality `OVERALL_PROMPT_TEMPLATE.md` files on the commit this was written against
(git HEAD `dc4ea6b`). Line numbers can drift as canon is edited — the **anchoring token**
column names the phrase to re-grep if a line moves.

Two classes of number:

- **gates.yaml-bound** — a deterministic threshold `validate_step.py` reads live from
  `vault/gates.yaml`. The knob binds it via `gates_key`; the run-local gates materializer
  (WS2) writes a knob-adjusted `output/studio/<run>/gates.yaml` and passes it with the new
  `validate_step.py --gates PATH` flag (WS0 canon micro-patch). These are the enforced
  numbers.
- **prose-bound** — a cardinality stated only in skill/template prose (pair count, panel
  shape, warm-up discard). The knob reaches the model via the resolver's **RUN PARAMETERS**
  block and, where load-bearing mid-instruction, an **overlay** (plan §3.3). No gate
  enforces these directly; the portfolio/QA stages catch gross violations.

> **Canon immutability (S1):** this census is a *reference*. Studio never edits the cited
> canon files — it restates the numbers in `flows/**` knob presets (which `--meta-check`
> skips, per the WS0 skiplist) and overrides them at run time via `--gates`.

---

## §3.1 core table (required) + everything else found

| Knob | Default | gates_key | Where the number lives today (file:line — anchor token) |
|---|---|---|---|
| `n_pairs` | 6 | — (prose) | `.claude/skills/lofn/SKILL.md:47` ("6 pair assignments"); `:103` ("Music default cardinality: 6 pairs × 4 variations = 24"); `:138` (per-pair invariant, "6 pairs × 4 variations = 24 outputs"); `.claude/skills/lofn-music/SKILL.md:29` ("6 pairs fan out"); `:43` ("**exactly 6** — only the Scientist downsizes"); `.claude/skills/lofn/EXECUTION.md:46` ("one wave of 6 pairs") |
| `n_variations_per_pair` | 4 | — (prose) | `.claude/skills/lofn/SKILL.md:103,138` ("× 4 variations = 24"); `.claude/skills/lofn-music/SKILL.md:29` ("6 pairs × 4 variations = 24 songs"); enforced downstream as `total_prompts` in `vault/gates.yaml:86` |
| `barbell_accessible` / `barbell_ambitious` | 3 / 3 | — (derived, prose) | `.claude/skills/lofn/SKILL.md:47` ("3 ACCESSIBLE + 3 AMBITIOUS"); `:103` ("3 ACCESSIBLE + 3 AMBITIOUS"); `.claude/skills/lofn-daily/SKILL.md:76` ("pairs 1–3 ACCESSIBLE, pairs 4–6 AMBITIOUS … best 3 from each arm"); `:149` ("3+3"). Derived: `ceil(n_pairs/2)` / `n_pairs − accessible`. |
| `n_concepts` (step 02) | 12 | — (prose) | `.claude/skills/lofn-music/SKILL.md:38` ("`step02_concepts.md` (12 concepts)"); `:43` ("which of the 12 concepts fills each slot"); `tools/explorer/pipeline_map.yaml:219` (g.tree_branches note ">=12 concepts at step 02"). Invariant: `n_concepts >= 2 * n_pairs`. |
| `panel_count` × `voices_per_panel` | 3 × 6 = 18 | — (prose) | `.claude/skills/lofn/SKILL.md:88` ("Convene THREE panels of 6 voices each = 18 voices"); `:45` ("18 voices, 3 Hyper-Skeptics"); `:29` ("all 18 panel voices"); `skills/music/OVERALL_PROMPT_TEMPLATE.md:28,43` ("Panel Ledger (18 voices)", "USE these exact 18 voices") |
| `n_flairs` | 15 | — (prose) | `.claude/skills/lofn/SKILL.md:97` ("15 Special Flairs"); `:29` ("all 15 Special Flairs"); `skills/music/OVERALL_PROMPT_TEMPLATE.md:23,51` ("The 15 Special Flairs", "15 Special Flairs (weave these throughout)") |
| `taxonomy_cardinality` | 50 | `taxonomy_cardinality` | `vault/gates.yaml:78`; step-00 prose ("exactly 50 aesthetics … 50 emotions … 50 frames … 50 genres") e.g. `skills/music/steps/00_Generate_Music_Aesthetics_And_Genres.md:39` |
| `step00_min_bytes` | 2000 | `step00_min_bytes` | `vault/gates.yaml:76` ("A thin Step-00 file means the tree never branched") |
| `music_prompt_chars` | [850, 1000] | `music_prompt_chars` | `vault/gates.yaml:20`; also hardcoded in `scripts/validate_step.py:44` DEFAULT_GATES + `:618` (`prompt_chars < 850 or > 1000`) |
| `music_prompt_chars_target` | [870, 960] | `music_prompt_chars_target` | `vault/gates.yaml:34` (write INTO the mid-band) |
| `music_prompt_hug_ceiling` | 985 | `music_prompt_hug_ceiling` | `vault/gates.yaml:35` (measured chars >= this → FLAG boundary_hugging) |
| `music_prompt_terminal_punctuation` | true | `music_prompt_terminal_punctuation` | `vault/gates.yaml:42` (HARD: prompt must end as a complete sentence) |
| `sung_lines` | [70, 120] | `sung_lines` | `vault/gates.yaml:27`; validator floor `scripts/validate_step.py:653` ("<60 triggers repair, target 70-120") |
| `sung_lines_target` | [78, 110] | `sung_lines_target` | `vault/gates.yaml:36` |
| `sung_lines_floor_hug` | 72 | `sung_lines_floor_hug` | `vault/gates.yaml:37` (sung lines <= this → FLAG floor_hugging) |
| `suno_lyrics_field_max` | 5000 | `suno_lyrics_field_max` | `vault/gates.yaml:23` (Suno LYRICS field hard render cap) |
| `suno_lyrics_field_target` | 4800 | `suno_lyrics_field_target` | `vault/gates.yaml:24` (soft target under the cap) |
| `max_sung_numeric_facts` | 1 | `max_sung_numeric_facts` | `vault/gates.yaml:50` (one-fact rule; FLAG only) |
| `image_min_words` | 80 | `image_min_words` | `vault/gates.yaml:82`; validator `scripts/validate_step.py:233`; daily prose `.claude/skills/lofn-daily/SKILL.md:90` ("noun-first present-tense ≥80 words") |
| `total_prompts` | 24 | `total_prompts` | `vault/gates.yaml:86` ("Total variation prompts across the 6 pairs (4 each)"). Derived: `n_pairs * n_variations_per_pair`. |
| `unique_line_ratio_floor` | 0.45 | `unique_line_ratio_floor` | `vault/gates.yaml:117` (below → FLAG collapse; chorus EXEMPT); also `scripts/validate_step.py:529` (`unique_ratio < 0.45`) |
| `ngram_collapse_n` | 4 | `ngram_collapse_n` | `vault/gates.yaml:118` |
| `concept_warmup_discard` | 17 (concepts), 14 (media) | — (prose) | Concepts: `skills/music/OVERALL_PROMPT_TEMPLATE.md:85` ("Generate 17 concepts … discard … start at Concept 18; use 18–50"); `image:72`, `story:72`, `video:72`. Media/arrangements: `skills/music/OVERALL_PROMPT_TEMPLATE.md:90` ("Brainstorm 14 … discard … start at Arrangement 15; use 15–27"); `image:73`, `story:73`, `video:73`. Also `skills/video/steps/02_…:544` ("generate 17 initial concepts"). |
| `daily_funnel` | 24 → 12 → 6 | — (prose) | `.claude/skills/lofn-daily/SKILL.md:14` ("24 prompts → top 12 → top 6"); `:22` ("music (24→6) and image (24→12→6)"); `:90` ("24 prompts → rank → top 12 → top 6"); `:89` ("24 songs (6×4)") |
| `max_concurrency` | 6 (12 → cap-and-stagger) | — (prose) | `.claude/skills/lofn/EXECUTION.md:46` ("Max concurrency: 6 standard (one wave of 6 pairs)"); `:203` ("Max concurrency: 6 standard"); `.claude/skills/lofn-daily/SKILL.md:94–96` ("cap-and-stagger … ~6 at a time, not all 12"); `:153` |
| `retry_max_attempts` | 3 | — (arg default) | `scripts/validate_with_retries.py:27` (`--max-attempts` default 3); usage line `:12`; `tools/explorer/pipeline_map.yaml:306` (validate_with_retries usage). FlowSpec `on_fail.max_attempts: 3`. |

---

## Additional numbers found (not in the §3.1 table — surfaced per WS0 "add any you find")

These are grounded numbers that are candidates for future knobs; they are **not** wired as
knobs in the two shipped presets (`classic-6x4`, `wide-10x2`) but are catalogued here so the
census is complete and the Flow Lab can promote them later.

| Number | Value | Where (file:line — anchor) | Notes |
|---|---|---|---|
| `banned_imperative_openers` | [Create, Design, Make, Render, Depict] | `vault/gates.yaml:92–97`; validator `scripts/validate_step.py:53,346` | HARD string gate (image noun-first opener). A list, not a scalar — a knob would be the list contents. |
| `ban_words` (haze words) | [ethereal, dreamlike, whimsical, gentle light, soft glow, magical, delicate] | `vault/gates.yaml:102–110`; validator `:54,242` | FLAG-level string list. |
| `house_lexicon` | 13 calcified golden phrases | `vault/gates.yaml:59–72`; validator `:64,279,318` | FLAG-only self-copy guard (golden output → house formula). |
| step-00 artifact min length (validator) | 800 chars | `scripts/validate_step.py:519` ("too short to be a real Lofn step") | A separate, hardcoded generic floor distinct from `step00_min_bytes` (2000). |
| generic repetition floor (validator) | 0.45 unique-line ratio | `scripts/validate_step.py:529` | Duplicated in gates.yaml as `unique_line_ratio_floor`; the `_validate` body uses a hardcoded 0.45 for the generic pre-check. |
| step-05 pair JSON band | 4–7 pairs | `scripts/validate_step.py:588` (`4 <= len(data) <= 7`) | The validator accepts 4–7 pairs in `concept_medium_pairs.json` — the practical range around `n_pairs=6`. A `n_pairs` knob outside [4,7] would need this validator band widened (noted for WS2/WS3). |
| full-section EMO headers floor | `max(6, lyric_count * 4)` | `scripts/validate_step.py:635` | Derived from `n_variations_per_pair` (4 per lyric); an implicit dependency on the variations knob. |
| sung-lines hard repair floor (validator) | 60 | `scripts/validate_step.py:653` ("<60 triggers repair") | Distinct from the gates.yaml `sung_lines[0]=70` band; the validator's own hard floor. |
| distinctiveness caps | prompt 0.58 / lyric 0.42 / 5-gram Jaccard 0.18 | `tools/explorer/pipeline_map.yaml:295` (g.portfolio note); `scripts/validate_portfolio_distinctiveness.py` (`--max-prompt-sim 0.58 --max-lyric-sim 0.42 --max-ngram-jaccard 0.18`, usage at `pipeline_map.yaml:305`) | Run-level portfolio gate thresholds — future knobs bound to that script's flags rather than gates.yaml. |
| orchestration cardinality | "6+ pairs × 4 outputs", "10 steps, 3 panels" | `skills/orchestration/SKILL.md:195` | Restates `n_pairs`/`n_variations`/`panel_count` from the orchestrator side; kept consistent by `--meta-check`. |

---

## How each knob reaches the model / the gate (plan §3.3)

1. **RUN PARAMETERS block** (resolver, WS2) — an authoritative studio-owned section appended
   to CREATIVE CONTEXT stating the run's numbers and instructing them to override any
   remembered value. Covers every prose-bound knob universally.
2. **Overlays** (`flows/overlays/*.patch.md`) — exact-match find→replace hunks for steps where
   a number is load-bearing mid-instruction (e.g. the step-05 "exactly 6" line). Apply-or-error,
   never fuzzy; canon files never touched.
3. **Run-local gates** — the materializer writes `output/studio/<run>/gates.yaml` from the
   `gates_key`-bound knobs and passes `--gates` to `validate_step.py`, so the deterministic
   checks enforce the *experiment's* numbers. This is the WS0 canon micro-patch's whole purpose.

## Invariants enforced by the presets

```
n_concepts >= 2 * n_pairs
barbell_accessible + barbell_ambitious == n_pairs
total_prompts == n_pairs * n_variations_per_pair
sung_lines[0] < sung_lines_target[0] < sung_lines_target[1] < sung_lines[1]
music_prompt_chars[0] < music_prompt_chars_target[0]  &&  music_prompt_chars_target[1] < music_prompt_chars[1]
```

Violations render inline in the Knobs UI and **block Run** (not Compile — you can preview a
broken preset, you can't spend money on one).
