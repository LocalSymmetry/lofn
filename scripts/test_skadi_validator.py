#!/usr/bin/env python3
"""Tests for scripts/validate_music_atlas.py — the SKAÐI reader.

Adversarial by design: each test fabricates the defect the check exists to catch
and asserts that the reader FAILS. A gate that cannot fail is a sentence.

House convention note: this repo keeps unit tests BESIDE the scripts they test
(`scripts/test_*.py`, nine existing files), not in a `scripts/tests/` package.
DESIGN v1 §4 suggested `scripts/tests/`; the existing convention wins.

    python3 -m unittest discover -s scripts -p "test_skadi*.py"
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_music_atlas as V  # noqa: E402

GATES_YAML = """\
music_prompt_chars: [850, 1000]
# --- MUSIC ATLAS (SKADI) ---
atlas_probe_slots: 6
atlas_probe_slots_min: 2
atlas_min_positive_controls: 1
atlas_min_negative_controls: 1
atlas_non_obvious_quota: 1
atlas_non_obvious_min_slots: 4
atlas_confidence_cap_n: 4
atlas_bank_min_n: 2
atlas_verdicts: [ARRIVES, PARTIAL, SMOOTHED, ABSENT, INVERTED, TRANSFORMS]
atlas_channels: [style, exclude, lyric_form, emo, section]
atlas_lanes: [NUMERIC, LISTENER, BOTH]
atlas_delta_moves: [NEW-BEHAVIOR, BEHAVIOR-CONFIRMED, SCOPE-NARROWED, OPERATOR-LAW, INSTRUMENT-ONLY, NONE, RE-AUTHORIZED]
atlas_streak_reauthorize: 3
atlas_scruff_rift_sightings: 3
atlas_expedition_prefix: output/expeditions/
"""


def md_table(cols, rows):
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


CLAIM_A = "A single BPM target ARRIVES within 5 BPM."
CLAIM_B = "Literal foley props are ABSENT or abstracted."


class Fixture:
    """A minimal but fully valid atlas on disk, plus knobs to break it."""

    def __init__(self, root: Path):
        self.root = root
        (root / "vault" / "music_atlas" / "seeds").mkdir(parents=True)
        (root / "evidence" / "listen").mkdir(parents=True)
        (root / "vault" / "gates.yaml").write_text(GATES_YAML, encoding="utf-8")
        (root / "evidence" / "measurements.json").write_text("{}", encoding="utf-8")
        (root / "evidence" / "listen" / "P1.blind.json").write_text("{}", encoding="utf-8")
        (root / "evidence" / "listen" / "P1.intent.json").write_text("{}", encoding="utf-8")
        self.banked = [self.claim_row("SK-001", "steady-tempo", "NUMERIC", CLAIM_A,
                                      "ARRIVES", 6, "MEDIUM",
                                      "`evidence/measurements.json`"),
                       self.claim_row("SK-002", "foley", "LISTENER", CLAIM_B,
                                      "ABSENT", 4, "MEDIUM",
                                      "`evidence/listen/P1.blind.json`, "
                                      "`evidence/listen/P1.intent.json`")]
        self.observed: list[list] = []
        self.conjectured: list[list] = []
        self.receipts = [["SK-001", f"`{V.sha12(CLAIM_A)}`", "SEED", "imported"],
                         ["SK-002", f"`{V.sha12(CLAIM_B)}`", "SEED", "imported"]]
        self.delta_rows: list[list] = []
        self.scruff = [["SCR-01", "2026-07-24", "an unrequested stutter", "P1", "WATCH"]]

    @staticmethod
    def claim_row(cid, cls, lane, claim, verdict, n, conf, evidence):
        return [cid, "CELL", f"any territory × {cls}", "style", lane, claim, verdict,
                n, conf, "v4.5", evidence, "the observation that would refute it"]

    def write(self):
        atlas = self.root / "vault" / "music_atlas"
        (atlas / "ATLAS.md").write_text("\n".join([
            "<!-- generated -->", "", "# ATLAS", "",
            "## BANKED", "", md_table(V.ATLAS_COLS, self.banked), "",
            "## OBSERVED", "", md_table(V.ATLAS_COLS, self.observed), "",
            "## CONJECTURED", "", md_table(V.ATLAS_COLS, self.conjectured), "",
            "## APPEND BLOCKS", "", "### 2026-08-17 seed additions", "",
            md_table(V.APPEND_COLS, self.receipts), "",
        ]), encoding="utf-8")
        (atlas / "TERRITORIES.md").write_text(
            md_table(V.TERRITORY_COLS,
                     [["Jazz", "Jazz, Bop", "Blues", "UNEXPLORED", "songs:1 · clips:0",
                       "(none yet)"]]), encoding="utf-8")
        (atlas / "SPEC_CLASSES.md").write_text(
            md_table(V.SPEC_CLASS_COLS, [
                ["steady-tempo", "NUMERIC", "one BPM target held",
                 "`scripts/measure_render.py` → `tempo_candidates()`", "ledger n=6"],
                ["foley", "LISTENER", "a literal prop sound",
                 "SKADI-BLIND-v1 Q6/Q7", "ledger n=4"],
            ]), encoding="utf-8")
        (atlas / "FRONTIER.md").write_text("\n".join([
            "# FRONTIER", "",
            "## COVERAGE", "",
            md_table(V.COVERAGE_COLS,
                     [["Jazz", "UNEXPLORED", 0, 0, "—", "(none yet)"]]), "",
            "## MENU", "",
            md_table(V.MENU_COLS,
                     [["P1", "Jazz × steady-tempo", "style", "does it arrive",
                       "control due", "PROPOSED"]]), "",
            "## SCRUFF SHELF", "", md_table(V.SCRUFF_COLS, self.scruff), "",
            "## CONTROL CHART", "",
            md_table(V.CONTROL_CHART_COLS,
                     [["steady-tempo", "baseline", "ARRIVES", "6 of 6", "v4.5"]]), "",
            "## DELTA LOG", "", md_table(V.DELTA_COLS, self.delta_rows), "",
            "## CONSUMPTION LOG", "", md_table(V.CONSUMPTION_COLS, []), "",
        ]), encoding="utf-8")
        (atlas / "EXPEDITION_LEDGER.md").write_text(
            md_table(V.LEDGER_COLS, []), encoding="utf-8")
        return self


class ValidatorCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.fx = Fixture(self.root)
        self._patches = [
            mock.patch.object(V, "REPO", self.root),
            mock.patch.object(V, "ATLAS_DIR", self.root / "vault" / "music_atlas"),
            mock.patch.object(V, "GATES", self.root / "vault" / "gates.yaml"),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._tmp.cleanup)
        for p in self._patches:
            self.addCleanup(p.stop)

    def run_all(self):
        self.fx.write()
        gates, _ = V.load_gates()
        rep = V.Report()
        V.run_all(gates, rep)
        return rep

    def failures(self, rep):
        return {r["check"]: r["detail"] for r in rep.rows if r["ok"] is False}


class TestGatesFailOpen(ValidatorCase):
    def test_missing_gates_file_uses_defaults_and_warns(self):
        (self.root / "vault" / "gates.yaml").unlink()
        gates, warnings = V.load_gates()
        self.assertTrue(warnings, "a missing gates file must warn, never pass silently")
        self.assertEqual(gates["atlas_probe_slots"], V.DEFAULTS["atlas_probe_slots"])
        self.assertEqual(gates["atlas_verdicts"], V.DEFAULTS["atlas_verdicts"])

    def test_unparseable_gates_file_falls_back_and_still_reads_inline_lists(self):
        # A tab makes PyYAML refuse the document outright.
        (self.root / "vault" / "gates.yaml").write_text(
            "\tbroken: [\n" + GATES_YAML, encoding="utf-8")
        gates, warnings = V.load_gates()
        self.assertTrue(warnings)
        self.assertEqual(gates["atlas_bank_min_n"], 2)
        # ⛔ The MUSIC ATLAS section is written entirely in inline flow lists. A
        # fallback that returns them as a string hands every vocabulary check a
        # set of single characters and passes.
        self.assertEqual(gates["atlas_verdicts"][0], "ARRIVES")
        self.assertIn("TRANSFORMS", gates["atlas_verdicts"])
        self.assertEqual(gates["atlas_expedition_prefix"], "output/expeditions/")

    def test_fallback_parses_block_lists_too(self):
        parsed = V._fallback_parse("house_lexicon:\n  - a phrase\n  - another\nn: 4\n")
        self.assertEqual(parsed["house_lexicon"], ["a phrase", "another"])
        self.assertEqual(parsed["n"], 4)

    def test_clean_fixture_passes(self):
        rep = self.run_all()
        self.assertEqual(self.failures(rep), {}, rep.render())


class TestAtlasSchema(ValidatorCase):
    def test_broken_evidence_path_fails(self):
        self.fx.banked[0][10] = "`evidence/does-not-exist.json`"
        rep = self.run_all()
        self.assertIn("every evidence path exists on disk", self.failures(rep))

    def test_n_below_cap_with_high_confidence_fails(self):
        self.fx.banked[0][7] = 2
        self.fx.banked[0][8] = "HIGH"
        rep = self.run_all()
        detail = self.failures(rep).get("confidence capped by n (gates atlas_confidence_cap_n)")
        self.assertIsNotNone(detail, rep.render())
        self.assertIn("SK-001", detail)

    def test_banked_below_min_n_fails(self):
        self.fx.banked[0][7] = 1
        rep = self.run_all()
        self.assertIn("BANKED respects the n floor (gates atlas_bank_min_n)",
                      self.failures(rep))

    def test_unknown_verdict_fails(self):
        self.fx.banked[0][6] = "MOSTLY-ARRIVED"
        rep = self.run_all()
        self.assertIn("row values use the canonical vocabulary", self.failures(rep))

    def test_empty_falsifier_fails(self):
        self.fx.banked[0][11] = ""
        rep = self.run_all()
        self.assertIn("scope, claim, models and falsifier are non-empty", self.failures(rep))

    def test_reworded_claim_breaks_its_receipt(self):
        self.fx.banked[0][5] = CLAIM_A + " Actually it always arrives."
        rep = self.run_all()
        self.assertIn("claim text matches its recorded sha12", self.failures(rep))

    def test_malformed_table_row_fails_loudly(self):
        self.fx.write()
        path = self.root / "vault" / "music_atlas" / "ATLAS.md"
        text = path.read_text(encoding="utf-8")
        # Drop one cell from a real row: the classic silent-skip case.
        broken = text.replace("| SK-001 | CELL |", "| SK-001 |", 1)
        path.write_text(broken, encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_atlas_schema(V.ATLAS_DIR, gates, rep)
        failed = self.failures(rep)
        self.assertIn("ATLAS.md parses against the row schema", failed)
        self.assertIn("malformed row", failed["ATLAS.md parses against the row schema"])

    def test_malformed_row_is_never_silently_dropped(self):
        rows, problems = V.parse_table(
            "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 |\n", ["a", "b"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(problems), 1)
        self.assertIn("malformed row at line 4", problems[0])

    def test_table_cells_preserves_pipes_in_code_spans(self):
        self.assertEqual(V.table_cells("| a | `x|y` | c |"), ["a", "`x|y`", "c"])
        self.assertEqual(V.table_cells(r"| a | x\|y | c |"), ["a", r"x\|y", "c"])


class TestLanes(ValidatorCase):
    def test_listener_claim_without_heard_evidence_fails(self):
        self.fx.banked[1][10] = "`evidence/measurements.json`"
        rep = self.run_all()
        detail = self.failures(rep).get("LISTENER-lane claims cite HEARD evidence")
        self.assertIsNotNone(detail, rep.render())
        self.assertIn("SK-002", detail)

    def test_conjectured_listener_row_is_not_held_to_the_lane_rule(self):
        # A registered prediction cannot have been heard: no render exists yet.
        # It carries n=0 and no confidence — see TestConjecturedShelf.
        self.fx.banked.pop(1)
        self.fx.receipts.pop(1)
        self.fx.conjectured.append(
            Fixture.claim_row("SKOP-01", "blend-of-two", "LISTENER", CLAIM_B,
                              "TRANSFORMS", 0, "—", "`evidence/measurements.json`"))
        self.fx.conjectured[-1][1] = "OPERATOR"
        self.fx.receipts.append(["SKOP-01", f"`{V.sha12(CLAIM_B)}`", "SEED", "imported"])
        rep = self.run_all()
        self.assertEqual(self.failures(rep), {}, rep.render())


class TestTerritoriesAndDelta(ValidatorCase):
    def test_unexplored_without_a_reason_fails(self):
        self.fx.write()
        path = self.root / "vault" / "music_atlas" / "TERRITORIES.md"
        path.write_text(path.read_text(encoding="utf-8").replace("(none yet)", ""),
                        encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_territory_states(V.ATLAS_DIR, gates, rep)
        self.assertIn("UNEXPLORED carries a reason or '(none yet)'", self.failures(rep))

    def test_closed_expedition_with_no_delta_row_fails(self):
        exp = self.root / "output" / "expeditions" / "2026-08-20_plate"
        exp.mkdir(parents=True)
        (exp / "INDEX.md").write_text("closed", encoding="utf-8")
        self.fx.write()
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_delta_log(V.ATLAS_DIR, gates, rep)
        detail = self.failures(rep).get("exactly one delta row per closed expedition")
        self.assertIsNotNone(detail, rep.render())
        self.assertIn("2026-08-20_plate", detail)

    def test_duplicate_delta_rows_fail(self):
        exp = self.root / "output" / "expeditions" / "2026-08-20_plate"
        exp.mkdir(parents=True)
        (exp / "INDEX.md").write_text("closed", encoding="utf-8")
        self.fx.delta_rows = [
            [1, "2026-08-20", "2026-08-20_plate", "NEW-BEHAVIOR", "1 new", "`evidence`"],
            [2, "2026-08-20", "2026-08-20_plate", "NONE", "0", "`evidence`"]]
        self.fx.write()
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_delta_log(V.ATLAS_DIR, gates, rep)
        self.assertIn("exactly one delta row per closed expedition", self.failures(rep))

    def test_non_moving_streak_flags_rather_than_fails(self):
        for i in range(3):
            exp = self.root / "output" / "expeditions" / f"2026-08-2{i}_plate"
            exp.mkdir(parents=True)
            (exp / "INDEX.md").write_text("closed", encoding="utf-8")
            self.fx.delta_rows.append(
                [i + 1, f"2026-08-2{i}", f"2026-08-2{i}_plate", "INSTRUMENT-ONLY", "0",
                 "`evidence`"])
        self.fx.write()
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_delta_log(V.ATLAS_DIR, gates, rep)
        streak = [r for r in rep.rows if "non-moving" in r["check"]][0]
        # A wall-map is a result: the streak is a FLAG that asks a human, never a FAIL.
        self.assertIsNone(streak["ok"])
        self.assertIn("RE-AUTHORIZE", streak["detail"])

    def test_scruff_rift_flags_at_threshold(self):
        self.fx.scruff = [["SCR-01", "2026-07-24", "a stutter", "P1, P4, P9", "WATCH"]]
        self.fx.write()
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_scruff(V.ATLAS_DIR, gates, rep)
        rift = [r for r in rep.rows if "rift detector" in r["check"]][0]
        self.assertIsNone(rift["ok"], "a rift is survey material, never a failure")
        self.assertIn("RIFT", rift["detail"])


class TestConjecturedShelf(ValidatorCase):
    """⛔ CONJECTURED IS A PREDICTION SHELF, NOT A PARKING SHELF (tester, 2026-08-17).

    `check_lanes` deliberately exempts CONJECTURED — a blind pass on an
    unrendered probe is impossible. That exemption was the whole shelf's
    governance: a row with n=9, HIGH confidence and measurement-only evidence
    parked there read as a registered prediction to every check and as a banked
    finding to every human.
    """

    def conjecture(self, n, confidence):
        row = Fixture.claim_row("SK-099", "foley", "LISTENER",
                                "An un-heard claim parked in CONJECTURED.",
                                "ARRIVES", n, confidence, "`evidence/measurements.json`")
        self.fx.conjectured.append(row)
        self.fx.receipts.append(
            ["SK-099", f"`{V.sha12('An un-heard claim parked in CONJECTURED.')}`",
             "SEED", "x"])
        return self.run_all()

    def test_a_prediction_carrying_observations_fails(self):
        detail = self.failures(self.conjecture(9, "HIGH")).get(
            "CONJECTURED rows are predictions, not observations")
        self.assertIsNotNone(detail)
        self.assertIn("n='9'", detail)
        self.assertIn("confidence='HIGH'", detail)

    def test_a_prediction_carrying_a_confidence_fails(self):
        detail = self.failures(self.conjecture(0, "LOW")).get(
            "CONJECTURED rows are predictions, not observations")
        self.assertIsNotNone(detail)
        self.assertIn("confidence='LOW'", detail)

    def test_a_prediction_with_no_n_and_no_confidence_passes(self):
        self.assertEqual(self.failures(self.conjecture(0, "—")), {})

    def test_a_dash_in_the_n_column_is_equally_legal(self):
        self.assertEqual(self.failures(self.conjecture("—", "—")), {})


class TestPlaceholdersNeverPass(ValidatorCase):
    """A placeholder is not empty, so every non-emptiness gate passes it."""

    def test_a_placeholder_model_epoch_fails(self):
        self.fx.banked[0][9] = "(declare the model epoch)"
        detail = self.failures(self.run_all()).get(
            "no cell carries a writer's placeholder text")
        self.assertIsNotNone(detail)
        self.assertIn("models", detail)
        # And it must NOT be caught only by the non-emptiness check, which passes:
        self.assertNotIn("scope, claim, models and falsifier are non-empty",
                         self.failures(self.run_all()))

    def test_a_placeholder_falsifier_fails(self):
        self.fx.banked[0][11] = "(name the observation that would refute this)"
        self.assertIn("no cell carries a writer's placeholder text",
                      self.failures(self.run_all()))

    def test_a_placeholder_epoch_on_the_control_chart_fails(self):
        self.fx.write()
        path = self.root / "vault" / "music_atlas" / "FRONTIER.md"
        path.write_text(path.read_text(encoding="utf-8")
                        .replace("| 6 of 6 | v4.5 |", "| 6 of 6 | (declare the model epoch) |"),
                        encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_control_chart(V.ATLAS_DIR, gates, rep)
        self.assertIn("every chart point names its model epoch", self.failures(rep))

    def test_the_menu_may_still_print_left_for_a_mind(self):
        # DESIGN §2 REQUIRES the literal in FRONTIER's MENU. The placeholder scan
        # is scoped to the ATLAS and the chart precisely so this stays legal.
        self.fx.write()
        path = self.root / "vault" / "music_atlas" / "FRONTIER.md"
        path.write_text(path.read_text(encoding="utf-8")
                        .replace("| does it arrive |", f"| {V.NON_OBVIOUS_PLACEHOLDER} |"),
                        encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.run_all(gates, rep)
        self.assertEqual(self.failures(rep), {}, rep.render())


class TestEvidencePathsAreReCheckable(ValidatorCase):
    """`_resolves` decides whether a stranger could re-open the citation."""

    def test_a_path_escaping_the_repo_fails(self):
        self.fx.banked[0][10] = "`../private-claw/TOOLS.md`"
        self.assertIn("every evidence path exists on disk", self.failures(self.run_all()))

    def test_a_path_that_climbs_out_and_back_fails(self):
        # `vault/../MEMORY.md` reads as a vault path and is not one.
        (self.root / "MEMORY.md").write_text("private", encoding="utf-8")
        self.fx.banked[0][10] = "`vault/../MEMORY.md`"
        self.assertIn("every evidence path exists on disk", self.failures(self.run_all()))

    def test_a_directory_is_not_a_receipt(self):
        (self.root / "evidence" / "adir").mkdir(parents=True, exist_ok=True)
        self.fx.banked[0][10] = "`evidence/adir`"
        self.assertIn("every evidence path exists on disk", self.failures(self.run_all()))

    def test_an_absolute_path_fails(self):
        self.fx.banked[0][10] = f"`{(self.root / 'evidence' / 'measurements.json')}`"
        self.assertIn("every evidence path exists on disk", self.failures(self.run_all()))

    def test_the_ordinary_repo_relative_file_still_resolves(self):
        self.assertEqual(self.failures(self.run_all()), {})
        self.assertTrue(V._resolves("evidence/measurements.json"))


class TestTierRederivation(ValidatorCase):
    """⛔ SKILL.md: the tier is "derived from artifacts on disk by skadi_bank.py
    and RE-DERIVED BY THE VALIDATOR". It was not, so a hand-written HEARD passed
    clean with no listen artifacts, no wav and no RENDERS.json."""

    def plate(self, tier="HEARD", artifacts=False, dir_exists=True):
        exp = self.root / "output" / "expeditions" / "2026-08-20_plate"
        if dir_exists:
            exp.mkdir(parents=True)
            (exp / "PLAN.md").write_text(
                "frontier_read: `vault/music_atlas/FRONTIER.md`\n\n"
                + md_table(V.PLAN_COLS,
                           [["P3", "PROBE", "Jazz × steady-tempo", "style", "h", "—", "p"]]),
                encoding="utf-8")
            if artifacts:
                (exp / "P3_song.md").write_text(
                    "## 1. MUSIC PROMPT\nx\n\n## 2. LYRICS\ny\n\n## 3. TITLE\nz\n",
                    encoding="utf-8")
                wav = exp / "wav" / "P3.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                wav.write_bytes(b"RIFF")
                listen = exp / "listen"
                listen.mkdir(parents=True, exist_ok=True)
                (listen / "P3.blind.json").write_text("{}", encoding="utf-8")
                (listen / "P3.intent.json").write_text("{}", encoding="utf-8")
                (exp / "RENDERS.json").write_text(json.dumps({"P3": {
                    "clip_id": "clip-3", "wav": wav.relative_to(self.root).as_posix(),
                    "measurements": {"duration_s": 1.0}, "matched_by": "exact"}}),
                    encoding="utf-8")
        self.fx.write()
        (self.root / "vault" / "music_atlas" / "EXPEDITION_LEDGER.md").write_text(
            md_table(V.LEDGER_COLS,
                     [["2026-08-20", "2026-08-20_plate", "P3", "Jazz × steady-tempo", "style",
                       "ARRIVES", "clip-3", "ARRIVES", tier, "—"]]), encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_tier_rederivation(V.ATLAS_DIR, gates, rep)
        return rep

    def test_a_fabricated_heard_tier_is_caught(self):
        detail = self.failures(self.plate()).get(
            "every ledger tier is re-derived from artifacts")
        self.assertIsNotNone(detail)
        self.assertIn("INFLATED", detail)
        self.assertIn("not even a package on disk", detail)

    def test_the_same_row_passes_once_the_artifacts_exist(self):
        rep = self.plate(artifacts=True)
        self.assertEqual(self.failures(rep), {}, rep.render())

    def test_a_tier_below_the_derived_one_is_never_raised(self):
        # The validator only ever LOWERS. A row banked before the listening pass
        # landed is honest, not inflated.
        rep = self.plate(tier="RENDERED", artifacts=True)
        self.assertEqual(self.failures(rep), {}, rep.render())

    def test_an_expedition_name_that_is_a_path_is_refused_not_followed(self):
        # A ledger row is DATA. `../..` in the expedition column must not send a
        # reader walking the disk looking for artifacts to believe.
        rep = self.plate(dir_exists=False)
        (self.root / "vault" / "music_atlas" / "EXPEDITION_LEDGER.md").write_text(
            md_table(V.LEDGER_COLS,
                     [["2026-08-20", "../../..", "P3", "Jazz × steady-tempo", "style",
                       "ARRIVES", "clip-3", "ARRIVES", "HEARD", "—"]]), encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_tier_rederivation(V.ATLAS_DIR, gates, rep)
        row = [r for r in rep.rows if "re-derived" in r["check"]][0]
        self.assertIsNot(row["ok"], False)
        self.assertIn("not a plain directory name", row["detail"])

    def test_a_row_with_no_expedition_directory_is_skipped_by_name(self):
        rep = self.plate(dir_exists=False)
        self.assertEqual(self.failures(rep), {}, rep.render())
        row = [r for r in rep.rows if "re-derived" in r["check"]][0]
        self.assertIn("skipped", row["detail"])
        self.assertIn("2026-08-20_plate/P3", row["detail"])

    def test_the_derivation_is_imported_not_reimplemented(self):
        # ⛔ One source of derivation, two readers. A second copy of the ladder
        # drifts, and the drift is invisible because both sides "check" it.
        import skadi_bank
        source = (Path(V.__file__).read_text(encoding="utf-8"))
        self.assertIn("import skadi_bank", source)
        self.assertNotIn("def stamp_tier", source)
        self.assertTrue(callable(skadi_bank.stamp_tier))


class TestLedgerClipCollision(ValidatorCase):
    def test_two_probes_bound_to_one_clip_fail(self):
        # attack_final.py case 4: the downstream shape of the matcher bug, and
        # the shape a hand-edited ledger takes.
        self.fx.write()
        (self.root / "vault" / "music_atlas" / "EXPEDITION_LEDGER.md").write_text(
            md_table(V.LEDGER_COLS, [
                ["2026-08-20", "2026-08-20_p", "P2", "Jazz × dynamics-arc", "style",
                 "ARRIVES", "clip-SAME", "ARRIVES", "HEARD", "—"],
                ["2026-08-20", "2026-08-20_p", "P3", "Jazz × dynamics-arc", "style",
                 "ARRIVES", "clip-SAME", "ARRIVES", "HEARD", "—"]]), encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_ledger_schema(V.ATLAS_DIR, gates, rep)
        detail = self.failures(rep).get("one clip id answers at most one probe")
        self.assertIsNotNone(detail, rep.render())
        self.assertIn("clip-SAME", detail)

    def test_unrendered_rows_do_not_collide_on_the_dash(self):
        self.fx.write()
        (self.root / "vault" / "music_atlas" / "EXPEDITION_LEDGER.md").write_text(
            md_table(V.LEDGER_COLS, [
                ["2026-08-20", "2026-08-20_p", "P2", "Jazz × dynamics-arc", "style",
                 "ARRIVES", "—", "PENDING", "SPECIFIED", "—"],
                ["2026-08-20", "2026-08-20_p", "P3", "Jazz × dynamics-arc", "style",
                 "ARRIVES", "—", "PENDING", "SPECIFIED", "—"]]), encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_ledger_schema(V.ATLAS_DIR, gates, rep)
        self.assertNotIn("one clip id answers at most one probe", self.failures(rep))


class TestPreflight(ValidatorCase):
    def make_plate(self, non_obvious_hypothesis: str, registered: bool = True):
        exp = self.root / "output" / "expeditions" / "2026-08-20_plate"
        exp.mkdir(parents=True)
        slots = [
            ["P1", "POSITIVE-CONTROL", "any territory × steady-tempo", "style",
             "expect it to arrive", "ARRIVES", "control due"],
            ["P2", "NEGATIVE-CONTROL", "any territory × foley", "style",
             "expect it to vanish", "ABSENT", "control due"],
            ["P3", "PROBE", "Jazz × steady-tempo", "style", "does it arrive", "—", "opens cell"],
            ["P4", "PROBE", "Jazz × foley", "style", "does it arrive", "—", "opens cell"],
            ["P5", "PROBE", "Jazz × foley", "exclude", "does it leave", "—", "opens channel"],
            ["P6", "NON-OBVIOUS", "Jazz × foley", "style", non_obvious_hypothesis, "—",
             "the non-obvious slot"],
        ]
        (exp / "PLAN.md").write_text(
            "frontier_read: `vault/music_atlas/FRONTIER.md`\n\n"
            + md_table(V.PLAN_COLS, slots), encoding="utf-8")
        when = "2026-08-20T09:00:00Z" if registered else "—"
        pred = "ARRIVES" if registered else "—"
        (exp / "PREDICTIONS.md").write_text(
            md_table(V.PREDICTION_COLS,
                     [[s[0], s[2], s[3], pred, when] for s in slots]), encoding="utf-8")
        self.fx.write()
        return exp

    def test_literal_placeholder_refuses_the_plate(self):
        exp = self.make_plate(V.NON_OBVIOUS_PLACEHOLDER)
        gates, _ = V.load_gates()
        rep = V.Report()
        V.preflight(exp, gates, rep)
        detail = self.failures(rep).get(
            "non-obvious slot was filled by a mind, not by the planner")
        self.assertIsNotNone(detail, rep.render())
        self.assertIn("P6", detail)

    def test_filled_plate_passes_preflight(self):
        exp = self.make_plate("ask for the prop in the LYRIC instead of the style field")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.preflight(exp, gates, rep)
        self.assertEqual(self.failures(rep), {}, rep.render())

    def test_unregistered_predictions_refuse_the_plate(self):
        exp = self.make_plate("a real idea", registered=False)
        gates, _ = V.load_gates()
        rep = V.Report()
        V.preflight(exp, gates, rep)
        self.assertIn("every slot has a prediction registered before the render",
                      self.failures(rep))

    def test_control_without_an_expected_verdict_refuses_the_plate(self):
        exp = self.make_plate("a real idea")
        text = (exp / "PLAN.md").read_text(encoding="utf-8")
        (exp / "PLAN.md").write_text(
            text.replace("| ARRIVES | control due", "| — | control due", 1), encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.preflight(exp, gates, rep)
        self.assertIn("every control declares the verdict it expects", self.failures(rep))

    def test_preflight_exit_code_is_three(self):
        self.make_plate(V.NON_OBVIOUS_PLACEHOLDER)
        with contextlib.redirect_stdout(io.StringIO()):
            code = V.main(["--preflight", "output/expeditions/2026-08-20_plate", "--json"])
        self.assertEqual(code, 3, "a refused plate must exit 3, distinct from a failed audit")


class TestDownScaledSlices(ValidatorCase):
    """⛔ A GATE THAT REFUSES LEGAL WORK GETS BYPASSED RATHER THAN OBEYED.

    SKILL.md Phase 1: "A down-scaled slice (2 pairs × 2 variations, the
    lofn-daily test-slice precedent) is legal for probe waves." Preflight
    demanded exactly `atlas_probe_slots` and refused the plate doctrine
    explicitly permits (tester, 2026-08-17).
    """

    SLOTS = [
        ["P1", "POSITIVE-CONTROL", "any territory × steady-tempo", "style",
         "expect arrival", "ARRIVES", "control due"],
        ["P2", "NEGATIVE-CONTROL", "any territory × foley", "style",
         "expect absence", "ABSENT", "control due"],
        ["P3", "PROBE", "Jazz × steady-tempo", "style", "does it arrive", "—", "opens cell"],
        ["P4", "NON-OBVIOUS", "Jazz × foley", "style", "a real idea nobody can justify",
         "—", "non-obvious"],
        ["P5", "PROBE", "Jazz × foley", "exclude", "does it leave", "—", "opens channel"],
        ["P6", "PROBE", "Jazz × foley", "emo", "does it survive", "—", "opens channel"],
        ["P7", "PROBE", "Jazz × foley", "section", "does it survive", "—", "opens channel"],
    ]

    def plate(self, n, slots=None):
        rows = list(slots if slots is not None else self.SLOTS[:n])
        exp = self.root / "output" / "expeditions" / "2026-08-20_plate"
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "PLAN.md").write_text(
            "frontier_read: `vault/music_atlas/FRONTIER.md`\n\n"
            + md_table(V.PLAN_COLS, rows), encoding="utf-8")
        (exp / "PREDICTIONS.md").write_text(
            md_table(V.PREDICTION_COLS,
                     [[s[0], s[2], s[3], "ARRIVES", "2026-08-20T09:00:00Z"] for s in rows]),
            encoding="utf-8")
        self.fx.write()
        gates, _ = V.load_gates()
        rep = V.Report()
        V.preflight(exp, gates, rep)
        return rep

    def row(self, rep, fragment):
        return [r for r in rep.rows if fragment in r["check"]][0]

    def test_a_full_six_slot_plate_passes(self):
        rep = self.plate(6)
        self.assertEqual(self.failures(rep), {}, rep.render())

    def test_a_four_slot_slice_passes_with_its_quota_enforced(self):
        rep = self.plate(4)
        self.assertEqual(self.failures(rep), {}, rep.render())
        self.assertIs(self.row(rep, "non-obvious slot present")["ok"], True)

    def test_the_smallest_legal_plate_is_the_two_controls(self):
        rep = self.plate(2)
        self.assertEqual(self.failures(rep), {}, rep.render())
        # ⛔ The quota is WAIVED here, and says so — it is not silently satisfied.
        quota = self.row(rep, "non-obvious slot present")
        self.assertIsNone(quota["ok"])
        self.assertIn("WAIVED", quota["detail"])
        self.assertIn("calibration slice", quota["detail"])

    def test_a_four_slot_slice_without_a_non_obvious_slot_is_refused(self):
        rep = self.plate(4, slots=[s for s in self.SLOTS[:5] if s[1] != "NON-OBVIOUS"][:4])
        self.assertIn("non-obvious slot present (gates atlas_non_obvious_quota)",
                      self.failures(rep))

    def test_a_plate_over_the_ceiling_is_refused(self):
        rep = self.plate(7)
        detail = self.failures(rep).get(
            "slot count within gates atlas_probe_slots_min..atlas_probe_slots")
        self.assertIsNotNone(detail, rep.render())
        self.assertIn("7 slot(s)", detail)

    def test_a_single_slot_plate_is_refused(self):
        self.assertIn("slot count within gates atlas_probe_slots_min..atlas_probe_slots",
                      self.failures(self.plate(1)))

    def test_the_control_minimums_hold_at_every_size(self):
        # ⛔ Down-scaling scales the PROBES, never the instrument: a slice with
        # no negative control measures nothing about the generator.
        rep = self.plate(2, slots=[self.SLOTS[0], self.SLOTS[2]])
        self.assertIn("controls assigned (gates atlas_min_positive_controls / "
                      "atlas_min_negative_controls)", self.failures(rep))

    def test_the_real_gates_file_declares_both_new_keys(self):
        # ⛔ Numbers live in vault/gates.yaml, never in a default. A key that
        # exists only as a DEFAULT is a threshold nobody can change without
        # editing code, which is how a gate stops being a gate.
        real = (Path(__file__).resolve().parent.parent / "vault" / "gates.yaml")             .read_text(encoding="utf-8")
        section = real.split("# --- MUSIC ATLAS (SKADI)")[1]
        self.assertIn("atlas_probe_slots_min: 2", section)
        self.assertIn("atlas_non_obvious_min_slots: 4", section)
        gates, _ = V.load_gates()
        self.assertEqual(gates["atlas_probe_slots_min"], 2)
        self.assertEqual(gates["atlas_non_obvious_min_slots"], 4)


class TestRegisteredUtcIsATimestamp(ValidatorCase):
    """⛔ THE ONE FIELD WHOSE JOB IS TO PROVE AN ORDER accepted free text, so
    "after we heard it" registered a prediction (tester, 2026-08-17)."""

    def plate(self, when, renders=None):
        exp = self.root / "output" / "expeditions" / "2026-08-20_plate"
        exp.mkdir(parents=True, exist_ok=True)
        slots = [
            ["P1", "POSITIVE-CONTROL", "any territory × steady-tempo", "style", "h",
             "ARRIVES", "control due"],
            ["P2", "NEGATIVE-CONTROL", "any territory × foley", "style", "h", "ABSENT",
             "control due"],
        ]
        (exp / "PLAN.md").write_text(
            "frontier_read: `vault/music_atlas/FRONTIER.md`\n\n"
            + md_table(V.PLAN_COLS, slots), encoding="utf-8")
        (exp / "PREDICTIONS.md").write_text(
            md_table(V.PREDICTION_COLS,
                     [[s[0], s[2], s[3], "ARRIVES", when] for s in slots]), encoding="utf-8")
        if renders is not None:
            (exp / "RENDERS.json").write_text(json.dumps(renders), encoding="utf-8")
        self.fx.write()
        return exp

    def preflight_exit(self, when, renders=None):
        self.plate(when, renders)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = V.main(["--preflight", "output/expeditions/2026-08-20_plate"])
        return code, buf.getvalue()

    def test_a_valid_iso_utc_stamp_passes(self):
        code, out = self.preflight_exit("2026-08-20T09:00:00Z")
        self.assertEqual(code, 0, out)

    def test_free_text_is_refused(self):
        code, out = self.preflight_exit("after we heard it")
        self.assertEqual(code, 3, out)
        self.assertIn("registered_utc is strict ISO-8601 UTC", out)

    def test_the_unfilled_template_literal_is_refused(self):
        code, out = self.preflight_exit(V.REGISTER_BEFORE_RENDER)
        self.assertEqual(code, 3, out)
        self.assertIn(V.REGISTER_BEFORE_RENDER, out)

    def test_a_local_timestamp_without_the_z_is_refused(self):
        # "UTC" is the contract; a naive stamp cannot be compared to one.
        self.assertEqual(self.preflight_exit("2026-08-20 09:00:00")[0], 3)
        self.assertEqual(self.preflight_exit("2026-08-20T09:00:00+02:00")[0], 3)

    def test_a_prediction_registered_after_its_render_is_caught(self):
        renders = {"P1": {"clip_id": "c1", "wav": None, "measurements": None,
                          "matched_by": "exact", "model": "v4.5",
                          "written_at_utc": "2026-08-20T08:00:00Z"}}
        code, out = self.preflight_exit("2026-08-20T09:00:00Z", renders)
        self.assertEqual(code, 3, out)
        self.assertIn("no prediction is registered after its render", out)
        self.assertIn("bound at 2026-08-20T08:00:00Z", out)

    def test_the_full_run_catches_it_too(self):
        # Preflight happens once, before the plate opens; the audit runs forever
        # after, and this is exactly the edit that would be made afterwards.
        self.plate("2026-08-20T09:00:00Z",
                   {"P1": {"clip_id": "c1", "written_at_utc": "2026-08-20T08:00:00Z"}})
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_predictions(V.ATLAS_DIR, gates, rep)
        detail = self.failures(rep).get("no prediction is registered after its render")
        self.assertIsNotNone(detail, rep.render())
        self.assertIn("2026-08-20_plate/P1", detail)

    def test_a_render_stamped_after_the_registration_is_the_honest_case(self):
        self.plate("2026-08-20T07:00:00Z",
                   {"P1": {"clip_id": "c1", "written_at_utc": "2026-08-20T08:00:00Z"}})
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_predictions(V.ATLAS_DIR, gates, rep)
        self.assertNotIn("no prediction is registered after its render", self.failures(rep))

    def test_the_parser_itself(self):
        self.assertIsNotNone(V.parse_iso_utc("2026-08-17T21:04:00Z"))
        self.assertIsNotNone(V.parse_iso_utc("2026-08-17T21:04Z"))
        for bad in ("", "—", "2026-08-17", "yesterday", "2026-08-17T21:04:00",
                    "2026-13-01T00:00:00Z", V.REGISTER_BEFORE_RENDER):
            self.assertIsNone(V.parse_iso_utc(bad), bad)


class TestGatesFallbackTypesMatchPyYAML(ValidatorCase):
    """⛔ D10: `[850, 1000]` loaded as `['850', '1000']` under the fallback, so
    the two loaders disagreed about the TYPE of half the file."""

    def test_inline_numeric_lists_are_numbers(self):
        parsed = V._fallback_parse("music_prompt_chars: [850, 1000]\nratio: [0.5, 2]\n")
        self.assertEqual(parsed["music_prompt_chars"], [850, 1000])
        self.assertEqual(parsed["ratio"], [0.5, 2])

    def test_block_lists_are_typed_too(self):
        parsed = V._fallback_parse("bounds:\n  - 12\n  - 3.5\n  - a phrase\n")
        self.assertEqual(parsed["bounds"], [12, 3.5, "a phrase"])

    def test_the_two_loaders_agree_on_the_real_gates_file(self):
        try:
            import yaml  # noqa: PLC0415
        except ImportError:  # pragma: no cover - PyYAML is present in this repo
            self.skipTest("PyYAML absent; the fallback is the only loader here")
        real = Path(__file__).resolve().parent.parent / "vault" / "gates.yaml"
        text = real.read_text(encoding="utf-8")
        loaded = yaml.safe_load(text) or {}
        fallback = V._fallback_parse(text)
        differing = {k: (loaded.get(k), fallback.get(k))
                     for k in set(loaded) | set(fallback) if loaded.get(k) != fallback.get(k)}
        self.assertEqual(differing, {},
                         "a fail-open fallback that changes types is a silent fallback")


if __name__ == "__main__":
    unittest.main()
