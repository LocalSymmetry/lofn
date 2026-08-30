#!/usr/bin/env python3
"""Tests for scripts/skadi_bank.py — tier stamping, control validity, banking.

The tier tests are the load-bearing ones: a tier must be a pure function of which
files exist, so each case adds exactly one artifact and asserts exactly one step
up the ladder.

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

import skadi_bank as B  # noqa: E402
import validate_music_atlas as V  # noqa: E402
from test_skadi_validator import Fixture, md_table  # noqa: E402

PACKAGE = ("## 1. MUSIC PROMPT\nA steady 96 BPM desert-blues figure, one guitar.\n\n"
           "## 1B. SUNO EXCLUDE PROMPT\nno brass\n\n"
           "## 2. LYRICS\n[Verse]\nsomething\n\n"
           "## 3. TITLE\nA Title\n")


class BankCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.atlas = self.root / "vault" / "music_atlas"
        self.fx = Fixture(self.root).write()
        self.exp = self.root / "output" / "expeditions" / "2026-08-20_plate"
        (self.exp / "packages").mkdir(parents=True)
        for p in (mock.patch.object(B, "REPO", self.root),
                  mock.patch.object(B, "ATLAS_DIR", self.atlas),
                  mock.patch.object(V, "REPO", self.root),
                  mock.patch.object(V, "ATLAS_DIR", self.atlas),
                  mock.patch.object(V, "GATES", self.root / "vault" / "gates.yaml")):
            p.start()
            self.addCleanup(p.stop)
        self.renders: dict[str, dict] = {}

    # ---------------------------------------------------------------- fixtures
    def slots(self, extra=(), control_cell="steady-tempo"):
        rows = [
            ["P1", "POSITIVE-CONTROL", f"any territory × {control_cell}", "style",
             "expect arrival", "ARRIVES", "control due"],
            ["P2", "NEGATIVE-CONTROL", "any territory × foley", "style",
             "expect absence", "ABSENT", "control due"],
        ]
        rows.extend(extra)
        return rows

    def write_plate(self, slots, verdicts):
        (self.exp / "PLAN.md").write_text(
            "frontier_read: `vault/music_atlas/FRONTIER.md`\n\n"
            + md_table(V.PLAN_COLS, slots), encoding="utf-8")
        (self.exp / "PREDICTIONS.md").write_text(
            md_table(V.PREDICTION_COLS,
                     [[s[0], s[2], s[3], s[5] if s[5] != "—" else "ARRIVES",
                       "2026-08-20T09:00:00Z"] for s in slots]), encoding="utf-8")
        (self.exp / "VERDICTS.md").write_text(
            md_table(B.VERDICT_COLS, verdicts), encoding="utf-8")
        (self.exp / "RENDERS.json").write_text(json.dumps(self.renders), encoding="utf-8")

    def add_package(self, pid):
        (self.exp / "packages" / f"{pid}_song.md").write_text(PACKAGE, encoding="utf-8")

    def add_render(self, pid, measured=False, model="v4.5"):
        wav = self.exp / "wav" / f"{pid}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFF-not-really")
        self.renders[pid] = {
            "clip_id": f"clip-{pid}", "matched_by": "exact",
            "wav": wav.relative_to(self.root).as_posix(),
            # `model` is what skadi_ingest.py copies off the snapshot row. Without
            # it the banker REFUSES the claim rather than writing a placeholder.
            "model": model,
            "written_at_utc": "2026-08-20T10:00:00Z",
            "measurements": {"duration_s": 180.0, "tempo_candidates_bpm": [96]}
            if measured else None,
        }

    def add_listen(self, pid, both=True):
        d = self.exp / "listen"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{pid}.blind.json").write_text("{}", encoding="utf-8")
        if both:
            (d / f"{pid}.intent.json").write_text("{}", encoding="utf-8")

    def bank(self, *extra):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = B.main([str(self.exp), *extra])
        return code, buf.getvalue() + err.getvalue()

    def ledger(self):
        rows, _ = V.parse_table((self.atlas / "EXPEDITION_LEDGER.md")
                                .read_text(encoding="utf-8"), V.LEDGER_COLS)
        return {r["probe_id"]: r for r in rows}

    def atlas_rows(self):
        text = (self.atlas / "ATLAS.md").read_text(encoding="utf-8")
        out = {}
        for tier in ("BANKED", "OBSERVED", "CONJECTURED"):
            rows, _ = V.parse_table(text, V.ATLAS_COLS, section=tier)
            for r in rows:
                r["_tier"] = tier
                out[V.strike(r["id"])[0]] = r
        return out


class TestTierStamping(BankCase):
    """A tier is a pure function of artifact presence. One artifact, one step."""

    def tier_of(self, pid):
        tier, evidence, notes = B.stamp_tier(self.exp, pid, self.renders)
        return tier, evidence, notes

    def test_no_package_is_not_even_specified(self):
        tier, _, notes = self.tier_of("P3")
        self.assertEqual(tier, "")
        self.assertIn("no package on disk", notes[0])

    def test_package_alone_is_specified(self):
        self.add_package("P3")
        tier, evidence, _ = self.tier_of("P3")
        self.assertEqual(tier, "SPECIFIED")
        self.assertTrue(evidence[0].endswith("P3_song.md"))

    def test_package_missing_a_canonical_heading_is_not_specified(self):
        (self.exp / "packages" / "P3_song.md").write_text(
            "## 1. MUSIC PROMPT\nsomething\n", encoding="utf-8")
        tier, _, notes = self.tier_of("P3")
        self.assertEqual(tier, "")
        self.assertIn("canonical heading", notes[0])

    def test_clip_plus_audio_is_rendered(self):
        self.add_package("P3")
        self.add_render("P3")
        tier, _, _ = self.tier_of("P3")
        self.assertEqual(tier, "RENDERED")

    def test_a_clip_id_without_audio_stays_specified(self):
        self.add_package("P3")
        self.renders["P3"] = {"clip_id": "clip-x", "wav": None, "measurements": None}
        tier, _, notes = self.tier_of("P3")
        self.assertEqual(tier, "SPECIFIED")
        self.assertIn("no audio on disk", notes[0])

    def test_measurements_make_it_measured(self):
        self.add_package("P3")
        self.add_render("P3", measured=True)
        tier, _, _ = self.tier_of("P3")
        self.assertEqual(tier, "MEASURED")

    def test_both_blind_turns_make_it_heard(self):
        self.add_package("P3")
        self.add_render("P3", measured=True)
        self.add_listen("P3")
        tier, evidence, _ = self.tier_of("P3")
        self.assertEqual(tier, "HEARD")
        self.assertTrue(any(e.endswith("P3.blind.json") for e in evidence))
        self.assertTrue(any(e.endswith("P3.intent.json") for e in evidence))

    def test_one_blind_turn_alone_is_not_heard(self):
        # ⛔ THE BLIND RULE needs both turns: audio alone, then the reveal.
        self.add_package("P3")
        self.add_render("P3", measured=True)
        self.add_listen("P3", both=False)
        tier, _, notes = self.tier_of("P3")
        self.assertEqual(tier, "MEASURED")
        self.assertTrue(any("BLIND RULE" in n for n in notes))

    def test_listening_without_a_render_cannot_be_heard(self):
        self.add_package("P3")
        self.add_listen("P3")
        tier, _, _ = self.tier_of("P3")
        self.assertEqual(tier, "SPECIFIED")


class TestBanking(BankCase):
    def numeric_probe(self, channel="style"):
        return ["P3", "PROBE", "Jazz × steady-tempo", channel, "does it arrive", "—",
                "opens cell"]

    def listener_probe(self):
        return ["P4", "PROBE", "Jazz × foley", "style", "does it arrive", "—", "opens cell"]

    def controls_ok(self):
        return [["P1", "ARRIVES", "—", "—", "—", "—", "control held"],
                ["P2", "ABSENT", "—", "—", "—", "—", "control held"]]

    def setup_numeric(self, verdict="ARRIVES", channel="style"):
        for pid in ("P1", "P2", "P3"):
            self.add_package(pid)
            self.add_render(pid, measured=True)
        self.write_plate(self.slots([self.numeric_probe(channel)]),
                         self.controls_ok()
                         + [["P3", verdict, "—", "a stray tape hiss under the intro",
                             "a render that measurably drifts from the asked tempo",
                             "—", "measured"]])

    def test_a_new_cell_banks_a_new_claim_at_n_equals_one(self):
        # `steady-tempo` through the `exclude` channel is a cell nothing claims.
        self.setup_numeric(channel="exclude")
        code, out = self.bank()
        self.assertEqual(code, 0, out)
        rows = self.atlas_rows()
        new = [r for rid, r in rows.items() if rid not in ("SK-001", "SK-002")]
        self.assertEqual(len(new), 1, out)
        self.assertEqual(new[0]["id"], "SK-003", "ids continue the existing sequence")
        self.assertEqual(new[0]["n"], "1")
        self.assertEqual(new[0]["confidence"], "LOW")
        self.assertEqual(new[0]["_tier"], "OBSERVED", "a fresh claim is n=1, not banked")
        self.assertIn("NEW-BEHAVIOR",
                      (self.exp / "DELTA_DRAFT.md").read_text(encoding="utf-8"))

    def test_a_new_claim_composes_a_mechanical_sentence_when_none_is_given(self):
        self.setup_numeric(channel="exclude")
        self.bank()
        new = next(r for rid, r in self.atlas_rows().items()
                   if rid not in ("SK-001", "SK-002"))
        self.assertIn("steady-tempo", new["claim"])
        self.assertIn("exclude", new["claim"])
        self.assertIn("ARRIVES", new["claim"])

    def test_the_banker_never_writes_high_confidence(self):
        self.setup_numeric()
        self.bank()
        for rid, row in self.atlas_rows().items():
            self.assertIn(row["confidence"], ("LOW", "MEDIUM"), rid)

    def test_matching_verdict_increments_n_and_promotes_to_banked(self):
        # SK-001 already claims steady-tempo/style ARRIVES at n=6.
        self.setup_numeric(verdict="ARRIVES")
        code, out = self.bank()
        self.assertEqual(code, 0, out)
        row = self.atlas_rows()["SK-001"]
        self.assertEqual(row["n"], "7")
        self.assertEqual(row["_tier"], "BANKED")
        self.assertIn("N-INCREMENT", (self.atlas / "ATLAS.md").read_text(encoding="utf-8"))

    def test_an_n_increment_does_not_break_the_claim_receipt(self):
        self.setup_numeric(verdict="ARRIVES")
        self.bank()
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_append_blocks(self.atlas, gates, rep)
        failed = [r for r in rep.rows if r["ok"] is False]
        self.assertEqual(failed, [], rep.render())

    def test_a_different_verdict_supersedes_rather_than_increments(self):
        self.setup_numeric(verdict="SMOOTHED")
        code, out = self.bank()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.atlas_rows()["SK-001"]["n"], "6", "the old row is untouched")
        self.assertIn("SCOPE-NARROWED",
                      (self.exp / "DELTA_DRAFT.md").read_text(encoding="utf-8"))

    def test_listener_lane_claim_is_refused_without_a_blind_pass(self):
        for pid in ("P1", "P2", "P4"):
            self.add_package(pid)
            self.add_render(pid, measured=True)
        self.write_plate(self.slots([self.listener_probe()]),
                         self.controls_ok()
                         + [["P4", "ABSENT", "—", "—", "a blind pass that names the prop",
                             "—", "note"]])
        code, out = self.bank()
        self.assertEqual(code, 1, out)
        self.assertIn("must cite a hearing", out)
        self.assertEqual(len(self.atlas_rows()), 2, "nothing new may have banked")

    def test_listener_lane_claim_banks_once_it_is_heard(self):
        for pid in ("P1", "P2", "P4"):
            self.add_package(pid)
            self.add_render(pid, measured=True)
        self.add_listen("P4")
        self.write_plate(self.slots([self.listener_probe()]),
                         self.controls_ok()
                         + [["P4", "ABSENT", "—", "—", "a blind pass that names the prop",
                             "—", "note"]])
        code, out = self.bank()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.atlas_rows()["SK-002"]["n"], "5")

    def test_the_ledger_is_append_only(self):
        self.setup_numeric()
        self.assertEqual(self.bank()[0], 0)
        code, out = self.bank()
        self.assertEqual(code, 1)
        self.assertIn("append-only", out)

    def test_harvest_is_mandatory(self):
        self.setup_numeric()
        (self.exp / "VERDICTS.md").write_text(
            md_table(B.VERDICT_COLS,
                     [["P1", "ARRIVES", "—", "", "—", "—", "n"],
                      ["P2", "ABSENT", "—", "—", "—", "—", "n"],
                      ["P3", "ARRIVES", "—", "—", "f", "—", "n"]]), encoding="utf-8")
        code, out = self.bank()
        self.assertEqual(code, 2)
        self.assertIn("harvest is mandatory", out)

    def test_a_verdict_outside_the_gates_vocabulary_is_refused(self):
        self.setup_numeric(verdict="MOSTLY-ARRIVED")
        code, out = self.bank()
        self.assertEqual(code, 2)
        self.assertIn("not in gates atlas_verdicts", out)

    def test_harvest_becomes_a_scruff_row(self):
        self.setup_numeric()
        self.bank()
        rows, _ = V.parse_table((self.atlas / "FRONTIER.md").read_text(encoding="utf-8"),
                                V.SCRUFF_COLS, section="SCRUFF SHELF")
        self.assertTrue(any("stray tape hiss" in r["description"] for r in rows))

    def test_a_named_scruff_id_accumulates_sightings_toward_the_rift(self):
        self.setup_numeric()
        (self.exp / "VERDICTS.md").write_text(
            md_table(B.VERDICT_COLS,
                     [["P1", "ARRIVES", "—", "—", "—", "—", "n"],
                      ["P2", "ABSENT", "—", "—", "—", "—", "n"],
                      ["P3", "ARRIVES", "—", "the same stutter again", "f", "SCR-01", "n"]]),
            encoding="utf-8")
        self.bank()
        rows, _ = V.parse_table((self.atlas / "FRONTIER.md").read_text(encoding="utf-8"),
                                V.SCRUFF_COLS, section="SCRUFF SHELF")
        scr01 = next(r for r in rows if r["scruff_id"] == "SCR-01")
        self.assertIn("P3", scr01["sightings"])
        self.assertIn("P1", scr01["sightings"], "the existing sighting must survive")

    def test_coverage_marks_the_territory_probed(self):
        self.setup_numeric()
        self.bank()
        rows, _ = V.parse_table((self.atlas / "FRONTIER.md").read_text(encoding="utf-8"),
                                V.COVERAGE_COLS, section="COVERAGE")
        jazz = next(r for r in rows if r["territory"] == "Jazz")
        self.assertEqual(jazz["state"], "PROBED")
        self.assertEqual(jazz["probes"], "1")
        self.assertEqual(jazz["last_expedition"], "2026-08-20_plate")

    def test_prediction_score_is_written_and_scored(self):
        self.setup_numeric()
        self.bank()
        text = (self.exp / "PREDICTION_SCORE.md").read_text(encoding="utf-8")
        self.assertIn("hit of", text)
        rows, _ = V.parse_table(text, ["probe_id", "predicted", "actual", "hit"])
        self.assertEqual({r["probe_id"]: r["hit"] for r in rows}["P3"], "HIT")

    def test_delta_draft_names_a_canonical_move(self):
        self.setup_numeric()
        self.bank()
        text = (self.exp / "DELTA_DRAFT.md").read_text(encoding="utf-8")
        gates, _ = V.load_gates()
        self.assertTrue(any(m in text for m in gates["atlas_delta_moves"]), text)
        # The banker drafts; a human appends the row and closes with INDEX.md.
        rows, _ = V.parse_table((self.atlas / "FRONTIER.md").read_text(encoding="utf-8"),
                                V.DELTA_COLS, section="DELTA LOG")
        self.assertEqual(rows, [], "the banker must not write the DELTA LOG row itself")


class TestNoPlaceholderEverBanks(BankCase):
    """⛔ A PLACEHOLDER IS NOT EMPTY, SO EVERY NON-EMPTINESS GATE PASSES IT.

    The banker used to write `(declare the model epoch)` into `models` and
    `(name the observation that would refute this)` into `falsifier`; the row
    then passed `models non-empty`, `falsifier non-empty` and the whole
    validator, while being scoped to no version of the generator and refutable
    by nothing (tester, 2026-08-17).
    """

    def probe(self):
        return ["P3", "PROBE", "Jazz × steady-tempo", "exclude", "does it arrive", "—",
                "opens cell"]

    def verdicts(self, falsifier="a render that measurably drifts from the asked tempo"):
        return [["P1", "ARRIVES", "—", "—", "—", "—", "held"],
                ["P2", "ABSENT", "—", "—", "—", "—", "held"],
                ["P3", "ARRIVES", "—", "—", falsifier, "—", "measured"]]

    def test_a_banked_row_carries_the_real_epoch_and_the_supplied_falsifier(self):
        for pid in ("P1", "P2", "P3"):
            self.add_package(pid)
            self.add_render(pid, measured=True, model="v4.5")
        self.write_plate(self.slots([self.probe()]), self.verdicts())
        code, out = self.bank()
        self.assertEqual(code, 0, out)
        new = next(r for rid, r in self.atlas_rows().items()
                   if rid not in ("SK-001", "SK-002"))
        self.assertEqual(new["models"], "v4.5")
        self.assertIn("measurably drifts", new["falsifier"])
        for ghost in V.PLACEHOLDER_STRINGS:
            self.assertNotIn(ghost, (self.atlas / "ATLAS.md").read_text(encoding="utf-8"))
            self.assertNotIn(ghost, (self.atlas / "FRONTIER.md").read_text(encoding="utf-8"))

    def test_the_validator_agrees_after_the_bank(self):
        # The tester's b_placeholder_models_passes_validator, inverted: the same
        # plate, banked, and the reader has nothing to say about it.
        for pid in ("P1", "P2", "P3"):
            self.add_package(pid)
            self.add_render(pid, measured=True, model="v4.5")
        self.write_plate(self.slots([self.probe()]), self.verdicts())
        self.bank()
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_atlas_schema(self.atlas, gates, rep)
        V.check_control_chart(self.atlas, gates, rep)
        self.assertEqual([r for r in rep.rows if r["ok"] is False], [], rep.render())

    def test_a_claim_with_no_model_epoch_is_refused_rather_than_placeheld(self):
        for pid in ("P1", "P2", "P3"):
            self.add_package(pid)
            self.add_render(pid, measured=True, model=None)
        self.write_plate(self.slots([self.probe()]), self.verdicts())
        code, out = self.bank()
        self.assertEqual(code, 1, out)
        self.assertIn("no model epoch", out)
        self.assertEqual(set(self.atlas_rows()), {"SK-001", "SK-002"},
                         "an incompletable claim banks nothing at all")

    def test_a_claim_with_no_falsifier_is_refused(self):
        for pid in ("P1", "P2", "P3"):
            self.add_package(pid)
            self.add_render(pid, measured=True)
        self.write_plate(self.slots([self.probe()]), self.verdicts(falsifier="—"))
        code, out = self.bank()
        self.assertEqual(code, 1, out)
        self.assertIn("no falsifier", out)
        self.assertEqual(set(self.atlas_rows()), {"SK-001", "SK-002"})

    def test_a_control_with_no_epoch_writes_no_chart_point_and_says_so(self):
        # A chart point with an invented epoch cannot be placed against the
        # rule-offs, so it is not written — and the absence is stated, not
        # silent.
        for pid in ("P1", "P2", "P3"):
            self.add_package(pid)
            self.add_render(pid, measured=True, model=None)
        self.write_plate(self.slots([self.probe()]), self.verdicts())
        code, out = self.bank()
        self.assertIn("no model epoch on its render", out)
        rows, _ = V.parse_table((self.atlas / "FRONTIER.md").read_text(encoding="utf-8"),
                                V.CONTROL_CHART_COLS, section="CONTROL CHART")
        self.assertEqual([r for r in rows if r["expedition"] == "2026-08-20_plate"], [])

    def test_an_increment_widens_the_model_scope(self):
        # SK-001 spans v4.5; a confirming probe from v5 makes the row span both,
        # and never silently replaces the older epoch.
        for pid in ("P1", "P2", "P3"):
            self.add_package(pid)
            self.add_render(pid, measured=True, model="v5")
        self.write_plate(
            self.slots([["P3", "PROBE", "Jazz × steady-tempo", "style", "does it arrive",
                         "—", "opens cell"]]), self.verdicts())
        code, out = self.bank()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.atlas_rows()["SK-001"]["models"], "v4.5, v5")


class TestOneClipOneProbe(BankCase):
    """⛔ THE SECOND LAYER of the double-binding fix (the first is the matcher,
    the third is `check_ledger_schema`). One render counted twice inflates every
    downstream number while every evidence path still resolves."""

    def probe(self, pid="P3", cell="Jazz × steady-tempo", channel="exclude"):
        return [pid, "PROBE", cell, channel, "does it arrive", "—", "opens cell"]

    def ok_verdicts(self, *pids):
        rows = [["P1", "ARRIVES", "—", "—", "—", "—", "held"],
                ["P2", "ABSENT", "—", "—", "—", "—", "held"]]
        rows += [[p, "ARRIVES", "—", "—", "a drifting render", "—", "n"] for p in pids]
        return rows

    def test_one_clip_bound_to_two_probes_in_renders_json_is_refused(self):
        for pid in ("P1", "P2", "P3", "P4"):
            self.add_package(pid)
            self.add_render(pid, measured=True)
        self.renders["P4"]["clip_id"] = self.renders["P3"]["clip_id"]
        self.write_plate(self.slots([self.probe("P3"), self.probe("P4", channel="style")]),
                         self.ok_verdicts("P3", "P4"))
        code, out = self.bank()
        self.assertEqual(code, 2, out)
        self.assertIn("one clip to several probes", out)
        self.assertFalse((self.atlas / "EXPEDITION_LEDGER.md").read_text(encoding="utf-8")
                         .count("P3"), "nothing is written on a refusal")

    def test_a_clip_already_banked_under_another_probe_is_refused(self):
        for pid in ("P1", "P2", "P3"):
            self.add_package(pid)
            self.add_render(pid, measured=True)
        self.write_plate(self.slots([self.probe("P3")]), self.ok_verdicts("P3"))
        self.assertEqual(self.bank()[0], 0)
        # A second plate, a new probe id — and the same audio file.
        led = self.atlas / "EXPEDITION_LEDGER.md"
        self.exp = self.root / "output" / "expeditions" / "2026-08-21_plate"
        (self.exp / "packages").mkdir(parents=True)
        self.renders = {}
        for pid in ("P1", "P2", "P5"):
            self.add_package(pid)
            self.add_render(pid, measured=True)
        self.renders["P5"]["clip_id"] = "clip-P3"
        self.write_plate(self.slots([self.probe("P5")]), self.ok_verdicts("P5"))
        code, out = self.bank()
        self.assertEqual(code, 2, out)
        self.assertIn("already bound to a probe", out)
        self.assertIn("2026-08-20_plate/P3", out)
        self.assertNotIn("2026-08-21_plate", led.read_text(encoding="utf-8"))


class TestVoidPlate(BankCase):
    def setup_void(self):
        for pid in ("P1", "P2", "P3"):
            self.add_package(pid)
            self.add_render(pid, measured=True)
        self.write_plate(
            self.slots([["P3", "PROBE", "Jazz × steady-tempo", "style", "does it arrive",
                         "—", "opens cell"]]),
            # ⛔ The positive control did NOT arrive. The instrument moved.
            [["P1", "SMOOTHED", "—", "the control itself smoothed", "—", "—", "n"],
             ["P2", "ABSENT", "—", "—", "—", "—", "n"],
             ["P3", "ARRIVES", "—", "—", "f", "—", "n"]])

    def test_a_failed_control_voids_every_row_on_the_plate(self):
        self.setup_void()
        code, out = self.bank()
        self.assertEqual(code, 1, out)
        for pid, row in self.ledger().items():
            self.assertEqual(row["verdict"], V.VOID_PLATE, pid)

    def test_a_void_plate_banks_nothing(self):
        self.setup_void()
        self.bank()
        self.assertEqual(set(self.atlas_rows()), {"SK-001", "SK-002"})
        self.assertEqual(self.atlas_rows()["SK-001"]["n"], "6")

    def test_a_void_plate_still_records_its_control_points(self):
        # An out-of-limits point is the most informative thing a plate produces.
        self.setup_void()
        self.bank()
        rows, _ = V.parse_table((self.atlas / "FRONTIER.md").read_text(encoding="utf-8"),
                                V.CONTROL_CHART_COLS, section="CONTROL CHART")
        point = next(r for r in rows if r["expedition"] == "2026-08-20_plate"
                     and r["control"] == "steady-tempo")
        self.assertEqual(point["verdict"], "SMOOTHED")

    def test_a_void_plate_delta_is_instrument_only(self):
        self.setup_void()
        self.bank()
        self.assertIn("INSTRUMENT-ONLY",
                      (self.exp / "DELTA_DRAFT.md").read_text(encoding="utf-8"))

    def test_the_validator_agrees_the_plate_is_void(self):
        self.setup_void()
        self.bank()
        (self.exp / "INDEX.md").write_text("closed", encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_controls(self.atlas, gates, rep)
        failed = {r["check"] for r in rep.rows if r["ok"] is False}
        self.assertNotIn("a failed control voids the whole plate", failed, rep.render())

    def test_the_validator_catches_an_unpropagated_void(self):
        self.setup_void()
        self.bank()
        (self.exp / "INDEX.md").write_text("closed", encoding="utf-8")
        led = self.atlas / "EXPEDITION_LEDGER.md"
        led.write_text(led.read_text(encoding="utf-8")
                       .replace(f"| {V.VOID_PLATE} | MEASURED | — |",
                                "| ARRIVES | MEASURED | — |"), encoding="utf-8")
        gates, _ = V.load_gates()
        rep = V.Report()
        V.check_controls(self.atlas, gates, rep)
        failed = {r["check"] for r in rep.rows if r["ok"] is False}
        self.assertIn("a failed control voids the whole plate", failed, rep.render())


if __name__ == "__main__":
    unittest.main()
