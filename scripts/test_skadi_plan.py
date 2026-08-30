#!/usr/bin/env python3
"""Tests for scripts/skadi_plan.py — the planner does arithmetic and nothing else.

    python3 -m unittest discover -s scripts -p "test_skadi*.py"
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import skadi_plan as P  # noqa: E402
import validate_music_atlas as V  # noqa: E402
from test_skadi_validator import Fixture  # noqa: E402


class PlanCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.atlas = self.root / "vault" / "music_atlas"
        self.fx = Fixture(self.root)
        # Add a second standing control so both kinds exist on the chart.
        self.fx.write()
        chart_path = self.atlas / "FRONTIER.md"
        chart_path.write_text(
            chart_path.read_text(encoding="utf-8").replace(
                "| steady-tempo | baseline | ARRIVES | 6 of 6 | v4.5 |",
                "| steady-tempo | baseline | ARRIVES | 6 of 6 | v4.5 |\n"
                "| foley | baseline | ABSENT | 4 of 4 | v4.5 |"),
            encoding="utf-8")
        for p in (mock.patch.object(P, "REPO", self.root),
                  mock.patch.object(P, "ATLAS_DIR", self.atlas),
                  mock.patch.object(V, "REPO", self.root),
                  mock.patch.object(V, "ATLAS_DIR", self.atlas),
                  mock.patch.object(V, "GATES", self.root / "vault" / "gates.yaml")):
            p.start()
            self.addCleanup(p.stop)
        self.out = self.root / "output" / "expeditions" / "2026-08-20_plate"

    def plan(self, *extra):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = P.main(["--out", str(self.out), *extra])
        return code, buf.getvalue() + err.getvalue()

    def slots(self):
        rows, problems = V.parse_table((self.out / "PLAN.md").read_text(encoding="utf-8"),
                                       V.PLAN_COLS)
        self.assertEqual(problems, [])
        return rows


class TestStandingControls(PlanCase):
    def test_kinds_are_derived_from_the_expected_verdict(self):
        state = P.load_state(V.load_gates()[0])
        controls = P.standing_controls(state["chart"])
        self.assertEqual(controls, [("POSITIVE-CONTROL", "steady-tempo", "ARRIVES"),
                                    ("NEGATIVE-CONTROL", "foley", "ABSENT")])

    def test_the_last_chart_point_is_the_expectation(self):
        path = self.atlas / "FRONTIER.md"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "| foley | baseline | ABSENT | 4 of 4 | v4.5 |",
            "| foley | baseline | ABSENT | 4 of 4 | v4.5 |\n"
            "| foley | 2026-08-19_plate | SMOOTHED | now partial | v5 |"), encoding="utf-8")
        state = P.load_state(V.load_gates()[0])
        controls = dict((c[1], c[2]) for c in P.standing_controls(state["chart"]))
        self.assertEqual(controls["foley"], "SMOOTHED")

    def test_a_chart_with_no_negative_control_refuses_to_plan(self):
        path = self.atlas / "FRONTIER.md"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "| foley | baseline | ABSENT | 4 of 4 | v4.5 |\n", ""), encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            self.plan()
        self.assertIn("A plate with no control proves nothing", str(ctx.exception))


class TestPlanShape(PlanCase):
    def test_slot_count_matches_the_gate(self):
        code, out = self.plan()
        self.assertEqual(code, 0, out)
        gates, _ = V.load_gates()
        self.assertEqual(len(self.slots()), gates["atlas_probe_slots"])

    def test_the_planner_leaves_the_non_obvious_slot_to_a_mind(self):
        self.plan()
        non = [s for s in self.slots() if s["kind"] == "NON-OBVIOUS"]
        self.assertEqual(len(non), 1)
        self.assertEqual(non[0]["hypothesis"], V.NON_OBVIOUS_PLACEHOLDER)
        self.assertEqual(non[0]["cell"], V.NON_OBVIOUS_PLACEHOLDER)

    def test_the_planner_registers_no_predictions(self):
        self.plan()
        rows, _ = V.parse_table((self.out / "PREDICTIONS.md").read_text(encoding="utf-8"),
                                V.PREDICTION_COLS)
        self.assertTrue(rows)
        for r in rows:
            self.assertIn(r["prediction"], V.EMPTY_CELL)
            # ⛔ NOT a dash and NOT a clock: the literal the ISO check refuses.
            # A dash would be "empty", which several readers treat as "not
            # supplied yet"; this says which supply is owed and by whom.
            self.assertEqual(r["registered_utc"], V.REGISTER_BEFORE_RENDER)
            self.assertIsNone(V.parse_iso_utc(r["registered_utc"]))

    def test_the_fresh_plan_is_refused_by_preflight(self):
        # ⛔ The planner's own output must not be able to open a plate.
        self.plan()
        gates, _ = V.load_gates()
        rep = V.Report()
        V.preflight(self.out, gates, rep)
        failed = {r["check"] for r in rep.rows if r["ok"] is False}
        self.assertIn("non-obvious slot was filled by a mind, not by the planner", failed)
        self.assertIn("every slot has a prediction registered before the render", failed)
        self.assertIn("registered_utc is strict ISO-8601 UTC (2026-08-17T21:04:00Z)", failed)

    def test_both_controls_are_assigned_with_expected_verdicts(self):
        self.plan()
        controls = [s for s in self.slots() if "CONTROL" in s["kind"]]
        self.assertEqual(len(controls), 2)
        for c in controls:
            self.assertNotIn(c["expected"], V.EMPTY_CELL)

    def test_every_price_is_arithmetic(self):
        self.plan()
        allowed = ("control due", "opens the", "moves ", "chases ", "the non-obvious slot")
        for s in self.slots():
            self.assertTrue(any(s["price"].startswith(a) or a in s["price"] for a in allowed),
                            f"{s['probe_id']} price is not map arithmetic: {s['price']!r}")

    def test_the_plan_carries_a_frontier_read_receipt(self):
        self.plan()
        self.assertIn("frontier_read: `vault/music_atlas/FRONTIER.md`",
                      (self.out / "PLAN.md").read_text(encoding="utf-8"))

    def test_candidates_come_from_more_than_one_generator(self):
        gates, _ = V.load_gates()
        ranked = P.rank_candidates(P.load_state(gates), gates)
        self.assertTrue(sum(1 for v in ranked.values() if v) >= 2,
                        "round-robin needs at least two live generators to diversify")

    def test_an_existing_plan_is_not_clobbered(self):
        self.plan()
        edited = (self.out / "PLAN.md").read_text(encoding="utf-8").replace(
            V.NON_OBVIOUS_PLACEHOLDER, "put the prop in the LYRIC instead", 1)
        (self.out / "PLAN.md").write_text(edited, encoding="utf-8")
        code, out = self.plan()
        self.assertEqual(code, 1)
        self.assertIn("REFUSED", out)
        self.assertIn("put the prop in the LYRIC instead",
                      (self.out / "PLAN.md").read_text(encoding="utf-8"))
        code, _ = self.plan("--force")
        self.assertEqual(code, 0)
        self.assertIn(V.NON_OBVIOUS_PLACEHOLDER,
                      (self.out / "PLAN.md").read_text(encoding="utf-8"))

    def test_dry_run_writes_nothing(self):
        code, out = self.plan("--dry-run")
        self.assertEqual(code, 0)
        self.assertFalse(self.out.exists())
        self.assertIn("PLAN —", out)

    def test_an_empty_atlas_is_an_error_not_an_empty_plan(self):
        (self.atlas / "ATLAS.md").unlink()
        with self.assertRaises(SystemExit) as ctx:
            self.plan()
        self.assertIn("skadi_seed.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
