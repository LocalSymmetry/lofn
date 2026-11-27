"""Regression coverage for the offline video cost preflight."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT = Path(__file__).with_name("video_cost_preflight.py")

class VideoPreflightTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--model", "minimax-h3", "--duration", "5", *args],
            env={**os.environ, "PYTHONIOENCODING": "cp1252"},
            capture_output=True, text=True, encoding="cp1252", check=False,
        )

    def test_non_utf8_console_prints_estimate_and_permission_notice(self):
        p = self.run_cli()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("WAVE TOTAL", p.stdout)
        self.assertIn("NOT permission to spend", p.stdout)

    def test_ceiling_refuses_render_estimate(self):
        p = self.run_cli("--ceiling", "0")
        self.assertEqual(p.returncode, 2, p.stderr)

    def test_invalid_values_never_get_a_pass(self):
        for args in (("--count", "0"), ("--takes", "0"), ("--count", "-2"),
                     ("--duration", "nan"), ("--ceiling", "inf")):
            with self.subTest(args=args):
                p = self.run_cli(*args)
                self.assertEqual(p.returncode, 3, p.stderr)
                self.assertNotIn("under ceiling", p.stdout)
