import unittest

from scripts.validate_step import _current_step_segment


class CurrentStepSegmentTests(unittest.TestCase):
    def test_selects_latest_requested_stage_from_cumulative_artifact(self):
        artifact = (
            "# Pair 01 Step 08 Generation\n"
            "## 1. MUSIC PROMPT\nhistorical prompt\n"
            "# Pair 01 Step 09 Artist Refinement\nintermediate\n"
            "# Pair 01 Step 10 Revision Synthesis\n"
            "## 1. MUSIC PROMPT\ncurrent finalist prompt\n"
            "## Terminal Step 10 Object\n{}\n"
        )

        segment = _current_step_segment(artifact, "10")
        self.assertTrue(segment.startswith("# Pair 01 Step 10"))
        self.assertIn("current finalist prompt", segment)
        self.assertNotIn("historical prompt", segment)

    def test_h2_terminal_heading_does_not_truncate_stage(self):
        artifact = (
            "# Pair 06 Step 10 — Revision and Synthesis\n"
            "current package\n"
            "## Terminal Step 10 Object\nterminal payload\n"
        )

        self.assertEqual(_current_step_segment(artifact, "10"), artifact)

    def test_falls_back_to_full_text_without_stage_heading(self):
        artifact = "standalone legacy artifact"

        self.assertEqual(_current_step_segment(artifact, "10"), artifact)


if __name__ == "__main__":
    unittest.main()
