import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_blind_music_set import (
    REQUIRED_FIELDS,
    current_step11_segment,
    render_candidate,
    terminal_packages,
)


class BlindMusicSetTests(unittest.TestCase):
    def test_terminal_packages_use_current_step11_only(self):
        packages = []
        for i in range(4):
            package = {key: f"{key}-{i}" for key in REQUIRED_FIELDS}
            packages.append(package)
        artifact = (
            "# Pair 01 Step 10 Revision\n```json\n{\"packages\": []}\n```\n"
            "# Step 11 — Pair 01 Enhancement\n"
            "```json\n"
            + json.dumps({"packages": packages})
            + "\n```\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.md"
            path.write_text(artifact, encoding="utf-8")
            parsed = terminal_packages(path)
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed[3]["title"], "title-3")
        self.assertNotIn("Step 10", current_step11_segment(artifact))

    def test_render_candidate_contains_only_blind_delivery_fields(self):
        package = {key: f"{key}-value" for key in REQUIRED_FIELDS}
        rendered = render_candidate(package)
        self.assertIn("## 1. MUSIC PROMPT", rendered)
        self.assertIn("## 1B. SUNO EXCLUDE PROMPT", rendered)
        self.assertIn("## 2. LYRICS", rendered)
        self.assertIn("## 3. TITLE", rendered)
        self.assertNotIn("source_final-value", rendered)
        self.assertNotIn("major_deviations-value", rendered)

    def test_terminal_packages_accept_unfenced_terminal_object(self):
        packages = [{key: f"{key}-{i}" for key in REQUIRED_FIELDS} for i in range(4)]
        artifact = (
            "# Step 11 — Pair 06 Enhancement\n"
            "## Terminal Synchronized Object\n\n"
            + json.dumps({"enhanced_packages": packages})
            + "\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.md"
            path.write_text(artifact, encoding="utf-8")
            parsed = terminal_packages(path)
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed[0]["title"], "title-0")


if __name__ == "__main__":
    unittest.main()
