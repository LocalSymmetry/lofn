import unittest

from scripts.validate_step09_distinctiveness import body, step09_segment


class Step09BodySelectionTests(unittest.TestCase):
    def test_uses_last_complete_output_in_cumulative_artifact(self):
        artifact = (
            "## 4. Complete Step Output\n"
            "historical package without critique language\n"
            "## 5. Execution Log\nold log\n"
            "## 4. Complete Step Output\n"
            "current Hyper-Skeptic objection and resolution\n"
            "## 5. Validation Result\nnew log\n"
        )

        self.assertEqual(
            body(artifact).strip(),
            "current Hyper-Skeptic objection and resolution",
        )

    def test_uses_last_model_response_when_complete_output_is_absent(self):
        artifact = (
            "## 3. Model Response / Creative Work\nold response\n"
            "## 5. Execution Log\nold log\n"
            "## 3. Model Response / Creative Work\ncurrent critic resolution\n"
            "## 5. Validation Result\nnew log\n"
        )

        self.assertEqual(body(artifact).strip(), "current critic resolution")

    def test_returns_full_text_when_no_delimited_body_exists(self):
        artifact = "Pair-specific critic objection and resolution."

        self.assertEqual(body(artifact), artifact)

    def test_step09_segment_ignores_cumulative_history(self):
        artifact = (
            "# Pair 01 Step 08 Generation\nold output\n"
            "## 4. Complete Step Output\nhistorical package\n"
            "# Pair 01 Step 09 — Artist Refinement\n"
            "## 3. Panel / Critic Deliberation Log\ncurrent objection\n"
            "## 4. Complete Step Output\ncurrent package\n"
        )

        current = step09_segment(artifact)
        self.assertTrue(current.startswith("# Pair 01 Step 09"))
        self.assertNotIn("historical package", current)
        self.assertIn("current objection", current)


if __name__ == "__main__":
    unittest.main()
