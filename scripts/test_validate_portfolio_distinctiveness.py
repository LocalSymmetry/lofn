import unittest

from scripts.validate_portfolio_distinctiveness import (
    first_variant,
    packages,
    prompt_blocks,
    prompt_only,
    stage_segment,
    step10_segment,
    variants,
)


class PortfolioStageSelectionTests(unittest.TestCase):
    def test_step10_segment_ignores_historical_variants(self):
        artifact = (
            "# Pair 01 Step 09 Artist Refinement\n"
            "## Variant Old\nold package\n"
            "# Pair 01 Step 10 Revision Synthesis\n"
            "## Variant New One\nnew package one\n"
            "## Variant New Two\nnew package two\n"
            "## Terminal Step 10 Object\n{}\n"
        )

        current = step10_segment(artifact)
        self.assertTrue(current.startswith("# Pair 01 Step 10"))
        self.assertNotIn("old package", current)
        self.assertIn("new package one", current)

    def test_h2_terminal_does_not_replace_h1_stage_boundary(self):
        artifact = (
            "# Pair 02 Step 10 — Revision and Synthesis\n"
            "## Variant One\ncurrent package\n"
            "## Terminal Step 10 Object\nterminal payload\n"
        )

        self.assertIn("current package", step10_segment(artifact))

    def test_first_variant_runs_within_current_segment(self):
        current = (
            "# Pair 03 Step 10 Revision Synthesis\n"
            "## Variant A\nfirst current package\n"
            "## Variation B\nsecond current package\n"
        )

        selected = first_variant(step10_segment(current))
        self.assertIn("first current package", selected)
        self.assertNotIn("second current package", selected)

    def test_variants_returns_every_current_finalist(self):
        current = (
            "# Pair 03 Step 10 Revision Synthesis\n"
            "## Variant A\nfirst current package\n"
            "## Variation B\nsecond current package\n"
            "## Variant C\nthird current package\n"
            "## Terminal Step 10 Object\n{}\n"
        )

        selected = variants(step10_segment(current))
        self.assertEqual(len(selected), 3)
        self.assertIn("first current package", selected[0])
        self.assertIn("second current package", selected[1])
        self.assertIn("third current package", selected[2])

    def test_variants_recognizes_revised_and_synthesized_prompt_headings(self):
        current = (
            "# Pair 05 Step 10 Revision Synthesis\n"
            "### Revised Prompt 1 — First\nfirst\n"
            "### Revised Prompt 2 — Second\nsecond\n"
            "### Synthesized Prompt 1 — Third\nthird\n"
            "### Synthesized Prompt 2 — Fourth\nfourth\n"
            "## Terminal synchronization payload\n{}\n"
        )

        selected = variants(current)
        self.assertEqual(len(selected), 4)
        self.assertIn("first", selected[0])
        self.assertIn("fourth", selected[3])

    def test_step11_segment_and_packages_ignore_cumulative_step10(self):
        artifact = (
            "# Pair 04 Step 10 Revision Synthesis\n"
            "## Variant Old\nold package\n"
            "# Step 11 — Pair 04 Enhancement\n"
            "## Package 1 — First\nfirst package\n"
            "## Enhanced Package 2 — Second\nsecond package\n"
        )

        current = stage_segment(artifact, 11)
        selected = packages(current)
        self.assertNotIn("old package", current)
        self.assertEqual(len(selected), 2)
        self.assertIn("first package", selected[0])
        self.assertIn("second package", selected[1])

    def test_prompt_only_does_not_include_exclude_field(self):
        block = (
            "## 1. MUSIC PROMPT\npositive style\n"
            "## 1B. SUNO EXCLUDE PROMPT\nnegative tokens\n"
            "## 2. LYRICS\nlyrics\n"
        )

        self.assertEqual(prompt_only(block).strip(), "positive style")

    def test_prompt_blocks_ignore_pair_heading_dialects(self):
        current = (
            "# Pair 06 Step 10 Revision Synthesis\n"
            "## REVISED PROMPT R1 — First\n"
            "## 1. MUSIC PROMPT\nfirst style\n## 2. LYRICS\nfirst lyric\n"
            "# Synthesized Final 1 — Second\n"
            "## 1. MUSIC PROMPT\nsecond style\n## 2. LYRICS\nsecond lyric\n"
        )

        selected = prompt_blocks(current)
        self.assertEqual(len(selected), 2)
        self.assertIn("first style", selected[0])
        self.assertIn("second style", selected[1])

    def test_step11_canonical_headings_extract_prompt_and_lyrics(self):
        current = (
            "# Refined Song\n"
            "## SUNO STYLE PROMPT\ncanonical style\n"
            "## SUNO EXCLUDE PROMPT\nnegative tokens\n"
            "## SUNO ENHANCED LYRICS\ncanonical lyric\n"
            "## 3. TITLE\nRefined Song\n"
        )

        selected = prompt_blocks(current)
        self.assertEqual(len(selected), 1)
        self.assertEqual(prompt_only(selected[0]).strip(), "canonical style")
        from scripts.validate_portfolio_distinctiveness import lyrics_only
        self.assertEqual(lyrics_only(selected[0]).strip(), "canonical lyric")


if __name__ == "__main__":
    unittest.main()
