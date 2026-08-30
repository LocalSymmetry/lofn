import tempfile
import unittest
from pathlib import Path

from skills.music.scripts.validate_suno_packages import (
    current_step11_segment,
    package_blocks,
    strip_md,
    validate,
)


def package(number: int, *, prompt_chars: int = 900) -> str:
    prompt = "x" * prompt_chars + "."
    exclude = ", ".join(f"failure_{i}" for i in range(45))
    sung = "\n".join(f"sung line {i}" for i in range(70))
    return (
        f"## Package {number} — Song {number}\n\n"
        "## 1. MUSIC PROMPT\n\n"
        f"{prompt}\n\n"
        "## 1B. SUNO EXCLUDE PROMPT\n\n"
        f"{exclude}\n\n"
        "## 2. LYRICS\n\n"
        "[Theme: test pressure]\n"
        "[SONG FORM: test return]\n"
        "[Verse - EMO:Apprehension - Adult mezzo - close]\n"
        f"{sung}\n"
        "*dry click*\n\n"
        "## 3. TITLE\n\n"
        f"Song {number}\n"
    )


def canonical_step11_package() -> str:
    prompt = "x" * 899 + "."
    exclude = ", ".join(f"failure_{i}" for i in range(45))
    sung = "\n".join(f"sung line {i}" for i in range(70))
    return (
        "# Step 11 Refined — Canonical Song\n\n"
        "## SUNO STYLE PROMPT\n\n"
        f"{prompt}\n\n"
        "## SUNO EXCLUDE PROMPT\n\n"
        f"{exclude}\n\n"
        "## SUNO ENHANCED LYRICS\n\n"
        "[Theme: test pressure]\n"
        "[SONG FORM: test return]\n"
        "[Verse – EMO:Apprehension – Adult mezzo, close]\n"
        f"{sung}\n"
        "*dry click*\n\n"
        "## 3. TITLE\n\n"
        "Canonical Song\n"
    )


class SunoPackageValidatorTests(unittest.TestCase):
    def test_cumulative_artifact_validates_all_current_step11_packages(self):
        artifact = (
            "# Pair 01 Step 10 Revision Synthesis\n"
            "## 1. MUSIC PROMPT\nlegacy short prompt\n"
            "# Step 11 — Pair 01 Enhancement\n"
            + "\n".join(package(i) for i in range(1, 5))
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.md"
            path.write_text(artifact, encoding="utf-8")
            self.assertEqual(validate(path), [])
        self.assertEqual(len(package_blocks(artifact)), 4)
        self.assertNotIn("legacy short prompt", current_step11_segment(artifact))

    def test_failure_in_later_package_is_not_silently_skipped(self):
        artifact = (
            "# Step 11 — Pair 02 Enhancement\n"
            + package(1)
            + package(2, prompt_chars=20)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.md"
            path.write_text(artifact, encoding="utf-8")
            errors = validate(path)
        self.assertTrue(any(error.startswith("package 2:") for error in errors))
        self.assertTrue(any("music prompt length" in error for error in errors))

    def test_strip_md_preserves_fenced_payload(self):
        self.assertEqual(strip_md("```text\nkeep me\n```"), "keep me")

    def test_step11_canonical_heading_aliases_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "canonical.md"
            path.write_text(canonical_step11_package(), encoding="utf-8")
            self.assertEqual(validate(path), [])


if __name__ == "__main__":
    unittest.main()
