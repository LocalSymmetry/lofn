import tempfile
import unittest
from pathlib import Path

from skills.music.scripts.validate_suno_packages import (
    DEFAULT_MAX_SUNG_WORDS,
    DEFAULT_MIN_SUNG_LINES,
    DEFAULT_MIN_SUNG_WORDS,
    load_length_gate,
    DEFAULT_SFX_CUE_MAX_CHARS,
    current_step11_segment,
    load_min_sung_lines,
    load_sfx_cue_max_chars,
    package_blocks,
    strip_md,
    validate,
    validate_package,
)


def package(number: int, *, prompt_chars: int = 900, cue: str | None = "dry click") -> str:
    prompt = "x" * prompt_chars + "."
    exclude = ", ".join(f"failure_{i}" for i in range(45))
    sung = "\n".join(f"this is one sung line {i}" for i in range(70))  # 70 lines x 6 words = 420
    cue_line = f"*{cue}*\n" if cue is not None else ""
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
        f"{cue_line}\n"
        "## 3. TITLE\n\n"
        f"Song {number}\n"
    )


def canonical_step11_package() -> str:
    prompt = "x" * 899 + "."
    exclude = ", ".join(f"failure_{i}" for i in range(45))
    sung = "\n".join(f"this is one sung line {i}" for i in range(70))  # 70 lines x 6 words = 420
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

    def test_revision_synthesis_ids_preserve_all_four_packages(self):
        ids = ["P02-R1", "P02-R2", "P02-S1", "P02-S2"]
        for prefix in ("", "Terminal "):
            with self.subTest(prefix=prefix):
                artifact = "\n".join(package(i).replace(f"## Package {i}", f"# {prefix}{identity}")
                                     for i, identity in enumerate(ids, 1))
                self.assertEqual(len(package_blocks(artifact)), 4)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "four.md"
                    path.write_text(artifact, encoding="utf-8")
                    self.assertEqual(validate(path), [])

    def test_v6_settings_heading_is_not_a_package_heading(self):
        # 2026-11-25: the step-11 contract pins a `## V6 SETTINGS` heading. The v-form of the
        # package heading matched it, so a one-package final split wrongly and raised SPLITTER
        # DEFECT. The model version in a settings heading is not a package number.
        text = canonical_step11_package() + "\n## V6 SETTINGS\n\n```text\nModel v6\n```\n"
        self.assertEqual(len(package_blocks(text)), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.md"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(validate(path), [])
        # A real v-form package heading is still a package heading, with or without the fix.
        four = "\n".join(package(i).replace(f"## Package {i} — Song {i}", f"# V{i} — Song {i}")
                         for i in range(1, 5))
        self.assertEqual(len(package_blocks(four)), 4)
        # And a v-form heading whose TITLE merely begins with the word is not swallowed by the
        # lookahead: only "V<n> SETTINGS" is excluded.
        titled = package(1).replace("## Package 1 — Song 1", "# V1 — Settings for a Small Room")
        self.assertEqual(len(package_blocks(titled + package(2))), 2)

    def test_fenced_field_excludes_adjacent_wrap_heading(self):
        self.assertEqual(strip_md("```text\nactual field\n```\n\n## Production notes\nnot pasted"), "actual field")

    def test_fence_markers_do_not_count_as_sung_lines(self):
        text = package(1).replace("[Theme: test pressure]", "```text\n[Theme: test pressure]")
        text = text.replace("*dry click*\n\n", "*dry click*\n```\n\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fenced.md"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(validate(path, min_sung_lines=70, max_sung_lines=70), [])
            self.assertTrue(any("only 70 probable sung lines" in e for e in validate(path, min_sung_lines=71)))

    def test_artist_sweep_uses_word_boundaries(self):
        for phrase, should_fail in (("arcade lead", False), ("Arca lead", True)):
            with self.subTest(phrase=phrase), tempfile.TemporaryDirectory() as tmp:
                text = package(1).replace("x" * 900, phrase + " " + "x" * (899 - len(phrase)))
                path = Path(tmp) / "artist_boundary.md"
                path.write_text(text, encoding="utf-8")
                artist_errors = [e for e in validate(path) if "artist names" in e]
                self.assertEqual(bool(artist_errors), should_fail)


class SfxCueCapTests(unittest.TestCase):
    """[2026-09-11] This validator shipped with ZERO SFX coverage while its
    hardcoded 40-char cap failed two finished songs on cues of 43 and 47 chars
    and reported them as MISSING. Both halves are pinned here: the cap's
    boundary, and the message telling absence from excess.
    """

    def _sfx_errors(self, cue, cap=60):
        text = package(1, cue=cue)
        return [e for e in validate_package(text, sfx_cue_max_chars=cap)
                if "SFX cue" in e]

    def test_short_cue_passes(self):
        self.assertEqual(self._sfx_errors("dry electronic click"), [])

    def test_cue_at_exactly_the_cap_passes(self):
        cue = "c" * 60
        self.assertEqual(len(cue), 60)
        self.assertEqual(self._sfx_errors(cue), [])

    def test_cue_one_char_over_the_cap_fails(self):
        errors = self._sfx_errors("c" * 61)
        self.assertEqual(len(errors), 1)
        self.assertIn("too long (61 chars, max 60)", errors[0])

    def test_cue_far_over_the_cap_fails_and_quotes_a_truncated_cue(self):
        cue = "SFX: " + "long " * 30            # 155 chars
        errors = self._sfx_errors(cue)
        self.assertEqual(len(errors), 1)
        self.assertIn(f"too long ({len(cue)} chars, max 60)", errors[0])
        self.assertIn("SFX: long long", errors[0])          # the cue is quoted
        self.assertIn("...", errors[0])                      # and truncated
        self.assertLess(len(errors[0]), 160)

    def test_missing_cue_reports_missing_not_too_long(self):
        errors = self._sfx_errors(None)
        self.assertEqual(errors, ["missing standalone *SFX cue* in lyrics"])
        self.assertNotIn("too long", errors[0])

    def test_over_long_cue_is_never_reported_as_missing(self):
        # The exact defect: present-but-long was reported as absent.
        errors = self._sfx_errors("c" * 99)
        self.assertTrue(errors)
        self.assertNotIn("missing", errors[0])

    def test_real_shipped_cues_at_43_and_47_chars_pass_at_the_gate_value(self):
        # The two metal-mike cues the old hardcoded 40 rejected.
        for cue in ("SFX: dry electronic click meets pick attack",
                    "SFX: throat-grit exhale into dry synthetic snap"):
            with self.subTest(cue=cue, length=len(cue)):
                self.assertIn(len(cue), (43, 47))
                self.assertEqual(self._sfx_errors(cue), [])
                self.assertNotEqual(self._sfx_errors(cue, cap=40), [])   # old cap did reject

    def test_any_one_cue_within_the_cap_satisfies_the_gate(self):
        text = package(1, cue="c" * 99).replace("*" + "c" * 99 + "*", "*" + "c" * 99 + "*\n*short one*")
        self.assertEqual([e for e in validate_package(text, sfx_cue_max_chars=60)
                          if "SFX cue" in e], [])

    def test_cap_is_read_from_gates_yaml_not_hardcoded(self):
        # A reader that always returns its default is indistinguishable from a
        # hardcode. Prove it returns the FILE's value.
        with tempfile.TemporaryDirectory() as tmp:
            gates = Path(tmp) / "gates.yaml"
            gates.write_text("sfx_cue_max_chars: 12\n", encoding="utf-8")
            self.assertEqual(load_sfx_cue_max_chars(gates), 12)

    def test_cap_fails_open_to_60_when_gates_is_missing_or_invalid(self):
        self.assertEqual(DEFAULT_SFX_CUE_MAX_CHARS, 60)
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no_such_gates.yaml"
            self.assertEqual(load_sfx_cue_max_chars(missing), 60)
            for body in ("sfx_cue_max_chars: not_a_number\n",
                         "sfx_cue_max_chars: -5\n",
                         "sfx_cue_max_chars: 0\n",
                         "# no key at all\nmusic_prompt_chars: [850, 1000]\n",
                         "\x00\x01 not yaml at all : : :\n"):
                with self.subTest(body=body[:24]):
                    gates = Path(tmp) / "gates.yaml"
                    gates.write_text(body, encoding="utf-8")
                    self.assertEqual(load_sfx_cue_max_chars(gates), 60)

    def test_live_gates_yaml_declares_the_key(self):
        # CLAUDE.md: "Numeric thresholds live in vault/gates.yaml".
        repo = Path(__file__).resolve().parent.parent
        raw = (repo / "vault" / "gates.yaml").read_text(encoding="utf-8")
        self.assertIn("sfx_cue_max_chars:", raw)
        self.assertEqual(load_sfx_cue_max_chars(repo / "vault" / "gates.yaml"), 60)

    @staticmethod
    def _with_sung(lines: int, words_per_line: int) -> str:
        body = "\n".join(" ".join(["word"] * (words_per_line - 1) + [str(i)]) for i in range(lines))
        return package(1).replace("\n".join(f"this is one sung line {i}" for i in range(70)), body)

    def test_v6_19_length_gate_is_sung_words_from_gates_yaml(self):
        repo = Path(__file__).resolve().parent.parent
        self.assertEqual((DEFAULT_MIN_SUNG_WORDS, DEFAULT_MAX_SUNG_WORDS), (300, 520))
        gate = load_length_gate(repo / "vault" / "gates.yaml")
        self.assertEqual(gate, {"mode": "words", "min_words": 300, "max_words": 520})
        # The line floor still exists for gates files written when lines were the gate.
        self.assertEqual(DEFAULT_MIN_SUNG_LINES, 45)
        self.assertEqual(load_min_sung_lines(repo / "vault" / "gates.yaml"), 45)

    def test_v6_19_line_count_alone_neither_passes_nor_fails_a_package(self):
        # Same 400 sung words in 40 short-count lines and in 100 lines: both pass.
        self.assertEqual(validate_package(self._with_sung(40, 10)), [])
        self.assertEqual(validate_package(self._with_sung(100, 4)), [])
        # 82 lines that hold only 246 words fail; 34 lines that hold 408 pass.
        short = validate_package(self._with_sung(82, 3))
        self.assertTrue(any("only 246 sung words in 82 lines, expected >=300" in e for e in short))
        self.assertEqual(validate_package(self._with_sung(34, 12)), [])

    def test_v6_19_word_ceiling_fails_and_boundaries_pass(self):
        self.assertEqual(validate_package(self._with_sung(60, 5)), [])     # 300, the floor
        self.assertEqual(validate_package(self._with_sung(65, 8)), [])     # 520, the ceiling
        over = validate_package(self._with_sung(66, 8))                    # 528
        self.assertTrue(any("528 sung words in 66 lines, expected <=520" in e for e in over))
        under = validate_package(self._with_sung(59, 5))                   # 295
        self.assertTrue(any("only 295 sung words in 59 lines, expected >=300" in e for e in under))

    def test_v6_19_a_declared_bound_replaces_the_house_gate(self):
        # A run that declares a patter form passes its own ceiling and is not re-failed by the default.
        self.assertEqual(validate_package(self._with_sung(80, 7), min_sung_words=460, max_sung_words=620), [])
        # A declared line floor checks lines only (the pre-11-25 call shape still works).
        self.assertTrue(any("only 44 probable sung lines, expected >=45" in e
                            for e in validate_package(self._with_sung(44, 3), min_sung_lines=45)))

    def test_v6_19_gates_file_written_for_lines_is_read_as_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "RUN_GATES.yaml"
            legacy.write_text("sung_lines: [52, 76]\nsung_lines_hard_floor: 45\n", encoding="utf-8")
            self.assertEqual(load_length_gate(legacy)["mode"], "lines")
            declared = Path(tmp) / "RUN_GATES_NEW.yaml"
            declared.write_text("length_gate: words\nsung_lines: [40, 100]\nsung_words_hard_floor: 460\n"
                                "sung_words_hard_ceiling: 620\n", encoding="utf-8")
            self.assertEqual(load_length_gate(declared), {"mode": "words", "min_words": 460, "max_words": 620})
            missing = Path(tmp) / "no_such.yaml"
            self.assertEqual(load_length_gate(missing), {"mode": "words", "min_words": 300, "max_words": 520})


if __name__ == "__main__":
    unittest.main()
