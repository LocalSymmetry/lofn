#!/usr/bin/env python3
"""Tests for scripts/skadi_listen.py — the blind pass, offline.

⛔ NOTHING HERE TOUCHES THE NETWORK. `poe_call` is replaced by a stub that fails
the test if it is ever reached, so a regression that starts contacting Poe from a
test run shows up as a failure rather than as a bill.

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

import run_render_audit_listener as L  # noqa: E402
import skadi_listen as SL  # noqa: E402
import validate_music_atlas as V  # noqa: E402

PACKAGE = ("## 1. MUSIC PROMPT\nA steady 96 BPM desert-blues figure.\n\n"
           "## 1B. SUNO EXCLUDE PROMPT\nno brass\n\n"
           "## 2. LYRICS\n[Verse]\nsomething\n\n"
           "## 3. TITLE\nA Title\n")


def never_call(*_args, **_kwargs):
    raise AssertionError("a test reached the network")


class TestQuestionnaire(unittest.TestCase):
    def test_it_is_the_render_audit_eight_plus_two(self):
        numbered = [ln for ln in SL.BLIND_TEMPLATE.splitlines()
                    if ln[:2].strip().rstrip(".").isdigit()]
        self.assertEqual(len(numbered), 10)
        self.assertTrue(numbered[8].startswith("9. Name the genre or scene"))
        self.assertTrue(numbered[9].startswith("10. What single technique"))

    def test_the_original_eight_are_verbatim(self):
        original = [ln for ln in L.BLIND_TEMPLATE.splitlines()
                    if ln[:2].strip().rstrip(".").isdigit()]
        mine = [ln for ln in SL.BLIND_TEMPLATE.splitlines()
                if ln[:2].strip().rstrip(".").isdigit()]
        self.assertEqual(original, mine[:8], "the repaired instrument is reused, not rewritten")

    def test_the_instruments_own_last_instruction_stays_last(self):
        self.assertEqual(SL.BLIND_TEMPLATE.splitlines()[-1],
                         L.BLIND_TEMPLATE.splitlines()[-1])

    def test_the_reveal_template_is_the_render_audits_own(self):
        self.assertIs(SL.REVEAL_TEMPLATE, L.REVEAL_TEMPLATE)
        self.assertIn("INVERTED", SL.REVEAL_TEMPLATE)
        self.assertIn("Do not rewrite or soften", SL.REVEAL_TEMPLATE)


class TestAttachment(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_wav_gets_its_own_media_type(self):
        wav = self.root / "a.wav"
        wav.write_bytes(b"RIFF")
        item = SL.attachment(wav, "track.wav")
        self.assertEqual(item["type"], "file", "Poe strips OpenAI-style input_audio")
        self.assertTrue(item["file"]["file_data"].startswith("data:audio/wav;base64,"))

    def test_mp3_keeps_the_instruments_own_type(self):
        mp3 = self.root / "a.mp3"
        mp3.write_bytes(b"ID3")
        item = SL.attachment(mp3, "track.mp3")
        self.assertTrue(item["file"]["file_data"].startswith("data:audio/mpeg;base64,"))


class TestSectionExtraction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_the_canonical_headings_are_read(self):
        p = self.root / "P1_song.md"
        p.write_text(PACKAGE, encoding="utf-8")
        intent = SL.intent_of(p)
        self.assertEqual(intent["style"], "A steady 96 BPM desert-blues figure.")
        self.assertEqual(intent["exclude"], "no brass")
        self.assertIn("[Verse]", intent["lyrics"])

    def test_a_package_without_an_exclude_field_says_so(self):
        p = self.root / "P1_song.md"
        p.write_text(PACKAGE.replace("## 1B. SUNO EXCLUDE PROMPT\nno brass\n\n", ""),
                     encoding="utf-8")
        self.assertIn("No separate exclude field", SL.intent_of(p)["exclude"])


class TestRefusalAndBlindness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.exp = self.root / "output" / "expeditions" / "2026-08-20_plate"
        (self.exp / "packages").mkdir(parents=True)
        (self.root / "vault").mkdir(parents=True)
        (self.root / "vault" / "gates.yaml").write_text("atlas_probe_slots: 6\n",
                                                        encoding="utf-8")
        (self.exp / "packages" / "P1_a-secret-title.md").write_text(PACKAGE, encoding="utf-8")
        wav = self.exp / "wav" / "P1.wav"
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"RIFF" + b"\0" * 64)
        (self.exp / "RENDERS.json").write_text(json.dumps({
            "P1": {"clip_id": "clip-1", "wav": wav.relative_to(self.root).as_posix(),
                   "measurements": {"duration_s": 180.0}, "matched_by": "map"}}),
            encoding="utf-8")
        for p in (mock.patch.object(SL, "REPO", self.root),
                  mock.patch.object(V, "REPO", self.root),
                  mock.patch.object(V, "GATES", self.root / "vault" / "gates.yaml"),
                  mock.patch.object(L, "poe_call", never_call)):
            p.start()
            self.addCleanup(p.stop)

    def run_listen(self, *extra, env=None):
        buf, err = io.StringIO(), io.StringIO()
        with mock.patch.dict("os.environ", env or {}, clear=False), \
                contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = SL.main([str(self.exp), *extra])
        return code, buf.getvalue() + err.getvalue()

    def test_missing_api_key_refuses_politely_and_contacts_nothing(self):
        with mock.patch.dict("os.environ", {"POE_API_KEY": ""}, clear=False):
            code, out = self.run_listen()
        self.assertEqual(code, 2)
        self.assertIn("REFUSED", out)
        self.assertIn("POE_API_KEY", out)

    def test_dry_run_writes_nothing(self):
        code, out = self.run_listen("--dry-run", env={"POE_API_KEY": ""})
        self.assertEqual(code, 0, out)
        self.assertFalse((self.exp / "listen").exists(),
                         "a dry run must not create the output directory")

    def test_the_blind_turn_carries_no_identifying_text(self):
        code, out = self.run_listen("--dry-run", env={"POE_API_KEY": ""})
        self.assertEqual(code, 0, out)
        blind = out.split("--- BLIND TURN")[1].split("--- INTENT TURN")[0]
        # ⛔ The blind surface includes the attachment filename. None of these
        # may appear before the listener has answered.
        for leak in ("P1", "desert-blues", "A Title", "a-secret-title", "clip-1",
                     "2026-08-20_plate"):
            self.assertNotIn(leak, blind, f"{leak!r} leaked into the blind turn")

    def test_the_attachment_filename_is_anonymous(self):
        wav = self.exp / "wav" / "P1.wav"
        item = SL.attachment(wav, f"track{wav.suffix}")
        self.assertEqual(item["file"]["filename"], "track.wav")

    def test_the_intent_turn_comes_second_and_reveals_the_ask(self):
        code, out = self.run_listen("--dry-run", env={"POE_API_KEY": ""})
        self.assertEqual(code, 0, out)
        self.assertLess(out.index("--- BLIND TURN"), out.index("--- INTENT TURN"))
        self.assertIn("desert-blues", out.split("--- INTENT TURN")[1])


if __name__ == "__main__":
    unittest.main()
