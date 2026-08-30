#!/usr/bin/env python3
"""Tests for scripts/skadi_ingest.py — the probe-to-clip matcher.

⛔ Never touches the network. `--fetch` is not exercised; its downloader is
`scripts/fetch_suno_wavs.py`, imported rather than re-derived, and covered by its
own idempotence contract.

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

import skadi_ingest as I  # noqa: E402
import validate_music_atlas as V  # noqa: E402

PROMPT_A = ("Hybrid lo-fi neo-soul meets glitchwave at 85 BPM in D minor: tape-warmed Rhodes "
            "chords and vinyl hiss weave with bit-crushed drums and granular vocal ghosts.")
PROMPT_B = ("Desert blues at 96 BPM in F dorian: a single hollow-body guitar, brushed kit, "
            "and a voice close enough to hear the room behind it.")


def package(prompt: str, title: str = "a title") -> str:
    return (f"## 1. MUSIC PROMPT\n{prompt}\n\n"
            f"## 1B. SUNO EXCLUDE PROMPT\nno brass\n\n"
            f"## 2. LYRICS\n[Verse]\nsomething\n\n"
            f"## 3. TITLE\n{title}\n")


def plan_md(ids):
    cols = "| " + " | ".join(V.PLAN_COLS) + " |"
    rule = "| " + " | ".join(["---"] * len(V.PLAN_COLS)) + " |"
    rows = [f"| {pid} | PROBE | Jazz × steady-tempo | style | does it arrive | — | opens cell |"
            for pid in ids]
    return "\n".join([cols, rule] + rows)


class IngestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.exp = self.root / "output" / "expeditions" / "2026-08-20_plate"
        (self.exp / "packages").mkdir(parents=True)
        (self.root / "data" / "snapshots" / "suno").mkdir(parents=True)
        (self.root / "vault").mkdir(parents=True)
        (self.root / "vault" / "gates.yaml").write_text("atlas_probe_slots: 6\n",
                                                        encoding="utf-8")
        for p in (mock.patch.object(I, "REPO", self.root),
                  mock.patch.object(V, "REPO", self.root),
                  mock.patch.object(V, "GATES", self.root / "vault" / "gates.yaml")):
            p.start()
            self.addCleanup(p.stop)

    def write_plan(self, ids):
        (self.exp / "PLAN.md").write_text(plan_md(ids), encoding="utf-8")

    def write_package(self, pid, prompt):
        (self.exp / "packages" / f"{pid}_a-song.md").write_text(package(prompt),
                                                                encoding="utf-8")

    def write_snapshot(self, rows):
        (self.root / "data" / "snapshots" / "suno" / "latest.json").write_text(
            json.dumps({"rows": rows}), encoding="utf-8")

    def run_ingest(self, *extra):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = I.main([str(self.exp), "--snapshot",
                           "data/snapshots/suno/latest.json", *extra])
        renders_path = self.exp / "RENDERS.json"
        renders = json.loads(renders_path.read_text(encoding="utf-8")) \
            if renders_path.exists() else {}
        return code, renders, err.getvalue()


class TestNormalize(unittest.TestCase):
    def test_case_punctuation_and_whitespace_are_flattened(self):
        self.assertEqual(I.normalize("A  B, **c**!"), "a b c")
        self.assertEqual(I.normalize("85 BPM in D minor:"), "85 bpm in d minor")

    def test_empty_input_normalizes_to_empty(self):
        self.assertEqual(I.normalize(""), "")
        self.assertEqual(I.normalize(None), "")


class TestMatcher(IngestCase):
    def test_exact_normalized_equality_matches(self):
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A},
                             {"id": "clip-b", "title": "B", "tags": PROMPT_B}])
        code, renders, _ = self.run_ingest()
        self.assertEqual(code, 0)
        self.assertEqual(renders["P1"]["clip_id"], "clip-a")
        self.assertEqual(renders["P1"]["matched_by"], "exact")

    def test_punctuation_and_case_still_match(self):
        # Normalization is the whole reason equality is usable: the style text
        # comes back from Suno through a web field, not a byte-for-byte channel.
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([{"id": "clip-a", "title": "A",
                              "tags": PROMPT_A.upper().replace(",", " ,  ")}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 0, err)
        self.assertEqual(renders["P1"]["clip_id"], "clip-a")

    def test_ambiguous_match_is_a_hard_error_listing_candidates(self):
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A},
                             {"id": "clip-a2", "title": "A again", "tags": PROMPT_A}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 1, "a wrong match is worse than no match")
        self.assertNotIn("P1", renders)
        self.assertIn("AMBIGUOUS", err)
        self.assertIn("clip-a", err)
        self.assertIn("clip-a2", err)
        self.assertIn("--map P1=", err)

    def test_zero_match_is_a_hard_error_listing_the_near_misses(self):
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([{"id": "clip-z", "title": "Z",
                              "tags": "solo bagpipe drone, no percussion, one chord"}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 1)
        self.assertNotIn("P1", renders)
        self.assertIn("NO match", err)
        self.assertIn("clip-z", err)
        self.assertIn("score=", err)

    def test_near_tie_above_the_floor_still_refuses(self):
        # ⛔ The margin exists so a field of near-identical prompts cannot be
        # resolved by a hundredth of a similarity point.
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        twin = PROMPT_A.replace("85 BPM", "86 BPM") + " Extra clause."
        twin2 = PROMPT_A.replace("85 BPM", "87 BPM") + " Other clause."
        self.write_snapshot([{"id": "c1", "title": "1", "tags": twin},
                             {"id": "c2", "title": "2", "tags": twin2}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 1)
        self.assertNotIn("P1", renders)

    def test_map_override_resolves_an_ambiguity(self):
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A},
                             {"id": "clip-a2", "title": "A again", "tags": PROMPT_A}])
        code, renders, _ = self.run_ingest("--map", "P1=clip-a2")
        self.assertEqual(code, 0)
        self.assertEqual(renders["P1"]["clip_id"], "clip-a2")
        self.assertEqual(renders["P1"]["matched_by"], "map")

    def test_map_to_an_unknown_clip_is_refused(self):
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A}])
        code, renders, err = self.run_ingest("--map", "P1=not-a-clip")
        self.assertEqual(code, 1)
        self.assertNotIn("P1", renders)
        self.assertIn("absent from the snapshot", err)

    def test_a_near_miss_is_a_candidate_and_never_a_binding(self):
        # ⛔ Similarity SCORES; it does not BIND. A prompt that nearly matches is
        # a question for a human, printed with its score, resolved with --map.
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        trimmed = PROMPT_A[:len(PROMPT_A) - 30] + " and a soft outro."
        self.write_snapshot([{"id": "clip-s", "title": "S", "tags": trimmed},
                             {"id": "clip-far", "title": "F", "tags": "polka accordion"}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 1, "a wrong match is worse than no match")
        self.assertNotIn("P1", renders)
        self.assertIn("clip-s", err)
        self.assertIn("score=", err)
        self.assertIn("--map P1=", err)


class TestTheDoubleBindingBug(IngestCase):
    """⛔ THE DEFECT THE PLATE'S OWN DESIGN MANUFACTURES (tester, 2026-08-17).

    SKILL.md Phase 1 prescribes `V1 absent · V2 standard · V3 pushed · V4
    crossed`, and "pushed" is routinely "V2 plus one more clause" — so V2's
    normalized prompt is a strict SUBSTRING of V3's. Under the old
    `t in want or want in t` rule, a V2 that was never rendered bound silently to
    V3's clip and the plate reported two observations of one audio file.
    """

    STANDARD = ("Desert blues at 96 BPM in F dorian: a single hollow-body guitar, brushed "
                "kit, and a voice close enough to hear the room behind it.")
    PUSHED = STANDARD + (" Push the last chorus: double the guitar an octave up and let the "
                         "room tail ring out three full seconds.")

    def test_a_pushed_variation_does_not_bind_the_standard_take(self):
        self.write_plan(["P2", "P3"])
        self.write_package("P2", self.STANDARD)
        self.write_package("P3", self.PUSHED)
        # Only the PUSHED take was rendered. The standard one is still on the
        # menu — the normal state of a system where renders are The Scientist's.
        self.write_snapshot([
            {"id": "clip-P3-pushed", "title": "the pushed take", "tags": self.PUSHED},
            {"id": "clip-unrelated", "title": "unrelated",
             "tags": "gregorian chant latin male choir cathedral reverb slow 60 bpm"}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 1, "the unrendered probe must not resolve")
        self.assertNotIn("P2", renders, "P2 was never rendered; it may not carry a clip")
        self.assertEqual(renders["P3"]["clip_id"], "clip-P3-pushed")
        self.assertIn("CONTAINS the probe's style text", err,
                      "the containment that used to BIND is now printed as a reason to look")

    def test_a_generic_one_word_tag_never_binds(self):
        # A snapshot row whose whole style field is "guitar" is contained in
        # almost every prompt we write. Containment scored it; equality does not.
        self.write_plan(["P1"])
        self.write_package("P1", self.STANDARD)
        self.write_snapshot([
            {"id": "clip-WRONG", "title": "an unrelated old song", "tags": "guitar"},
            {"id": "clip-RIGHT", "title": "the actual render",
             "tags": "Desert blues 96 BPM F dorian hollow body guitar brushed kit"}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 1)
        self.assertNotIn("P1", renders)
        self.assertIn("clip-WRONG", err, "the near-miss is still listed, as a candidate")

    def test_two_probes_cannot_share_one_clip(self):
        # Two variations that ended up with the SAME prompt text, one render.
        # Each matches exactly — and exactly is not enough: the plate would count
        # one audio file twice.
        self.write_plan(["P1", "P2"])
        self.write_package("P1", PROMPT_A)
        self.write_package("P2", PROMPT_A)
        self.write_snapshot([{"id": "clip-one", "title": "one", "tags": PROMPT_A}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 1)
        self.assertEqual(renders, {}, "neither binding is written")
        self.assertIn("COLLISION", err)
        self.assertIn("--map P1=", err)
        self.assertIn("--map P2=", err)

    def test_map_cannot_bind_two_probes_to_one_clip(self):
        # ⛔ The invariant holds over --map too: a hand-written map is exactly as
        # able to double-count a render as a matcher is.
        self.write_plan(["P1", "P2"])
        self.write_package("P1", PROMPT_A)
        self.write_package("P2", PROMPT_B)
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A},
                             {"id": "clip-b", "title": "B", "tags": PROMPT_B}])
        code, renders, err = self.run_ingest("--map", "P1=clip-a", "--map", "P2=clip-a")
        self.assertEqual(code, 1)
        self.assertEqual(renders, {})
        self.assertIn("COLLISION: clip clip-a", err)

    def test_distinct_probes_on_distinct_clips_still_pass(self):
        # The invariant must not refuse the normal case.
        self.write_plan(["P1", "P2"])
        self.write_package("P1", PROMPT_A)
        self.write_package("P2", PROMPT_B)
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A},
                             {"id": "clip-b", "title": "B", "tags": PROMPT_B}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 0, err)
        self.assertEqual({p: r["clip_id"] for p, r in renders.items()},
                         {"P1": "clip-a", "P2": "clip-b"})


class TestBindingRecordsWhatTheBankerNeeds(IngestCase):
    def test_the_model_epoch_and_the_bind_time_are_recorded(self):
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A,
                              "major_model_version": "v4.5"}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 0, err)
        # Without this the banker wrote "(declare the model epoch)" into the
        # claim's `models` column, which passed every non-emptiness gate.
        self.assertEqual(renders["P1"]["model"], "v4.5")
        # And this is what makes "registered before the render" checkable.
        self.assertIsNotNone(V.parse_iso_utc(renders["P1"]["written_at_utc"]))

    def test_a_snapshot_row_with_no_epoch_records_none_rather_than_a_guess(self):
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 0, err)
        self.assertIsNone(renders["P1"]["model"], "no epoch is not an epoch")


class TestOfflineBinding(IngestCase):
    """`--measure` must work on WAVs already on disk, and a re-run must not need
    the network to find a file the manifest already records."""

    def wav_dir(self):
        d = self.exp / "wav"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def plant_wav(self, cid, title="A Title", body=b"RIFF" + b"\0" * 60):
        d = self.wav_dir()
        wav = d / f"{title}__{cid[:8]}.wav"
        wav.write_bytes(body)
        (d / "manifest.json").write_text(json.dumps({
            "source": "https://example.invalid/{clip_id}",
            "snapshot": "data/snapshots/suno/latest.json",
            "files": [{"id": cid, "title": title,
                       "path": wav.relative_to(self.root).as_posix(),
                       "bytes": len(body), "sha256": "0" * 64}]}), encoding="utf-8")
        return wav

    def test_measure_alone_finds_a_wav_the_manifest_records(self):
        # ⛔ D11a: `--measure` used to be unreachable without `--fetch`, because
        # only the fetch step ever set `wav`. Measurement reads a file; if the
        # file is there, the measurement must be possible offline.
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([{"id": "clip-a", "title": "A Title", "tags": PROMPT_A}])
        self.plant_wav("clip-a")
        with mock.patch.object(I, "measure", lambda wav: {"duration_s": 61.0}):
            code, renders, err = self.run_ingest("--measure")
        self.assertEqual(code, 0, err)
        self.assertTrue(renders["P1"]["wav"].endswith("__clip-a.wav"), renders["P1"]["wav"])
        self.assertEqual(renders["P1"]["measurements"], {"duration_s": 61.0})

    def test_a_wav_named_by_the_durable_key_is_found_without_a_manifest(self):
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([{"id": "clip-a", "title": "A Title", "tags": PROMPT_A}])
        d = self.wav_dir()
        (d / "A Title__clip-a.wav").write_bytes(b"RIFF")
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 0, err)
        self.assertTrue(renders["P1"]["wav"].endswith("A Title__clip-a.wav"))

    def test_an_offline_rerun_rebinds_from_the_manifest_with_no_network_call(self):
        # ⛔ D11b: the old path asked the CDN for a HEAD first and `continue`d
        # past the file when the request failed — so a machine holding every byte
        # it needed could not re-bind them. The manifest IS the record.
        import fetch_suno_wavs as F
        wav = self.plant_wav("clip-a")
        rows = [{"id": "clip-a", "title": "A Title"}]

        def no_network(*a, **k):
            raise AssertionError("the manifest already had this file; no request may be made")

        with mock.patch.object(F, "head", no_network), \
                mock.patch.object(F, "download", no_network):
            paths = I.fetch_wavs(["clip-a"], rows, self.wav_dir(),
                                 "data/snapshots/suno/latest.json")
        self.assertEqual(paths, {"clip-a": wav})

    def test_a_manifest_whose_bytes_disagree_still_asks(self):
        # The pre-check is not "trust the manifest"; it is "trust the manifest
        # AND the file". A truncated file must still be re-fetched.
        import fetch_suno_wavs as F
        self.plant_wav("clip-a")
        (self.wav_dir() / "A Title__clip-a.wav").write_bytes(b"RIFF")  # now short
        rows = [{"id": "clip-a", "title": "A Title"}]
        asked = []
        with mock.patch.object(F, "head", lambda cid, **k: asked.append(cid) or (200, 4)), \
                mock.patch.object(F, "download",
                                  lambda cid, dest, exp, **k: {"bytes": 4, "sha256": "x"}):
            I.fetch_wavs(["clip-a"], rows, self.wav_dir(),
                         "data/snapshots/suno/latest.json")
        self.assertEqual(asked, ["clip-a"])


class TestPackageExtraction(IngestCase):
    def test_missing_package_is_an_error(self):
        self.write_plan(["P1"])
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A}])
        code, _, err = self.run_ingest()
        self.assertEqual(code, 1)
        self.assertIn("no package found", err)

    def test_two_packages_for_one_probe_is_an_error(self):
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        (self.exp / "packages" / "P1_another.md").write_text(package(PROMPT_B),
                                                             encoding="utf-8")
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A}])
        code, _, err = self.run_ingest()
        self.assertEqual(code, 1)
        self.assertIn("candidate packages", err)

    def test_empty_music_prompt_is_an_error_not_a_match_of_zero(self):
        self.write_plan(["P1"])
        (self.exp / "packages" / "P1_a.md").write_text(
            "## 1. MUSIC PROMPT\n\n## 2. LYRICS\nx\n\n## 3. TITLE\nt\n", encoding="utf-8")
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A}])
        code, _, err = self.run_ingest()
        self.assertEqual(code, 1)
        self.assertIn("EMPTY MUSIC PROMPT", err)

    def test_probe_id_prefix_does_not_bleed_across_probes(self):
        # P1 must not claim P10's package.
        self.write_plan(["P1", "P10"])
        self.write_package("P1", PROMPT_A)
        self.write_package("P10", PROMPT_B)
        self.write_snapshot([{"id": "clip-a", "title": "A", "tags": PROMPT_A},
                             {"id": "clip-b", "title": "B", "tags": PROMPT_B}])
        code, renders, err = self.run_ingest()
        self.assertEqual(code, 0, err)
        self.assertEqual(renders["P1"]["clip_id"], "clip-a")
        self.assertEqual(renders["P10"]["clip_id"], "clip-b")

    def test_empty_snapshot_is_an_error(self):
        self.write_plan(["P1"])
        self.write_package("P1", PROMPT_A)
        self.write_snapshot([])
        with self.assertRaises(SystemExit) as ctx:
            self.run_ingest()
        self.assertIn("0 rows", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
