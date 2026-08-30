#!/usr/bin/env python3
"""Tests for scripts/skadi_import_archive.py — the archive sweep, and the gate it added.

Adversarial by design, in the house style of `test_skadi_validator.py`: each test
fabricates the defect the code exists to refuse and asserts that it is refused.
The ones that matter most here are the two ways an archive sweep lies —
**n inflated by counting one render twice**, and **n inflated by counting only the
renders that agreed** — plus the gate that stops a big archive `n` from buying
HIGH confidence it never earned.

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

import skadi_import_archive as I  # noqa: E402
import validate_music_atlas as V  # noqa: E402


def md_table(cols, rows):
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def meas(**kw):
    """A measurement object with the shape `measure_render.audit()` returns."""
    base = {
        "file": "x.wav", "duration_s": 200.0, "sample_rate": 48000, "peak": 0.9,
        "crest_db": 12.0, "loudness_spread_db": 14.0, "opening_2s_db": -20.0,
        "quiet_gaps": [], "deepest_mid_dip": {"at_s": 100.0, "depth_db": 6.0,
                                              "width_s": 0.4},
        "tempo_candidates_bpm": [120, 60, 90, 150],
        "bands_by_quarter_db": {"20-80Hz": [0.0, 1.0, 2.0, 3.0]},
        "stereo": {"correlation": 0.9, "side_mid": 0.2},
        "width_dynamics": {"width_min": 0.0, "width_max": 1.0, "corr_width_level": 0.0},
        "sustained_tones_hz": [], "quarter_tone_pairs": [],
    }
    base.update(kw)
    return base


# --------------------------------------------------------------- the intent parsers

class TestSpelledNumbers(unittest.TestCase):
    """The house writes tempo both ways. A parser that reads one is half an instrument."""

    def test_reads_the_forms_the_catalogue_actually_uses(self):
        for words, want in [("a hundred and thirty-six ", 136), ("eighty-eight ", 88),
                            ("one hundred and fifty-two ", 152), ("a hundred and ten ", 110),
                            ("ninety ", 90), ("a hundred ", 100)]:
            self.assertEqual(I.spelled_number(words), want, words)

    def test_an_unreadable_phrase_is_no_ask_never_a_guess(self):
        for words in ("roughly ", "some number of ", "", "   ", "quite fast "):
            self.assertIsNone(I.spelled_number(words), words)

    def test_out_of_range_is_refused(self):
        # `five` beats per minute is not a tempo; neither is `a hundred hundred`.
        self.assertIsNone(I.spelled_number("five "))
        self.assertIsNone(I.spelled_number("a hundred hundred "))

    def test_only_the_trailing_run_of_number_words_is_read(self):
        # "...in G major, one hundred and forty BPM" must not absorb the key.
        self.assertEqual(I.spelled_number("in G major, one hundred and forty "), 140)


class TestBpmTargets(unittest.TestCase):
    def test_numerals_and_words_both_land(self):
        self.assertEqual(I.bpm_targets("Jersey club bounce, 140 BPM, G minor"), [140])
        self.assertEqual(I.bpm_targets("trip-hop at eighty-eight BPM in G minor"), [88])
        self.assertEqual(
            I.bpm_targets("UK garage at a hundred and thirty-six BPM in A minor"), [136])

    def test_the_same_value_twice_is_one_target(self):
        self.assertEqual(I.bpm_targets("172 BPM ... still 172 BPM"), [172])

    def test_two_distinct_values_are_two_targets(self):
        # This is what makes a clip tempo-dramaturgy rather than steady-tempo.
        self.assertEqual(I.bpm_targets("opens at 92 BPM and lifts to 140 BPM"), [92, 140])

    def test_no_bpm_is_no_ask(self):
        self.assertEqual(I.bpm_targets("dense trip-hop, warm and slow"), [])

    def test_a_frequency_is_not_a_tempo(self):
        self.assertEqual(I.bpm_targets("a 110 Hz mains hum dead centre"), [])


class TestAskDetectors(unittest.TestCase):
    def test_quiet_open_fires_on_the_real_phrasings(self):
        for text in ["Opens in near-silence with frost-air pads",
                     "begins in silence, then a breath",
                     "the first five seconds are near-silent",
                     "fades in from silence"]:
            self.assertTrue(I.QUIET_OPEN.search(text), text)

    def test_quiet_open_does_not_fire_on_its_opposite(self):
        # ⛔ "full level from bar one" is the NEGATION of this ask. A detector that
        # caught it would score a loud opening as a failed quiet one.
        for text in ["full level from bar one, no intro: chopped amen already running",
                     "full height from bar one — first five seconds: a triplet kick"]:
            self.assertFalse(I.QUIET_OPEN.search(text), text)

    def test_terminal_and_interrupting_silence_are_different_asks(self):
        # The single most useful row in the behaviour ledger is that these two are
        # not one behaviour. A parser that conflated them would erase it.
        ending = "End on sub-bass, tape hiss, and absolute silence."
        middle = "collapses the verse into a single beat of charged silence before the chorus"
        self.assertTrue(I.TERMINAL_SILENCE.search(ending))
        self.assertFalse(I.TERMINAL_SILENCE.search(middle))
        self.assertTrue(I.INTERRUPTING_SILENCE.search(middle))

    def test_widen_and_sub_build_fire_on_the_benchmark_text(self):
        arch = ("chorus widens like a triple-arch panorama over the Swiss Alps; "
                "final chorus returns with more sub and more sky")
        self.assertTrue(I.WIDEN.search(arch))
        self.assertTrue(I.SUB_BUILD.search(arch))

    def test_named_frequency_is_found_in_both_word_orders(self):
        self.assertEqual(I.hz_targets("a 110 Hz mains hum dead centre"), [110])
        self.assertEqual(I.hz_targets("a fluorescent mains hum at 110 Hz"), [110])
        self.assertEqual(I.hz_targets("mains hum sits dead centre at a hundred and ten hertz"),
                         [110])
        self.assertEqual(I.hz_targets("dense trip-hop, warm and slow"), [])


# ---------------------------------------------------------------------- the tests

class TestMeasurementTests(unittest.TestCase):
    def test_steady_tempo_folds_octaves_and_reproduces_the_deep_dive(self):
        # Triple Arch: asked 110, candidates [54, 72, 107, 143]. 107 is within 5;
        # 54 is the half. The seed's "6 of 6 within 5 BPM" is this reading.
        verdict, why = I.test_steady_tempo(
            meas(tempo_candidates_bpm=[54, 72, 107, 143]), [110], 5)
        self.assertEqual(verdict, "ARRIVES")
        self.assertIn("110", why)

    def test_steady_tempo_reports_absent_when_nothing_is_near(self):
        verdict, _ = I.test_steady_tempo(
            meas(tempo_candidates_bpm=[53, 80, 54, 107]), [140], 5)
        self.assertEqual(verdict, "ABSENT")

    def test_a_missing_measurement_is_an_instrument_gap_not_an_absence(self):
        # ⛔ None means "the instrument cannot see this", which is not the same
        # claim as "the generator did not do it".
        verdict, _ = I.test_steady_tempo(meas(tempo_candidates_bpm=[]), [120], 5)
        self.assertIsNone(verdict)
        self.assertIsNone(I.test_widen(meas(width_dynamics={}), 0.1)[0])

    def test_quiet_open_threshold(self):
        self.assertEqual(I.test_quiet_open(meas(opening_2s_db=-46.8), -30)[0], "ARRIVES")
        self.assertEqual(I.test_quiet_open(meas(opening_2s_db=-17.5), -30)[0], "ABSENT")

    def test_terminal_silence_only_counts_a_gap_at_the_end(self):
        end = meas(duration_s=246.8, quiet_gaps=[[0.0, 1.2], [244.75, 2.0]])
        mid = meas(duration_s=246.8, quiet_gaps=[[0.0, 1.2], [100.0, 2.0]])
        self.assertEqual(I.test_terminal_silence(end)[0], "ARRIVES")
        self.assertEqual(I.test_terminal_silence(mid)[0], "SMOOTHED")

    def test_interrupting_silence_only_counts_a_gap_in_the_body(self):
        mid = meas(duration_s=200.0, quiet_gaps=[[100.0, 1.5]])
        edge = meas(duration_s=200.0, quiet_gaps=[[0.0, 1.5], [198.0, 2.0]])
        self.assertEqual(I.test_interrupting_silence(mid)[0], "ARRIVES")
        self.assertEqual(I.test_interrupting_silence(edge)[0], "SMOOTHED")

    def test_sub_build_reads_the_quarters(self):
        up = meas(bands_by_quarter_db={"20-80Hz": [0.0, 8.4, 10.6, 11.4]})
        flat = meas(bands_by_quarter_db={"20-80Hz": [0.0, 0.5, 1.0, 1.2]})
        self.assertEqual(I.test_sub_build(up, 3)[0], "ARRIVES")
        self.assertEqual(I.test_sub_build(flat, 3)[0], "SMOOTHED")

    def test_widen_has_three_outcomes_and_inversion_is_one_of_them(self):
        # INWARD COSMOS is a technique, not a defect — but it is also not an arrival.
        inv = meas(width_dynamics={"corr_width_level": -0.426})
        flat = meas(width_dynamics={"corr_width_level": -0.011})
        up = meas(width_dynamics={"corr_width_level": 0.4})
        self.assertEqual(I.test_widen(inv, 0.10)[0], "INVERTED")
        self.assertEqual(I.test_widen(flat, 0.10)[0], "ABSENT")
        self.assertEqual(I.test_widen(up, 0.10)[0], "ARRIVES")

    def test_a_frequency_below_the_analysis_window_is_a_gap_not_an_absence(self):
        # ⛔ tonal_components() searches 70–6000 Hz. A 38 Hz sub the instrument
        # never looked for must not be reported as a sub that failed to arrive.
        verdict, why = I.test_named_tone(meas(), [38], 50)
        self.assertIsNone(verdict)
        self.assertIn("outside", why)
        self.assertEqual(I.test_named_tone(meas(sustained_tones_hz=[110.4]), [110], 50)[0],
                         "ARRIVES")
        self.assertEqual(I.test_named_tone(meas(sustained_tones_hz=[443.0]), [110], 50)[0],
                         "ABSENT")


# ------------------------------------------------------------- n counts trials

class TestVerdictGrouping(unittest.TestCase):
    def test_a_unanimous_group_keeps_its_verdict_and_counts_everything(self):
        rows = [{"verdict": "ARRIVES"}] * 4
        self.assertEqual(I.verdict_of_group(rows), ("ARRIVES", 4, 4))

    def test_a_split_group_is_PARTIAL_and_keeps_the_denominator(self):
        # ⛔ THE CONFIRMATION COUNTER. Taking the majority verdict with n = the
        # majority count would publish "ARRIVES, n=40" from 41 renders of which one
        # disagreed — the same evidence with the inconvenient part deleted.
        rows = [{"verdict": "ARRIVES"}] * 40 + [{"verdict": "ABSENT"}]
        verdict, k, m = I.verdict_of_group(rows)
        self.assertEqual(verdict, "PARTIAL")
        self.assertEqual((k, m), (40, 41))
        text = I.claim_text("steady-tempo", verdict, k, m, ["v5.5"])
        self.assertIn("40 of 41", text)

    def test_the_claim_text_of_a_unanimous_row_says_all(self):
        self.assertIn("all 3", I.claim_text("sub-build", "ARRIVES", 3, 3, ["v5"]))


class TestAssembly(unittest.TestCase):
    """Epoch discipline: agreeing epochs merge, disagreeing epochs never do."""

    LANES = {"steady-tempo": "NUMERIC", "sub-build": "NUMERIC"}

    @staticmethod
    def obs(cls, epoch, verdict, n, first=0):
        return [{"clip": f"{cls[:2]}{epoch}{i + first:02d}", "epoch": epoch,
                 "spec_class": cls, "verdict": verdict, "measure": "m",
                 "audio": "a.wav", "snapshot": "latest.json", "origin": "master",
                 "title": "t"} for i in range(n)]

    @staticmethod
    def atlas_row(rid, cls, verdict, n, models):
        return {"id": rid, "kind": "CELL", "scope": f"any territory × {cls}",
                "channel": "style", "lane": "NUMERIC", "claim": "a claim",
                "verdict": verdict, "n": str(n), "confidence": "MEDIUM",
                "models": models, "evidence": "`x.json`", "falsifier": "f"}

    def test_agreeing_epochs_merge_into_one_row_listing_both(self):
        obs = self.obs("sub-build", "v5", "ARRIVES", 2) + self.obs("sub-build", "v5.5",
                                                                   "ARRIVES", 3)
        new, inc, drift = I.assemble(obs, self.LANES, [], {})
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["epochs"], ["v5", "v5.5"])
        self.assertEqual(new[0]["m"], 5)
        self.assertEqual(drift, [])

    def test_disagreeing_epochs_split_and_raise_a_drift_candidate(self):
        obs = self.obs("sub-build", "v4.5", "SMOOTHED", 2) + self.obs("sub-build", "v5.5",
                                                                      "ARRIVES", 3)
        new, inc, drift = I.assemble(obs, self.LANES, [], {})
        self.assertEqual(len(new), 2)
        self.assertEqual({tuple(r["epochs"]) for r in new}, {("v4.5",), ("v5.5",)})
        self.assertTrue(any("disagree" in d for d in drift), drift)

    def test_an_agreeing_existing_row_is_incremented_never_reworded(self):
        prior = [self.atlas_row("SK-006", "sub-build", "ARRIVES", 1, "v4.5")]
        new, inc, drift = I.assemble(self.obs("sub-build", "v5.5", "ARRIVES", 3),
                                     self.LANES, prior, {})
        self.assertEqual(new, [])
        self.assertEqual(len(inc), 1)
        self.assertEqual(inc[0]["id"], "SK-006")
        self.assertEqual(inc[0]["add_n"], 3)
        self.assertEqual(inc[0]["epochs"], ["v5.5"])

    def test_a_disagreeing_existing_row_gets_a_new_row_and_a_drift_line_not_an_edit(self):
        prior = [self.atlas_row("SK-001", "steady-tempo", "ARRIVES", 6, "v4.5")]
        new, inc, drift = I.assemble(self.obs("steady-tempo", "v4.5", "ABSENT", 7),
                                     self.LANES, prior, {})
        self.assertEqual(inc, [])
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["verdict"], "ABSENT")
        self.assertTrue(any("SK-001" in d and "NOT edited" in d for d in drift), drift)

    def test_a_mixed_epoch_never_increments_an_existing_row(self):
        # ⛔ An increment cannot reword its claim (the append receipt is a sha of
        # that text), so a PARTIAL result must not be folded into a row that says
        # ARRIVES. It becomes its own row carrying `k of m`.
        prior = [self.atlas_row("SK-001", "steady-tempo", "ARRIVES", 6, "v4.5")]
        obs = self.obs("steady-tempo", "v5.5", "ARRIVES", 40) + \
            self.obs("steady-tempo", "v5.5", "ABSENT", 1, first=90)
        new, inc, drift = I.assemble(obs, self.LANES, prior, {})
        self.assertEqual(inc, [])
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["verdict"], "PARTIAL")
        self.assertEqual((new[0]["k"], new[0]["m"]), (40, 41))

    def test_ids_are_stable_across_runs(self):
        rows = [{"key": "sub-build|style|ARRIVES|v5.5", "spec_class": "sub-build"}]
        I.allocate_ids(rows, [self.atlas_row("SK-013", "x", "ARRIVES", 1, "v4.5")], {})
        first = rows[0]["id"]
        self.assertEqual(first, "SK-014")
        again = [{"key": "sub-build|style|ARRIVES|v5.5", "spec_class": "sub-build"}]
        I.allocate_ids(again, [self.atlas_row("SK-013", "x", "ARRIVES", 1, "v4.5")],
                       {"claims": {"sub-build|style|ARRIVES|v5.5": {"id": first}}})
        self.assertEqual(again[0]["id"], first,
                         "a re-run must reuse the id it minted, or every run renames the map")


# ------------------------------------------------------------ one clip, one observation

class ImportFixture(unittest.TestCase):
    """A miniature repo: the importer's paths are all module-level, so they patch."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        atlas = self.root / "vault" / "music_atlas" / "seeds"
        atlas.mkdir(parents=True)
        (self.root / "output" / "distribution" / "wav_masters").mkdir(parents=True)
        self._patches = [
            mock.patch.object(I, "REPO", self.root),
            mock.patch.object(I, "ATLAS_DIR", self.root / "vault" / "music_atlas"),
            mock.patch.object(I, "MASTERS",
                              self.root / "output" / "distribution" / "wav_masters"),
            mock.patch.object(I, "CACHE", atlas / "MEASUREMENT_CACHE.json"),
            mock.patch.object(I, "EVIDENCE", atlas / "ARCHIVE_EVIDENCE.json"),
            mock.patch.object(I, "AUDITS", self.root / "output" / "render-audits"),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def snap(cid, tags, epoch="v5.5", title="t"):
        return {"id": cid + "-0000-0000-0000-000000000000", "title": title, "tags": tags,
                "major_model_version": epoch, "model_name": "chirp-fenix"}


class TestDeduplication(ImportFixture):
    def test_a_byte_identical_twin_is_one_observation_not_two(self):
        # ⛔ The album pipeline keeps a plain-titled copy beside the clip-id one.
        # Counted twice, that is n inflated with evidence paths that all resolve.
        m = meas()
        cache = {
            "output/distribution/wav_masters/A__aaaaaaaa.wav": {"bytes": 1, "measurements": m},
            "output/distribution/wav_masters/A copy__aaaaaaaa.wav": {"bytes": 1,
                                                                     "measurements": dict(m)},
        }
        snaps = {"aaaaaaaa": self.snap("aaaaaaaa", "140 BPM")}
        clips, extras, _ = I.build_clips(cache, snaps, {"aaaaaaaa": "latest.json"})
        self.assertEqual(len(clips), 1)
        self.assertEqual(len(extras["duplicate_file"]), 1)

    def test_the_clip_id_bearing_twin_is_the_one_that_survives(self):
        # ⛔ `Song.wav` sorts BEFORE `Song__abcd1234.wav`. Read in plain sort order
        # the anonymous copy claims the fingerprint first and the real master is
        # discarded as its duplicate — the dedup rule deleting the evidence.
        m = meas()
        cache = {
            "output/distribution/wav_masters/Song.wav": {"bytes": 1, "measurements": dict(m)},
            "output/distribution/wav_masters/Song__aaaaaaaa.wav": {"bytes": 1,
                                                                   "measurements": m},
        }
        snaps = {"aaaaaaaa": self.snap("aaaaaaaa", "140 BPM")}
        clips, extras, _ = I.build_clips(cache, snaps, {"aaaaaaaa": "latest.json"})
        self.assertEqual(list(clips), ["aaaaaaaa"])
        self.assertIn("Song.wav", extras["duplicate_file"][0])

    def test_a_file_with_no_clip_id_is_refused_never_matched_by_title(self):
        cache = {"output/distribution/wav_masters/The Costume.wav":
                 {"bytes": 1, "measurements": meas()},
                 "output/distribution/wav_masters/Other__bbbbbbbb.wav":
                 {"bytes": 2, "measurements": meas(peak=0.5)}}
        snaps = {"bbbbbbbb": self.snap("bbbbbbbb", "140 BPM"),
                 "99999999": self.snap("99999999", "140 BPM", title="The Costume")}
        clips, extras, _ = I.build_clips(cache, snaps, {"bbbbbbbb": "latest.json"})
        self.assertEqual(list(clips), ["bbbbbbbb"])
        self.assertEqual(len(extras["no_clip_id"]), 1)
        self.assertNotIn("99999999", clips, "a title match is not a binding")

    def test_a_clip_with_no_epoch_cannot_carry_a_claim(self):
        cache = {"output/distribution/wav_masters/S__cccccccc.wav":
                 {"bytes": 1, "measurements": meas()},
                 "output/distribution/wav_masters/T__dddddddd.wav":
                 {"bytes": 2, "measurements": meas(peak=0.5)}}
        snaps = {"cccccccc": self.snap("cccccccc", "140 BPM", epoch=""),
                 "dddddddd": self.snap("dddddddd", "140 BPM")}
        clips, extras, _ = I.build_clips(cache, snaps,
                                         {"cccccccc": "latest.json", "dddddddd": "latest.json"})
        self.assertEqual(list(clips), ["dddddddd"])
        self.assertEqual(extras["no_epoch"], ["cccccccc"])

    def test_an_all_unusable_corpus_stops_the_import(self):
        cache = {"output/distribution/wav_masters/S__cccccccc.wav":
                 {"bytes": 1, "measurements": meas()}}
        snaps = {"cccccccc": self.snap("cccccccc", "140 BPM", epoch="")}
        with self.assertRaises(SystemExit):
            I.build_clips(cache, snaps, {"cccccccc": "latest.json"})

    def test_a_non_suno_render_never_enters_the_atlas(self):
        cache = {"output/distribution/wav_masters/thing__dddddddd.lyria.wav":
                 {"bytes": 1, "measurements": meas()},
                 "output/distribution/wav_masters/ok__eeeeeeee.wav":
                 {"bytes": 2, "measurements": meas(peak=0.5)}}
        snaps = {"dddddddd": self.snap("dddddddd", "140 BPM"),
                 "eeeeeeee": self.snap("eeeeeeee", "140 BPM")}
        clips, extras, _ = I.build_clips(cache, snaps,
                                         {"dddddddd": "latest.json", "eeeeeeee": "latest.json"})
        self.assertEqual(list(clips), ["eeeeeeee"])
        self.assertEqual(len(extras["non_suno"]), 1)


class TestEmptyExtractionIsAnError(ImportFixture):
    def test_no_audio_masters_stops_the_import(self):
        with self.assertRaises(SystemExit):
            I.audio_masters()

    def test_a_corpus_with_no_intents_stops_the_import(self):
        # ⛔ Zero observations and a broken parser look identical in the output.
        clip = I.Clip("ffffffff", self.snap("ffffffff", "warm and slow, no numbers"),
                      "latest.json")
        clip.measurements = meas()
        with self.assertRaises(SystemExit):
            I.observations({"ffffffff": clip}, {})

    def test_an_absent_snapshot_directory_stops_the_import(self):
        with mock.patch.object(I, "SNAPSHOT_DIR", self.root / "nope"):
            with self.assertRaises(SystemExit):
                I.snapshot_index()


class TestObservationsRespectTheAsk(ImportFixture):
    def test_a_track_that_does_not_ask_contributes_nothing(self):
        asks = I.Clip("11111111", self.snap("11111111", "trip-hop at 88 BPM"), "s.json")
        asks.measurements = meas(tempo_candidates_bpm=[88, 44, 176, 120])
        silent = I.Clip("22222222", self.snap("22222222", "warm dense trip-hop"), "s.json")
        silent.measurements = meas()
        obs, counts, gaps = I.observations({"11111111": asks, "22222222": silent}, {})
        self.assertEqual([o["clip"] for o in obs], ["11111111"])
        self.assertEqual(counts["steady-tempo"], 1)

    def test_two_tempo_targets_are_an_ask_the_instrument_cannot_adjudicate(self):
        # It is counted, and it is counted as a GAP — never as a smoothing.
        c = I.Clip("33333333", self.snap("33333333", "opens at 92 BPM, lifts to 140 BPM"),
                   "s.json")
        c.measurements = meas()
        other = I.Clip("44444444", self.snap("44444444", "at 120 BPM"), "s.json")
        other.measurements = meas()
        obs, counts, gaps = I.observations({"33333333": c, "44444444": other}, {})
        self.assertEqual(counts["tempo-dramaturgy"], 1)
        self.assertNotIn("tempo-dramaturgy", {o["spec_class"] for o in obs})
        self.assertTrue(any("tempo-dramaturgy" in g for g in gaps))


# ------------------------------------------------------------------- idempotency

ATLAS_HEAD = """<!-- generated -->

# ATLAS

## BANKED

{banked}

## OBSERVED

{observed}

## CONJECTURED

{conjectured}

## APPEND BLOCKS

### 2026-08-17 skadi-seed additions

{receipts}
"""


class TestAtlasWriting(ImportFixture):
    CLAIM = "A single BPM target ARRIVES."

    def seed_atlas(self):
        banked = [["SK-006", "CELL", "any territory × sub-build", "style", "NUMERIC",
                   self.CLAIM, "ARRIVES", "2", "MEDIUM", "v4.5", "`vault/gates.yaml`",
                   "a falsifier"]]
        (self.root / "vault" / "music_atlas" / "ATLAS.md").write_text(
            ATLAS_HEAD.format(
                banked=md_table(V.ATLAS_COLS, banked),
                observed=md_table(V.ATLAS_COLS, []),
                conjectured=md_table(V.ATLAS_COLS, []),
                receipts=md_table(V.APPEND_COLS,
                                  [["SK-006", f"`{V.sha12(self.CLAIM)}`", "SEED", "n"]])),
            encoding="utf-8")
        (self.root / "vault" / "gates.yaml").write_text("x: 1\n", encoding="utf-8")

    def test_running_twice_reproduces_the_same_atlas_byte_for_byte(self):
        self.seed_atlas()
        new_rows = [{"id": "SK-014", "spec_class": "width-vs-level", "verdict": "INVERTED",
                     "k": 3, "m": 3, "epochs": ["v5.5"], "lane": "NUMERIC",
                     "claim": "it narrows", "n": 3, "confidence": "MEDIUM",
                     "key": "width-vs-level|style|INVERTED|v5.5"}]
        inc = [{"id": "SK-006", "spec_class": "sub-build", "add_n": 5, "epochs": ["v5.5"]}]
        bases = {"SK-006": {"n": "2", "models": "v4.5", "evidence": "`vault/gates.yaml`"}}
        ev = {"SK-014": ["vault/gates.yaml"], "SK-006": ["vault/gates.yaml"]}
        gates = {"atlas_bank_min_n": 2}

        first, _ = I.write_atlas(new_rows, inc, bases, ev, gates)
        (self.root / "vault" / "music_atlas" / "ATLAS.md").write_text(first, encoding="utf-8")
        second, _ = I.write_atlas(new_rows, inc, bases, ev, gates)
        self.assertEqual(first, second, "the import must be a pure function of its inputs")

    def test_an_increment_is_applied_to_the_recorded_base_not_compounded(self):
        # ⛔ Without a recorded base, run two turns n=7 into n=12 and the atlas
        # grows an n that no render supports.
        self.seed_atlas()
        inc = [{"id": "SK-006", "spec_class": "sub-build", "add_n": 5, "epochs": ["v5.5"]}]
        bases = {"SK-006": {"n": "2", "models": "v4.5", "evidence": "`vault/gates.yaml`"}}
        ev = {"SK-006": ["vault/gates.yaml"]}
        first, _ = I.write_atlas([], inc, bases, ev, {"atlas_bank_min_n": 2})
        self.assertIn("| 7 |", first)
        (self.root / "vault" / "music_atlas" / "ATLAS.md").write_text(first, encoding="utf-8")
        second, _ = I.write_atlas([], inc, bases, ev, {"atlas_bank_min_n": 2})
        self.assertIn("| 7 |", second)
        self.assertNotIn("| 12 |", second)

    def test_an_increment_widens_models_and_never_rewords_the_claim(self):
        self.seed_atlas()
        inc = [{"id": "SK-006", "spec_class": "sub-build", "add_n": 5, "epochs": ["v5.5"]}]
        bases = {"SK-006": {"n": "2", "models": "v4.5", "evidence": "`vault/gates.yaml`"}}
        text, receipts = I.write_atlas([], inc, bases, {"SK-006": ["vault/gates.yaml"]},
                                       {"atlas_bank_min_n": 2})
        self.assertIn("v4.5, v5.5", text)
        self.assertIn(self.CLAIM, text)
        # The receipt must still be the sha of the UNCHANGED claim text, or the
        # row breaks its own audit trail.
        self.assertEqual(receipts[0][1], f"`{V.sha12(self.CLAIM)}`")

    def test_the_archive_never_writes_HIGH(self):
        self.seed_atlas()
        rows = [{"id": "SK-014", "spec_class": "sub-build", "verdict": "ARRIVES", "k": 99,
                 "m": 99, "epochs": ["v5.5"], "lane": "NUMERIC", "claim": "c", "n": 99,
                 "confidence": "MEDIUM", "key": "k"}]
        text, _ = I.write_atlas(rows, [], {}, {"SK-014": ["vault/gates.yaml"]},
                                {"atlas_bank_min_n": 2})
        row = next(line for line in text.splitlines() if line.startswith("| SK-014 |"))
        self.assertEqual(V.table_cells(row)[V.ATLAS_COLS.index("confidence")], "MEDIUM")
        self.assertEqual(V.table_cells(row)[V.ATLAS_COLS.index("n")], "99")


class TestFrontierAndNotesAreReplaced(ImportFixture):
    def test_the_drift_note_is_replaced_not_appended_twice(self):
        front = self.root / "vault" / "music_atlas" / "FRONTIER.md"
        front.write_text("# FRONTIER\n\n## CONTROL CHART\n\nx\n\n## DELTA LOG\n\ny\n",
                         encoding="utf-8")
        first = I.write_frontier(["**a** — one"], [])
        front.write_text(first, encoding="utf-8")
        second = I.write_frontier(["**a** — one"], [])
        self.assertEqual(first, second)
        self.assertEqual(second.count(I.DRIFT_START), 1)

    def test_a_frontier_with_no_delta_log_is_refused_not_appended_to(self):
        (self.root / "vault" / "music_atlas" / "FRONTIER.md").write_text(
            "# FRONTIER\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            I.write_frontier(["x"], [])

    def test_the_seed_notes_section_is_replaced_not_duplicated(self):
        notes = self.root / "vault" / "music_atlas" / "seeds" / "SEED_NOTES.md"
        notes.write_text("# SEED NOTES\n\n## 1. A\n\nbody\n", encoding="utf-8")
        with mock.patch.object(I, "NOTES", notes):
            first = I.write_notes(f"## {I.NOTES_SECTION}\n\nhello\n")
            notes.write_text(first, encoding="utf-8")
            second = I.write_notes(f"## {I.NOTES_SECTION}\n\nhello\n")
        self.assertEqual(first, second)
        self.assertEqual(second.count(f"## {I.NOTES_SECTION}"), 1)
        self.assertIn("## 1. A", second)


# ------------------------------------- the gate this import added to the validator

VALIDATOR_ATLAS = """<!-- generated -->

# ATLAS

## BANKED

{banked}

## OBSERVED

{observed}

## CONJECTURED

{conjectured}

## APPEND BLOCKS

### 2026-08-17 seed additions

{receipts}
"""


class TestHighRequiresAPlate(unittest.TestCase):
    """⛔ n IS NOT PROVENANCE. The archive can reach an n a plate never will."""

    CLAIM = "A single BPM target ARRIVES."

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.root / "vault" / "music_atlas").mkdir(parents=True)
        (self.root / "output" / "expeditions" / "2026-08-20_p1").mkdir(parents=True)
        (self.root / "output" / "expeditions" / "2026-08-20_p1" / "RENDERS.json").write_text(
            "{}", encoding="utf-8")
        (self.root / "vault" / "music_atlas" / "seeds").mkdir(parents=True)
        (self.root / "vault" / "music_atlas" / "seeds" / "ARCHIVE_EVIDENCE.json").write_text(
            "{}", encoding="utf-8")
        self._patches = [mock.patch.object(V, "REPO", self.root),
                         mock.patch.object(V, "ATLAS_DIR", self.root / "vault" / "music_atlas")]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def write(self, evidence, confidence="HIGH", n=46):
        row = ["SK-014", "CELL", "any territory × steady-tempo", "style", "NUMERIC",
               self.CLAIM, "ARRIVES", str(n), confidence, "v5.5", evidence, "a falsifier"]
        (self.root / "vault" / "music_atlas" / "ATLAS.md").write_text(
            VALIDATOR_ATLAS.format(
                banked=md_table(V.ATLAS_COLS, [row]),
                observed=md_table(V.ATLAS_COLS, []),
                conjectured=md_table(V.ATLAS_COLS, []),
                receipts=md_table(V.APPEND_COLS,
                                  [["SK-014", f"`{V.sha12(self.CLAIM)}`", "NEW", "n"]])),
            encoding="utf-8")

    def run_check(self, gates=None):
        rep = V.Report()
        V.check_archive_provenance(self.root, gates or dict(V.DEFAULTS), rep)
        return rep

    def name(self):
        return ("HIGH-confidence claims cite an expedition plate "
                "(gates atlas_high_requires_plate)")

    def test_archive_only_evidence_cannot_be_HIGH(self):
        self.write("`vault/music_atlas/seeds/ARCHIVE_EVIDENCE.json`")
        rep = self.run_check()
        row = next(r for r in rep.rows if r["check"] == self.name())
        self.assertIs(row["ok"], False, rep.render())
        self.assertIn("SK-014", row["detail"])
        self.assertIn("n=46", row["detail"])

    def test_a_row_citing_a_real_plate_may_be_HIGH(self):
        self.write("`vault/music_atlas/seeds/ARCHIVE_EVIDENCE.json`, "
                   "`output/expeditions/2026-08-20_p1/RENDERS.json`")
        rep = self.run_check()
        row = next(r for r in rep.rows if r["check"] == self.name())
        self.assertIs(row["ok"], True, rep.render())

    def test_an_expedition_path_that_does_not_exist_does_not_count(self):
        # ⛔ A citation that looks like a plate is not a plate.
        self.write("`output/expeditions/2026-08-20_p1/NOT_THERE.json`")
        row = next(r for r in self.run_check().rows if r["check"] == self.name())
        self.assertIs(row["ok"], False)

    def test_medium_confidence_on_archive_evidence_is_fine(self):
        self.write("`vault/music_atlas/seeds/ARCHIVE_EVIDENCE.json`", confidence="MEDIUM")
        row = next(r for r in self.run_check().rows if r["check"] == self.name())
        self.assertIs(row["ok"], True)
        self.assertIn("0 HIGH row(s)", row["detail"])

    def test_the_gate_can_be_switched_off_and_says_so(self):
        self.write("`vault/music_atlas/seeds/ARCHIVE_EVIDENCE.json`")
        gates = dict(V.DEFAULTS)
        gates["atlas_high_requires_plate"] = False
        row = next(r for r in self.run_check(gates).rows if r["check"] == self.name())
        self.assertIsNone(row["ok"])
        self.assertIn("switched off", row["detail"])

    def test_the_check_is_registered_in_the_run(self):
        # A check that is never called is a comment.
        self.assertIn(V.check_archive_provenance, V.CHECKS)

    def test_the_gate_key_is_in_the_real_gates_file(self):
        text = (Path(__file__).resolve().parent.parent / "vault" / "gates.yaml").read_text(
            encoding="utf-8")
        self.assertIn("atlas_high_requires_plate", text)


if __name__ == "__main__":
    unittest.main()
