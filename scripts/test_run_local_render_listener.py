#!/usr/bin/env python3
"""Focused no-model tests for run_local_render_listener.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_local_render_listener as listener


class WindowScheduleTest(unittest.TestCase):
    def test_default_windows_span_opening_interiors_and_ending(self) -> None:
        windows = listener.schedule_windows(240.0)

        self.assertEqual([item.start_seconds for item in windows], [0.0, 72.0, 144.0, 216.0])
        self.assertEqual(
            [item.role for item in windows],
            ["opening", "interior_01", "interior_02", "ending"],
        )
        self.assertEqual(windows[-1].start_seconds + windows[-1].duration_seconds, 240.0)
        self.assertEqual(
            [item.blind_dict() for item in windows],
            [
                {"window_id": "window_01", "role": "opening"},
                {"window_id": "window_02", "role": "interior"},
                {"window_id": "window_03", "role": "interior"},
                {"window_id": "window_04", "role": "ending"},
            ],
        )

    def test_short_audio_is_one_full_track_window(self) -> None:
        windows = listener.schedule_windows(11.25, window_seconds=24.0, windows=4)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].role, "full_track")
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[0].duration_seconds, 11.25)
        self.assertEqual(
            windows[0].blind_dict(),
            {"window_id": "window_01", "role": "opening"},
        )
        self.assertEqual(
            windows[0].private_dict(),
            {
                "window_id": "window_01",
                "selection_role": "full_track",
                "source_start_seconds": 0.0,
                "source_end_seconds": 11.25,
                "requested_excerpt_duration_seconds": 11.25,
            },
        )

    def test_nonpositive_schedule_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            listener.schedule_windows(0.0)
        with self.assertRaises(ValueError):
            listener.schedule_windows(60.0, windows=0)


class AnonymousNamingTest(unittest.TestCase):
    def test_anonymous_name_contains_only_neutral_indices(self) -> None:
        secret_source_name = "Artist - Revealing Song Title.mp3"
        name = listener.anonymous_window_filename(12, 4)

        self.assertEqual(name, "audio_012_window_04.wav")
        self.assertRegex(name, r"\Aaudio_\d{3,}_window_\d{2,}\.wav\Z")
        self.assertNotIn(Path(secret_source_name).stem.casefold(), name.casefold())

    def test_processor_path_must_be_in_neutral_temp_directory(self) -> None:
        valid = Path("C:/Temp/lofn_audio_deadbeef/audio_001_window_01.wav")
        listener.validate_anonymous_model_path(valid)

        with self.assertRaises(ValueError):
            listener.validate_anonymous_model_path(
                Path("C:/Music/Revealing Title/Revealing Title.wav")
            )


class PromptQuarantineTest(unittest.TestCase):
    def test_blind_prompt_has_exactly_eight_neutral_questions(self) -> None:
        numbered = re.findall(r"(?m)^(\d+)\. ", listener.BLIND_WINDOW_PROMPT)

        self.assertEqual(numbered, [str(number) for number in range(1, 9)])
        lowered = listener.BLIND_WINDOW_PROMPT.casefold()
        for forbidden in ("filename", "artist", "title", "lyrics", "intent"):
            self.assertNotIn(forbidden, lowered)

    def test_window_conversation_contains_no_source_identity(self) -> None:
        source = Path("C:/Music/Artist - Secret Intent Title.mp3")
        anonymous = Path("C:/Temp/lofn_audio_deadbeef/audio_001_window_01.wav")
        conversation = listener.build_window_conversation(anonymous)
        serialized = json.dumps(conversation, ensure_ascii=False)

        self.assertEqual(len(conversation), 1, "each window must start a fresh conversation")
        self.assertNotIn(str(source), serialized)
        self.assertNotIn(source.name, serialized)
        self.assertEqual(conversation[0]["content"][0]["audio"], str(anonymous))
        self.assertEqual(
            conversation[0]["content"][0],
            {"type": "audio", "audio": str(anonymous)},
        )
        listener.assert_source_quarantined(conversation, source)

    def test_source_path_is_rejected_if_injected(self) -> None:
        source = Path("C:/Music/Artist - Secret Intent Title.mp3")
        conversation = [{"role": "user", "content": str(source)}]

        with self.assertRaises(ValueError):
            listener.assert_source_quarantined(conversation, source)

    def test_source_title_ancestor_is_rejected_before_inference(self) -> None:
        source = Path("C:/Music/Secret Intent Title.mp3")
        anonymous = Path(
            "C:/Secret Intent Title/lofn_audio_deadbeef/audio_001_window_01.wav"
        )
        conversation = listener.build_window_conversation(anonymous)

        with self.assertRaises(ValueError):
            listener.assert_source_quarantined(conversation, source)


class SourceMapSeparationTest(unittest.TestCase):
    def test_identity_and_blind_records_are_physically_separate(self) -> None:
        source = listener.PreparedSource(
            source_path=Path("C:/Music/Artist - Secret Intent Title.mp3"),
            sha256="a" * 64,
            duration_seconds=9876.54321,
            song_number=1,
            song_id="song_001",
            private_window_schedule=[
                {
                    "window_id": "window_04",
                    "selection_role": "ending",
                    "source_start_seconds": 9851.86431,
                    "source_end_seconds": 9876.54321,
                    "requested_excerpt_duration_seconds": 24.6789,
                }
            ],
        )
        identity = listener.build_identity_payload(
            [source], requested_window_seconds=24.6789, requested_windows=4
        )
        blind = listener.build_blind_report_payload()
        blind["songs"].append(
            {
                "song_id": source.song_id,
                "status": "complete",
                "windows": [
                    {
                        "window_id": "window_04",
                        "role": "ending",
                        "status": "ok",
                        "raw_report": "The excerpt closes on a held chord.",
                    }
                ],
                "errors": [],
            }
        )

        identity_json = json.dumps(identity, ensure_ascii=False)
        blind_json = json.dumps(blind, ensure_ascii=False)
        self.assertIn(source.source_path.name, identity_json)
        self.assertIn(source.sha256, identity_json)
        for private_value in ("9876.54321", "9851.86431", "24.6789"):
            self.assertIn(private_value, identity_json)
            self.assertNotIn(private_value, blind_json)
        self.assertNotIn(source.source_path.name, blind_json)
        self.assertNotIn(source.sha256, blind_json)
        for forbidden_key in (
            "duration_seconds",
            "source_start_seconds",
            "source_end_seconds",
            "source_offset_seconds",
            "requested_excerpt_duration_seconds",
            "requested_window_duration_seconds",
            "window_seconds",
        ):
            self.assertNotIn(f'"{forbidden_key}"', blind_json)
        self.assertEqual(
            blind["numeric_and_stereo_analysis"],
            "separate; not inferred by this mono listener",
        )

    def test_synthesis_prompt_uses_successful_window_reports_only(self) -> None:
        song = {
            "song_id": "song_001",
            "windows": [
                {
                    "window_id": "window_01",
                    "role": "opening",
                    "source_offset_seconds": 1234.56789,
                    "excerpt_duration_seconds": 98.76543,
                    "source_duration_seconds": 9876.54321,
                    "source_path": "C:/Music/Artist - Private Title.mp3",
                    "status": "ok",
                    "raw_report": "A dry alto enters over a sparse pulse.",
                },
                {
                    "window_id": "window_02",
                    "role": "ending",
                    "source_offset_seconds": 9753.08642,
                    "excerpt_duration_seconds": 98.76543,
                    "status": "error",
                    "error": {"message": "private failure"},
                },
            ],
        }
        conversation = listener.build_synthesis_conversation(song)
        synthesis_prompt = conversation[0]["content"][0]["text"]
        serialized = json.dumps(conversation, ensure_ascii=False)

        self.assertIn("A dry alto enters", serialized)
        self.assertNotIn("private failure", serialized)
        self.assertNotIn("song_001", serialized)
        self.assertNotIn("Private Title", serialized)
        for private_value in ("1234.56789", "98.76543", "9876.54321"):
            self.assertNotIn(private_value, serialized)
        for forbidden_key in (
            "source_offset_seconds",
            "excerpt_duration_seconds",
            "source_duration_seconds",
            "source_path",
        ):
            self.assertNotIn(forbidden_key, serialized)
        self.assertIn('"window_id": "window_01"', synthesis_prompt)
        self.assertIn('"role": "opening"', synthesis_prompt)

    def test_synthesis_projects_first_answer_but_raw_text_stays_unmodified(self) -> None:
        raw = (
            "1. Dry vocal over a clipped pulse.\n"
            "2. The phrase rises once, then settles.\n"
            "Human: Please describe an unrelated image.\n"
            "A swimmer in blue water."
        )
        song = {
            "windows": [
                {
                    "window_id": "window_01",
                    "role": "opening",
                    "status": "ok",
                    "raw_report": raw,
                }
            ]
        }

        conversation = listener.build_synthesis_conversation(song)
        serialized = json.dumps(conversation, ensure_ascii=False)

        self.assertEqual(song["windows"][0]["raw_report"], raw)
        self.assertIn("The phrase rises once, then settles.", serialized)
        self.assertNotIn("swimmer", serialized)
        self.assertNotIn("Human:", serialized)

    def test_first_response_projection_rejects_continuation_only_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "no usable first answer"):
            listener.project_first_response("Human: invented next turn")

    def test_numbered_projection_removes_instruction_echo(self) -> None:
        raw = (
            "1. Describe structure.\n2. Describe voice.\n3. Describe instruments.\n"
            "4. Describe events.\n5. Describe uncertainty.\n6. Summarize.\n\n"
            "1. The opening is spare.\n2. A dry alto sings.\n"
            "3. Muted drums support organ.\n4. A rest interrupts the hook.\n"
            "5. The final word is uncertain.\n6. The sound communicates resolve.\n"
            "Human: unrelated continuation"
        )

        projected = listener.project_numbered_response(raw, 6)

        self.assertTrue(projected.startswith("1. The opening is spare."))
        self.assertNotIn("Describe structure", projected)
        self.assertNotIn("Human:", projected)

    def test_synthesis_rejects_nonsemantic_private_role(self) -> None:
        song = {
            "windows": [
                {
                    "window_id": "window_01",
                    "role": "full_track",
                    "status": "ok",
                    "raw_report": "Anonymous blind prose.",
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "semantic role"):
            listener.build_synthesis_conversation(song)

    def test_preparation_is_deterministic_and_preserves_missing_file_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_path = root / "Z Revealing Name.mp3"
            second_path = root / "A Other Name.mp3"
            missing_path = root / "Missing Name.mp3"
            first_path.write_bytes(b"first audio bytes")
            second_path.write_bytes(b"second audio bytes")

            with mock.patch.object(listener, "probe_duration", return_value=10.0):
                first_run = listener.prepare_sources(
                    [first_path, missing_path, second_path]
                )
                second_run = listener.prepare_sources(
                    [second_path, first_path, missing_path]
                )

        first_mapping = {item.sha256: item.song_id for item in first_run if item.ready}
        second_mapping = {item.sha256: item.song_id for item in second_run if item.ready}
        self.assertEqual(first_mapping, second_mapping)
        self.assertEqual(sum(item.ready for item in first_run), 2)
        missing = next(item for item in first_run if item.source_path.name == missing_path.name)
        self.assertEqual(missing.song_id, "song_003")
        self.assertEqual(missing.errors[0]["stage"], "input")

    def test_main_persists_raw_report_before_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "listener-output"
            source_path = root / "Revealing Source Name.mp3"
            prepared = listener.PreparedSource(
                source_path=source_path,
                sha256="b" * 64,
                duration_seconds=12.0,
                song_number=1,
                song_id="song_001",
            )

            def fake_generate(_listener, conversation, max_new_tokens):
                del _listener, max_new_tokens
                first_content = conversation[0]["content"][0]
                if first_content["type"] == "audio":
                    return "Unmodified blind window report."
                raw_path = output_dir / listener.RAW_REPORT_RELATIVE_PATH
                self.assertTrue(raw_path.is_file())
                raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
                window = raw_payload["songs"][0]["windows"][0]
                self.assertEqual(window["status"], "ok")
                self.assertEqual(window["raw_report"], "Unmodified blind window report.")
                return "Blind synthesis from that report."

            with (
                mock.patch.object(listener, "prepare_sources", return_value=[prepared]),
                mock.patch.object(listener, "load_local_listener", return_value=object()),
                mock.patch.object(listener, "extract_window", return_value=None),
                mock.patch.object(listener, "generate_greedy", side_effect=fake_generate),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = listener.main(
                    [
                        str(source_path),
                        "--model-dir",
                        str(root / "local-model"),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            private = json.loads(
                (output_dir / listener.PRIVATE_RELATIVE_PATH).read_text(
                    encoding="utf-8"
                )
            )
            raw = json.loads(
                (output_dir / listener.RAW_REPORT_RELATIVE_PATH).read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(
                (output_dir / listener.SYNTHESIS_RELATIVE_PATH).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                summary["songs"][0]["raw_synthesis"],
                "Blind synthesis from that report.",
            )
            self.assertEqual(
                summary["songs"][0]["usable_synthesis"],
                "Blind synthesis from that report.",
            )
            self.assertEqual(private["songs"][0]["duration_seconds"], 12.0)
            self.assertEqual(
                private["window_request"]["requested_window_duration_seconds"],
                24.0,
            )
            self.assertEqual(
                private["songs"][0]["window_schedule"][0],
                {
                    "window_id": "window_01",
                    "selection_role": "full_track",
                    "source_start_seconds": 0.0,
                    "source_end_seconds": 12.0,
                    "requested_excerpt_duration_seconds": 12.0,
                },
            )
            self.assertEqual(
                set(raw["songs"][0]["windows"][0]),
                {"window_id", "role", "status", "raw_report"},
            )
            self.assertEqual(raw["songs"][0]["windows"][0]["role"], "opening")
            raw_json = json.dumps(raw, ensure_ascii=False)
            summary_json = json.dumps(summary, ensure_ascii=False)
            for public_json in (raw_json, summary_json):
                self.assertNotIn("Revealing Source Name", public_json)
                self.assertNotIn("12.0", public_json)
                self.assertNotIn("24.0", public_json)
                for forbidden_key in (
                    "duration_seconds",
                    "source_start_seconds",
                    "source_end_seconds",
                    "source_offset_seconds",
                    "requested_excerpt_duration_seconds",
                    "requested_window_duration_seconds",
                    "window_seconds",
                ):
                    self.assertNotIn(f'"{forbidden_key}"', public_json)

    def test_main_preserves_window_error_and_continues_to_next_song(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "listener-output"
            sources = [
                listener.PreparedSource(
                    source_path=root / "Broken Render.mp3",
                    sha256="1" * 64,
                    duration_seconds=12.0,
                    song_number=1,
                    song_id="song_001",
                ),
                listener.PreparedSource(
                    source_path=root / "Good Render.mp3",
                    sha256="2" * 64,
                    duration_seconds=12.0,
                    song_number=2,
                    song_id="song_002",
                ),
            ]

            def fake_extract(source_path, destination_path, window):
                del destination_path, window
                if source_path.name == "Broken Render.mp3":
                    raise RuntimeError(f"decode failed for {source_path}")

            def fake_generate(_listener, conversation, max_new_tokens):
                del _listener, max_new_tokens
                if conversation[0]["content"][0]["type"] == "audio":
                    return "Blind report for the second song."
                return "Blind synthesis for the second song."

            with (
                mock.patch.object(listener, "prepare_sources", return_value=sources),
                mock.patch.object(listener, "load_local_listener", return_value=object()),
                mock.patch.object(listener, "extract_window", side_effect=fake_extract),
                mock.patch.object(listener, "generate_greedy", side_effect=fake_generate),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = listener.main(
                    [
                        str(sources[0].source_path),
                        str(sources[1].source_path),
                        "--model-dir",
                        str(root / "local-model"),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 1)
            raw = json.loads(
                (output_dir / listener.RAW_REPORT_RELATIVE_PATH).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [song["status"] for song in raw["songs"]], ["failed", "complete"]
            )
            self.assertEqual(raw["songs"][0]["windows"][0]["status"], "error")
            self.assertNotIn("Broken Render.mp3", json.dumps(raw, ensure_ascii=False))
            self.assertEqual(
                raw["songs"][1]["windows"][0]["raw_report"],
                "Blind report for the second song.",
            )

    def test_main_continues_after_one_window_error_without_timing_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "listener-output"
            source = listener.PreparedSource(
                source_path=root / "Private Long Render.mp3",
                sha256="3" * 64,
                duration_seconds=271.2345,
                song_number=1,
                song_id="song_001",
            )

            def fake_extract(_source_path, _destination_path, window):
                if window.index == 2:
                    raise RuntimeError(
                        f"private offset {window.start_seconds}; "
                        f"private duration {window.duration_seconds}"
                    )

            def fake_generate(_listener, conversation, max_new_tokens):
                del _listener, max_new_tokens
                if conversation[0]["content"][0]["type"] == "audio":
                    return "Anonymous report with no wrapper timing."
                return "Anonymous synthesis with no wrapper timing."

            with (
                mock.patch.object(listener, "prepare_sources", return_value=[source]),
                mock.patch.object(listener, "load_local_listener", return_value=object()),
                mock.patch.object(listener, "extract_window", side_effect=fake_extract),
                mock.patch.object(listener, "generate_greedy", side_effect=fake_generate),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = listener.main(
                    [
                        str(source.source_path),
                        "--model-dir",
                        str(root / "local-model"),
                        "--output-dir",
                        str(output_dir),
                        "--window-seconds",
                        "23.4567",
                    ]
                )

            self.assertEqual(exit_code, 1)
            private = json.loads(
                (output_dir / listener.PRIVATE_RELATIVE_PATH).read_text(
                    encoding="utf-8"
                )
            )
            raw = json.loads(
                (output_dir / listener.RAW_REPORT_RELATIVE_PATH).read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(
                (output_dir / listener.SYNTHESIS_RELATIVE_PATH).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [window["status"] for window in raw["songs"][0]["windows"]],
                ["ok", "error", "ok", "ok"],
            )
            self.assertEqual(summary["songs"][0]["window_reports_used"], 3)
            self.assertEqual(summary["songs"][0]["status"], "complete")
            self.assertIn("private offset", json.dumps(private, ensure_ascii=False))
            public_json = json.dumps({"raw": raw, "summary": summary}, ensure_ascii=False)
            self.assertNotIn("private offset", public_json)
            self.assertNotIn("Private Long Render", public_json)
            for forbidden_key in (
                "duration_seconds",
                "source_start_seconds",
                "source_end_seconds",
                "source_offset_seconds",
                "requested_excerpt_duration_seconds",
                "requested_window_duration_seconds",
                "window_seconds",
            ):
                self.assertNotIn(f'"{forbidden_key}"', public_json)
            for private_value in ("271.2345", "23.4567"):
                self.assertIn(private_value, json.dumps(private, ensure_ascii=False))
                self.assertNotIn(private_value, public_json)


class ModelConfigurationTest(unittest.TestCase):
    def test_generation_discourages_runaway_dialogue_and_transcript_loops(self) -> None:
        class FakeTensor:
            shape = (1, 2)

            def __getitem__(self, _key):
                return self

        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeProcessor:
            def apply_chat_template(self, *_args, **_kwargs):
                return FakeInputs(input_ids=FakeTensor())

            def batch_decode(self, *_args, **_kwargs):
                return ["1. A bounded answer."]

        class FakeModel:
            device = "cpu"

            def __init__(self):
                self.kwargs = None

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return FakeTensor()

        model = FakeModel()
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        loaded = listener.LoadedListener(FakeProcessor(), model, fake_torch)

        listener.generate_greedy(loaded, [], max_new_tokens=123)

        self.assertEqual(model.kwargs["max_new_tokens"], 123)
        self.assertEqual(model.kwargs["no_repeat_ngram_size"], 8)
        self.assertEqual(model.kwargs["repetition_penalty"], 1.05)

    def test_empty_decode_is_an_error_not_a_success(self) -> None:
        class FakeTensor:
            shape = (1, 2)

            def __getitem__(self, _key):
                return self

        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeProcessor:
            def apply_chat_template(self, *_args, **_kwargs):
                return FakeInputs(input_ids=FakeTensor())

            def batch_decode(self, *_args, **_kwargs):
                return ["   "]

        class FakeModel:
            device = "cpu"

            def generate(self, **_kwargs):
                return FakeTensor()

        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        loaded = listener.LoadedListener(FakeProcessor(), FakeModel(), fake_torch)

        with self.assertRaisesRegex(RuntimeError, "empty decoded response"):
            listener.generate_greedy(loaded, [], max_new_tokens=8)

    def test_loader_requests_nf4_sdpa_and_local_files_only(self) -> None:
        calls = {}

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                calls["quantization"] = self

        class FakeAutoProcessor:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls["processor"] = (path, kwargs)
                return object()

        class FakeLoadedModel:
            def eval(self):
                calls["eval"] = True

        class FakeModelClass:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls["model"] = (path, kwargs)
                return FakeLoadedModel()

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = object()
        fake_torch.manual_seed = mock.Mock()
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=mock.Mock(),
        )
        fake_torch.backends = types.SimpleNamespace(
            cudnn=types.SimpleNamespace(benchmark=True)
        )

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoProcessor = FakeAutoProcessor
        fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
        fake_transformers.Qwen2_5OmniThinkerForConditionalGeneration = FakeModelClass

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory) / "model"
            model_dir.mkdir()
            with (
                mock.patch.dict(
                    sys.modules,
                    {"torch": fake_torch, "transformers": fake_transformers},
                ),
                mock.patch.dict(os.environ, {}, clear=False),
            ):
                loaded = listener.load_local_listener(model_dir)
                offline_environment = {
                    name: os.environ.get(name)
                    for name in (
                        "HF_HUB_OFFLINE",
                        "TRANSFORMERS_OFFLINE",
                        "HF_HUB_DISABLE_TELEMETRY",
                    )
                }

        quantization = calls["quantization"].kwargs
        model_kwargs = calls["model"][1]
        self.assertTrue(quantization["load_in_4bit"])
        self.assertEqual(quantization["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(quantization["bnb_4bit_use_double_quant"])
        self.assertIs(quantization["bnb_4bit_compute_dtype"], fake_torch.float16)
        self.assertEqual(model_kwargs["attn_implementation"], "sdpa")
        self.assertEqual(model_kwargs["device_map"], "auto")
        self.assertTrue(model_kwargs["local_files_only"])
        self.assertIs(model_kwargs["torch_dtype"], fake_torch.float16)
        self.assertTrue(calls["processor"][1]["local_files_only"])
        self.assertFalse(calls["processor"][1]["trust_remote_code"])
        self.assertFalse(calls["processor"][1]["use_fast"])
        self.assertFalse(model_kwargs["trust_remote_code"])
        self.assertEqual(
            offline_environment,
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            },
        )
        self.assertTrue(calls["eval"])
        self.assertIs(loaded.torch, fake_torch)


class CommandConstructionTest(unittest.TestCase):
    def test_ffmpeg_builds_metadata_free_mono_16khz_pcm(self) -> None:
        source = Path("C:/Music/Artist - Revealing Title.mp3")
        destination = Path("C:/Temp/lofn_audio_deadbeef/audio_001_window_01.wav")
        command = listener.build_ffmpeg_command(
            "ffmpeg", source, destination, start_seconds=72.5, duration_seconds=24.0
        )

        self.assertEqual(command[0], "ffmpeg")
        self.assertEqual(command[command.index("-i") + 1], str(source))
        self.assertEqual(command[command.index("-ss") + 1], "72.5")
        self.assertEqual(command[command.index("-t") + 1], "24")
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")
        self.assertEqual(command[command.index("-map_metadata") + 1], "-1")
        self.assertEqual(command[command.index("-map_chapters") + 1], "-1")
        self.assertEqual(command[-1], str(destination))
        self.assertNotIn(source.name, destination.name)

    def test_cli_defaults_are_four_twenty_four_second_windows(self) -> None:
        args = listener.build_parser().parse_args(
            [
                "one.mp3",
                "--model-dir",
                "C:/models/qwen",
                "--output-dir",
                "out",
            ]
        )

        self.assertEqual(args.window_seconds, 24.0)
        self.assertEqual(args.windows, 4)


if __name__ == "__main__":
    unittest.main()
