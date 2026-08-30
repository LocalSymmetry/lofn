#!/usr/bin/env python3
"""Run a local, quarantined blind listening pass over rendered audio.

The script deliberately has no reveal phase. Source identity, hashes, exact
timing, and operational errors live under ``private/``. The listening model
receives only metadata-free, mono 16 kHz WAV excerpts with neutral filenames
and the fixed blind prompt below. Blind-public window wrappers expose only an
anonymous window ID and a coarse semantic role. Numeric and stereo measurement
remain the job of ``scripts/measure_render.py``.

Example:
    python scripts/run_local_render_listener.py output/*.mp3 \
        --model-dir C:/models/Qwen2.5-Omni-3B \
        --output-dir output/render-listener
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


DEFAULT_WINDOW_SECONDS = 24.0
DEFAULT_WINDOWS = 4
WINDOW_MAX_NEW_TOKENS = 320
SYNTHESIS_MAX_NEW_TOKENS = 448

PRIVATE_RELATIVE_PATH = Path("private/source_identity_map.json")
RAW_REPORT_RELATIVE_PATH = Path("blind/window_reports.raw.json")
SYNTHESIS_RELATIVE_PATH = Path("blind/song_summaries.json")

MODEL_DESCRIPTION = "Qwen2.5-Omni-3B Thinker (local 4-bit NF4, SDPA)"

# This is intentionally fixed. Do not interpolate a title, known words, source
# prompt, filename, or intended effect into it.
BLIND_WINDOW_PROMPT = """Listen to this anonymous audio excerpt and describe only what is actually audible. Do not infer hidden context. Answer exactly eight numbered items, using one or two concise sentences per item. Do not repeat these questions. Stop immediately after item 8; do not continue as a dialogue.
1. What structural role does the excerpt appear to play? Give timestamps relative to the excerpt.
2. Describe the vocal register, delivery, distance, doubling, and reverberation.
3. Describe the instruments, timbres, rhythm, density, and movement of energy.
4. Describe the emotional trajectory without inventing a backstory.
5. Identify the single most distinctive audible event and its timestamp.
6. Identify any artifact, glitch, or apparent mistake, and say whether it works musically.
7. Transcribe only words or vocal sounds you can confidently make out; mark uncertainty explicitly.
8. In one sentence, what does the excerpt communicate from sound alone?"""

BLIND_SYNTHESIS_INSTRUCTION = """Consolidate the blind window reports below for one anonymous track. Use only those reports and their anonymous semantic-role metadata. Do not invent missing evidence or infer hidden context. Answer exactly six numbered items. Do not repeat these instructions. Stop immediately after item 6; do not continue as a dialogue.

Return these six numbered items:
1. Likely large-scale structure and energy arc across opening, interior, and ending.
2. Vocal character plus any confidently audible recurring words, sounds, or refrain; preserve uncertainty.
3. Instrumental, rhythmic, spatial, and timbral character.
4. Strongest distinctive events, retaining excerpt-relative timestamps when the reports provide them.
5. Artifacts or mistakes, agreements or conflicts among reports, and the most important uncertainty.
6. One final sentence describing what the track communicates from sound alone."""

_ANONYMOUS_FILE_RE = re.compile(r"audio_\d{3,}_window_\d{2,}\.wav\Z")
_ANONYMOUS_DIR_RE = re.compile(r"lofn_audio_[A-Za-z0-9_.-]+\Z")
_BLIND_WINDOW_ID_RE = re.compile(r"window_\d{2,}\Z")
_BLIND_WINDOW_ROLES = frozenset({"opening", "interior", "ending"})
_PUBLIC_ERROR_MESSAGE = "Details are quarantined in the private source map."
_SPURIOUS_CONTINUATION_RE = re.compile(
    r"(?im)^(?:human\s*:|assistant\s*:|user\s*:|question\s*:|answer\s*:|"
    r"you are an ai assistant\b|<\|im_start\|>)"
)


@dataclass(frozen=True)
class WindowSpec:
    """One deterministic excerpt location within a source render."""

    index: int
    role: str
    start_seconds: float
    duration_seconds: float

    @property
    def blind_role(self) -> str:
        """Return a coarse role that cannot identify an absolute source span."""

        if self.role == "opening" or self.role == "ending":
            return self.role
        if self.role == "full_track":
            # Revealing that a short excerpt is the whole track makes its
            # excerpt-relative timestamps reconstruct the source duration.
            return "opening"
        if self.role.startswith("interior_"):
            return "interior"
        raise ValueError(f"unsupported private window role: {self.role}")

    def blind_dict(self) -> dict[str, Any]:
        """Return the strict allowlist used by blind-public artifacts."""

        return {
            "window_id": f"window_{self.index:02d}",
            "role": self.blind_role,
        }

    def private_dict(self) -> dict[str, Any]:
        """Return the exact schedule; this may only be written under private/."""

        return {
            "window_id": f"window_{self.index:02d}",
            "selection_role": self.role,
            "source_start_seconds": self.start_seconds,
            "source_end_seconds": round(
                self.start_seconds + self.duration_seconds, 6
            ),
            "requested_excerpt_duration_seconds": self.duration_seconds,
        }


@dataclass
class PreparedSource:
    """Private source information. Never pass this object to the processor."""

    source_path: Path
    sha256: str | None = None
    duration_seconds: float | None = None
    song_number: int = 0
    song_id: str = ""
    private_window_schedule: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.sha256 is not None and self.duration_seconds is not None

    def identity_dict(self) -> dict[str, Any]:
        return {
            "song_id": self.song_id,
            "source_path": str(self.source_path),
            "sha256": self.sha256,
            "duration_seconds": self.duration_seconds,
            "window_schedule": list(self.private_window_schedule),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class LoadedListener:
    """The single processor/model load shared by every window and synthesis."""

    processor: Any
    model: Any
    torch: Any


class ExternalCommandError(RuntimeError):
    """Compact subprocess failure that retains stderr for the JSON audit trail."""

    def __init__(self, program: str, returncode: int, stderr: str):
        detail = stderr.strip() or "no stderr"
        super().__init__(f"{program} exited with code {returncode}: {detail[-4000:]}")
        self.program = program
        self.returncode = returncode
        self.stderr = stderr


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local Qwen2.5-Omni blind listening on anonymous audio windows."
    )
    parser.add_argument("audio_paths", nargs="+", type=Path)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Local Qwen2.5-Omni model directory; remote model IDs are rejected.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--window-seconds",
        type=positive_float,
        default=DEFAULT_WINDOW_SECONDS,
    )
    parser.add_argument("--windows", type=positive_int, default=DEFAULT_WINDOWS)
    return parser


def format_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def schedule_windows(
    duration_seconds: float,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    windows: int = DEFAULT_WINDOWS,
) -> list[WindowSpec]:
    """Return evenly spaced opening/interior/ending excerpts.

    The first excerpt begins at zero and the last excerpt ends with the source.
    Intermediate excerpts are evenly spaced between them. Short sources use one
    full-track excerpt rather than duplicate windows.
    """

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and greater than zero")
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and greater than zero")
    if windows <= 0:
        raise ValueError("windows must be greater than zero")

    excerpt_seconds = min(duration_seconds, window_seconds)
    if duration_seconds <= window_seconds or windows == 1:
        role = "full_track" if duration_seconds <= window_seconds else "opening"
        return [
            WindowSpec(
                index=1,
                role=role,
                start_seconds=0.0,
                duration_seconds=round(excerpt_seconds, 6),
            )
        ]

    last_start = duration_seconds - window_seconds
    result: list[WindowSpec] = []
    for offset in range(windows):
        start = last_start * offset / (windows - 1)
        if offset == 0:
            role = "opening"
        elif offset == windows - 1:
            role = "ending"
        else:
            role = f"interior_{offset:02d}"
        result.append(
            WindowSpec(
                index=offset + 1,
                role=role,
                start_seconds=round(start, 6),
                duration_seconds=round(window_seconds, 6),
            )
        )
    return result


def anonymous_window_filename(song_number: int, window_number: int) -> str:
    if song_number <= 0 or window_number <= 0:
        raise ValueError("anonymous indices must be greater than zero")
    return f"audio_{song_number:03d}_window_{window_number:02d}.wav"


def validate_anonymous_model_path(audio_path: Path) -> None:
    """Fail closed if a processor-facing path does not use our neutral scheme."""

    if not _ANONYMOUS_FILE_RE.fullmatch(audio_path.name):
        raise ValueError(f"non-anonymous processor filename rejected: {audio_path.name}")
    if not _ANONYMOUS_DIR_RE.fullmatch(audio_path.parent.name):
        raise ValueError("processor audio must live in a neutral lofn_audio_* temp directory")


def build_window_conversation(anonymous_audio_path: Path) -> list[dict[str, Any]]:
    """Build a fresh one-turn conversation containing no source metadata."""

    validate_anonymous_model_path(anonymous_audio_path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(anonymous_audio_path)},
                {"type": "text", "text": BLIND_WINDOW_PROMPT},
            ],
        }
    ]


def project_first_response(raw_response: str) -> str:
    """Return the first answer while preserving the raw model text elsewhere.

    Small local multimodal models sometimes finish a useful answer and then
    hallucinate a new ``Human:``/``Question:`` turn instead of emitting EOS.
    The durable ``raw_report`` remains byte-for-byte model output; only this
    allowlisted projection is supplied to the cross-window synthesis and saved
    as the usable song-level answer.
    """

    if not isinstance(raw_response, str):
        raise TypeError("raw response must be text")
    marker = _SPURIOUS_CONTINUATION_RE.search(raw_response)
    projected = raw_response[: marker.start()] if marker else raw_response
    projected = projected.strip()
    if not projected:
        raise ValueError("model response contained no usable first answer")
    return projected


def project_numbered_response(raw_response: str, expected_items: int) -> str:
    """Select the final complete numbered answer from the first model turn.

    A weak model may echo the numbered instruction and then emit the actual
    numbered answer. Choosing the final complete 1..N block removes that echo
    without rewriting any musical observation. If no complete block is found,
    retain the first-response projection so evidence is never silently lost.
    """

    if expected_items <= 0:
        raise ValueError("expected_items must be greater than zero")
    projected = project_first_response(raw_response)
    starts = [match.start() for match in re.finditer(r"(?m)^1\.\s", projected)]
    for start in reversed(starts):
        candidate = projected[start:].strip()
        if all(
            re.search(rf"(?m)^{item}\.\s", candidate)
            for item in range(1, expected_items + 1)
        ):
            return candidate
    return projected


def build_synthesis_conversation(song_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a fresh turn from an allowlisted projection of blind reports.

    Exact source timing may exist on the caller's private objects, so never
    serialize a window record wholesale here.
    """

    window_reports = []
    for window in song_report.get("windows", []):
        if window.get("status") != "ok":
            continue
        window_id = window.get("window_id")
        role = window.get("role")
        raw_report = window.get("raw_report")
        if not isinstance(window_id, str) or not _BLIND_WINDOW_ID_RE.fullmatch(
            window_id
        ):
            raise ValueError("invalid anonymous window ID in blind report")
        if role not in _BLIND_WINDOW_ROLES:
            raise ValueError("invalid semantic role in blind report")
        if not isinstance(raw_report, str):
            raise ValueError("successful blind window has no raw report text")
        window_reports.append(
            {
                "window_id": window_id,
                "role": role,
                "blind_report": project_numbered_response(raw_report, 8),
            }
        )
    if not window_reports:
        raise ValueError("cannot synthesize a song with no successful window reports")

    report_text = json.dumps(window_reports, ensure_ascii=False, indent=2)
    prompt = f"{BLIND_SYNTHESIS_INSTRUCTION}\n\nBLIND WINDOW REPORTS\n{report_text}"
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def assert_source_quarantined(
    conversation: Sequence[dict[str, Any]], source_path: Path
) -> None:
    """Prove the injected turn contains neither source path nor source filename."""

    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [item for child in value.values() for item in strings(child)]
        if isinstance(value, (list, tuple)):
            return [item for child in value for item in strings(child)]
        return []

    visible = "\n".join(strings(conversation)).casefold()
    forbidden = (str(source_path.resolve()), source_path.name)
    for value in forbidden:
        if value and value.casefold() in visible:
            raise ValueError("source identity leaked into a model-visible conversation")

    source_path_parts = {source_path.name.casefold(), source_path.stem.casefold()}
    for message in conversation:
        for content in message.get("content", []):
            if not isinstance(content, dict) or content.get("type") != "audio":
                continue
            audio_value = content.get("audio")
            if not isinstance(audio_value, str):
                continue
            visible_path_parts = {part.casefold() for part in Path(audio_value).parts}
            if source_path_parts & visible_path_parts:
                raise ValueError("source identity leaked into a model-visible audio path")


def build_ffprobe_command(ffprobe: str, source_path: Path) -> list[str]:
    return [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(source_path),
    ]


def build_ffmpeg_command(
    ffmpeg: str,
    source_path: Path,
    destination_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> list[str]:
    """Build an accurate, metadata-free mono 16 kHz PCM extraction command."""

    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-ss",
        format_seconds(start_seconds),
        "-t",
        format_seconds(duration_seconds),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination_path),
    ]


def run_command(command: Sequence[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise ExternalCommandError(command[0], result.returncode, result.stderr)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    result = run_command(build_ffprobe_command(ffprobe, path), timeout_seconds=120)
    try:
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("ffprobe returned no valid format duration") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"ffprobe returned invalid duration: {duration!r}")
    return round(duration, 6)


def extract_window(
    source_path: Path,
    destination_path: Path,
    window: WindowSpec,
    ffmpeg: str = "ffmpeg",
) -> None:
    validate_anonymous_model_path(destination_path)
    command = build_ffmpeg_command(
        ffmpeg,
        source_path,
        destination_path,
        window.start_seconds,
        window.duration_seconds,
    )
    run_command(command, timeout_seconds=300)
    if not destination_path.is_file() or destination_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg reported success but produced no audio")


def error_record(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def quarantine_error(
    record: dict[str, str], source_path: Path | None = None, temp_root: Path | None = None
) -> dict[str, str]:
    # Do not echo subprocess/model diagnostics into blind-public artifacts.
    # Besides paths and titles, those diagnostics can contain exact offsets,
    # durations, or media metadata. Full details already live in private/.
    del source_path, temp_root
    return {
        "stage": record["stage"],
        "error_type": record["error_type"],
        "message": _PUBLIC_ERROR_MESSAGE,
    }


def prepare_sources(
    audio_paths: Sequence[Path], ffprobe: str = "ffprobe"
) -> list[PreparedSource]:
    """Hash and probe inputs, then give them deterministic content-sorted IDs."""

    prepared: list[PreparedSource] = []
    for supplied_path in audio_paths:
        source = PreparedSource(source_path=supplied_path.expanduser().resolve())
        if not source.source_path.is_file():
            source.errors.append(
                error_record("input", FileNotFoundError(f"audio file not found: {source.source_path}"))
            )
            prepared.append(source)
            continue
        try:
            source.sha256 = sha256_file(source.source_path)
        except Exception as exc:  # Preserve the failure and continue with other inputs.
            source.errors.append(error_record("sha256", exc))
        try:
            source.duration_seconds = probe_duration(source.source_path, ffprobe=ffprobe)
        except Exception as exc:  # Preserve the failure and continue with other inputs.
            source.errors.append(error_record("ffprobe", exc))
        prepared.append(source)

    prepared.sort(
        key=lambda item: (
            item.sha256 is None,
            item.sha256 or "",
            str(item.source_path).casefold(),
        )
    )
    for number, source in enumerate(prepared, start=1):
        source.song_number = number
        source.song_id = f"song_{number:03d}"
    return prepared


def build_identity_payload(
    sources: Sequence[PreparedSource],
    requested_window_seconds: float,
    requested_windows: int,
    run_errors: Sequence[dict[str, str]] = (),
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "classification": "PRIVATE SOURCE MAP - keep outside blind review",
        "ordering": "SHA-256 ascending, then resolved path",
        "window_request": {
            "requested_window_duration_seconds": requested_window_seconds,
            "requested_window_count": requested_windows,
        },
        "songs": [source.identity_dict() for source in sources],
        "run_errors": list(run_errors),
    }


def build_blind_report_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "raw blind window reports",
        "model": MODEL_DESCRIPTION,
        "blind_rule": (
            "Only anonymous metadata-free mono excerpts and the fixed prompt were supplied. "
            "Window wrappers contain only anonymous IDs and semantic roles; source identity, "
            "exact timing, and later comparison material are quarantined elsewhere."
        ),
        "numeric_and_stereo_analysis": "separate; not inferred by this mono listener",
        "prompt": BLIND_WINDOW_PROMPT,
        "songs": [],
        "run_errors": [],
    }


def build_synthesis_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "song-level synthesis from blind window reports only",
        "model": MODEL_DESCRIPTION,
        "songs": [],
        "run_errors": [],
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def load_local_listener(model_dir: Path) -> LoadedListener:
    """Load the official Qwen Thinker once, offline, in 4-bit NF4 with SDPA."""

    resolved_model_dir = model_dir.expanduser().resolve()
    if not resolved_model_dir.is_dir():
        raise FileNotFoundError(f"local model directory not found: {resolved_model_dir}")

    # Fail closed against an accidental Hub fetch. This runner never uploads or
    # downloads audio, prompts, reports, or model data.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(
        str(resolved_model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False,
    )
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(resolved_model_dir),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return LoadedListener(processor=processor, model=model, torch=torch)


def generate_greedy(
    listener: LoadedListener,
    conversation: Sequence[dict[str, Any]],
    max_new_tokens: int,
) -> str:
    """Run one fresh conversation and decode only newly generated tokens."""

    inputs = listener.processor.apply_chat_template(
        list(conversation),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(listener.model.device)
    input_length = inputs["input_ids"].shape[1]
    with listener.torch.inference_mode():
        generated_ids = listener.model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            repetition_penalty=1.05,
            no_repeat_ngram_size=8,
            max_new_tokens=max_new_tokens,
        )
    new_ids = generated_ids[:, input_length:]
    decoded = listener.processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not decoded:
        raise RuntimeError("model returned no decoded response")
    response = decoded[0].strip()
    if not response:
        raise RuntimeError("model returned an empty decoded response")
    return response


def source_public_errors(source: PreparedSource) -> list[dict[str, str]]:
    return [quarantine_error(item, source_path=source.source_path) for item in source.errors]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    identity_path = output_dir / PRIVATE_RELATIVE_PATH
    raw_report_path = output_dir / RAW_REPORT_RELATIVE_PATH
    synthesis_path = output_dir / SYNTHESIS_RELATIVE_PATH

    sources = prepare_sources(args.audio_paths)
    private_run_errors: list[dict[str, str]] = []
    blind_payload = build_blind_report_payload()
    synthesis_payload = build_synthesis_payload()

    def persist_identity() -> None:
        write_json_atomic(
            identity_path,
            build_identity_payload(
                sources,
                requested_window_seconds=args.window_seconds,
                requested_windows=args.windows,
                run_errors=private_run_errors,
            ),
        )

    persist_identity()

    for source in sources:
        if source.ready:
            continue
        public_errors = source_public_errors(source)
        blind_payload["songs"].append(
            {
                "song_id": source.song_id,
                "status": "source_error",
                "windows": [],
                "errors": public_errors,
            }
        )
        synthesis_payload["songs"].append(
            {
                "song_id": source.song_id,
                "status": "source_error",
                "window_reports_used": 0,
                "errors": public_errors,
            }
        )

    write_json_atomic(raw_report_path, blind_payload)
    write_json_atomic(synthesis_path, synthesis_payload)

    viable_sources = [source for source in sources if source.ready]
    if not viable_sources:
        print("No readable audio inputs; errors were saved to JSON.", file=sys.stderr)
        return 1

    try:
        listener = load_local_listener(args.model_dir)
    except Exception as exc:
        private_error = error_record("model_load", exc)
        private_run_errors.append(private_error)
        public_error = quarantine_error(
            private_error, temp_root=args.model_dir.expanduser().resolve()
        )
        blind_payload["run_errors"].append(public_error)
        synthesis_payload["run_errors"].append(public_error)
        persist_identity()
        write_json_atomic(raw_report_path, blind_payload)
        write_json_atomic(synthesis_path, synthesis_payload)
        print("Local listener model failed to load; error saved to JSON.", file=sys.stderr)
        return 2

    any_errors = bool(private_run_errors) or any(source.errors for source in sources)
    with tempfile.TemporaryDirectory(
        prefix="lofn_audio_", ignore_cleanup_errors=True
    ) as temporary_directory:
        temp_root = Path(temporary_directory)
        for source_index, source in enumerate(viable_sources, start=1):
            print(
                f"[{source_index}/{len(viable_sources)}] {source.song_id}: blind windows",
                flush=True,
            )
            assert source.duration_seconds is not None
            schedule = schedule_windows(
                source.duration_seconds,
                window_seconds=args.window_seconds,
                windows=args.windows,
            )
            source.private_window_schedule = [
                window.private_dict() for window in schedule
            ]
            persist_identity()
            song_report: dict[str, Any] = {
                "song_id": source.song_id,
                "status": "in_progress",
                "windows": [],
                "errors": [],
            }
            blind_payload["songs"].append(song_report)
            blind_payload["songs"].sort(key=lambda item: item["song_id"])
            write_json_atomic(raw_report_path, blind_payload)

            for window in schedule:
                window_record = window.blind_dict()
                window_record["status"] = "in_progress"
                song_report["windows"].append(window_record)
                write_json_atomic(raw_report_path, blind_payload)

                anonymous_path = temp_root / anonymous_window_filename(
                    source.song_number, window.index
                )
                try:
                    extract_window(source.source_path, anonymous_path, window)
                    conversation = build_window_conversation(anonymous_path)
                    assert_source_quarantined(conversation, source.source_path)
                    window_record["raw_report"] = generate_greedy(
                        listener,
                        conversation,
                        max_new_tokens=WINDOW_MAX_NEW_TOKENS,
                    )
                    window_record["status"] = "ok"
                except Exception as exc:
                    any_errors = True
                    private_error = error_record(
                        f"{window_record['window_id']}_blind", exc
                    )
                    source.errors.append(private_error)
                    public_error = quarantine_error(
                        private_error,
                        source_path=source.source_path,
                        temp_root=temp_root,
                    )
                    window_record["status"] = "error"
                    window_record["error"] = public_error
                    song_report["errors"].append(public_error)
                    persist_identity()
                # The unmodified blind response is durable before any later
                # consolidation step can run.
                write_json_atomic(raw_report_path, blind_payload)

            successful_windows = [
                window for window in song_report["windows"] if window["status"] == "ok"
            ]
            if not successful_windows:
                song_report["status"] = "failed"
            elif song_report["errors"]:
                song_report["status"] = "partial"
            else:
                song_report["status"] = "complete"

            # This write is intentionally before the synthesis call. There is
            # no reveal phase in this program.
            write_json_atomic(raw_report_path, blind_payload)

            summary_record: dict[str, Any] = {
                "song_id": source.song_id,
                "status": "in_progress",
                "window_reports_used": len(successful_windows),
                "errors": [],
            }
            synthesis_payload["songs"].append(summary_record)
            synthesis_payload["songs"].sort(key=lambda item: item["song_id"])
            write_json_atomic(synthesis_path, synthesis_payload)

            if successful_windows:
                try:
                    synthesis_conversation = build_synthesis_conversation(song_report)
                    summary_record["raw_synthesis"] = generate_greedy(
                        listener,
                        synthesis_conversation,
                        max_new_tokens=SYNTHESIS_MAX_NEW_TOKENS,
                    )
                    summary_record["usable_synthesis"] = project_numbered_response(
                        summary_record["raw_synthesis"], 6
                    )
                    summary_record["status"] = "complete"
                except Exception as exc:
                    any_errors = True
                    private_error = error_record("blind_synthesis", exc)
                    source.errors.append(private_error)
                    public_error = quarantine_error(
                        private_error,
                        source_path=source.source_path,
                        temp_root=temp_root,
                    )
                    summary_record["status"] = "error"
                    summary_record["errors"].append(public_error)
                    persist_identity()
            else:
                summary_record["status"] = "no_window_reports"
                summary_record["errors"].append(
                    {
                        "stage": "blind_synthesis",
                        "error_type": "NoWindowReports",
                        "message": "No successful blind window reports were available.",
                    }
                )
            write_json_atomic(synthesis_path, synthesis_payload)

    persist_identity()
    write_json_atomic(raw_report_path, blind_payload)
    write_json_atomic(synthesis_path, synthesis_payload)
    print(f"Blind window reports: {raw_report_path}", flush=True)
    print(f"Blind song summaries: {synthesis_path}", flush=True)
    print(f"Private identity map: {identity_path}", flush=True)
    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
