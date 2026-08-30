#!/usr/bin/env python3
"""Run the listening half of a Lofn render audit through Poe.

The caller must provide POE_API_KEY in the process environment. The script
never writes the key, never sends source intent in the blind turn, and uses
anonymous attachment filenames so the track title cannot leak into that turn.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "output/render-audits/2026-07-24-eight-song-deep-dive"
MEASUREMENTS_PATH = AUDIT_ROOT / "measurements.json"
ENDPOINT = "https://api.poe.com/v1/chat/completions"
MODEL = "gpt-audio-1.5"
MAX_SERVER_RETRIES = 3


TRACKS = [
    {
        "id": "AUD-01",
        "slug": "harder-to-reach",
        "title": "Harder to Reach",
        "audio": Path(r"C:\Users\gulzi\Downloads\Harder to Reach.mp3"),
        "source": ROOT
        / "output/daily/2026-05-23/daily-run/pair_01/step10_final_suno_package.md",
        "format": "harder",
    },
    {
        "id": "AUD-02",
        "slug": "the-beautiful-thing-i-cut",
        "title": "The Beautiful Thing I Cut",
        "audio": Path(r"C:\Users\gulzi\Downloads\The Beautiful Thing I Cut.mp3"),
        "source": ROOT
        / "output/daily/2026-05-13/scientist-run/music/06_pair3_variations.md",
        "format": "beautiful",
    },
    {
        "id": "AUD-03",
        "slug": "the-caterpillar",
        "title": "The Caterpillar",
        "audio": Path(r"C:\Users\gulzi\Downloads\The Caterpillar.mp3"),
        "source": ROOT / "output/suno_render_prima-materia-v2.md",
        "format": "caterpillar",
    },
    {
        "id": "AUD-04",
        "slug": "the-cell-named-after-no-one",
        "title": "The Cell Named After No One",
        "audio": Path(r"C:\Users\gulzi\Downloads\The Cell Named After No One.mp3"),
        "source": ROOT
        / "output/daily/2026-07-04-lofn-prime-test/music/pair_04_step10_revision_synthesis.md",
        "format": "cell",
    },
    {
        "id": "AUD-05",
        "slug": "burning-bush-with-a-barcode",
        "title": "Burning Bush With a Barcode",
        "audio": Path(r"C:\Users\gulzi\Downloads\Burning Bush With a Barcode.mp3"),
        "source": ROOT / "output/songs/2026-07-06_P05V1_burning-bush-with-a-barcode.md",
        "format": "burning",
    },
    {
        "id": "AUD-06",
        "slug": "start-with-one",
        "title": "Start With One",
        "audio": Path(r"C:\Users\gulzi\Downloads\Start With One.mp3"),
        "source": ROOT / "output/songs/20260713_start_with_one_P05_R1.md",
        "format": "start",
    },
    {
        "id": "AUD-07",
        "slug": "triple-arch-over-me",
        "title": "Triple Arch Over Me",
        "audio": Path(r"C:\Users\gulzi\Downloads\Triple Arch Over Me.mp3"),
        "source": ROOT / "skills/music/references/suno_format_example_triple_arch.md",
        "format": "triple",
    },
    {
        "id": "AUD-08",
        "slug": "it-wasnt-even-locked",
        "title": "It Wasn't Even Locked",
        "audio": Path(r"C:\Users\gulzi\Downloads\It Wasn't Even Locked.mp3"),
        "upload_audio": Path(
            r"C:\Users\gulzi\AppData\Local\Temp\lofn-render-audit-20260724\AUD-08-clean.mp3"
        ),
        "source": ROOT / "output/songs/2026-07-24/p03-it-wasn-t-even-locked.md",
        "format": "locked",
    },
]


BLIND_TEMPLATE = """\
Listen to this track and describe only what you actually hear. Do not guess at intent.
The file is {duration:.1f} seconds long ({duration_clock}); never report a timestamp beyond that duration. If you are unsure, say unsure rather than inventing.
1. Structure with timestamps — where sections begin and end.
2. Any silence or near-silence longer than one second — where, and how long.
3. Sustained non-musical sound (hum, drone, whine, motor, machine noise) — where, and roughly what pitch.
4. Stereo image — what sits left, right, centre; does the width change, and where?
5. The vocal — register, distance from the mic, dry or reverberant, doubled or single.
6. The single most distinctive sonic event in the track, and its timestamp.
7. Anything that sounds like an artifact, a glitch, or a mistake — and whether it works.
8. In one sentence: what is this song about, from the audio alone?
Finish with a one-line timestamp-confidence note."""


REVEAL_TEMPLATE = """\
Here is what was asked for.

STYLE PROMPT
{style}

EXCLUDE PROMPT
{exclude}

LYRICS
{lyrics}

For each specified element, say ARRIVED / PARTIAL / ABSENT / INVERTED, with the timestamp that justifies it. Flag anything the render does that was not asked for and is better than what was. Do not rewrite or soften your blind observations now that you can see the intent."""


def lines(text: str) -> list[str]:
    return text.splitlines()


def line_range(text: str, start: int, end: int) -> str:
    return "\n".join(lines(text)[start - 1 : end]).strip()


def between(text: str, start: str, end: str | None) -> str:
    a = text.index(start) + len(start)
    b = text.index(end, a) if end else len(text)
    return text[a:b].strip()


def strip_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1]
    if value.endswith("```"):
        value = value.rsplit("\n", 1)[0]
    return value.strip()


def fenced_after(text: str, heading: str) -> str:
    tail = text[text.index(heading) + len(heading) :]
    opening = tail.index("```")
    body_start = tail.index("\n", opening) + 1
    body_end = tail.index("```", body_start)
    return tail[body_start:body_end].strip()


def extract_intent(track: dict) -> dict[str, str]:
    text = track["source"].read_text(encoding="utf-8")
    fmt = track["format"]

    if fmt == "harder":
        return {
            "style": line_range(text, 114, 118),
            "exclude": "No separate exclude field. Negative constraints are embedded in the production prompt and mix notes.",
            "lyrics": line_range(text, 6, 110),
        }
    if fmt == "beautiful":
        return {
            "style": line_range(text, 498, 498),
            "exclude": "No separate exclude field. The final sentence of the music prompt supplies the negative constraints.",
            "lyrics": line_range(text, 501, 638),
        }
    if fmt == "caterpillar":
        style = line_range(text, 8, 8)
        prefix = 'style_prompt: "'
        if style.startswith(prefix) and style.endswith('"'):
            style = style[len(prefix) : -1]
        return {
            "style": style,
            "exclude": "No separate exclude field. The bans are embedded in the style prompt.",
            "lyrics": line_range(text, 18, 131),
        }
    if fmt == "cell":
        return {
            "style": line_range(text, 253, 253),
            "exclude": line_range(text, 256, 256),
            "lyrics": line_range(text, 259, 363),
        }
    if fmt == "burning":
        return {
            "style": between(text, "## MUSIC PROMPT (Suno style — 977 chars)", "## EXCLUDE PROMPT"),
            "exclude": between(text, "## EXCLUDE PROMPT", "## LYRICS"),
            "lyrics": between(text, "## LYRICS", "## Bold Choice"),
        }
    if fmt == "start":
        return {
            "style": between(text, "### 1. MUSIC PROMPT  (SUNO STYLE PROMPT)", "### 2. EXCLUDE PROMPT"),
            "exclude": between(text, "### 2. EXCLUDE PROMPT", "### 3. LYRICS  (SUNO LYRICS)"),
            "lyrics": between(text, "### 3. LYRICS  (SUNO LYRICS)", "### 4. TITLE"),
        }
    if fmt == "triple":
        return {
            "style": fenced_after(text, "## Music Prompt"),
            "exclude": "No separate exclude field. Negative constraints are embedded at the end of the music prompt.",
            "lyrics": fenced_after(text, "## Lyrics"),
        }
    if fmt == "locked":
        return {
            "style": strip_fence(between(text, "## 1. MUSIC PROMPT", "## 2. EXCLUDE PROMPT").split("\n\n", 1)[1]),
            "exclude": strip_fence(between(text, "## 2. EXCLUDE PROMPT", "## 3. SUNO LYRICS").split("\n\n", 1)[1]),
            "lyrics": strip_fence(between(text, "## 3. SUNO LYRICS", "\n---").split("\n\n", 1)[1]),
        }
    raise ValueError(f"Unknown source format: {fmt}")


def duration_clock(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:04.1f}"


def decode_content(content: str) -> str:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return content.strip()
    if isinstance(value, dict):
        for key in ("final", "content", "text", "analysis"):
            if isinstance(value.get(key), str):
                return value[key].strip()
    return content.strip()


class PoeRequestError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Poe request failed with HTTP {status}")
        self.status = status
        self.body = body


def poe_call(api_key: str, messages: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "stream": False,
        "enable_audio_output": False,
        "auto_max_tokens": False,
        "max_tokens": 4096,
        "messages": messages,
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/LocalSymmetry/private-claw",
            "X-Title": "Lofn Render Audit",
        },
    )
    for attempt in range(1, MAX_SERVER_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            retryable = 500 <= exc.code <= 599 and attempt < MAX_SERVER_RETRIES
            if not retryable:
                raise PoeRequestError(exc.code, body) from exc
            delay = 2 ** attempt
            print(
                f"POE_RETRY status={exc.code} attempt={attempt}/{MAX_SERVER_RETRIES} "
                f"backoff_s={delay}",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def attachment(audio: Path, anonymous_name: str) -> dict:
    encoded = base64.b64encode(audio.read_bytes()).decode("ascii")
    return {
        "type": "file",
        "file": {
            "filename": anonymous_name,
            "file_data": f"data:audio/mpeg;base64,{encoded}",
        },
    }


def public_response(response: dict) -> dict:
    choice = response["choices"][0]
    return {
        "id": response.get("id"),
        "model": response.get("model"),
        "created": response.get("created"),
        "usage": response.get("usage"),
        "finish_reason": choice.get("finish_reason"),
        "raw_content": choice["message"].get("content", ""),
        "content": decode_content(choice["message"].get("content", "")),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    api_key = os.environ.get("POE_API_KEY", "").strip()
    if not api_key.startswith("sk-poe-"):
        print("POE_API_KEY missing or malformed", file=sys.stderr)
        return 2

    measurements = json.loads(MEASUREMENTS_PATH.read_text(encoding="utf-8-sig"))
    by_file = {item["file"]: item for item in measurements}

    manifest = {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "blind_rule": "Audio + canonical blind prompt first; intent supplied only after the blind response.",
        "tracks": [],
    }

    for index, track in enumerate(TRACKS, start=1):
        out_dir = AUDIT_ROOT / track["slug"]
        out_dir.mkdir(parents=True, exist_ok=True)
        blind_path = out_dir / "blind_listener.json"
        reveal_path = out_dir / "intent_comparison.json"
        intent = extract_intent(track)
        measurement = by_file[track["audio"].name]
        duration = float(measurement["duration_s"])
        blind_prompt = BLIND_TEMPLATE.format(
            duration=duration, duration_clock=duration_clock(duration)
        )
        upload_audio = track.get("upload_audio", track["audio"])
        if not upload_audio.exists():
            upload_audio = track["audio"]
        file_item = attachment(upload_audio, f"{track['id']}.mp3")
        first_user = {
            "role": "user",
            "content": [{"type": "text", "text": blind_prompt}, file_item],
        }

        if blind_path.exists():
            blind = json.loads(blind_path.read_text(encoding="utf-8"))
            blind["content"] = decode_content(blind["raw_content"])
            write_json(blind_path, blind)
            print(f"[{index}/8] BLIND_REUSE {track['id']}", flush=True)
        else:
            print(f"[{index}/8] BLIND {track['id']}", flush=True)
            try:
                blind_raw = poe_call(api_key, [first_user])
            except PoeRequestError as exc:
                write_json(
                    out_dir / "listener_error.json",
                    {"stage": "blind", "status": exc.status, "body": exc.body},
                )
                raise
            blind = public_response(blind_raw)
            write_json(blind_path, blind)
            print(f"[{index}/8] BLIND_DONE {track['id']}", flush=True)

        reveal_prompt = REVEAL_TEMPLATE.format(**intent)
        second_messages = [
            first_user,
            {"role": "assistant", "content": blind["raw_content"]},
            {"role": "user", "content": reveal_prompt},
        ]
        if reveal_path.exists():
            reveal = json.loads(reveal_path.read_text(encoding="utf-8"))
            reveal["content"] = decode_content(reveal["raw_content"])
            write_json(reveal_path, reveal)
            print(f"[{index}/8] REVEAL_REUSE {track['id']}", flush=True)
        else:
            print(f"[{index}/8] REVEAL {track['id']}", flush=True)
            try:
                reveal_raw = poe_call(api_key, second_messages)
            except PoeRequestError as exc:
                write_json(
                    out_dir / "listener_error.json",
                    {"stage": "reveal", "status": exc.status, "body": exc.body},
                )
                raise
            reveal = public_response(reveal_raw)
            write_json(reveal_path, reveal)
            print(f"[{index}/8] REVEAL_DONE {track['id']}", flush=True)

        live_error = out_dir / "listener_error.json"
        recovered_error = out_dir / "listener_error_recovered.json"
        if live_error.exists():
            live_error.replace(recovered_error)

        packet = (
            f"# Intent packet — {track['title']}\n\n"
            f"Source: `{track['source'].relative_to(ROOT).as_posix()}`\n\n"
            f"## Style prompt\n\n{intent['style']}\n\n"
            f"## Exclude prompt\n\n{intent['exclude']}\n\n"
            f"## Lyrics\n\n{intent['lyrics']}\n"
        )
        (out_dir / "INTENT_PACKET.md").write_text(packet, encoding="utf-8")
        manifest["tracks"].append(
            {
                "id": track["id"],
                "title": track["title"],
                "slug": track["slug"],
                "duration_s": duration,
                "source": track["source"].relative_to(ROOT).as_posix(),
                "blind_response": f"{track['slug']}/blind_listener.json",
                "intent_response": f"{track['slug']}/intent_comparison.json",
            }
        )
        time.sleep(0.25)

    write_json(AUDIT_ROOT / "listener_manifest.json", manifest)
    print("LISTENER_BATCH_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
