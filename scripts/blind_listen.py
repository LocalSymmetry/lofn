#!/usr/bin/env python3
"""blind_listen.py — the listening half of a render audit, for arbitrary files.

THE BLIND RULE (`.claude/skills/lofn-render-audit/SKILL.md`): the audio goes out
ALONE. No title, no style prompt, no lyrics, no intent, no expectations — a model
shown the intent confirms the intent. Filenames are anonymised before upload so
the title cannot leak through the attachment name.

`scripts/run_render_audit_listener.py` does this for the 2026-07-24 eight-song
deep dive and has that run's tracks and source-file parsers hard-coded. This one
takes any file and is the reusable form.

    POE_API_KEY=... python3 scripts/blind_listen.py OUT.json a.mp3 b.mp3
    POE_API_KEY=... python3 scripts/blind_listen.py OUT.json a.mp3 --reveal intent_a.txt

The optional second turn (`--reveal`, one file per track, same order) supplies
the intent AFTER the blind answer is already on record, and asks for the
ARRIVED / PARTIAL / ABSENT / INVERTED table.

AUTONOMY.md: spends Poe points (metered), writes locally, publishes nothing.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://api.poe.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-audio-1.5"
MAX_RETRIES = 3

BLIND_PROMPT = """\
Listen to this track and describe only what you actually hear. Do not guess at intent.
The file is {duration:.1f} seconds long ({clock}); never report a timestamp beyond that duration. If you are unsure, say unsure rather than inventing.
1. Structure with timestamps - where sections begin and end.
2. Any silence or near-silence longer than one second - where, and how long.
3. Sustained non-musical sound (hum, drone, whine, motor, machine noise) - where, and roughly what pitch.
4. Stereo image - what sits left, right, centre; does the width change, and where?
5. The vocal - register, distance from the mic, dry or reverberant, doubled or single.
6. The single most distinctive sonic event in the track, and its timestamp.
7. Anything that sounds like an artifact, a glitch, or a mistake - and whether it works.
8. In one sentence: what is this song about, from the audio alone?
Finish with a one-line timestamp-confidence note."""

REVEAL_PROMPT = """\
Here is what was asked for.

{intent}

For each specified element, say ARRIVED / PARTIAL / ABSENT / INVERTED, with the timestamp that justifies it. Flag anything the render does that was not asked for and is better than what was. Do not rewrite or soften your blind observations now that you can see the intent."""


def clockstr(seconds: float) -> str:
    m = int(seconds // 60)
    return "%d:%04.1f" % (m, seconds - m * 60)


def attachment(path: Path, anon_name: str) -> dict:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "file",
        "file": {
            "filename": anon_name,
            "file_data": "data:audio/mpeg;base64," + encoded,
        },
    }


def poe_call(api_key: str, model: str, messages: list[dict], max_tokens: int = 4096) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "enable_audio_output": False,
        "auto_max_tokens": False,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "X-Title": "Lofn Blind Listen",
        },
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if not (500 <= exc.code <= 599) or attempt == MAX_RETRIES:
                raise RuntimeError("HTTP %d: %s" % (exc.code, body[:800])) from exc
            delay = 2 ** attempt
            print("  retry status=%d attempt=%d backoff=%ds" % (exc.code, attempt, delay), file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def content_of(response: dict) -> str:
    return (response["choices"][0]["message"].get("content") or "").strip()


def duration_of(path: Path) -> float:
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="path for the JSON transcript")
    ap.add_argument("audio", nargs="+", help="audio files, in order")
    ap.add_argument("--reveal", action="append", default=[], help="intent text file for the second turn (same order as audio)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--duration", action="append", type=float, default=[], help="duration in seconds per track (skips soundfile)")
    args = ap.parse_args()

    api_key = os.environ.get("POE_API_KEY", "").strip()
    if not api_key.startswith("sk-poe-"):
        print("POE_API_KEY missing or malformed", file=sys.stderr)
        return 2

    results = []
    for i, raw in enumerate(args.audio):
        path = Path(raw)
        dur = args.duration[i] if i < len(args.duration) else duration_of(path)
        anon = "track_%02d.mp3" % (i + 1)
        prompt = BLIND_PROMPT.format(duration=dur, clock=clockstr(dur))
        print("[%d/%d] blind pass on %s (as %s, %.1fs)" % (i + 1, len(args.audio), path.name, anon, dur), file=sys.stderr)

        blind_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    attachment(path, anon),
                ],
            }
        ]
        resp = poe_call(api_key, args.model, blind_messages)
        blind = content_of(resp)
        entry = {
            "audio_file": str(path),
            "anon_name": anon,
            "duration_s": dur,
            "model": resp.get("model") or args.model,
            "blind_prompt": prompt,
            "blind_response": blind,
            "blind_usage": resp.get("usage"),
        }

        if i < len(args.reveal):
            intent = Path(args.reveal[i]).read_text(encoding="utf-8")
            print("      reveal pass", file=sys.stderr)
            reveal_messages = blind_messages + [
                {"role": "assistant", "content": blind},
                {"role": "user", "content": REVEAL_PROMPT.format(intent=intent)},
            ]
            r2 = poe_call(api_key, args.model, reveal_messages)
            entry["reveal_intent_file"] = args.reveal[i]
            entry["reveal_response"] = content_of(r2)
            entry["reveal_usage"] = r2.get("usage")

        results.append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote %s" % out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
