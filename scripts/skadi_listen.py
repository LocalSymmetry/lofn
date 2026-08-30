#!/usr/bin/env python3
"""skadi_listen.py — the SKAÐI blind listening pass (protocol SKADI-BLIND-v1).

⛔ THE BLIND RULE. The listener hears the audio ALONE first. The blind turn
carries the audio and the questions and nothing else — no style prompt, no
title, no probe id, no expedition name. The intent is revealed only in a second
turn, on top of the blind answer, and the reveal explicitly forbids softening
what was already said. A fused "does this serve the ask?" question cannot return
NO, and a gate that cannot fail is a sentence.

⭐ REUSE THE REPAIRED INSTRUMENT, NEVER RE-DERIVE IT. Every piece of Poe plumbing
here is imported from `scripts/run_render_audit_listener.py`: the anonymous
`type=file` data-URI attachment (Poe strips OpenAI-style `input_audio`), the
retry/backoff, the response redaction, and the decoder that prefers `final` over
`analysis`. The questionnaire is that script's BLIND_TEMPLATE verbatim plus two
questions, and its REVEAL_TEMPLATE verbatim.

The ONE adaptation, declared: `attachment()` hardcodes an `audio/mpeg` data URI
because the deep dive uploaded MP3s. Skaði's ingest fetches WAVs, so the media
type is corrected per file suffix here. If only a WAV exists the payload is
warned about loudly — base64 of a 30 MB master is ~40 MB of request body, and
this script does not transcode.

Writes `listen/<probe_id>.blind.json`, `listen/<probe_id>.intent.json` and
`listen/listener_manifest.json` (model + protocol version + the blind-rule
statement). Existing answers are REUSED, never re-billed.

USAGE
    POE_API_KEY=sk-poe-... python3 scripts/skadi_listen.py output/expeditions/<dir>
    python3 scripts/skadi_listen.py <dir> --probe P3

EXIT   0 done · 1 one or more probes could not be heard · 2 missing key / input error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_music_atlas import REPO, load_gates, read  # noqa: E402
import run_render_audit_listener as L  # noqa: E402  — the repaired instrument

PROTOCOL = "SKADI-BLIND-v1"

# The render-audit eight, verbatim, plus the two questions the atlas needs: a
# genre call (territory coverage is accounted on it) and a salience call (which
# technique actually led). Inserted BEFORE the closing confidence note so the
# instrument's own last instruction stays last.
_HEAD, _, _TAIL = L.BLIND_TEMPLATE.rpartition("\n")
BLIND_TEMPLATE = (
    _HEAD
    + "\n9. Name the genre or scene in your own words."
    + "\n10. What single technique or production choice is most salient?\n"
    + _TAIL
)

REVEAL_TEMPLATE = L.REVEAL_TEMPLATE

PACKAGE_SECTIONS = {
    "style": (r"^## 1\. MUSIC PROMPT\s*$",
              r"^## 1B\. SUNO EXCLUDE PROMPT\s*$|^## 2\. LYRICS\s*$"),
    "exclude": (r"^## 1B\. SUNO EXCLUDE PROMPT\s*$", r"^## 2\. LYRICS\s*$"),
    "lyrics": (r"^## 2\. LYRICS\s*$", r"^## 3\. TITLE\s*$|\Z"),
}
MEDIA_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
               ".flac": "audio/flac", ".ogg": "audio/ogg"}


def section(text: str, name: str) -> str:
    start_pat, end_pat = PACKAGE_SECTIONS[name]
    start = re.search(start_pat, text, re.M)
    if not start:
        return ""
    rest = text[start.end():]
    end = re.search(end_pat, rest, re.M)
    return (rest[:end.start()] if end else rest).strip()


def intent_of(package: Path) -> dict[str, str]:
    text = read(package)
    out = {k: section(text, k) for k in PACKAGE_SECTIONS}
    if not out["exclude"]:
        out["exclude"] = ("No separate exclude field in this package. Negative constraints, if "
                          "any, are embedded in the style prompt.")
    return out


def attachment(audio: Path, anonymous_name: str) -> dict:
    """`L.attachment` with the media type corrected for the actual file."""
    item = L.attachment(audio, anonymous_name)
    media = MEDIA_TYPES.get(audio.suffix.lower(), "application/octet-stream")
    item["file"]["file_data"] = re.sub(r"^data:[^;]+;base64,", f"data:{media};base64,",
                                       item["file"]["file_data"])
    return item


def pick_audio(wav_rel: str | None) -> Path | None:
    """Prefer an MP3 sitting beside the master: a base64 WAV is ~40 MB of body."""
    if not wav_rel:
        return None
    wav = REPO / wav_rel
    mp3 = wav.with_suffix(".mp3")
    if mp3.exists():
        return mp3
    if wav.exists():
        if wav.suffix.lower() == ".wav":
            size = wav.stat().st_size
            print(f"[warn] {wav.name} is a {size / 1e6:.0f} MB WAV and will be sent base64 "
                  f"(~{size * 4 / 3e6:.0f} MB of request body). Place an .mp3 beside it to "
                  "avoid this; this script does not transcode.", file=sys.stderr)
        return wav
    return None


def duration_of(entry: dict, audio: Path | None) -> float | None:
    meas = entry.get("measurements") or {}
    if isinstance(meas, dict) and meas.get("duration_s"):
        return float(meas["duration_s"])
    if entry.get("duration"):
        return float(entry["duration"])
    if audio is not None:
        try:
            import soundfile as sf  # noqa: PLC0415 — optional dependency
            info = sf.info(str(audio))
            return float(info.frames) / float(info.samplerate)
        except Exception:  # noqa: BLE001
            return None
    return None


def find_package(expedition: Path, pid: str) -> Path | None:
    for p in sorted(expedition.rglob("*.md")):
        if re.match(rf"^{re.escape(pid)}(?![0-9A-Za-z])", p.name) \
                and re.search(r"^## 1\. MUSIC PROMPT\s*$", read(p), re.M):
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the SKAÐI blind listening pass.")
    ap.add_argument("expedition", help="expedition directory")
    ap.add_argument("--probe", default=None, help="listen to one probe only")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the turns; contact nothing")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    load_gates()  # a reader opens the file

    expedition = Path(args.expedition)
    if not expedition.is_absolute():
        expedition = REPO / expedition
    renders_path = expedition / "RENDERS.json"
    if not renders_path.exists():
        print(f"error: {renders_path} absent — run scripts/skadi_ingest.py first",
              file=sys.stderr)
        return 2
    renders: dict[str, dict] = json.loads(renders_path.read_text(encoding="utf-8"))
    if not renders:
        print("error: RENDERS.json holds 0 probes — an empty extraction is an ERROR, "
              "never a listening pass over nothing", file=sys.stderr)
        return 2
    if args.probe:
        renders = {k: v for k, v in renders.items() if k == args.probe}
        if not renders:
            print(f"error: probe {args.probe!r} is not in RENDERS.json", file=sys.stderr)
            return 2

    api_key = os.environ.get("POE_API_KEY", "").strip()
    if not args.dry_run and not api_key.startswith("sk-poe-"):
        print("REFUSED: POE_API_KEY is missing or malformed.\n"
              "  skadi_listen.py is the only part of Skaði that reaches the network, and it "
              "does so deliberately.\n"
              "  Set POE_API_KEY (see TOOLS.md) and re-run, or use --dry-run to inspect the "
              "turns without contacting anything.", file=sys.stderr)
        return 2

    out_dir = expedition / "listen"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": PROTOCOL,
        "model": L.MODEL,
        "endpoint": L.ENDPOINT,
        "plumbing": "scripts/run_render_audit_listener.py (imported, not re-derived)",
        "blind_rule": ("Audio + the SKADI-BLIND-v1 questions first, alone. Intent is supplied "
                       "only after the blind response, and the reveal forbids softening it."),
        "questions": BLIND_TEMPLATE,
        "probes": [],
    }

    failed: list[str] = []
    for pid in sorted(renders):
        entry = renders[pid]
        audio = pick_audio(entry.get("wav"))
        if audio is None:
            failed.append(f"{pid}: no audio on disk (RENDERS.json wav={entry.get('wav')!r}) — "
                          "run skadi_ingest.py --fetch")
            continue
        duration = duration_of(entry, audio)
        if duration is None:
            failed.append(f"{pid}: no duration available — run skadi_ingest.py --measure so "
                          "the blind prompt can bound its timestamps")
            continue
        package = find_package(expedition, pid)
        if package is None:
            failed.append(f"{pid}: no package found for the intent turn")
            continue

        blind_prompt = BLIND_TEMPLATE.format(duration=duration,
                                             duration_clock=L.duration_clock(duration))
        # ⛔ The anonymous filename carries no probe id, no title, no slug. The
        # attachment name is part of the blind surface.
        file_item = attachment(audio, f"track{audio.suffix.lower()}")
        first_user = {"role": "user",
                      "content": [{"type": "text", "text": blind_prompt}, file_item]}
        reveal_prompt = REVEAL_TEMPLATE.format(**intent_of(package))

        if args.dry_run:
            print(f"\n===== {pid} =====\nAUDIO {audio}  ({duration:.1f}s)\n"
                  f"--- BLIND TURN (audio + text below, nothing else) ---\n{blind_prompt}\n"
                  f"--- INTENT TURN (only after the blind answer) ---\n{reveal_prompt[:400]}…")
            manifest["probes"].append({"probe_id": pid, "clip_id": entry.get("clip_id"),
                                       "audio": audio.name, "dry_run": True})
            continue

        blind_path = out_dir / f"{pid}.blind.json"
        intent_path = out_dir / f"{pid}.intent.json"
        try:
            if blind_path.exists():
                blind = json.loads(blind_path.read_text(encoding="utf-8"))
                blind["content"] = L.decode_content(blind["raw_content"])
                L.write_json(blind_path, blind)
                print(f"[{pid}] BLIND_REUSE", flush=True)
            else:
                print(f"[{pid}] BLIND", flush=True)
                blind = L.public_response(L.poe_call(api_key, [first_user]))
                L.write_json(blind_path, blind)

            if intent_path.exists():
                reveal = json.loads(intent_path.read_text(encoding="utf-8"))
                reveal["content"] = L.decode_content(reveal["raw_content"])
                L.write_json(intent_path, reveal)
                print(f"[{pid}] INTENT_REUSE", flush=True)
            else:
                print(f"[{pid}] INTENT", flush=True)
                reveal = L.public_response(L.poe_call(api_key, [
                    first_user,
                    {"role": "assistant", "content": blind["raw_content"]},
                    {"role": "user", "content": reveal_prompt},
                ]))
                L.write_json(intent_path, reveal)
        except L.PoeRequestError as exc:
            L.write_json(out_dir / f"{pid}.listener_error.json",
                         {"probe": pid, "status": exc.status, "body": exc.body})
            failed.append(f"{pid}: Poe HTTP {exc.status} — see listen/{pid}.listener_error.json")
            continue

        manifest["probes"].append({
            "probe_id": pid, "clip_id": entry.get("clip_id"), "audio": audio.name,
            "blind": blind_path.name, "intent": intent_path.name,
            "duration_s": round(duration, 1),
        })

    if args.dry_run:
        print(f"\n(dry run — nothing written, nothing contacted; "
              f"{len(manifest['probes'])} of {len(renders)} probe(s) would be heard)")
    else:
        L.write_json(out_dir / "listener_manifest.json", manifest)
        print(f"WROTE     {out_dir / 'listener_manifest.json'}")
        print(f"{len(manifest['probes'])} of {len(renders)} probe(s) heard")
    if failed:
        print("\n⛔ NOT HEARD:", file=sys.stderr)
        for f in failed:
            print("  " + f, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
