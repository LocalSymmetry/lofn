#!/usr/bin/env python3
"""verify_canvas.py — check rendered Canvas assets against Spotify's published specs.

Spotify Canvas (from the upload dialog, 2026-08-09):
  * 9:16 aspect ratio
  * at least 720px tall
  * MP4 or JPEG
  * (Canvas plays as a silent 3-8s loop; audio in the file is ignored, not rejected)

Usage:
    python3 scripts/verify_canvas.py output/distribution/canvas/renders/*.mp4

No network. Reports what it EXTRACTED before what it CONCLUDED, per COMPETITION_LEARNINGS
L25/L28/L31 — a checker that silently extracts nothing must never print PASS.
"""
import json
import subprocess
import sys
from pathlib import Path

TARGET_RATIO = 9 / 16          # 0.5625
RATIO_TOL = 0.01               # within 1% is fine; Spotify letterboxes tiny deviations
MIN_HEIGHT = 720
DUR_MIN, DUR_MAX = 3.0, 8.0    # Canvas loop window


def probe(path: Path) -> dict:
    """ffprobe → dict. Capture BYTES and decode UTF-8 explicitly.

    NEVER use text=True here: on Windows that decodes with cp1252 and silently
    mangles non-ASCII output (vault/RUN_LEDGER.md, 2026-08-04).
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace").strip())
    return json.loads(out.stdout.decode("utf-8"))


def check(path: Path) -> tuple[bool, list[str], dict]:
    facts, problems = {}, []
    if path.suffix.lower() in (".jpg", ".jpeg"):
        from PIL import Image
        with Image.open(path) as im:
            facts["w"], facts["h"] = im.size
        facts["dur"] = None
        facts["kind"] = "still"
    else:
        info = probe(path)
        vs = [s for s in info["streams"] if s.get("codec_type") == "video"]
        if not vs:
            return False, ["no video stream found"], facts
        v = vs[0]
        facts["w"], facts["h"] = int(v["width"]), int(v["height"])
        facts["codec"] = v.get("codec_name")
        facts["kind"] = "video"
        facts["has_audio"] = any(s.get("codec_type") == "audio" for s in info["streams"])
        try:
            facts["dur"] = float(info["format"]["duration"])
        except (KeyError, ValueError):
            facts["dur"] = None
        if facts["codec"] not in ("h264", "hevc"):
            problems.append(f"codec {facts['codec']} — Spotify wants H.264 MP4")

    facts["ratio"] = facts["w"] / facts["h"]
    facts["mb"] = path.stat().st_size / 1_048_576

    if abs(facts["ratio"] - TARGET_RATIO) > RATIO_TOL:
        problems.append(f"ratio {facts['ratio']:.4f} != 9:16 ({TARGET_RATIO:.4f})")
    if facts["h"] < MIN_HEIGHT:
        problems.append(f"height {facts['h']}px < {MIN_HEIGHT}px floor")
    if facts["kind"] == "video":
        if facts["dur"] is None:
            problems.append("duration unreadable — cannot verify the 3-8s window")
        elif not (DUR_MIN <= facts["dur"] <= DUR_MAX):
            problems.append(f"duration {facts['dur']:.2f}s outside {DUR_MIN}-{DUR_MAX}s")
    return not problems, problems, facts


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv if Path(p).is_file()]
    if not paths:
        print("EXTRACTED 0 files — nothing checked. This is a bug report, not a PASS.")
        return 2

    print(f"EXTRACTED {len(paths)} file(s) to check against Canvas specs\n")
    print(f"{'file':<42}{'WxH':>12}{'ratio':>8}{'dur':>7}{'MB':>7}  verdict")
    print("-" * 92)
    failed = 0
    for p in sorted(paths):
        try:
            ok, problems, f = check(p)
        except Exception as e:                      # noqa: BLE001 - surface, never swallow
            print(f"{p.name[:41]:<42}{'ERROR':>12}  {type(e).__name__}: {e}")
            failed += 1
            continue
        dur = f"{f['dur']:.2f}s" if f.get("dur") else "  -  "
        print(f"{p.name[:41]:<42}{f['w']}x{f['h']:>6}{f['ratio']:>8.4f}{dur:>7}"
              f"{f['mb']:>7.1f}  {'PASS' if ok else 'FAIL'}")
        for pr in problems:
            print(f"{'':<42}  ⛔ {pr}")
        failed += (not ok)

    print("-" * 92)
    print(f"{len(paths) - failed}/{len(paths)} pass Canvas specs")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
