from __future__ import annotations

import hashlib
from pathlib import Path


MUSIC = Path(__file__).resolve().parent
ICB = (MUSIC / "CREATIVE_CONTEXT.md").read_bytes()


def build(step: int) -> Path:
    body = MUSIC / f"_step{step:02d}_body.md"
    out = MUSIC / {
        0: "step00_aesthetics_and_genres.md",
        1: "step01_essence_and_facets.md",
        2: "step02_concepts.md",
        3: "step03_artist_and_critique.md",
        4: "step04_medium.md",
        5: "step05_refine_medium.md",
    }[step]
    payload = ICB + body.read_bytes()
    out.write_bytes(payload)
    if not out.read_bytes().startswith(ICB):
        raise SystemExit(f"ICB prefix failed: {out}")
    print(f"{out.name}: bytes={len(payload)} sha256={hashlib.sha256(payload).hexdigest()}")
    return out


if __name__ == "__main__":
    import sys

    steps = [int(value) for value in sys.argv[1:]] or list(range(6))
    for item in steps:
        build(item)

