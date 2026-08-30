from __future__ import annotations

from pathlib import Path


MUSIC = Path(__file__).resolve().parent
icb = (MUSIC / "CREATIVE_CONTEXT.md").read_bytes()

for path in sorted(MUSIC.glob("pair_??_step10_final_package_enhanced.md")):
    payload = path.read_bytes()
    if not payload.startswith(icb):
        path.write_bytes(icb + b"\n\n" + payload)
        print(f"prefixed {path.name}")
    else:
        print(f"already-prefixed {path.name}")
    if not path.read_bytes().startswith(icb):
        raise SystemExit(f"prefix repair failed: {path}")

