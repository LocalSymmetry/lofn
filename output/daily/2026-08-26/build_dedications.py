"""Collect the measured dedication blocks from the six canonical finalists."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def main() -> None:
    packages = sorted((ROOT / "selected").glob("P*.md"))
    if len(packages) != 6:
        raise SystemExit(f"expected 6 selected packages, found {len(packages)}")

    parts = [
        "# Dedications — 2026-08-26 Music Daily",
        "",
        "Each dedication is copied verbatim from its canonical selected package and measured as characters of dedication body. Every entry is within the 500-character hard ceiling; expanded lineage lives in `CREDITS.md`.",
    ]
    for path in packages:
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].split("—", 1)[-1].strip()
        match = re.search(r"^## DEDICATION\s*$\n(.*?)(?=^## |\Z)", text, re.M | re.S)
        if not match:
            raise SystemExit(f"missing dedication: {path.name}")
        body = match.group(1).strip()
        if len(body) > 500:
            raise SystemExit(f"dedication over cap ({len(body)}): {path.name}")
        parts.extend(("", f"## {title}", "", body, "", f"*Measured length: {len(body)} characters.*"))

    (ROOT / "DEDICATIONS.md").write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print("DEDICATIONS.md: " + ", ".join(
        f"{path.name}"
        for path in packages
    ))


if __name__ == "__main__":
    main()
