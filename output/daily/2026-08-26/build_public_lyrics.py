"""Build the reader-facing lyric booklet from the six canonical selected packages."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def public_lyrics(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"^## SUNO ENHANCED LYRICS\s*$\n(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"missing lyric field: {path.name}")

    out = []
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if re.match(r"^\[(?:Theme|SONG FORM|Disc_[^:]+):", line, re.IGNORECASE):
            continue
        header = re.match(r"^\[([^\]]+?)\]$", line)
        if header:
            # The canonical lyric packages use hyphen, en dash, and em dash
            # separators across generations. Public lyrics keep only the
            # human-readable section name and remove all EMO/performance tails.
            name = re.split(
                r"\s+[\-–—]\s+EMO:",
                header.group(1),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            out.extend(("", f"### {name}", ""))
            continue
        out.append(raw.rstrip())
    return "\n".join(out).strip()


def main() -> None:
    selected = sorted((ROOT / "selected").glob("P*.md"))
    if len(selected) != 6:
        raise SystemExit(f"expected 6 selected packages, found {len(selected)}")

    parts = [
        "# Public Lyrics — 2026-08-26 Music Daily",
        "",
        "Reader-facing lyric sheets. Generator instructions, EMO tags, channel notation, and production ledgers are intentionally omitted; the canonical Suno fields remain in `selected/`.",
    ]
    for path in selected:
        first = path.read_text(encoding="utf-8").splitlines()[0]
        title = first.split("—", 1)[-1].strip() if "—" in first else first.lstrip("# ")
        parts.extend(("", "---", "", f"## {title}", "", public_lyrics(path)))

    (ROOT / "PUBLIC_LYRICS.md").write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"PUBLIC_LYRICS.md: {len(selected)} songs")


if __name__ == "__main__":
    main()
