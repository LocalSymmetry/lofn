"""Build one copy-ready Suno bundle from the six canonical selected packages."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def extract(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"missing section: {heading}")
    return match.group(1).strip()


def main() -> None:
    packages = sorted((ROOT / "selected").glob("P*.md"))
    if len(packages) != 6:
        raise SystemExit(f"expected 6 selected packages, found {len(packages)}")

    parts = [
        "# Copy-Ready Suno Prompts — 2026-08-26 Music Daily",
        "",
        "Paste each fenced block into Suno's matching field: Style, Exclude Styles, and Lyrics. The fields below are copied verbatim from the QA-cleared canonical packages; no artist names appear in generator-facing text.",
    ]

    for index, path in enumerate(packages, 1):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].split("—", 1)[-1].strip()
        style = extract(text, "SUNO STYLE PROMPT")
        exclude = extract(text, "SUNO EXCLUDE PROMPT")
        lyrics = extract(text, "SUNO ENHANCED LYRICS")
        parts.extend(
            (
                "",
                "---",
                "",
                f"## {index}. {title}",
                "",
                f"*Style: {len(style)} chars · Exclude: {len(exclude)} chars · Lyrics: {len(lyrics)} chars*",
                "",
                "### Style",
                "",
                "```text",
                style,
                "```",
                "",
                "### Exclude Styles",
                "",
                "```text",
                exclude,
                "```",
                "",
                "### Lyrics",
                "",
                "```text",
                lyrics,
                "```",
            )
        )

    target = ROOT / "SUNO_PROMPTS.md"
    target.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"{target.name}: {len(packages)} songs, {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
