from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LIVE = ROOT / "vault/COMPETITION_LEARNINGS.md"
ARCHIVE = ROOT / "vault/COMPETITION_LEARNINGS_ARCHIVE.md"
ROW_RE = re.compile(r"^\| (L\d+) \|.*$", re.M)


def rows(text: str) -> dict[str, str]:
    return {match.group(1): match.group(0) for match in ROW_RE.finditer(text)}


def main() -> None:
    previous = subprocess.check_output(
        ["git", "show", "HEAD:vault/COMPETITION_LEARNINGS.md"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    current = LIVE.read_text(encoding="utf-8")
    old_rows = rows(previous)
    current_rows = rows(current)
    removed = [old_rows[key] for key in old_rows if key not in current_rows]
    if not removed:
        print("no pruned rows to archive")
        return

    if ARCHIVE.exists():
        archive_text = ARCHIVE.read_text(encoding="utf-8")
    else:
        archive_text = (
            "# Competition Learnings Archive\n\n"
            "Rows pruned from the live ~25-entry advisory ledger are preserved here verbatim. "
            "Archived rows are historical evidence, not active dispatch constraints.\n\n"
            "| # | theme-type | venue | modality | verdict | confidence | transferable rule (ADVISORY) |\n"
            "|---|-----------|-------|----------|---------|-----------|------------------------------|\n"
        )
    archived_ids = set(rows(archive_text))
    additions = [row for row in removed if re.match(r"^\| (L\d+) ", row).group(1) not in archived_ids]
    if additions:
        ARCHIVE.write_text(
            archive_text.rstrip() + "\n" + "\n".join(additions) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(f"archived={len(additions)} preserved_total={len(rows(ARCHIVE.read_text(encoding='utf-8')))}")


if __name__ == "__main__":
    main()
