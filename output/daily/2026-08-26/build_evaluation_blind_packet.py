from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ICB = (RUN_DIR / "CREATIVE_CONTEXT.md").read_text(encoding="utf-8")
SALT = "2026-08-26-daily-blind-v1"
HEADING = re.compile(r"^# V([1-4])\s+[—-]\s+(.+?)\s*$", re.M)


records: list[dict[str, str | int]] = []
for pair in range(1, 7):
    nn = f"{pair:02d}"
    path = RUN_DIR / f"pair_{nn}_step10_final_package_enhanced.md"
    text = path.read_text(encoding="utf-8")
    if not text.startswith(ICB):
        raise SystemExit(f"ICB prefix mismatch: {path.name}")
    tail = text[len(ICB):]
    matches = list(HEADING.finditer(tail))
    if len(matches) != 4:
        raise SystemExit(f"expected four V headings in {path.name}; found {len(matches)}")
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(tail)
        variation = int(match.group(1))
        title = match.group(2).strip()
        body = tail[match.end():end].strip()
        # Remove pair/variation labels from sidecars while preserving all
        # renderable fields and the title itself.
        body = re.sub(rf"\bP{nn}\b", "PAIR", body, flags=re.I)
        body = re.sub(rf"\bV{variation}\b", "CANDIDATE", body)
        key = hashlib.sha256(f"{SALT}|{nn}|{variation}|{title}".encode()).hexdigest()
        records.append({
            "key": key,
            "pair": nn,
            "variation": variation,
            "title": title,
            "source": path.name,
            "body": body,
        })

records.sort(key=lambda item: str(item["key"]))
packet_parts = [
    "# 2026-08-26 Music Daily — Blind Evaluation Input",
    "",
    "Twenty-four Step-11 text packages in deterministic blind order. Pair and variation identity are withheld. No audio exists; judge only the stated song, production notation, lyric/form craft, and declared render risk.",
]
blind_map = []
for index, record in enumerate(records, start=1):
    blind_id = f"C{index:02d}"
    packet_parts.extend([
        "",
        f"# BLIND CANDIDATE {blind_id}",
        "",
        f"## 3. TITLE\n\n{record['title']}",
        "",
        str(record["body"]),
    ])
    blind_map.append({
        "blind_id": blind_id,
        "pair": record["pair"],
        "variation": record["variation"],
        "title": record["title"],
        "source": record["source"],
    })

(RUN_DIR / "EVALUATION_BLIND_INPUT.md").write_text(
    "\n".join(packet_parts).rstrip() + "\n",
    encoding="utf-8",
    newline="\n",
)
(RUN_DIR / ".evaluation_blind_map.json").write_text(
    json.dumps(blind_map, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
    newline="\n",
)
print(f"blind candidates: {len(records)}")
print("wrote EVALUATION_BLIND_INPUT.md and sealed map")
