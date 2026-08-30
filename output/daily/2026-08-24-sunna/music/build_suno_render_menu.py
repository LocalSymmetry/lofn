from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


MUSIC = Path(__file__).resolve().parent
OUT = MUSIC / "suno_render"

SELECTIONS = [
    (1, 1, "P01V01", "look_with_your_eyes", "5334eaa297cbf8bdd3e77b984424c259194ae989f5df1d42df21a2f86eaf5ee5"),
    (2, 3, "P02V03", "leave_it_crooked", "ca4be5e80f33826b84b97219e18eb98b1861ff2e1b4b8c629b92aa21f598e87c"),
    (3, 1, "P03V01", "pretty_on_purpose", "8b03f585141ca1c9463b74fee3314dad198c40192c140777166771dcd64e72fe"),
    (4, 4, "P04V04", "again_before_credits", "ca5c8b5d3353eded81dbe13043d762b83c69c1ec066b839088abe5ba0ae3571d"),
    (5, 4, "P05V04", "scan_me_twice", "be0563368962c5b0fda0ec7ff511c33f501cfa9ad415d523ed898501324aa53b"),
    (6, 4, "P06V04", "high_tide_high_gloss", "08ee797f373acbfa0c96d32137f74088192a2a054dd98bb0f19e8ccbef2bf8d5"),
]


def selected_block(path: Path, variation: int) -> tuple[bytes, str]:
    raw = path.read_bytes()
    matches = list(re.finditer(
        rb"(?im)^#{1,4}[ \t]+variation[ \t]+([1-9])\b[^\r\n]*(?:\r?\n)", raw
    ))
    for index, match in enumerate(matches):
        if int(match.group(1)) != variation:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        # Pair-level execution/postamble begins after the final package and is
        # not part of the selected block.
        tail = raw[match.start():end]
        post = re.search(
            rb"(?im)^##[ \t]+(?:11\.[ \t]+Cross-Variation|"
            rb"8\.[ \t]+Step[ -]11[ \t]+Execution Log|Pair-Level[ \t]+Execution Log)",
            tail,
        )
        if post:
            tail = tail[:post.start()]
        return tail, tail.decode("utf-8")
    raise ValueError(f"{path.name}: variation {variation} not found")


def clean(text: str) -> str:
    text = re.sub(r"^```[^\n]*$", "", text, flags=re.M).strip()
    if text.startswith("`") and text.endswith("`") and "\n" not in text:
        text = text.strip("`").strip()
    return text


def field(block: str, start: str, end: str | None = None) -> str:
    stop = rf"(?=^##\s+{end}\s*$)" if end else r"(?=^##\s+|\Z)"
    match = re.search(rf"^##\s+{start}\s*$\n(?P<body>.*?){stop}", block, re.M | re.S | re.I)
    if not match:
        raise ValueError(f"missing field {start!r}")
    return clean(match.group("body"))


def named_section(block: str, name_pattern: str) -> str:
    heading = re.search(rf"^(?P<hash>##|###)\s+(?P<title>{name_pattern})\s*$", block, re.M | re.I)
    if not heading:
        raise ValueError(f"missing section {name_pattern!r}")
    level = len(heading.group("hash"))
    following = block[heading.end():]
    next_heading = re.search(rf"(?m)^#{{2,{level}}}\s+", following)
    body = following[:next_heading.start()] if next_heading else following
    return clean(body)


def dedication(block: str) -> str:
    return named_section(block, r"(?:5\.\s+)?DEDICATION(?:\s+DRAFT)?")


def lineage(block: str) -> str:
    return named_section(block, r"(?:4\.\s+)?LINEAGE(?:\s*&\s*CREDIT|\s*/\s*SOURCE NOTE)?")


def main() -> None:
    expected_names = {
        *(f"{index:02d}_{slug}.md" for index, (_p, _v, _c, slug, _h) in enumerate(SELECTIONS, start=1)),
        "INDEX.md",
        "build_manifest.json",
        "QA_REPORT.md",
        "P04_FINAL_RECHECK.md",
    }
    if OUT.exists():
        extras = {p.name for p in OUT.iterdir()} - expected_names
        if extras:
            raise FileExistsError(f"refusing to overwrite render menu with unexpected files: {sorted(extras)}")
    else:
        OUT.mkdir(parents=True)
    manifest = []
    dedication_rows = ["# SUNNA R4 — Dedications", ""]
    credit_rows = ["# SUNNA R4 — Lineage & Credits", ""]
    index_rows = [
        "# SUNNA R4 — Suno Render Menu",
        "",
        "**Status:** text packages only; no render, listening pass, upload, publication, or spend has occurred.",
        "",
    ]
    for ordinal, (pair, variation, candidate_id, slug, expected_hash) in enumerate(SELECTIONS, start=1):
        source = MUSIC / f"pair_{pair:02d}_step10_final_package_enhanced.md"
        raw_block, block = selected_block(source, variation)
        canonical_block = raw_block.replace(b"\r\n", b"\n")
        actual_hash = hashlib.sha256(canonical_block).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"{candidate_id}: frozen source hash changed: {actual_hash} != {expected_hash}")
        prompt = field(block, r"1\.\s+MUSIC PROMPT", r"1B\.\s+SUNO EXCLUDE PROMPT")
        exclude = field(block, r"1B\.\s+SUNO EXCLUDE PROMPT", r"2\.\s+LYRICS")
        lyrics = field(block, r"2\.\s+LYRICS", r"3\.\s+TITLE")
        title = field(block, r"3\.\s+TITLE")
        ded = dedication(block)
        cred = lineage(block)
        if len(ded) > 500:
            raise ValueError(f"{candidate_id}: dedication {len(ded)} chars exceeds 500")
        filename = f"{ordinal:02d}_{slug}.md"
        body = (
            f"# {title} — Suno Render Package\n\n"
            f"**Candidate:** {candidate_id} · **Status:** text-only finalist, not yet rendered or heard\n\n"
            f"## 1. MUSIC PROMPT\n\n{prompt}\n\n"
            f"## 1B. SUNO EXCLUDE PROMPT\n\n{exclude}\n\n"
            f"## 2. LYRICS\n\n{lyrics}\n\n"
            f"## 3. TITLE\n\n{title}\n\n"
            f"## 4. DEDICATION\n\n{ded}\n\n"
            f"## 5. LINEAGE & CREDIT\n\n{cred}\n"
        )
        (OUT / filename).write_text(body, encoding="utf-8", newline="\n")
        manifest.append({
            "ordinal": ordinal,
            "candidate_id": candidate_id,
            "title": title,
            "filename": filename,
            "source": str(source.relative_to(MUSIC.parent.parent.parent.parent)),
            "source_block_sha256": actual_hash,
            "music_prompt_chars": len(prompt),
            "exclude_chars": len(exclude),
            "lyrics_chars": len(lyrics),
            "dedication_chars": len(ded),
        })
        index_rows.append(f"{ordinal}. [{title}]({filename}) — `{candidate_id}`")
        dedication_rows.extend([f"## {ordinal}. {title}", "", ded, ""])
        credit_rows.extend([f"## {ordinal}. {title}", "", cred, ""])
    (OUT / "INDEX.md").write_text("\n".join(index_rows) + "\n", encoding="utf-8", newline="\n")
    (MUSIC / "DEDICATIONS.md").write_text("\n".join(dedication_rows), encoding="utf-8", newline="\n")
    (MUSIC / "CREDITS.md").write_text("\n".join(credit_rows), encoding="utf-8", newline="\n")
    (OUT / "build_manifest.json").write_text(
        json.dumps({"status": "TEXT_ONLY_NOT_RENDERED", "selections": manifest}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
