#!/usr/bin/env python3
"""Deterministic lyric extraction and narrow cohesion proxies for the preregistered cohort.

These proxies do not decide artistic quality. They prepare a reproducible evidence packet
for a clean-context blind reader and the source-grounded panel.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "hers", "him", "his", "i", "if", "in", "into",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "so", "that",
    "the", "their", "them", "they", "this", "those", "to", "too", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your", "yours",
}

CARRY_OPENERS = {
    "and", "but", "so", "because", "then", "therefore", "yet", "still", "now", "instead",
    "this", "that", "these", "those", "it", "its", "they", "them", "their", "she", "her",
    "he", "him", "his", "we", "our", "i", "my", "you", "your", "here", "there", "again",
}

AUX_VERBS = {
    "am", "are", "be", "been", "being", "can", "can't", "cannot", "could", "did", "didn't",
    "do", "does", "doesn't", "don't", "had", "has", "have", "haven't", "is", "isn't", "may",
    "might", "must", "shall", "should", "was", "wasn't", "were", "weren't", "will", "won't",
    "would", "wouldn't",
}

COMMON_VERBS = {
    "ask", "asks", "asked", "answer", "answers", "answered", "bend", "bent", "bite", "bites",
    "break", "breaks", "bring", "brings", "call", "calls", "carry", "carries", "change", "changes",
    "choose", "chooses", "close", "closes", "come", "comes", "cut", "cuts", "dance", "dances",
    "drop", "drops", "feel", "feels", "fill", "fills", "find", "finds", "give", "gives", "go",
    "goes", "got", "get", "gets", "hear", "hears", "hold", "holds", "keep", "keeps", "kept",
    "know", "knows", "land", "lands", "lean", "leans", "leave", "leaves", "let", "lets", "lift",
    "lifts", "look", "looks", "lose", "loses", "love", "loves", "make", "makes", "made", "mean",
    "means", "miss", "misses", "move", "moves", "name", "names", "need", "needs", "open", "opens",
    "press", "presses", "put", "puts", "refuse", "refuses", "remember", "remembers", "return",
    "returns", "run", "runs", "say", "says", "see", "sees", "set", "sets", "shake", "shakes",
    "sit", "sits", "stand", "stands", "stay", "stays", "step", "steps", "stop", "stops", "take",
    "takes", "teach", "teaches", "tell", "tells", "think", "thinks", "turn", "turns", "wait",
    "waits", "walk", "walks", "want", "wants", "watch", "watches", "write", "writes", "wrote",
}

ABSTRACT = {
    "answer", "cause", "choice", "decision", "exception", "fact", "fault", "fear", "future", "grief",
    "hope", "idea", "lesson", "light", "love", "meaning", "memory", "name", "nothing", "promise",
    "purpose", "question", "reason", "recognition", "rule", "something", "time", "truth", "want",
}

ACTION = {
    "ask", "answer", "bend", "bite", "break", "bring", "call", "carry", "change", "choose", "close",
    "come", "cut", "dance", "drop", "fill", "find", "give", "go", "hold", "keep", "land", "lean",
    "leave", "lift", "look", "lose", "make", "miss", "move", "name", "open", "press", "put", "refuse",
    "return", "run", "say", "set", "shake", "sit", "stand", "stay", "step", "stop", "take", "teach",
    "tell", "turn", "wait", "walk", "watch", "write",
}

ENGINE_TERMS = {
    "answer slot", "byte-identical", "call and answer", "call-answer", "cause missing", "checksum",
    "consonant", "delayed premise", "exact chorus", "exact return", "grammar", "inversion", "missing cause",
    "missing grammar", "owner switch", "rekey", "rhyme debt", "semantic", "self-amending", "timbral rhyme",
    "vowel migration", "zeugma",
}


def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8", errors="replace")


def clean_heading(text: str) -> str:
    return re.sub(r"[*_`]", "", text).strip().strip("—·- ")


def extract_lyrics(block: str) -> tuple[str, str, str]:
    """Return (lyrics payload, theme, song form) from one package block."""
    lines = block.splitlines()
    start = None
    level = None
    for i, line in enumerate(lines):
        bare_label = clean_heading(line).upper()
        if line.strip().startswith("**") and (bare_label.startswith("LYRICS") or bare_label.startswith("③ LYRICS")):
            start = i + 1
            level = 6
            break
        m = HEADING_RE.match(line)
        if not m:
            continue
        label = clean_heading(m.group(2)).upper()
        if "LYRICS" in label and not label.startswith(("CLEAN LYRICS", "LYRICS FIELD")):
            start = i + 1
            level = len(m.group(1))
            break
    if start is None:
        raise ValueError("lyrics heading not found")
    first_content = start
    while first_content < len(lines) and not lines[first_content].strip():
        first_content += 1
    if first_content < len(lines) and lines[first_content].strip().startswith("```"):
        fence_end = next(
            (j for j in range(first_content + 1, len(lines)) if lines[j].strip().startswith("```")),
            None,
        )
        if fence_end is None:
            raise ValueError("lyrics code fence is not closed")
        payload = "\n".join(lines[first_content + 1:fence_end]).strip()
        theme_m = re.search(r"\[Theme:\s*(.*?)\]", payload, flags=re.I | re.S)
        form_m = re.search(r"\[SONG FORM:\s*(.*?)\]", payload, flags=re.I | re.S)
        return payload, (theme_m.group(1).strip() if theme_m else ""), (form_m.group(1).strip() if form_m else "")
    end = len(lines)
    for j in range(start, len(lines)):
        m = HEADING_RE.match(lines[j])
        if m and len(m.group(1)) <= int(level):
            end = j
            break
    payload_lines = lines[start:end]
    while payload_lines and payload_lines[0].strip().startswith("```"):
        payload_lines.pop(0)
    while payload_lines and payload_lines[-1].strip().startswith("```"):
        payload_lines.pop()
    payload = "\n".join(payload_lines).strip()
    theme_m = re.search(r"\[Theme:\s*(.*?)\]", payload, flags=re.I | re.S)
    form_m = re.search(r"\[SONG FORM:\s*(.*?)\]", payload, flags=re.I | re.S)
    return payload, (theme_m.group(1).strip() if theme_m else ""), (form_m.group(1).strip() if form_m else "")


def split_at_matches(text: str, pattern: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(pattern, text, flags=re.M | re.I))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((clean_heading(m.group(1)), text[m.start():end]))
    return out


def song_lines(payload: str) -> list[str]:
    out = []
    for raw in payload.splitlines():
        line = raw.strip()
        if not line or line.startswith("```") or line.startswith("[") or line.startswith("*"):
            continue
        if line.startswith(("#", "---")):
            continue
        # Remove common Markdown quotation/list decoration without deleting sung punctuation.
        line = re.sub(r"^(?:>|[-+]\s+)", "", line).strip()
        if line:
            out.append(line)
    return out


def words(line: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in WORD_RE.findall(line)]


def stem(w: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ed", "ies", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def has_verb(ws: list[str]) -> bool:
    if any(w in AUX_VERBS or w in COMMON_VERBS for w in ws):
        return True
    return any(len(w) > 5 and w.endswith(("ed", "ing")) for w in ws)


def engine_count(theme: str, form: str) -> int:
    text = f"{theme} {form}".lower()
    found = {term for term in ENGINE_TERMS if term in text}
    # Named form stages are a separate load signal, but capped so a slash-heavy roadmap
    # does not dominate the semantic-device count.
    stages = min(3, max(0, text.count("→") + text.count(" / ") - 1))
    return len(found) + stages


def metrics(rec: dict) -> dict:
    lines = rec["lines"]
    token_lines = [words(x) for x in lines]
    all_words = [w for row in token_lines for w in row]
    n = len(lines)
    pair_n = max(0, n - 1)
    carries = 0
    for a, b in zip(token_lines, token_lines[1:]):
        ca = {stem(w) for w in a if w not in STOP and len(w) > 2}
        cb = {stem(w) for w in b if w not in STOP and len(w) > 2}
        opener = b[0] if b else ""
        if ca & cb or opener in CARRY_OPENERS:
            carries += 1
    normalized = [" ".join(row) for row in token_lines if row]
    repeats = len(normalized) - len(set(normalized))
    fragment_flags = [not has_verb(row) for row in token_lines]
    list_members = [False] * n
    run_start = 0
    for i in range(n + 1):
        if i == n or not fragment_flags[i]:
            if i - run_start >= 3:
                for j in range(run_start, i):
                    list_members[j] = True
            run_start = i + 1
    abstract_hits = sum(w in ABSTRACT for w in all_words)
    action_hits = sum(stem(w) in ACTION for w in all_words)
    section_count = sum(1 for line in rec["payload"].splitlines() if line.strip().startswith("[") and "EMO:" in line)
    return {
        "cohort": rec["cohort"],
        "run": rec["run"],
        "title": rec["title"],
        "source": rec["source"],
        "sung_lines": n,
        "words": len(all_words),
        "words_per_line": round(len(all_words) / n, 3) if n else None,
        "ultrashort_rate": round(sum(len(row) <= 3 for row in token_lines) / n, 4) if n else None,
        "fragment_proxy_rate": round(sum(fragment_flags) / n, 4) if n else None,
        "list_run_rate": round(sum(list_members) / n, 4) if n else None,
        "adjacent_carry_proxy": round(carries / pair_n, 4) if pair_n else None,
        "exact_repeat_ratio": round(repeats / n, 4) if n else None,
        "abstract_lexicon_per_100w": round(100 * abstract_hits / len(all_words), 3) if all_words else None,
        "action_lexicon_per_100w": round(100 * action_hits / len(all_words), 3) if all_words else None,
        "section_count": section_count,
        "engine_proxy_count": engine_count(rec.get("theme", ""), rec.get("form", "")),
    }


def package_record(cohort: str, run: str, title: str, source: str, block: str) -> dict:
    try:
        payload, theme, form = extract_lyrics(block)
    except ValueError as exc:
        raise ValueError(f"{exc}: run={run!r} title={title!r} source={source!r}") from exc
    return {
        "cohort": cohort,
        "run": run,
        "title": title,
        "source": source,
        "payload": payload,
        "theme": theme,
        "form": form,
        "lines": song_lines(payload),
    }


def high_bar() -> list[dict]:
    out = []
    triple_source = "output/distribution/LYRICS/SIDE_A/A17 - Triple Arch Over Me.txt"
    triple_payload = read(triple_source).strip()
    out.append({
        "cohort": "A_HIGH_BAR", "run": "2026-04-22", "title": "Triple Arch Over Me",
        "source": triple_source, "payload": triple_payload, "theme": "", "form": "",
        "lines": song_lines(triple_payload),
    })
    for title, source, run in [
        ("Spare Hair Tie", "output/songs/20260809_the-spare-hair-tie_P04_V2.md", "2026-08-09"),
        ("Burning Bush With a Barcode", "output/songs/2026-07-06_P05V1_burning-bush-with-a-barcode.md", "2026-07-06"),
        ("The Wedding Takes It Louder", "output/songs/20260821_the_wedding_takes_it_louder_P06_V2.md", "2026-08-21"),
    ]:
        out.append(package_record("A_HIGH_BAR", run, title, source, read(source)))
    ask_source = "output/daily/2026-08-17-dialtone2/SUNO_GUIDES/DIAL_TONE_2_TOP_SIX.md"
    ask_text = read(ask_source)
    start = ask_text.index("## AMBITIOUS · “ASK FOR IT BY NAME”")
    ask_block = ask_text[start:]
    out.append(package_record("A_HIGH_BAR", "2026-08-17", "ASK FOR IT BY NAME", ask_source, ask_block))
    return out


def criticized_batch() -> list[dict]:
    base = REPO / "output/daily/2026-08-29-lofn-cleverness-unbound/selected"
    out = []
    for path in sorted(base.glob("[0-9][0-9]_*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', text, flags=re.M)
        title = m.group(1).strip().strip('"\'') if m else clean_heading(next(x for x in text.splitlines() if x.startswith("# "))[2:])
        out.append(package_record("B_2026_08_29", "2026-08-29", title, path.relative_to(REPO).as_posix(), text))
    return out


TIMELINE_COMBINED = [
    ("2026-08-21", "output/daily/2026-08-21/TOP6_PASTE_READY.md", r"^## (?:ACCESSIBLE|AMBITIOUS) #\d+ — \*\*(.+?)\*\*"),
    ("2026-08-22", "output/daily/2026-08-22-lofn-prime-every-element/TOP6_PASTE_READY.md", r"^# P\d+V\d+ — (.+)$"),
    ("2026-08-26", "output/daily/2026-08-26-opus-masterpiece/SUNO_PACKAGES.md", r"^### \d+ · (.+)$"),
    ("2026-08-27", "output/daily/2026-08-27-high-energy-registers/SUNO_PACKAGES.md", r"^## \d+ · (.+)$"),
    ("2026-08-28", "output/daily/2026-08-28-three-act-mania/SUNO_PACKAGES.md", r"^# \d+\. (.+)$"),
]


def timeline() -> list[dict]:
    out = []
    for run, source, pattern in TIMELINE_COMBINED:
        text = read(source)
        chunks = split_at_matches(text, pattern)
        for title, block in chunks:
            out.append(package_record("C_TIMELINE", run, title, source, block))

    for run, directory, glob_pattern in [
        ("2026-08-24", "output/daily/2026-08-24/music/suno_render", "[0-9][0-9]_*.md"),
        ("2026-08-25", "output/daily/2026-08-25-sunna-modern-swagger/music/selected", "*.md"),
        ("2026-08-29", "output/daily/2026-08-29-lofn-cleverness-unbound/selected", "[0-9][0-9]_*.md"),
    ]:
        for path in sorted((REPO / directory).glob(glob_pattern)):
            text = path.read_text(encoding="utf-8", errors="replace")
            m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', text, flags=re.M)
            if m:
                title = m.group(1).strip().strip('"\'')
            else:
                candidates = [clean_heading(x[2:]) for x in text.splitlines() if x.startswith("# ") and "BEGIN CREATIVE" not in x]
                title = candidates[-1] if candidates else path.stem
            out.append(package_record("C_TIMELINE", run, title, path.relative_to(REPO).as_posix(), text))
    return out


def median(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.median(vals), 4) if vals else None


def main() -> None:
    primary = high_bar() + criticized_batch()
    timeline_rows = timeline()
    if len(high_bar()) != 5 or len(criticized_batch()) != 6:
        raise SystemExit("primary cohort cardinality mismatch")
    timeline_counts = {}
    for r in timeline_rows:
        timeline_counts[r["run"]] = timeline_counts.get(r["run"], 0) + 1
    expected_runs = {"2026-08-21", "2026-08-22", "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28", "2026-08-29"}
    if set(timeline_counts) != expected_runs or any(n != 6 for n in timeline_counts.values()):
        raise SystemExit(f"timeline cardinality mismatch: {timeline_counts}")

    all_records = primary + timeline_rows
    measured = [metrics(r) for r in all_records]
    fields = list(measured[0])
    with (HERE / "MEASUREMENTS.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(measured)
    (HERE / "CORPUS.json").write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")

    # Deterministically shuffled blind packet. Titles, sources, dates, and cohort labels are withheld.
    rng = random.Random(20260830)
    shuffled = list(primary)
    rng.shuffle(shuffled)
    key = []
    packet = [
        "# Blind Reader Packet — Cohesion Regression",
        "",
        "For each coded song, judge only the lyric text. Do not infer cohort from style or formatting.",
        "Report: literal situation, singer's present want, hard-cut estimate (0–100%), five compulsion signals,",
        "and whether fragments/lists form pressure or disconnection. Cite exact lines. Do not rank until all are read.",
        "",
    ]
    for i, rec in enumerate(shuffled, 1):
        code = f"S{i:02d}"
        key.append({"code": code, "cohort": rec["cohort"], "title": rec["title"], "source": rec["source"]})
        packet.extend([f"## {code}", "", "```text", *rec["lines"], "```", ""])
    (HERE / "BLIND_READER_PACKET.md").write_text("\n".join(packet), encoding="utf-8")
    (HERE / "BLIND_KEY.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    a = [r for r in measured if r["cohort"] == "A_HIGH_BAR"]
    b = [r for r in measured if r["cohort"] == "B_2026_08_29"]
    metric_keys = [
        "sung_lines", "words_per_line", "ultrashort_rate", "fragment_proxy_rate", "list_run_rate",
        "adjacent_carry_proxy", "exact_repeat_ratio", "abstract_lexicon_per_100w",
        "action_lexicon_per_100w", "section_count", "engine_proxy_count",
    ]
    summary = {
        "corpus_hash": hashlib.sha256(json.dumps(all_records, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "primary_n": {"A_HIGH_BAR": len(a), "B_2026_08_29": len(b)},
        "timeline_n": timeline_counts,
        "medians": {
            "A_HIGH_BAR": {k: median(a, k) for k in metric_keys},
            "B_2026_08_29": {k: median(b, k) for k in metric_keys},
        },
        "cautions": [
            "All deterministic continuity/fragment/list/abstract metrics are narrow proxies, not verdicts.",
            "The blinded reader and panel must retain counterexamples and may reject proxy validity.",
            "Audio claims remain out of scope without the recent renders.",
        ],
    }
    (HERE / "MEASUREMENT_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
