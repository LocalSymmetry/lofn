#!/usr/bin/env python3
"""skadi_ingest.py — bind rendered clips to the probes that asked for them.

Matches each probe's package (its `## 1. MUSIC PROMPT` block) against the style
prompt text (`tags`) of the rows in a Suno snapshot, then optionally pulls the
WAVs and measures them. Writes `<expedition-dir>/RENDERS.json`.

⛔ A WRONG MATCH IS WORSE THAN NO MATCH. The whole atlas rests on which render
answered which ask, so an ambiguous match and a zero match are both hard ERRORS
that list their candidates and stop. They are resolved by a human with
`--map <probe_id>=<clip_id>`, never by the matcher picking the best of a bad
field. The clip id is the only durable key; filenames and titles are not.

⛔ AUTO-MATCH IS EXACT NORMALIZED EQUALITY, AND NOTHING ELSE. Containment and
similarity are CANDIDATE SCORERS for the error message; neither can ever bind.
The reason is the plate's own design: SKILL.md Phase 1 prescribes the variation
structure `V1 absent · V2 standard · V3 pushed · V4 crossed`, and "pushed" is
routinely "V2 plus one more clause" — so V2's prompt is a strict SUBSTRING of
V3's, and scores ~0.7 similar to it. Under containment (or under a similarity
floor) a V2 that was never rendered binds silently to V3's clip, with
`matched_by` reading like a certainty. The plate then reports two observations
of one audio file. Found by the tester, 2026-08-17; the same collision is caught
again downstream by `skadi_bank.py` and by `check_ledger_schema`, because this
one is worth three layers.

⛔ ONE CLIP ANSWERS AT MOST ONE PROBE, AND ONE PROBE HAS AT MOST ONE CLIP. The
uniqueness invariant is enforced over the whole plate AFTER matching, `--map`
included — a hand-written map is exactly as capable of binding two probes to one
render as a matcher is.

PACKAGE CONVENTION (pinned by `skills/music/scripts/validate_suno_packages.py`,
never read off the artifacts): a probe's package is a markdown file under the
expedition directory whose filename begins with the probe id, containing exactly
one `## 1. MUSIC PROMPT` section. Zero or several is an ERROR, not a guess.

`--fetch` reuses `scripts/fetch_suno_wavs.py` (idempotent by Content-Length,
`.part` writes, manifest-is-the-record). `--measure` calls
`scripts/measure_render.py::audit()`. Both are imported, never shelled out, and
`measure_render` is imported LAZILY so ingest works on a machine with no
numpy/soundfile.

⛔ THE MANIFEST IS READ BEFORE THE NETWORK, AND `--measure` STANDS ALONE. Any WAV
already on disk — recorded in `wav/manifest.json` at its byte length, or simply
named with the clip id the way `fetch_suno_wavs.safe_stem` names it — is bound
with no request of any kind. So `--measure` works on a machine that is offline,
or on a clip whose CDN object has aged out, and a failed HEAD can no longer
delete work already paid for.

USAGE
    python3 scripts/skadi_ingest.py output/expeditions/<dir>
    python3 scripts/skadi_ingest.py <dir> --map P3=abc-123 --fetch --measure
    python3 scripts/skadi_ingest.py <dir> --measure        # offline, wavs already pulled

EXIT   0 all probes bound · 1 one or more probes unresolved · 2 input error
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_music_atlas import (  # noqa: E402
    PLAN_COLS, REPO, load_gates, parse_table, read,
)

DEFAULT_SNAPSHOT = "data/snapshots/suno/latest.json"

# How many scored candidates an unresolved probe prints. NOT a gate: a report
# width, not a threshold that governs a claim.
# SPEC-GAP: DESIGN v1 §4 fixed the matching STRATEGY as "exact-substring then
# similarity" and named no cutoff. Both of those rungs are now SCORERS only —
# see the module docstring — so no cutoff exists to name.
CANDIDATES_SHOWN = 5

PROMPT_RE = re.compile(
    r"^## 1\. MUSIC PROMPT\s*\n(?P<body>.*?)(?=^## 1B\. SUNO EXCLUDE PROMPT\s*$|"
    r"^## 2\. LYRICS\s*$|\Z)", re.M | re.S)
PROMPT_HEADING_RE = re.compile(r"^## 1\. MUSIC PROMPT\s*$", re.M)


def normalize(text: str) -> str:
    """Case-folded, punctuation-flattened, whitespace-collapsed."""
    text = re.sub(r"[`*_>#]+", " ", text or "")
    text = re.sub(r"[^0-9a-z]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def probe_ids(expedition: Path) -> list[str]:
    rows, problems = parse_table(read(expedition / "PLAN.md"), PLAN_COLS)
    if problems or not rows:
        raise SystemExit(f"ERROR: {expedition / 'PLAN.md'} unreadable: "
                         f"{'; '.join(problems) or 'no slots'}")
    return [r["probe_id"] for r in rows]


def find_packages(expedition: Path, pid: str) -> list[Path]:
    hits = []
    for p in sorted(expedition.rglob("*.md")):
        if not re.match(rf"^{re.escape(pid)}(?![0-9A-Za-z])", p.name):
            continue
        if PROMPT_HEADING_RE.search(read(p)):
            hits.append(p)
    return hits


def probe_prompts(expedition: Path, ids: list[str]) -> tuple[dict[str, tuple[Path, str]], list[str]]:
    """-> ({probe_id: (package_path, music_prompt)}, errors). Cardinality asserted."""
    out: dict[str, tuple[Path, str]] = {}
    errors: list[str] = []
    for pid in ids:
        hits = find_packages(expedition, pid)
        if not hits:
            errors.append(f"{pid}: no package found — expected a markdown file under "
                          f"{expedition.name}/ whose name starts with {pid!r} and which "
                          "contains exactly one '## 1. MUSIC PROMPT' section")
            continue
        if len(hits) > 1:
            errors.append(f"{pid}: {len(hits)} candidate packages — "
                          + ", ".join(h.name for h in hits))
            continue
        text = read(hits[0])
        n_headings = len(PROMPT_HEADING_RE.findall(text))
        if n_headings != 1:
            errors.append(f"{pid}: {hits[0].name} holds {n_headings} '## 1. MUSIC PROMPT' "
                          "heading(s); the contract is exactly one")
            continue
        body = PROMPT_RE.search(text)
        prompt = (body.group("body") if body else "").strip()
        if not prompt:
            errors.append(f"{pid}: {hits[0].name} has an EMPTY MUSIC PROMPT — an empty "
                          "extraction is an ERROR, never a match of zero")
            continue
        out[pid] = (hits[0], prompt)
    return out, errors


def load_snapshot(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"ERROR: snapshot absent at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") if isinstance(data, dict) else data
    if not rows:
        raise SystemExit(f"ERROR: snapshot {path} holds 0 rows — an empty extraction is an "
                         "ERROR, never a matcher that finds nothing")
    return rows


def candidates_for(want: str, rows: list[dict]) -> list[str]:
    """The near-misses, ranked, FOR THE HUMAN WHO WILL RESOLVE THIS BY HAND.

    Similarity and containment live here and only here. A scorer that cannot
    bind cannot mis-bind, and it is free to be as generous as it likes — its
    output is a list to read, not a decision.
    """
    scored = sorted(
        ((difflib.SequenceMatcher(None, want, t).ratio(), t, r)
         for r in rows if (t := normalize(r.get("tags") or ""))),
        key=lambda triple: (-triple[0], triple[2]["id"]))
    if not scored:
        return ["snapshot holds no row with a non-empty `tags` field"]
    out = []
    for score, tags, r in scored[:CANDIDATES_SHOWN]:
        if tags == want:
            how = "IDENTICAL text"
        elif tags in want:
            how = "its style text is CONTAINED IN the probe's"
        elif want in tags:
            how = "CONTAINS the probe's style text"
        else:
            how = "no containment"
        out.append(f"{r['id']}  score={score:.3f}  {how}  {(r.get('title') or '')[:40]}")
    return out


def match_probe(prompt: str, rows: list[dict]) -> tuple[str | None, str, str, list[str]]:
    """-> (clip_id or None, matched_by, reason, candidate descriptions).

    ⛔ EXACT NORMALIZED EQUALITY IS THE ONLY THING THAT BINDS. `reason` is
    `AMBIGUOUS` (several clips carry that exact text) or `NO` (none does), and
    both are resolved by a human with `--map`.
    """
    want = normalize(prompt)
    if not want:
        return None, "", "NO", ["probe prompt normalized to the empty string"]

    exact = [r for r in rows if normalize(r.get("tags") or "") == want]
    if len(exact) == 1:
        return exact[0]["id"], "exact", "", []
    if len(exact) > 1:
        return None, "", "AMBIGUOUS", [
            f"{r['id']}  IDENTICAL text  {(r.get('title') or '')[:40]}" for r in exact]
    return None, "", "NO", candidates_for(want, rows)


def parse_map(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"ERROR: --map expects probe_id=clip_id, got {pair!r}")
        pid, cid = pair.split("=", 1)
        out[pid.strip()] = cid.strip()
    return out


def manifest_entries(wav_dir: Path) -> dict[str, dict]:
    """`{clip_id: manifest row}` — the record of what has already been pulled."""
    path = wav_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {e["id"]: e for e in prior.get("files", []) if isinstance(e, dict) and e.get("id")}


def recorded_wav(entry: dict, wav_dir: Path, cid: str) -> Path | None:
    """The file this manifest row records, IF it is on disk at the recorded length.

    ⛔ THE MANIFEST IS THE RECORD, AND THE RECORD IS READABLE OFFLINE. A wav that
    was downloaded, verified and written to the manifest is bound from the
    manifest — with no HEAD, no request, no network of any kind. The old path
    asked the CDN first and `continue`d past the file when the request failed,
    so a machine holding every byte it needed could not re-bind them; being
    offline (or a 403 on a clip whose object has aged out) deleted work already
    paid for.
    """
    size = entry.get("bytes")
    if not isinstance(size, int) or size <= 0:
        return None
    rel = entry.get("path")
    candidates = []
    if isinstance(rel, str) and rel:
        candidates.append(REPO / rel)
    candidates.extend(sorted(wav_dir.glob(f"*__{cid[:8]}.wav")))
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size == size:
                return candidate
        except OSError:
            continue
    return None


def existing_wavs(clip_ids: list[str], wav_dir: Path) -> dict[str, Path]:
    """Bind clips to WAVs ALREADY on disk. Never touches the network.

    This is what makes `--measure` usable without `--fetch`: measurement reads a
    file, so requiring a download step to reach a file that is already there made
    the offline half of the instrument unreachable.
    """
    if not wav_dir.is_dir():
        return {}
    entries = manifest_entries(wav_dir)
    out: dict[str, Path] = {}
    for cid in clip_ids:
        found = recorded_wav(entries.get(cid) or {}, wav_dir, cid)
        if found is None:
            # No manifest row (or a stale one): fall back to the durable key in
            # the filename, which `fetch_suno_wavs.safe_stem` puts there.
            hits = sorted(p for p in wav_dir.glob(f"*__{cid[:8]}.wav") if p.is_file())
            found = hits[0] if len(hits) == 1 else None
        if found is not None:
            out[cid] = found
    return out


def fetch_wavs(clip_ids: list[str], rows: list[dict], out_dir: Path,
               snapshot_rel: str) -> dict[str, Path]:
    """Reuse the repaired downloader rather than re-deriving its idempotence."""
    import fetch_suno_wavs as F  # imported here so ingest runs without network intent

    by_id = {r["id"]: r for r in rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    entries: dict[str, dict] = manifest_entries(out_dir)

    paths: dict[str, Path] = {}
    for cid in clip_ids:
        row = by_id.get(cid, {"id": cid, "title": None})
        dest = out_dir / f"{F.safe_stem(row.get('title'), cid)}.wav"
        # ⛔ THE MANIFEST IS CONSULTED BEFORE THE NETWORK, NOT AFTER IT.
        have = recorded_wav(entries.get(cid) or {}, out_dir, cid)
        if have is not None:
            print(f"[fetch] {cid} skip (manifest records {have.name} at "
                  f"{entries[cid]['bytes']:,} bytes; no request made)")
            paths[cid] = have
            continue
        try:
            status, expected = F.head(cid)
        except Exception as err:  # noqa: BLE001 — recorded, never silently dropped
            print(f"[fetch] {cid} HEAD failed: {type(err).__name__}: {err}", file=sys.stderr)
            if not entries.get(cid, {}).get("sha256"):
                entries[cid] = {"id": cid, "title": row.get("title"),
                                "error": f"{type(err).__name__}: {err}"}
            continue
        if dest.exists() and expected is not None and dest.stat().st_size == expected \
                and entries.get(cid, {}).get("sha256"):
            print(f"[fetch] {cid} skip (have)")
            paths[cid] = dest
            continue
        try:
            result = F.download(cid, dest, expected)
        except Exception as err:  # noqa: BLE001
            print(f"[fetch] {cid} FAILED: {type(err).__name__}: {err}", file=sys.stderr)
            entries[cid] = {"id": cid, "title": row.get("title"),
                            "error": f"{type(err).__name__}: {err}"}
            continue
        entries[cid] = {"id": cid, "title": row.get("title"),
                        "path": str(dest.relative_to(REPO)).replace("\\", "/"),
                        "created_at": row.get("created_at"), "duration": row.get("duration"),
                        **result}
        paths[cid] = dest
        print(f"[fetch] {cid} ok  {result['bytes'] / 1e6:.1f} MB")
    F.write_manifest(manifest_path, entries, snapshot_rel)
    return paths


def measure(wav: Path) -> dict:
    """Lazy import: `measure_render` needs numpy + soundfile, and ingest must work
    on a machine that has neither."""
    try:
        from measure_render import audit  # noqa: PLC0415
    except ImportError as err:
        raise SystemExit(f"ERROR: --measure needs scripts/measure_render.py's dependencies "
                         f"(numpy, soundfile): {err}")
    return audit(str(wav))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bind rendered Suno clips to SKAÐI probes.")
    ap.add_argument("expedition", help="expedition directory")
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--map", action="append", default=[], metavar="PROBE=CLIP_ID",
                    help="resolve an ambiguous or missing match by hand (repeatable)")
    ap.add_argument("--fetch", action="store_true", help="download the matched WAVs")
    ap.add_argument("--measure", action="store_true",
                    help="run scripts/measure_render.py::audit on each WAV")
    ap.add_argument("--wav-dir", default=None,
                    help="where WAVs live (default: <expedition>/wav)")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    load_gates()  # a reader opens the file, even when it needs no threshold today

    expedition = Path(args.expedition)
    if not expedition.is_absolute():
        expedition = REPO / expedition
    if not expedition.is_dir():
        print(f"error: no such expedition directory: {expedition}", file=sys.stderr)
        return 2

    ids = probe_ids(expedition)
    prompts, errors = probe_prompts(expedition, ids)
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.is_absolute():
        snapshot_path = REPO / snapshot_path
    rows = load_snapshot(snapshot_path)
    overrides = parse_map(args.map)
    known = {r["id"] for r in rows}

    print(f"[EXTRACT] {len(prompts)} of {len(ids)} probe package(s) read · "
          f"{len(rows)} snapshot row(s)", file=sys.stderr)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    by_id = {r["id"]: r for r in rows}

    def bind(cid: str, how: str) -> dict:
        """One probe's RENDERS.json entry.

        `model` is read off the snapshot row, never invented: `skadi_bank.py`
        writes it into the claim's `models` column, and the alternative was the
        placeholder `(declare the model epoch)`, which satisfied every
        non-emptiness gate while scoping the claim to nothing.

        `written_at_utc` is when this binding was made. `validate_music_atlas.py`
        compares it against `registered_utc` — a prediction stamped after its
        clip was bound is not a prediction, and that is now checkable rather
        than promised.
        """
        row = by_id.get(cid, {})
        epoch = (row.get("major_model_version") or "").strip()
        return {"clip_id": cid, "wav": None, "measurements": None, "matched_by": how,
                "model": epoch or None, "written_at_utc": now}

    renders: dict[str, dict] = {}
    unresolved: list[str] = list(errors)
    for pid in ids:
        if pid in overrides:
            cid = overrides[pid]
            if cid not in known:
                unresolved.append(f"{pid}: --map names clip {cid!r}, absent from the snapshot")
                continue
            renders[pid] = bind(cid, "map")
            continue
        if pid not in prompts:
            continue  # its package error is already recorded
        cid, how, reason, candidates = match_probe(prompts[pid][1], rows)
        if cid is None:
            unresolved.append(
                f"{pid}: {reason} match. Candidates:\n      " + "\n      ".join(candidates)
                + f"\n      Resolve with --map {pid}=<clip_id>")
            continue
        renders[pid] = bind(cid, how)

    # ⛔ THE UNIQUENESS INVARIANT, over the whole plate, after --map has had its
    # say. Two probes on one clip is one render counted twice: every downstream
    # number (n, coverage, the prediction score) inflates while every evidence
    # path still resolves, which is the hardest kind of wrong to see.
    holders: dict[str, list[str]] = {}
    for pid, entry in renders.items():
        holders.setdefault(entry["clip_id"], []).append(pid)
    for cid, pids in sorted(holders.items()):
        if len(pids) < 2:
            continue
        for pid in pids:
            renders.pop(pid, None)
        unresolved.append(
            f"COLLISION: clip {cid} was bound to {len(pids)} probes ({', '.join(sorted(pids))}). "
            "One render answers one ask; none of these are written.\n      "
            "Resolve by naming a distinct clip per probe: "
            + " ".join(f"--map {pid}=<clip_id>" for pid in sorted(pids)))

    wav_dir = Path(args.wav_dir) if args.wav_dir else expedition / "wav"
    if not wav_dir.is_absolute():
        wav_dir = REPO / wav_dir

    # Bind what is already on disk BEFORE any network intent. `--measure` alone
    # is a legitimate re-run over WAVs that were pulled on an earlier pass.
    if renders:
        already = existing_wavs([r["clip_id"] for r in renders.values()], wav_dir)
        for entry in renders.values():
            got = already.get(entry["clip_id"])
            if got is not None:
                entry["wav"] = got.relative_to(REPO).as_posix() \
                    if got.is_relative_to(REPO) else str(got)

    if args.fetch and renders:
        try:
            snap_rel = snapshot_path.relative_to(REPO).as_posix()
        except ValueError:
            snap_rel = str(snapshot_path)
        paths = fetch_wavs([r["clip_id"] for r in renders.values()], rows, wav_dir, snap_rel)
        for pid, entry in renders.items():
            got = paths.get(entry["clip_id"])
            if got:
                entry["wav"] = got.relative_to(REPO).as_posix() \
                    if got.is_relative_to(REPO) else str(got)

    if args.measure:
        for pid, entry in renders.items():
            rel = entry.get("wav")
            wav = (REPO / rel) if rel else None
            if wav is None or not wav.exists():
                print(f"[measure] {pid}: no wav for clip {entry['clip_id']} under "
                      f"{wav_dir} (manifest or *__{entry['clip_id'][:8]}.wav) — pull it with "
                      "--fetch, or point --wav-dir at where it already lives",
                      file=sys.stderr)
                continue
            entry["measurements"] = measure(wav)

    out = expedition / "RENDERS.json"
    out.write_text(json.dumps(renders, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE     {out}")
    print(f"{len(renders)} of {len(ids)} probe(s) bound to a clip")
    for pid, entry in sorted(renders.items()):
        print(f"  {pid}  {entry['clip_id']}  ({entry['matched_by']})")

    if unresolved:
        print("\n⛔ UNRESOLVED — a wrong match is worse than no match, so these stop here:",
              file=sys.stderr)
        for u in unresolved:
            print("  " + u, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
