from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path


MUSIC = Path(__file__).resolve().parent
OUT = MUSIC / "qa_recheck_p01v01"
KEY = MUSIC / "qa_recheck_p01v01_PRIVATE.json"


def load_pool_builder():
    path = MUSIC / "build_blind_qa_pool.py"
    spec = importlib.util.spec_from_file_location("pool_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if OUT.exists() or KEY.exists():
        raise FileExistsError("refusing to overwrite existing P01V01 blind recheck")
    builder = load_pool_builder()
    candidate = next(x for x in builder.extract_candidates() if x["candidate_id"] == "P01V01")
    entries = [candidate, builder.extract_golden(), builder.extract_decoy()]
    random.Random("sunna-p01v01-repair-recheck-20260824").shuffle(entries)
    OUT.mkdir(parents=True)
    key_rows = []
    for label, entry in zip(("A", "B", "C"), entries, strict=True):
        content = entry.pop("content")
        (OUT / f"package_{label}.md").write_text(content, encoding="utf-8", newline="\n")
        key_rows.append({
            "label": label,
            **entry,
            "bytes": len(content.encode("utf-8")),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    KEY.write_text(json.dumps({
        "classification": "PRIVATE BLIND KEY — RECHECK JUDGE MUST NOT READ BEFORE STAGE 2",
        "packages": key_rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("built 3-member repaired P01V01 blind recheck")


if __name__ == "__main__":
    main()
