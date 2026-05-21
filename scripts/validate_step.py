#!/usr/bin/env python3
"""Minimal deterministic validator for Lofn step artifacts.

This is not a creative-quality judge; it catches collapsed/stub/template artifacts before
agents advance to the next step.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    if len(sys.argv) != 3:
        return fail("usage: validate_step.py <step> <file>")
    step = str(sys.argv[1]).zfill(2)
    path = Path(sys.argv[2])
    if not path.exists() or not path.is_file():
        return fail(f"missing file: {path}")
    name = path.name
    canonical = (
        name.startswith("step")
        or (name.startswith("pair_") and "_step" in name)
    )
    if not canonical:
        print(f"STEP {step} SKIPPED: {path} is not a canonical step artifact")
        return 0

    text = path.read_text(errors="replace")
    lower = text.lower()

    if len(text.strip()) < 800:
        return fail("artifact is too short to be a real Lofn step")
    if re.search(r"\b(lorem ipsum|todo|tbd|placeholder|similar arrangement|song n|genre n)\b", lower):
        return fail("artifact contains placeholder/template language")
    if re.search(r"\bline\s+\d+\b", lower):
        return fail("artifact contains numbered placeholder lyric lines like 'line 1'")
    # Catch copy-paste repetition masquerading as step depth.
    nonempty = [ln.strip().lower() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith('#')]
    if len(nonempty) >= 6:
        unique_ratio = len(set(nonempty)) / len(nonempty)
        if unique_ratio < 0.45:
            return fail(f"artifact is excessively repetitive (unique line ratio {unique_ratio:.2f})")
    paras = [p.strip().lower() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 120]
    if len(paras) >= 3 and len(set(paras)) <= len(paras) // 2:
        return fail("artifact repeats large paragraph blocks instead of developing the step")
    if re.search(r"steps?[_ -]?0?6[_ -]?0?10", path.name.lower()):
        return fail("collapsed Steps 06-10 file is not a canonical step artifact")

    # Provenance gate: every canonical step must prove a real prompt/response chain.
    # Accept old section names briefly for legacy artifacts, but require the new
    # stronger sections for all newly generated artifacts when present.
    required_sections = [
        "## 0. Step Provenance",
        "## 1. Input Context Digest",
        "## 2. Step Template Requirements Applied",
    ]
    missing_sections = [sec for sec in required_sections if sec.lower() not in lower]
    if missing_sections:
        return fail("artifact missing step provenance sections: " + ", ".join(missing_sections))
    has_new_contract = "## 4. complete step output" in lower or "## 5. execution log" in lower
    if has_new_contract:
        for sec in [
            "## 3. Panel / Critic Deliberation Log",
            "## 4. Complete Step Output",
            "## 5. Execution Log",
            "## 6. Self-Critique Against Step Requirements",
        ]:
            if sec.lower() not in lower:
                return fail(f"artifact missing required complete-output section: {sec}")
    else:
        for sec in [
            "## 3. Model Response / Creative Work",
            "## 4. Self-Critique Against Step Requirements",
        ]:
            if sec.lower() not in lower:
                return fail(f"artifact missing legacy creative/provenance section: {sec}")
    if "what this step would do" in lower or "would generate" in lower or "would produce" in lower:
        return fail("artifact describes what the step would do instead of containing complete step output")
    if "panel / critic deliberation log" in lower:
        panel_section = re.search(r"## 3\. Panel / Critic Deliberation Log(.*?)(?=\n## 4\.)", text, re.I | re.S)
        if panel_section:
            ps = panel_section.group(1).lower()
            for marker in ["devil", "hyper-skeptic", "resolution"]:
                if marker not in ps:
                    return fail(f"panel deliberation log missing marker: {marker}")
    if "step file loaded:" not in lower:
        return fail("artifact missing explicit loaded step file path")
    if "input artifacts used:" not in lower:
        return fail("artifact missing explicit input artifact list")
    if "validation command:" not in lower:
        return fail("artifact missing validation command provenance")

    checks = {
        "00": ["aesthetic", "emotion", "genre"],
        "01": ["essence", "facet", "style"],
        "02": ["concept"],
        "03": ["artist", "critique"],
        "04": ["medium"],
        "05": ["pair", "concept", "medium"],
        "06": ["facet"],
        "07": ["song guide"],
        "08": ["prompt"],
        "09": ["artist", "refin"],
        "10": ["prompt"],
    }
    for needle in checks.get(step, []):
        if needle not in lower:
            return fail(f"step {step} artifact missing expected marker: {needle}")

    is_music = "music" in str(path).lower() or "song" in lower or "lyrics" in lower
    if is_music and step in {"08", "10"}:
        if "lyrics" not in lower:
            return fail("music song artifact missing lyrics")
        prompt_count = len(re.findall(r"^##\s*1\.\s*music prompt\b|^\[suno style prompt\s*:\]", text, re.I | re.M))
        lyric_count = len(re.findall(r"^##\s*2\.\s*lyrics\b", text, re.I | re.M))
        song_form_count = len(re.findall(r"\[song form\s*:[^\]]+\]", text, re.I))
        full_emo_headers = len(re.findall(r"^\[[^\]\n]+[-–—]\s*EMO\s*:[^\]\n]+[-–—][^\]\n]+[-–—][^\]\n]+\]", text, re.I | re.M))
        bare_emo_headers = len(re.findall(r"^\[\s*EMO\s*:", text, re.I | re.M))
        if prompt_count < 1:
            return fail("music song artifact missing standalone ## 1. MUSIC PROMPT")
        if lyric_count < 1:
            return fail("music song artifact missing ## 2. LYRICS section")
        if song_form_count < lyric_count:
            return fail(f"music song artifact missing bracketed [SONG FORM:] declarations ({song_form_count}/{lyric_count})")
        if full_emo_headers < max(6, lyric_count * 4):
            return fail(f"music song artifact has too few full section EMO headers ({full_emo_headers}); bare [EMO:...] is not enough")
        if bare_emo_headers:
            return fail("music song artifact uses bare [EMO:...] headers instead of [Section - EMO:... - Voice - Cue]")
        if re.search(r"^\s*EMO\s+HEADER\s*:", text, re.I | re.M):
            return fail("music song artifact contains prose EMO HEADER lines")
        if re.search(r"^\s*SONG FORM\s*:", text, re.I | re.M):
            return fail("music song artifact contains plain SONG FORM: text instead of [SONG FORM: ...]")
        if not re.search(r"^\*[^*\n]{1,50}\*\s*$", text, re.M):
            return fail("music song artifact missing standalone SFX cue")

    print(f"STEP {step} PASSED: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
