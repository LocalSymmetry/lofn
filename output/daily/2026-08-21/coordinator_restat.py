#!/usr/bin/env python3
"""Coordinator re-stat + RUN_STATE rebuild for daily-music-2026-08-21.

REUSED INSTRUMENT — copied verbatim from output/daily/2026-08-18-canonical/ and
re-pointed, per the standing rule: REUSE THE REPAIRED INSTRUMENT, NEVER RE-DERIVE IT.
Two additions for this run only, both marked [2026-08-21]: the COHORT GATE sweep
(collective pronouns in sung lines) and the DEDICATION character measurement.

THE SUBAGENT'S RETURN IS A CLAIM. THIS SCRIPT IS THE PROOF.

Three runs of record established that when a pair agent's measured claim was properly
re-measured, THE AGENT WAS RIGHT AND THE DISCREPANCY WAS THE COORDINATOR'S. So this
script exists to stop the coordinator guessing, not to catch the pairs.

Instrument-credibility rules baked in (L25 / L28 / L31):
  * PRINT WHAT WAS EXTRACTED before trusting what was concluded.
  * An EMPTY EXTRACTION IS A HARD ERROR, never a score.
  * ASSERT the extraction count equals the expected cardinality (24 prompts).
  * MODE-DETECT the heading convention once per file; never "canonical-first with a
    loose fallback" — that only reorders the failure.
  * WORD BOUNDARIES on every token sweep (`garage` contains `rage`; `ornament`
    contains `amen`).
  * SWEEP SUNG LINES ONLY for lyric properties — prose and appendices produced five
    false extractions on 2026-08-13, every one the coordinator's.

Usage:  python3 output/daily/2026-08-18-canonical/coordinator_restat.py
"""
import hashlib
import json
import pathlib
import re
import sys

RUN = pathlib.Path("output/daily/2026-08-21")
PAIRS = [f"{i:02d}" for i in range(1, 7)]
STEPS = ["step06_facets", "step07_song_guides", "step08_generation",
         "step09_artist_refined", "step10_revision_synthesis",
         "step10_final_package_enhanced"]  # step 11 output
EXPECTED_PROMPTS = 24

# gates.yaml mirrors (single source is vault/gates.yaml)
G = dict(prompt_lo=850, prompt_hi=1000, prompt_hug=985,
         lyrics_max=5000, lyrics_target=4800,
         lines_lo=70, lines_hi=120, lines_hug=72,
         excl_lo=400, excl_hi=1000)

BANNED_OPENERS = ["Begin in", "Begin with", "Use ", "Build the track from",
                  "Chronology:", "For an adult human singer"]
BANNED_TOKENS = ["snarl", "phonk", "supersaw", "airhorn"]
BREAK_TOKENS = ["amen", "funky drummer", "think break", "apache", "chopped break"]
AGGRO_TOKENS = ["rage", "furious", "aggressive", "brutal", "violent", "snarling"]


def wordsweep(text, token):
    """Word-boundary count. NEVER use `token in text` for a token sweep."""
    return len(re.findall(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", text.lower()))


WORDNUM = {"one": "1", "two": "2", "three": "3", "four": "4"}

# Observed across the six shipped files (measured, not assumed):
#   `# VARIATION 1 — TITLE`      (pairs 01,02,03,04)
#   `# ▚ VARIATION 1 — TITLE`    (pair 05 — a decorative glyph before the word)
#   `# 1. VARIATION ONE — TITLE` (pair 06 — an ordinal prefix AND a word-number)
# One pattern, tolerant of a leading glyph / ordinal, accepting digit or word numbers.
# NOT "canonical-first with a loose fallback" — that only reorders the failure (L31).
VARIATION_RE = re.compile(
    r"^#{1,4}[ \t]*(?:[^\w\s#]+[ \t]*)?(?:\d[.)][ \t]*)?"  # [2026-08-21] glyph RUN, not one glyph
    r"(?:(?:VARIATION|SONG)[ \t]+V?(?:([1-4])|(ONE|TWO|THREE|FOUR))|V([1-4]))\b.*$",  # [2026-08-21] `VARIATION V1` form
    re.M | re.I,
)


def split_variations(text):
    """Mode-detect the variation heading convention ONCE, then use that table alone."""
    hits = list(VARIATION_RE.finditer(text))
    if len(hits) < 2:
        return [], None
    mode = ("v-form" if hits[0].group(3)
            else "digit" if hits[0].group(1) else "word-number")
    if mode != "v-form" and hits[0].group(0).lstrip("# \t")[:1] not in "VSvs":
        mode += "+prefix"
    out = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        vid = m.group(1) or m.group(3) or WORDNUM[m.group(2).lower()]
        out.append((vid, text[m.start():end]))
    return out, mode


def grab_block(chunk, *headings):
    """Pull a labelled section's body out of one variation chunk."""
    for h in headings:
        # [2026-08-21] Same leading-glyph-run and ordinal tolerance VARIATION_RE
        # already has. Without it `### ⭐ DEDICATION` and `## 4. DEDICATION` extracted
        # EMPTY and were silently skipped rather than flagged.
        m = re.search(r"^#{1,4}[ \t]*(?:[^\w\s#]+[ \t]*)?(?:\d+[A-Za-z]?[.)][ \t]*)?"
                      + h + r".*?$", chunk, re.M | re.I)
        if not m:
            continue
        rest = chunk[m.end():]
        nxt = re.search(r"^#{1,4}\s+", rest, re.M)
        body = rest[:nxt.start()] if nxt else rest
        fenced = re.findall(r"```[a-z]*\n(.*?)```", body, re.S)
        return (fenced[0] if fenced else body).strip()
    return ""


def sung_lines(lyrics):
    """Sung lines ONLY — no headers, no bracket tags, no SFX cues, no blanks."""
    out = []
    for ln in lyrics.splitlines():
        s = ln.strip()
        if not s or s.startswith("[") or s.startswith("*") or s.startswith("#"):
            continue
        if re.match(r"^\(.*\)$", s) and len(s) < 4:
            continue
        out.append(s)
    return out


def measure_dedication(chunk):
    """⭐ THE ONE implementation of the dedication character count.

    A dedication is the PARAGRAPH. A trailing italic note, a self-reported count in
    the heading, and a section rule are all ABOUT the dedication and are not part of
    it. Counting them fabricated exactly 83 characters on four compliant dedications
    on 2026-08-21, and five characters on 2026-08-15, and 158 on 2026-08-14 — always
    over, always in this field, always bouncing something that was fine.

    Every reader in this run calls THIS. Nobody re-implements it.
    """
    body = grab_block(chunk, r"DEDICATION")
    if not body:
        return ""
    body = re.sub(r"\n\s*-{3,}\s*$", "", body).strip()          # trailing rule
    body = re.sub(r"\n\s*\*\([^)]*\)\*\s*$", "", body).strip()  # trailing italic note
    body = re.sub(r"^\*\([^)]*\)\*\s*\n", "", body).strip()      # leading italic note
    return body


rows, findings, all_prompts, all_lyrics = [], [], {}, {}

for p in PAIRS:
    for st in STEPS:
        fp = RUN / f"pair_{p}_{st}.md"
        rec = dict(pair=p, step=st, canonical_path=str(fp).replace("\\", "/"),
                   exists=fp.exists(), byte_size=0, sha="", status="pending",
                   gate_verdict="", attempt_count=1)
        if fp.exists():
            b = fp.read_bytes()
            rec.update(byte_size=len(b), sha=hashlib.sha256(b).hexdigest()[:12])
            if len(b) < 800:
                rec["gate_verdict"] = "FAIL:stub"
                findings.append(f"P{p}/{st}: {len(b)} bytes — stub/truncated")
            else:
                # ⚠️ QA-corrected: this used to read "PASS:exists", which is an EXISTENCE
                # CHECK WEARING THE WORD PASS. Every row in the manifest said PASS while
                # nothing in it had been gated. Renamed so the manifest cannot overclaim;
                # measured verdicts are attached below, per artifact, from the deep re-stat.
                rec["status"], rec["gate_verdict"] = "done", "exists:non-trivial"
        rows.append(rec)

    # ---- deep re-stat of the step-10 package
    # Which step-10 artifact to re-stat. Default: the pair's own synthesis.
    # `--enhanced` points this reader at the STEP-11 packages instead.
    # Why here and not in verify_enhanced.py: that reader's block() takes the FIRST
    # heading match in the whole file, so on a four-variation enhanced file it stats
    # V1 and silently reports the pair "clean" — a gate that cannot fail. Found by the
    # P02 step-11 tier, 2026-08-18. Its own docstring forbids widening it (L31), so the
    # variation-aware instrument reads the variation-shaped files instead.
    _suffix = ("step10_final_package_enhanced" if "--enhanced" in sys.argv
               else "step10_revision_synthesis")
    fp10 = RUN / f"pair_{p}_{_suffix}.md"
    if not fp10.exists():
        findings.append(f"P{p}: step10 MISSING — no package to re-stat")
        continue
    text = fp10.read_text(encoding="utf-8", errors="replace")
    variations, mode = split_variations(text)
    print(f"\n=== PAIR {p} — step10 ({fp10.stat().st_size:,} bytes) ===")
    print(f"  heading mode detected : {mode}")
    print(f"  variations extracted  : {len(variations)}  (expect 4)")
    if len(variations) != 4:
        findings.append(f"P{p}: extracted {len(variations)} variations, expected 4 "
                        f"— EXTRACTION ERROR, not a finding about the artifact")
        continue

    for vid, chunk in variations:
        # Heading alias table. 2026-08-18-canonical's FORMAT CONTRACT shape is listed
        # first; the 2026-08-14 shapes are kept so this instrument still reads that run.
        # This is an alias table, not "canonical-first with a loose fallback" (L31).
        prompt = grab_block(chunk, r"SUNO STYLE PROMPT", r"1\.\s*MUSIC PROMPT", r"MUSIC PROMPT")
        excl = grab_block(chunk, r"SUNO EXCLUDE\b", r"1B\.\s*SUNO EXCLUDE PROMPT", r"EXCLUDE PROMPT")
        lyr = grab_block(chunk, r"SUNO LYRICS", r"2\.\s*LYRICS", r"3\.\s*LYRICS", r"LYRICS PROMPT", r"LYRICS")
        title = grab_block(chunk, r"3\.\s*TITLE", r"TITLE")
        if not title:
            # v-form files carry the title in the variation heading itself: `# V1 — TITLE`
            mt = re.match(r"^#{1,4}[ \t]*V[1-4][ \t]*[—–-][ \t]*(.+?)[ \t]*$", chunk, re.M)
            title = mt.group(1).strip() if mt else ""
        if not prompt or not lyr:
            findings.append(f"P{p}/V{vid}: EMPTY EXTRACTION (prompt={len(prompt)}, "
                            f"lyrics={len(lyr)}) — HARD ERROR, not a score")
            continue
        sl = sung_lines(lyr)
        all_prompts[f"{p}V{vid}"] = prompt
        all_lyrics[f"{p}V{vid}"] = "\n".join(sl)
        flags = []
        if not (G["prompt_lo"] <= len(prompt) <= G["prompt_hi"]):
            flags.append(f"PROMPT_CHARS={len(prompt)}")
        elif len(prompt) >= G["prompt_hug"]:
            flags.append(f"HUG(prompt {len(prompt)})")
        if prompt and prompt.rstrip()[-1] not in ".!?":
            flags.append("NO_TERMINAL_PUNCT")
        if len(lyr) >= G["lyrics_max"]:
            flags.append(f"LYRICS_OVER_CAP={len(lyr)}")
        elif len(lyr) > G["lyrics_target"]:
            flags.append(f"lyrics>{G['lyrics_target']}({len(lyr)})")
        if not (G["lines_lo"] <= len(sl) <= G["lines_hi"]):
            flags.append(f"SUNG_LINES={len(sl)}")
        elif len(sl) <= G["lines_hug"]:
            flags.append(f"HUG(lines {len(sl)})")
        if excl and not (G["excl_lo"] <= len(excl) <= G["excl_hi"]):
            flags.append(f"EXCLUDE_CHARS={len(excl)}")
        if not excl:
            flags.append("EXCLUDE_MISSING")
        for op in BANNED_OPENERS:
            if prompt.lower().startswith(op.lower()):
                flags.append(f"BANNED_OPENER:{op}")
        for t in BANNED_TOKENS:
            if wordsweep(prompt, t):
                flags.append(f"BANNED_TOKEN:{t}")
        if not re.search(r"\[Theme:", lyr):
            flags.append("NO_THEME_TAG")
        if not re.search(r"\[SONG FORM:", lyr, re.I):
            flags.append("NO_SONGFORM_TAG")
        if not re.search(r"^\s*\*[^*]+\*\s*$", lyr, re.M):
            flags.append("NO_SFX_CUE")
        bare = len(re.findall(r"EMO:\s*(?:AWE|INDIGNATION)\s*[-\]]", lyr, re.I))
        if bare:
            flags.append(f"BARE_EMO_TAGS={bare}")
        # [2026-08-21] THE COHORT GATE — delete every collective pronoun. Word
        # boundaries are mandatory: "ours" contains "our", "wearing" contains "we".
        # Sung lines ONLY; a "we" in a bracket header or an appendix is not sung.
        cohort = {t: wordsweep("\n".join(sl), t)
                  for t in ("we", "us", "our", "ours", "everyone", "everybody", "were")
                  if t != "were" and wordsweep("\n".join(sl), t)}
        if cohort:
            flags.append(f"COHORT_GATE={cohort}")
        # [2026-08-21] DEDICATION — measured in code, never by eye. The 2026-08-15 run
        # had three instruments disagree on one dedication and one FABRICATED five
        # characters of over-cap violation. Extract, strip the fence, measure.
        # [2026-08-21] ⭐ ONE implementation of the dedication measurement.
        # Ledger, 2026-08-15: "THE FIX WAS NOT THE REGEX — TWO IMPLEMENTATIONS OF ONE
        # MEASUREMENT IS THE DEFECT." Three recurrences now (08-14, 08-15, 08-21), and
        # every re-implementation has OVER-counted, and an over-count always bounces a
        # compliant artifact. This function is the only place the count is computed.
        ded = measure_dedication(chunk)
        ded_heading_present = bool(re.search(r"^#{1,4}[^\n]*DEDICATION", chunk, re.M | re.I))
        if ded_heading_present and not ded:
            findings.append(f"P{p}/V{vid}: DEDICATION heading present but extraction EMPTY "
                            f"— HARD ERROR, the cap was NOT measured (never a silent skip)")
        if ded:
            # [2026-08-21 REPAIR] A dedication is the PARAGRAPH. A trailing italic
            # metadata note and a section rule are ABOUT the dedication, not part of
            # it. Swallowing them fabricated exactly 83 chars on all four of Pair 06's
            # dedications and would have bounced four compliant artifacts as OVER_CAP.
            # This is the 2026-08-15 "invented five characters" failure, same field,
            # one week later, in a different instrument. Measure the object, not the
            # commentary beside it.
            ded = re.sub(r"\n\s*-{3,}\s*$", "", ded).strip()
            ded = re.sub(r"\n\s*\*\([^)]*\)\*\s*$", "", ded).strip()
            if len(ded) > 500:
                flags.append(f"DEDICATION_OVER_CAP={len(ded)}")
            else:
                flags.append(f"ded={len(ded)}c")
        digits = re.findall(r"(?<![\w#])\d{1,4}(?![\w%])", "\n".join(sl))
        if digits:
            flags.append(f"SUNG_DIGITS={digits[:4]}")
        # NAMED figures trigger an OWED credit. Generic break language does NOT —
        # crediting Coleman for an unnamed "chopped break" would be a FALSE credit, and
        # a false credit makes the true ones worthless. My first version of this check
        # conflated the two and flagged `chopped break` on 5 variations; a detector
        # written in response to a finding gets the same scrutiny as the finding.
        named = {"amen": "Gregory C. Coleman / The Winstons",
                 "funky drummer": "Clyde Stubblefield",
                 "apache": "the Incredible Bongo Band"}
        brk = [f"{t} -> {who}" for t, who in named.items()
               if re.search(r"(?<![a-z])" + t + r"(?![a-z])", prompt.lower())]
        if re.search(r"(?<![a-z])think break(?![a-z])", prompt.lower()):
            brk.append("think break -> Lyn Collins")
        generic = bool(re.search(r"(?<![a-z])(chopped |re)?break(beat)?s?(?![a-z])", prompt.lower())) and not brk
        print(f"  V{vid}: prompt {len(prompt):>4}c | excl {len(excl):>4}c | "
              f"lyrics {len(lyr):>4}c | sung {len(sl):>3} | title {'y' if title else 'N'} "
              f"| {'; '.join(flags) if flags else 'clean'}")
        if brk:
            print(f"        NAMED FIGURE in STYLE -> CREDIT OWED: {brk}")
        elif generic:
            print(f"        generic break language only — NO named figure, so NO credit owed "
                  f"(crediting anyway would be a false credit)")
        for f in flags:
            if f.split("(")[0].isupper():
                findings.append(f"P{p}/V{vid}: {f}")
        # attach a MEASURED verdict to the step-10 row, so the manifest carries gate
        # results and not merely a stat() call (QA finding: every gate_verdict was an
        # existence check, and the six SHIPPED artifacts appeared in zero rows at all)
        for r in rows:
            if r["pair"] == p and r["step"] == "step10_revision_synthesis":
                prev = r["gate_verdict"]
                mark = f"V{vid}:{'FLAG' if flags else 'clean'}"
                r["gate_verdict"] = (prev + " " + mark) if "V" in prev else f"measured {mark}"

# ---------- portfolio distinctiveness (assert cardinality BEFORE concluding)
print(f"\n=== PORTFOLIO ===")
print(f"  prompts extracted: {len(all_prompts)} / {EXPECTED_PROMPTS}")
print(f"  lyrics  extracted: {len(all_lyrics)} / {EXPECTED_PROMPTS}")
if len(all_prompts) != EXPECTED_PROMPTS:
    print(f"  !! EXTRACTION COUNT != {EXPECTED_PROMPTS} — similarity numbers below are "
          f"NOT trustworthy and are a bug report, not a finding.")

if len(all_lyrics) >= 2:
    from difflib import SequenceMatcher

    def ngrams(t, n=5):
        w = re.findall(r"[a-z']+", t.lower())
        return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}

    worst_l = worst_p = worst_j = (0.0, "", "")
    keys = sorted(all_lyrics)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[:2] == b[:2]:
                continue  # within-pair variations are meant to be siblings
            sl = SequenceMatcher(None, all_lyrics[a], all_lyrics[b]).ratio()
            sp = SequenceMatcher(None, all_prompts[a], all_prompts[b]).ratio()
            ga, gb = ngrams(all_lyrics[a]), ngrams(all_lyrics[b])
            j = len(ga & gb) / len(ga | gb) if (ga | gb) else 0.0
            worst_l = max(worst_l, (sl, a, b))
            worst_p = max(worst_p, (sp, a, b))
            worst_j = max(worst_j, (j, a, b))
    print(f"  worst cross-pair LYRIC  similarity: {worst_l[0]:.3f}  ({worst_l[1]} vs {worst_l[2]})  ceiling 0.42")
    print(f"  worst cross-pair PROMPT similarity: {worst_p[0]:.3f}  ({worst_p[1]} vs {worst_p[2]})  ceiling 0.58")
    print(f"  worst cross-pair 5-gram Jaccard   : {worst_j[0]:.4f} ({worst_j[1]} vs {worst_j[2]})  ceiling 0.18")
    for val, ceil, label in ((worst_l[0], 0.42, "lyric"), (worst_p[0], 0.58, "prompt"),
                             (worst_j[0], 0.18, "5-gram")):
        if val > ceil:
            findings.append(f"PORTFOLIO: {label} similarity {val:.3f} > {ceil}")

# ---------- [2026-08-21] WITHIN-PAIR similarity — THE AXIS WITH NO VALIDATOR
# Ledger, 2026-08-15: "L41, fourth confirmation, and the level is named. Cross-pair
# similarity measured 0.074/0.055 against 0.42/0.58 ceilings with ZERO shared sung
# lines - so residue relocated to the one axis with NO VALIDATOR: WITHIN-pair prompt
# similarity. P05's four MUSIC PROMPTs are 0.497 similar to each other, 9x the
# cross-pair maximum, sharing an identical 30-word sentence. ONE PROMPT WEARING FOUR
# HATS. Third consecutive run where residue lands exactly one level below the fix."
#
# This is that reader. It is deliberately a FLAG, never a fail: within-pair variations
# ARE meant to be siblings - they share an arrangement, a tempo, a key and a concept by
# design, and that is the pair-isolation law working, not breaking. What it catches is
# the degenerate case: four prompts that are one prompt with nouns swapped.
#
# The ceiling is set from the ONE measured datum we have (0.497 = "one prompt wearing
# four hats") and the cross-pair maximum (0.058). A sibling band has to sit well above
# the cross-pair figure; 0.45 sits just under the known-bad case. n=1 - it is a
# tripwire, not a verdict, and it says so when it fires.
print(f"\n=== WITHIN-PAIR SIMILARITY (the axis that had no reader until today) ===")
WITHIN_PROMPT_FLAG = 0.45
WITHIN_LYRIC_FLAG = 0.45
for _p in PAIRS:
    ks = sorted(k for k in all_prompts if k[:2] == _p)
    if len(ks) < 2:
        continue
    from difflib import SequenceMatcher as _SM
    pr = [(_SM(None, all_prompts[a], all_prompts[b]).ratio(), a, b)
          for i, a in enumerate(ks) for b in ks[i + 1:]]
    ly = [(_SM(None, all_lyrics[a], all_lyrics[b]).ratio(), a, b)
          for i, a in enumerate(ks) for b in ks[i + 1:]]
    wp, wl = max(pr), max(ly)
    mp = sum(x[0] for x in pr) / len(pr)
    ml = sum(x[0] for x in ly) / len(ly)
    mark = ""
    if wp[0] > WITHIN_PROMPT_FLAG:
        mark += f"  <-- FLAG prompt {wp[1]}~{wp[2]}"
        findings.append(f"P{_p}: WITHIN-PAIR prompt similarity {wp[0]:.3f} > {WITHIN_PROMPT_FLAG} "
                        f"({wp[1]} vs {wp[2]}) - FLAG to the Somatic read, not an auto-fail")
    if wl[0] > WITHIN_LYRIC_FLAG:
        mark += f"  <-- FLAG lyric {wl[1]}~{wl[2]}"
        findings.append(f"P{_p}: WITHIN-PAIR lyric similarity {wl[0]:.3f} > {WITHIN_LYRIC_FLAG} "
                        f"({wl[1]} vs {wl[2]}) - FLAG to the Somatic read, not an auto-fail")
    print(f"  P{_p}: prompt worst {wp[0]:.3f} / mean {mp:.3f} | "
          f"lyric worst {wl[0]:.3f} / mean {ml:.3f}{mark}")
print(f"  reference: the 2026-08-15 known-bad case measured 0.497 (one prompt, four hats); "
      f"that run's cross-pair maximum was 0.058. Flag band {WITHIN_PROMPT_FLAG}, n=1 - a tripwire, not a verdict.")

# ---------- anger audit, WITH WORD BOUNDARIES
print(f"\n=== ANGER AUDIT (word-boundary; `garage`/`Baltimore` contain `rage` as substrings) ===")
aggro_pairs = set()
for k, pr in all_prompts.items():
    hits = {t: wordsweep(pr, t) for t in AGGRO_TOKENS if wordsweep(pr, t)}
    if hits:
        aggro_pairs.add(k[:2])
        print(f"  {k}: {hits}")
naive = sum(1 for pr in all_prompts.values() if "rage" in pr.lower())
print(f"  pairs with aggression tokens: {len(aggro_pairs)} of 6  (cap 2)")
print(f"  [control] naive substring 'rage' would have matched {naive}/24 prompts — "
      f"this is why word boundaries are mandatory")
if len(aggro_pairs) > 2:
    findings.append(f"ANGER AUDIT: {len(aggro_pairs)} of 6 aggressive, cap is 2")

# ---------- write RUN_STATE (LAST action, so it never claims an absent artifact)
icb = RUN / "CREATIVE_CONTEXT.md"
icb_sha = hashlib.sha256(icb.read_bytes()).hexdigest()[:16] if icb.exists() else "MISSING"
done = sum(1 for r in rows if r["status"] == "done")
manifest = [
    "# RUN_STATE — disk-derived manifest · `daily-music-2026-08-21`",
    "",
    "Rebuilt by stat-ing the run directory. **Disk is authority** — if this file and disk",
    "disagree, re-derive from disk. Written as the LAST action of the wave so it can never",
    "claim an artifact that is not on disk.",
    "",
    f"`icb_sha: {icb_sha}`  ·  `icb_bytes: {icb.stat().st_size if icb.exists() else 0}`",
    f"`artifacts_done: {done}/{len(rows)}`",
    "",
    "| step | pair | exists | bytes | sha | gate | status |",
    "|---|---|---|---|---|---|---|",
]
for r in rows:
    manifest.append(f"| {r['step']} | {r['pair']} | {'yes' if r['exists'] else 'NO'} | "
                    f"{r['byte_size']:,} | `{r['sha']}` | {r['gate_verdict']} | {r['status']} |")
manifest += ["", "## Findings from this re-stat", ""]
manifest += [f"- {f}" for f in findings] or ["- none"]
(RUN / "RUN_STATE.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")

print(f"\n=== RUN_STATE ===")
print(f"  artifacts done: {done}/{len(rows)}   icb_sha {icb_sha}")
print(f"  findings: {len(findings)}")
for f in findings:
    print(f"    - {f}")
json.dump({"rows": rows, "findings": findings}, open(RUN / "GATE_REPORT.json", "w"), indent=1)
