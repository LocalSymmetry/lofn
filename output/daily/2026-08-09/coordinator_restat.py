# -*- coding: utf-8 -*-
"""COORDINATOR RE-STAT — run 2026-08-09_daily_music_genz.

The pair agents' RETURN envelopes are CLAIMS. This is the proof.

DESIGN, written against five recorded instrument failures (L25 / L28 / L31):
  1. STATE WHAT WE WERE POINTED AT. A harness that does not print its input set is
     indistinguishable from a broken subject -- a regression harness once printed basenames
     only, making 60 files from unrelated image runs look like the subject failing.
  2. MODE DETECTION, NOT PATTERN PRIORITY. Detect each file's heading convention ONCE, then
     match that table alone. "Canonical-first with a loose fallback" does not remove the
     failure mode, it REORDERS it -- the loose fallback then over-matches a canonical file
     and grabs prose headings as content.
  3. AN EMPTY EXTRACTION IS A HARD ERROR, NEVER A SCORE. Two empty strings compare as 1.000;
     a strict-only matcher once extracted zero blocks and printed CLEAN on all three
     absolution scans of a run's most important gate.
  4. ASSERT THE EXTRACTION COUNT EQUALS THE EXPECTED CARDINALITY before concluding anything.
  5. PRINT WHAT WAS EXTRACTED BEFORE WHAT WAS CONCLUDED.

The canonical contract comes from skills/music/scripts/validate_suno_packages.py -- the
VALIDATOR, never the artifacts.

Usage:  python3 output/daily/2026-08-09/coordinator_restat.py
"""
from __future__ import print_function
import io, os, re, sys, glob, json, difflib

RUN = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(RUN, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

EXPECTED_PAIRS = 6
VARIATIONS_PER_PAIR = 4
EXPECTED_PACKAGES = EXPECTED_PAIRS * VARIATIONS_PER_PAIR      # 24

# ---- thresholds: single-sourced from vault/gates.yaml, never hand-restated -------------
def load_gates():
    g, path = {}, os.path.join(REPO, "vault", "gates.yaml")
    txt = io.open(path, encoding="utf-8").read()
    def num(key, default):
        m = re.search(r"^%s:\s*([0-9.]+)" % re.escape(key), txt, re.M)
        return float(m.group(1)) if m else default
    def pair(key, default):
        m = re.search(r"^%s:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]" % re.escape(key), txt, re.M)
        return (float(m.group(1)), float(m.group(2))) if m else default
    g["prompt_band"]   = pair("music_prompt_chars", (850, 1000))
    g["prompt_target"] = pair("music_prompt_chars_target", (870, 960))
    g["hug"]           = num("music_prompt_hug_ceiling", 985)
    g["lyr_max"]       = num("suno_lyrics_field_max", 5000)
    g["lyr_target"]    = num("suno_lyrics_field_target", 4800)
    g["lines_band"]    = pair("sung_lines", (70, 120))
    g["lines_hug"]     = num("sung_lines_floor_hug", 72)
    g["rhyme_floor"]   = num("rhyme_return_floor", 0.30)
    g["line_floor"]    = num("line_return_floor", 0.20)
    g["wpl_ceiling"]   = num("mean_words_per_line_ceiling", 7.5)
    g["allit_floor"]   = num("alliteration_per_100w_floor", 11.0)
    g["lyric_sim"]     = num("portfolio_max_lyric_similarity", 0.42)
    g["prompt_sim"]    = num("portfolio_max_prompt_similarity", 0.58)
    return g

# ---- heading-convention MODE DETECTION -------------------------------------------------
CONVENTIONS = {
    "canonical": {                      # what validate_suno_packages.py enforces
        "prompt":  r"^## 1\. MUSIC PROMPT\s*$",
        "exclude": r"^## 1B\. SUNO EXCLUDE PROMPT\s*$",
        "lyrics":  r"^## 2\. LYRICS\s*$",
        "title":   r"^## 3\. TITLE\s*$",
    },
    "legacy_3lyrics": {                 # the ## 3. LYRICS convention seen on 2026-08-05
        "prompt":  r"^## 1\. MUSIC PROMPT\s*$",
        "exclude": r"^## 2\. EXCLUDE PROMPT\s*$",
        "lyrics":  r"^## 3\. LYRICS(?: PROMPT)?\s*$",
        "title":   r"^## 4\. TITLE\s*$",
    },
    "bare": {                           # ## MUSIC PROMPT / ## LYRICS, unnumbered
        "prompt":  r"^## MUSIC PROMPT\s*$",
        "exclude": r"^## (?:SUNO )?EXCLUDE PROMPT\s*$",
        "lyrics":  r"^## LYRICS\s*$",
        "title":   r"^## TITLE\s*$",
    },
}

def detect_mode(text):
    """Score every convention, pick the winner ONCE, then use only that table."""
    scores = {}
    for name, tbl in CONVENTIONS.items():
        scores[name] = sum(len(re.findall(rx, text, re.M)) for rx in tbl.values())
    best = max(scores, key=lambda k: scores[k])
    return (best, scores) if scores[best] > 0 else (None, scores)

def extract_blocks(text, mode):
    """Return list of dicts, one per package, using ONLY the detected convention."""
    tbl = CONVENTIONS[mode]
    heads = []
    for kind, rx in tbl.items():
        for m in re.finditer(rx, text, re.M):
            heads.append((m.start(), m.end(), kind))
    heads.sort()
    out, cur = [], None
    for i, (s, e, kind) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        body = text[e:end].strip()
        if kind == "prompt":
            if cur:
                out.append(cur)
            cur = {"prompt": body, "exclude": "", "lyrics": "", "title": ""}
        elif cur is not None:
            cur[kind] = body
    if cur:
        out.append(cur)
    return out

def defence(s):
    """Strip markdown fences so a fenced prompt is measured as its own text (the 2026-08-03 bug)."""
    s = re.sub(r"^```[a-zA-Z]*\s*$", "", s, flags=re.M)
    return s.strip()

def sung_lines(lyrics):
    """Delegate to the SHIPPED definition. Do not re-derive it.

    The doctrine this run dispatched to six agents says: measure with
    measure_soundcraft.py, "never by eye and never by re-deriving your own window."
    The coordinator's first version re-derived BOTH the line filter and (via a
    non-existent 'strict_end_rhyme' key that silently defaulted to 0.0) the rhyme
    number itself -- and would have reported RHYME_BELOW_FLOOR on all 24 songs.
    A missing dict key returning a default is a silent wrong answer; ask for the key
    the function actually returns, and let the module own the extraction."""
    from measure_soundcraft import sung
    return list(sung(lyrics))

BANNED_PHRASES = [
    "put the phone down", "look up from", "touch grass", "doomscroll", "screen time",
    "brain rot", "we used to",
]
COHORT_WORDS = [r"\bthis generation\b", r"\byoung people\b", r"\bkids these days\b", r"\bgen z\b"]
HOUSE_LEXICON = [
    "more sub and more sky", "small astonished laugh", "triple arch", "small enough to understand",
    "frost-air pad", "starfield percussion", "zodiacal glow", "clear silver tone", "dew-bright",
    "glass harmonica sheen", "crystalline arpeggios", "make my little fear",
]

def main():
    gates = load_gates()
    paths = sorted(glob.glob(os.path.join(RUN, "pair_*_step11_final_package_enhanced.md")))
    tier = "step11"
    if len(paths) < EXPECTED_PAIRS:
        p10 = sorted(glob.glob(os.path.join(RUN, "pair_*_step10_final_package.md")))
        if len(p10) > len(paths):
            paths, tier = p10, "step10"

    # (1) STATE WHAT WE WERE POINTED AT -- always, before any conclusion.
    print("=" * 78)
    print("COORDINATOR RE-STAT  |  run 2026-08-09_daily_music_genz  |  tier = %s" % tier)
    print("pointed at : %s" % os.path.join(RUN, "pair_*_%s*.md" % tier))
    print("files found: %d" % len(paths))
    for p in paths:
        print("   - %-52s %8d bytes" % (os.path.basename(p), os.path.getsize(p)))
    print("=" * 78)
    if not paths:
        print("ERROR: zero input files. This is a harness/point-at failure, not a clean run.")
        return 2

    rows, all_lyrics, all_prompts, hard_fail = [], {}, {}, []
    for p in paths:
        pair = re.search(r"pair_(\d+)_", os.path.basename(p)).group(1)
        text = io.open(p, encoding="utf-8").read()
        mode, scores = detect_mode(text)
        if mode is None:
            hard_fail.append("pair %s: NO heading convention matched. scores=%s "
                             "-- broken matcher or broken artifact, NOT a clean file." % (pair, scores))
            continue
        blocks = extract_blocks(text, mode)
        blocks = [b for b in blocks if defence(b["prompt"]) and defence(b["lyrics"])]
        print("\npair %s  mode=%-14s scores=%s  packages extracted=%d"
              % (pair, mode, scores, len(blocks)))
        if not blocks:
            hard_fail.append("pair %s: extracted ZERO packages (mode=%s). HARD ERROR, never a score."
                             % (pair, mode))
            continue
        if len(blocks) != VARIATIONS_PER_PAIR:
            hard_fail.append("pair %s: extracted %d packages, expected %d."
                             % (pair, len(blocks), VARIATIONS_PER_PAIR))

        for i, b in enumerate(blocks, 1):
            prom, lyr = defence(b["prompt"]), defence(b["lyrics"])
            title = defence(b["title"]).splitlines()[0].strip() if b["title"] else ""
            sl = sung_lines(lyr)
            r = {
                "pair": pair, "v": i, "title": title[:60],
                "prompt_chars": len(prom), "lyrics_chars": len(lyr), "sung": len(sl),
                "flags": [],
            }
            lo, hi = gates["prompt_band"]
            if not (lo <= r["prompt_chars"] <= hi):
                r["flags"].append("PROMPT_OUT_OF_BAND")
            if r["prompt_chars"] >= gates["hug"]:
                r["flags"].append("PROMPT_HUG")
            if not re.search(r"[.!?]\s*$", prom):
                r["flags"].append("NO_TERMINAL_PUNCT")
            if r["lyrics_chars"] >= gates["lyr_max"]:
                r["flags"].append("LYRICS_OVER_5000")
            lo, hi = gates["lines_band"]
            if not (lo <= r["sung"] <= hi):
                r["flags"].append("SUNG_LINES_OUT_OF_BAND")
            if r["sung"] <= gates["lines_hug"]:
                r["flags"].append("SUNG_FLOOR_HUG")
            if re.search(r"\bEMO:\s*(AWE|INDIGNATION)\b", lyr):
                r["flags"].append("BARE_MODE_IN_EMO")
            if not re.search(r"\[Theme:", lyr):
                r["flags"].append("NO_THEME_HEADER")
            if not re.search(r"\[SONG FORM:", lyr, re.I):
                r["flags"].append("NO_SONGFORM_HEADER")
            low = (prom + "\n" + lyr).lower()
            for ph in BANNED_PHRASES:
                if ph in low:
                    r["flags"].append("BAN_D2:%s" % ph)
            for rx in COHORT_WORDS:
                if re.search(rx, lyr, re.I):
                    r["flags"].append("COHORT_WORD:%s" % rx)
            for ph in HOUSE_LEXICON:
                if ph in low:
                    r["flags"].append("HOUSE_LEXICON:%s" % ph)
            # a sung numeral written as digits (digits inside [brackets] are fine -- unvoiced)
            for ln in sl:
                if re.search(r"(?<![\[\w])\d{2,}(?![\]\w])", ln):
                    r["flags"].append("SUNG_DIGITS:%s" % ln[:40])
                    break
            try:
                from measure_soundcraft import profile
                pr = profile(sl)
                if "end_rhyme" not in pr:                     # never accept a silent default
                    raise KeyError("profile() has no 'end_rhyme' key: %s" % sorted(pr))
                r["rhyme"] = round(pr["end_rhyme"], 3)
                r["lret"]  = round(pr.get("line_return", 0.0), 3)
                r["wpl"]   = round(pr.get("words_per_line", 0.0), 2)
                r["allit"] = round(pr.get("allit_per_100w", 0.0), 2)
                if r["rhyme"] < gates["rhyme_floor"]: r["flags"].append("RHYME_BELOW_FLOOR")
                if r["lret"]  < gates["line_floor"]:  r["flags"].append("LINE_RETURN_BELOW_FLOOR")
                if r["wpl"]   > gates["wpl_ceiling"]: r["flags"].append("WPL_OVER_CEILING")
                if r["allit"] < gates["allit_floor"]: r["flags"].append("ALLIT_BELOW_FLOOR")
            except Exception as e:
                r["soundcraft_error"] = str(e)[:80]
            rows.append(r)
            all_lyrics["%s.%d" % (pair, i)] = lyr
            all_prompts["%s.%d" % (pair, i)] = prom
            print("   v%d  %-40s prompt=%4d  lyrics=%4d  sung=%3d  %s"
                  % (i, (title[:38] or "(no title)"), r["prompt_chars"], r["lyrics_chars"],
                     r["sung"], ",".join(r["flags"]) or "-"))

    # (4) CARDINALITY ASSERTION -- before any verdict.
    print("\n" + "=" * 78)
    print("EXTRACTION TOTAL: %d packages (expected %d)" % (len(rows), EXPECTED_PACKAGES))
    if len(rows) != EXPECTED_PACKAGES:
        hard_fail.append("portfolio cardinality: %d != %d" % (len(rows), EXPECTED_PACKAGES))

    # cross-pair distinctiveness -- only ACROSS pairs, never within
    print("\nCROSS-PAIR DISTINCTIVENESS (ceilings: lyric %.2f / prompt %.2f)"
          % (gates["lyric_sim"], gates["prompt_sim"]))
    worst_l = worst_p = (0.0, "", "")
    n = 0
    for a in sorted(all_lyrics):
        for b in sorted(all_lyrics):
            if a >= b or a.split(".")[0] == b.split(".")[0]:
                continue
            n += 1
            sl_ = difflib.SequenceMatcher(None, all_lyrics[a], all_lyrics[b]).ratio()
            sp_ = difflib.SequenceMatcher(None, all_prompts[a], all_prompts[b]).ratio()
            if sl_ > worst_l[0]: worst_l = (sl_, a, b)
            if sp_ > worst_p[0]: worst_p = (sp_, a, b)
    print("  comparisons: %d" % n)
    print("  worst lyric : %.3f  (%s vs %s)  %s" % (worst_l[0], worst_l[1], worst_l[2],
          "BREACH" if worst_l[0] > gates["lyric_sim"] else "ok"))
    print("  worst prompt: %.3f  (%s vs %s)  %s" % (worst_p[0], worst_p[1], worst_p[2],
          "BREACH" if worst_p[0] > gates["prompt_sim"] else "ok"))
    if worst_l[0] > gates["lyric_sim"]:  hard_fail.append("lyric similarity breach %.3f" % worst_l[0])
    if worst_p[0] > gates["prompt_sim"]: hard_fail.append("prompt similarity breach %.3f" % worst_p[0])

    print("\n" + "=" * 78)
    flagged = [r for r in rows if r["flags"]]
    print("packages with >=1 flag: %d of %d" % (len(flagged), len(rows)))
    for h in hard_fail:
        print("  HARD: %s" % h)
    print("VERDICT:", "CLEAN" if (not hard_fail and not flagged) else
                      ("FLAGS ONLY" if not hard_fail else "HARD FAIL"))
    io.open(os.path.join(RUN, "GATE_REPORT.json"), "w", encoding="utf-8").write(
        json.dumps({"tier": tier, "files": [os.path.basename(p) for p in paths],
                    "rows": rows, "hard_fail": hard_fail,
                    "worst_lyric_sim": worst_l[0], "worst_prompt_sim": worst_p[0],
                    "comparisons": n}, indent=2))
    print("wrote GATE_REPORT.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
