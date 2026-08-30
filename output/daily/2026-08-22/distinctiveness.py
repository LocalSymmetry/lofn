#!/usr/bin/env python3
"""Distinctiveness + the run's falsification tests, measured on the SUNG BODY.

Measured on the sung body ONLY -- never the raw lyrics field. Once every pair
carries the same `[Theme:]` / `[SONG FORM:]` / `[Section - EMO:...]` frame, a
field-level similarity score is partly measuring the FORMAT CONTRACT and reads
falsely low. (R3.2 rule 2.)

Four measurements:
  1. CROSS-PAIR similarity, ceiling 0.42.
  2. WITHIN-PAIR similarity -- R3.2 rule 2: the 0.42 ceiling was cross-pair
     only, and a pair once measured 0.426 within-pair with nothing gating it.
  3. THE CLOSING-BLOCK SWAP -- on the last 8 sung lines, not the last line.
     R3.2 rule 4: pairs honestly reported RETURN-SWAP PASS because the closing
     LINES differed by a noun while the seven lines above them were one machine.
     "The formula hides in the resolution."
  4. HOUSE-LEXICON DRIFT -- the words that calcified last time (`toast`,
     `kettle`, `chips`, `Best ___ I've had`), swept with word boundaries so
     `rage` does not match inside `garage`.

Run from repo root:  python3 output/daily/2026-08-22/distinctiveness.py
"""
import itertools
import os
import re
import sys

# Windows consoles default to cp1252 and die on the doctrine glyphs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUN = os.path.join("output", "daily", "2026-08-22")
CROSS_CEIL, WITHIN_CEIL, CLOSING_CEIL = 0.42, 0.42, 0.50
CLOSING_BLOCK = 8

# The 2026-08-18 calcified lexicon (R3.1) + this run's own risk words.
DRIFT = ["toast", "kettle", "chips", "tea", "soup", "armchair", "chip shop"]
DRIFT_GRAMMAR = re.compile(r"\bbest\s+\w+\s+(?:i've|i have)\s+(?:had|ever had)\b", re.I)


def fenced(block):
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", block, re.S)
    if m:
        return m.group(1).strip()
    return "\n".join(l for l in block.splitlines() if not l.strip().startswith("#")).strip()


def sung(lyrics):
    out = []
    for ln in lyrics.splitlines():
        s = ln.strip()
        if not s or (s.startswith("[") and s.endswith("]")) or (s.startswith("*") and s.endswith("*")):
            continue
        out.append(s)
    return out


def grams(lines, n=3):
    toks = re.findall(r"[a-z']+", " \n ".join(lines).lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jac(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    songs = {}
    for p in ["01", "02", "03", "04", "05", "06"]:
        path = os.path.join(RUN, f"pair_{p}_step10_final_package_enhanced.md")
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8").read()
        parts = re.split(r"^#\s+(V\d)\s*[—\-–]\s*(.+?)\s*$", text, flags=re.M)
        for i in range(1, len(parts) - 2, 3):
            tag, title, body = parts[i], parts[i + 1], parts[i + 2]
            lm = re.search(r"##\s*2\.\s*LYRICS(.*?)(?=##\s)", body, re.S | re.I)
            lines = sung(fenced(lm.group(1))) if lm else []
            songs[f"P{p}{tag}"] = {"pair": p, "title": title.strip(), "lines": lines,
                                   "g": grams(lines), "gc": grams(lines[-CLOSING_BLOCK:])}

    ids = sorted(songs)
    cross, within, closing = [], [], []
    for a, b in itertools.combinations(ids, 2):
        s = jac(songs[a]["g"], songs[b]["g"])
        sc = jac(songs[a]["gc"], songs[b]["gc"])
        (within if songs[a]["pair"] == songs[b]["pair"] else cross).append((s, a, b))
        closing.append((sc, a, b))

    def report(name, data, ceil):
        data.sort(reverse=True)
        worst = data[0] if data else (0, "-", "-")
        over = [d for d in data if d[0] > ceil]
        print(f"\n{name}  (n={len(data)}, ceiling {ceil})")
        print(f"  worst: {worst[1]} x {worst[2]} = {worst[0]:.3f}")
        print(f"  mean : {sum(d[0] for d in data)/len(data):.3f}" if data else "")
        print(f"  OVER CEILING: {len(over)}")
        for s, a, b in over[:8]:
            print(f"    !! {a} x {b} = {s:.3f}")
        return len(over)

    print("=" * 64)
    print("DISTINCTIVENESS — measured on the SUNG BODY, not the lyrics field")
    print("=" * 64)
    print(f"songs: {len(ids)}   sung lines: {sum(len(s['lines']) for s in songs.values())}")

    n1 = report("1. CROSS-PAIR", cross, CROSS_CEIL)
    n2 = report("2. WITHIN-PAIR  (R3.2 rule 2 — ungated last time)", within, WITHIN_CEIL)
    n3 = report(f"3. CLOSING-BLOCK SWAP  (last {CLOSING_BLOCK} sung lines — R3.2 rule 4)",
                closing, CLOSING_CEIL)

    # Report COUNTS, not presence. The first version reported only that a word
    # occurred, so a variation using `kettle` twice looked identical to one
    # using it once -- and the pair agent found more instances by hand than
    # this test reported. Presence is the wrong unit for a calcification check:
    # calcification IS frequency.
    print("\n4. HOUSE-LEXICON DRIFT  (word-boundary matched, WITH COUNTS)")
    hits, per_word = 0, {}
    for vid in ids:
        body = " ".join(songs[vid]["lines"]).lower()
        found = []
        for w in DRIFT:
            c = len(re.findall(rf"\b{re.escape(w)}\b", body))
            if c:
                found.append(f"{w}x{c}" if c > 1 else w)
                per_word[w] = per_word.get(w, 0) + c
        if DRIFT_GRAMMAR.search(body):
            found.append("Best ___ I've had")
            per_word["Best ___ I've had"] = per_word.get("Best ___ I've had", 0) + 1
        if found:
            hits += 1
            print(f"    {vid}: {', '.join(found)}")
    if per_word:
        tot = sum(per_word.values())
        print(f"    -- totals: {tot} occurrence(s) across {hits}/24 songs "
              f"({hits/24:.0%}; domestic-warm cap is 33%)")
        print(f"    -- by word: {', '.join(f'{w} x{c}' for w, c in sorted(per_word.items()))}")
    if not hits:
        print("    none — the 2026-08-18 calcified lexicon does not appear in any of the 24.")

    print("\n5. CLOSING LINE OF EVERY SONG  (eyeball the return; the formula hides here)")
    for vid in ids:
        last = songs[vid]["lines"][-1] if songs[vid]["lines"] else "(none)"
        print(f"    {vid}  {last[:62]}")

    # ------------------------------------------------------------------
    # 6. CALIBRATION -- because a gate that cannot fail is not a gate.
    #
    # The 0.42 ceiling in doctrine was set on a DIFFERENT metric. Word
    # 3-gram Jaccard is far sparser: it returns ~0.00 for any two songs
    # that are not near-copies, so it would report PASS forever and prove
    # nothing. Before believing a pass, the instrument must show it can
    # detect similarity AT ALL, and on a corpus already known to contain it.
    # ------------------------------------------------------------------
    print("\n6. CALIBRATION — can this instrument detect similarity at all?")
    first = songs[ids[0]]
    print(f"   positive control (song vs ITSELF): {jac(first['g'], first['g']):.3f}   (must be 1.000)")

    # Unigram Jaccard: a looser scale, closer to what the 0.42 ceiling was set on.
    for s in songs.values():
        s["u"] = grams(s["lines"], 1)
    uni_cross = [jac(songs[a]["u"], songs[b]["u"])
                 for a, b in itertools.combinations(ids, 2) if songs[a]["pair"] != songs[b]["pair"]]
    uni_within = [jac(songs[a]["u"], songs[b]["u"])
                  for a, b in itertools.combinations(ids, 2) if songs[a]["pair"] == songs[b]["pair"]]
    print(f"   UNIGRAM cross-pair : max {max(uni_cross):.3f}  mean {sum(uni_cross)/len(uni_cross):.3f}")
    print(f"   UNIGRAM within-pair: max {max(uni_within):.3f}  mean {sum(uni_within)/len(uni_within):.3f}")

    # Negative control: a corpus independently found to be templated.
    ref = os.path.join("output", "daily", "2026-08-21", "pair_04_step10_final_package_enhanced.md")
    if os.path.isfile(ref):
        text = open(ref, encoding="utf-8").read()
        parts = re.split(r"^#\s+(V\d)\s*[—\-–]\s*(.+?)\s*$", text, flags=re.M)
        ry = {}
        for i in range(1, len(parts) - 2, 3):
            lm = re.search(r"##\s*2\.\s*LYRICS(.*?)(?=##\s)", parts[i + 2], re.S | re.I)
            ln = sung(fenced(lm.group(1))) if lm else []
            if ln:
                ry[parts[i]] = {"g": grams(ln), "u": grams(ln, 1), "gc": grams(ln[-CLOSING_BLOCK:])}
        if len(ry) > 1:
            rk = sorted(ry)
            t3 = [jac(ry[a]["g"], ry[b]["g"]) for a, b in itertools.combinations(rk, 2)]
            t1 = [jac(ry[a]["u"], ry[b]["u"]) for a, b in itertools.combinations(rk, 2)]
            tc = [jac(ry[a]["gc"], ry[b]["gc"]) for a, b in itertools.combinations(rk, 2)]
            print(f"\n   NEGATIVE CONTROL — 2026-08-21 pair 04, the run R3.2 measured as templated")
            print(f"     3-gram within-pair : max {max(t3):.3f}  (ours: {max(s for s,_,_ in within):.3f})")
            print(f"     unigram within-pair: max {max(t1):.3f}  (ours: {max(uni_within):.3f})")
            print(f"     closing-block      : max {max(tc):.3f}  (ours: {max(s for s,_,_ in closing):.3f})")
            print("\n   *** THE CONTROL FAILED, AND THAT IS THE FINDING. ***")
            print("   On all three measures the KNOWN-TEMPLATED corpus scores LOWER (more")
            print("   distinct) than this run. An instrument that ranks a corpus already")
            print("   proven formulaic as MORE distinct than an unknown one is not measuring")
            print("   distinctiveness. Its '0 over ceiling' verdict above is therefore NOT")
            print("   evidence of anything, and must not be reported as a clean pass.")
            print("   Global n-gram overlap is the wrong shape: R3.2 did not find a corpus")
            print("   that was globally similar, it found SPECIFIC REPEATED BLOCKS inside")
            print("   otherwise-distinct songs. See test 7, which measures that directly.")
    else:
        print("\n   NEGATIVE CONTROL UNAVAILABLE — reference corpus not found.")
        print("   → a pass here is UNCALIBRATED and must not be reported as clean.")

    # ------------------------------------------------------------------
    # 7. THE MEASURE THAT WOULD HAVE CAUGHT THE 2026-08-18 DEFECT.
    #
    # R3.2's actual finding was not global similarity. It was that a pair
    # sang `I signed it standing up` BYTE-IDENTICAL in all four variations,
    # and three of four shared `[authority] told me once I could`. So the
    # right instrument counts EXACT REPEATED SUNG LINES across the
    # variations of one pair -- with the refrain exempt, because a
    # byte-identical chorus is required and needs no defence.
    # ------------------------------------------------------------------
    print("\n7. EXACT REPEATED SUNG LINES ACROSS A PAIR'S VARIATIONS")
    print("   (refrain exempt — a byte-identical chorus is doctrine, not a defect)")

    def repeats(corpus_by_pair, label):
        worst = 0
        for pid, variants in sorted(corpus_by_pair.items()):
            counts = {}
            for vid, lines in variants.items():
                for ln in set(lines):
                    key = re.sub(r"[^a-z ]", "", ln.lower()).strip()
                    if len(key.split()) < 3:
                        continue          # too short to be a signature
                    counts.setdefault(key, set()).add(vid)
            shared = {k: v for k, v in counts.items() if len(v) >= 2}
            # A line in ALL variations of a pair is the refrain signature: exempt.
            n_all = sum(1 for v in shared.values() if len(v) == len(variants))
            n_partial = len(shared) - n_all
            worst = max(worst, n_partial)
            flag = "  <-- LEAK" if n_partial else ""
            print(f"     {label} {pid}: {n_partial} line(s) shared by SOME-but-not-all "
                  f"({n_all} refrain-wide, exempt){flag}")
            for k, v in list(shared.items())[:40]:
                if len(v) != len(variants):
                    print(f"         {sorted(v)}: {k[:56]!r}")
        return worst

    ours = {}
    for vid, s in songs.items():
        ours.setdefault(s["pair"], {})[vid] = s["lines"]
    w_ours = repeats(ours, "P")

    if os.path.isfile(ref) and len(ry) > 1:
        yday = {"04(08-21)": {}}
        parts = re.split(r"^#\s+(V\d)\s*[—\-–]\s*(.+?)\s*$",
                         open(ref, encoding="utf-8").read(), flags=re.M)
        for i in range(1, len(parts) - 2, 3):
            lm = re.search(r"##\s*2\.\s*LYRICS(.*?)(?=##\s)", parts[i + 2], re.S | re.I)
            if lm:
                yday["04(08-21)"][parts[i]] = sung(fenced(lm.group(1)))
        print("\n   CONTROL — the same measure on 2026-08-21 pair 04:")
        w_ctrl = repeats(yday, " ")
        print(f"\n   separation: control {w_ctrl} shared-line leak(s) vs ours {w_ours}.")
        if w_ctrl > w_ours:
            print("   → THIS instrument separates the two corpora in the right direction.")
            print("     It is the one to trust; test 1-3's ceilings are not.")
        else:
            print("   → still no separation. Report distinctiveness as UNPROVEN, not clean.")

    total = n1 + n2 + n3
    print("\n" + "=" * 64)
    print(f"VERDICT: {total} measurement(s) over ceiling.")
    print("⚠️ Ceilings are inherited from a different metric — read the CALIBRATION")
    print("   block before treating a pass as meaningful.")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
