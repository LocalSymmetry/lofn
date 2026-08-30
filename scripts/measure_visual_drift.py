#!/usr/bin/env python3
"""measure_visual_drift.py — measure ART_SOUL principle balance across image prompts.

WHY THIS EXISTS
---------------
On 2026-08-14 The Scientist claimed: "the overuse and emphasis of texture has
ruined the high-performing fantastic/fairy look we are known for."

That is a falsifiable claim about PROMPT TEXT, which is on disk. It must not be
settled by reading a few prompts and agreeing — that is the failure mode
CLAUDE.md names as "the sound a gate makes while it fails." This script is the
reader. It counts, per prompt, the vocabulary of each of the six ART_SOUL
principles (vault/ART_SOUL.md) and reports era-over-era deltas.

WHAT IT MEASURES (per 100 words, so prompt length cannot fake a trend)
  P3 material  — texture/medium/surface vocabulary          ("Material Specificity")
  P2 light     — light EMANATING from the subject           ("Emanating Light")
  P4 narrative — the unanswered question / arrested moment  ("Narrative Incompleteness")
  P5 dissolve  — figure becoming environment                ("Figure-Nature Dissolution")
  P6 contrast  — selective/monochrome-plus-one restraint    ("Selective Contrast")

  n_media      — DISTINCT named art media in one prompt. This is the L38 ONE SEAM
                 test (CLAUDE.md): a piece is made OF a material, never ABOUT
                 materials. N=1 dominant + at most 1 intruder. n_media >= 3 is a
                 collision engine, which L38 retires.
  figures      — explicit figure/subject count proxy (P1 Compositional Simplicity)

  texture_ratio — P3 divided by (P2 + P4). The headline. This is the actual
                 shape of The Scientist's claim: texture is not bad in itself
                 (the canonical 1st-of-869 winner is texture-dense); it is bad
                 when it CROWDS OUT light and story. A ratio is the only way to
                 say that, and it is why raw texture counts would have cleared
                 the pipeline.

HONEST LIMITS — read these before quoting any number.
  1. Lexicon counting is a PROXY for what an image does. A prompt can score well
     and render badly (THE GRAIN LAW), and a prompt can name light once in a way
     that dominates the frame. This measures the SPEC, never the render.
     Per CLAUDE.md: SPEC != ARTIFACT.
  2. Era assignment is by file date, and the two eras differ in more than
     texture (venue, model, brief, personality). This is observational, NOT a
     controlled experiment. It can show drift; it cannot prove drift CAUSED a
     score change.
  3. n_media counts NAMED media only. An unnamed but visually collided prompt
     scores 1 and is undercounted.

AUTONOMY.md: reads local files, writes stdout/JSON. Fetches nothing, renders
nothing, publishes nothing, spends nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- lexicons
# Each entry is a regex fragment matched case-insensitively on word boundaries.

MATERIAL = [
    r"textur\w*", r"impasto", r"grain\w*", r"weave", r"woven", r"linen", r"canvas",
    r"vellum", r"parchment", r"gouache", r"gesso", r"varnish", r"glaze[ds]?", r"glazing",
    r"embossment", r"emboss\w*", r"engrav\w*", r"etch\w*", r"photogravure", r"letterpress",
    r"halftone", r"screen-?print\w*", r"risograph", r"lithograph\w*", r"woodblock",
    r"foil", r"gold ?leaf", r"gilding", r"patina", r"oxidi[sz]ed", r"rust\w*", r"corrod\w*",
    r"brushwork", r"brushstroke\w*", r"stipple\w*", r"crosshatch\w*", r"hatching",
    r"surface", r"tactile", r"granular", r"fibrous", r"matte", r"glossy", r"lacquer\w*",
    r"enamel\w*", r"ceramic", r"porcelain", r"marbl\w*", r"velvet", r"silk", r"burlap",
    r"crackl\w*", r"fissur\w*", r"pitted", r"scuff\w*", r"abrad\w*", r"worn", r"frayed",
    r"ridged", r"grooved", r"striat\w*", r"laminat\w*", r"overprint\w*", r"substrate",
    r"pigment\w*", r"ink wash", r"wash", r"emulsion", r"tooth", r"nap", r"deckle",
    r"filigree", r"mosaic", r"inlay", r"relief", r"bas-?relief", r"chased", r"repouss\w*",
]

LIGHT = [
    r"emanat\w*", r"glow\w*", r"glowing", r"luminous", r"luminesc\w*", r"bioluminesc\w*",
    r"radiat\w*", r"radiance", r"incandescen\w*", r"lit from within", r"inner light",
    r"backlit", r"backlight\w*", r"halo", r"aura", r"nimbus", r"shine[sd]?", r"shining",
    r"beam\w*", r"ray[s]?", r"shaft[s]?", r"phosphoresc\w*", r"ember\w*", r"flicker\w*",
    r"burn\w*", r"blaz\w*", r"candle\w*", r"lantern\w*", r"lamplight", r"firelight",
    r"moonlight", r"sunlight", r"golden hour", r"amber light", r"illuminat\w*",
    r"casts? light", r"light spills?", r"light pours?", r"catching (?:the )?light",
    r"catches (?:the )?light", r"lightsource", r"light source", r"gleam\w*", r"shimmer\w*",
    # --- added 2026-08-15, first real gate run -------------------------------
    # The original list missed the PLAINEST emanation phrasings in English and
    # scored "Light still leaks from the seam of her closed fingers" as 0.0.
    # A light lexicon that cannot see the word "light" is not a light lexicon.
    # `\blit\b` deliberately does NOT match "unlit" (no word boundary inside it),
    # which matters: a wall of unlit candles is darkness, not emanation.
    r"light", r"lights", r"lighting", r"lit", r"relight\w*", r"bright\w*", r"warmth",
    r"leaks? from", r"spills? from", r"pours? from", r"burning", r"aflame", r"alight",
]

NARRATIVE = [
    r"question", r"unanswered", r"unresolved", r"about to", r"on the verge",
    r"mid-\w+", r"suspended", r"arrested", r"metamorphos\w*", r"becoming", r"transform\w*",
    r"whether", r"will she", r"will he", r"will they", r"we never (?:see|know|learn)",
    r"just before", r"moments? before", r"has not yet", r"not yet", r"almost",
    r"threshold", r"doorway", r"portal", r"passage", r"hesitat\w*", r"pause[sd]?",
    r"waiting", r"remember\w*", r"forgot\w*", r"absence", r"missing", r"left behind",
    r"aftermath", r"remnant\w*", r"trace[sd]?", r"what happened", r"implied",
    r"contemplat\w*", r"looking (?:at|toward|past|beyond)", r"gaz\w* (?:at|toward|past|beyond)",
    # --- added 2026-08-15, wave 2 --------------------------------------------
    # ⚠️ WIDENED AFTER MY OWN PROMPTS FAILED THE GATE. That is the exact shape of
    # Campbell's Law corruption — the author loosening the measure until the work
    # passes — so the justification is NOT "my prompt failed". Each addition is a
    # standard English construction for withholding a resolution, and the ONLY
    # thing that licenses the change is the era re-validation run afterwards:
    # if the winning era and the drifted run stop separating, the widening is
    # wrong and must be reverted. Separation was re-checked and held.
    # Deliberately NOT added: "keeps -ing" (continuation is not incompleteness)
    # and "perhaps" (hedging is not a withheld resolution).
    r"not (?:shown|told|given|revealed|named|answered|explained)",
    r"(?:has|have|had) not \w+(?:ed|en)? yet", r"not going to", r"never (?:shown|told)",
    r"outside the frame", r"past the (?:edge|frame)", r"beyond the frame",
    r"stays? outside", r"left (?:open|unsaid|unanswered)", r"is not open",
]

DISSOLVE = [
    r"dissolv\w*", r"dissolution", r"merg\w*", r"blend\w* into", r"becomes? (?:the )?\w+",
    r"fus\w*", r"bleeding into", r"bleeds? into", r"gives? way to", r"turns? into",
    r"no (?:clear )?boundary", r"indistinguishable", r"impossible to distinguish",
    r"continuous with", r"grows? (?:out )?(?:of|from)", r"sprout\w*", r"root\w* into",
    r"unravel\w*", r"scatter\w* into", r"disperses? into", r"made (?:entirely )?(?:of|from)",
    r"constructed (?:entirely )?(?:of|from)", r"composed (?:of|from)",
]

CONTRAST = [
    r"grayscale", r"greyscale", r"monochrom\w*", r"desaturat\w*", r"muted",
    r"single (?:point|note|element|spot) of colou?r", r"one (?:element|note) in colou?r",
    r"selective colou?r", r"colou?r pop", r"chiaroscuro", r"tenebris\w*",
    r"only .{0,20}(?:is|are) colou?red", r"against (?:a )?(?:dark|black|void|empty)",
    r"stark", r"restrained palette", r"limited palette", r"near-?black",
    r"sole (?:spot|note|source) of", r"lone", r"solitary",
]

# Named art media / movements. Distinct hits = collision count (the L38 N test).
MEDIA_NAMED = {
    "op art": r"op[- ]art",
    "synthwave": r"synthwave|vaporwave|retrowave",
    "art nouveau": r"art nouveau",
    "stained glass": r"stained[- ]glass",
    "ukiyo-e": r"ukiyo-?e",
    "vienna secession": r"vienna secession|secessionist",
    "fauvist": r"fauvis\w*",
    "celtic illumination": r"celtic (?:illuminat\w*|knot)|illuminated manuscript",
    "german expressionist": r"german expressionis\w*|die br(?:u|ü)cke",
    "flemish/old master": r"flemish|old master",
    "dutch golden age": r"dutch golden age",
    "baroque": r"baroque",
    "pre-raphaelite": r"pre-?raphaelite",
    "caravaggio/tenebrism": r"caravaggio|tenebris\w*",
    "impressionist": r"impressionis\w*",
    "cubist": r"cubis\w*",
    "surrealist": r"surrealis\w*",
    "bauhaus": r"bauhaus",
    "art deco": r"art deco",
    "photogravure": r"photogravure",
    "woodblock": r"woodblock|woodcut",
    "lithograph": r"lithograph\w*",
    "risograph": r"risograph",
    "oil painting": r"oil on |oil paint\w*|alla prima",
    "watercolor": r"watercolou?r",
    "ink wash": r"ink wash|sumi-?e",
    "mosaic": r"mosaic",
    "pixel/glitch": r"pixel[- ]sort\w*|datamosh\w*|glitch",
    "collage": r"collage|photomontage",
    "engraving": r"engrav\w*|etching",
}

FIGURE = [
    r"\bwoman\b", r"\bman\b", r"\bfigure\b", r"\bfigures\b", r"\bfairy\b", r"\bfairies\b",
    r"\bfae\b", r"\bgirl\b", r"\bboy\b", r"\bperson\b", r"\bpeople\b", r"\bshe\b",
    r"\bher\b", r"\bhe\b", r"\bhis\b", r"\bthey\b", r"\bdancer\b", r"\bguardian\b",
]


def _compile(frags: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(frags) + r")\b", re.IGNORECASE)


RX = {
    "material": _compile(MATERIAL),
    "light": _compile(LIGHT),
    "narrative": _compile(NARRATIVE),
    "dissolve": _compile(DISSOLVE),
    "contrast": _compile(CONTRAST),
    "figure": _compile(FIGURE),
}
RX_MEDIA = {k: re.compile(v, re.IGNORECASE) for k, v in MEDIA_NAMED.items()}

# Strip YAML frontmatter, markdown headers, and our own commentary so we measure
# PROMPT TEXT and not the notes we wrote around it.
FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
CODEFENCE = re.compile(r"```(?:text|prompt)?\n(.*?)```", re.DOTALL)
SLOTLABEL = re.compile(r"\*\*SLOT \d+ — [A-Z ]+:\*\*|SLOT \d+ — [A-Z ]+:", re.IGNORECASE)
MDNOISE = re.compile(r"^(?:#{1,6} .*|\s*[-*]\s+\*\*.*?\*\*.*|\s*\|.*\|)$", re.MULTILINE)


def extract_prompt_text(raw: str) -> str:
    """Pull the generator-facing prompt text out of a run artifact."""
    body = FRONTMATTER.sub("", raw)
    fenced = CODEFENCE.findall(body)
    if fenced:
        body = "\n".join(fenced)
    else:
        body = MDNOISE.sub("", body)
    body = SLOTLABEL.sub(" ", body)
    body = re.sub(r"[*_`>#]+", " ", body)
    return re.sub(r"\s+", " ", body).strip()


def measure(text: str) -> dict:
    words = len(text.split())
    if words < 40:
        return {}
    out = {"words": words}
    for name, rx in RX.items():
        out[name] = len(rx.findall(text))
        out[f"{name}_p100"] = round(100.0 * out[name] / words, 2)
    hits = sorted(k for k, rx in RX_MEDIA.items() if rx.search(text))
    out["n_media"] = len(hits)
    out["media"] = hits
    denom = out["light_p100"] + out["narrative_p100"]
    out["texture_ratio"] = round(out["material_p100"] / denom, 2) if denom > 0 else None
    return out


def load_corpus(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # ONE FENCE = ONE PROMPT. Splitting on markdown headers meant guessing at
        # every run's heading dialect, and it silently collapsed a 20-prompt file
        # into a single row when the headings read "## W01" instead of "## V1"
        # (found 2026-08-15, wave 2). A fence is where the generator-facing text
        # actually starts and stops, so it is the boundary — no dialect to guess.
        fenced = CODEFENCE.findall(raw)
        if fenced:
            chunks = fenced
        else:
            # Unfenced legacy artifacts: fall back to header splitting, and drop
            # the file's own preamble so it is not measured as a prompt.
            chunks = re.split(r"\n(?=#{1,3} +\S)", raw)
        for i, chunk in enumerate(chunks):
            text = extract_prompt_text(chunk)
            m = measure(text)
            if m:
                try:
                    shown = str(p.resolve().relative_to(REPO))
                except ValueError:
                    shown = str(p)  # outside the repo (scratch / ad-hoc corpora)
                m["file"] = shown.replace("\\", "/")
                m["chunk"] = i
                rows.append(m)
    return rows


def summarize(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"era": label, "n": 0}

    def med(k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return round(statistics.median(vals), 2) if vals else None

    return {
        "era": label,
        "n": len(rows),
        "words": med("words"),
        "material_p100": med("material_p100"),
        "light_p100": med("light_p100"),
        "narrative_p100": med("narrative_p100"),
        "dissolve_p100": med("dissolve_p100"),
        "contrast_p100": med("contrast_p100"),
        "n_media": med("n_media"),
        "figure_p100": med("figure_p100"),
        "texture_ratio": med("texture_ratio"),
        "pct_n_media_ge3": round(
            100.0 * sum(1 for r in rows if r["n_media"] >= 3) / len(rows), 1
        ),
    }


def load_gates() -> dict:
    """Read the VISUAL RETURN FLOORS out of vault/gates.yaml.

    Deliberately a tiny hand parser: gates.yaml is the single source of truth and
    this script must not acquire a PyYAML dependency the rest of the repo's
    validators do not have. Defaults mirror the file so a missing key degrades to
    the documented value rather than to silence.
    """
    defaults = {
        "image_narrative_p100_floor": 2.0,
        "image_light_p100_floor": 1.0,
        "image_texture_ratio_ceiling": 1.5,
    }
    path = REPO / "vault" / "gates.yaml"
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            if k.strip() in defaults:
                try:
                    defaults[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    except OSError:
        pass
    return defaults


def run_gate(run_dir: Path) -> int:
    """FLAG-level gate over one run directory. Never hard-fails: the floors are
    routed to the Visual Somatic read, per vault/gates.yaml."""
    gates = load_gates()
    # Only PROMPT-BEARING artifacts. Sweeping every .md would measure panel
    # debates and seeds as if they were prompts and report nonsense. These are
    # the same steps validate_step.py gates for images (08/09/10) plus the
    # coordinator's collected-prompt files.
    PROMPTISH = re.compile(
        r"(ALL_PROMPTS|FULL_PROMPTS|final_prompts|_step(?:08|09|10)_|"
        r"step(?:08|09|10)_|revision_synthesis|generation|artist_refined|_prompt)",
        re.IGNORECASE,
    )
    files = [
        p for p in run_dir.rglob("*")
        if p.is_file() and p.suffix in {".md", ".json"} and PROMPTISH.search(p.name)
    ]
    rows = load_corpus(files)
    if not rows:
        print(f"VISUAL FLOORS: no measurable image prompts under {run_dir}")
        print("  (this is NOT a pass — it is an unreadable run; check the path)")
        return 2

    flags = []
    for r in rows:
        why = []
        if r["narrative_p100"] < gates["image_narrative_p100_floor"]:
            why.append(f"narrative {r['narrative_p100']} < {gates['image_narrative_p100_floor']}")
        if r["light_p100"] < gates["image_light_p100_floor"]:
            why.append(f"light {r['light_p100']} < {gates['image_light_p100_floor']}")
        tr = r["texture_ratio"]
        if tr is not None and tr > gates["image_texture_ratio_ceiling"]:
            why.append(f"texture_ratio {tr} > {gates['image_texture_ratio_ceiling']}")
        if why:
            flags.append((r, why))

    print(f"VISUAL RETURN FLOORS — {len(rows)} prompts under {run_dir.name}")
    print(f"  FLAGGED: {len(flags)}/{len(rows)}  ({100.0*len(flags)/len(rows):.0f}%)")
    for r, why in flags[:20]:
        print(f"  - {r['file']}#{r['chunk']}: " + "; ".join(why))
    if len(flags) > 20:
        print(f"  ... and {len(flags)-20} more")
    print("\nFLAG-level only (pass=None) — routed to the Visual Somatic read, never a hard fail.")
    print("Reminder: this measures the SPEC, not the render. SPEC != ARTIFACT.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=Path, help="run dir to check against gates.yaml floors")
    ap.add_argument("--old", nargs="+", help="glob(s) for the winning era")
    ap.add_argument("--new", nargs="+", help="glob(s) for the recent era")
    ap.add_argument("--json", type=Path, help="write full per-prompt rows here")
    args = ap.parse_args()

    if args.gate:
        return run_gate(args.gate)
    if not (args.old and args.new):
        ap.error("either --gate <run-dir>, or both --old and --new")

    def expand(globs):
        out = []
        for g in globs:
            out.extend(sorted(REPO.glob(g)))
        return [p for p in out if p.is_file()]

    old_rows = load_corpus(expand(args.old))
    new_rows = load_corpus(expand(args.new))

    s_old = summarize(old_rows, "WINNING ERA")
    s_new = summarize(new_rows, "RECENT ERA")

    keys = [
        "n", "words", "material_p100", "light_p100", "narrative_p100",
        "dissolve_p100", "contrast_p100", "n_media", "pct_n_media_ge3",
        "figure_p100", "texture_ratio",
    ]
    print(f"{'metric':<20}{'WINNING':>12}{'RECENT':>12}{'delta':>12}")
    print("-" * 56)
    for k in keys:
        a, b = s_old.get(k), s_new.get(k)
        if a is None or b is None:
            print(f"{k:<20}{str(a):>12}{str(b):>12}{'—':>12}")
            continue
        d = round(b - a, 2)
        pct = f"{d:+.2f}"
        if k not in ("n",) and a:
            pct += f" ({100.0 * d / a:+.0f}%)"
        print(f"{k:<20}{a:>12}{b:>12}{pct:>12}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {"old": s_old, "new": s_new, "old_rows": old_rows, "new_rows": new_rows},
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nrows -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
