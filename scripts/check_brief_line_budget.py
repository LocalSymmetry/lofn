#!/usr/bin/env python3
"""check_brief_line_budget.py -- does the section architecture we PRESCRIBED admit
any song that clears the sung-line floor?

WHY THIS EXISTS. On 2026-08-26 (`2026-08-26-pantheon-rare-registers`) FIVE OF SIX
isolation briefs handed a pair an exact section skeleton -- named sections with
exact stanza lengths -- and never once evaluated what those numbers multiply out
to. Two pairs discovered before writing a line that the arithmetic does not close:

    P05: Refrain x6 + fields 8/6/4/2. The 20 field lines and the SIX refrains are
         fixed; the refrain LENGTH is not. At 7 lines that is 62 against a floor
         of 70. It had to spend a PERMISSION CLAUSE deviation to ship.
    P06: stanzas 6/5/4/3 = 18 fixed lines + 3 choruses. Clearing 70 needs a
         SEVENTEEN-line chorus. It reinterpreted every stanza count as PAIRS.

The brief states the floor in one section and a skeleton in another, and nothing
multiplies. ⛔ A constraint set that has to be adjudicated is a constraint set that
was not finished -- and this one was not even evaluated.

⭐ SO THE CHECK IS THE DECLARATION, NOT A PARSER. A prose matcher over six freely
worded briefs is exactly the brittle instrument that produced four false positives
on the day this was found. Instead, every isolation brief must carry ONE line:

    LINE BUDGET: 20 + 6*9 = 74

The reader evaluates the left side, checks it equals the stated total, and checks
the total lands inside the sung-line band. **The act of writing the line IS the
check** -- the coordinator cannot state a budget without doing the multiplication.

    python3 scripts/check_brief_line_budget.py <run-dir> [--gates PATH]

`--gates PATH` (2026-09-07) reads the sung-line band from an explicitly declared
run-local gates file instead of vault/gates.yaml, mirroring validate_step.py.

`--pairs 01 02 03` (2026-10-16, `2026-10-16_lofn-unmix-insane`) measures only the named
pairs' briefs. A MIXED-FORM run declares two gates files (sung pairs 32-64 lines,
instrumental pairs 0), and one band cannot judge both: without this flag every run
of the reader fails the other form's briefs, and a reader that always exits 3 is a
reader nobody reads. Run it once per form, each with its own gates file and pairs:

    python3 scripts/check_brief_line_budget.py <run> --gates RUN_GATES.yaml --pairs 01 02 03
    python3 scripts/check_brief_line_budget.py <run> --gates RUN_GATES_INSTRUMENTAL.yaml --pairs 04 05 06

A named pair with no brief is reported MISSING (exit 3), never skipped.

⭐ WORD BUDGET (V6-19, 2026-11-25). Length is budgeted in SUNG WORDS since 2026-11-25:
about 400 sung words gives a usual 3:30, and once words are known the line count adds
nothing (output/analysis/2026-11-25_v6-330-length-budget/RESULT.md). Under a gates file
with `length_gate: words`, every brief must also carry ONE line, in the same grammar:

    WORD BUDGET: 20*6 + 6*9*5 = 390

and the total must land inside `sung_words_hard_floor`..`sung_words_hard_ceiling`. The
LINE BUDGET line is still required, because it is the skeleton's arithmetic; its band
is then the advisory `sung_lines` range. A gates file that declares `sung_lines` and no
`length_gate` is read as it was written (lines gate, no word line required).

Exit 0 = every brief declares a budget that closes. Exit 3 = a brief is missing the
declaration, or its arithmetic is wrong, or its total cannot reach the floor.
Exit 2 = no briefs found, which is reported rather than passed.

Read-only. Writes nothing, renders nothing, spends nothing.
"""
import os
import sys
import json
import re

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LABEL = "LINE BUDGET:"
WORD_LABEL = "WORD BUDGET:"
ALLOWED = set("0123456789+*() ")


def band(repo_root, gates_path=None):
    """The sung-line band, from vault/gates.yaml. Never hard-code a threshold.

    2026-09-07: `gates_path` lets a run point this reader at an explicitly declared
    run-local gates file (the same `--gates PATH` validate_step.py accepts), so a form
    exception declared BEFORE writing is read by the same instrument that checks the
    briefs. The default is still vault/gates.yaml; nothing changes for runs that do
    not declare one. The effective band is printed so the declaration is visible.
    """
    path = gates_path or os.path.join(repo_root, "vault", "gates.yaml")
    lo, hi = 70, 120
    try:
        with open(path, encoding="utf-8") as source:
            lines = source.readlines()
        for index, line in enumerate(lines):
            s = line.split("#", 1)[0].strip()
            if s.startswith("sung_lines:"):
                value = s.split(":", 1)[1].strip()
                if value:
                    nums = [int(t.strip()) for t in value.strip("[]").split(",")]
                else:
                    # YAML serializers may emit the same band as a block list.
                    nums = []
                    for following in lines[index + 1:]:
                        item = following.split("#", 1)[0].strip()
                        if not item:
                            continue
                        match = re.fullmatch(r"-\s+(\d+)", item)
                        if not match:
                            break
                        nums.append(int(match.group(1)))
                if len(nums) == 2:
                    lo, hi = nums
                break
    except OSError:
        pass
    return lo, hi


def word_gate(repo_root, gates_path=None):
    """-> (mode, lo, hi). `mode` is 'words' or 'lines'; lo/hi are the hard word bounds.

    Same rule as validate_step.load_gates: a gates file that declares `sung_lines` and no
    `length_gate` was written when lines were the gate.
    """
    path = gates_path or os.path.join(repo_root, "vault", "gates.yaml")
    mode, lo, hi, saw_lines = None, 300, 520, False
    try:
        with open(path, encoding="utf-8") as source:
            for line in source:
                s = line.split("#", 1)[0].rstrip()
                m = re.match(r"length_gate:\s*(words|lines)\s*$", s, re.I)
                if m:
                    mode = m.group(1).lower()
                m = re.match(r"sung_words_hard_floor:\s*(\d+)\s*$", s)
                if m:
                    lo = int(m.group(1))
                m = re.match(r"sung_words_hard_ceiling:\s*(\d+)\s*$", s)
                if m:
                    hi = int(m.group(1))
                if re.match(r"sung_lines:", s):
                    saw_lines = True
    except OSError:
        pass
    if mode is None:
        mode = "lines" if saw_lines else "words"
    return mode, lo, hi


def evaluate(expr):
    """Evaluate a budget expression. Digits, + * and parentheses only -- nothing
    else is accepted, so there is no code path from a brief into this process."""
    if not expr or not set(expr) <= ALLOWED:
        return None
    try:
        return int(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def declarations(text, label=None):
    """Read explicit declarations, including JSON string values, never prose.

    A seed may declare four different legal variation budgets. Check all four;
    JSON's closing quote/comma is serialization, not part of the arithmetic.
    """
    marker = "# ===== END CREATIVE CONTEXT ====="
    if marker in text:
        text = text.rsplit(marker, 1)[1]
    label = label or LABEL
    found = []
    for line in text.splitlines():
        if label not in line:
            continue
        value = re.search(r'"(?:line_budget|word_budget|budget)"\s*:\s*("(?:\\.|[^"\\])*")\s*,?\s*$', line)
        if value:
            line = json.loads(value.group(1))
        stated = line.split(label, 1)[1].strip().strip("*` ")
        if stated not in found:
            found.append(stated)
    return found


def budget_result(stated, lo, hi):
    if "=" not in stated:
        return False, "MALFORMED -- '" + stated + "' (want: <expr> = <total>)"
    expr, total = stated.rsplit("=", 1)
    got = evaluate(expr.strip())
    try:
        claimed = int(total.strip())
    except ValueError:
        claimed = None
    if got is None or claimed is None:
        return False, "UNPARSEABLE -- '" + stated + "'"
    if got != claimed:
        return False, "ARITHMETIC WRONG -- " + expr.strip() + " is " + str(got) + ", declared " + str(claimed)
    if not (lo <= claimed <= hi):
        return False, "OUT OF BAND -- " + str(claimed) + " not in " + str(lo) + "-" + str(hi) + "  (" + stated + ")"
    return True, "OK -- " + stated


def main():
    argv = list(sys.argv[1:])
    gates_path = None
    if "--gates" in argv:
        i = argv.index("--gates")
        if i + 1 >= len(argv):
            print("usage: check_brief_line_budget.py <run-dir> [--gates PATH]")
            return 2
        gates_path = argv[i + 1]
        del argv[i:i + 2]
        if not os.path.isfile(gates_path):
            print("FAIL: --gates file not found: " + gates_path)
            return 2
    only = None
    if "--pairs" in argv:
        i = argv.index("--pairs")
        j = i + 1
        while j < len(argv) and argv[j].isdigit():
            j += 1
        only = [p.zfill(2) for p in argv[i + 1:j]]
        del argv[i:j]
        if not only:
            print("usage: check_brief_line_budget.py <run-dir> [--gates PATH] [--pairs 01 02 03]")
            return 2
    if len(argv) != 1:
        print("usage: check_brief_line_budget.py <run-dir> [--gates PATH] [--pairs 01 02 03]")
        return 2
    run = argv[0]
    if not os.path.isdir(run):
        print("FAIL: not a directory: " + run)
        return 2

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lo, hi = band(repo_root, gates_path)
    mode, wlo, whi = word_gate(repo_root, gates_path)

    briefs = sorted(f for f in os.listdir(run)
                    if f.startswith("pair_") and "step05_isolated_brief" in f)
    missing_named = []
    if only is not None:
        briefs = [f for f in briefs if f.split("_")[1] in only]
        missing_named = [p for p in only if not any(f.split("_")[1] == p for f in briefs)]
    if not briefs:
        print("FAIL: no isolation briefs found -- nothing measured.")
        print("      A checker that measures nothing must not return success.")
        return 2

    print("BRIEF LINE BUDGET -- band " + str(lo) + "-" + str(hi) + " sung lines"
          + ("  (advisory range; length gate is words)" if mode == "words" else "")
          + ("  [run-local gates: " + gates_path + "]" if gates_path else "  [vault/gates.yaml]"))
    if mode == "words":
        print("BRIEF WORD BUDGET -- band " + str(wlo) + "-" + str(whi) + " sung words")
    bad = []
    for f in briefs:
        pair = f.split("_")[1]
        with open(os.path.join(run, f), encoding="utf-8") as handle:
            brief_text = handle.read()
        stated_budgets = declarations(brief_text)
        if mode == "words":
            word_budgets = declarations(brief_text, WORD_LABEL)
            if not word_budgets:
                print("  P" + pair + ": MISSING -- no '" + WORD_LABEL + "' declaration")
                if pair not in bad:
                    bad.append(pair)
            for index, stated in enumerate(word_budgets, 1):
                ok, message = budget_result(stated, wlo, whi)
                suffix = "[" + str(index) + "/" + str(len(word_budgets)) + "]" if len(word_budgets) > 1 else ""
                print("  P" + pair + suffix + " words: " + message)
                if not ok and pair not in bad:
                    bad.append(pair)
        if not stated_budgets:
            print("  P" + pair + ": MISSING -- no '" + LABEL + "' declaration")
            if pair not in bad:
                bad.append(pair)
            continue
        for index, stated in enumerate(stated_budgets, 1):
            ok, message = budget_result(stated, lo, hi)
            suffix = "[" + str(index) + "/" + str(len(stated_budgets)) + "]" if len(stated_budgets) > 1 else ""
            print("  P" + pair + suffix + ": " + message)
            if not ok and pair not in bad:
                bad.append(pair)

    for pair in missing_named:
        print("  P" + pair + ": MISSING -- named in --pairs, but no isolation brief exists")
        bad.append(pair)

    print()
    if bad:
        print("  RESULT: " + str(len(bad)) + " of " + str(len(briefs) + len(missing_named))
              + " brief(s) do not close: " + ", ".join("P" + p for p in bad))
        print("  Fix the skeleton BEFORE dispatch. A pair that has to break your")
        print("  arithmetic before writing a line is a pair you sent out unfinished.")
        return 3
    print("  RESULT: all " + str(len(briefs)) + " briefs declare a budget that closes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
