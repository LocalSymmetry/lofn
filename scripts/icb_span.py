"""One implementation of "remove the Immutable Continuity Block (ICB) from an artifact before parsing it".

Import this; never re-derive it. History: four readers each re-derived the step and disagreed —
  * check_palette_floor.py stripped an exact byte-0 prefix and reported ERROR for any package without one
    (measured 2026-10-01: it refused all six finals of output/2026-09-30_forge-hub-instrumentals);
  * check_field_census.py stripped an exact prefix only if the file started with it;
  * check_instrumental_package.py stripped a marker-delimited span anywhere in the file;
  * check_generator_negatives.py did nothing, and read the ICB's own template headings as a lyrics field (exit 2).
A step-11 final MUST begin with the run's full ICB, byte-exact (skills/music/steps/11, "Immutable-prefix contract":
"fix the reader, never drop the prefix"), so the normal case is "exact". An attested final (a sha256 line instead
of the ICB, as on 2026-09-30) still reads correctly and returns how=None.
(Docstring corrected 2026-10-01: it said that finals carry an attestation instead of the ICB, which contradicted the
contract it cites. A step-11 author on 2026-10-01_caterpillar-drawl-funk read the contradiction and asked which
rule held.)

The ICB is the run's `CREATIVE_CONTEXT.md`. Its content sits between the OVERALL_PROMPT_TEMPLATE markers
`# ===== BEGIN CREATIVE CONTEXT ... =====` and `# ===== END CREATIVE CONTEXT =====`.
"""
import os
import re

MARKED = re.compile(r"# ===== BEGIN CREATIVE CONTEXT.*?# ===== END CREATIVE CONTEXT =====[^\n]*\n?", re.S)


def load_icb(run_dir):
    """The run's frozen ICB text, or "" when the run has no CREATIVE_CONTEXT.md."""
    p = os.path.join(run_dir, "CREATIVE_CONTEXT.md")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def strip_icb(text, icb=""):
    """-> (text_without_the_icb, how), how in {"exact", "marked", None}.

    exact:  the run's own ICB text is removed wherever it occurs (prefix or embedded); tolerant of CRLF/LF drift
            between the artifact and CREATIVE_CONTEXT.md (in that case the returned text is LF-normalized).
    marked: otherwise, every span between the template's BEGIN/END CREATIVE CONTEXT markers is removed.
    None:   nothing to strip; the text is returned unchanged (the attested-final case).
    """
    if icb and icb.strip():
        if icb in text:
            return text.replace(icb, ""), "exact"
        t, i = text.replace("\r\n", "\n"), icb.replace("\r\n", "\n")
        if i in t:
            return t.replace(i, ""), "exact"
    out, n = MARKED.subn("", text)
    if n:
        return out, "marked"
    return text, None
