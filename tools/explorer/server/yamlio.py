"""Byte-stable YAML round-trip — the engine behind acceptance gate #1.

Empirically validated (docs/explorer-ground-truth.md §3) against all 254 canonical
edit-surface files: 250 round-trip byte-identical; the 4 that don't are detected by
`can_roundtrip()` and served raw-text-only. Strategy:

  read bytes -> strip/record BOM -> record dominant EOL -> decode -> LF-normalize
  -> ruamel round-trip (per-file sequence-indent detection, no wrapping, quotes kept)
  -> dump -> re-apply EOL + BOM.

WRITES in this app are almost always raw bytes from the editor (files.py). ruamel is
used to (a) decide structured-vs-raw editing, (b) parse cards/tables for display, and
(c) the rare structured write (gates.yaml table mode, library regen).
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from ruamel.yaml import YAML

_SEQ_RE = re.compile(r"^(?P<indent> *)-(?: |$)", re.MULTILINE)
_BOM = b"\xef\xbb\xbf"


@dataclass
class FileMeta:
    eol: str          # "\r\n" | "\n"
    bom: bool
    seq_indent: tuple[int, int, int]  # (mapping, sequence, offset)


def detect_eol(data: bytes) -> str:
    crlf = data.count(b"\r\n")
    lf = data.count(b"\n") - crlf
    return "\r\n" if crlf > 0 and crlf >= lf else "\n"


def detect_seq_indent(text: str) -> tuple[int, int, int]:
    """Recover ruamel (mapping, sequence, offset) from the first block dash.

    Dash at absolute column C (parent at col 0) -> offset=C, sequence=C+2. Covers
    both families: top-level item lists (C=0 -> 2/0) and nested lists under a
    column-0 key (C=2 -> 4/2). Mapping-only files use harmless defaults.
    """
    m = _SEQ_RE.search(text)
    if not m:
        return 2, 4, 2
    c = len(m.group("indent"))
    return 2, c + 2, c


def make_yaml(seq_indent: tuple[int, int, int]) -> YAML:
    mp, sq, off = seq_indent
    y = YAML()  # round-trip mode
    y.preserve_quotes = True
    y.width = 1 << 20  # never wrap long scalars
    y.indent(mapping=mp, sequence=sq, offset=off)
    return y


def read_meta(raw: bytes) -> tuple[str, FileMeta]:
    """Return (LF-normalized decoded text, FileMeta) from raw bytes."""
    bom = raw.startswith(_BOM)
    body = raw[len(_BOM):] if bom else raw
    eol = detect_eol(body)
    text = body.decode("utf-8")
    lf_text = text.replace("\r\n", "\n")
    return lf_text, FileMeta(eol=eol, bom=bom, seq_indent=detect_seq_indent(lf_text))


def load(raw: bytes):
    """Parse YAML into a round-trip structure (raises on invalid YAML)."""
    lf_text, meta = read_meta(raw)
    y = make_yaml(meta.seq_indent)
    return y.load(lf_text), meta


def dump(data, meta: FileMeta) -> bytes:
    """Serialize a round-trip structure back to bytes, re-applying EOL/BOM."""
    y = make_yaml(meta.seq_indent)
    buf = io.StringIO()
    y.dump(data, buf)
    out = buf.getvalue().replace("\n", meta.eol).encode("utf-8")
    return (_BOM + out) if meta.bom else out


def can_roundtrip(raw: bytes) -> bool:
    """True iff load->dump reproduces the exact original bytes.

    Files that fail this are served raw-text-only (still byte-safe, since raw
    writes echo bytes). Never raises — a parse error means 'raw only'.
    """
    try:
        data, meta = load(raw)
        return dump(data, meta) == raw
    except Exception:
        return False
