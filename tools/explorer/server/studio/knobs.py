"""Knobs — the magic-numbers-as-parameters model (plan §3, §6, WS2).

Every magic number in the Lofn pipeline (the 6 pairs, the 4 variations, the
850–1000 char band, …) becomes a KNOB with a range, an optional derivation, and
invariants. This module owns:

  * ``SafeEvaluator`` — a *tiny* expression evaluator for derived knobs and
    invariants. Integer arithmetic + ``ceil / floor / min / max`` ONLY. There is
    **no ``eval()``** anywhere in this module (plan §3.2 non-negotiable): the
    expressions are parsed with the stdlib ``ast`` module and walked node-by-node
    against a hard allowlist. Anything outside the allowlist raises.
  * ``KnobSet`` — a loaded ``*.knobs.yaml`` preset: literal values, derived
    ``expr`` knobs (with optional ``override`` pin), ``min_expr`` lower bounds,
    ``gates_key`` bindings, and the fully-resolved value map.
  * invariant checking — evaluate each declared invariant plus the implicit
    ``min_expr`` bounds; a violation blocks Run (not Compile) per plan §3.2.
  * preset load / save / duplicate — the Knobs-UI operations (plan §8.2).

DUMB BACKEND: knob values are *only* numbers/bands. This module never renders a
prompt (that's ``resolver.py``) and never runs a gate (that's ``validate_*.py``).
The run-local ``gates.yaml`` materializer (``materialize_gates``) writes the
``gates_key``-bound knob values into a file the validators read via ``--gates``.

CANON IMMUTABILITY (S1): presets live under ``tools/explorer/flows/presets/**``;
this module writes nothing outside that tree (and ``output/studio/**`` for the
run-local gates file, which the engine owns).
"""
from __future__ import annotations

import ast
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# A knob value is a scalar (int/float/bool) or a 2-element band [lo, hi].
Scalar = Union[int, float, bool]
Band = List[float]
KnobValue = Union[Scalar, Band]


# ---------------------------------------------------------------------------
# The SAFE evaluator (NO eval()). ast-walked, hard allowlist.
# ---------------------------------------------------------------------------

class EvalError(ValueError):
    """Raised when an expression is malformed or steps outside the allowlist."""


def _to_number(v: Any) -> Union[int, float]:
    """Coerce a knob value to a number for arithmetic.

    Bands are NOT numbers — a bare band used in arithmetic is an error; band
    elements are reached with subscript (``sung_lines[0]``). ``bool`` is a Python
    ``int`` subclass and is allowed (True==1) so a boolean knob can participate in
    e.g. ``min(...)``, but we keep it as-is otherwise.
    """
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    raise EvalError(f"expected a number, got {type(v).__name__}: {v!r}")


class SafeEvaluator:
    """Evaluate a knob expression against a name→value environment.

    Grammar (everything else raises ``EvalError``):
      * int / float literals
      * names bound in the environment (other knobs)
      * subscript of a band by an int literal or expression: ``sung_lines[0]``
      * binary  + - * / // %  and unary + -
      * comparisons  <  <=  >  >=  ==  !=  (chained, for invariants)
      * boolean  and / or / not  (for invariants)
      * calls to the whitelisted functions ONLY: ceil, floor, min, max, abs, int

    Division uses TRUE division for ``/`` (matching Python) but derived knobs in
    the census use ``ceil(n_pairs / 2)`` so the result is re-integerized by the
    surrounding ceil/floor. ``//`` is floor division. There is no attribute
    access, no comprehension, no lambda, no name that isn't in the env or the
    function allowlist — the walker rejects the node type outright.
    """

    _FUNCS = {
        "ceil": lambda x: math.ceil(x),
        "floor": lambda x: math.floor(x),
        "min": lambda *xs: min(xs),
        "max": lambda *xs: max(xs),
        "abs": lambda x: abs(x),
        "int": lambda x: int(x),
    }

    _BINOPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
    }

    _CMPOPS = {
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
    }

    def __init__(self, env: Dict[str, KnobValue]):
        self.env = env

    def eval(self, expr: str) -> Any:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise EvalError(f"cannot parse expression {expr!r}: {e}") from e
        return self._walk(tree.body, expr)

    # -- the walker ---------------------------------------------------------
    def _walk(self, node: ast.AST, expr: str) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            if isinstance(node.value, bool):
                return node.value
            raise EvalError(f"disallowed literal {node.value!r} in {expr!r}")

        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise EvalError(f"unknown name {node.id!r} in {expr!r}")
            return self.env[node.id]

        if isinstance(node, ast.Subscript):
            base = self._walk(node.value, expr)
            idx = self._walk(node.slice, expr)
            if not isinstance(base, (list, tuple)):
                raise EvalError(f"cannot subscript non-band {base!r} in {expr!r}")
            try:
                return base[int(idx)]
            except (IndexError, TypeError) as e:
                raise EvalError(f"bad band index in {expr!r}: {e}") from e

        if isinstance(node, ast.UnaryOp):
            operand = _to_number(self._walk(node.operand, expr))
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.Not):
                return not self._walk(node.operand, expr)
            raise EvalError(f"disallowed unary operator in {expr!r}")

        if isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in self._BINOPS:
                raise EvalError(f"disallowed binary operator {op.__name__} in {expr!r}")
            left = _to_number(self._walk(node.left, expr))
            right = _to_number(self._walk(node.right, expr))
            if op in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise EvalError(f"division by zero in {expr!r}")
            return self._BINOPS[op](left, right)

        if isinstance(node, ast.BoolOp):
            values = [self._walk(v, expr) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise EvalError(f"disallowed boolean operator in {expr!r}")

        if isinstance(node, ast.Compare):
            left = self._walk(node.left, expr)
            result = True
            for op, comparator in zip(node.ops, node.comparators):
                right = self._walk(comparator, expr)
                op_t = type(op)
                if op_t not in self._CMPOPS:
                    raise EvalError(f"disallowed comparison {op_t.__name__} in {expr!r}")
                # Numeric coercion for ordering; equality works on raw values too.
                if op_t in (ast.Eq, ast.NotEq):
                    result = result and self._CMPOPS[op_t](left, right)
                else:
                    result = result and self._CMPOPS[op_t](_to_number(left), _to_number(right))
                left = right
            return result

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise EvalError(f"only bare function calls allowed in {expr!r}")
            fname = node.func.id
            if fname not in self._FUNCS:
                raise EvalError(f"disallowed function {fname!r} in {expr!r}")
            if node.keywords:
                raise EvalError(f"keyword arguments not allowed in {expr!r}")
            args = [_to_number(self._walk(a, expr)) for a in node.args]
            return self._FUNCS[fname](*args)

        raise EvalError(f"disallowed expression node {type(node).__name__} in {expr!r}")


# ---------------------------------------------------------------------------
# YAML I/O — ruamel round-trip aware, LF-normalized (v1 yamlio doctrine).
# ---------------------------------------------------------------------------

def _yaml():
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _plainify(obj):
    """ruamel scalar/seq -> json/python-native (recursive)."""
    if hasattr(obj, "items"):
        return {str(k): _plainify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plainify(x) for x in obj]
    return obj


def load_knobs_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return _plainify(_yaml().load(text))


def dump_knobs_yaml_str(data: Any) -> str:
    buf = io.StringIO()
    _yaml().dump(data, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# KnobSet — a loaded preset with a resolved value map.
# ---------------------------------------------------------------------------

@dataclass
class KnobDef:
    """One knob's declaration (mirrors knobs.schema.json)."""

    name: str
    value: Optional[KnobValue] = None
    expr: Optional[str] = None
    min_expr: Optional[str] = None
    override: Optional[KnobValue] = None
    range: Optional[List[float]] = None
    gates_key: Optional[str] = None
    desc: Optional[str] = None

    @property
    def derived(self) -> bool:
        return self.expr is not None


@dataclass
class InvariantResult:
    expr: str
    ok: bool
    detail: str = ""


@dataclass
class KnobSet:
    """A loaded, RESOLVED knobs preset.

    ``resolved`` maps every knob name to its final value (literals as-is; derived
    knobs computed via the safe evaluator, unless ``override`` pins them). The
    resolution is topological-by-fixpoint: derived knobs may reference other
    derived knobs; we iterate to a fixed point and raise on a cycle.
    """

    name: str
    desc: str
    defs: Dict[str, KnobDef]
    invariants: List[str]
    resolved: Dict[str, KnobValue] = field(default_factory=dict)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "KnobSet":
        defs: Dict[str, KnobDef] = {}
        for name, spec in (data.get("knobs") or {}).items():
            spec = spec or {}
            defs[name] = KnobDef(
                name=name,
                value=spec.get("value"),
                expr=spec.get("expr"),
                min_expr=spec.get("min_expr"),
                override=spec.get("override"),
                range=spec.get("range"),
                gates_key=spec.get("gates_key"),
                desc=spec.get("desc"),
            )
        ks = cls(
            name=str(data.get("name", "")),
            desc=str(data.get("desc", "")),
            defs=defs,
            invariants=list(data.get("invariants") or []),
        )
        ks.resolve()
        return ks

    @classmethod
    def load(cls, path: Path) -> "KnobSet":
        return cls.from_dict(load_knobs_yaml(path))

    # -- resolution ---------------------------------------------------------
    def resolve(self) -> Dict[str, KnobValue]:
        """Compute the final value of every knob (fixpoint over derived exprs).

        A derived knob with a non-null ``override`` is PINNED to the override and
        never evaluates its ``expr`` (plan §3.2: "a derived knob may still be
        pinned via override"). Literal knobs take their ``value``. Raises
        ``EvalError`` on an unresolvable cycle or an expression that references a
        name that can never resolve.
        """
        resolved: Dict[str, KnobValue] = {}
        # Seed literals + pinned overrides.
        pending: Dict[str, KnobDef] = {}
        for name, d in self.defs.items():
            if d.derived and d.override is None:
                pending[name] = d
            elif d.derived and d.override is not None:
                resolved[name] = d.override
            else:
                resolved[name] = d.value

        # Fixpoint: repeatedly try to evaluate pending derived knobs whose free
        # names are all already resolved.
        progress = True
        while pending and progress:
            progress = False
            for name in list(pending):
                d = pending[name]
                try:
                    val = SafeEvaluator(resolved).eval(d.expr)  # type: ignore[arg-type]
                except EvalError:
                    # A referenced name isn't resolved yet — retry next pass.
                    continue
                resolved[name] = val
                del pending[name]
                progress = True

        if pending:
            raise EvalError(
                "unresolvable derived knobs (cycle or unknown reference): "
                + ", ".join(f"{n}={self.defs[n].expr!r}" for n in pending)
            )

        self.resolved = resolved
        return resolved

    # -- invariants ---------------------------------------------------------
    def check_invariants(self) -> List[InvariantResult]:
        """Evaluate every declared invariant + each knob's implicit ``min_expr``.

        Returns one ``InvariantResult`` per check. ``ok`` False anywhere means the
        preset BLOCKS Run (plan §3.2). Compile is allowed on a broken preset.
        A malformed invariant (evaluator error) is reported as a FAILED check
        rather than crashing the caller.
        """
        results: List[InvariantResult] = []
        ev = SafeEvaluator(self.resolved)

        for expr in self.invariants:
            try:
                val = ev.eval(expr)
                ok = bool(val)
                results.append(InvariantResult(expr=expr, ok=ok,
                                               detail="" if ok else "evaluated False"))
            except EvalError as e:
                results.append(InvariantResult(expr=expr, ok=False, detail=str(e)))

        # Implicit min_expr bounds: value >= min_expr.
        for name, d in self.defs.items():
            if not d.min_expr:
                continue
            check = f"{name} >= {d.min_expr}"
            try:
                bound = ev.eval(d.min_expr)
                cur = self.resolved.get(name)
                cur_n = _to_number(cur)
                ok = cur_n >= _to_number(bound)
                detail = "" if ok else f"{name}={cur} < min {bound}"
                results.append(InvariantResult(expr=check, ok=ok, detail=detail))
            except EvalError as e:
                results.append(InvariantResult(expr=check, ok=False, detail=str(e)))

        return results

    def invariant_violations(self) -> List[InvariantResult]:
        return [r for r in self.check_invariants() if not r.ok]

    @property
    def ok(self) -> bool:
        """True when every invariant + min_expr bound holds (Run is allowed)."""
        return not self.invariant_violations()

    # -- gates binding ------------------------------------------------------
    def gates_bindings(self) -> Dict[str, KnobValue]:
        """gates.yaml key → resolved value, for every ``gates_key``-bound knob."""
        out: Dict[str, KnobValue] = {}
        for name, d in self.defs.items():
            if d.gates_key:
                out[d.gates_key] = self.resolved[name]
        return out

    # -- editing (Knobs UI) -------------------------------------------------
    def with_value(self, name: str, value: KnobValue) -> "KnobSet":
        """Return a NEW KnobSet with one knob's literal value changed + re-resolved.

        The signature "feel the knob" operation (plan §8.2): drag n_pairs 6→10 and
        everything derived moves. Copy-on-value; the original is untouched. A
        derived knob can be pinned by setting its ``override`` — use
        ``with_override`` for that instead.
        """
        new = self._clone()
        if name not in new.defs:
            raise KeyError(f"unknown knob {name!r}")
        d = new.defs[name]
        if d.derived and d.override is None:
            raise ValueError(
                f"{name!r} is derived (expr={d.expr!r}); pin it via override, "
                f"not value"
            )
        new.defs[name] = _replace_knob(d, value=value)
        new.resolve()
        return new

    def with_override(self, name: str, override: Optional[KnobValue]) -> "KnobSet":
        """Pin (or un-pin, with None) a derived knob's value."""
        new = self._clone()
        if name not in new.defs:
            raise KeyError(f"unknown knob {name!r}")
        new.defs[name] = _replace_knob(new.defs[name], override=override)
        new.resolve()
        return new

    def _clone(self) -> "KnobSet":
        return KnobSet(
            name=self.name,
            desc=self.desc,
            defs={k: _replace_knob(v) for k, v in self.defs.items()},
            invariants=list(self.invariants),
            resolved=dict(self.resolved),
        )

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        """Round-trip back to the ``*.knobs.yaml`` shape (for save/duplicate)."""
        knobs: Dict[str, dict] = {}
        for name, d in self.defs.items():
            spec: Dict[str, Any] = {}
            if d.value is not None:
                spec["value"] = d.value
            if d.expr is not None:
                spec["expr"] = d.expr
            if d.min_expr is not None:
                spec["min_expr"] = d.min_expr
            if d.override is not None:
                spec["override"] = d.override
            elif d.derived:
                # Preserve the explicit `override: null` the schema expects on a
                # derived knob so a round-trip is stable.
                spec["override"] = None
            if d.range is not None:
                spec["range"] = d.range
            if d.gates_key is not None:
                spec["gates_key"] = d.gates_key
            if d.desc is not None:
                spec["desc"] = d.desc
            knobs[name] = spec
        out: Dict[str, Any] = {}
        if self.name:
            out["name"] = self.name
        if self.desc:
            out["desc"] = self.desc
        out["knobs"] = knobs
        if self.invariants:
            out["invariants"] = list(self.invariants)
        return out

    def summary(self) -> dict:
        """Compact view for the Knobs UI / /compile response: resolved values +
        which are derived + invariant status. Never contains anything secret."""
        return {
            "name": self.name,
            "desc": self.desc,
            "resolved": dict(self.resolved),
            "derived": [n for n, d in self.defs.items() if d.derived],
            "gates_bindings": self.gates_bindings(),
            "invariants": [
                {"expr": r.expr, "ok": r.ok, "detail": r.detail}
                for r in self.check_invariants()
            ],
            "ok": self.ok,
        }


def _replace_knob(d: KnobDef, **changes) -> KnobDef:
    return KnobDef(
        name=changes.get("name", d.name),
        value=changes.get("value", d.value),
        expr=changes.get("expr", d.expr),
        min_expr=changes.get("min_expr", d.min_expr),
        override=changes.get("override", d.override),
        range=changes.get("range", d.range),
        gates_key=changes.get("gates_key", d.gates_key),
        desc=changes.get("desc", d.desc),
    )


# ---------------------------------------------------------------------------
# Preset load / save / duplicate (plan §8.2 Knobs UI operations).
# ---------------------------------------------------------------------------

def presets_dir(flows_dir: Path) -> Path:
    return flows_dir / "presets"


def preset_path(flows_dir: Path, name: str) -> Path:
    """Presets are named ``<name>.knobs.yaml`` under flows/presets/."""
    stem = name[:-len(".knobs.yaml")] if name.endswith(".knobs.yaml") else name
    return presets_dir(flows_dir) / f"{stem}.knobs.yaml"


def load_preset(flows_dir: Path, name: str) -> KnobSet:
    return KnobSet.load(preset_path(flows_dir, name))


def save_preset(flows_dir: Path, knobset: KnobSet, name: Optional[str] = None) -> Path:
    """Write a KnobSet to flows/presets/<name>.knobs.yaml (byte-safe, LF).

    CANON IMMUTABILITY: only ever writes under flows/presets/**. The caller (API)
    is responsible for the path-jail; this helper computes the path from the
    presets dir and never escapes it.
    """
    stem = name or knobset.name
    if not stem:
        raise ValueError("preset needs a name to save under")
    out = preset_path(flows_dir, stem)
    data = knobset.to_dict()
    if name:
        data["name"] = stem[:-len(".knobs.yaml")] if stem.endswith(".knobs.yaml") else stem
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_knobs_yaml_str(data), encoding="utf-8", newline="\n")
    return out


def duplicate_preset(flows_dir: Path, src_name: str, new_name: str) -> Path:
    """Load a preset and save it under a new name (the UI 'duplicate' op)."""
    ks = load_preset(flows_dir, src_name)
    ks.name = new_name[:-len(".knobs.yaml")] if new_name.endswith(".knobs.yaml") else new_name
    return save_preset(flows_dir, ks, name=new_name)


def list_presets(flows_dir: Path) -> List[str]:
    d = presets_dir(flows_dir)
    if not d.exists():
        return []
    return sorted(p.name[:-len(".knobs.yaml")] for p in d.glob("*.knobs.yaml"))


# ---------------------------------------------------------------------------
# Run-local gates materializer (plan §3.3 step 3, §6 WS2 deliverable 3).
# ---------------------------------------------------------------------------

def build_run_gates(base_gates: dict, knobset: KnobSet) -> dict:
    """Overlay the ``gates_key``-bound knob values onto a copy of ``base_gates``.

    Starts from the canon ``vault/gates.yaml`` mapping (so keys the studio does
    NOT knob — house_lexicon, ban_words, terminal-punctuation flags — pass through
    unchanged) and writes each bound knob's resolved value over its ``gates_key``.
    The result is what the engine materializes to
    ``output/studio/<run>/gates.yaml`` for ``validate_step.py --gates``.
    """
    out = dict(base_gates)
    for gates_key, value in knobset.gates_bindings().items():
        out[gates_key] = value
    # Derive the run-local concept-pair band from n_pairs so a wide portfolio
    # (n_pairs=10) passes validate_step.py step-05, which reads `pair_count_band`
    # from the loaded gates (fail-open default [4, 7]). This is emitted ONLY into
    # the run-local materialized gates — canon vault/gates.yaml is never touched.
    n_pairs = knobset.resolved.get("n_pairs")
    if isinstance(n_pairs, (int, float)) and not isinstance(n_pairs, bool):
        np = int(n_pairs)
        out["pair_count_band"] = [np, np]
    return out


def materialize_gates(
    run_dir: Path,
    base_gates: dict,
    knobset: KnobSet,
) -> Path:
    """Write ``<run_dir>/gates.yaml`` from base gates + knob overrides.

    The engine (WS3) calls this once per run; ``resolver.compile`` can call it in
    dry-run to prove the file the validators will read. Byte-safe, LF-normalized.
    Returns the written path. The engine OWNS output/studio/** — this is the only
    place this module writes outside flows/presets/.
    """
    out = build_run_gates(base_gates, knobset)
    dest = run_dir / "gates.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(dump_knobs_yaml_str(out), encoding="utf-8", newline="\n")
    return dest


def load_base_gates(repo_root: Path) -> dict:
    """Load the canon vault/gates.yaml as a plain mapping (READ-ONLY)."""
    p = repo_root / "vault" / "gates.yaml"
    if not p.exists():
        return {}
    return _plainify(_yaml().load(p.read_text(encoding="utf-8"))) or {}
