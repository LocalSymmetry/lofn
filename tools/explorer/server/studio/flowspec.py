"""FlowSpec — the studio's executable document: load, validate, version, import.

The Two-Truths doctrine (plan §0) keeps Canon (production prose Claude reads) and
Studio (the engine reading a FlowSpec) separate. This module owns the Studio side
of that boundary:

  * ``load_flowspec`` / ``validate_flowspec`` — read + schema-validate a
    ``*.flow.yaml`` (ruamel round-trip aware, mirrors the v1 yamlio doctrine).
  * ``next_version`` / ``copy_on_write`` — a flow that has EVER produced a run is
    immutable; edits fork to ``@N+1`` so provenance never dangles (plan §4).
  * ``flow_has_run`` — the "immutable once a run exists" helper the API enforces.
  * ``canon_import`` — GENERATE ``flows/lofn-<modality>@1.flow.yaml`` for all four
    modalities from ``pipeline_map.yaml`` + each step's Required Inputs + the
    legacy ``model_defaults.yaml`` (model_tiers heritage). Its node/gate set is
    cross-checked against the manifest — drift is a build failure (plan §4).

DUMB BACKEND: this module never reimplements a gate. It only records which gate
ids guard which step edge (from ``pipeline_map.yaml``); the engine shells out to
the repo's own ``scripts/validate_*.py`` at run time.

CANON IMMUTABILITY (acceptance gate S1): the importer only READS canon and only
WRITES under ``tools/explorer/flows/**``.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo / path discovery (walk up to the markers the v1 config.py uses).
# ---------------------------------------------------------------------------

_MARKERS = ["SOUL.md", "vault/gates.yaml", "skills/orchestration"]


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up until a dir contains all repo markers (v1 config.py contract)."""
    start = (start or Path(__file__)).resolve()
    for d in [start, *start.parents]:
        if all((d / m).exists() for m in _MARKERS):
            return d
    raise RuntimeError(f"repo root not found from {start} (markers: {_MARKERS})")


def _studio_dir() -> Path:
    return Path(__file__).resolve().parent


def _schemas_dir() -> Path:
    return _studio_dir() / "schemas"


def flows_dir(root: Path | None = None) -> Path:
    return (root or find_repo_root()) / "tools" / "explorer" / "flows"


# ---------------------------------------------------------------------------
# YAML I/O — ruamel round-trip, LF-normalized, no wrapping (v1 yamlio doctrine).
# ---------------------------------------------------------------------------

def _yaml():
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.width = 4096  # never wrap long strings
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def load_yaml(path: Path) -> Any:
    """Load YAML with ruamel (preferred), falling back to safe types."""
    text = path.read_text(encoding="utf-8")
    return _yaml().load(text)


def dump_yaml_str(data: Any) -> str:
    buf = io.StringIO()
    _yaml().dump(data, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Schema validation.
# ---------------------------------------------------------------------------

def _load_schema(name: str) -> dict:
    return json.loads((_schemas_dir() / name).read_text(encoding="utf-8"))


def validate_against(instance: Any, schema_name: str) -> list[str]:
    """Return a list of human-readable validation errors ([] == valid).

    Uses jsonschema when available; never raises on a validation miss — it
    reports. (An import failure of jsonschema itself IS raised, loudly, because
    that is a build-environment bug, not a data bug.)
    """
    import jsonschema  # required in the studio venv; fail loud if absent

    schema = _load_schema(schema_name)
    validator = jsonschema.Draft7Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in err.path) or "<root>"
        errors.append(f"{loc}: {err.message}")
    return errors


def validate_flowspec(spec: dict) -> list[str]:
    return validate_against(spec, "flowspec.schema.json")


def validate_knobs(preset: dict) -> list[str]:
    return validate_against(preset, "knobs.schema.json")


def validate_run(run: dict) -> list[str]:
    return validate_against(run, "run.schema.json")


@dataclass
class FlowSpec:
    """A loaded, validated FlowSpec plus its on-disk provenance."""

    path: Path
    data: dict

    @property
    def name(self) -> str:
        return self.data["flow"]

    @property
    def version(self) -> int:
        return int(self.data["version"])

    @property
    def modality(self) -> str:
        return self.data["modality"]

    def file_sha(self) -> str:
        return sha256_bytes(self.path.read_bytes())


def load_flowspec(path: Path, *, validate: bool = True) -> FlowSpec:
    data = load_yaml(path)
    # ruamel returns CommentedMap; normalize to plain dict for jsonschema.
    plain = json.loads(json.dumps(data, default=_plainify))
    if validate:
        errors = validate_flowspec(plain)
        if errors:
            raise ValueError(
                f"FlowSpec {path} failed schema validation:\n  - "
                + "\n  - ".join(errors)
            )
    return FlowSpec(path=path, data=plain)


def _plainify(obj):  # ruamel scalar/seq -> json-native
    if hasattr(obj, "items"):
        return dict(obj)
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Hashing.
# ---------------------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


# ---------------------------------------------------------------------------
# Versioning + immutability (plan §4).
# ---------------------------------------------------------------------------

_FLOW_FILE_RE = re.compile(r"^(?P<name>[a-z0-9-]+)@(?P<ver>\d+)(?P<suffix>-[a-z0-9-]+)?\.flow\.yaml$")


def flow_filename(name: str, version: int, suffix: str | None = None) -> str:
    sfx = f"-{suffix}" if suffix else ""
    return f"{name}@{version}{sfx}.flow.yaml"


def list_flow_versions(name: str, root: Path | None = None) -> list[int]:
    """Return the sorted versions of a flow family present on disk."""
    d = flows_dir(root)
    if not d.exists():
        return []
    out: list[int] = []
    for fp in d.glob(f"{name}@*.flow.yaml"):
        m = _FLOW_FILE_RE.match(fp.name)
        if m and m.group("name") == name:
            out.append(int(m.group("ver")))
    return sorted(set(out))


def next_version(name: str, root: Path | None = None) -> int:
    versions = list_flow_versions(name, root)
    return (max(versions) + 1) if versions else 1


def run_dir_root(root: Path | None = None) -> Path:
    return (root or find_repo_root()) / "output" / "studio"


def flow_has_run(name: str, version: int, root: Path | None = None) -> bool:
    """The "immutable once a run exists" helper (plan §4, API §9).

    A flow that has EVER produced a run is immutable — the API must fork edits to
    ``@N+1`` instead of overwriting. We answer this by scanning every studio
    ``run.json`` for a matching ``flow.name``/``flow.version``. Disk is authority
    (the rebuild-manifest doctrine): no separate index to drift.
    """
    rroot = run_dir_root(root)
    if not rroot.exists():
        return False
    for rj in rroot.glob("*/run.json"):
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        flow = data.get("flow") or {}
        if flow.get("name") == name and int(flow.get("version", -1)) == int(version):
            return True
    return False


def copy_on_write(
    src: FlowSpec | Path,
    *,
    suffix: str | None = None,
    root: Path | None = None,
) -> Path:
    """Fork a flow to the next version (provenance never dangles).

    Returns the path written under ``flows/``. ``base`` is stamped to
    ``<name>@<src-version>`` so the lineage is explicit. Does NOT overwrite the
    source. The caller (API) uses this whenever ``flow_has_run`` is true.
    """
    root = root or find_repo_root()
    spec = src if isinstance(src, FlowSpec) else load_flowspec(src)
    new_ver = next_version(spec.name, root)
    forked = dict(spec.data)
    forked["version"] = new_ver
    forked["base"] = f"{spec.name}@{spec.version}"
    out = flows_dir(root) / flow_filename(spec.name, new_ver, suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_yaml_str(forked), encoding="utf-8", newline="\n")
    return out


# ---------------------------------------------------------------------------
# Canon importer (plan §4, WS0 deliverable 2/3).
# ---------------------------------------------------------------------------

# The fan-out boundary: coordinator lane is steps 00..05, the pair fan-out is
# 06.. onward. This mirrors EXECUTION.md (step00..step05 are single-threaded
# coordinator artifacts; pair_NN_step06.. are per-pair fan-out artifacts).
_FANOUT_FIRST_STEP = 6

# Step number -> canonical artifact slug, keyed by the modality's node title.
# We derive the emitted filename from the manifest node id + a slugified title so
# the importer's `emits` matches the naming the validators/rebuild_manifest expect.
_STEP_SLUGS = {
    "00": "aesthetics_genres",
    "01": "essence_facets",
    "02": "concepts",
    "03": "artist_critique",
    "04": "medium",
    "05": "refine_medium",
    "06": "facets",
    "07": "guides",
    "08": "generation",
    "09": "artist_refined",
    "10": "revision_synthesis",
    "11": "enhancement",
}

# Legacy model_defaults.yaml modality keys (art == image).
_MODALITY_TO_DEFAULTS_KEY = {
    "image": "art",
    "music": "music",
    "video": "video",
    "story": "story",
}


def _git_head_sha(root: Path) -> str:
    """Best-effort git HEAD sha for provenance ('canon@<sha>'). 'unknown' on any miss."""
    head = root / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = root / ".git" / ref.split(" ", 1)[1].strip()
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:12]
            # packed-refs fallback
            packed = root / ".git" / "packed-refs"
            if packed.exists():
                target = ref.split(" ", 1)[1].strip()
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.strip().endswith(target):
                        return line.split(" ", 1)[0][:12]
        return ref[:12]
    except Exception:
        return "unknown"


def _load_pipeline_map(root: Path) -> dict:
    m = root / "tools" / "explorer" / "pipeline_map.yaml"
    data = load_yaml(m)
    return json.loads(json.dumps(data, default=_plainify))


def _load_model_defaults(root: Path) -> dict:
    p = root / "docs" / "reference" / "lofn-legacy-llm" / "model_defaults.yaml"
    if not p.exists():
        return {}
    data = load_yaml(p)
    return json.loads(json.dumps(data, default=_plainify))


def _rail_for(pmap: dict, modality: str) -> dict:
    for rail in pmap.get("rails", []):
        if rail.get("id") == modality:
            return rail
    raise KeyError(f"modality rail '{modality}' not in pipeline_map.yaml")


def _node_step_num(node_id: str) -> str:
    """'music.08' -> '08'."""
    return node_id.split(".", 1)[1].split(".", 1)[0]


def _consumes_for(step_num: str, modality: str) -> list[str]:
    """Canonical artifacts a step reads (beyond the always-present CREATIVE CONTEXT).

    Grounded in the pipeline shape (EXECUTION.md §5 artifact layout) rather than the
    prose Required-Inputs list (which restates the taxonomy slots that already live
    in CREATIVE CONTEXT). Coordinator steps chain to the prior coordinator artifact;
    the first fan-out step reads the pair assignments; later fan-out steps chain to
    the prior pair artifact of the same pair.
    """
    n = int(step_num)
    if n == 0:
        return []  # only CREATIVE CONTEXT
    if n <= 5:
        prev = f"{n - 1:02d}"
        return [_emit_name(prev, modality, fanout=False)]
    if n == _FANOUT_FIRST_STEP:
        # first fan-out step reads step-05's pair assignments (+ coordinator lane)
        return ["step05_refine_medium.md", "concept_medium_pairs.json"]
    prev = f"{n - 1:02d}"
    return [_emit_name(prev, modality, fanout=True)]


def _emit_name(step_num: str, modality: str, *, fanout: bool) -> str:
    slug = _STEP_SLUGS.get(step_num, step_num)
    if fanout:
        return f"pair_{{pair}}_step{step_num}_{slug}.md"
    return f"step{step_num}_{slug}.md"


def _tier_for(step_num: str) -> str:
    """Assign a model tier by role (heritage: concept_medium vs prompt).

    Coordinator/divergence steps (00..05) are 'coordinator'; the pair generation
    body (06..10) is 'pair'; the final enhancement (11) is 'enhance'. This matches
    the FlowSpec model_tiers example in plan §4.
    """
    n = int(step_num)
    if n <= 5:
        return "coordinator"
    if n >= 11:
        return "enhance"
    return "pair"


def _build_model_tiers(modality: str, defaults: dict) -> dict:
    """Build the model_tiers block. Anthropic-native default per plan §7; the legacy
    model_defaults.yaml supplies the heritage note of which models each role used.

    Per plan §4 the baseline import runs Anthropic (claude-opus-4-8) so a $0/mock or
    a live Anthropic smoke run works out of the box; the legacy per-modality lists are
    a reference the operator can retarget in the Flow Lab, not a hard binding.
    """
    return {
        "coordinator": {"provider": "anthropic", "model": "claude-opus-4-8", "max_tokens": 16000},
        "pair": {"provider": "anthropic", "model": "claude-opus-4-8", "max_tokens": 32000},
        "enhance": {"provider": "anthropic", "model": "claude-opus-4-8", "max_tokens": 32000},
    }


def _on_fail_for(step_num: str, gates: list[str], modality: str) -> dict:
    """Route on_fail by role: the step-11 enhancement quarantines the pair (the Andon
    Cord); pair-generation steps retry (validate_with_retries semantics); the rest
    retry as well. Coordinator step 00 retries. This mirrors the engine loop (plan §5).
    """
    if "g.andon" in gates:
        return {"policy": "quarantine_pair"}
    return {"policy": "retry", "max_attempts": 3}


def build_flowspec(modality: str, root: Path | None = None) -> dict:
    """Generate the @1 FlowSpec dict for a modality from canon.

    Reads pipeline_map.yaml (nodes + edge_gates + creative_context_slots + router
    template) and model_defaults.yaml. Pure function of canon — writes nothing.
    """
    root = root or find_repo_root()
    pmap = _load_pipeline_map(root)
    defaults = _load_model_defaults(root)
    rail = _rail_for(pmap, modality)
    slots = list(pmap.get("creative_context_slots", []))
    edge_gates = pmap.get("edge_gates", {})
    template = (rail.get("router") or {}).get("template", "")

    coordinator_steps: list[dict] = []
    pair_steps: list[dict] = []

    for node in rail.get("nodes", []):
        node_id = node["id"]
        step_num = _node_step_num(node_id)
        n = int(step_num)
        fanout = n >= _FANOUT_FIRST_STEP
        gates = list(edge_gates.get(node_id, []))
        # Map the manifest path (repo-relative) straight through as the prompt source.
        step = {
            "id": step_num,
            "title": node.get("title", ""),
            "kind": "llm",
            "prompt": node["path"],
            "overlay": None,
            "tier": _tier_for(step_num),
            "emits": _emit_name(step_num, modality, fanout=fanout),
            "consumes": _consumes_for(step_num, modality),
            "gates": gates,
            "on_fail": _on_fail_for(step_num, gates, modality),
        }
        (pair_steps if fanout else coordinator_steps).append(step)

    stages: list[dict] = [
        {"id": "coordinator", "steps": coordinator_steps},
    ]
    if pair_steps:
        stages.append({
            "id": "pairs",
            "foreach": "pair",
            "concurrency": "{max_concurrency}",
            "steps": pair_steps,
        })
    # Run-level portfolio distinctiveness validator (a pure-validator step; no LLM).
    stages.append({
        "id": "portfolio",
        "steps": [{
            "id": "distinctiveness",
            "title": "Portfolio distinctiveness",
            "kind": "validator",
            "prompt": None,
            "overlay": None,
            "tier": None,
            "emits": None,
            "consumes": [],
            "run": "scripts/validate_portfolio_distinctiveness.py",
            "gates": ["g.portfolio"],
        }],
    })

    spec = {
        "flow": f"lofn-{modality}",
        "version": 1,
        "modality": modality,
        "base": f"canon@{_git_head_sha(root)}",
        "knobs_defaults": "presets/classic-6x4.knobs.yaml",
        "context": {
            "template": template,
            "slots": slots,
        },
        "model_tiers": _build_model_tiers(modality, defaults),
        "stages": stages,
    }
    return spec


def canon_import(modality: str, *, root: Path | None = None, write: bool = True) -> Path:
    """Generate flows/lofn-<modality>@1.flow.yaml from canon and (by default) write it.

    Validates against the schema, cross-checks node/gate coverage against the
    manifest (raising on drift), and writes under flows/. Returns the output path.
    """
    root = root or find_repo_root()
    spec = build_flowspec(modality, root)

    errors = validate_flowspec(spec)
    if errors:
        raise ValueError(
            f"generated FlowSpec for {modality} failed schema validation:\n  - "
            + "\n  - ".join(errors)
        )

    drift = verify_against_manifest(spec, root)
    if drift:
        raise ValueError(
            f"importer node/gate set for {modality} drifted from pipeline_map.yaml:\n  - "
            + "\n  - ".join(drift)
        )

    out = flows_dir(root) / flow_filename(spec["flow"], 1)
    if write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dump_yaml_str(spec), encoding="utf-8", newline="\n")
    return out


def canon_import_all(*, root: Path | None = None, write: bool = True) -> list[Path]:
    return [canon_import(m, root=root, write=write) for m in ("music", "image", "video", "story")]


# ---------------------------------------------------------------------------
# Self-verify: importer node-set + gate-set MATCH pipeline_map.yaml (plan §4).
# ---------------------------------------------------------------------------

def _spec_step_ids(spec: dict) -> set[str]:
    ids: set[str] = set()
    for stage in spec["stages"]:
        for step in stage.get("steps", []):
            if step.get("kind") == "validator":
                continue
            ids.add(step["id"])
    return ids


def _spec_gates(spec: dict) -> dict[str, list[str]]:
    """step_id -> gate list, for LLM steps."""
    out: dict[str, list[str]] = {}
    for stage in spec["stages"]:
        for step in stage.get("steps", []):
            if step.get("kind") == "validator":
                continue
            out[step["id"]] = list(step.get("gates", []))
    return out


def verify_against_manifest(spec: dict, root: Path | None = None) -> list[str]:
    """Return a list of drift messages ([] == the importer matches the manifest).

    NODE PARITY: every manifest node for the rail is a step in the spec and vice
    versa. GATE PARITY: each step's gate list equals the manifest's edge_gates for
    that node (empty where the manifest declares none).
    """
    root = root or find_repo_root()
    pmap = _load_pipeline_map(root)
    modality = spec["modality"]
    rail = _rail_for(pmap, modality)
    edge_gates = pmap.get("edge_gates", {})

    manifest_steps = {_node_step_num(n["id"]) for n in rail.get("nodes", [])}
    spec_steps = _spec_step_ids(spec)

    drift: list[str] = []
    for missing in sorted(manifest_steps - spec_steps):
        drift.append(f"manifest node step {missing} missing from FlowSpec")
    for extra in sorted(spec_steps - manifest_steps):
        drift.append(f"FlowSpec step {extra} has no manifest node")

    spec_gates = _spec_gates(spec)
    for node in rail.get("nodes", []):
        step_num = _node_step_num(node["id"])
        want = sorted(edge_gates.get(node["id"], []))
        got = sorted(spec_gates.get(step_num, []))
        if want != got:
            drift.append(
                f"step {step_num} gate set {got} != manifest edge_gates {want}"
            )
    return drift


if __name__ == "__main__":  # convenience: python -m ... flowspec, or run directly
    import sys

    root = find_repo_root()
    written = canon_import_all(root=root, write=True)
    for p in written:
        rel = p.relative_to(root).as_posix()
        spec = load_flowspec(p)
        drift = verify_against_manifest(spec.data, root)
        status = "OK" if not drift else f"DRIFT: {drift}"
        print(f"wrote {rel}  [{status}]")
    sys.exit(0)
