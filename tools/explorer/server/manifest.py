"""Load and expose tools/explorer/pipeline_map.yaml.

The manifest is the app's single structural truth. This module parses it and
offers typed-ish accessors; it does not interpret the repo layout itself.
"""
from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML


class Manifest:
    def __init__(self, data: dict, path: Path):
        self.data = data
        self.path = path

    # -- accessors --------------------------------------------------------
    @property
    def version(self) -> int:
        return int(self.data.get("version", 0))

    @property
    def write_allowlist(self) -> list[str]:
        return list(self.data.get("write_allowlist", []))

    @property
    def repo_root_markers(self) -> list[str]:
        return list(self.data.get("repo_root_markers", []))

    @property
    def rails(self) -> list[dict]:
        return self.data.get("rails", [])

    @property
    def gates(self) -> list[dict]:
        return self.data.get("gates", [])

    @property
    def validators(self) -> list[dict]:
        return self.data.get("validators", [])

    @property
    def libraries(self) -> list[dict]:
        return self.data.get("libraries", [])

    @property
    def gates_file(self) -> str:
        return self.data.get("gates_file", "vault/gates.yaml")

    def library(self, lib_id: str) -> dict | None:
        return next((l for l in self.libraries if l.get("id") == lib_id), None)

    def all_nodes(self) -> list[dict]:
        out = []
        for rail in self.rails:
            for node in rail.get("nodes", []):
                out.append({**node, "rail": rail["id"]})
        return out

    def node(self, node_id: str) -> dict | None:
        for rail in self.rails:
            for n in rail.get("nodes", []):
                if n.get("id") == node_id:
                    return {**n, "rail": rail["id"]}
                for v in n.get("variants", []) or []:
                    if v.get("id") == node_id:
                        return {**v, "rail": rail["id"], "variant_of": n["id"]}
        return None

    def public_dict(self) -> dict:
        """The whole manifest, as the frontend consumes it (plain JSON)."""
        return self.data


def load_manifest(path: Path) -> Manifest:
    text = Path(path).read_text(encoding="utf-8")
    data = YAML(typ="safe").load(text)
    return Manifest(data, Path(path))
