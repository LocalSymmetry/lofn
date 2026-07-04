// Types mirroring the backend API (see server/*.py). Kept pragmatic; manifest
// sub-objects are loosely typed where the shape is stable and read straight through.

export type Severity = "hard" | "flag" | "prose" | "mixed";

export interface NodeVariant { id: string; title: string; path: string; }
export interface RailNode {
  id: string;
  title: string;
  path: string;
  rail?: string;
  edition_compressed?: string;
  variants?: NodeVariant[];
}
export interface Rail {
  id: string;
  title: string;
  kind: string;
  router?: Record<string, string>;
  nodes: RailNode[];
}
export interface Gate {
  id: string;
  title: string;
  level: "step" | "run" | "orchestration";
  severity: Severity;
  keys: string[];
  validator: string | null;
  prose?: string;
  note?: string;
}
export interface Validator {
  id: string;
  path: string;
  level: string;
  usage: string;
}
export interface LibrarySpec {
  id: string;
  title: string;
  kind: string;
  card?: string;
}
export interface Manifest {
  version: number;
  rails: Rail[];
  gates: Gate[];
  validators: Validator[];
  libraries: any[];
  dispatch_edges: { from: string; to: string; label: string }[];
  edge_gates: Record<string, string[]>;
  creative_context_slots: string[];
  step_outline_headings: string[];
  gates_file: string;
  execution_layer: { id: string; skill: string; execution?: string }[];
  reference_dirs: { id: string; dir: string }[];
}
export interface VerifyResult {
  ok: boolean;
  missing?: { path: string; context: string }[];
  orphan_step_files?: string[];
  error?: string;
}
export interface ManifestResponse {
  repo_root: string;
  manifest: Manifest;
  verify: VerifyResult;
}

export interface FileView {
  path: string;
  text: string;
  eol: "crlf" | "lf";
  bom: boolean;
  sha: string;
  mtime: number;
  size: number;
  writable: boolean;
  roundtrip: boolean;
  is_yaml: boolean;
}

export type GitStatus = Record<string, string>;

export interface GateGroupEntry { key: string; value: any; }
export interface GateGroup { section: string; entries: GateGroupEntry[]; }
export interface GatesData {
  path: string;
  sha: string;
  text: string;
  eol: string;
  writable: boolean;
  roundtrip: boolean;
  groups: GateGroup[];
  flat: Record<string, any>;
}
export interface MetaCheckRow { file: string; label: string; value: string; }
export interface MetaCheck {
  summary: string;
  rows: MetaCheckRow[];
  count: number;
  raw_stdout: string;
  raw_stderr: string;
  duration_ms: number;
}

export interface LibraryItemRef {
  name: string;
  file: string;
  slug: string;
  is_variant: boolean;
}
export interface LibraryList {
  id: string;
  kind: string;
  title: string;
  count: number;
  variant_count?: number;
  items: LibraryItemRef[] | string[];
  columns?: string[];
  rows?: string[][];
  path?: string;
  sha?: string;
}
export interface Card {
  name: string;
  type: "personality" | "panel";
  headings?: string[];
  preview?: string;
  sections?: { title: string; items: string[] }[];
}
export interface LibraryItem { view: FileView; card: Card | null; }
export interface SyncStatus {
  id: string;
  applicable: boolean;
  canonical_count?: number;
  index_count?: number | null;
  monolith_count?: number | null;
  missing_from_index?: string[];
  stale_in_index?: string[];
  regen?: string;
}
export interface ValidatorResult {
  cmd: string;
  returncode: number;
  stdout: string;
  stderr: string;
  duration_ms: number;
}
