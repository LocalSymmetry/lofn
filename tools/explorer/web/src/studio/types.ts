// Studio types — mirror the backend studio schemas + response shapes.
// Sources of truth:
//   server/studio/schemas/{flowspec,knobs,run}.schema.json
//   server/studio/{api,compile,runs,resolver,knobs}.py (response payloads)
// Kept pragmatic: shapes that are stable and read straight through are loosely
// typed; the load-bearing contracts (compile result, run.json, knob preset) are
// modeled precisely so the view agents get real autocomplete.

// -- FlowSpec (flowspec.schema.json) ---------------------------------------

export type Modality = "music" | "image" | "video" | "story";
export type Provider =
  | "anthropic" | "openai" | "openrouter" | "poe" | "gemini" | "mock";

export interface ModelTier {
  provider: Provider;
  model: string;
  max_tokens?: number;
  params?: Record<string, unknown>;
}

export type StepKind = "llm" | "validator";
export type OnFailPolicy = "block" | "flag" | "retry" | "quarantine_pair";

export interface OnFail {
  policy: OnFailPolicy;
  max_attempts?: number;
}

export interface FlowStep {
  id: string;
  title?: string;
  kind?: StepKind;               // default "llm"
  prompt?: string | null;
  overlay?: string | null;
  tier?: string | null;
  emits?: string | null;
  consumes?: string[];
  run?: string | null;           // validator script path (kind:validator)
  gates: string[];
  on_fail?: OnFail;
}

export interface FlowStage {
  id: string;
  foreach?: "pair";              // present => fan-out stage
  concurrency?: string | number;
  steps: FlowStep[];
}

export interface FlowContext {
  template: string;
  slots: string[];
}

export interface FlowSpec {
  flow: string;
  version: number;
  modality: Modality;
  base: string;
  knobs_defaults?: string;
  context: FlowContext;
  model_tiers: Record<string, ModelTier>;
  stages: FlowStage[];
  // provenance + immutability (added by GET /flows/{name})
  file?: string;
  file_sha?: string;
  immutable?: boolean;           // has ever produced a run => read-only (plan §4)
  versions?: number[];           // all @N versions of this flow family on disk
}

/** GET /flows list entry (summary). */
export interface FlowSummary {
  flow: string;
  version: number;
  modality?: Modality;
  title?: string;
  file?: string;
  file_sha?: string;
  immutable?: boolean;           // read-only if a run exists
  versions?: number[];           // sibling versions of this flow family
}

/** GET /flows/{name}/raw — verbatim on-disk YAML for the raw-YAML tab. */
export interface FlowRaw {
  flow: string;
  version: number;
  file: string;
  yaml: string;
  immutable: boolean;
}

/** POST /flows/{name}/version result (copy-on-write to @N+1). */
export interface ForkResult {
  flow: string;
  version: number;
  file: string;
  file_sha: string;
}

/** POST /presets result (save / duplicate). */
export interface PresetSaveResult {
  name: string;
  file: string;
}

// -- Knobs (knobs.schema.json) ---------------------------------------------

/** A knob value: scalar number, boolean, or a [lo, hi] band. */
export type KnobValue = number | boolean | [number, number];

export interface KnobDef {
  value?: KnobValue;
  expr?: string;                 // derivation over other knobs
  min_expr?: string;             // lower-bound invariant expr
  override?: number | boolean | null | KnobValue;
  range?: [number, number];      // UI slider clamp (inclusive)
  gates_key?: string;            // vault/gates.yaml key this knob writes
  desc?: string;
}

/** A *.knobs.yaml preset (the definitions). */
export interface KnobSetSpec {
  name?: string;
  desc?: string;
  knobs: Record<string, KnobDef>;
  invariants?: string[];
}

export interface InvariantResult {
  expr: string;
  ok: boolean;
  detail: string;
}

/** KnobSet.summary() — the compact resolved view /compile returns. */
export interface KnobSummary {
  name: string;
  desc: string;
  resolved: Record<string, KnobValue>;
  derived: string[];
  gates_bindings: Record<string, KnobValue>;
  invariants: InvariantResult[];
  ok: boolean;
}

/** GET /presets list entry. */
export interface PresetSummary {
  name: string;
  desc?: string;
  file?: string;
}

/** knobs / inline map accepted by compile & launch (preset name, full spec,
    bare knob map, or null for the flow default). */
export type KnobsRequest =
  | string
  | KnobSetSpec
  | Record<string, KnobDef>
  | null;

// -- Compile (compile.py CompileResult.as_dict) ----------------------------

export interface ResolvedPrompt {
  system: string | null;
  user: string;
  meta: Record<string, unknown>;
  text_sha: string;
}

export interface StepCompile {
  step: string;
  pair: number | null;
  resolved: ResolvedPrompt;
  est_tokens_in: number;
  est_tokens_out_ceiling: number;
  cost_estimate: number | null;
  priced: boolean;
  cost_note: string;
  diff: string | null;           // unified diff vs baseline (null if no baseline)
}

export interface CompileTotals {
  est_tokens_in: number;
  est_tokens_out_ceiling: number;
  cost_estimate: number | null;
  cost_estimate_note: string;
}

export interface InvariantViolation {
  expr: string;
  detail: string;
}

export interface CompileResult {
  flow: string;
  flow_version: number;
  knobs: KnobSummary;
  steps: StepCompile[];
  totals: CompileTotals;
  invariant_violations: InvariantViolation[];
  run_allowed: boolean;
}

export interface CompileRequest {
  flow: string;
  version?: number;
  knobs?: KnobsRequest;
  context?: Record<string, string>;
  step?: string;
  baseline?: KnobsRequest;       // diff target ("feel the knob")
}

// -- Gates preview (api.py /gates/preview) ---------------------------------

export interface GatesPreview {
  gates: Record<string, unknown>;
  bindings: Record<string, KnobValue>;
}

export interface GatesPreviewRequest {
  flow?: string;
  version?: number;
  knobs?: KnobsRequest;
}

// -- Runs (run.schema.json + runs.py) --------------------------------------

export type RunStatus =
  | "pending" | "running" | "complete" | "failed" | "cancelled";

export type StepStatus =
  | "pending" | "resolving" | "calling" | "validating" | "done"
  | "flagged" | "retrying" | "failed" | "quarantined" | "held_for_human";

export interface RunRef {
  path?: string;
  sha?: string;
  name?: string;
}

export interface RunContext {
  seed?: RunRef;
  personality?: RunRef;
  panel?: RunRef;
  meta_prompt?: RunRef;
  input?: string;
}

export interface RunFlowRef {
  name: string;
  version: number;
  file_sha: string;
}

export interface RunBudget {
  cap_usd: number;
  spend_usd?: number;
  aborted_on_cap?: boolean;
}

export interface RunTotals {
  tokens_in?: number;
  tokens_out?: number;
  cost_usd?: number;
  duration_s?: number;
  pairs_total?: number;
  pairs_quarantined?: number;
}

export interface GateResult {
  gate: string;
  pass: boolean | null;          // true=pass, false=hard fail, null=FLAG (amber)
  detail?: string;
}

export interface StepResult {
  stage?: string;
  step: string;
  pair?: string | null;
  provider?: string;
  model?: string;
  tokens_in?: number;
  tokens_out?: number;
  cost_usd?: number | null;
  duration_s?: number;
  attempts?: number;
  artifact?: string | null;
  artifact_sha?: string | null;
  gate_results?: GateResult[];
  status: StepStatus;
}

/** run.json (full record from GET /runs/{id}). */
export interface RunState {
  id: string;
  title: string;
  created: string;
  flow: RunFlowRef;
  knobs: Record<string, KnobValue>;
  overlays?: { path: string; sha: string }[];
  canon_sha: string;
  canon_dirty?: boolean;
  context: RunContext;
  model_tiers: Record<string, ModelTier>;
  steps?: StepResult[];
  totals?: RunTotals;
  budget: RunBudget;
  status: RunStatus;
  verdicts?: Record<string, unknown>[];
  quarantine_banner?: unknown;
}

/** GET /runs list entry (runs.py list_runs summary). */
export interface RunSummary {
  id: string;
  title: string;
  created: string;
  flow: RunFlowRef;
  status: RunStatus;
  totals?: RunTotals;
  budget?: RunBudget;
  quarantine_banner?: unknown;
}

export interface LaunchRequest {
  flow: string;
  version?: number;
  knobs?: KnobsRequest;
  context?: Record<string, string>;     // slot -> verbatim ICB value
  title?: string;
  budget_cap_usd: number;               // REQUIRED (no run without a cap)
  dry_run?: boolean;
  experimental_qa?: boolean;
}

// -- Providers (api.py /providers) -----------------------------------------

export interface ProviderStatus {
  provider: Provider | string;
  configured: boolean;
  model_list?: string[];
}

export interface ProviderTestResult {
  provider: string;
  ok: boolean;
  note?: string;
}

// -- Run control responses -------------------------------------------------

export interface CancelResult { run_id: string; cancelling: boolean; }
export interface DecisionResult { run_id: string; pair: string | null; decision: string; }
export interface DecisionRequest {
  pair?: string | null;                 // held pair label ("03"); null = run-level
  decision: "proceed" | "kill";
}

// -- Runlog / SSE events (runstore.jsonl frames tailed by useRunStream) -----
// The runlog is an open event stream (plan §5); frames carry an `event` tag and
// arbitrary payload fields. `run_end` terminates the stream. Loosely typed by
// design — the view agents narrow on `event`.

export interface RunlogEvent {
  event: string;                        // "step_start" | "gate" | "run_end" | ...
  ts?: string;
  step?: string;
  pair?: string | null;
  status?: StepStatus;
  [k: string]: unknown;
}

// -- Judge / Compare (server/studio/judge.py + api.py /judge) ----------------
// POST /judge returns BOTH the OPEN comparison and the fail-closed blind set.

/** Compact unified-diff summary for one artifact (judge._line_diff). */
export interface ArtifactDiff {
  added: number;
  removed: number;
  identical: boolean;
  unified: string;                      // raw unified-diff block (render in UnifiedDiff)
}

/** pass/fail/flag counts from a step's recorded gate_results. */
export interface GateCounts { pass: number; fail: number; flag: number; }

/** One aligned artifact across the two runs (compare.artifacts[]). */
export interface CompareArtifact {
  artifact: string;                     // canonical name, matched across A/B
  in_a: boolean;
  in_b: boolean;
  diff: ArtifactDiff;
  gate_delta: { a: GateCounts; b: GateCounts };
}

export interface CompareCostDelta {
  a_cost_usd: number;
  b_cost_usd: number;
  delta_cost_usd: number;
  a_tokens_in: number;
  b_tokens_in: number;
  a_tokens_out: number;
  b_tokens_out: number;
  delta_tokens_in: number;
  delta_tokens_out: number;
}

export interface CompareKnobDelta { knob: string; a: KnobValue | null; b: KnobValue | null; }

/** judge.compare() — the OPEN, provenance-full side-by-side. */
export interface CompareResult {
  run_a: string | null;
  run_b: string | null;
  artifacts: CompareArtifact[];
  cost_delta: CompareCostDelta;
  knob_delta: CompareKnobDelta[];
}

/** One provenance-STRIPPED blind card (no run/pair id — only two payloads). */
export interface BlindMatchup {
  matchup_id: string;                   // "M01", ...
  A: string;                            // payload text shown as side A
  B: string;                            // payload text shown as side B
}

/** A matchup that failed the fail-closed extraction (not emitted as a card). */
export interface BlindDropped {
  artifact: string;
  reason: string;
  a_ok: string;
  b_ok: string;
}

/** judge.blind_judge() — cards + coordinator-only de-blinding key. */
export interface BlindSet {
  run_a: string;
  run_b: string;
  seed: number;
  matchups: BlindMatchup[];
  /** matchup_id -> {A: run_id, B: run_id, artifact}. Coordinator-only. */
  key: Record<string, { A: string; B: string; artifact: string }>;
  dropped: BlindDropped[];
  n_matchups: number;
  n_dropped: number;
}

export interface JudgeRequest {
  run_a: string;
  run_b: string;
  seed?: number;                        // randomizes A/B sides deterministically
}

/** POST /judge — both views in one payload. */
export interface JudgeResult {
  compare: CompareResult;
  blind: BlindSet;
}

/** A recorded verdict (the OEC is the operator's taste, at n=1). */
export interface Verdict {
  matchup_id?: string;
  artifact?: string;
  winner?: string;                      // "A" | "B" | "tie" | de-blinded run id
  note?: string;
  kind?: string;
  ts?: string;
  [k: string]: unknown;
}

export interface VerdictRequest {
  run_a: string;
  run_b: string;
  verdict: Verdict;
}

export interface VerdictRecordResult {
  run_a: string;
  run_b: string;
  verdict: Verdict;                     // echoed with server-stamped ts
}

// -- Promote (server/studio/promote.py + api.py /promote/*) ------------------

/** One prose-patch preview/apply entry. */
export interface PatchPreview {
  file: string;
  ok: boolean;
  detail: string;
  diff: string;
}

/** gates.yaml value-edit preview block. */
export interface GatesPromoteBlock {
  file: string;
  applied: Record<string, KnobValue>;
  diff: string;
  changed: boolean;
}

export interface PromotePreviewRequest {
  /** gates.yaml key -> proposed value (EXISTING keys only). */
  gate_values?: Record<string, KnobValue>;
  /** exact, unique, non-regex prose swaps. */
  patches?: { file: string; old: string; new: string }[];
}

/** POST /promote/preview — the diffs a promotion WOULD produce (no write). */
export interface PromotePreview {
  gates: GatesPromoteBlock | null;
  patches: PatchPreview[];
  skipped_keys: string[];
}

/** POST /promote/apply — after the byte-safe write + --meta-check. */
export interface PromoteApplyResult {
  applied_gates: Record<string, KnobValue>;
  applied_patches: { file: string; applied: boolean; detail: string }[];
  errors: string[];
  meta_check: { exit_code: number; ok: boolean; stdout: string; stderr: string };
  git_status: string;
}

export interface PromoteBriefRequest {
  run_a?: string;
  run_b?: string;
  title?: string;
  summary?: string;
  structural_changes?: { kind?: string; where?: string; what?: string }[];
  knob_delta?: CompareKnobDelta[];
}

/** POST /promote/brief — the structural-change hand-off text. */
export interface PromoteBriefResult { brief: string; }
