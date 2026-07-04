// Studio API client — typed fetch wrapper for /api/studio/* (plan §9).
// Mirrors the v1 web/src/api.ts style: a `j<T>` response helper + typed returns.
// Same-origin; the vite dev proxy forwards /api. KEYS never cross this boundary —
// /providers returns {configured} only; every runlog/run frame is redacted server-side.
//
// BACKEND STATUS (read server/studio/api.py): providers, providers/{p}/test,
// compile, gates/preview, and runs/* are IMPLEMENTED. flows list/get/import,
// presets, judge, and promote are FORTHCOMING — their client methods are present
// with final signatures so the view agents can build against them; the not-yet-
// wired ones call the intended route (a 404 today) rather than throwing, EXCEPT
// import/judge/promote mutations, which reject early with a clear message so a
// view never silently POSTs to a dead route. Swap the bodies out when the routes land.

import type {
  CompileRequest, CompileResult,
  GatesPreview, GatesPreviewRequest,
  FlowSpec, FlowSummary, FlowRaw, ForkResult,
  PresetSummary, KnobSetSpec, PresetSaveResult,
  ProviderStatus, ProviderTestResult,
  LaunchRequest, RunState, RunSummary,
  CancelResult, DecisionRequest, DecisionResult, RunlogEvent,
  JudgeRequest, JudgeResult, VerdictRequest, VerdictRecordResult,
  PromotePreviewRequest, PromotePreview, PromoteApplyResult,
  PromoteBriefRequest, PromoteBriefResult,
} from "./types";

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let body: any = {};
    try { body = await res.json(); } catch { /* ignore */ }
    const err = new Error(body.error || body.detail || `HTTP ${res.status}`) as any;
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return res.json() as Promise<T>;
}

const q = (s: string) => encodeURIComponent(s);
const BASE = "/api/studio";

const jsonPost = (path: string, body: unknown) =>
  fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });

/** Guard for routes whose backend isn't wired yet: reject with a clear message
    rather than firing a mutation at a dead endpoint. */
const notWired = <T>(what: string): Promise<T> =>
  Promise.reject(
    Object.assign(new Error(`${what} is not implemented in the studio backend yet`), { status: 501 }),
  );

export const studioApi = {
  // -- Flows (list/get/raw/fork IMPLEMENTED; import FORTHCOMING) ------------
  flows: () => fetch(`${BASE}/flows`).then(j<FlowSummary[]>),
  flow: (name: string, version = 1) =>
    fetch(`${BASE}/flows/${q(name)}?version=${version}`).then(j<FlowSpec>),
  /** Verbatim on-disk YAML for the raw-YAML Monaco tab. */
  flowRaw: (name: string, version = 1) =>
    fetch(`${BASE}/flows/${q(name)}/raw?version=${version}`).then(j<FlowRaw>),
  /** Copy-on-write to @N+1 ("New version from this"). Pass `yaml` to write the
      raw-editor buffer as the new version (schema-validated server-side); omit
      it to duplicate the source verbatim. Never overwrites the source. */
  forkFlow: (name: string, version: number, yaml?: string) =>
    jsonPost(`/flows/${q(name)}/version`, yaml == null ? { version } : { version, yaml })
      .then(j<ForkResult>),
  importFlow: (_body: { source?: string }): Promise<FlowSummary> =>
    notWired("flow import"),

  // -- Presets (list/get/save IMPLEMENTED) --------------------------------
  presets: (flow?: string) =>
    fetch(`${BASE}/presets${flow ? `?flow=${q(flow)}` : ""}`).then(j<PresetSummary[]>),
  preset: (name: string) =>
    fetch(`${BASE}/presets/${q(name)}`).then(j<KnobSetSpec>),
  /** Save / duplicate a preset (name collision overwrites). Writes only under
      flows/presets/**. `knobs` is the full KnobSetSpec. */
  savePreset: (name: string, knobs: KnobSetSpec) =>
    jsonPost("/presets", { name, knobs }).then(j<PresetSaveResult>),

  // -- Compile (IMPLEMENTED) ----------------------------------------------
  compile: (body: CompileRequest) => jsonPost("/compile", body).then(j<CompileResult>),

  // -- Gates preview (IMPLEMENTED) ----------------------------------------
  gatesPreview: (body: GatesPreviewRequest) =>
    jsonPost("/gates/preview", body).then(j<GatesPreview>),

  // -- Providers (IMPLEMENTED) --------------------------------------------
  providers: () => fetch(`${BASE}/providers`).then(j<ProviderStatus[]>),
  testProvider: (provider: string) =>
    jsonPost(`/providers/${q(provider)}/test`, {}).then(j<ProviderTestResult>),

  // -- Runs (IMPLEMENTED) --------------------------------------------------
  launchRun: (body: LaunchRequest) => jsonPost("/runs", body).then(j<RunState>),
  runs: () => fetch(`${BASE}/runs`).then(j<RunSummary[]>),
  run: (id: string) => fetch(`${BASE}/runs/${q(id)}`).then(j<RunState>),
  cancelRun: (id: string) => jsonPost(`/runs/${q(id)}/cancel`, {}).then(j<CancelResult>),
  resumeRun: (id: string, budget_cap_usd?: number) =>
    jsonPost(`/runs/${q(id)}/resume`, budget_cap_usd == null ? {} : { budget_cap_usd })
      .then(j<RunState>),
  decideRun: (id: string, body: DecisionRequest) =>
    jsonPost(`/runs/${q(id)}/decision`, body).then(j<DecisionResult>),

  /** SSE endpoint URL for a run's event stream. Tail it with `useRunStream`
      (EventSource) rather than fetching — this returns text/event-stream. */
  runEventsUrl: (id: string) => `${BASE}/runs/${q(id)}/events`,

  // -- Judge / Compare (IMPLEMENTED) --------------------------------------
  /** OPEN compare (aligned diffs + gate/cost/knob deltas) AND the fail-closed
      blind set, over two finished runs. The blind `key` is coordinator-only. */
  judge: (body: JudgeRequest) => jsonPost("/judge", body).then(j<JudgeResult>),
  /** Append a verdict to BOTH runs' run.json (byte-safe, redacted). */
  judgeVerdict: (body: VerdictRequest) =>
    jsonPost("/judge/verdict", body).then(j<VerdictRecordResult>),

  // -- Promote (IMPLEMENTED) ----------------------------------------------
  /** The diffs a promotion WOULD produce (gates value edits + prose patches), no write. */
  promotePreview: (body: PromotePreviewRequest) =>
    jsonPost("/promote/preview", body).then(j<PromotePreview>),
  /** AUTO-APPLY value + simple-patch changes (byte-safe), run --meta-check, return git status. */
  promoteApply: (body: PromotePreviewRequest) =>
    jsonPost("/promote/apply", body).then(j<PromoteApplyResult>),
  /** Generate the PROMOTION_BRIEF.md text for structural (hand-applied) changes. */
  promoteBrief: (body: PromoteBriefRequest) =>
    jsonPost("/promote/brief", body).then(j<PromoteBriefResult>),
};

export type { RunlogEvent };
export type StudioApi = typeof studioApi;
