import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { studioApi } from "../api";
import { useRunStream, CostBar, StatusLight } from "../components";
import { StackedStrip, DotStrip } from "../../components/charts";
import type {
  RunState, StepResult, StepStatus, GateResult, RunlogEvent,
} from "../types";
import { ArtifactViewer } from "./RunBench-Artifact";

const TERMINAL: Record<string, true> = { complete: true, failed: true, cancelled: true };

/** A pending HOLD_FOR_HUMAN, surfaced from a `hold_for_human` runlog frame and
    cleared by a `hold_resolved` frame (or a decision POST). */
interface Hold {
  pair: string | null;
  step: string;
  gate?: string;
  detail?: string;
  message?: string;
}

interface Props {
  runId: string;
  onExit: () => void;
}

/**
 * Run Bench — live monitor / replay (plan §8.3). One lane per pair PLUS a
 * coordinator lane; step cards fill as the run.json steps land; gate chips light
 * green/amber/red; a cost ticker vs the budget bar; quarantine banners; a
 * HOLD_FOR_HUMAN modal (proceed/kill). Cancel/Resume drive the run service.
 *
 * A FINISHED run renders identically from its runlog (replay): run.json carries
 * the full structured step records, the SSE stream replays every frame from the
 * start, and both consumers reconstruct the same view — no live-only state.
 */
export function LiveMonitor({ runId, onExit }: Props) {
  const [run, setRun] = useState<RunState | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [openArtifact, setOpenArtifact] = useState<string | null>(null);
  const { events, connected, ended } = useRunStream(runId);

  const terminal = run != null && TERMINAL[run.status];

  // Poll run.json for the structured step records (SSE `step` frames are light;
  // run.json.steps carries the gate results, artifacts, tokens, costs). Stop
  // polling once terminal AND the stream ended (replay done).
  const refresh = useCallback(() => {
    studioApi.run(runId).then(setRun).catch((e) => setLoadErr(e?.message || String(e)));
  }, [runId]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (terminal && ended) return;                 // done — no more polling
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, [terminal, ended, refresh]);

  // Also refresh promptly when a lifecycle frame arrives (snappier than the poll).
  const lastLifecycle = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i].event;
      if (e === "step" || e === "run_end" || e === "pair_done" || e === "quarantine") return events.length;
    }
    return 0;
  }, [events]);
  useEffect(() => { if (lastLifecycle) refresh(); }, [lastLifecycle, refresh]);

  // -- lanes: coordinator (pair null) + one per observed pair label --------
  const lanes = useMemo(() => buildLanes(run?.steps || [], events), [run?.steps, events]);

  // cost composition (where the spend went, aggregated by step across pairs) +
  // latency (which step stalls). Memoized off run.json.steps so they don't
  // redraw on every SSE tick — lane-level, never one chart per step card.
  const costByStep = useMemo(() => {
    const m = new Map<string, number>();
    for (const s of run?.steps || []) if (s.cost_usd && s.cost_usd > 0) m.set(s.step, (m.get(s.step) || 0) + s.cost_usd);
    return Array.from(m, ([key, value]) => ({ key, value }));
  }, [run?.steps]);
  const maxDur = useMemo(() => (run?.steps || []).reduce((mx, s) => Math.max(mx, s.duration_s || 0), 0), [run?.steps]);

  // -- pending human hold (from the event stream) --------------------------
  const hold = useMemo(() => pendingHold(events), [events]);

  // -- quarantine banner ---------------------------------------------------
  const banner = useMemo(() => quarantineBanner(run, events), [run, events]);

  // -- cost ticker ---------------------------------------------------------
  const spend = run?.budget?.spend_usd ?? run?.totals?.cost_usd ?? 0;
  const cap = run?.budget?.cap_usd ?? 0;
  const abortedOnCap = run?.budget?.aborted_on_cap;

  const cancel = () => act("cancel", () => studioApi.cancelRun(runId));
  const resume = () => {
    let raise: number | undefined;
    if (abortedOnCap) {
      const v = window.prompt("Raise the budget cap to resume (USD):", String(cap + 5));
      if (v == null) return;
      const n = parseFloat(v);
      if (Number.isFinite(n) && n >= 0) raise = n;
    }
    act("resume", () => studioApi.resumeRun(runId, raise));
  };
  const decide = (pair: string | null, decision: "proceed" | "kill") =>
    act(`decide-${decision}`, () => studioApi.decideRun(runId, { pair, decision }).then(() => { }));

  function act(label: string, fn: () => Promise<unknown>) {
    setBusy(label);
    fn().then(refresh).catch((e) => setLoadErr(e?.message || String(e))).finally(() => setBusy(null));
  }

  if (loadErr && !run) {
    return <div className="banner err" style={{ borderRadius: 6 }}>{loadErr}
      <button className="btn sm" onClick={onExit}>back</button></div>;
  }
  if (!run) return <div className="empty row" style={{ justifyContent: "center" }}><span className="spin" /></div>;

  return (
    <div style={{ display: "grid", gap: 12 }}>
      {/* header: title, status, controls, cost */}
      <div className="card"><div className="card-bd" style={{ display: "grid", gap: 10 }}>
        <div className="row" style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
          <div className="row" style={{ gap: 8 }}>
            <button className="btn sm ghost" onClick={onExit}>← runs</button>
            <b>{run.title}</b>
            <StatusChip status={run.status} live={connected} />
            {run.context?.input && <span className="faint mono" style={{ fontSize: 11 }} title={run.context.input}>“{trunc(run.context.input, 60)}”</span>}
          </div>
          <div className="row" style={{ gap: 6 }}>
            <span className="chip" title="flow provenance">
              <span className="mono" style={{ fontSize: 10.5 }}>{run.flow.name}@{run.flow.version}</span>
              <span className="faint mono" style={{ fontSize: 10 }}>{run.flow.file_sha?.slice(0, 7)}</span>
            </span>
            {!terminal && <button className="btn sm" disabled={busy != null} onClick={cancel}>{busy === "cancel" ? "…" : "Cancel"}</button>}
            {terminal && run.status !== "complete" && (
              <button className="btn sm primary" disabled={busy != null} onClick={resume}>{busy === "resume" ? "…" : "Resume"}</button>
            )}
          </div>
        </div>
        <CostBar spend={spend} cap={cap} unpriced={spend === 0 && (run.totals?.tokens_out ?? 0) > 0} />
        {costByStep.length > 1 && (
          <div style={{ display: "grid", gap: 3 }}>
            <span className="faint mono" style={{ fontSize: 10 }}>where the spend went — by step</span>
            <StackedStrip segments={costByStep} labelTop={3} fmt={(n) => (n >= 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(3)}`)} />
          </div>
        )}
        <div className="row" style={{ gap: 12, flexWrap: "wrap", fontSize: 11 }}>
          <Meter k="tokens in" v={(run.totals?.tokens_in ?? 0).toLocaleString()} />
          <Meter k="tokens out" v={(run.totals?.tokens_out ?? 0).toLocaleString()} />
          <Meter k="pairs" v={`${run.totals?.pairs_total ?? lanes.filter((l) => l.pair !== null).length}${(run.totals?.pairs_quarantined ?? 0) ? ` · ${run.totals?.pairs_quarantined} quarantined` : ""}`} />
          {run.totals?.duration_s != null && <Meter k="duration" v={`${run.totals.duration_s.toFixed(1)}s`} />}
          {run.canon_dirty && <span className="badge warn"><span className="d" />canon dirty at launch</span>}
        </div>
      </div></div>

      {/* quarantine banner — LOUD, before anything else */}
      {banner && (
        <div className="banner warn" style={{ borderRadius: 6, fontWeight: 550 }}>
          <span className="chip flag"><span className="d" />quarantine</span>
          <span>{banner}</span>
        </div>
      )}
      {abortedOnCap && (
        <div className="banner err" style={{ borderRadius: 6 }}>
          <b>budget cap reached</b> — run halted at ${cap.toFixed(2)}. Resume to raise the cap and continue.
        </div>
      )}

      {/* lanes */}
      <div style={{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))" }}>
        {lanes.map((lane) => (
          <Lane key={lane.key} lane={lane} maxDur={maxDur} onOpenArtifact={setOpenArtifact} />
        ))}
        {lanes.length === 0 && <div className="empty">waiting for the first step…</div>}
      </div>

      {loadErr && <div className="banner err" style={{ borderRadius: 6 }}><span className="mono" style={{ fontSize: 11 }}>{loadErr}</span></div>}

      {/* HOLD_FOR_HUMAN modal */}
      {hold && !terminal && (
        <HoldModal hold={hold} busy={busy} onDecide={decide} />
      )}

      {/* artifact viewer (Monaco) */}
      {openArtifact && (
        <ArtifactViewer runId={runId} artifact={openArtifact} onClose={() => setOpenArtifact(null)} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Lane + step card
// ---------------------------------------------------------------------------

interface LaneModel {
  key: string;
  pair: string | null;         // null = coordinator lane
  label: string;
  steps: StepResult[];
  quarantined: boolean;
}

function Lane({ lane, maxDur, onOpenArtifact }: { lane: LaneModel; maxDur: number; onOpenArtifact: (a: string) => void }) {
  const timed = lane.steps.some((s) => s.duration_s);
  return (
    <div className="card" style={lane.quarantined ? { borderColor: "color-mix(in srgb, var(--sev-flag) 45%, var(--border))" } : undefined}>
      <div className="card-hd row" style={{ justifyContent: "space-between", padding: "8px 12px" }}>
        <span className="row" style={{ gap: 6 }}>
          <span style={{ fontSize: 12 }}>{lane.pair === null ? "◆" : "▸"}</span>
          <b style={{ fontSize: 12.5 }}>{lane.label}</b>
        </span>
        {lane.quarantined
          ? <span className="chip flag"><span className="d" />quarantined</span>
          : <span className="faint mono" style={{ fontSize: 10.5 }}>{lane.steps.filter((s) => s.status === "done" || s.status === "flagged").length}/{lane.steps.length}</span>}
      </div>
      {maxDur > 0 && timed && (
        <div style={{ padding: "4px 12px 0" }} title="per-step timing — the amber dot is this lane's slowest step">
          <DotStrip values={lane.steps.map((s) => ({ label: s.step, value: s.duration_s || 0 }))} max={maxDur} width={150} fmt={(n) => `${n.toFixed(1)}s`} />
        </div>
      )}
      <div style={{ display: "grid", gap: 6, padding: 8 }}>
        {lane.steps.map((s) => <StepCard key={`${s.step}:${s.pair ?? ""}`} s={s} onOpenArtifact={onOpenArtifact} />)}
      </div>
    </div>
  );
}

function StepCard({ s, onOpenArtifact }: { s: StepResult; onOpenArtifact: (a: string) => void }) {
  const dot = statusColor(s.status);
  const active = s.status === "resolving" || s.status === "calling" || s.status === "validating" || s.status === "retrying";
  return (
    <div style={{
      border: "1px solid var(--border)", borderRadius: 6, padding: "6px 8px",
      background: "var(--surface-2)", display: "grid", gap: 4,
    }}>
      <div className="row" style={{ justifyContent: "space-between", gap: 6 }}>
        <span className="row" style={{ gap: 6, minWidth: 0 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: dot, flexShrink: 0 }}
                className={active ? "pulse" : ""} />
          <span className="mono" style={{ fontSize: 11.5, whiteSpace: "nowrap" }}>{s.step}</span>
          {s.attempts != null && s.attempts > 1 && <span className="faint mono" style={{ fontSize: 10 }}>×{s.attempts}</span>}
        </span>
        <span className="faint mono" style={{ fontSize: 10 }}>{s.status}</span>
      </div>
      {(s.gate_results && s.gate_results.length > 0) && (
        <div className="row" style={{ gap: 4, flexWrap: "wrap" }}>
          {s.gate_results.map((g) => <GateChip key={g.gate} g={g} />)}
        </div>
      )}
      <div className="row" style={{ justifyContent: "space-between", gap: 6 }}>
        <span className="faint mono" style={{ fontSize: 9.5 }}>
          {s.tokens_out != null ? `${s.tokens_out}t` : ""}{s.cost_usd != null ? ` · $${s.cost_usd.toFixed(4)}` : ""}
        </span>
        {s.artifact && (
          <button className="btn sm ghost" style={{ fontSize: 10, padding: "1px 6px" }}
                  onClick={() => onOpenArtifact(s.artifact!)} title={s.artifact_sha || undefined}>
            {trunc(s.artifact, 22)}
          </button>
        )}
      </div>
    </div>
  );
}

function GateChip({ g }: { g: GateResult }) {
  // pass === true -> green/ok ; false -> hard red ; null -> amber FLAG
  const cls = g.pass === true ? "" : g.pass === false ? "hard" : "flag";
  const badge = g.pass === true ? "ok" : g.pass === false ? "err" : "warn";
  return (
    <span className={`badge ${badge}`} title={g.detail || g.gate} style={{ fontSize: 9.5 }}>
      <span className="d" />{shortGate(g.gate)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// HOLD_FOR_HUMAN modal
// ---------------------------------------------------------------------------

function HoldModal({ hold, busy, onDecide }: {
  hold: Hold;
  busy: string | null;
  onDecide: (pair: string | null, d: "proceed" | "kill") => void;
}) {
  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", zIndex: 50,
      display: "flex", alignItems: "center", justifyContent: "center", padding: 20,
    }}>
      <div className="card" style={{ maxWidth: 520, width: "100%", borderColor: "var(--sev-flag)" }}>
        <div className="card-hd row" style={{ gap: 8 }}>
          <span className="chip flag"><span className="d" />HOLD_FOR_HUMAN</span>
          <b>a human must decide</b>
        </div>
        <div className="card-bd" style={{ display: "grid", gap: 10 }}>
          <div className="row" style={{ gap: 8 }}>
            <span className="mono" style={{ fontSize: 11 }}>step {hold.step}</span>
            {hold.pair && <span className="mono" style={{ fontSize: 11 }}>pair {hold.pair}</span>}
            {hold.gate && <span className="badge warn"><span className="d" />{shortGate(hold.gate)}</span>}
          </div>
          {hold.message && <div style={{ fontSize: 12.5, color: "var(--sev-flag)" }}>{hold.message}</div>}
          {hold.detail && <div className="mono scroll" style={{ fontSize: 11, maxHeight: 160, background: "var(--surface-2)", padding: 8, borderRadius: 6, whiteSpace: "pre-wrap" }}>{hold.detail}</div>}
          <div className="row" style={{ gap: 8, justifyContent: "flex-end" }}>
            <button className="btn" disabled={busy != null} onClick={() => onDecide(hold.pair, "kill")}
                    style={{ borderColor: "var(--sev-hard)", color: "var(--sev-hard)" }}>
              {busy === "decide-kill" ? "…" : "Kill pair"}
            </button>
            <button className="btn primary" disabled={busy != null} onClick={() => onDecide(hold.pair, "proceed")}>
              {busy === "decide-proceed" ? "…" : "Proceed"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// derivations from run.json.steps + the runlog event stream
// ---------------------------------------------------------------------------

function buildLanes(steps: StepResult[], events: RunlogEvent[]): LaneModel[] {
  const byPair = new Map<string | null, StepResult[]>();
  const order: (string | null)[] = [];
  const push = (pair: string | null, s: StepResult) => {
    if (!byPair.has(pair)) { byPair.set(pair, []); order.push(pair); }
    byPair.get(pair)!.push(s);
  };
  for (const s of steps) push(s.pair ?? null, s);

  // Seed lanes for pairs the event stream announced before run.json caught up.
  for (const e of events) {
    if (e.event === "pair_start" && typeof e.pair === "string" && !byPair.has(e.pair)) {
      byPair.set(e.pair, []); order.push(e.pair);
    }
  }

  const quarantinedPairs = new Set<string>();
  for (const e of events) {
    if ((e.event === "pair_quarantined" || e.event === "quarantine") && typeof e.pair === "string") {
      quarantinedPairs.add(e.pair);
    }
  }
  for (const s of steps) if (s.status === "quarantined" && s.pair) quarantinedPairs.add(s.pair);

  // coordinator lane first, then pair lanes in numeric order
  order.sort((a, b) => {
    if (a === null) return -1;
    if (b === null) return 1;
    return String(a).localeCompare(String(b), undefined, { numeric: true });
  });

  return order.map((pair) => ({
    key: pair === null ? "__coordinator" : `pair-${pair}`,
    pair,
    label: pair === null ? "coordinator" : `pair ${pair}`,
    steps: byPair.get(pair) || [],
    quarantined: pair !== null && quarantinedPairs.has(pair),
  }));
}

function pendingHold(events: RunlogEvent[]): Hold | null {
  // Latest unresolved hold_for_human (a matching hold_resolved with the same
  // pair after it clears it).
  const holds: Hold[] = [];
  const resolved = new Set<string>();
  for (const e of events) {
    const pair = (e.pair ?? null) as string | null;
    const key = `${e.step}:${pair ?? ""}`;
    if (e.event === "hold_for_human") {
      holds.push({
        pair, step: String(e.step),
        gate: e.gate as string | undefined,
        detail: e.detail as string | undefined,
        message: e.message as string | undefined,
      });
    } else if (e.event === "hold_resolved") {
      resolved.add(key);
    }
  }
  for (let i = holds.length - 1; i >= 0; i--) {
    const h = holds[i];
    if (!resolved.has(`${h.step}:${h.pair ?? ""}`)) return h;
  }
  return null;
}

function quarantineBanner(run: RunState | null, events: RunlogEvent[]): string | null {
  if (run?.quarantine_banner) {
    return typeof run.quarantine_banner === "string"
      ? run.quarantine_banner
      : (run.quarantine_banner as any)?.message ?? JSON.stringify(run.quarantine_banner);
  }
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].event === "quarantine_banner") return String(events[i].message ?? "");
  }
  return null;
}

// ---------------------------------------------------------------------------
// tiny presentational helpers
// ---------------------------------------------------------------------------

function Meter({ k, v }: { k: string; v: string }) {
  return <span className="faint mono"><span className="faint">{k}</span> <span style={{ color: "var(--text-dim)" }}>{v}</span></span>;
}

function StatusChip({ status, live }: { status: string; live: boolean }) {
  const badge = status === "complete" ? "ok" : status === "failed" ? "err" : status === "cancelled" ? "warn" : "";
  return (
    <span className={`badge ${badge}`}>
      <span className={`d ${status === "running" ? "pulse" : ""}`} />{status}{status === "running" && live ? " ·live" : ""}
    </span>
  );
}

function statusColor(s: StepStatus): string {
  switch (s) {
    case "done": return "var(--ok)";
    case "flagged": return "var(--sev-flag)";
    case "failed":
    case "quarantined": return "var(--sev-hard)";
    case "held_for_human": return "var(--accent-2)";
    case "retrying": return "var(--sev-flag)";
    case "calling":
    case "resolving":
    case "validating": return "var(--accent)";
    default: return "var(--text-faint)";
  }
}

function shortGate(g: string): string {
  return g.replace(/^g\./, "").replace(/_/g, " ");
}
function trunc(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
