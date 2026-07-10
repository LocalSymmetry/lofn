import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useStore } from "../store";
import { useHealth, ago, type CheckState } from "../health";

/**
 * The Bridge — "what needs my attention." The instrument's five vitals, promoted
 * out of four separate rooms into one triage ledger: each row is a status + a
 * count + the inline offenders + a freshness stamp + one drill into the fix.
 * Calm and nearly empty when all-green (a healthy cockpit looks calm); red routes,
 * green frees. Every number is a link into the fix, never a mere readout.
 */
type Tone = "ok" | "warn" | "err" | "info" | "idle";

export function Bridge() {
  const { manifest, verify, git } = useStore();
  const drift = useHealth((s) => s.drift);
  const sync = useHealth((s) => s.sync);
  const lastRun = useHealth((s) => s.lastRun);
  const checkDrift = useHealth((s) => s.checkDrift);
  const checkSync = useHealth((s) => s.checkSync);
  const checkLastRun = useHealth((s) => s.checkLastRun);
  const nav = useNavigate();

  useEffect(() => {
    if (!manifest) return;
    checkDrift();
    checkSync(manifest.libraries.map((l: any) => ({ id: l.id, title: l.title })));
    checkLastRun();
  }, [manifest, checkDrift, checkSync, checkLastRun]);

  if (!manifest) return null;
  const missing = verify?.missing || [];
  const orphans = verify?.orphan_step_files || [];
  const manifestOk = (verify?.ok ?? true) && missing.length === 0 && orphans.length === 0;
  const dirty = Object.keys(git);
  const runS = lastRun.summary;

  const allClear =
    manifestOk && dirty.length === 0 &&
    drift.state === "done" && drift.count === 0 &&
    sync.state === "done" && sync.drifted.length === 0;

  const refreshAll = () => {
    checkDrift();
    checkSync(manifest.libraries.map((l: any) => ({ id: l.id, title: l.title })));
    checkLastRun();
  };

  return (
    <>
      <div className="hd">
        <h1>Bridge</h1>
        <span className="faint">what needs your attention</span>
        <div className="spacer" />
        <button className="btn sm" onClick={refreshAll}>↻ re-check all</button>
      </div>
      <div className="scroll pad" style={{ display: "flex", flexDirection: "column", gap: 10, maxWidth: 900 }}>
        {allClear && (
          <div className="card gilt-edge"><div className="card-bd row" style={{ gap: 10 }}>
            <span className="badge canon"><span className="d" />all clear</span>
            <span className="muted">manifest honest · no drift · library in sync · HEAD clean. The instrument is calm — reach for the <a href="#/atlas">Atlas</a> or the <a href="#/studio/run">Run Bench</a>.</span>
          </div></div>
        )}

        {/* 1 — manifest honesty */}
        <LedgerRow
          tone={manifestOk ? "ok" : "err"}
          state="done"
          label="Manifest"
          headline={manifestOk ? "honest" : `${missing.length + orphans.length} disagreement(s)`}
          fresh={null}
          drill={{ label: "Atlas →", to: "/atlas" }}>
          {!manifestOk && (
            <Offenders items={[
              ...missing.map((m) => ({ text: `missing: ${m.path}`, to: `/edit?path=${encodeURIComponent(m.path)}` })),
              ...orphans.map((p) => ({ text: `orphan step file: ${p}`, to: `/edit?path=${encodeURIComponent(p)}` })),
            ]} />
          )}
        </LedgerRow>

        {/* 2 — prose↔YAML drift */}
        <LedgerRow
          tone={drift.state !== "done" ? "idle" : drift.count === 0 ? "ok" : "warn"}
          state={drift.state}
          label="Drift"
          headline={drift.state === "checking" ? "scanning…" : drift.state === "unknown" ? "not checked" : drift.count === 0 ? "prose agrees with gates.yaml" : `${drift.count} restated threshold(s) disagree`}
          fresh={drift.ranAt}
          onRefresh={checkDrift}
          drill={drift.count > 0 ? { label: "reconcile →", to: "/gates?tab=drift" } : undefined}>
          {drift.count > 0 && (
            <Offenders items={dedupe(drift.rows.map((r) => r.file)).map((f) => ({ text: f, to: `/edit?path=${encodeURIComponent(f)}` }))} />
          )}
        </LedgerRow>

        {/* 3 — library sync-drift */}
        <LedgerRow
          tone={sync.state !== "done" ? "idle" : sync.drifted.length === 0 ? "ok" : "warn"}
          state={sync.state}
          label="Library sync"
          headline={sync.state === "checking" ? "checking…" : sync.state === "unknown" ? "not checked" : sync.drifted.length === 0 ? `${sync.checked} collection(s) in sync` : `${sync.drifted.length} collection(s) drifted`}
          fresh={sync.ranAt}
          onRefresh={() => checkSync(manifest.libraries.map((l: any) => ({ id: l.id, title: l.title })))}
          drill={sync.drifted.length > 0 ? { label: "regenerate →", to: `/library?c=${sync.drifted[0].id}` } : undefined}>
          {sync.drifted.length > 0 && (
            <Offenders items={sync.drifted.map((d) => ({
              text: `${d.title}: ${d.missing} missing · ${d.stale} stale in derived`,
              to: `/library?c=${d.id}`,
            }))} />
          )}
        </LedgerRow>

        {/* 4 — working tree */}
        <LedgerRow
          tone={dirty.length === 0 ? "ok" : "info"}
          state="done"
          label="Working tree"
          headline={dirty.length === 0 ? "HEAD clean" : `${dirty.length} uncommitted file(s)`}
          fresh={null}>
          {dirty.length > 0 && (
            <div className="row wrap" style={{ gap: 6, marginTop: 4 }}>
              {Object.entries(district(dirty)).map(([d, n]) => (
                <span key={d} className="badge dirty"><span className="d" />{d} · {n}</span>
              ))}
              <span className="spacer" />
              {dirty.slice(0, 6).map((p) => (
                <a key={p} className="mono faint" style={{ fontSize: 10.5 }} href={`#/edit?path=${encodeURIComponent(p)}`}>{shortPath(p)}</a>
              ))}
              {dirty.length > 6 && <span className="faint" style={{ fontSize: 10.5 }}>+{dirty.length - 6} more</span>}
            </div>
          )}
        </LedgerRow>

        {/* 5 — last run */}
        <LedgerRow
          tone={runToneFor(lastRun.state, runS)}
          state={lastRun.state}
          label="Last run"
          headline={
            lastRun.state === "checking" ? "checking…" :
            !runS ? "no studio runs yet" :
            `${runS.status}${runS.budget?.aborted_on_cap ? " · budget capped" : ""}${runS.totals?.pairs_quarantined ? ` · ${runS.totals.pairs_quarantined} quarantined` : ""}`
          }
          fresh={lastRun.ranAt}
          onRefresh={checkLastRun}
          drill={runS ? { label: "Run Bench →", to: "/studio/run" } : undefined}>
          {runS && (
            <div className="faint mono" style={{ fontSize: 10.5, marginTop: 3 }}>
              {runS.title} · {runS.flow?.name}@{runS.flow?.version}
              {runS.totals?.cost_usd != null ? ` · $${runS.totals.cost_usd.toFixed(3)}` : ""}
            </div>
          )}
        </LedgerRow>
      </div>
    </>
  );
}

function LedgerRow({ tone, state, label, headline, fresh, onRefresh, drill, children }: {
  tone: Tone; state: CheckState; label: string; headline: string; fresh: number | null;
  onRefresh?: () => void; drill?: { label: string; to: string }; children?: React.ReactNode;
}) {
  const nav = useNavigate();
  const badgeClass = tone === "ok" ? "ok" : tone === "err" ? "err" : tone === "warn" ? "warn" : tone === "info" ? "dirty" : "";
  return (
    <div className="card">
      <div className="card-bd">
        <div className="row" style={{ gap: 10 }}>
          <Led tone={tone} spinning={state === "checking"} />
          <b style={{ minWidth: 108 }}>{label}</b>
          <span className={`badge ${badgeClass}`}><span className="d" />{headline}</span>
          <span className="spacer" />
          {fresh !== null && <span className="faint mono" style={{ fontSize: 10.5 }} title="freshness — a stale green is not a fact">{ago(fresh)}</span>}
          {onRefresh && <button className="btn ghost sm" title="re-check" onClick={onRefresh}>↻</button>}
          {drill && <button className="btn sm primary" onClick={() => nav(drill.to)}>{drill.label}</button>}
        </div>
        {children}
      </div>
    </div>
  );
}

/** by-exception LED: near-invisible when ok, recruits color only on deviation. */
function Led({ tone, spinning }: { tone: Tone; spinning?: boolean }) {
  if (spinning) return <span className="spin" style={{ width: 12, height: 12 }} />;
  const color =
    tone === "err" ? "var(--sev-hard)" :
    tone === "warn" ? "var(--sev-flag)" :
    tone === "info" ? "var(--dirty)" :
    tone === "idle" ? "var(--text-faint)" :
    "var(--border-strong)"; // ok = quiet; green whispers, it does not shout
  const glow = tone === "err" || tone === "warn";
  return <span style={{
    width: 10, height: 10, borderRadius: "50%", background: color, flex: "none",
    boxShadow: glow ? `0 0 8px ${color}` : "none",
  }} />;
}

function Offenders({ items }: { items: { text: string; to: string }[] }) {
  return (
    <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 2 }}>
      {items.slice(0, 8).map((it, i) => (
        <a key={i} className="mono" style={{ fontSize: 11, color: "var(--text-dim)" }} href={`#${it.to}`}>{it.text}</a>
      ))}
      {items.length > 8 && <span className="faint" style={{ fontSize: 10.5 }}>+{items.length - 8} more</span>}
    </div>
  );
}

const dedupe = (a: string[]) => Array.from(new Set(a));
const shortPath = (p: string) => p.split("/").slice(-2).join("/");
function district(paths: string[]): Record<string, number> {
  const d: Record<string, number> = {};
  for (const p of paths) {
    const top = p.split("/")[0] || "·";
    d[top] = (d[top] || 0) + 1;
  }
  return d;
}
function runToneFor(state: CheckState, s: any): Tone {
  if (state === "checking") return "idle";
  if (!s) return "idle";
  if (s.budget?.aborted_on_cap) return "warn";
  if (s.status === "failed" || s.status === "cancelled") return "err";
  if (s.status === "complete") return "ok";
  return "info";
}
