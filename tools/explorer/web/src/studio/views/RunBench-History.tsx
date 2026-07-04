import { useEffect, useState } from "react";
import { studioApi } from "../api";
import type { RunSummary } from "../types";

/**
 * Run Bench — run-history table (plan §8.3). Provenance columns: flow@ver,
 * knobs summary, cost vs cap, status, verdicts. Click a row to open its live
 * monitor (a finished run replays identically from its runlog).
 */
export function RunHistory({ onOpen, refreshKey }: { onOpen: (id: string) => void; refreshKey: number }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    studioApi.runs()
      .then((r) => { if (live) setRuns(r); })
      .catch((e) => { if (live) setError(e?.message || String(e)); });
    return () => { live = false; };
  }, [refreshKey]);

  return (
    <div className="card"><div className="card-bd" style={{ padding: 0 }}>
      <div className="row" style={{ justifyContent: "space-between", padding: "10px 12px", borderBottom: "1px solid var(--border)" }}>
        <b>Run history</b>
        <span className="faint mono" style={{ fontSize: 11 }}>{runs ? `${runs.length} run(s)` : ""}</span>
      </div>
      {error && <div className="banner err" style={{ borderRadius: 0 }}><span className="mono" style={{ fontSize: 11 }}>{error}</span></div>}
      {runs && runs.length === 0 && <div className="empty">no runs yet — launch one above.</div>}
      {runs && runs.length > 0 && (
        <div className="scroll" style={{ overflowX: "auto" }}>
          <table className="grid">
            <thead>
              <tr>
                <th>run</th>
                <th>flow@ver</th>
                <th>status</th>
                <th style={{ textAlign: "right" }}>cost / cap</th>
                <th>pairs</th>
                <th>verdicts</th>
                <th>created</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} style={{ cursor: "pointer" }} onClick={() => onOpen(r.id)}>
                  <td className="mono" style={{ fontSize: 11 }} title={r.id}>
                    <span style={{ color: "var(--accent)" }}>{r.title}</span>
                    <div className="faint" style={{ fontSize: 9.5 }}>{r.id}</div>
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {r.flow?.name}@{r.flow?.version}
                    <div className="faint" style={{ fontSize: 9.5 }}>{r.flow?.file_sha?.slice(0, 7)}</div>
                  </td>
                  <td><StatusCell status={r.status} quarantine={r.quarantine_banner} /></td>
                  <td className="mono" style={{ fontSize: 11, textAlign: "right" }}>
                    {money(r.totals?.cost_usd ?? r.budget?.spend_usd)}
                    <span className="faint"> / {money(r.budget?.cap_usd)}</span>
                    {r.budget?.aborted_on_cap && <div className="faint" style={{ color: "var(--sev-hard)", fontSize: 9.5 }}>cap hit</div>}
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {r.totals?.pairs_total ?? "—"}
                    {(r.totals?.pairs_quarantined ?? 0) > 0 && (
                      <span style={{ color: "var(--sev-flag)" }}> · {r.totals?.pairs_quarantined}q</span>
                    )}
                  </td>
                  <td><VerdictCell verdicts={(r as any).verdicts} /></td>
                  <td className="faint mono" style={{ fontSize: 10.5 }}>{fmtTime(r.created)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {!runs && !error && <div className="empty row" style={{ justifyContent: "center" }}><span className="spin" /></div>}
    </div></div>
  );
}

function StatusCell({ status, quarantine }: { status: string; quarantine?: unknown }) {
  const badge = status === "complete" ? "ok" : status === "failed" ? "err"
    : status === "cancelled" ? "warn" : "";
  return (
    <span className="row" style={{ gap: 4 }}>
      <span className={`badge ${badge}`}><span className={`d ${status === "running" ? "pulse" : ""}`} />{status}</span>
      {quarantine ? <span className="chip flag" title={bannerText(quarantine)}><span className="d" />q</span> : null}
    </span>
  );
}

function VerdictCell({ verdicts }: { verdicts?: unknown[] }) {
  if (!verdicts || verdicts.length === 0) return <span className="faint">—</span>;
  return (
    <span className="row" style={{ gap: 4, flexWrap: "wrap" }}>
      {verdicts.slice(0, 3).map((v: any, i) => (
        <span key={i} className="chip" style={{ fontSize: 9.5 }} title={JSON.stringify(v)}>
          {String(v?.verdict ?? v?.winner ?? v?.result ?? "verdict")}
        </span>
      ))}
      {verdicts.length > 3 && <span className="faint">+{verdicts.length - 3}</span>}
    </span>
  );
}

function money(n?: number): string {
  if (n == null) return "—";
  return n >= 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(3)}`;
}
function fmtTime(iso?: string): string {
  if (!iso) return "";
  try { return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
  catch { return iso; }
}
function bannerText(b: unknown): string {
  if (typeof b === "string") return b;
  return (b as any)?.message ?? JSON.stringify(b);
}
