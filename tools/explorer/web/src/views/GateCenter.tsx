import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { Editor } from "../components/Editor";
import { useStore } from "../store";
import type { GatesData, MetaCheck, ValidatorResult } from "../types";

type Tab = "thresholds" | "drift" | "map" | "validators" | "raw";

export function GateCenter() {
  const { manifest } = useStore();
  const [params] = useSearchParams();
  const [tab, setTab] = useState<Tab>(params.get("gate") ? "map" : "thresholds");
  const [gates, setGates] = useState<GatesData | null>(null);
  const highlight = params.get("gate");

  useEffect(() => { api.gates().then(setGates).catch(() => setGates(null)); }, []);

  const tabs: [Tab, string][] = [
    ["thresholds", "Thresholds"], ["drift", "Drift"], ["map", "Gate map"],
    ["validators", "Validators"], ["raw", "gates.yaml"],
  ];

  return (
    <>
      <div className="hd">
        <h1>Gate Center</h1>
        <div className="row" style={{ gap: 4, marginLeft: 12 }}>
          {tabs.map(([t, label]) => (
            <button key={t} className={`btn sm ${tab === t ? "primary" : "ghost"}`} onClick={() => setTab(t)}>{label}</button>
          ))}
        </div>
        <div className="spacer" />
        <span className="faint mono" style={{ fontSize: 11 }}>{manifest?.gates_file}</span>
      </div>
      {tab === "thresholds" && <Thresholds gates={gates} />}
      {tab === "drift" && <Drift />}
      {tab === "map" && <GateMap highlight={highlight} />}
      {tab === "validators" && <Validators />}
      {tab === "raw" && <div style={{ flex: 1, minHeight: 0 }}><Editor path={manifest!.gates_file} /></div>}
    </>
  );
}

function Thresholds({ gates }: { gates: GatesData | null }) {
  if (!gates) return <div className="empty row" style={{ justifyContent: "center" }}><span className="spin" /></div>;
  return (
    <div className="scroll pad" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 12, alignContent: "start" }}>
      {gates.groups.map((grp) => (
        <div key={grp.section} className="card">
          <div className="card-hd" style={{ fontSize: 12 }}>{grp.section}</div>
          <div className="card-bd" style={{ padding: 0 }}>
            <table className="grid">
              <tbody>
                {grp.entries.map((e) => (
                  <tr key={e.key}>
                    <td className="mono" style={{ color: "var(--text-dim)" }}>{e.key}</td>
                    <td className="mono" style={{ color: "var(--accent)", textAlign: "right" }}>{fmt(e.value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}

function Drift() {
  const [mc, setMc] = useState<MetaCheck | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => { setLoading(true); try { setMc(await api.metacheck()); } finally { setLoading(false); } };
  useEffect(() => { run(); }, []);
  return (
    <div className="scroll pad">
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="btn primary" onClick={run} disabled={loading}>{loading ? "scanning…" : "Re-run meta-check"}</button>
        {mc && <span className="faint">{mc.summary || `${mc.count} disagreement(s)`} · {mc.duration_ms}ms</span>}
      </div>
      <p className="muted" style={{ maxWidth: 640, fontSize: 12 }}>
        Prose vs <span className="mono">gates.yaml</span> drift — a restated threshold in a skill/reference file that disagrees with the single source. WARN-only in the pipeline; here it's the fix list.
      </p>
      {!mc ? null : mc.rows.length === 0 ? (
        <div className="card"><div className="card-bd row"><span className="badge ok"><span className="d" />no drift</span> every restated number agrees with gates.yaml.</div></div>
      ) : (
        <div className="card"><table className="grid">
          <thead><tr><th>File</th><th>Threshold</th><th>Restated as</th><th></th></tr></thead>
          <tbody>
            {mc.rows.map((r, i) => (
              <tr key={i}>
                <td className="mono" style={{ fontSize: 11 }}>{r.file}</td>
                <td className="mono" style={{ color: "var(--text-dim)" }}>{r.label}</td>
                <td className="mono" style={{ color: "var(--sev-flag)" }}>{r.value}</td>
                <td><a href={`#/edit?path=${encodeURIComponent(r.file)}`}>open →</a></td>
              </tr>
            ))}
          </tbody>
        </table></div>
      )}
    </div>
  );
}

function GateMap({ highlight }: { highlight: string | null }) {
  const { manifest } = useStore();
  if (!manifest) return null;
  return (
    <div className="scroll pad" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {manifest.gates.map((g) => (
        <div key={g.id} id={g.id} className="card" style={{ outline: highlight === g.id ? "2px solid var(--accent)" : "none" }}>
          <div className="card-bd" style={{ display: "grid", gridTemplateColumns: "150px 1fr", gap: 12 }}>
            <div>
              <span className={`chip ${g.severity}`}><span className="d" />{g.severity}</span>
              <div className="faint" style={{ fontSize: 10.5, marginTop: 6 }}>level: {g.level}</div>
            </div>
            <div>
              <b>{g.title}</b>
              {g.note && <div className="muted" style={{ fontSize: 12, marginTop: 3 }}>{g.note}</div>}
              <div className="row wrap" style={{ gap: 6, marginTop: 6 }}>
                {g.keys.map((k) => <span key={k} className="mono badge" style={{ fontSize: 10 }}>{k}</span>)}
                {g.validator && <span className="faint mono" style={{ fontSize: 10.5 }}>▸ {g.validator}</span>}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function Validators() {
  const { manifest } = useStore();
  const [argMap, setArgMap] = useState<Record<string, string>>({});
  const [out, setOut] = useState<Record<string, ValidatorResult>>({});
  const [busy, setBusy] = useState<string | null>(null);
  if (!manifest) return null;
  const run = async (id: string) => {
    setBusy(id);
    try {
      const args = (argMap[id] || "").trim().split(/\s+/).filter(Boolean);
      const res = await api.runValidator(id, args);
      setOut((o) => ({ ...o, [id]: res }));
    } catch (e: any) { setOut((o) => ({ ...o, [id]: { cmd: id, returncode: -1, stdout: "", stderr: e.message, duration_ms: 0 } })); }
    finally { setBusy(null); }
  };
  return (
    <div className="scroll pad" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {manifest.validators.map((v) => (
        <div key={v.id} className="card">
          <div className="card-bd">
            <div className="row"><b className="mono">{v.id}</b><span className="badge">{v.level}</span><span className="spacer" /><span className="faint mono" style={{ fontSize: 10.5 }}>{v.path}</span></div>
            <div className="mono faint" style={{ fontSize: 11, margin: "6px 0" }}>{v.usage}</div>
            <div className="row">
              <input className="input mono" style={{ fontSize: 11 }} placeholder="args (e.g. --meta-check --root .)" value={argMap[v.id] || ""}
                onChange={(e) => setArgMap((m) => ({ ...m, [v.id]: e.target.value }))} />
              <button className="btn sm primary" disabled={busy === v.id} onClick={() => run(v.id)}>{busy === v.id ? "…" : "Run"}</button>
            </div>
            {out[v.id] && (
              <div className="mono" style={{ fontSize: 11, marginTop: 8, background: "var(--bg)", borderRadius: 5, padding: 8, maxHeight: 220, overflow: "auto" }}>
                <div className={out[v.id].returncode === 0 ? "" : ""} style={{ color: out[v.id].returncode === 0 ? "var(--ok)" : "var(--sev-hard)" }}>exit {out[v.id].returncode} · {out[v.id].duration_ms}ms</div>
                {out[v.id].stdout && <pre style={{ margin: "4px 0", whiteSpace: "pre-wrap" }}>{out[v.id].stdout}</pre>}
                {out[v.id].stderr && <pre style={{ margin: 0, whiteSpace: "pre-wrap", color: "var(--text-faint)" }}>{out[v.id].stderr}</pre>}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

const fmt = (v: any) => Array.isArray(v) ? `[${v.join(", ")}]` : typeof v === "boolean" ? String(v) : String(v);
