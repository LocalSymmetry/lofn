import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useStore } from "../store";
import { ArcFan, HypsometricKey } from "../components/charts";
import type { Gate, Rail, RailNode } from "../types";

const SEV_ORDER = ["hard", "mixed", "flag", "prose"];

export function Atlas() {
  const { manifest, git } = useStore();
  const nav = useNavigate();
  const [showGates, setShowGates] = useState(true);
  const [showEditions, setShowEditions] = useState(false);
  const [showExec, setShowExec] = useState(false);

  const gateById = useMemo(() => {
    const m: Record<string, Gate> = {};
    manifest?.gates.forEach((g) => (m[g.id] = g));
    return m;
  }, [manifest]);

  if (!manifest) return null;
  const edgeGates: Record<string, string[]> = (manifest as any).edge_gates || {};
  // the dispatch fan-out: the orchestrator freezes once and fans to every
  // modality rail. Rendered as an arc-fan, in each rail's own hue.
  const dispatchTargets = manifest.rails
    .filter((r) => r.kind !== "orchestrator")
    .map((r) => ({ label: r.id, color: `var(--rail-${r.id})` }));

  return (
    <>
      <div className="hd">
        <h1>Atlas</h1>
        <span className="faint">orchestrator → 4 modality rails · gates ride the edges</span>
        <div className="spacer" />
        <Toggle on={showGates} set={setShowGates} label="Gates" />
        <Toggle on={showEditions} set={setShowEditions} label="Editions" />
        <Toggle on={showExec} set={setShowExec} label="Execution layer" />
      </div>
      <Legend />
      <div className="scroll pad" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {manifest.rails.map((rail) => (
          <RailRow key={rail.id} rail={rail} git={git} edgeGates={edgeGates}
            gateById={gateById} showGates={showGates} showEditions={showEditions}
            dispatchTargets={dispatchTargets}
            onOpen={(p) => nav(`/edit?path=${encodeURIComponent(p)}`)} />
        ))}
        {showExec && <ExecStrip layer={(manifest as any).execution_layer} onOpen={(p) => nav(`/edit?path=${encodeURIComponent(p)}`)} />}
      </div>
    </>
  );
}

function RailRow({ rail, git, edgeGates, gateById, showGates, showEditions, dispatchTargets, onOpen }: {
  rail: Rail; git: Record<string, string>; edgeGates: Record<string, string[]>;
  gateById: Record<string, Gate>; showGates: boolean; showEditions: boolean;
  dispatchTargets: { label: string; color: string }[]; onOpen: (p: string) => void;
}) {
  const hue = `var(--rail-${rail.id})`;
  const isOrch = rail.kind === "orchestrator";
  return (
    <div style={{ display: "grid", gridTemplateColumns: "132px 1fr", gap: 12, alignItems: "stretch" }}>
      <div style={{ borderLeft: `3px solid ${hue}`, paddingLeft: 10, display: "flex", flexDirection: "column", justifyContent: "center" }}>
        <div className="title" style={{ color: hue }}>{rail.title}</div>
        <div className="faint" style={{ fontSize: 11 }}>{rail.nodes.length} steps</div>
        {rail.router?.skill && (
          <button className="btn ghost sm" style={{ marginTop: 4, justifyContent: "flex-start", padding: "2px 4px" }} onClick={() => onOpen(rail.router!.skill)}>SKILL.md</button>
        )}
      </div>
      <div className="scroll" style={{ overflowX: "auto", overflowY: "hidden", paddingBottom: 6 }}>
        <div style={{ display: "flex", alignItems: "stretch", gap: 0, minWidth: "min-content" }}>
          {rail.nodes.map((node, i) => {
            const isLast = i === rail.nodes.length - 1;
            // Gates validate a step's OUTPUT -> render on its OUTGOING edge (after
            // the card). Consistent with the editor's "gates guarding this step".
            const gates = showGates ? (edgeGates[node.id] || []).map((id) => gateById[id]).filter(Boolean) : [];
            return (
              <div key={node.id} style={{ display: "flex", alignItems: "stretch" }}>
                <NodeCard node={node} hue={hue} dirty={!!git[node.path]} showEditions={showEditions} onOpen={onOpen} />
                {(!isLast || gates.length > 0) && <EdgeGates gates={gates} hasNext={!isLast} />}
              </div>
            );
          })}
          {isOrch && (
            <div style={{ display: "flex", alignItems: "center", paddingLeft: 10 }}>
              <ArcFan source="orchestrator" targets={dispatchTargets} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function NodeCard({ node, hue, dirty, showEditions, onOpen }: {
  node: RailNode; hue: string; dirty: boolean; showEditions: boolean; onOpen: (p: string) => void;
}) {
  const idNum = node.id.split(".")[1];
  return (
    <div className="card" style={{ width: 152, minWidth: 152, cursor: "pointer", display: "flex", flexDirection: "column" }}
      onClick={() => onOpen(node.path)} title={node.path}>
      <div style={{ padding: "7px 9px", display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
        <div className="row" style={{ gap: 6 }}>
          <span className="mono" style={{ color: hue, fontSize: 11, fontWeight: 600 }}>{idNum}</span>
          {dirty && <span title="modified" style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--dirty)" }} />}
          <span className="spacer" />
          {node.edition_compressed && <span className="faint" style={{ fontSize: 9.5 }} title="has compressed edition">⊟</span>}
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.3 }}>{node.title}</div>
        {node.variants?.length ? <div className="faint" style={{ fontSize: 10 }}>+{node.variants.length} variant</div> : null}
      </div>
      {showEditions && node.edition_compressed && (
        <button className="btn ghost sm" style={{ borderTop: "1px solid var(--border)", borderRadius: 0, justifyContent: "flex-start", color: "var(--text-faint)" }}
          onClick={(e) => { e.stopPropagation(); onOpen(node.edition_compressed!); }}>compressed →</button>
      )}
      {showEditions && node.variants?.map((v) => (
        <button key={v.id} className="btn ghost sm" style={{ borderTop: "1px solid var(--border)", borderRadius: 0, justifyContent: "flex-start", color: "var(--text-faint)" }}
          onClick={(e) => { e.stopPropagation(); onOpen(v.path); }}>{v.title} →</button>
      ))}
    </div>
  );
}

function EdgeGates({ gates, hasNext }: { gates: Gate[]; hasNext: boolean }) {
  const nav = useNavigate();
  const sorted = [...gates].sort((a, b) => SEV_ORDER.indexOf(a.severity) - SEV_ORDER.indexOf(b.severity));
  return (
    <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", gap: 3, padding: "0 8px", minWidth: gates.length ? 96 : 34 }}>
      {hasNext && (
        <div style={{ height: 1, background: "var(--border-strong)", position: "relative" }}>
          <span style={{ position: "absolute", right: -2, top: -7, color: "var(--border-strong)" }}>▸</span>
        </div>
      )}
      {sorted.map((g) => (
        <span key={g.id} className={`chip ${g.severity}`} title={g.note || g.title}
          onClick={(e) => { e.stopPropagation(); nav(`/gates?gate=${g.id}`); }}>
          <span className="d" />{g.title.replace(/^(Music|Image) /, "")}
        </span>
      ))}
    </div>
  );
}

function ExecStrip({ layer, onOpen }: { layer: any[]; onOpen: (p: string) => void }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "132px 1fr", gap: 12 }}>
      <div style={{ borderLeft: "3px solid var(--border-strong)", paddingLeft: 10, display: "flex", alignItems: "center" }}>
        <div className="title faint">Execution layer</div>
      </div>
      <div className="row wrap">
        {(layer || []).map((x) => (
          <div key={x.id} className="card row" style={{ padding: "6px 10px", gap: 8 }}>
            <span className="mono" style={{ fontSize: 11 }}>{x.id}</span>
            <button className="btn ghost sm" onClick={() => onOpen(x.skill)}>SKILL</button>
            {x.execution && <button className="btn ghost sm" onClick={() => onOpen(x.execution)}>EXEC</button>}
          </div>
        ))}
      </div>
    </div>
  );
}

function Toggle({ on, set, label }: { on: boolean; set: (b: boolean) => void; label: string }) {
  return (
    <button className={`btn sm ${on ? "primary" : ""}`} onClick={() => set(!on)}>{label}</button>
  );
}

function Legend() {
  return (
    <div className="row wrap" style={{ padding: "8px 16px", borderBottom: "1px solid var(--border)", gap: 16, alignItems: "center" }}>
      <HypsometricKey variant="legend" />
      <span className="faint" style={{ fontSize: 11, maxWidth: 360, lineHeight: 1.4 }}>
        Severity as altitude of light — canon at the pale zenith, the hard-fail Andon at the horizon. Gates ride each step's outgoing edge; the orchestrator freezes once and fans to every rail.
      </span>
    </div>
  );
}
