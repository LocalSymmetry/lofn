import { useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Editor } from "../components/Editor";
import { useStore } from "../store";
import type { FileView, Gate, RailNode } from "../types";

export function EditorView() {
  const [params] = useSearchParams();
  const path = params.get("path") || "";
  const { manifest } = useStore();
  const nav = useNavigate();
  const [view, setView] = useState<FileView | null>(null);

  const located = useMemo(() => {
    if (!manifest) return null;
    for (const rail of manifest.rails) {
      const idx = rail.nodes.findIndex((n) => n.path === path);
      if (idx >= 0) return { rail, node: rail.nodes[idx], idx };
      for (const n of rail.nodes) {
        const v = (n.variants || []).find((x) => x.path === path);
        if (v) return { rail, node: n, idx: rail.nodes.indexOf(n), variantTitle: v.title };
      }
    }
    return null;
  }, [manifest, path]);

  if (!path) return <div className="empty">No file selected. Use the Atlas or ⌘K.</div>;

  const edgeGates: Record<string, string[]> = (manifest as any)?.edge_gates || {};
  const gateById: Record<string, Gate> = {};
  manifest?.gates.forEach((g) => (gateById[g.id] = g));
  const stepGates = located ? (edgeGates[located.node.id] || []).map((id) => gateById[id]).filter(Boolean) : [];

  const prev = located && located.idx > 0 ? located.rail.nodes[located.idx - 1] : null;
  const next = located && located.idx < located.rail.nodes.length - 1 ? located.rail.nodes[located.idx + 1] : null;

  return (
    <>
      <div className="hd">
        {located ? (
          <>
            <span className="mono" style={{ color: `var(--rail-${located.rail.id})`, fontWeight: 600 }}>{located.node.id}</span>
            <h1>{(located as any).variantTitle || located.node.title}</h1>
            <span className="faint mono" style={{ fontSize: 11 }}>{located.rail.title}</span>
          </>
        ) : (
          <h1 className="mono" style={{ fontSize: 12.5 }}>{path}</h1>
        )}
        <div className="spacer" />
        {prev && <button className="btn sm" onClick={() => nav(`/edit?path=${encodeURIComponent(prev.path)}`)}>← {prev.id}</button>}
        {next && <button className="btn sm" onClick={() => nav(`/edit?path=${encodeURIComponent(next.path)}`)}>{next.id} →</button>}
      </div>
      <div style={{ flex: 1, minHeight: 0, display: "grid", gridTemplateColumns: located ? "1fr 316px" : "1fr" }}>
        <Editor key={path} path={path} onLoaded={setView} />
        {located && <ContextPanel node={located.node} rail={located.rail.id} gates={stepGates} view={view} slots={manifest!.creative_context_slots} />}
      </div>
    </>
  );
}

function ContextPanel({ node, rail, gates, view, slots }: {
  node: RailNode; rail: string; gates: Gate[]; view: FileView | null; slots: string[];
}) {
  const nav = useNavigate();
  const text = view?.text || "";
  const headings = useMemo(() => {
    const out: { level: number; text: string }[] = [];
    for (const line of text.split("\n")) {
      const m = /^(#{1,4})\s+(.*\S)\s*$/.exec(line);
      if (m) out.push({ level: m[1].length, text: m[2] });
    }
    return out;
  }, [text]);
  const slotPresence = useMemo(() => slots.map((s) => ({ s, present: text.includes(`{${s}}`) })), [slots, text]);
  const anySlots = slotPresence.some((x) => x.present);

  return (
    <div className="scroll" style={{ borderLeft: "1px solid var(--border)", background: "var(--surface)" }}>
      <Section title="Gates guarding this step">
        {gates.length === 0 ? <div className="faint" style={{ fontSize: 11.5 }}>none declared on this edge</div> :
          gates.map((g) => (
            <div key={g.id} className="card" style={{ marginBottom: 6, cursor: "pointer" }} onClick={() => nav(`/gates?gate=${g.id}`)}>
              <div className="card-bd" style={{ padding: 9 }}>
                <div className="row"><span className={`chip ${g.severity}`}><span className="d" />{g.severity}</span><b style={{ fontSize: 12 }}>{g.title}</b></div>
                {g.note && <div className="faint" style={{ fontSize: 11, marginTop: 4 }}>{g.note}</div>}
                {g.keys.length > 0 && <div className="mono" style={{ fontSize: 10, marginTop: 4, color: "var(--accent)" }}>{g.keys.join(" · ")}</div>}
              </div>
            </div>
          ))}
      </Section>

      {anySlots && (
        <Section title="Creative Context slots">
          <div className="row wrap" style={{ gap: 5 }}>
            {slotPresence.map((x) => (
              <span key={x.s} className={`badge ${x.present ? "ok" : ""}`} style={{ opacity: x.present ? 1 : 0.45 }}>
                <span className="d" />{x.s}
              </span>
            ))}
          </div>
        </Section>
      )}

      <Section title={`Outline · ${headings.length}`}>
        {headings.map((h, i) => (
          <div key={i} className="faint" style={{ fontSize: 11.5, paddingLeft: (h.level - 1) * 10, color: h.level <= 2 ? "var(--text-dim)" : "var(--text-faint)" }}>{h.text}</div>
        ))}
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--border)" }}>
      <div className="nav-section" style={{ padding: "0 0 8px" }}>{title}</div>
      {children}
    </div>
  );
}
