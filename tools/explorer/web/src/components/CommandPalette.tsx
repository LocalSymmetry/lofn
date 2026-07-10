import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useStore } from "../store";

interface Entry { label: string; hint: string; run: () => void; }

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const nav = useNavigate();
  const manifest = useStore((s) => s.manifest);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const openIt = () => { setOpen(true); setQ(""); setSel(0); };
    window.addEventListener("open-palette", openIt);
    return () => window.removeEventListener("open-palette", openIt);
  }, []);
  useEffect(() => { if (open) setTimeout(() => inputRef.current?.focus(), 10); }, [open]);

  const entries = useMemo<Entry[]>(() => {
    if (!manifest) return [];
    const e: Entry[] = [];
    const go = (to: string) => () => { nav(to); setOpen(false); };
    e.push({ label: "Bridge", hint: "home · vitals", run: go("/") });
    e.push({ label: "Atlas", hint: "view", run: go("/atlas") });
    e.push({ label: "Gate Center", hint: "view", run: go("/gates") });
    e.push({ label: "Gate Center · Coverage", hint: "rail × severity", run: go("/gates?tab=coverage") });
    e.push({ label: "Gate Center · Drift", hint: "reconcile", run: go("/gates?tab=drift") });
    e.push({ label: "Working set", hint: "g w · diff-first", run: go("/changes") });
    e.push({ label: "Library", hint: "view", run: go("/library") });
    e.push({ label: "Studio · Flow Lab", hint: "g s f", run: go("/studio/flow") });
    e.push({ label: "Studio · Run Bench", hint: "g s r", run: go("/studio/run") });
    e.push({ label: "Studio · Compare", hint: "g s c", run: go("/studio/compare") });
    for (const rail of manifest.rails) {
      for (const [k, v] of Object.entries(rail.router || {})) {
        e.push({ label: `${rail.title} · ${k}`, hint: v, run: go(`/edit?path=${encodeURIComponent(v)}`) });
      }
      for (const n of rail.nodes) {
        e.push({ label: `${rail.title} · ${n.id} ${n.title}`, hint: n.path, run: go(`/edit?path=${encodeURIComponent(n.path)}`) });
        for (const varr of n.variants || []) e.push({ label: `${rail.title} · ${varr.title}`, hint: varr.path, run: go(`/edit?path=${encodeURIComponent(varr.path)}`) });
      }
    }
    for (const x of manifest.execution_layer || []) {
      e.push({ label: `.claude · ${x.id}`, hint: x.skill, run: go(`/edit?path=${encodeURIComponent(x.skill)}`) });
      if (x.execution) e.push({ label: `.claude · ${x.id} EXECUTION`, hint: x.execution, run: go(`/edit?path=${encodeURIComponent(x.execution)}`) });
    }
    for (const l of manifest.libraries) e.push({ label: `Library · ${l.title}`, hint: l.kind, run: go(`/library?c=${l.id}`) });
    return e;
  }, [manifest, nav]);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return entries.slice(0, 60);
    return entries.filter((x) => (x.label + " " + x.hint).toLowerCase().includes(s)).slice(0, 60);
  }, [q, entries]);

  useEffect(() => { setSel(0); }, [q]);
  if (!open) return null;

  return (
    <div onClick={() => setOpen(false)}
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", display: "flex", justifyContent: "center", alignItems: "flex-start", paddingTop: "12vh", zIndex: 100 }}>
      <div onClick={(e) => e.stopPropagation()} className="card" style={{ width: 620, maxWidth: "90vw", boxShadow: "var(--shadow)", overflow: "hidden" }}>
        <input ref={inputRef} className="input" placeholder="Jump to a step, router, gate, library…"
          value={q} onChange={(e) => setQ(e.target.value)}
          style={{ border: "none", borderBottom: "1px solid var(--border)", borderRadius: 0, padding: "12px 14px", fontSize: 14 }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(s + 1, filtered.length - 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
            else if (e.key === "Enter") { filtered[sel]?.run(); }
            else if (e.key === "Escape") setOpen(false);
          }} />
        <div className="scroll" style={{ maxHeight: 380 }}>
          {filtered.length === 0 && <div className="empty">no matches</div>}
          {filtered.map((x, i) => (
            <div key={i} onMouseEnter={() => setSel(i)} onClick={() => x.run()}
              style={{ display: "flex", gap: 10, padding: "7px 14px", cursor: "pointer", background: i === sel ? "var(--surface-3)" : "transparent", alignItems: "baseline" }}>
              <span>{x.label}</span>
              <span className="faint mono spacer" style={{ fontSize: 10.5, textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{x.hint}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
