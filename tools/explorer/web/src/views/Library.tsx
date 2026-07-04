import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { Editor } from "../components/Editor";
import { useStore } from "../store";
import type { Card, LibraryItemRef, LibraryList, LibrarySpec, SyncStatus } from "../types";

export function Library() {
  const { manifest } = useStore();
  const [specs, setSpecs] = useState<LibrarySpec[]>([]);
  const [params, setParams] = useSearchParams();
  const cid = params.get("c") || "personalities";
  useEffect(() => { api.libraries().then(setSpecs); }, []);
  const spec = specs.find((s) => s.id === cid) || manifest?.libraries.find((l: any) => l.id === cid);

  return (
    <>
      <div className="hd">
        <h1>Library</h1>
        <div className="row wrap" style={{ gap: 4, marginLeft: 10 }}>
          {(specs.length ? specs : manifest?.libraries || []).map((s: any) => (
            <button key={s.id} className={`btn sm ${s.id === cid ? "primary" : "ghost"}`}
              onClick={() => setParams({ c: s.id })}>{s.title}</button>
          ))}
        </div>
      </div>
      {spec && <CollectionView key={cid} spec={spec as LibrarySpec} />}
    </>
  );
}

function CollectionView({ spec }: { spec: LibrarySpec }) {
  if (spec.kind === "yaml-collection") return <YamlCollection spec={spec} />;
  if (spec.kind === "csv") return <CsvCollection spec={spec} />;
  return <ListCollection spec={spec} />;
}

/* ---------------- YAML collections (personalities, panels) ---------------- */
function YamlCollection({ spec }: { spec: LibrarySpec }) {
  const [list, setList] = useState<LibraryList | null>(null);
  const [sync, setSync] = useState<SyncStatus | null>(null);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<string | null>(null);
  const [card, setCard] = useState<Card | null>(null);
  const [showVariants, setShowVariants] = useState(false);
  const [regenning, setRegenning] = useState(false);
  const refreshGit = useStore((s) => s.refreshGit);

  const loadList = () => api.library(spec.id, q || undefined).then(setList);
  useEffect(() => { loadList(); }, [spec.id, q]);
  useEffect(() => { api.librarySync(spec.id).then(setSync); }, [spec.id]);
  useEffect(() => {
    if (!sel) { setCard(null); return; }
    api.libraryItem(spec.id, sel).then((r) => setCard(r.card));
  }, [sel, spec.id]);

  const items = ((list?.items as LibraryItemRef[]) || []).filter((i) => showVariants || !i.is_variant);
  const drift = sync && sync.applicable &&
    ((sync.index_count != null && sync.index_count !== sync.canonical_count) ||
     (sync.monolith_count != null && sync.monolith_count !== sync.canonical_count) ||
     (sync.missing_from_index?.length || 0) > 0);

  const regen = async () => {
    setRegenning(true);
    try { const r = await api.libraryRegen(spec.id); setSync(r.sync); refreshGit(); }
    finally { setRegenning(false); }
  };

  return (
    <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
      {sync?.applicable && (
        <div className={`banner ${drift ? "warn" : ""}`} style={!drift ? { background: "var(--surface)", borderBottom: "1px solid var(--border)", color: "var(--text-dim)" } : {}}>
          <span className={`badge ${drift ? "warn" : "ok"}`}><span className="d" />{drift ? "derived drift" : "in sync"}</span>
          <span className="faint mono" style={{ fontSize: 11 }}>
            canonical {sync.canonical_count} · index {sync.index_count ?? "—"} · monolith {sync.monolith_count ?? "—"}
          </span>
          {drift ? <span className="faint" style={{ fontSize: 11.5 }}>derived files (monolith + indexes) are regen-only.</span> : null}
          <div className="spacer" />
          {sync.regen && <button className="btn sm primary" disabled={regenning} onClick={regen} title={sync.regen}>{regenning ? "regenerating…" : "Regenerate derived"}</button>}
        </div>
      )}
      <div style={{ flex: 1, minHeight: 0, display: "grid", gridTemplateColumns: "268px 1fr" }}>
        <div style={{ borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div style={{ padding: 8, borderBottom: "1px solid var(--border)" }}>
            <input className="input" placeholder={`search ${spec.title.toLowerCase()}…`} value={q} onChange={(e) => setQ(e.target.value)} />
            <div className="row" style={{ marginTop: 6, justifyContent: "space-between" }}>
              <span className="faint" style={{ fontSize: 11 }}>{list?.count ?? "…"} items{list?.variant_count ? ` · ${list.variant_count} variants` : ""}</span>
              {!!list?.variant_count && <label className="row faint" style={{ fontSize: 11, gap: 4, cursor: "pointer" }}>
                <input type="checkbox" checked={showVariants} onChange={(e) => setShowVariants(e.target.checked)} /> variants
              </label>}
            </div>
          </div>
          <div className="scroll">
            {items.map((it) => (
              <div key={it.file} onClick={() => setSel(it.file)}
                className="row" style={{ padding: "6px 10px", cursor: "pointer", gap: 6, background: sel === it.file ? "var(--surface-3)" : "transparent", borderBottom: "1px solid var(--surface-2)" }}>
                <span style={{ fontSize: 12.5 }}>{it.name}</span>
                {it.is_variant && <span className="badge" style={{ fontSize: 9.5 }}>variant</span>}
                <span className="spacer" />
                <GitDot path={it.file} />
              </div>
            ))}
            {items.length === 0 && <div className="empty">no matches</div>}
          </div>
        </div>
        {sel ? (
          <div style={{ display: "grid", gridTemplateRows: card ? "auto 1fr" : "1fr", minHeight: 0 }}>
            {card && <CardView card={card} />}
            <Editor key={sel} path={sel} />
          </div>
        ) : <div className="empty">select an item to edit</div>}
      </div>
    </div>
  );
}

function CardView({ card }: { card: Card }) {
  return (
    <div style={{ borderBottom: "1px solid var(--border)", background: "var(--surface)", padding: 12, maxHeight: 210, overflow: "auto" }}>
      <div className="row" style={{ marginBottom: 6 }}><b>{card.name}</b><span className="badge">{card.type}</span><span className="faint" style={{ fontSize: 10.5 }}>display-only · edit below</span></div>
      {card.type === "panel" ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 10 }}>
          {card.sections?.map((s) => (
            <div key={s.title}>
              <div className="nav-section" style={{ padding: "0 0 4px" }}>{s.title}</div>
              {s.items.slice(0, 8).map((it, i) => <div key={i} className="faint" style={{ fontSize: 11, lineHeight: 1.35 }}>{it}</div>)}
            </div>
          ))}
        </div>
      ) : (
        <>
          <div className="row wrap" style={{ gap: 5, marginBottom: 6 }}>
            {card.headings?.slice(0, 14).map((h, i) => <span key={i} className="badge" style={{ fontSize: 10 }}>{h}</span>)}
          </div>
          <div className="faint" style={{ fontSize: 11.5, whiteSpace: "pre-wrap" }}>{card.preview}</div>
        </>
      )}
    </div>
  );
}

/* ---------------- line / comma lists (aesthetics, genres, film_styles) ---- */
function ListCollection({ spec }: { spec: LibrarySpec }) {
  const [list, setList] = useState<LibraryList | null>(null);
  const [q, setQ] = useState("");
  const load = () => api.library(spec.id).then(setList);
  useEffect(() => { load(); }, [spec.id]);
  const terms = (list?.items as string[]) || [];
  const filtered = useMemo(() => q ? terms.filter((t) => t.toLowerCase().includes(q.toLowerCase())) : terms, [q, terms]);

  return (
    <div style={{ flex: 1, minHeight: 0, display: "grid", gridTemplateColumns: "1fr 300px" }}>
      {list?.path ? <Editor key={list.path} path={list.path} /> : <div className="empty row" style={{ justifyContent: "center" }}><span className="spin" /></div>}
      <div style={{ borderLeft: "1px solid var(--border)", display: "flex", flexDirection: "column", minHeight: 0, background: "var(--surface)" }}>
        <div style={{ padding: 8, borderBottom: "1px solid var(--border)" }}>
          <input className="input" placeholder="find term…" value={q} onChange={(e) => setQ(e.target.value)} />
          <div className="row" style={{ marginTop: 6, justifyContent: "space-between" }}>
            <span className="faint" style={{ fontSize: 11 }}>{terms.length} terms{spec.kind === "comma-list" ? " (comma-separated)" : ""}</span>
            <button className="btn sm ghost" onClick={load}>reload</button>
          </div>
        </div>
        <div className="scroll pad" style={{ display: "flex", flexWrap: "wrap", gap: 5, alignContent: "flex-start" }}>
          {filtered.slice(0, 800).map((t, i) => <span key={i} className="badge" style={{ fontSize: 11 }}>{t}</span>)}
          {filtered.length > 800 && <span className="faint">+{filtered.length - 800} more…</span>}
        </div>
      </div>
    </div>
  );
}

/* ---------------- CSV (frames) -------------------------------------------- */
function CsvCollection({ spec }: { spec: LibrarySpec }) {
  const [list, setList] = useState<LibraryList | null>(null);
  const [q, setQ] = useState("");
  const load = () => api.library(spec.id, q || undefined).then(setList);
  useEffect(() => { load(); }, [spec.id, q]);
  return (
    <div style={{ flex: 1, minHeight: 0, display: "grid", gridTemplateRows: "auto 1fr" }}>
      <div className="row" style={{ padding: 8, borderBottom: "1px solid var(--border)", background: "var(--surface)" }}>
        <input className="input mono" style={{ maxWidth: 280, fontSize: 12 }} placeholder="filter rows…" value={q} onChange={(e) => setQ(e.target.value)} />
        <span className="faint" style={{ fontSize: 11 }}>{list?.count ?? "…"} rows · {(list?.columns || []).join(" · ")}</span>
      </div>
      <div style={{ minHeight: 0, display: "grid", gridTemplateColumns: "1fr 1fr" }}>
        <div className="scroll">
          <table className="grid">
            <thead><tr>{(list?.columns || []).map((c) => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>
              {(list?.rows || []).slice(0, 1500).map((r, i) => (
                <tr key={i}>{r.map((c, j) => <td key={j} className={j === 0 ? "mono" : ""} style={j === 0 ? { color: "var(--text-dim)", fontSize: 11 } : { fontSize: 11.5 }}>{c}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
        {list?.path && <Editor key={list.path} path={list.path} />}
      </div>
    </div>
  );
}

function GitDot({ path }: { path: string }) {
  const dirty = useStore((s) => !!s.git[path]);
  if (!dirty) return null;
  return <span title="modified" style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--dirty)" }} />;
}
