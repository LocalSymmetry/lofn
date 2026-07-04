import Monaco from "@monaco-editor/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useStore } from "../store";
import type { FileView } from "../types";
import { DiffPanel } from "./DiffPanel";

function langOf(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase();
  return ({ md: "markdown", markdown: "markdown", yaml: "yaml", yml: "yaml",
    json: "json", py: "python", csv: "plaintext", txt: "plaintext" } as any)[ext || ""] || "plaintext";
}

export function Editor({ path, onLoaded }: { path: string; onLoaded?: (v: FileView) => void }) {
  const [view, setView] = useState<FileView | null>(null);
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [conflict, setConflict] = useState<string | null>(null);
  const [showDiff, setShowDiff] = useState(false);
  const [saving, setSaving] = useState(false);
  const refreshGit = useStore((s) => s.refreshGit);
  const git = useStore((s) => s.git[path] || "clean");
  const themedRef = useRef(false);

  const load = useCallback(async () => {
    setErr(null); setConflict(null);
    try {
      const v = await api.readFile(path);
      setView(v); setText(v.text); onLoaded?.(v);
    } catch (e: any) { setErr(e.message); }
  }, [path, onLoaded]);

  useEffect(() => { load(); }, [load]);

  const dirty = view != null && text !== view.text;

  const save = useCallback(async () => {
    if (!view || !view.writable || saving) return;
    setSaving(true); setErr(null); setConflict(null);
    try {
      const res = await api.writeFile(path, text, view.sha);
      setView(res.view); setText(res.view.text); onLoaded?.(res.view);
      refreshGit();
    } catch (e: any) {
      if (e.status === 409) setConflict(e.body?.current_sha || "changed on disk");
      else setErr(e.message);
    } finally { setSaving(false); }
  }, [view, text, path, saving, onLoaded, refreshGit]);

  // Ctrl/Cmd+S
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === "s" || e.key === "S") && (e.metaKey || e.ctrlKey)) { e.preventDefault(); save(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [save]);

  const revert = useCallback(async () => {
    if (!confirm(`Revert ${path} to HEAD? Local changes are discarded.`)) return;
    const r = await api.revert(path);
    if (r.reverted) { await load(); refreshGit(); }
    else alert(r.reason || "could not revert");
  }, [path, load, refreshGit]);

  if (err) return <div className="empty" style={{ color: "var(--sev-hard)" }}>{err}</div>;
  if (!view) return <div className="empty row" style={{ justifyContent: "center" }}><span className="spin" /></div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div className="row" style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)", background: "var(--surface)", gap: 8 }}>
        <button className="btn primary sm" disabled={!view.writable || !dirty || saving} onClick={save}>
          {saving ? "saving…" : "Save"} <span className="kbd">⌘S</span>
        </button>
        <button className="btn sm" disabled={!dirty} onClick={() => setText(view.text)}>Reset</button>
        <button className={`btn sm ${showDiff ? "primary" : ""}`} disabled={git === "clean"} onClick={() => setShowDiff((s) => !s)}>Diff</button>
        <button className="btn sm ghost" disabled={git === "clean"} onClick={revert}>Revert</button>
        <div className="spacer" />
        {dirty && <span className="badge dirty"><span className="d" />unsaved</span>}
        {git !== "clean" && <span className="badge dirty" title="git">{git}</span>}
        {view.is_yaml && !view.roundtrip && <span className="badge warn" title="Irregular YAML — structured view unavailable; raw editing is byte-safe.">raw-mode</span>}
        {!view.writable && <span className="badge">read-only</span>}
        <span className="faint mono" style={{ fontSize: 10.5 }}>{view.eol} · {view.size}b · {view.sha.slice(0, 8)}</span>
      </div>

      {conflict && (
        <div className="banner warn">
          <b>Changed on disk</b> since you opened it (concurrent edit / pipeline run).
          <button className="btn sm" onClick={load}>Reload disk version</button>
          <span className="faint mono">disk sha {conflict.slice(0, 8)}</span>
        </div>
      )}

      <div style={{ flex: 1, minHeight: 0, display: showDiff ? "grid" : "block", gridTemplateColumns: showDiff ? "1fr 1fr" : "1fr" }}>
        <Monaco
          language={langOf(path)}
          value={text}
          onChange={(v) => setText(v ?? "")}
          theme="lofn-dark"
          beforeMount={(monaco) => {
            if (themedRef.current) return;
            themedRef.current = true;
            monaco.editor.defineTheme("lofn-dark", {
              base: "vs-dark", inherit: true, rules: [],
              colors: { "editor.background": "#15171e", "editorGutter.background": "#15171e" },
            });
          }}
          options={{
            fontSize: 12.5, fontFamily: "ui-monospace, Cascadia Code, Consolas, monospace",
            minimap: { enabled: false }, wordWrap: "on", readOnly: !view.writable,
            scrollBeyondLastLine: false, renderWhitespace: "selection", tabSize: 2,
            smoothScrolling: true, padding: { top: 8 },
          }}
        />
        {showDiff && <DiffPanel path={path} />}
      </div>
    </div>
  );
}
