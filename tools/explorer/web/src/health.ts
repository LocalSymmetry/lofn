// The instrument's health vector — the five signals the Context panel named, each
// already a backend endpoint. Kept in its own store so the always-present state
// spine (App) and the Bridge (home) read ONE source of truth. Cheap signals
// (manifest, working-tree) live in the main store and render instantly; the
// on-demand ones (drift, sync, last-run) run scripts, so they carry a freshness
// stamp and stay "unknown" until checked — a stale green is a lie, not a fact.
import { create } from "zustand";
import { api } from "./api";
import { studioApi } from "./studio/api";
import type { MetaCheckRow } from "./types";
import type { RunSummary } from "./studio/types";

export type CheckState = "unknown" | "checking" | "done";
export interface SyncDrift { id: string; title: string; missing: number; stale: number; }

interface HealthState {
  drift: { state: CheckState; count: number; rows: MetaCheckRow[]; ranAt: number | null };
  sync: { state: CheckState; drifted: SyncDrift[]; checked: number; ranAt: number | null };
  lastRun: { state: CheckState; summary: RunSummary | null; ranAt: number | null };
  checkDrift: () => Promise<void>;
  checkSync: (libs: { id: string; title: string }[]) => Promise<void>;
  checkLastRun: () => Promise<void>;
}

export const useHealth = create<HealthState>((set) => ({
  drift: { state: "unknown", count: 0, rows: [], ranAt: null },
  sync: { state: "unknown", drifted: [], checked: 0, ranAt: null },
  lastRun: { state: "unknown", summary: null, ranAt: null },

  checkDrift: async () => {
    set((s) => ({ drift: { ...s.drift, state: "checking" } }));
    try {
      const mc = await api.metacheck();
      set({ drift: { state: "done", count: mc.count ?? mc.rows.length, rows: mc.rows, ranAt: Date.now() } });
    } catch { set((s) => ({ drift: { ...s.drift, state: "unknown" } })); }
  },

  checkSync: async (libs) => {
    set((s) => ({ sync: { ...s.sync, state: "checking" } }));
    const drifted: SyncDrift[] = [];
    let checked = 0;
    await Promise.all(libs.map(async (l) => {
      try {
        const r = await api.librarySync(l.id);
        if (!r.applicable) return;
        checked += 1;
        const missing = r.missing_from_index?.length || 0;
        const stale = r.stale_in_index?.length || 0;
        const countDrift =
          (r.index_count != null && r.index_count !== r.canonical_count) ||
          (r.monolith_count != null && r.monolith_count !== r.canonical_count);
        if (missing || stale || countDrift) drifted.push({ id: l.id, title: l.title, missing, stale });
      } catch { /* a non-applicable / erroring collection is simply skipped */ }
    }));
    set({ sync: { state: "done", drifted, checked, ranAt: Date.now() } });
  },

  checkLastRun: async () => {
    set((s) => ({ lastRun: { ...s.lastRun, state: "checking" } }));
    try {
      const runs = await studioApi.runs();
      const latest = runs && runs.length
        ? runs.slice().sort((a, b) => (a.created < b.created ? 1 : -1))[0]
        : null;
      set({ lastRun: { state: "done", summary: latest, ranAt: Date.now() } });
    } catch {
      // the studio backend may be absent in a plain v1 launch — treat as "no runs".
      set((s) => ({ lastRun: { ...s.lastRun, state: "done", summary: null } }));
    }
  },
}));

/** "3s ago" / "2m ago" — freshness, so a green is never faith-based. */
export function ago(ms: number | null): string {
  if (ms == null) return "not checked";
  const d = Math.max(0, Date.now() - ms);
  if (d < 1500) return "just now";
  if (d < 60_000) return `${Math.round(d / 1000)}s ago`;
  if (d < 3_600_000) return `${Math.round(d / 60_000)}m ago`;
  return `${Math.round(d / 3_600_000)}h ago`;
}
