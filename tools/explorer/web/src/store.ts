import { create } from "zustand";
import { api } from "./api";
import type { GitStatus, Manifest, VerifyResult } from "./types";

// Per-rail segmentation hue (dimension -> visual variable, the data-viz seat's
// token). One hue per rail, used identically on the Atlas and anywhere a rail
// is named. CSS mirrors these in :root for chips that can't read JS.
export const RAIL_HUE: Record<string, string> = {
  orchestrator: "var(--rail-orchestrator)",
  music: "var(--rail-music)",
  image: "var(--rail-image)",
  video: "var(--rail-video)",
  story: "var(--rail-story)",
};

interface AppState {
  loading: boolean;
  error: string | null;
  repoRoot: string;
  manifest: Manifest | null;
  verify: VerifyResult | null;
  git: GitStatus;
  load: () => Promise<void>;
  refreshGit: () => Promise<void>;
  gitOf: (path?: string) => string;
}

export const useStore = create<AppState>((set, get) => ({
  loading: true,
  error: null,
  repoRoot: "",
  manifest: null,
  verify: null,
  git: {},
  load: async () => {
    set({ loading: true, error: null });
    try {
      const [m, git] = await Promise.all([api.manifest(), api.gitStatus()]);
      set({
        loading: false,
        repoRoot: m.repo_root,
        manifest: m.manifest,
        verify: m.verify,
        git,
      });
    } catch (e: any) {
      set({ loading: false, error: e.message || String(e) });
    }
  },
  refreshGit: async () => {
    try {
      set({ git: await api.gitStatus() });
    } catch { /* keep previous */ }
  },
  gitOf: (path?: string) => (path ? get().git[path] || "clean" : "clean"),
}));
