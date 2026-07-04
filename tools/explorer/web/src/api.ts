import type {
  FileView, GatesData, GitStatus, LibraryItem, LibraryList,
  ManifestResponse, MetaCheck, SyncStatus, ValidatorResult, LibrarySpec,
} from "./types";

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let body: any = {};
    try { body = await res.json(); } catch { /* ignore */ }
    const err = new Error(body.error || body.detail || `HTTP ${res.status}`) as any;
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return res.json() as Promise<T>;
}

const q = (s: string) => encodeURIComponent(s);

export const api = {
  manifest: () => fetch("/api/manifest").then(j<ManifestResponse>),
  gitStatus: () => fetch("/api/git/status").then(j<GitStatus>),

  readFile: (path: string) => fetch(`/api/file?path=${q(path)}`).then(j<FileView>),
  writeFile: (path: string, text: string, base_sha: string | null) =>
    fetch("/api/file", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, text, base_sha }),
    }).then(j<{ view: FileView; git: string }>),

  diff: (path: string) => fetch(`/api/git/diff?path=${q(path)}`).then(j<{ path: string; diff: string }>),
  revert: (path: string) =>
    fetch("/api/git/revert", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }).then(j<{ reverted: boolean; reason?: string; view?: FileView }>),

  gates: () => fetch("/api/gates").then(j<GatesData>),
  metacheck: () => fetch("/api/gates/metacheck", { method: "POST" }).then(j<MetaCheck>),

  validators: () => fetch("/api/validators").then(j<any[]>),
  runValidator: (id: string, args: string[]) =>
    fetch(`/api/validators/${q(id)}/run`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ args }),
    }).then(j<ValidatorResult>),

  libraries: () => fetch("/api/libraries").then(j<LibrarySpec[]>),
  library: (id: string, query?: string) =>
    fetch(`/api/libraries/${q(id)}${query ? `?q=${q(query)}` : ""}`).then(j<LibraryList>),
  libraryItem: (id: string, path: string) =>
    fetch(`/api/libraries/${q(id)}/item?path=${q(path)}`).then(j<LibraryItem>),
  writeLibraryItem: (id: string, path: string, text: string, base_sha: string | null) =>
    fetch(`/api/libraries/${q(id)}/item`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, text, base_sha }),
    }).then(j<{ view: FileView; git: string }>),
  librarySync: (id: string) => fetch(`/api/libraries/${q(id)}/sync`).then(j<SyncStatus>),
  libraryRegen: (id: string) =>
    fetch(`/api/libraries/${q(id)}/regen`, { method: "POST" }).then(
      j<{ returncode: number; stdout: string; stderr: string; sync: SyncStatus }>,
    ),
};
