"""FastAPI app for the Lofn Prompt Explorer.

Dumb-backend doctrine (plan §3.4): file CRUD behind a path jail, YAML round-trip,
subprocessing the repo's own validators, git status/diff, and index regen. No
threshold or check is reimplemented here.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings, load_settings
from .files import ConflictError, FileService, PathJailError
from .gates import GatesService
from .gitio import GitService
from .libraries import LibraryService
from .manifest import load_manifest
from .validators import ValidatorRunner


class WriteBody(BaseModel):
    path: str
    text: str
    base_sha: str | None = None


class PathBody(BaseModel):
    path: str


class RunBody(BaseModel):
    args: list[str] = []
    timeout: int = 120


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    root = settings.repo_root
    manifest = load_manifest(settings.manifest_path)

    files = FileService(root, manifest.write_allowlist)
    git = GitService(root)
    runner = ValidatorRunner(root, settings.python_exe, manifest)
    gates = GatesService(root, files, runner, manifest.gates_file)
    libraries = LibraryService(root, manifest, files)

    verify = _run_verify(settings)

    app = FastAPI(title="Lofn Prompt Explorer", version="1.0")

    # -- studio (v2) additive routes -------------------------------------
    # The Creative Studio registers /api/studio/* onto this same app. Kept
    # import-guarded so a partial studio build never breaks the v1 explorer.
    try:
        from .studio.api import register_studio

        register_studio(app, root)
    except Exception:  # pragma: no cover - studio optional at boot
        pass

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- exception mapping ------------------------------------------------
    @app.exception_handler(PathJailError)
    async def _jail(_req, exc):  # noqa
        return JSONResponse(status_code=403, content={"error": str(exc)})

    @app.exception_handler(ConflictError)
    async def _conflict(_req, exc):  # noqa
        return JSONResponse(status_code=409, content={"error": str(exc), "current_sha": exc.current_sha})

    @app.exception_handler(FileNotFoundError)
    async def _nf(_req, exc):  # noqa
        return JSONResponse(status_code=404, content={"error": f"not found: {exc}"})

    # -- meta -------------------------------------------------------------
    @app.get("/api/health")
    def health():
        return {"ok": True, "repo_root": str(root)}

    @app.get("/api/manifest")
    def get_manifest():
        return {"repo_root": str(root), "manifest": manifest.public_dict(), "verify": verify}

    # -- files ------------------------------------------------------------
    @app.get("/api/file")
    def get_file(path: str = Query(...)):
        return files.read(path).__dict__

    @app.put("/api/file")
    def put_file(body: WriteBody):
        view = files.write_text(body.path, body.text, base_sha=body.base_sha)
        return {"view": view.__dict__, "git": git.status_of(view.path)}

    # -- git --------------------------------------------------------------
    @app.get("/api/git/status")
    def git_status():
        return git.status_map()

    @app.get("/api/git/diff")
    def git_diff(path: str = Query(...)):
        return {"path": path, "diff": git.diff(path)}

    @app.post("/api/git/revert")
    def git_revert(body: PathBody):
        res = git.revert(body.path)
        if res.get("reverted"):
            try:
                res["view"] = files.read(body.path).__dict__
            except FileNotFoundError:
                res["view"] = None
        return res

    # -- gates ------------------------------------------------------------
    @app.get("/api/gates")
    def get_gates():
        return gates.read()

    @app.post("/api/gates/metacheck")
    def gates_metacheck():
        return gates.meta_check()

    # -- validators -------------------------------------------------------
    @app.get("/api/validators")
    def list_validators():
        return manifest.validators

    @app.post("/api/validators/{vid}/run")
    def run_validator(vid: str, body: RunBody):
        try:
            return runner.run_by_id(vid, body.args, timeout=body.timeout)
        except PermissionError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # -- libraries --------------------------------------------------------
    @app.get("/api/libraries")
    def list_collections():
        return libraries.collections()

    @app.get("/api/libraries/{lib_id}")
    def list_items(lib_id: str, q: str | None = None):
        try:
            return libraries.list_items(lib_id, q)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no library '{lib_id}'")

    @app.get("/api/libraries/{lib_id}/item")
    def get_item(lib_id: str, path: str = Query(...)):
        try:
            return libraries.get_item(lib_id, path)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no library '{lib_id}'")

    @app.put("/api/libraries/{lib_id}/item")
    def put_item(lib_id: str, body: WriteBody):
        try:
            view = libraries.put_item(lib_id, body.path, body.text, body.base_sha)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no library '{lib_id}'")
        return {"view": view, "git": git.status_of(body.path)}

    @app.get("/api/libraries/{lib_id}/sync")
    def library_sync(lib_id: str):
        try:
            return libraries.sync_status(lib_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no library '{lib_id}'")

    @app.post("/api/libraries/{lib_id}/regen")
    def library_regen(lib_id: str):
        lib = manifest.library(lib_id)
        if not lib or not lib.get("regen"):
            raise HTTPException(status_code=404, detail=f"no regen for '{lib_id}'")
        script = root / "scripts" / "regen_library.py"
        cp = subprocess.run(
            [settings.python_exe, str(script), lib_id, "--root", str(root)],
            cwd=str(root), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120,
        )
        return {
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "sync": libraries.sync_status(lib_id),
        }

    # -- static (production build) ---------------------------------------
    dist = settings.web_dist
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(dist / "index.html")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            candidate = dist / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")
    else:
        @app.get("/")
        def dev_notice():
            return {
                "message": "Backend up. Web build not found — run the Vite dev server "
                "(cd tools/explorer/web && npm run dev) or `npm run build` for production.",
                "api": "/api/health",
            }

    return app


def _run_verify(settings: Settings) -> dict:
    """Run scripts/verify_pipeline_map.py --json at startup (manifest honesty, gate 6)."""
    script = settings.repo_root / "scripts" / "verify_pipeline_map.py"
    try:
        cp = subprocess.run(
            [settings.python_exe, str(script), "--json",
             "--map", str(settings.manifest_path), "--root", str(settings.repo_root)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        import json
        data = json.loads(cp.stdout or "{}")
        data["ok"] = data.get("ok", cp.returncode == 0)
        return data
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": f"verify failed to run: {e}"}


app = create_app()
