"""Entry point: `python -m tools.explorer.server` (or via run.ps1)."""
from __future__ import annotations

import uvicorn

from .config import load_settings
from .main import create_app


def run() -> None:
    settings = load_settings()
    app = create_app(settings)
    print(f"Lofn Prompt Explorer — repo {settings.repo_root}")
    print(f"  http://{settings.host}:{settings.port}   (API at /api/health)")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    run()
