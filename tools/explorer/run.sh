#!/usr/bin/env bash
# Lofn Prompt Explorer — one-command launch (macOS / Linux).
#   ./tools/explorer/run.sh            # or:  bash tools/explorer/run.sh
# Creates the venv + installs deps + builds the web UI on first run, then serves
# the whole thing from one local process at http://127.0.0.1:8765.
# Pass --build to force a fresh web build after changing the frontend.
# Override the bootstrap interpreter with:  PYTHON=/path/to/python ./run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="$SCRIPT_DIR/server/.venv"
PY="$VENV/bin/python"
URL="http://127.0.0.1:8765"

# Resolve a system Python for venv creation (prefer python3; override with $PYTHON).
BOOT_PY="${PYTHON:-python3}"
if ! command -v "$BOOT_PY" >/dev/null 2>&1; then BOOT_PY="python"; fi
if ! command -v "$BOOT_PY" >/dev/null 2>&1; then
  echo "[explorer] error: no python3/python on PATH. Install Python 3.11+." >&2
  exit 1
fi

if [ ! -x "$PY" ]; then
  echo "[explorer] creating venv + installing backend deps..."
  "$BOOT_PY" -m venv "$VENV"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r "$SCRIPT_DIR/server/requirements.txt"
fi

# Build the web UI on first run, or whenever --build is passed.
BUILD=0
for arg in "$@"; do
  if [ "$arg" = "--build" ]; then BUILD=1; fi
done
if [ ! -f "$SCRIPT_DIR/web/dist/index.html" ] || [ "$BUILD" -eq 1 ]; then
  echo "[explorer] building web UI..."
  (
    cd "$SCRIPT_DIR/web"
    [ -d node_modules ] || npm install --no-fund --no-audit
    npm run build
  )
fi

# Open the browser once the server has had a moment to bind (non-fatal if headless).
OPENER=""
if command -v open >/dev/null 2>&1; then OPENER="open"
elif command -v xdg-open >/dev/null 2>&1; then OPENER="xdg-open"; fi
if [ -n "$OPENER" ]; then ( sleep 1; "$OPENER" "$URL" >/dev/null 2>&1 || true ) & fi

echo "[explorer] serving $URL  (Ctrl+C to stop)"
cd "$ROOT"
export PYTHONIOENCODING="utf-8"
exec "$PY" -m tools.explorer.server
