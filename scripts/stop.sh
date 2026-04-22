#!/usr/bin/env bash
# ── Stop the Whisper STT server ──────────────────────────────────────
# Usage: ./scripts/stop.sh [--force]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$PROJECT_DIR/run"
PID_FILE="$RUN_DIR/server.pid"
PORT="${PORT:-8000}"
FORCE="${1:-}"

stop_pid() {
    local pid=$1
    local signal=${2:-TERM}
    
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Process $pid not running."
        return 1
    fi

    echo -n "Sending SIG$signal to PID $pid..."
    kill -"$signal" "$pid" 2>/dev/null || true

    # Wait up to 10 seconds for graceful shutdown
    for i in $(seq 1 10); do
        sleep 1
        echo -n "."
        if ! kill -0 "$pid" 2>/dev/null; then
            echo " stopped."
            return 0
        fi
    done

    # Graceful shutdown timed out — force kill
    echo " still running, sending SIGKILL..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "  Killed."
        return 0
    fi
    echo "  WARNING: Process $pid could not be killed."
    return 1
}

echo "═══ Stopping Whisper STT Server ═══"

# ── Stop via PID file ────────────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if [[ "$FORCE" == "--force" || "$FORCE" == "-f" ]]; then
        echo "Force stopping PID $PID..."
        kill -9 "$PID" 2>/dev/null || true
        sleep 1
        echo "Done."
    else
        stop_pid "$PID" "TERM"
    fi
    rm -f "$PID_FILE"
else
    echo "No PID file found at $PID_FILE"
fi

# ── Clean up any orphaned processes on the port ──────────────────────
ORPHANS=$(lsof -ti :"$PORT" 2>/dev/null || true)
if [ -n "$ORPHANS" ]; then
    echo "Cleaning up orphaned processes on port $PORT: $ORPHANS"
    echo "$ORPHANS" | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# ── Verify ───────────────────────────────────────────────────────────
if lsof -ti :"$PORT" &>/dev/null; then
    echo "WARNING: Port $PORT is still in use!"
    exit 1
else
    echo ""
    echo "Server stopped. Port $PORT is free."
fi
