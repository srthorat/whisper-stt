#!/usr/bin/env bash
# ── Start the Whisper STT server ─────────────────────────────────────
# Usage: ./scripts/start.sh [--foreground]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/dev/whisper-env}"
RUN_DIR="$PROJECT_DIR/run"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$LOG_DIR/server.log"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

# Max log size before rotation (10 MB)
LOG_MAX_BYTES=${LOG_MAX_BYTES:-10485760}
# Number of rotated logs to keep
LOG_BACKUP_COUNT=${LOG_BACKUP_COUNT:-5}

# ── Pre-flight checks ───────────────────────────────────────────────
if [ ! -f "$VENV/bin/python" ]; then
    echo "ERROR: virtualenv not found at $VENV"
    echo "  Create it: python3 -m venv $VENV && $VENV/bin/pip install -r $PROJECT_DIR/requirements.txt"
    exit 1
fi

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Server already running (PID $OLD_PID)"
        echo "  Use: ./scripts/stop.sh"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Check port availability
if lsof -ti :"$PORT" &>/dev/null; then
    echo "ERROR: Port $PORT is already in use"
    lsof -ti :"$PORT" | head -3
    exit 1
fi

# ── Ensure run & log directories ────────────────────────────────────
mkdir -p "$RUN_DIR" "$LOG_DIR"

# ── Log rotation ────────────────────────────────────────────────────
rotate_log() {
    if [ ! -f "$LOG_FILE" ]; then
        return
    fi
    local size
    size=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
    if [ "$size" -ge "$LOG_MAX_BYTES" ]; then
        echo "[log-rotate] Rotating $LOG_FILE ($((size / 1048576)) MB)"
        # Shift existing rotated logs
        for i in $(seq $((LOG_BACKUP_COUNT - 1)) -1 1); do
            [ -f "${LOG_FILE}.$i" ] && mv "${LOG_FILE}.$i" "${LOG_FILE}.$((i + 1))"
        done
        mv "$LOG_FILE" "${LOG_FILE}.1"
        # Compress the rotated log in background
        gzip -f "${LOG_FILE}.1" 2>/dev/null &
        # Remove oldest if over backup count
        rm -f "${LOG_FILE}.$((LOG_BACKUP_COUNT + 1))"*
    fi
}
rotate_log

# ── Export CUDA env (belt-and-suspenders with .env) ──────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CT2_CUDA_ALLOW_TF32="${CT2_CUDA_ALLOW_TF32:-1}"

echo "═══ Whisper STT Server ═══"
echo "  Project:  $PROJECT_DIR"
echo "  Venv:     $VENV"
echo "  Bind:     $HOST:$PORT"
echo "  PID file: $PID_FILE"
echo "  Log:      $LOG_FILE"

# ── Foreground mode ─────────────────────────────────────────────────
if [[ "${1:-}" == "--foreground" || "${1:-}" == "-f" ]]; then
    echo "  Mode:     foreground (Ctrl+C to stop)"
    echo ""
    cd "$PROJECT_DIR"
    exec "$VENV/bin/python" -m uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info
fi

# ── Background (daemon) mode ────────────────────────────────────────
echo "  Mode:     background (daemon)"
echo ""

cd "$PROJECT_DIR"
nohup "$VENV/bin/python" -m uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info \
    >> "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Wait for health check
echo -n "Starting (PID $SERVER_PID)..."
for i in $(seq 1 30); do
    sleep 1
    echo -n "."
    if curl -sf "http://127.0.0.1:$PORT/healthz" > /dev/null 2>&1; then
        echo " OK"
        echo ""
        curl -s "http://127.0.0.1:$PORT/healthz" | python3 -m json.tool
        echo ""
        echo "Server started successfully."
        echo "  Stop:    ./scripts/stop.sh"
        echo "  Status:  ./scripts/status.sh"
        echo "  Logs:    tail -f $LOG_FILE"
        exit 0
    fi
    # Check if process died
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo " FAILED"
        echo ""
        echo "Server process exited unexpectedly. Last 20 lines of log:"
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
done

echo " TIMEOUT"
echo ""
echo "Server did not become healthy within 30 seconds."
echo "Check logs: tail -f $LOG_FILE"
exit 1
