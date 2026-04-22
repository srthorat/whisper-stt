#!/usr/bin/env bash
# ── Whisper STT server status & monitoring ───────────────────────────
# Usage: ./scripts/status.sh [--watch]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$PROJECT_DIR/run"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$PROJECT_DIR/logs/server.log"
PORT="${PORT:-8000}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_status() {
    echo -e "${BOLD}═══ Whisper STT Server Status ═══${NC}"
    echo ""

    # ── 1. Process status ────────────────────────────────────────────
    echo -e "${BOLD}[Process]${NC}"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            UPTIME=$(ps -o etime= -p "$PID" 2>/dev/null | xargs)
            RSS=$(ps -o rss= -p "$PID" 2>/dev/null | xargs)
            RSS_MB=$((${RSS:-0} / 1024))
            CPU=$(ps -o %cpu= -p "$PID" 2>/dev/null | xargs)
            echo -e "  Status:    ${GREEN}RUNNING${NC}"
            echo "  PID:       $PID"
            echo "  Uptime:    $UPTIME"
            echo "  RAM:       ${RSS_MB} MB"
            echo "  CPU:       ${CPU}%"
        else
            echo -e "  Status:    ${RED}DEAD${NC} (stale PID file: $PID)"
        fi
    else
        # Check if something is on the port anyway
        PORT_PID=$(lsof -ti :"$PORT" 2>/dev/null | head -1 || true)
        if [ -n "$PORT_PID" ]; then
            echo -e "  Status:    ${YELLOW}RUNNING (no PID file)${NC}"
            echo "  PID:       $PORT_PID (detected via port $PORT)"
        else
            echo -e "  Status:    ${RED}STOPPED${NC}"
        fi
    fi
    echo ""

    # ── 2. Health check ──────────────────────────────────────────────
    echo -e "${BOLD}[Health]${NC}"
    HEALTH=$(curl -sf "http://127.0.0.1:$PORT/healthz" 2>/dev/null || true)
    if [ -n "$HEALTH" ]; then
        STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "?")
        WORKERS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('workers_active','?'))" 2>/dev/null || echo "?")
        MODEL=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded','?'))" 2>/dev/null || echo "?")
        CALLS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active_calls','?'))" 2>/dev/null || echo "?")
        
        if [ "$STATUS" = "ok" ]; then
            echo -e "  Endpoint:  ${GREEN}HEALTHY${NC}"
        else
            echo -e "  Endpoint:  ${YELLOW}$STATUS${NC}"
        fi
        echo "  Workers:   $WORKERS"
        echo "  Model:     $MODEL"
        echo "  Calls:     $CALLS"
    else
        echo -e "  Endpoint:  ${RED}UNREACHABLE${NC} (http://127.0.0.1:$PORT/healthz)"
    fi
    echo ""

    # ── 3. GPU status ────────────────────────────────────────────────
    echo -e "${BOLD}[GPU]${NC}"
    if command -v nvidia-smi &>/dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,persistence_mode,pstate --format=csv,noheader,nounits 2>/dev/null || echo "")
        if [ -n "$GPU_INFO" ]; then
            IFS=',' read -r NAME TEMP UTIL MEM_USED MEM_TOTAL POWER POWER_LIM PERSIST PSTATE <<< "$GPU_INFO"
            echo "  GPU:       $(echo $NAME | xargs)"
            echo "  Temp:      $(echo $TEMP | xargs)°C"
            echo "  Util:      $(echo $UTIL | xargs)%"
            echo "  VRAM:      $(echo $MEM_USED | xargs) / $(echo $MEM_TOTAL | xargs) MiB"
            echo "  Power:     $(echo $POWER | xargs) / $(echo $POWER_LIM | xargs) W"
            echo "  Persist:   $(echo $PERSIST | xargs)"
            echo "  PerfState: $(echo $PSTATE | xargs)"
        fi
    else
        echo "  nvidia-smi not available"
    fi
    echo ""

    # ── 4. System resources ──────────────────────────────────────────
    echo -e "${BOLD}[System]${NC}"
    MEM_INFO=$(free -m | awk 'NR==2{printf "  RAM:       %s / %s MB (%.0f%% used)\n", $3, $2, $3/$2*100}')
    echo "$MEM_INFO"
    LOAD=$(cat /proc/loadavg | awk '{printf "  Load:      %s %s %s (1/5/15 min)\n", $1, $2, $3}')
    echo "$LOAD"
    FD_USED=$(cat /proc/sys/fs/file-nr | awk '{printf "  Open FDs:  %s / %s\n", $1, $3}')
    echo "$FD_USED"
    echo ""

    # ── 5. Prometheus metrics (summary) ──────────────────────────────
    echo -e "${BOLD}[Metrics]${NC}"
    METRICS=$(curl -sf "http://127.0.0.1:$PORT/metrics" 2>/dev/null || true)
    if [ -n "$METRICS" ]; then
        QUEUE=$(echo "$METRICS" | grep '^stt_gpu_queue_size ' | awk '{print $2}' || echo "0")
        FINALS=$(echo "$METRICS" | grep 'stt_transcript_events_total{event_type="final"}' | awk '{print $2}' || echo "0")
        INTERIMS=$(echo "$METRICS" | grep 'stt_transcript_events_total{event_type="interim"}' | awk '{print $2}' || echo "0")
        ERRORS=$(echo "$METRICS" | awk '/^stt_errors_total\{/{sum+=$2}END{print int(sum+0)}')
        DROPPED=$(echo "$METRICS" | awk '/^stt_dropped_jobs_total\{/{sum+=$2}END{print int(sum+0)}')
        BYTES=$(echo "$METRICS" | grep '^stt_audio_bytes_ingested_total ' | awk '{print $2}' || echo "0")
        
        # Inference latency
        INF_COUNT=$(echo "$METRICS" | grep '^stt_inference_seconds_count ' | awk '{print $2}' || echo "0")
        INF_SUM=$(echo "$METRICS" | grep '^stt_inference_seconds_sum ' | awk '{print $2}' || echo "0")
        AVG_INF="0"
        if [ -n "$INF_COUNT" ] && [ "$INF_COUNT" != "0" ] && [ "$INF_COUNT" != "0.0" ]; then
            AVG_INF=$(python3 -c "c=float('${INF_COUNT}'); s=float('${INF_SUM}'); print(f'{s/c:.3f}' if c>0 else '0')" 2>/dev/null || echo "?")
        fi

        echo "  Queue:        ${QUEUE:-0}"
        echo "  Finals:       ${FINALS:-0}"
        echo "  Interims:     ${INTERIMS:-0}"
        echo "  Errors:       ${ERRORS:-0}"
        echo "  Dropped:      ${DROPPED:-0}"
        echo "  Inferences:   ${INF_COUNT:-0} (avg ${AVG_INF}s)"
        if [ -n "$BYTES" ] && [ "$BYTES" != "0" ] && [ "$BYTES" != "0.0" ]; then
            BYTES_MB=$(python3 -c "print(f'{float(${BYTES})/1048576:.1f}')" 2>/dev/null || echo "?")
            echo "  Audio:        ${BYTES_MB} MB ingested"
        fi
    else
        echo "  (metrics endpoint unreachable)"
    fi
    echo ""

    # ── 6. Recent log lines ──────────────────────────────────────────
    echo -e "${BOLD}[Recent Logs]${NC}"
    if [ -f "$LOG_FILE" ]; then
        tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
    else
        echo "  (no log file)"
    fi
    echo ""
}

# ── Watch mode: refresh every 5 seconds ─────────────────────────────
if [[ "${1:-}" == "--watch" || "${1:-}" == "-w" ]]; then
    while true; do
        clear
        print_status
        echo -e "${CYAN}Refreshing every 5s... (Ctrl+C to exit)${NC}"
        sleep 5
    done
else
    print_status
fi
