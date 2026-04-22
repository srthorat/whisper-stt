#!/usr/bin/env bash
# ── Production OS tuning for GPU STT server ──────────────────────────
# Run once with: sudo bash scripts/os-tune.sh
# Persists via sysctl.d and systemd; safe to re-run (idempotent).
set -euo pipefail

echo "═══ Whisper STT — Production OS Tuning ═══"
echo ""

# ── 1. GPU: persistence mode + max clocks ────────────────────────────
echo "[GPU] Enabling persistence mode..."
nvidia-smi -pm 1 2>/dev/null || echo "  (skipped — nvidia-smi not available)"

# Lock GPU clocks to max for consistent inference latency
# L4 max clocks: graphics=2040MHz, memory=6251MHz
echo "[GPU] Setting max GPU clock speeds..."
nvidia-smi -ac 6251,2040 2>/dev/null || echo "  (skipped — clock lock not supported)"

# Disable ECC if enabled (saves ~5% memory, reduces latency)
# NOTE: Requires reboot to take effect; uncomment if desired
# nvidia-smi -e 0 2>/dev/null || true

# ── 2. Kernel: memory and scheduler ─────────────────────────────────
echo "[KERNEL] Applying sysctl tuning..."

cat > /etc/sysctl.d/90-whisper-stt.conf << 'SYSCTL'
# ── Memory ──
# Reduce swap aggressiveness (GPU workload = latency sensitive)
vm.swappiness = 10
# Reduce dirty page flush delays
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5
# Disable OOM-kill overcommit surprises
vm.overcommit_memory = 0

# ── Network (WebSocket connections) ──
# Allow TIME_WAIT socket reuse
net.ipv4.tcp_tw_reuse = 1
# Faster connection teardown
net.ipv4.tcp_fin_timeout = 15
# Larger listen backlog for burst WebSocket connections
net.core.somaxconn = 8192
net.ipv4.tcp_max_syn_backlog = 8192
net.core.netdev_max_backlog = 8192
# TCP keepalive for long-lived WebSocket sessions
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 6
# Increase local port range for outbound connections
net.ipv4.ip_local_port_range = 1024 65535

# ── Scheduler ──
# Disable autogroup (single-purpose server)
kernel.sched_autogroup_enabled = 0
SYSCTL

sysctl --system > /dev/null 2>&1
echo "  Applied /etc/sysctl.d/90-whisper-stt.conf"

# ── 3. File descriptor limits ────────────────────────────────────────
echo "[LIMITS] Setting file descriptor limits..."

grep -q "whisper-stt" /etc/security/limits.d/90-whisper-stt.conf 2>/dev/null || \
cat > /etc/security/limits.d/90-whisper-stt.conf << 'LIMITS'
# whisper-stt: raise fd limits for WebSocket connections
*    soft    nofile    65536
*    hard    nofile    65536
*    soft    nproc     32768
*    hard    nproc     32768
LIMITS
echo "  Applied /etc/security/limits.d/90-whisper-stt.conf"

# ── 4. Transparent Huge Pages (disable for latency) ──────────────────
echo "[THP] Disabling Transparent Huge Pages..."
if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo never > /sys/kernel/mm/transparent_hugepage/enabled
    echo never > /sys/kernel/mm/transparent_hugepage/defrag
    echo "  THP disabled"
else
    echo "  (not available)"
fi

# ── 5. CPU governor: performance mode ────────────────────────────────
echo "[CPU] Setting performance governor..."
if command -v cpupower &> /dev/null; then
    cpupower frequency-set -g performance 2>/dev/null || echo "  (cpupower failed)"
elif [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$cpu" 2>/dev/null || true
    done
    echo "  Set performance governor"
else
    echo "  (no cpufreq — likely fixed-clock instance)"
fi

# ── 6. IRQ affinity: keep GPU IRQ off CPU 0 ──────────────────────────
echo "[IRQ] Tuning IRQ affinity..."
# Pin GPU interrupts to CPUs 1-7, leave CPU 0 for network/userspace
GPU_IRQ=$(cat /proc/interrupts | grep -i nvidia | awk '{print $1}' | tr -d ':' | head -1)
if [ -n "$GPU_IRQ" ] && [ -f "/proc/irq/$GPU_IRQ/smp_affinity" ]; then
    echo fe > "/proc/irq/$GPU_IRQ/smp_affinity" 2>/dev/null || echo "  (irq affinity failed)"
    echo "  GPU IRQ $GPU_IRQ pinned to CPUs 1-7"
else
    echo "  (skipped — GPU IRQ not found)"
fi

# ── 7. NUMA: verify single-socket layout ────────────────────────────
echo "[NUMA] Checking topology..."
if command -v numactl &> /dev/null; then
    NODES=$(numactl --hardware | grep "available:" | awk '{print $2}')
    echo "  $NODES NUMA node(s)"
else
    echo "  (numactl not installed)"
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "═══ Tuning complete ═══"
echo ""
echo "  Swappiness:       $(sysctl -n vm.swappiness)"
echo "  Somaxconn:        $(sysctl -n net.core.somaxconn)"
echo "  FD limit (soft):  $(ulimit -Sn)"
echo "  GPU persistence:  $(nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep 'Persistence Mode' | awk '{print $NF}' || echo 'N/A')"
echo "  GPU perf state:   $(nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep 'Performance State' | awk '{print $NF}' || echo 'N/A')"
echo ""
echo "NOTE: File descriptor limits require re-login to take effect."
echo "NOTE: THP settings do not persist across reboot — add to rc.local or systemd unit."
