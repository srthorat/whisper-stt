#!/usr/bin/env python3
"""
Load test: concurrent WebSocket connections streaming real or synthetic
16 kHz PCM audio.  Captures server metrics every 10 seconds and writes
a timestamped report to results/load_test_<timestamp>.txt

Usage:
  source ~/dev/whisper-env/bin/activate
  # With real audio (recommended):
  python scripts/load_test.py --wav-file /path/to/audio.wav -c 100 -d 60
  # With synthetic tone:
  python scripts/load_test.py -c 100 -d 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import struct
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import websockets

# ── defaults ────────────────────────────────────────────────────────────
DEFAULT_URL = "ws://127.0.0.1:8000"
METRICS_URL = "http://127.0.0.1:8000/metrics"
HEALTH_URL = "http://127.0.0.1:8000/healthz"
SAMPLE_RATE = 16_000
FRAME_MS = 40
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)    # 640
FRAME_BYTES = FRAME_SAMPLES * 2                        # 1280
FRAMES_PER_SEC = 1000 // FRAME_MS                      # 25


# ── synthetic audio ─────────────────────────────────────────────────────

def generate_tone_frame(frame_idx: int, freq: float = 440.0) -> bytes:
    """Generate a single frame of PCM 16-bit sine tone."""
    buf = bytearray(FRAME_BYTES)
    for i in range(FRAME_SAMPLES):
        t = (frame_idx * FRAME_SAMPLES + i) / SAMPLE_RATE
        val = int(3000 * math.sin(2 * math.pi * freq * t))
        struct.pack_into("<h", buf, i * 2, max(-32768, min(32767, val)))
    return bytes(buf)


# Pre-generate 1 second of frames to avoid per-frame CPU overhead
PREBUILT_FRAMES = [generate_tone_frame(i) for i in range(FRAMES_PER_SEC)]


# ── WAV file loader ─────────────────────────────────────────────────────

def load_wav_frames(wav_path: str, frame_ms: int = FRAME_MS) -> tuple[list[bytes], int]:
    """Load a WAV file and split it into fixed-size PCM frames.
    
    Returns (list_of_frame_bytes, num_channels).
    Each frame contains `frame_ms` ms of interleaved PCM16LE samples.
    """
    with wave.open(wav_path, 'rb') as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sr != SAMPLE_RATE:
        raise ValueError(f"WAV sample rate {sr} != expected {SAMPLE_RATE}. Resample first.")
    if width != 2:
        raise ValueError(f"WAV sample width {width} != 2 (16-bit PCM required).")

    samples_per_frame = int(sr * frame_ms / 1000) * channels
    bytes_per_frame = samples_per_frame * 2  # 16-bit
    
    frames = []
    offset = 0
    while offset + bytes_per_frame <= len(raw):
        frames.append(raw[offset : offset + bytes_per_frame])
        offset += bytes_per_frame
    
    return frames, channels


# ── per-connection stats ────────────────────────────────────────────────

@dataclass
class ClientStats:
    client_id: int = 0
    connected_at: float = 0.0
    disconnected_at: float = 0.0
    frames_sent: int = 0
    interims_received: int = 0
    finals_received: int = 0
    errors: int = 0
    status: str = "pending"     # pending | connected | streaming | done | error


# ── single client coroutine ─────────────────────────────────────────────

async def run_client(
    client_id: int,
    url: str,
    duration_sec: int,
    stats: ClientStats,
    stop_event: asyncio.Event,
    wav_frames: list[bytes] | None = None,
    wav_channels: int = 1,
) -> None:
    stats.client_id = client_id
    stats.status = "pending"
    ws_url = f"{url.rstrip('/')}/ws/call/load-{client_id}"

    try:
        async with websockets.connect(ws_url, max_size=None, open_timeout=10) as ws:
            stats.connected_at = time.time()
            stats.status = "connected"

            # ── send start ──
            channels = wav_channels if wav_frames else 1
            start_evt: dict = {
                "type": "start",
                "sample_rate": SAMPLE_RATE,
                "channels": channels,
            }
            if channels == 2:
                start_evt["left_speaker"] = "agent"
                start_evt["right_speaker"] = "customer"
            else:
                start_evt["speaker"] = f"client-{client_id}"
            await ws.send(json.dumps(start_evt))

            # ── receiver task ──
            async def _recv():
                try:
                    async for msg in ws:
                        if isinstance(msg, str):
                            try:
                                evt = json.loads(msg)
                            except json.JSONDecodeError:
                                continue
                            etype = evt.get("type", "")
                            if etype == "interim":
                                stats.interims_received += 1
                            elif etype == "final":
                                stats.finals_received += 1
                            elif etype == "error":
                                stats.errors += 1
                except Exception:
                    pass

            recv_task = asyncio.create_task(_recv())

            # ── stream audio ──
            stats.status = "streaming"
            if wav_frames:
                # Stream WAV frames (loop if duration > file length)
                total_frames_needed = FRAMES_PER_SEC * duration_sec
                for frame_idx in range(total_frames_needed):
                    if stop_event.is_set():
                        break
                    frame = wav_frames[frame_idx % len(wav_frames)]
                    await ws.send(frame)
                    stats.frames_sent += 1
                    await asyncio.sleep(FRAME_MS / 1000.0)
            else:
                # Synthetic tone fallback
                total_frames = FRAMES_PER_SEC * duration_sec
                for frame_idx in range(total_frames):
                    if stop_event.is_set():
                        break
                    frame = PREBUILT_FRAMES[frame_idx % FRAMES_PER_SEC]
                    await ws.send(frame)
                    stats.frames_sent += 1
                    await asyncio.sleep(FRAME_MS / 1000.0)

            # ── stop ──
            await ws.send(json.dumps({"type": "flush"}))
            await asyncio.sleep(0.5)
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.sleep(0.3)

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

            stats.status = "done"

    except Exception as exc:
        stats.status = f"error: {exc}"
        stats.errors += 1
    finally:
        stats.disconnected_at = time.time()


# ── metrics collector ────────────────────────────────────────────────────

async def collect_metrics_loop(
    interval_sec: int,
    duration_sec: int,
    all_stats: list[ClientStats],
    snapshots: list[dict],
    stop_event: asyncio.Event,
) -> None:
    """Collect server + client metrics every `interval_sec`."""
    import urllib.request

    start_time = time.time()

    while not stop_event.is_set():
        await asyncio.sleep(interval_sec)
        elapsed = time.time() - start_time

        # ── client-side aggregates ──
        connected = sum(1 for s in all_stats if s.status in ("connected", "streaming"))
        done = sum(1 for s in all_stats if s.status == "done")
        errored = sum(1 for s in all_stats if s.status.startswith("error"))
        total_frames = sum(s.frames_sent for s in all_stats)
        total_interims = sum(s.interims_received for s in all_stats)
        total_finals = sum(s.finals_received for s in all_stats)
        total_errors = sum(s.errors for s in all_stats)

        # ── server-side metrics ──
        server_metrics = {}
        try:
            with urllib.request.urlopen(METRICS_URL, timeout=2) as resp:
                body = resp.read().decode()
            for line in body.splitlines():
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    server_metrics[parts[0]] = parts[1]
        except Exception:
            pass

        # ── server health ──
        health = {}
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                health = json.loads(resp.read().decode())
        except Exception:
            pass

        snap = {
            "elapsed_sec": round(elapsed, 1),
            "ts": time.strftime("%H:%M:%S"),
            # client side
            "active_connections": connected,
            "completed": done,
            "errored": errored,
            "frames_sent": total_frames,
            "interims": total_interims,
            "finals": total_finals,
            "client_errors": total_errors,
            # server side
            "server_active_calls": health.get("active_calls", "?"),
            "server_status": health.get("status", "?"),
            "gpu_queue": server_metrics.get("stt_gpu_queue_size", "0"),
            "inference_count": server_metrics.get("stt_inference_seconds_count", "0"),
            "inference_sum_sec": server_metrics.get("stt_inference_seconds_sum", "0"),
            "audio_bytes": server_metrics.get("stt_audio_bytes_ingested_total", "0"),
            "transcript_finals": server_metrics.get(
                'stt_transcript_events_total{event_type="final"}', "0"
            ),
            "transcript_interims": server_metrics.get(
                'stt_transcript_events_total{event_type="interim"}', "0"
            ),
        }
        snapshots.append(snap)

        # live print
        print(
            f"[{snap['ts']}] {snap['elapsed_sec']:>5.0f}s | "
            f"conn={snap['active_connections']:>3} done={snap['completed']:>3} err={snap['errored']:>3} | "
            f"frames={snap['frames_sent']:>7} interims={snap['interims']:>5} finals={snap['finals']:>5} | "
            f"srv_calls={snap['server_active_calls']} gpu_q={snap['gpu_queue']} "
            f"infer={snap['inference_count']}"
        )


# ── GPU snapshot ─────────────────────────────────────────────────────────

def gpu_snapshot() -> str:
    """Capture nvidia-smi summary."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return out
    except Exception as exc:
        return f"(unavailable: {exc})"


# ── report writer ────────────────────────────────────────────────────────

def write_report(
    report_path: Path,
    args: argparse.Namespace,
    all_stats: list[ClientStats],
    snapshots: list[dict],
    wall_time: float,
) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  WHISPER STT — LOAD TEST REPORT")
    lines.append("=" * 80)
    lines.append(f"  Date:            {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Connections:     {args.connections}")
    lines.append(f"  Duration:        {args.duration}s per client")
    lines.append(f"  Ramp:            {args.ramp} clients/batch, {args.ramp_delay}s between")
    lines.append(f"  Wall time:       {wall_time:.1f}s")
    lines.append(f"  GPU:             {gpu_snapshot()}")
    lines.append("")

    # ── summary ──
    total_frames = sum(s.frames_sent for s in all_stats)
    total_interims = sum(s.interims_received for s in all_stats)
    total_finals = sum(s.finals_received for s in all_stats)
    total_errors = sum(s.errors for s in all_stats)
    done_count = sum(1 for s in all_stats if s.status == "done")
    err_count = sum(1 for s in all_stats if s.status.startswith("error"))

    total_audio_sec = total_frames * FRAME_MS / 1000.0
    lines.append("── Summary ──────────────────────────────────────────────")
    lines.append(f"  Clients completed:   {done_count} / {len(all_stats)}")
    lines.append(f"  Clients errored:     {err_count}")
    lines.append(f"  Total frames sent:   {total_frames:,}")
    lines.append(f"  Total audio:         {total_audio_sec:,.1f}s ({total_audio_sec/60:.1f} min)")
    lines.append(f"  Total interims:      {total_interims:,}")
    lines.append(f"  Total finals:        {total_finals:,}")
    lines.append(f"  Total errors:        {total_errors:,}")
    lines.append(f"  Avg interims/client: {total_interims / max(done_count, 1):.1f}")
    lines.append(f"  Avg finals/client:   {total_finals / max(done_count, 1):.1f}")
    lines.append("")

    # ── connection durations ──
    durations = [
        s.disconnected_at - s.connected_at
        for s in all_stats
        if s.connected_at > 0 and s.disconnected_at > 0
    ]
    if durations:
        lines.append("── Connection Durations ─────────────────────────────────")
        lines.append(f"  Min:    {min(durations):.1f}s")
        lines.append(f"  Max:    {max(durations):.1f}s")
        lines.append(f"  Mean:   {sum(durations) / len(durations):.1f}s")
        lines.append("")

    # ── timeline (10s snapshots) ──
    lines.append("── Timeline (every 10s) ────────────────────────────────")
    header = (
        f"{'Time':>6} {'Elapsed':>7} │ {'Active':>6} {'Done':>5} {'Err':>4} │ "
        f"{'Frames':>8} {'Interims':>8} {'Finals':>7} │ "
        f"{'SrvCalls':>8} {'GPU_Q':>5} {'Infer#':>7}"
    )
    lines.append(header)
    lines.append("─" * len(header))
    for s in snapshots:
        lines.append(
            f"{s['ts']:>6} {s['elapsed_sec']:>6.0f}s │ "
            f"{s['active_connections']:>6} {s['completed']:>5} {s['errored']:>4} │ "
            f"{s['frames_sent']:>8} {s['interims']:>8} {s['finals']:>7} │ "
            f"{str(s['server_active_calls']):>8} {s['gpu_queue']:>5} {s['inference_count']:>7}"
        )
    lines.append("")

    # ── per-client breakdown ──
    lines.append("── Per-Client Breakdown ─────────────────────────────────")
    lines.append(
        f"{'ID':>4} {'Status':>10} {'Frames':>7} {'Interims':>8} {'Finals':>7} {'Errors':>6} {'Duration':>8}"
    )
    lines.append("─" * 60)
    for s in sorted(all_stats, key=lambda x: x.client_id):
        dur = (
            f"{s.disconnected_at - s.connected_at:.1f}s"
            if s.connected_at > 0 and s.disconnected_at > 0
            else "n/a"
        )
        status = s.status if len(s.status) <= 10 else s.status[:10]
        lines.append(
            f"{s.client_id:>4} {status:>10} {s.frames_sent:>7} "
            f"{s.interims_received:>8} {s.finals_received:>7} {s.errors:>6} {dur:>8}"
        )
    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    print(f"\n{'=' * 60}")
    print(report_text)
    print(f"\nReport saved to: {report_path}")


# ── main orchestrator ────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    all_stats: list[ClientStats] = [ClientStats() for _ in range(args.connections)]
    snapshots: list[dict] = []
    stop_event = asyncio.Event()

    # Load WAV file if provided
    wav_frames: list[bytes] | None = None
    wav_channels = 1
    if args.wav_file:
        print(f"Loading WAV: {args.wav_file}")
        wav_frames, wav_channels = load_wav_frames(args.wav_file)
        wav_duration = len(wav_frames) * FRAME_MS / 1000.0
        print(f"  Frames: {len(wav_frames)}, Channels: {wav_channels}, Duration: {wav_duration:.1f}s")
        if args.duration > wav_duration:
            print(f"  Note: audio will loop ({wav_duration:.0f}s file × {args.duration}s requested)")
    else:
        print(f"Using synthetic tone (no --wav-file specified)")

    print(f"\n═══ Load Test ═══")
    print(f"  Target:      {args.url}")
    print(f"  Connections: {args.connections}")
    print(f"  Duration:    {args.duration}s streaming per client")
    print(f"  Audio:       {'WAV: ' + args.wav_file if args.wav_file else 'synthetic tone'}")
    print(f"  Channels:    {wav_channels}")
    print(f"  Ramp:        {args.ramp} clients every {args.ramp_delay}s")
    print(f"  GPU:         {gpu_snapshot()}")
    print()

    # Start metrics collector
    metrics_task = asyncio.create_task(
        collect_metrics_loop(
            interval_sec=args.interval,
            duration_sec=args.duration,
            all_stats=all_stats,
            snapshots=snapshots,
            stop_event=stop_event,
        )
    )

    # Ramp up clients
    wall_start = time.time()
    client_tasks: list[asyncio.Task] = []

    for batch_start in range(0, args.connections, args.ramp):
        batch_end = min(batch_start + args.ramp, args.connections)
        for cid in range(batch_start, batch_end):
            task = asyncio.create_task(
                run_client(
                    cid, args.url, args.duration, all_stats[cid], stop_event,
                    wav_frames=wav_frames, wav_channels=wav_channels,
                ),
                name=f"client-{cid}",
            )
            client_tasks.append(task)
        
        launched = batch_end
        print(f"[ramp] Launched {launched}/{args.connections} clients")
        
        if batch_end < args.connections:
            await asyncio.sleep(args.ramp_delay)

    # Wait for all clients to finish
    await asyncio.gather(*client_tasks, return_exceptions=True)
    wall_time = time.time() - wall_start

    # Let one more metrics snapshot happen
    await asyncio.sleep(args.interval + 1)
    stop_event.set()
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass

    # Write report
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.output_dir) / f"load_test_{ts}.txt"
    write_report(report_path, args, all_stats, snapshots, wall_time)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whisper STT load test")
    p.add_argument("--url", default=DEFAULT_URL, help="WebSocket base URL")
    p.add_argument("--connections", "-c", type=int, default=100, help="Number of concurrent clients")
    p.add_argument("--duration", "-d", type=int, default=60, help="Streaming duration per client (seconds)")
    p.add_argument("--interval", "-i", type=int, default=10, help="Metrics snapshot interval (seconds)")
    p.add_argument("--ramp", type=int, default=10, help="Clients to launch per ramp batch")
    p.add_argument("--ramp-delay", type=float, default=1.0, help="Delay between ramp batches (seconds)")
    p.add_argument("--wav-file", type=str, default=None, help="WAV file to stream (16kHz PCM16LE). Omit for synthetic tone.")
    p.add_argument("--output-dir", default="results", help="Directory for report files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
