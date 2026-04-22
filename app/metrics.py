from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

QUEUE_SIZE = Gauge("stt_gpu_queue_size", "Current GPU inference queue size")
ACTIVE_SESSIONS = Gauge("stt_active_sessions", "Number of active websocket sessions")
INFERENCE_SECONDS = Histogram(
    "stt_inference_seconds",
    "Whisper inference latency in seconds",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0),
)
EMIT_SECONDS = Histogram(
    "stt_emit_seconds",
    "Websocket emit latency in seconds",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
)
TRANSCRIPT_EVENTS = Counter(
    "stt_transcript_events_total",
    "Transcript events emitted",
    ["event_type"],
)
DROPPED_JOBS = Counter("stt_dropped_jobs_total", "Dropped inference jobs", ["reason"])
ERROR_EVENTS = Counter("stt_errors_total", "Server error count", ["kind"])
BYTES_INGESTED = Counter("stt_audio_bytes_ingested_total", "Incoming audio bytes")


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
