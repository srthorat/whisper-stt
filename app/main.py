"""FastAPI WebSocket server for streaming speech-to-text.

Audio arrives as PCM-16LE frames over WebSocket.  Each channel is handled
by a Session that accumulates audio during speech, periodically submits
for GPU inference, and emits confirmed/preview text via LocalAgreement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect

from .config import settings
from .metrics import ACTIVE_SESSIONS, BYTES_INGESTED, EMIT_SECONDS, ERROR_EVENTS, render_metrics
from .sessions import Session
from .transcriber import TranscriberService

# Enable DEBUG for hypothesis_buffer during development
logging.getLogger("app.hypothesis_buffer").setLevel(logging.DEBUG)
_hb_handler = logging.StreamHandler()
_hb_handler.setFormatter(logging.Formatter("%(name)s %(message)s"))
logging.getLogger("app.hypothesis_buffer").addHandler(_hb_handler)

app = FastAPI(title="single-gpu-whisper-stt")
transcriber = TranscriberService()
registry_lock = asyncio.Lock()


# -------------------------------------------------------------------- #
# Call context
# -------------------------------------------------------------------- #


@dataclass
class CallContext:
    call_id: str
    ws: WebSocket
    channels: int
    sample_rate: int
    started_at: float = field(default_factory=time.time)
    last_audio_at: float = field(default_factory=time.time)
    last_event_at: float = field(default_factory=time.time)
    sessions: list[Session] = field(default_factory=list)


active_calls: dict[str, CallContext] = {}
cleanup_task: asyncio.Task | None = None


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #


def _decode_pcm16le(payload: bytes, channels: int) -> tuple[np.ndarray, np.ndarray | None]:
    samples = np.frombuffer(payload, dtype=np.int16)
    if channels <= 0:
        return np.zeros(0, dtype=np.float32), None
    if samples.size % channels != 0:
        samples = samples[: samples.size - (samples.size % channels)]
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32), (
            np.zeros(0, dtype=np.float32) if channels == 2 else None
        )
    if channels == 1:
        return samples.astype(np.float32) / 32768.0, None
    stereo = samples.reshape(-1, channels).astype(np.float32) / 32768.0
    return stereo[:, 0], stereo[:, 1]


def _is_authorized(ws: WebSocket) -> bool:
    if not settings.api_key:
        return True
    provided = ws.headers.get("x-api-key") or ws.query_params.get("api_key")
    return bool(provided and provided == settings.api_key)


async def _safe_emit(ws: WebSocket, lock: asyncio.Lock, payload: dict[str, Any]) -> None:
    emit_start = time.perf_counter()
    async with lock:
        await asyncio.wait_for(
            ws.send_text(json.dumps(payload)),
            timeout=settings.ws_send_timeout_seconds,
        )
    EMIT_SECONDS.observe(time.perf_counter() - emit_start)


# -------------------------------------------------------------------- #
# Channel audio processing (streaming)
# -------------------------------------------------------------------- #


async def _process_channel_audio(channel: Session, audio: np.ndarray) -> None:
    """Process one audio chunk for a single channel.

    Speech state machine:
      silence → speech_start  →  begin new segment, maybe queue inference
      speech_continues        →  accumulate, queue inference when ready
      speech_end              →  final flush (last transcription + commit all)
      silence                 →  no-op
    """
    event = channel.process_audio(audio)

    if event == "speech_start":
        # New speech segment started — first chunk already in buffer.
        # If already enough audio (unlikely on first chunk), queue inference.
        if channel.should_infer():
            await transcriber.submit_streaming(channel)

    elif event == "speech_continues":
        if channel.should_infer():
            await transcriber.submit_streaming(channel)

    elif event == "speech_end":
        # End of speech — submit final-flush to commit remaining text
        await transcriber.submit_streaming_final(channel)


# -------------------------------------------------------------------- #
# Session cleanup
# -------------------------------------------------------------------- #


async def _close_and_cleanup(call_id: str, reason: str) -> None:
    async with registry_lock:
        ctx = active_calls.pop(call_id, None)
        ACTIVE_SESSIONS.set(len(active_calls))

    if not ctx:
        return

    for channel_session in ctx.sessions:
        await transcriber.flush_final(channel_session)

    try:
        await ctx.ws.send_text(
            json.dumps({"type": "stopped", "call_id": call_id, "reason": reason})
        )
    except Exception:
        pass
    try:
        await ctx.ws.close()
    except Exception:
        pass


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)
        now = time.time()
        stale_calls: list[tuple[str, str]] = []
        async with registry_lock:
            for call_id, ctx in active_calls.items():
                if now - ctx.last_audio_at >= settings.session_idle_timeout_seconds:
                    if now - ctx.last_event_at >= settings.session_idle_timeout_seconds:
                        stale_calls.append((call_id, "idle_timeout"))
                elif now - ctx.started_at >= settings.session_max_duration_seconds:
                    stale_calls.append((call_id, "max_duration"))

        for call_id, reason in stale_calls:
            await _close_and_cleanup(call_id, reason)


# -------------------------------------------------------------------- #
# FastAPI lifecycle
# -------------------------------------------------------------------- #


@app.on_event("startup")
async def startup() -> None:
    global cleanup_task
    await transcriber.start()
    cleanup_task = asyncio.create_task(_cleanup_loop(), name="session-cleanup")


@app.on_event("shutdown")
async def shutdown() -> None:
    global cleanup_task
    if cleanup_task:
        cleanup_task.cancel()
        cleanup_task = None

    async with registry_lock:
        call_ids = list(active_calls.keys())

    for call_id in call_ids:
        await _close_and_cleanup(call_id, "shutdown")

    await transcriber.stop()


# -------------------------------------------------------------------- #
# HTTP endpoints
# -------------------------------------------------------------------- #


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    workers_running = transcriber.worker_tasks and any(
        not t.done() for t in transcriber.worker_tasks
    )
    return {
        "status": "ok" if workers_running else "degraded",
        "workers_active": workers_running,
        "model_loaded": transcriber.model is not None,
        "active_calls": len(active_calls),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return await healthz()


@app.get("/metrics")
async def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


# -------------------------------------------------------------------- #
# WebSocket endpoint
# -------------------------------------------------------------------- #


@app.websocket("/ws/call/{call_id}")
async def call_stream(ws: WebSocket, call_id: str) -> None:
    if not _is_authorized(ws):
        ERROR_EVENTS.labels(kind="auth_failed").inc()
        await ws.close(code=4401)
        return

    async with registry_lock:
        if len(active_calls) >= settings.max_concurrent_sessions:
            ERROR_EVENTS.labels(kind="max_sessions_reached").inc()
            await ws.close(code=4429)
            return

    await ws.accept()
    ws_send_lock = asyncio.Lock()

    # ── Start handshake ──
    try:
        first = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
        start = json.loads(first)
    except Exception:
        ERROR_EVENTS.labels(kind="bad_start").inc()
        await ws.close(code=4400)
        return

    if start.get("type") != "start":
        ERROR_EVENTS.labels(kind="missing_start").inc()
        await ws.close(code=4400)
        return

    sample_rate = int(start.get("sample_rate") or settings.sample_rate)
    channels = int(start.get("channels") or settings.channels)
    if sample_rate != settings.sample_rate or channels not in (1, 2):
        ERROR_EVENTS.labels(kind="bad_audio_format").inc()
        await ws.close(code=4400)
        return

    async def emit_fn(payload: dict[str, Any]) -> None:
        try:
            await _safe_emit(ws, ws_send_lock, payload)
        except Exception:
            ERROR_EVENTS.labels(kind="ws_slow_consumer").inc()

    mono_speaker = str(start.get("speaker") or "mono")
    left_speaker = str(start.get("left_speaker") or "agent")
    right_speaker = str(start.get("right_speaker") or "customer")
    call_language = str(start.get("language") or "").strip().lower()

    # ── Create sessions ──
    sessions: list[Session] = []
    channel_map: list[dict[str, Any]] = []

    if channels == 1:
        sessions.append(
            Session(
                call_id=call_id, speaker=mono_speaker,
                channel_id=0, emit_fn=emit_fn, language=call_language,
            )
        )
        channel_map.append({"channel_id": 0, "speaker": mono_speaker, "channel_name": "mono"})
    else:
        sessions.append(
            Session(
                call_id=call_id, speaker=left_speaker,
                channel_id=0, emit_fn=emit_fn, language=call_language,
            )
        )
        sessions.append(
            Session(
                call_id=call_id, speaker=right_speaker,
                channel_id=1, emit_fn=emit_fn, language=call_language,
            )
        )
        channel_map.append({"channel_id": 0, "speaker": left_speaker, "channel_name": "left"})
        channel_map.append({"channel_id": 1, "speaker": right_speaker, "channel_name": "right"})

    ctx = CallContext(
        call_id=call_id, ws=ws, channels=channels,
        sample_rate=sample_rate, sessions=sessions,
    )

    async with registry_lock:
        active_calls[call_id] = ctx
        ACTIVE_SESSIONS.set(len(active_calls))

    await emit_fn({"type": "started", "call_id": call_id, "channel_map": channel_map})

    # ── Main receive loop ──
    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # ── Binary audio frame ──
            if "bytes" in message and message["bytes"] is not None:
                frame = message["bytes"]
                if len(frame) > settings.max_frame_bytes:
                    ERROR_EVENTS.labels(kind="frame_too_large").inc()
                    await emit_fn({
                        "type": "error", "call_id": call_id,
                        "code": "frame_too_large",
                        "detail": f"Frame size {len(frame)} exceeds limit {settings.max_frame_bytes}",
                        "ts": time.time(),
                    })
                    continue

                BYTES_INGESTED.inc(len(frame))
                now = time.time()
                ctx.last_audio_at = now
                ctx.last_event_at = now

                if ctx.channels == 1:
                    mono_audio, _ = _decode_pcm16le(frame, 1)
                    await _process_channel_audio(ctx.sessions[0], mono_audio)
                else:
                    left_audio, right_audio = _decode_pcm16le(frame, 2)
                    await _process_channel_audio(ctx.sessions[0], left_audio)
                    await _process_channel_audio(
                        ctx.sessions[1],
                        right_audio if right_audio is not None else np.zeros(0, dtype=np.float32),
                    )
                continue

            # ── Text control events ──
            if "text" in message and message["text"]:
                try:
                    event = json.loads(message["text"])
                except json.JSONDecodeError:
                    ERROR_EVENTS.labels(kind="bad_json_event").inc()
                    await emit_fn({
                        "type": "error", "call_id": call_id,
                        "code": "bad_json", "detail": "Invalid JSON in text message",
                        "ts": time.time(),
                    })
                    continue

                event_type = str(event.get("type") or "").lower()
                if event_type == "ping":
                    ctx.last_event_at = time.time()
                    await emit_fn({"type": "pong", "call_id": call_id})
                elif event_type == "flush":
                    ctx.last_event_at = time.time()
                    for channel in ctx.sessions:
                        await transcriber.flush_final(channel)
                elif event_type == "stop":
                    ctx.last_event_at = time.time()
                    break
                else:
                    ERROR_EVENTS.labels(kind="unknown_event").inc()
                    await emit_fn({
                        "type": "error", "call_id": call_id,
                        "code": "unknown_event",
                        "detail": f"Unknown event type: {event_type}",
                        "ts": time.time(),
                    })

    except WebSocketDisconnect:
        pass
    except Exception:
        ERROR_EVENTS.labels(kind="ws_handler_exception").inc()
    finally:
        for channel in ctx.sessions:
            await transcriber.flush_final(channel)

        async with registry_lock:
            active_calls.pop(call_id, None)
            ACTIVE_SESSIONS.set(len(active_calls))
