#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import signal
import struct
import time
from typing import Optional

import websockets


def build_start_event(channels: int) -> dict:
    payload = {
        "type": "start",
        "sample_rate": 16000,
        "channels": channels,
    }
    if channels == 2:
        payload["left_speaker"] = "agent"
        payload["right_speaker"] = "customer"
    else:
        payload["speaker"] = "caller"
    return payload


def make_pcm_frame(channels: int, frame_ms: int, sample_rate: int, phase: float) -> tuple[bytes, float]:
    samples = int(sample_rate * (frame_ms / 1000.0))
    freq = 440.0
    amp = 0.12
    values: list[int] = []

    for index in range(samples):
        t = phase + (index / sample_rate)
        mono = int(max(-1.0, min(1.0, amp * math.sin(2 * math.pi * freq * t))) * 32767)
        if channels == 1:
            values.append(mono)
        else:
            left = mono
            right = int(max(-32767, min(32767, mono * 0.8)))
            values.extend((left, right))

    frame = struct.pack("<" + "h" * len(values), *values)
    return frame, phase + (samples / sample_rate)


async def recv_loop(ws) -> None:
    async for message in ws:
        if isinstance(message, bytes):
            print(f"[recv:binary] {len(message)} bytes")
            continue
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            print(f"[recv:text] {message}")
            continue
        print("[recv]", json.dumps(event, ensure_ascii=False))


async def ping_loop(ws, interval: float, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        if stop_event.is_set():
            return
        await ws.send(json.dumps({"type": "ping"}))


async def stream_audio(ws, channels: int, duration_seconds: float, frame_ms: int) -> None:
    sample_rate = 16000
    phase = 0.0
    started = time.time()
    while (time.time() - started) < duration_seconds:
        frame, phase = make_pcm_frame(channels, frame_ms, sample_rate, phase)
        await ws.send(frame)
        await asyncio.sleep(frame_ms / 1000.0)


async def run_client(url: str, call_id: str, channels: int, duration: float, ping_interval: float, frame_ms: int) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop(*_: object) -> None:
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _stop)
    loop.add_signal_handler(signal.SIGTERM, _stop)

    async with websockets.connect(f"{url.rstrip('/')}/ws/call/{call_id}", max_size=None) as ws:
        await ws.send(json.dumps(build_start_event(channels=channels)))

        receiver = asyncio.create_task(recv_loop(ws), name="ws-recv")
        pinger = asyncio.create_task(ping_loop(ws, ping_interval, stop_event), name="ws-ping")

        try:
            await stream_audio(ws, channels=channels, duration_seconds=duration, frame_ms=frame_ms)
            await ws.send(json.dumps({"type": "flush"}))
            await asyncio.sleep(0.5)
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.sleep(0.5)
        finally:
            stop_event.set()
            pinger.cancel()
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pinger
            with contextlib.suppress(asyncio.CancelledError):
                await receiver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo websocket client for whisper-stt server")
    parser.add_argument("--url", default="ws://127.0.0.1:8000", help="Base websocket URL (without /ws/call/{call_id})")
    parser.add_argument("--call-id", default="call-123")
    parser.add_argument("--channels", type=int, choices=(1, 2), default=2)
    parser.add_argument("--duration", type=float, default=6.0, help="Audio streaming duration in seconds")
    parser.add_argument("--ping-interval", type=float, default=5.0, help="Keepalive ping interval in seconds")
    parser.add_argument("--frame-ms", type=int, choices=(20, 40), default=20, help="Audio frame size in ms")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_client(
            url=args.url,
            call_id=args.call_id,
            channels=args.channels,
            duration=args.duration,
            ping_interval=args.ping_interval,
            frame_ms=args.frame_ms,
        )
    )
