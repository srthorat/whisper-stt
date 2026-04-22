#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import struct
import time
import wave

import websockets


def build_start_event(channels: int, language: str | None = None) -> dict:
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
    if language:
        payload["language"] = language
    return payload


async def recv_loop(ws) -> None:
    try:
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
    except Exception:
        # Connection closed normally after stop
        pass


async def ping_loop(ws, interval: float, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        if stop_event.is_set():
            return
        await ws.send(json.dumps({"type": "ping"}))


async def stream_wav_file(ws, wav_path: str, frame_ms: int) -> None:
    """Stream a WAV file to the websocket in chunks."""
    with wave.open(wav_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        
        print(f"[audio-file] {wav_path}")
        print(f"[audio-info] rate={sample_rate}Hz, channels={channels}, width={sample_width}bytes")
        
        # Calculate samples per frame
        samples_per_frame = int(sample_rate * (frame_ms / 1000.0))
        bytes_per_frame = samples_per_frame * channels * sample_width
        
        frame_count = 0
        while True:
            frame_data = wf.readframes(samples_per_frame)
            if not frame_data:
                break
            
            await ws.send(frame_data)
            frame_count += 1
            
            # Simulate real-time streaming
            await asyncio.sleep(frame_ms / 1000.0)
        
        print(f"[audio-complete] Sent {frame_count} frames")


async def run_client(url: str, call_id: str, wav_path: str, ping_interval: float, frame_ms: int, language: str | None = None) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop(*_: object) -> None:
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _stop)
    loop.add_signal_handler(signal.SIGTERM, _stop)

    # Read WAV file info to determine channels
    with wave.open(wav_path, 'rb') as wf:
        channels = wf.getnchannels()

    async with websockets.connect(f"{url.rstrip('/')}/ws/call/{call_id}", max_size=None) as ws:
        print(f"[connected] {url}/ws/call/{call_id}")
        
        # Send start event
        start_event = build_start_event(channels=channels, language=language)
        await ws.send(json.dumps(start_event))
        print(f"[sent] {json.dumps(start_event, ensure_ascii=False)}")

        receiver = asyncio.create_task(recv_loop(ws), name="ws-recv")
        pinger = asyncio.create_task(ping_loop(ws, ping_interval, stop_event), name="ws-ping")

        try:
            await stream_wav_file(ws, wav_path=wav_path, frame_ms=frame_ms)
            
            # Send flush and stop
            await ws.send(json.dumps({"type": "flush"}))
            print("[sent] flush")
            await asyncio.sleep(1.0)
            
            await ws.send(json.dumps({"type": "stop"}))
            print("[sent] stop")
            await asyncio.sleep(0.5)
        finally:
            stop_event.set()
            pinger.cancel()
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pinger
            with contextlib.suppress(asyncio.CancelledError, websockets.exceptions.ConnectionClosedError):
                await receiver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whisper-stt server with a WAV file")
    parser.add_argument("--url", default="ws://127.0.0.1:8000", help="Base websocket URL")
    parser.add_argument("--call-id", default="test-call")
    parser.add_argument("--wav-file", required=True, help="Path to WAV file (16kHz, 16-bit PCM)")
    parser.add_argument("--ping-interval", type=float, default=5.0, help="Keepalive ping interval")
    parser.add_argument("--frame-ms", type=int, choices=(20, 40), default=40, help="Audio frame size in ms")
    parser.add_argument("--language", type=str, default=None,
                        help="Language code (e.g. hi, en, es). Omit for auto-detect.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_client(
            url=args.url,
            call_id=args.call_id,
            wav_path=args.wav_file,
            ping_interval=args.ping_interval,
            frame_ms=args.frame_ms,
            language=args.language,
        )
    )
