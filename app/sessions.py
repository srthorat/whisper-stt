"""Per-channel streaming session with LocalAgreement hypothesis buffer.

Each Session accumulates audio during speech, periodically submits for
GPU inference, and uses LocalAgreement to decide which words are
confirmed (final) vs. still unstable (interim preview).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import logging

import numpy as np
import torch
import webrtcvad

from .config import settings
from .hypothesis_buffer import HypothesisBuffer, TimedWord

logger = logging.getLogger(__name__)

EmitFn = Callable[[dict], Awaitable[None]]


# -------------------------------------------------------------------- #
# VAD
# -------------------------------------------------------------------- #


class WebRTCSpeechGate:
    """WebRTC VAD-based speech detector."""

    def __init__(self) -> None:
        self._vad = webrtcvad.Vad(settings.vad_mode)
        self._frame_samples = int(settings.sample_rate * (settings.vad_frame_ms / 1000.0))
        self._frame_bytes = self._frame_samples * 2

    def has_voice(self, mono_samples: np.ndarray) -> bool:
        if mono_samples.size < self._frame_samples:
            return self._energy_fallback(mono_samples)

        pcm = (np.clip(mono_samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        raw = pcm.tobytes()
        for idx in range(0, len(raw) - self._frame_bytes + 1, self._frame_bytes):
            frame = raw[idx : idx + self._frame_bytes]
            if self._vad.is_speech(frame, settings.sample_rate):
                return True
        return self._energy_fallback(mono_samples)

    @staticmethod
    def _energy_fallback(mono_samples: np.ndarray) -> bool:
        if mono_samples.size == 0:
            return False
        energy = float(np.sqrt(np.mean(np.square(mono_samples))))
        return energy >= settings.vad_energy_threshold


class SileroSpeechGate:
    """Silero VAD-based speech detector (ONNX, higher accuracy)."""

    CHUNK_SAMPLES = 512  # Silero requires exactly 512 samples at 16kHz (32ms)

    def __init__(self) -> None:
        from silero_vad import load_silero_vad
        self._model = load_silero_vad(onnx=True)
        self._threshold = settings.silero_vad_threshold
        logger.info("Silero VAD loaded (ONNX, threshold=%.2f)", self._threshold)

    def has_voice(self, mono_samples: np.ndarray) -> bool:
        if mono_samples.size < self.CHUNK_SAMPLES:
            return self._energy_fallback(mono_samples)

        # Process in 512-sample chunks, return True if any chunk exceeds threshold
        for start in range(0, mono_samples.size - self.CHUNK_SAMPLES + 1, self.CHUNK_SAMPLES):
            chunk = mono_samples[start : start + self.CHUNK_SAMPLES]
            tensor = torch.from_numpy(chunk)
            prob = self._model(tensor, settings.sample_rate)
            if prob.item() >= self._threshold:
                return True
        return self._energy_fallback(mono_samples)

    @staticmethod
    def _energy_fallback(mono_samples: np.ndarray) -> bool:
        if mono_samples.size == 0:
            return False
        energy = float(np.sqrt(np.mean(np.square(mono_samples))))
        return energy >= settings.vad_energy_threshold


def _create_speech_gate():
    if settings.vad_engine == "silero":
        return SileroSpeechGate()
    return WebRTCSpeechGate()


speech_gate = _create_speech_gate()


# -------------------------------------------------------------------- #
# Session
# -------------------------------------------------------------------- #


@dataclass
class Session:
    """Per-channel streaming transcription session."""

    call_id: str
    speaker: str
    channel_id: int
    emit_fn: EmitFn
    language: str = ""  # per-session language override

    # ── Audio buffer (grows during speech, trimmed after commitment) ──
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    buffer_time_offset: float = 0.0  # segment-relative offset (seconds)

    # ── LocalAgreement ────────────────────────────────────────────────
    hypothesis_buffer: HypothesisBuffer = field(default_factory=HypothesisBuffer)

    # ── Text state ────────────────────────────────────────────────────
    confirmed: str = ""   # all confirmed text (full session lifetime)
    unstable: str = ""    # current unconfirmed preview text

    # ── Timing ────────────────────────────────────────────────────────
    created_at: float = field(default_factory=time.time)
    last_activity_ts: float = field(default_factory=time.time)
    last_voice_ts: float = field(default_factory=time.time)
    last_inference_ts: float = 0.0
    audio_cursor: int = 0  # new samples since last inference

    # ── GPU job control ───────────────────────────────────────────────
    has_pending_job: bool = False
    needs_complete: bool = False  # set when speech_end while job is pending

    # ── Speech state machine ──────────────────────────────────────────
    is_speaking: bool = False
    speech_frames: int = 0
    _silence_start_ts: float = 0.0

    # ================================================================ #
    # Audio processing
    # ================================================================ #

    def process_audio(self, mono_samples: np.ndarray) -> str:
        """Process incoming audio chunk: VAD + buffer management.

        Audio is accumulated continuously during speech (no reset between
        speech segments).  Buffer trimming handles unbounded growth.

        Returns one of:
          ``"speech_start"``     — speech just began
          ``"speech_continues"`` — speech ongoing / brief intra-speech silence
          ``"speech_end"``       — sustained silence after speech
          ``"silence"``          — no speech
        """
        if mono_samples.size == 0:
            return "silence"

        has_voice = speech_gate.has_voice(mono_samples)
        now = time.time()
        self.last_activity_ts = now

        if has_voice:
            self.last_voice_ts = now
            self.speech_frames += 1
            self._silence_start_ts = 0.0

            if not self.is_speaking:
                if self.speech_frames >= settings.min_speech_frames:
                    self.is_speaking = True
                    self._append_to_buffer(mono_samples)
                    return "speech_start"
                return "silence"

            self._append_to_buffer(mono_samples)
            return "speech_continues"

        # ── Silence ──
        self.speech_frames = 0

        if self.is_speaking:
            # Include brief silence in the buffer (padding the tail)
            self._append_to_buffer(mono_samples)

            if self._silence_start_ts == 0.0:
                self._silence_start_ts = now

            if now - self._silence_start_ts >= settings.vad_silence_seconds:
                self.is_speaking = False
                self._silence_start_ts = 0.0
                return "speech_end"
            return "speech_continues"

        return "silence"

    def _append_to_buffer(self, mono_samples: np.ndarray) -> None:
        self.audio_buffer = np.concatenate([self.audio_buffer, mono_samples])
        self.audio_cursor += mono_samples.size

    # ================================================================ #
    # Inference scheduling
    # ================================================================ #

    def should_infer(self) -> bool:
        """True when enough new audio has accumulated for another inference."""
        if self.has_pending_job or not self.is_speaking:
            return False
        min_samples = int(settings.sample_rate * settings.streaming_min_chunk_sec)
        return self.audio_cursor >= min_samples

    def get_inference_snapshot(self) -> tuple[np.ndarray, float, str]:
        """Return *(audio_copy, buffer_time_offset, initial_prompt)*."""
        audio = self.audio_buffer.copy()
        offset = self.buffer_time_offset
        # Last 200 chars of confirmed text for Whisper context continuity
        prompt = self.confirmed[-200:].strip() if self.confirmed else ""
        return audio, offset, prompt

    def mark_submitted(self) -> None:
        self.has_pending_job = True
        self.audio_cursor = 0
        self.last_inference_ts = time.time()

    def mark_done(self) -> None:
        self.has_pending_job = False

    # ================================================================ #
    # Buffer trimming
    # ================================================================ #

    def trim_buffer(self, trim_time: float) -> None:
        """Trim audio buffer up to *trim_time* (segment-relative seconds)."""
        if trim_time <= self.buffer_time_offset:
            return
        cut_seconds = trim_time - self.buffer_time_offset
        cut_samples = int(cut_seconds * settings.sample_rate)
        if cut_samples >= self.audio_buffer.size:
            self.audio_buffer = np.zeros(0, dtype=np.float32)
        else:
            self.audio_buffer = self.audio_buffer[cut_samples:]
        self.buffer_time_offset = trim_time
        self.hypothesis_buffer.pop_committed(trim_time)

    def maybe_trim_buffer(self) -> None:
        """Trim audio buffer if it exceeds the configured maximum.

        Keeps at least 3 s of committed audio for n-gram dedup anchoring.
        """
        buffer_duration = self.audio_buffer.size / settings.sample_rate
        if buffer_duration <= settings.streaming_max_buffer_sec:
            return

        committed = self.hypothesis_buffer.committed_in_buffer
        if not committed:
            # No committed words — hard-trim if way over limit
            if buffer_duration > settings.streaming_max_buffer_sec * 2:
                excess = buffer_duration - settings.streaming_max_buffer_sec
                cut_samples = int(excess * settings.sample_rate)
                self.audio_buffer = self.audio_buffer[cut_samples:]
                self.buffer_time_offset += excess
            return

        # Keep ≥ 3 s of committed audio before last_committed_time
        keep_window = 3.0
        candidate_time = self.hypothesis_buffer.last_committed_time - keep_window
        trim_candidates = [w.end for w in committed if w.end <= candidate_time]
        if trim_candidates:
            self.trim_buffer(trim_candidates[-1])

    # ================================================================ #
    # Speech segment lifecycle
    # ================================================================ #

    def flush_unconfirmed(self) -> list[TimedWord]:
        """Flush remaining unconfirmed words (on session close)."""
        return self.hypothesis_buffer.complete()

    # ================================================================ #
    # Emit helpers
    # ================================================================ #

    async def emit(self, msg_type: str, text: str) -> None:
        payload = {
            "call_id": self.call_id,
            "speaker": self.speaker,
            "channel_id": self.channel_id,
            "type": msg_type,
            "text": text,
            "ts": time.time(),
            "queued_at": getattr(self, "_infer_queued_at", None),
        }
        await self.emit_fn(payload)
