"""GPU inference worker with LocalAgreement streaming.

Architecture:
  1. Sessions queue InferenceJobs (audio snapshot + metadata).
  2. A single GPU worker dequeues jobs, runs Whisper with word_timestamps,
     and returns timestamped words.
  3. Post-inference, the session's HypothesisBuffer applies LocalAgreement
     to commit stable words → emit "final", preview unstable → emit "interim".
  4. Audio buffer is trimmed after commitment to bound inference cost.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from .config import settings
from .hypothesis_buffer import TimedWord
from .metrics import DROPPED_JOBS, ERROR_EVENTS, INFERENCE_SECONDS, QUEUE_SIZE, TRANSCRIPT_EVENTS
from .sessions import Session

# -------------------------------------------------------------------- #
# Hallucination filter
# -------------------------------------------------------------------- #

_HALLUCINATION_PATTERNS: set[str] = {
    "thank you.", "thank you", "thanks.", "thanks",
    "thank you so much.", "thank you for watching.",
    "bye.", "bye-bye.", "goodbye.", "gracias.", "grazie.",
    "you're welcome.", "you", ".", "...", "you.",
    "laughter", "laughter.", "(laughter)", "[laughter]",
    "music", "music.", "applause", "silence",
    "i'll see you next time.", "see you next time.",
    "subscribe.", "please subscribe.", "like and subscribe.",
    "thank you for listening.",
}


def _is_hallucination(text: str) -> bool:
    if not settings.hallucination_filter:
        return False
    cleaned = text.strip().lower()
    if cleaned in _HALLUCINATION_PATTERNS:
        return True
    if re.match(r'^(thank you\.?\s*){2,}$', cleaned, re.IGNORECASE):
        return True
    return False


def _dedup_repetitions(text: str) -> str:
    """Remove repeated phrases/sentences caused by Whisper decoder loops."""
    if not text or len(text) < 20:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 1:
        parts = text.split(' or ')
        if len(parts) >= 3:
            first = parts[0].strip().lower()
            repeats = sum(
                1 for p in parts[1:]
                if p.strip().lower() == first
                or first.endswith(p.strip().lower())
                or p.strip().lower().startswith(first[: min(len(first), 10)])
            )
            if repeats >= 2:
                return f"{parts[0].strip()} or {parts[1].strip()}"
        return text

    seen: list[str] = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if not seen or s_clean.lower() != seen[-1].lower():
            seen.append(s_clean)

    result = " ".join(seen)
    if len(result) < len(text) * 0.3 and len(text) > 50:
        return seen[0] if seen else ""
    return result


def _strip_confirmed_overlap(new_text: str, confirmed: str) -> str:
    """Strip already-confirmed text from new transcription.

    Prevents duplication when Whisper re-produces committed context after
    buffer trimming or during final-flush re-transcription.
    """
    if not confirmed or not new_text:
        return new_text

    conf_words = confirmed.lower().split()
    new_lower_words = new_text.lower().split()
    new_raw_words = new_text.strip().split()

    if not new_lower_words or not conf_words:
        return new_text

    # Check 1: suffix of confirmed matches prefix of new (classic overlap)
    best_overlap = 0
    for n in range(min(len(conf_words), len(new_lower_words)), 0, -1):
        if conf_words[-n:] == new_lower_words[:n]:
            best_overlap = n
            break

    if best_overlap > 0:
        remaining = new_raw_words[best_overlap:]
        return " ".join(remaining).strip()

    # Check 2: entire new text is already contained in confirmed tail
    new_clean = " ".join(new_lower_words)
    # Check against the last 3× the length of new_text in confirmed
    check_len = len(new_clean) * 3
    conf_tail = " ".join(conf_words).strip()[-check_len:] if conf_words else ""
    if new_clean in conf_tail:
        return ""

    # Check 3: new text starts with extra words then overlaps with confirmed tail
    # E.g., confirmed: "X Y Z", new: "A B X Y Z W" → strip "X Y Z", keep "A B W"
    for start in range(1, min(len(new_lower_words), 4)):
        for n in range(min(len(conf_words), len(new_lower_words) - start), 2, -1):
            if conf_words[-n:] == new_lower_words[start : start + n]:
                before = new_raw_words[:start]
                after = new_raw_words[start + n :]
                result = " ".join(before + after).strip()
                return result

    return new_text


# -------------------------------------------------------------------- #
# Inference job
# -------------------------------------------------------------------- #


@dataclass
class InferenceJob:
    session: Session
    audio: np.ndarray
    buffer_time_offset: float = 0.0
    initial_prompt: str = ""
    is_speech_end: bool = False  # commit all remaining on completion
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other: "InferenceJob") -> bool:
        """FIFO ordering."""
        return self.created_at < other.created_at


# -------------------------------------------------------------------- #
# Transcriber service
# -------------------------------------------------------------------- #


class TranscriberService:
    def __init__(self) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(settings.cuda_device_index))
        self.model = WhisperModel(
            settings.whisper_model,
            device=settings.device,
            device_index=settings.cuda_device_index,
            compute_type=settings.compute_type,
        )
        self.queue: asyncio.PriorityQueue[InferenceJob] = asyncio.PriorityQueue()
        self._pending_jobs: dict[int, InferenceJob] = {}  # id(session) → latest job
        self.worker_tasks: list[asyncio.Task] = []
        self._inference_lock = asyncio.Lock()
        self._stopping = False

    # ── Lifecycle ──────────────────────────────────────────────────── #

    async def start(self) -> None:
        if self.worker_tasks and any(not t.done() for t in self.worker_tasks):
            return
        self._stopping = False
        self.worker_tasks = [
            asyncio.create_task(self._gpu_worker(idx), name=f"gpu-worker-{idx}")
            for idx in range(settings.gpu_worker_count)
        ]

    async def stop(self) -> None:
        self._stopping = True
        for t in self.worker_tasks:
            t.cancel()
        for t in self.worker_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self.worker_tasks = []

    # ── Submit methods ─────────────────────────────────────────────── #

    async def submit_streaming(self, session: Session) -> bool:
        """Queue an incremental streaming inference for the session."""
        if session.has_pending_job:
            return False

        audio, offset, prompt = session.get_inference_snapshot()
        if audio.size == 0:
            return False

        session.mark_submitted()
        job = InferenceJob(
            session=session,
            audio=audio,
            buffer_time_offset=offset,
            initial_prompt=prompt,
        )
        self._track_job(session, job)
        await self.queue.put(job)
        QUEUE_SIZE.set(self.queue.qsize())
        return True

    async def submit_streaming_force(
        self, session: Session, is_speech_end: bool = False
    ) -> bool:
        """Submit inference for remaining audio (e.g. on speech_end).

        Like submit_streaming but ignores the min_chunk_sec threshold.
        """
        if session.has_pending_job:
            return False

        audio, offset, prompt = session.get_inference_snapshot()
        if audio.size == 0:
            return False

        session.mark_submitted()
        job = InferenceJob(
            session=session,
            audio=audio,
            buffer_time_offset=offset,
            initial_prompt=prompt,
            is_speech_end=is_speech_end,
        )
        self._track_job(session, job)
        await self.queue.put(job)
        QUEUE_SIZE.set(self.queue.qsize())
        return True

    async def submit_streaming_final(self, session: Session) -> None:
        """Handle end of speech segment.

        Submits a final inference for any remaining un-transcribed audio.
        If a job is already pending, marks the session so the pending job
        will commit remaining words when it finishes.
        """
        if session.has_pending_job:
            # Flag so the running/queued job commits everything on completion
            session.needs_complete = True
            return

        # Force-submit remaining audio regardless of min_chunk_sec
        await self.submit_streaming_force(session, is_speech_end=True)

    async def flush_final(self, session: Session) -> None:
        """Force flush on session close / stop event.

        Cancels any in-flight job and emits all remaining unconfirmed text.
        """
        if session.has_pending_job:
            self._cancel_pending(session)
            running = getattr(session, '_running_job', None)
            if running is not None:
                running._cancelled = True  # type: ignore[attr-defined]
            session.mark_done()

        await self._flush_remaining(session)

    # ── Job tracking ───────────────────────────────────────────────── #

    def _track_job(self, session: Session, job: InferenceJob) -> None:
        sid = id(session)
        old = self._pending_jobs.get(sid)
        if old is not None:
            old._cancelled = True  # type: ignore[attr-defined]
            DROPPED_JOBS.labels(reason="stale_streaming_replaced").inc()
        self._pending_jobs[sid] = job

    def _cancel_pending(self, session: Session) -> None:
        sid = id(session)
        old = self._pending_jobs.pop(sid, None)
        if old is not None:
            old._cancelled = True  # type: ignore[attr-defined]
            DROPPED_JOBS.labels(reason="superseded_by_final").inc()

    # ── GPU worker loop ────────────────────────────────────────────── #

    async def _gpu_worker(self, worker_id: int) -> None:
        while not self._stopping:
            first = await self.queue.get()
            QUEUE_SIZE.set(self.queue.qsize())
            jobs = [first]
            # Drain a small batch for efficiency
            while len(jobs) < settings.max_batch_jobs:
                try:
                    jobs.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            for job in jobs:
                try:
                    if getattr(job, "_cancelled", False):
                        QUEUE_SIZE.set(self.queue.qsize())
                        continue
                    # Clear tracking when processing
                    sid = id(job.session)
                    if self._pending_jobs.get(sid) is job:
                        del self._pending_jobs[sid]
                    await self._run_job(job)
                finally:
                    job.session.mark_done()
                    self.queue.task_done()
            QUEUE_SIZE.set(self.queue.qsize())

    # ── Core inference ─────────────────────────────────────────────── #

    def _transcribe_words(
        self, samples: np.ndarray, session: Session, initial_prompt: str
    ) -> list[TimedWord]:
        """Run Whisper with word_timestamps and return TimedWord list.

        Timestamps are relative to the audio chunk start (0-based).
        """
        queue_depth = self.queue.qsize()
        use_fast = (
            settings.adaptive_quality
            and queue_depth >= settings.adaptive_queue_threshold
        )

        beam_size = settings.fast_beam_size if use_fast else settings.whisper_beam_size
        patience = settings.fast_patience if use_fast else settings.whisper_patience

        # Always use word_timestamps — LocalAgreement requires them
        hal_silence = (
            settings.whisper_hallucination_silence_threshold
            if settings.whisper_hallucination_silence_threshold > 0
            else None
        )

        effective_language = session.language or settings.whisper_language or None

        segments, _info = self.model.transcribe(
            samples,
            beam_size=beam_size,
            language=effective_language,
            vad_filter=False,
            temperature=settings.whisper_temperature,
            no_speech_threshold=settings.whisper_no_speech_threshold,
            initial_prompt=initial_prompt or None,
            condition_on_previous_text=False,
            hotwords=settings.whisper_hotwords.strip() or None,
            repetition_penalty=settings.whisper_repetition_penalty,
            no_repeat_ngram_size=settings.whisper_no_repeat_ngram_size,
            patience=patience,
            word_timestamps=True,
            hallucination_silence_threshold=hal_silence,
            compression_ratio_threshold=settings.whisper_compression_ratio_threshold,
            log_prob_threshold=settings.whisper_log_prob_threshold,
            length_penalty=settings.whisper_length_penalty,
        )

        words: list[TimedWord] = []
        for seg in segments:
            # Skip high no-speech segments
            if seg.no_speech_prob > 0.9:
                continue
            if _is_hallucination(seg.text.strip()):
                continue
            if seg.words:
                for w in seg.words:
                    words.append(
                        TimedWord(
                            start=w.start,
                            end=w.end,
                            text=w.word,
                            probability=w.probability,
                        )
                    )
        return words

    async def _run_job(self, job: InferenceJob) -> None:
        session = job.session
        audio = job.audio

        if audio.size == 0:
            return

        # Track running job on session for cancellation from submit_streaming_final
        session._running_job = job  # type: ignore[attr-defined]

        infer_start = time.perf_counter()
        session._infer_queued_at = time.time()  # wall-clock for latency tracking
        try:
            async with self._inference_lock:
                words = await asyncio.to_thread(
                    self._transcribe_words,
                    audio.astype(np.float32),
                    session,
                    job.initial_prompt,
                )
        except RuntimeError as err:
            ERROR_EVENTS.labels(kind="gpu_runtime").inc()
            detail = str(err)
            code = "gpu_oom" if "out of memory" in detail.lower() else "inference_runtime_error"
            if code == "gpu_oom":
                DROPPED_JOBS.labels(reason="gpu_oom").inc()
            await session.emit_fn(
                {
                    "type": "error",
                    "call_id": session.call_id,
                    "channel_id": session.channel_id,
                    "speaker": session.speaker,
                    "code": code,
                    "detail": detail,
                    "ts": time.time(),
                }
            )
            return
        except Exception as err:
            ERROR_EVENTS.labels(kind="inference_exception").inc()
            await session.emit_fn(
                {
                    "type": "error",
                    "call_id": session.call_id,
                    "channel_id": session.channel_id,
                    "speaker": session.speaker,
                    "code": "inference_failed",
                    "detail": str(err),
                    "ts": time.time(),
                }
            )
            return
        finally:
            INFERENCE_SECONDS.observe(time.perf_counter() - infer_start)
            session._running_job = None  # type: ignore[attr-defined]

        # If cancelled while GPU was running (e.g. by submit_streaming_final),
        # skip post-processing to avoid overwriting flushed hypothesis state
        if getattr(job, '_cancelled', False):
            return

        should_complete = job.is_speech_end or session.needs_complete
        await self._handle_streaming_result(
            session, words, job.buffer_time_offset, commit_remaining=should_complete,
        )
        if should_complete:
            session.needs_complete = False

    # ── Post-inference processing ──────────────────────────────────── #

    async def _handle_streaming_result(
        self,
        session: Session,
        words: list[TimedWord],
        buffer_time_offset: float,
        commit_remaining: bool = False,
    ) -> None:
        """Run LocalAgreement, emit finals/interims, trim buffer."""

        # Insert into hypothesis buffer and run LocalAgreement
        if words:
            session.hypothesis_buffer.insert(words, buffer_time_offset)
            committed = session.hypothesis_buffer.flush()
        else:
            committed = []

        # Emit committed words as "final"
        if committed:
            text = "".join(w.text for w in committed).strip()
            text = _dedup_repetitions(text)
            text = _strip_confirmed_overlap(text, session.confirmed)
            if text and not _is_hallucination(text):
                session.confirmed = f"{session.confirmed} {text}".strip()
                await session.emit("final", text)
                TRANSCRIPT_EVENTS.labels(event_type="final").inc()

        # On speech_end: commit all remaining unconfirmed words
        if commit_remaining:
            remaining = session.hypothesis_buffer.complete()
            if remaining:
                tail_text = "".join(w.text for w in remaining).strip()
                tail_text = _dedup_repetitions(tail_text)
                tail_text = _strip_confirmed_overlap(tail_text, session.confirmed)
                if tail_text and not _is_hallucination(tail_text):
                    session.confirmed = f"{session.confirmed} {tail_text}".strip()
                    await session.emit("final", tail_text)
                    TRANSCRIPT_EVENTS.labels(event_type="final").inc()
                # Update tracking so future inferences don't re-emit
                session.hypothesis_buffer.last_committed_time = max(
                    session.hypothesis_buffer.last_committed_time,
                    remaining[-1].end,
                )
                session.hypothesis_buffer.committed_in_buffer.extend(remaining)
            session.unstable = ""
            return

        # Emit unconfirmed words as "interim" (only during ongoing speech)
        unconfirmed = session.hypothesis_buffer.get_unconfirmed_text()
        if unconfirmed != session.unstable:
            session.unstable = unconfirmed
            if unconfirmed:
                await session.emit("interim", unconfirmed)
                TRANSCRIPT_EVENTS.labels(event_type="interim").inc()

        # Buffer trimming
        session.maybe_trim_buffer()

        # If speech has ended while we were inferring, commit all
        # remaining unconfirmed words as final.  This mirrors
        # whisper_streaming's finish() / SimulWhisper's refresh_segment(complete=True).
        if not session.is_speaking:
            await self._flush_remaining(session)

    async def _flush_remaining(self, session: Session) -> None:
        """Commit any remaining unconfirmed words as a final (session close)."""
        remaining = session.flush_unconfirmed()
        if remaining:
            text = "".join(w.text for w in remaining).strip()
            text = _dedup_repetitions(text)
            text = _strip_confirmed_overlap(text, session.confirmed)
            if text and not _is_hallucination(text):
                session.confirmed = f"{session.confirmed} {text}".strip()
                await session.emit("final", text)
                TRANSCRIPT_EVENTS.labels(event_type="final").inc()
            # Update last_committed_time so future inferences don't re-emit
            session.hypothesis_buffer.last_committed_time = max(
                session.hypothesis_buffer.last_committed_time,
                remaining[-1].end,
            )
            session.hypothesis_buffer.committed_in_buffer.extend(remaining)
        session.unstable = ""

    # ── Queue management ───────────────────────────────────────────── #

    async def _drop_stale_jobs_if_needed(self) -> None:
        """Drop old streaming jobs when queue is too deep."""
        queue_size = self.queue.qsize()
        if queue_size <= settings.queue_drop_threshold:
            return

        kept: list[InferenceJob] = []
        dropped = 0
        target_drop = max(1, queue_size - settings.queue_drop_threshold)

        while True:
            try:
                job = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if dropped < target_drop:
                job.session.mark_done()
                self.queue.task_done()
                dropped += 1
                DROPPED_JOBS.labels(reason="queue_backpressure").inc()
                continue
            self.queue.task_done()
            kept.append(job)

        for job in kept:
            await self.queue.put(job)
        QUEUE_SIZE.set(self.queue.qsize())
