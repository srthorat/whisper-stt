import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (sets CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH, etc.)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=False)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


@dataclass
class Settings:
    # ── Core audio / streaming ───────────────────────────────────────────
    sample_rate: int = 16_000
    channels: int = 2
    # ── Queue / back-pressure ───────────────────────────────────────────
    queue_drop_threshold: int = 60
    max_batch_jobs: int = 4
    gpu_worker_count: int = 1
    # ── Adaptive quality (dynamic scaling under GPU pressure) ───────────
    adaptive_quality: bool = True
    adaptive_queue_threshold: int = 20      # Queue depth to trigger fast mode
    fast_beam_size: int = 1                 # Beam size in fast mode
    fast_patience: float = 1.0              # Patience in fast mode
    # ── VAD ─────────────────────────────────────────────────────────────
    vad_engine: str = "silero"             # "webrtcvad" or "silero"
    vad_mode: int = 3
    vad_frame_ms: int = 30
    vad_energy_threshold: float = 0.03
    vad_silence_seconds: float = 0.35
    silero_vad_threshold: float = 0.5     # Silero speech probability threshold
    # ── Whisper / GPU ───────────────────────────────────────────────────
    whisper_model: str = "turbo"
    device: str = "cuda"
    cuda_device_index: int = 0
    compute_type: str = "int8_float16"
    # ── Session lifecycle ───────────────────────────────────────────────
    session_idle_timeout_seconds: float = 20.0
    session_max_duration_seconds: float = 14_400.0
    cleanup_interval_seconds: float = 5.0
    max_concurrent_sessions: int = 100
    max_frame_bytes: int = 32_768
    api_key: str = ""
    ws_send_timeout_seconds: float = 2.0
    # ── Streaming / LocalAgreement ──────────────────────────────────────
    streaming_min_chunk_sec: float = 1.0   # Min new audio before queuing inference
    streaming_max_buffer_sec: float = 15.0 # Max audio buffer before trimming
    min_speech_frames: int = 2             # Speech debounce (VAD frames)
    # ── Whisper quality tuning ──────────────────────────────────────────
    whisper_language: str = ""
    whisper_beam_size: int = 5
    whisper_temperature: float = 0.0
    whisper_no_speech_threshold: float = 0.6
    whisper_condition_on_previous: bool = True
    hallucination_filter: bool = True
    # ── Advanced tuning (faster-whisper / CTranslate2) ──────────────────
    whisper_hotwords: str = ""              # Comma-separated hint phrases
    whisper_repetition_penalty: float = 1.15  # Penalise repeated tokens
    whisper_no_repeat_ngram_size: int = 3     # Block trigram repetitions
    whisper_patience: float = 1.5             # Beam-search patience factor
    whisper_word_timestamps: bool = True      # Word-level timestamps via cross-attention
    whisper_hallucination_silence_threshold: float = 1.0  # Skip hallucinated silence (needs word_timestamps)
    whisper_compression_ratio_threshold: float = 2.4      # Filter high-compression segments
    whisper_log_prob_threshold: float = -1.0               # Filter low-probability segments
    whisper_length_penalty: float = 1.0                    # Exponential length penalty


def load_settings() -> Settings:
    s = Settings(
        sample_rate=_get_int("SAMPLE_RATE", 16_000),
        channels=_get_int("CHANNELS", 2),
        queue_drop_threshold=_get_int("QUEUE_DROP_THRESHOLD", 60),
        max_batch_jobs=_get_int("MAX_BATCH_JOBS", 4),
        gpu_worker_count=_get_int("GPU_WORKER_COUNT", 1),
        adaptive_quality=_get_str("ADAPTIVE_QUALITY", "true").lower() in ("true", "1", "yes"),
        adaptive_queue_threshold=_get_int("ADAPTIVE_QUEUE_THRESHOLD", 20),
        fast_beam_size=_get_int("FAST_BEAM_SIZE", 1),
        fast_patience=_get_float("FAST_PATIENCE", 1.0),
        vad_engine=_get_str("VAD_ENGINE", "webrtcvad"),
        vad_mode=_get_int("VAD_MODE", 3),
        vad_frame_ms=_get_int("VAD_FRAME_MS", 30),
        vad_energy_threshold=_get_float("VAD_ENERGY_THRESHOLD", 0.03),
        vad_silence_seconds=_get_float("VAD_SILENCE_SECONDS", 0.35),
        silero_vad_threshold=_get_float("SILERO_VAD_THRESHOLD", 0.5),
        whisper_model=_get_str("WHISPER_MODEL", "turbo"),
        device=_get_str("WHISPER_DEVICE", "cuda"),
        cuda_device_index=_get_int("CUDA_DEVICE_INDEX", 0),
        compute_type=_get_str("WHISPER_COMPUTE_TYPE", "int8_float16"),
        session_idle_timeout_seconds=_get_float("SESSION_IDLE_TIMEOUT_SECONDS", 20.0),
        session_max_duration_seconds=_get_float("SESSION_MAX_DURATION_SECONDS", 14_400.0),
        cleanup_interval_seconds=_get_float("CLEANUP_INTERVAL_SECONDS", 5.0),
        max_concurrent_sessions=_get_int("MAX_CONCURRENT_SESSIONS", 100),
        max_frame_bytes=_get_int("MAX_FRAME_BYTES", 32_768),
        api_key=_get_str("API_KEY", ""),
        ws_send_timeout_seconds=_get_float("WS_SEND_TIMEOUT_SECONDS", 2.0),
        streaming_min_chunk_sec=_get_float("STREAMING_MIN_CHUNK_SEC", 1.0),
        streaming_max_buffer_sec=_get_float("STREAMING_MAX_BUFFER_SEC", 15.0),
        min_speech_frames=_get_int("MIN_SPEECH_FRAMES", 2),
        whisper_language=_get_str("WHISPER_LANGUAGE", ""),
        whisper_beam_size=_get_int("WHISPER_BEAM_SIZE", 5),
        whisper_temperature=_get_float("WHISPER_TEMPERATURE", 0.0),
        whisper_no_speech_threshold=_get_float("WHISPER_NO_SPEECH_THRESHOLD", 0.6),
        whisper_condition_on_previous=_get_str("WHISPER_CONDITION_ON_PREVIOUS", "true").lower() in ("true", "1", "yes"),
        hallucination_filter=_get_str("HALLUCINATION_FILTER", "true").lower() in ("true", "1", "yes"),
        whisper_hotwords=_get_str("WHISPER_HOTWORDS", ""),
        whisper_repetition_penalty=_get_float("WHISPER_REPETITION_PENALTY", 1.15),
        whisper_no_repeat_ngram_size=_get_int("WHISPER_NO_REPEAT_NGRAM_SIZE", 3),
        whisper_patience=_get_float("WHISPER_PATIENCE", 1.5),
        whisper_word_timestamps=_get_str("WHISPER_WORD_TIMESTAMPS", "true").lower() in ("true", "1", "yes"),
        whisper_hallucination_silence_threshold=_get_float("WHISPER_HALLUCINATION_SILENCE_THRESHOLD", 1.0),
        whisper_compression_ratio_threshold=_get_float("WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4),
        whisper_log_prob_threshold=_get_float("WHISPER_LOG_PROB_THRESHOLD", -1.0),
        whisper_length_penalty=_get_float("WHISPER_LENGTH_PENALTY", 1.0),
    )
    
    # Validate configuration
    if s.vad_engine not in ("webrtcvad", "silero"):
        raise ValueError(f"VAD_ENGINE must be 'webrtcvad' or 'silero' (got {s.vad_engine})")
    if s.vad_engine == "webrtcvad" and s.vad_frame_ms not in (10, 20, 30):
        raise ValueError(f"vad_frame_ms must be 10, 20, or 30 (got {s.vad_frame_ms})")
    if s.session_idle_timeout_seconds <= 0:
        raise ValueError(f"session_idle_timeout_seconds must be positive (got {s.session_idle_timeout_seconds})")
    if s.session_max_duration_seconds <= 0:
        raise ValueError(f"session_max_duration_seconds must be positive (got {s.session_max_duration_seconds})")
    if s.cleanup_interval_seconds <= 0:
        raise ValueError(f"cleanup_interval_seconds must be positive (got {s.cleanup_interval_seconds})")
    if s.max_concurrent_sessions <= 0:
        raise ValueError(f"max_concurrent_sessions must be positive (got {s.max_concurrent_sessions})")
    
    return s


settings = load_settings()
