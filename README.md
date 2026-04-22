# Whisper Turbo Streaming STT Server

Production-grade WebSocket speech-to-text server built on **faster-whisper turbo** with true streaming via **LocalAgreement**, utterance-based VAD, defence-in-depth hallucination filtering, and Prometheus metrics.

Benchmarked against **6 candidate models** (Whisper turbo, Voxtral 4B, Moonshine Medium, SenseVoice Small, Qwen3-ASR-0.6B, Whisper CPU) across quality, speed, concurrency, and cost. **Whisper turbo is the winner for streaming** — see [Model Benchmark Results](#model-benchmark-results) below. **Qwen3-ASR-0.6B** is a strong contender for batch/non-streaming workloads (99+ concurrent RT streams on a single L4).

---

## Architecture

### High-Level Data Flow

```
WebSocket clients ──PCM 16kHz──►  FastAPI  ──►  Silero VAD  ──►  GPU queue  ──►  Whisper turbo
  (stereo/mono)                     │            │                  │               │
                                    │       utterance FSM      back-pressure   word_timestamps
                                    │    (speech_start/end)    + adaptive      + hallucination
                                    │            │              quality          filter
                                    │            │                │               │
                                    ◄──── interim (preview) ◄─── LocalAgreement ─┘
                                    ◄──── final  (committed) ◄─── (2-pass confirm)
```

### Streaming Pipeline Detail

The server implements a **true streaming architecture** — audio is processed incrementally as it arrives, not in batch after the call ends. The pipeline has 5 stages:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: AUDIO INGESTION                                                        │
│                                                                                 │
│  WebSocket ──► PCM-16LE decode ──► stereo demux (L/R channels)                  │
│  • 16 kHz, 16-bit signed, little-endian                                         │
│  • 20-40ms frames recommended                                                   │
│  • Each channel → independent Session                                           │
└──────────────────────────────┬──────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: VOICE ACTIVITY DETECTION                                               │
│                                                                                 │
│  Silero VAD (ONNX) ──► speech/silence classification per 512-sample chunk       │
│  • Threshold: 0.5 probability                                                   │
│  • Energy fallback for short frames                                             │
│  • Utterance FSM: silence ──► speech_start ──► speech_continues ──► speech_end   │
│  • Speech debounce: 2 voiced frames to trigger                                  │
│  • Silence duration: 350ms to end utterance                                     │
└──────────────────────────────┬──────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: AUDIO BUFFER + INFERENCE SCHEDULING                                    │
│                                                                                 │
│  Audio accumulates during speech in a growing buffer (up to 15s, then trimmed)  │
│  • Every 0.5s of new audio → snapshot + queue InferenceJob                      │
│  • On speech_end → force-submit remaining audio                                 │
│  • One job per session (supersedes stale jobs)                                  │
│  • Queue back-pressure: drop oldest jobs at depth > 120                         │
└──────────────────────────────┬──────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Stage 4: GPU INFERENCE                                                          │
│                                                                                 │
│  Single async GPU worker dequeues jobs (batches up to 8)                        │
│  • faster-whisper turbo, int8_float16 quantization                              │
│  • word_timestamps=True (cross-attention alignment)                             │
│  • Adaptive quality: beam=1 when queue > 5, beam=5 otherwise                    │
│  • Language: auto-detect or per-call pinning (99 languages)                     │
│  • initial_prompt from last 200 chars of confirmed text                         │
│  • Post-inference: hallucination filter (30+ patterns) + dedup                  │
└──────────────────────────────┬──────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Stage 5: LocalAgreement HYPOTHESIS BUFFER                                       │
│                                                                                 │
│  Words from consecutive passes are compared word-by-word:                       │
│  • Longest common prefix → COMMITTED (emit "final")                             │
│  • Remaining → UNCONFIRMED (emit "interim" preview)                             │
│  • 1-word lookahead realignment handles insertions/deletions                    │
│  • N-gram dedup (up to 8-grams) prevents re-emission                           │
│  • On speech_end → commit all remaining as final                                │
│  • Buffer trimmed after commitment to bound inference cost                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Implementation |
|-----------|---------------|
| Model | `faster-whisper` turbo, int8_float16, ~1.2 GB VRAM |
| VAD | Silero VAD (ONNX, threshold=0.5) + energy fallback → utterance state machine |
| Streaming | LocalAgreement — 2-pass word-level confirmation with lookahead realignment |
| Inference | Single async GPU worker, adaptive quality (beam=5 normal / beam=1 under load) |
| Anti-hallucination | 30+ pattern filter, `hallucination_silence_threshold`, compression ratio filter, log-prob filter, repetition dedup, prefix stripping |
| Transport | WebSocket `/ws/call/{call_id}`, stereo (L/R) or mono |
| Language | Auto-detect (99 languages) or per-call pinning via `start` event |
| Monitoring | Prometheus metrics + `/healthz` + full status dashboard |

## Performance

- **GPU RTFx**: ~64× real-time (single stream, L4 GPU)
- **Streaming TTFT**: ~1.4s first interim text
- **Concurrent streams**: 10–15 real-time streams per L4 GPU
- **Accuracy**: ~91 % word accuracy (vs AWS Transcribe ~96.5 %, Google STT ~92.5 %)
- **Cost**: ~$8–12 / 1000 hrs (120–180× cheaper than cloud providers)
- **VRAM**: 1.2 GB model + ~100 MB per active stream
- **Hardware**: NVIDIA L4 GPU (g6.2xlarge) — 23 GB VRAM

---

## Model Benchmark Results

We evaluated 6 candidate models/configurations for production streaming STT on an **NVIDIA L4 GPU** (AWS g6.2xlarge, 23 GB VRAM, 300 GB/s bandwidth, 8 vCPUs). **Whisper turbo is the clear winner** for production streaming. **Qwen3-ASR-0.6B** is the best alternative for batch/concurrent workloads with multilingual support.

### 1. Whisper Large-v3 Turbo (GPU) — WINNER ✓

The production model. CTranslate2/faster-whisper with int8_float16 quantization.

| Metric | Value |
|--------|-------|
| Architecture | Encoder-decoder (autoregressive), 809M params |
| VRAM | 1.2 GB |
| Single-stream RTFx | **~64×** real-time |
| Batch RTFx | ~70× (GPU-bottlenecked, flat regardless of instances) |
| Streaming | LocalAgreement (true streaming, word-level confirmation) |
| TTFT | ~1.4s (first interim) |
| Languages | 99 (auto-detect or pinned) |
| Hindi | Partial support |
| English quality | Very good (~91% word accuracy) |
| Concurrent streams (safe) | **10–15** real-time on L4 |
| Concurrent streams (max) | 30 (all succeed, quality degrades at 20+) |

**Concurrent streaming results (beam=1, 0.5s chunks):**

| Streams | Success | Med TTFT | Max TTFT | Wall Time | Real-time? |
|---------|---------|----------|----------|-----------|------------|
| 1 | 1/1 | 1,391ms | 1,391ms | 14.9s | ✓ |
| 5 | 5/5 | 1,751ms | 2,324ms | 15.2s | ✓ |
| 10 | 10/10 | 2,466ms | 3,753ms | 15.3s | ✓ |
| 15 | 15/15 | 3,198ms | 5,206ms | 15.2s | ✓ |
| 20 | 20/20 | 3,927ms | 6,670ms | 15.4s | ✓ |
| 25 | 25/25 | 4,685ms | 8,164ms | 15.3s | ✓ |
| 30 | 30/30 | 5,428ms | 9,638ms | 15.3s | ✓ |

Multi-worker tested (1–5 instances on same GPU): total throughput flat at ~70× RTFx. **GPU compute is the bottleneck**, not model count.

### 2. Voxtral Realtime 4B (vLLM)

Mistral's multimodal LLM-based ASR via vLLM nightly with `--max-model-len 8192 --enforce-eager`.

| Metric | Value |
|--------|-------|
| Architecture | LLM-based (Pixtral encoder + Tekken decoder), 4B params |
| VRAM | **21.5 GB** (nearly fills L4) |
| Transformers RTFx | ~1.26× (barely real-time) |
| vLLM `/v1/audio/transcriptions` RTFx | 1.88–1.96× |
| vLLM WebSocket streaming RTFx | 1.74–1.79× |
| TTFT (WebSocket) | ~500ms |
| Languages | Multilingual (best Hindi of all tested) |
| English quality | Good |
| Concurrent streams (max) | **4** (then OOM) |
| Total throughput at 4 streams | ~7× RTFx |

**Verdict:** Best multilingual quality (especially Hindi) but too slow and memory-hungry for production streaming. Only 4 concurrent streams on an L4.

### 3. Moonshine Medium (ONNX CPU streaming)

Useful Knowledge's Moonshine via `moonshine-voice` 0.0.49 with ONNX runtime.

| Metric | Value |
|--------|-------|
| Architecture | Encoder-decoder, ~82M params |
| VRAM | 510 MB (GPU batch) / **0 MB** (CPU ONNX streaming) |
| GPU batch RTFx | 15–23× single, **850×** at batch=768 |
| CPU streaming RTFx | 2.6–3.0× |
| CPU streaming TTFT | 110–165ms |
| Languages | **English only** |
| English quality | Good (6.65% WER on LibriSpeech) |
| Concurrent CPU streams | 2–3 real-time max |

**Verdict:** Excellent for English-only, CPU-only deployments. Best WER of all models tested (6.65%). But English-only and CPU streaming is limited to 2–3 concurrent streams.

### 4. SenseVoice Small (streaming-sensevoice wrapper)

Alibaba's CTC-based model via [pengzhendong/streaming-sensevoice](https://github.com/pengzhendong/streaming-sensevoice).

| Metric | Value |
|--------|-------|
| Architecture | CTC-based (non-autoregressive), SANM encoder, ~234M params |
| VRAM | 994 MB |
| Streaming RTFx (greedy) | **10.4×** |
| Streaming RTFx (beam=3) | 10.4× (CTC decoder is cheap) |
| Batch (non-streaming) RTFx | **74–183×** (blazing fast) |
| Streaming TTFT | **58ms** (fastest of all models) |
| Languages | zh, en, ja, ko, yue + emotion/event detection |
| Hindi | **Not supported** (outputs garbled characters) |
| English quality | Good (simple audio), moderate (complex names) |
| Concurrent streams (safe) | **3–4** real-time |

**Chunk size scaling (linear):** chunk=5 → 5.4×, chunk=10 → 10.2×, chunk=20 → 19.5×, chunk=40 → 35.8×

**Concurrent streaming:**

| Streams | RTFx/stream | Real-time? | Throughput | VRAM |
|---------|------------|-----------|------------|------|
| 1 | 10.3× | ✓ | 9.9× | 2,558 MB |
| 2 | 6.2× | ✓ | 11.8× | 2,622 MB |
| 3 | 3.7× | ✓ | 10.5× | 2,716 MB |
| 5 | 1.1× | Barely | 5.5× | 2,854 MB |
| 8 | 0.5× | ✗ | 4.3× | 3,128 MB |

**Verdict:** Fastest TTFT (58ms) and excellent batch throughput, but streaming is 6× slower than Whisper and only supports 3–4 concurrent streams. No Hindi.

### 5. Qwen3-ASR-0.6B (Transformers)

Qwen/Alibaba's encoder-decoder ASR model via `qwen-asr` 0.0.6. 52 languages + 22 Chinese dialects.

| Metric | Value |
|--------|-------|
| Architecture | Encoder-decoder (based on Qwen3-Omni), 0.6B params |
| VRAM | 1.8 GB (model), 5.8 GB (batch=256), 16.5 GB (batch=99 concurrent) |
| Single-stream RTFx (English) | **8–14×** |
| Single-stream RTFx (Hindi) | **3.1×** (autoregressive Hindi script generation) |
| Batch RTFx | **170×** at batch=32+ |
| Streaming | Not available (vLLM 0.16 incompatible with qwen-asr 0.0.6) |
| TTFT | N/A (batch-only, no streaming) |
| Languages | **52** + auto-detection |
| Hindi | **Excellent** — native Devanagari output, auto-detected |
| English quality | Very good |
| Concurrent RT streams (L4) | **99+** (batch mode) |

**Single-file results:**

| Audio | Duration | RTFx | Language |
|-------|----------|------|----------|
| simplest-short | 12.1s | 8.1× | English (auto) |
| simple | 29.6s | 13.7× | English (auto) |
| hard (names) | 32.2s | 10.4× | English (auto) |
| hindi | 16.1s | 3.1× | Hindi (auto) |

**Batch throughput scaling (simplest-short × N):**

| Batch | Time | RTFx | VRAM |
|-------|------|------|------|
| 1 | 1.48s | 8.2× | 2 GB |
| 8 | 1.63s | 59.5× | 4 GB |
| 32 | 2.28s | 170× | 4 GB |
| 64 | 4.89s | 158× | 6 GB |
| 128 | 9.17s | 169× | 6 GB |
| 256 | 18.3s | 169× | 6 GB |

**Concurrent real-time capacity (batch mode):**

| Streams | Batch Time | Audio Dur | Real-time? | VRAM |
|---------|-----------|-----------|------------|------|
| 30 (mixed 12–32s) | 8.1s | — | ✓ all | 16.5 GB |
| 32 (uniform 12.1s) | 2.2s | 12.1s | ✓ | 16.5 GB |
| 64 (uniform 12.1s) | 4.5s | 12.1s | ✓ | 16.5 GB |
| 96 (uniform 12.1s) | 6.6s | 12.1s | ✓ | 16.5 GB |
| 99 (uniform 12.1s) | 8.2s | 12.1s | ✓ | 16.5 GB |

**Verdict:** Remarkable batch throughput and concurrency — **99+ real-time streams on a single L4**. Excellent multilingual quality with native Hindi support. However, no streaming capability (incremental results) with current tooling. Best suited for batch transcription or non-streaming concurrent workloads.

### 6. Whisper Large-v3 Turbo (CPU)

Same model as #1 but running on CPU with int8 quantization.

| Metric | Value |
|--------|-------|
| VRAM | 0 (CPU-only) |
| RTFx (8 threads) | 1.7–2.2× |
| RTFx (4 threads) | 1.8× |
| RTFx (2 threads) | 1.0× (barely real-time) |
| RTFx (1 thread) | 0.5× (not real-time) |
| Quality | Identical to GPU |
| Concurrent streams | **1** (throughput flat at 1.8× regardless of concurrency) |

**CPU pinning test (8 vCPUs total):**

| Config | RTFx/instance | Real-time? | Total Throughput |
|--------|--------------|-----------|-----------------|
| 1 inst × 8 threads | 1.68× | 1/1 ✓ | 0.9× |
| 2 inst × 4 threads | 1.07× | 2/2 barely | 1.0× |
| 4 inst × 2 threads | 0.57× | 0/4 ✗ | 1.1× |
| 8 inst × 1 thread | 0.31× | 0/8 ✗ | 1.2× |

**Verdict:** Works for 1 stream. CPU is fully saturated — no concurrency possible. Only viable as a single-stream fallback or for offline batch processing.

### Final Comparison

| Metric | Whisper Turbo (GPU) | Qwen3-ASR-0.6B | Voxtral 4B | Moonshine Med | SenseVoice Small | Whisper (CPU) |
|--------|:------------------:|:--------------:|:----------:|:-------------:|:----------------:|:-------------:|
| **Streaming RTFx** | **64×** | N/A (batch only) | 1.7–1.9× | 2.6–3.0× | 10.4× | 1.7× |
| **Batch RTFx** | 70× | **170×** | N/A | 850× (batched) | 74–183× | 1.7× |
| **TTFT** | 1.4s | N/A | 500ms | 110ms | **58ms** | N/A |
| **Max RT streams (L4)** | 10–15 (streaming) | **99+** (batch) | 4 | 2–3 (CPU) | 3–4 | 1 |
| **VRAM** | 1.2 GB | 1.8 GB (16.5 GB peak) | 21.5 GB | 0 (CPU) | 1.0 GB | 0 |
| **English quality** | Very good | Very good | Good | Good (6.65% WER) | Good | Very good |
| **Hindi** | Partial | **Excellent** | **Excellent** | None | None | Partial |
| **Languages** | 99 | 52 | Many | English only | 5 | 99 |
| **Production ready** | **✓ (streaming)** | **✓ (batch)** | ✗ | ✗ | ✗ | ✗ |

### Why Whisper Turbo Wins (Streaming)

1. **6× faster streaming** than the next GPU competitor (SenseVoice at 10×)
2. **True incremental streaming** — word-level confirmation via LocalAgreement as audio arrives
3. **99 language support** with auto-detection — no model changes needed
4. **Mature ecosystem** — faster-whisper/CTranslate2, production-proven, actively maintained
5. **Only 1.2 GB VRAM** — leaves headroom for other workloads or burst capacity
6. **Total throughput stays high under concurrency** — 30 streams all succeed on L4

### When to Consider Qwen3-ASR-0.6B Instead

1. **99+ concurrent real-time streams** in batch mode on a single L4 (vs 10–15 for Whisper streaming)
2. **Excellent multilingual quality** — native Hindi Devanagari output, 52 languages with auto-detection
3. **170× batch throughput** — ideal for post-call transcription or high-volume batch workloads
4. **Small model** (0.6B params, 1.8 GB VRAM) — efficient and cost-effective
5. **Caveat:** No streaming support currently (vLLM backend required, incompatible with vLLM ≥0.16)

---

## Hardware Guide: Achieving 100% Real-time with Zero Transcript Loss

For production deployments requiring **100 concurrent real-time streams with zero quality degradation**, here are the hardware options ranked by cost-efficiency.

### Key Insight: Memory Bandwidth is the Bottleneck

Whisper turbo's autoregressive decoder is **memory-bandwidth bound**. The number of real-time streams a GPU can support scales linearly with memory bandwidth, not VRAM size or CUDA cores.

| GPU | Memory Bandwidth | Safe RT Streams | VRAM |
|-----|-----------------|----------------|------|
| T4 (g4dn) | 320 GB/s | ~8–10 | 16 GB |
| L4 (g6) | 300 GB/s | 10–15 | 24 GB |
| A10G (g5) | 600 GB/s | ~20 | 24 GB |
| L40S (g6e) | 864 GB/s | ~25–30 | 48 GB |
| A100 (p4d) | 2,039 GB/s | ~50–60 | 80 GB |

### Option 1: Single Instance — g6e.12xlarge (4× L40S) — Simplest

| Spec | Value |
|------|-------|
| GPUs | 4× NVIDIA L40S |
| Total bandwidth | 3,456 GB/s |
| Estimated capacity | **~100–120 concurrent RT streams** |
| VRAM | 192 GB total |
| vCPUs | 48 |
| Network | 100 Gbps |
| **On-demand price** | **$10.49/hr** |
| **Per-stream cost** | **$0.105/hr** |

**Pros:** Single machine, no load balancer, simplest operations, no networking overhead.
**Cons:** Single point of failure, ~50% more expensive per stream than fleet options.

**Architecture:** Run 4 Whisper server processes (one per GPU), local nginx round-robin.

```
            ┌──────────────────────────────────────────┐
            │         g6e.12xlarge (4× L40S)           │
            │                                          │
            │  nginx :8000 (round-robin WebSocket)     │
            │    ├── whisper-stt :8001  (GPU 0, ~25s)  │
            │    ├── whisper-stt :8002  (GPU 1, ~25s)  │
            │    ├── whisper-stt :8003  (GPU 2, ~25s)  │
            │    └── whisper-stt :8004  (GPU 3, ~25s)  │
            │                                          │
            └──────────────────────────────────────────┘
```

### Option 2: Fleet — 4× g6e.xlarge (4× L40S) — Best Value Large GPU

| Spec | Value |
|------|-------|
| Instances | 4× g6e.xlarge (1× L40S each) |
| Total bandwidth | 3,456 GB/s |
| Estimated capacity | **~100–120 concurrent RT streams** |
| **On-demand price** | **$7.44/hr** ($1.86 × 4) |
| **Per-stream cost** | **$0.074/hr** |

**Pros:** Fault tolerant (lose 1 = lose 25%), 30% cheaper than single instance.
**Cons:** Needs ALB + 4 instances to manage.

### Option 3: Fleet — 7× g5.xlarge (7× A10G) — Best Per-Stream Value

| Spec | Value |
|------|-------|
| Instances | 7× g5.xlarge (1× A10G each) |
| Total bandwidth | 4,200 GB/s |
| Estimated capacity | **~105 concurrent RT streams** |
| **On-demand price** | **$7.07/hr** ($1.01 × 7) |
| **Per-stream cost** | **$0.071/hr** |

**Pros:** Cheapest per-stream of GPU options, proven hardware, best fault tolerance.
**Cons:** 7 instances to manage, A10G (older Ampere architecture).

### Option 4: Fleet — 13× g4dn.xlarge (13× T4) — Cheapest Absolute

| Spec | Value |
|------|-------|
| Instances | 13× g4dn.xlarge (1× T4 each) |
| Total bandwidth | 4,160 GB/s |
| Estimated capacity | **~104 concurrent RT streams** |
| **On-demand price** | **$6.89/hr** ($0.53 × 13) |
| **Per-stream cost** | **$0.069/hr** |

**Pros:** Cheapest absolute cost.
**Cons:** 13 instances, T4 is older hardware (Turing), less headroom per GPU.

### Option 5: Single Instance — p4d.24xlarge (8× A100) — Maximum Headroom

| Spec | Value |
|------|-------|
| GPUs | 8× NVIDIA A100 (80 GB) |
| Total bandwidth | 16,312 GB/s |
| Estimated capacity | **~400+ concurrent RT streams** |
| **On-demand price** | **$32.77/hr** |
| **Per-stream cost** | **$0.082/hr** (at 400 streams) |

**Pros:** Massive headroom for growth, single machine.
**Cons:** Expensive absolute cost, single point of failure, overkill for 100 streams.

### Cost Comparison Summary (100 streams)

| Option | Config | $/hr | $/stream/hr | Fault Tolerance | Ops Complexity |
|--------|--------|------|-------------|-----------------|----------------|
| g4dn.xlarge fleet | 13× T4 | **$6.89** | **$0.069** | Best (8% per node) | High (13 nodes) |
| g5.xlarge fleet | 7× A10G | $7.07 | $0.071 | Good (14% per node) | Medium (7 nodes) |
| g6e.xlarge fleet | 4× L40S | $7.44 | $0.074 | Good (25% per node) | Low (4 nodes) |
| **g6e.12xlarge** | **4× L40S** | **$10.49** | **$0.105** | **None (SPOF)** | **Lowest (1 node)** |
| p4d.24xlarge | 8× A100 | $32.77 | $0.328 | None (SPOF) | Lowest (1 node) |

### Recommendation

| Priority | Best Choice |
|----------|-------------|
| **Simplest ops** | g6e.12xlarge — 1 machine, 4 GPUs, ~100 streams |
| **Best cost** | 7× g5.xlarge — $7.07/hr, proven A10G, good fault tolerance |
| **Cheapest absolute** | 13× g4dn.xlarge — $6.89/hr, but 13 instances |
| **Future growth** | g6e.xlarge fleet — add/remove instances, linear scaling |

### Fleet Architecture (Multi-Instance)

```
                        ┌─────────────────┐
                        │    AWS ALB       │
                        │   (WebSocket     │
                        │   sticky sess.)  │
                        └────────┬────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
   ┌──────┴──────┐       ┌──────┴──────┐       ┌──────┴──────┐
   │ g5.xlarge   │       │ g5.xlarge   │       │ g5.xlarge   │  ...×N
   │ A10G        │       │ A10G        │       │ A10G        │
   │ ~15 streams │       │ ~15 streams │       │ ~15 streams │
   │ whisper-stt │       │ whisper-stt │       │ whisper-stt │
   └─────────────┘       └─────────────┘       └─────────────┘
```

- **Load balancer:** AWS ALB with WebSocket sticky sessions (target: active connection count)
- **Each instance:** Runs this whisper-stt server with identical config
- **Auto-scaling:** Scale on `ActiveConnectionCount` per target — add instance when approaching safe stream limit
- **Health checks:** ALB uses `/healthz` endpoint
- **Savings Plans pricing** reduces fleet costs by 30–60% (e.g., g5.xlarge: $1.01 → $0.64/hr with 1yr no-upfront)

## Quick start

```bash
# 1. Create venv and install
python3 -m venv ~/dev/whisper-env
source ~/dev/whisper-env/bin/activate
pip install -r requirements.txt          # or requirements-lock.txt for exact versions

# 2. Copy config
cp .env.example .env                     # edit LD_LIBRARY_PATH if your CUDA path differs

# 3. Apply production OS tuning (one-time, requires sudo)
sudo bash scripts/os-tune.sh

# 4. Start server
./scripts/start.sh                       # background daemon (auto health-check)
./scripts/start.sh --foreground          # or foreground (Ctrl+C to stop)
```

## Operations

### Start / Stop / Status

```bash
./scripts/start.sh              # Start as background daemon with health-check
./scripts/start.sh -f           # Start in foreground (Ctrl+C to stop)
./scripts/stop.sh               # Graceful shutdown (SIGTERM → 10s → SIGKILL)
./scripts/stop.sh --force       # Immediate SIGKILL
./scripts/status.sh             # Full dashboard (process, health, GPU, metrics, logs)
./scripts/status.sh --watch     # Auto-refresh every 5 seconds
```

### Status dashboard

`./scripts/status.sh` shows:

```
═══ Whisper STT Server Status ═══

[Process]  PID, uptime, RAM, CPU
[Health]   /healthz endpoint, workers, model, active calls
[GPU]      NVIDIA L4 — temp, utilization, VRAM, power, persistence, perf state
[System]   RAM, load average, file descriptors
[Metrics]  Queue depth, finals/interims count, errors, avg inference latency, audio ingested
[Logs]     Last 5 log lines
```

### OS production tuning

`sudo bash scripts/os-tune.sh` applies (idempotent, safe to re-run):

| Tuning | Before | After |
|--------|--------|-------|
| GPU persistence mode | off (P8 idle) | Enabled (P0) |
| GPU clock lock | dynamic | Max (2040/6251 MHz) |
| vm.swappiness | 60 | 10 |
| net.core.somaxconn | 4096 | 8192 |
| TCP keepalive | 7200 s | 60 s |
| TCP fin timeout | 60 s | 15 s |
| Transparent Huge Pages | always | never |
| IRQ affinity | default | GPU pinned to CPUs 1-7 |
| File descriptors | 1024 | 65536 |

### Log rotation

Two layers of log rotation are configured:

| Layer | Mechanism | When | Details |
|-------|-----------|------|---------|
| **Built-in** | `start.sh` rotates on launch | Each `start.sh` invocation | Rotates if `server.log` > 10 MB, keeps 5 compressed backups |
| **System** | logrotate cron (daily) | System cron | Rotates if > 10 MB, keeps 14 days compressed, `copytruncate` |

```bash
# Install system logrotate config
sudo cp scripts/whisper-stt.logrotate /etc/logrotate.d/whisper-stt

# Dry-run test
sudo logrotate -d /etc/logrotate.d/whisper-stt
```

PID is tracked in `run/server.pid` (auto-created by start script).

### systemd (optional)

```bash
sudo cp scripts/whisper-stt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now whisper-stt

# Then use standard systemctl commands:
sudo systemctl status whisper-stt
sudo journalctl -u whisper-stt -f
```

## Test

```bash
# Stream a WAV file and print events
python scripts/test_with_file.py --wav-file /path/to/audio.wav --call-id test-1

# Pin language for the call (e.g. Hindi, Spanish, English)
python scripts/test_with_file.py --wav-file hindi.wav --language hi
python scripts/test_with_file.py --wav-file spanish.wav --language es

# Omit --language for automatic language detection
python scripts/test_with_file.py --wav-file any_language.wav

# Synthetic audio demo (verifies start/audio/ping/stop flow)
python scripts/ws_client_demo.py --url ws://127.0.0.1:8000 --call-id demo-1 --channels 2 --duration 8
```

## WebSocket protocol

### Endpoint

`ws://<host>:8000/ws/call/{call_id}`

Optional auth: set `API_KEY` in `.env`, then send `x-api-key` header or `?api_key=` query param.

### 1. Start (required first message)

```json
{
  "type": "start",
  "sample_rate": 16000,
  "channels": 2,
  "left_speaker": "agent",
  "right_speaker": "customer",
  "language": "hi"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `type` | yes | Must be `"start"` |
| `sample_rate` | yes | Must be `16000` |
| `channels` | yes | `1` (mono) or `2` (stereo) |
| `speaker` | mono | Speaker label for mono |
| `left_speaker` / `right_speaker` | stereo | Speaker labels for stereo |
| `language` | no | ISO 639-1 code (`en`, `hi`, `es`, …). Omit for auto-detect. |

### 2. Stream audio

Send binary frames of **16 kHz, 16-bit signed PCM little-endian** (20–40 ms chunks recommended).  
Stereo: interleaved L/R samples. Mono: set `channels: 1`.

### 3. Control events

| Client sends | Server responds |
|-------------|----------------|
| `{"type":"ping"}` | `{"type":"pong","call_id":"..."}` |
| `{"type":"flush"}` | Flushes pending audio as finals |
| `{"type":"stop"}` | Emits final `stopped` event, cleans up |

### 4. Server events

```jsonc
// Connection established
{"type":"started","call_id":"call-123","channel_map":[{"channel_id":0,"speaker":"agent","channel_name":"left"},{"channel_id":1,"speaker":"customer","channel_name":"right"}]}

// Real-time partial transcript
{"type":"interim","call_id":"call-123","speaker":"agent","channel_id":0,"text":"hello thank you for","ts":1760000000.11}

// Committed utterance
{"type":"final","call_id":"call-123","speaker":"customer","channel_id":1,"text":"Hi, I need help with my order status.","ts":1760000001.04}

// Session ended
{"type":"stopped","call_id":"call-123"}
```

## Configuration

All settings load from `.env` via `python-dotenv` (no manual `source` / `export` needed).  
See [.env.example](.env.example) for the full annotated list. Key tuning knobs:

| Variable | Default | Purpose |
|----------|---------|---------|
| `WHISPER_MODEL` | `turbo` | Model size (turbo recommended for streaming) |
| `WHISPER_BEAM_SIZE` | `5` | Beam search width (higher = better quality, slower) |
| `WHISPER_LANGUAGE` | _(empty)_ | Server default language. Empty = auto-detect. Per-call `language` in start event overrides this. |
| `VAD_MODE` | `3` | webrtcvad aggressiveness 0–3 (3 = most aggressive) |
| `VAD_SILENCE_SECONDS` | `0.35` | Silence duration to end an utterance |
| `MAX_UTTERANCE_DURATION_SEC` | `6.0` | Force-split long utterances |
| `WHISPER_REPETITION_PENALTY` | `1.15` | Penalise repeated tokens |
| `WHISPER_NO_REPEAT_NGRAM_SIZE` | `3` | Block trigram repetitions |
| `WHISPER_PATIENCE` | `1.5` | Beam search patience factor |
| `WHISPER_WORD_TIMESTAMPS` | `true` | Required for hallucination silence detection |
| `WHISPER_HALLUCINATION_SILENCE_THRESHOLD` | `1.0` | Skip hallucinated silence segments |
| `WHISPER_HOTWORDS` | *(empty)* | Comma-separated domain terms (e.g. `Benoni,mitkadmim`) |
| `HALLUCINATION_FILTER` | `true` | Pattern-based hallucination rejection |
| `MAX_CONCURRENT_SESSIONS` | `100` | Guard against resource exhaustion |
| `API_KEY` | *(empty)* | WebSocket auth (disabled when empty) |

## Monitoring

| Endpoint | Format |
|----------|--------|
| `GET /healthz` | JSON — worker status, model loaded, active calls |
| `GET /health` | Alias for `/healthz` |
| `GET /metrics` | Prometheus — queue size, inference latency, transcript events, errors |

## Project structure

```
whisper-stt/
├── .env                    # Production config (git-ignored)
├── .env.example            # Annotated config template
├── requirements.txt        # Minimum version constraints
├── requirements-lock.txt   # Pinned production versions
├── run/                    # PID file (server.pid) — git-ignored
├── logs/                   # Server logs — git-ignored
├── app/
│   ├── __init__.py
│   ├── config.py           # Settings dataclass + .env auto-loading
│   ├── main.py             # FastAPI app, WebSocket handler, PCM decoding
│   ├── sessions.py         # Session state, SpeechGate (VAD), utterance FSM
│   ├── transcriber.py      # GPU worker, inference, hallucination filter, dedup
│   └── metrics.py          # Prometheus counters / histograms
└── scripts/
    ├── start.sh                # Start server (daemon or foreground)
    ├── stop.sh                 # Graceful / force stop
    ├── status.sh               # Full monitoring dashboard
    ├── os-tune.sh              # One-time production OS tuning (sudo)
    ├── whisper-stt.service     # systemd unit file
    ├── whisper-stt.logrotate   # System logrotate config
    ├── test_with_file.py       # Stream WAV file over WebSocket
    ├── test_stereo_separation.py
    └── ws_client_demo.py       # Synthetic audio demo client
```

## Design decisions

1. **Utterance-based segmentation** — VAD detects speech→silence transitions to form natural utterances, rather than fixed-window chunking. Produces cleaner finals.
2. **`condition_on_previous_text=False`** — Disabled to prevent Whisper decoder repetition loops. Context is instead passed via `initial_prompt` from per-utterance confirmed text.
3. **`turbo` over `large-v3`** — Identical word accuracy in testing, but turbo is 3–5× faster. Critical for streaming latency.
4. **Hallucination defence-in-depth** — 30+ pattern blacklist + `hallucination_silence_threshold` + compression ratio filter + log probability filter + repetition dedup.
5. **Prefix stripping** — As utterance audio grows and is re-transcribed, already-confirmed text is stripped to avoid duplicate finals.
6. **CUDA env vars in `.env`** — `CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH`, etc. loaded automatically via `python-dotenv`, eliminating manual shell exports.
