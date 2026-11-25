# SpectraGuard

**Real-Time Audio Provenance & Adversarial Deepfake Detection Engine**

SpectraGuard combines **Active DSP Watermarking** (`1.5 kHz – 4.0 kHz` speech-masked Hadamard spread-spectrum signatures) with **Passive Adversarial Deepfake Detection** (a Temperature-Scaled CNN-Conformer quantized to **INT8 ONNX Runtime**), **Time-Frequency Grad-CAM** attribution, a calibrated **3-way abstention gate**, and a **SHA-256 hash-chained audit ledger**.

---

## System Architecture

```text
Incoming 16 kHz Audio (REST / WAV / WS Ring-Buffer 32,000 samples @ 16,000 hop)
   │
   ├──► [Active Provenance] SpreadSpectrumWatermarker (1.5–4.0 kHz, 32-bit, seed=42)
   │       └─► Recovered 32-bit Hex Payload + Correlation Confidence
   │
   ├──► [Passive Detector] 64-bin Log-Mel (n_fft=512, hop=256)
   │       ├─► INT8 ONNX Runtime (SpectraGuardNet: 3x Conv2d + 2-layer Transformer)
   │       │      └─► Calibrated Spoof Probability (Temperature Scaling T=0.7867)
   │       │             ├─► p < 0.40       ──► AUTHENTIC
   │       │             ├─► 0.40 <= p <= 0.60 ──► STEP_UP_OTP
   │       │             └─► p > 0.60       ──► SPOOF_DETECTED
   │       └─► On-Demand Time-Frequency Grad-CAM ([64 x T] Saliency Heatmap)
   │
   └──► [Tamper-Evident Ledger] SQLAlchemy + SHA-256 Hash-Chained Audit DB
           └─► row_hash = SHA256(canonical_json(fields + prev_hash))
```

---

## Verified Benchmarks

All metrics below were directly measured by running `python -m ml.train_export` and `pytest -v` in `.venv` (`CPUExecutionProvider`):

| Subsystem | Metric | Measured Result | Gate |
| :--- | :--- | :--- | :--- |
| **Active Watermark** | Embedding SNR | `33.00 dB` | `> 32.0 dB` |
| **Active Watermark** | Clean Bit Recovery | `100%` (`32/32` bits) | `100%` |
| **Active Watermark** | Telephony (`300–3400 Hz`) & `20 dB` AWGN | `>= 90%` (`32/32` on benchmark clip) | `>= 90%` |
| **CNN-Conformer** | Held-Out Validation ROC-AUC | `1.0000` | `> 0.92` |
| **CNN-Conformer** | Equal Error Rate (EER) | `0.00%` | `< 10.0%` |
| **Calibration** | ECE (Before → After Temp Scaling) | `0.0064` → `0.0039` (`T = 0.7867`) | `< 0.08` |
| **Quantization** | FP32 → INT8 ONNX Size | `408.6 KB` → `185.0 KB` (`54.7%` smaller) | `< FP32` |
| **Quantization** | Max Probability Drift (PT vs. INT8 ORT) | `0.00106` | `< 0.08` |
| **Runtime Speed** | INT8 ONNX Latency (50 consecutive calls) | `3.01 ms` mean / `2.61 ms` p50 / `10.44 ms` p99 | `p99 < 25.0 ms` |

---

## Quickstart

### 1. Local Python Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
python -m ml.train_export
pytest -v
python main.py
```

Open `http://localhost:8000` for the interactive SpectraGuard dashboard.

### 2. Docker Compose (FastAPI + PostgreSQL 16)

```bash
docker compose up --build
```

The API and dashboard become available at `http://localhost:8000`, backed by `postgres:16-alpine` via `DATABASE_URL`.

---

## API Reference

- `GET /` — Dark-themed interactive verification, Grad-CAM canvas, WebSocket streamer & audit ledger UI
- `GET /health` — Service health, ONNX provider, and audit chain status
- `POST /v1/verify-audio` — Verify JSON `float32` PCM samples (`samples`, `embed_watermark`, `watermark_hex`, `include_gradcam`)
- `POST /v1/verify-wav` — Multipart `.wav` upload resampled/converted to 16 kHz mono
- `WS /v1/ws/stream` — Real-time 2-second (`32,000`-sample) sliding ring-buffer stream emitting verdicts every `16,000`-sample hop
- `GET /v1/audit/verify` — Walk and verify the full SHA-256 hash chain
- `GET /v1/audit/logs?limit=25` — Fetch recent hash-chained audit records
