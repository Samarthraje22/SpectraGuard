"""FastAPI REST and WebSocket ring-buffer streaming server for SpectraGuard."""

from __future__ import annotations

import io
import math
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy import signal
from scipy.io import wavfile

from app.audit_db import ProvenanceLedger
from app.inference import SpectraGuardEngine
from ml.corruptions import add_gaussian_noise, telephony_bandpass
from ml.train_export import _generate_genuine_clip, _generate_spoof_clip


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


TARGET_SAMPLE_RATE = 16000
WINDOW_SAMPLES = 32000
HOP_SAMPLES = 16000


class VerifyAudioRequest(BaseModel):
    """JSON payload for /v1/verify-audio."""

    samples: list[float] = Field(..., min_length=1)
    embed_watermark: bool = False
    watermark_hex: str | None = None
    include_gradcam: bool = False


def _decode_wav_to_16k_mono(raw_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes into 16 kHz mono float32 waveform in [-1.0, 1.0]."""
    try:
        sr, data = wavfile.read(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {exc}") from exc

    if data.size == 0:
        raise HTTPException(status_code=400, detail="WAV file contains no audio samples")

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        # Handle unsigned 8-bit PCM ([0, 255]) vs signed integer PCM
        if data.dtype == np.uint8:
            audio = (data.astype(np.float32) - 128.0) / 128.0
        else:
            scale = float(max(abs(info.min), abs(info.max)))
            audio = data.astype(np.float32) / scale
    else:
        audio = data.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise HTTPException(status_code=400, detail=f"Unsupported WAV shape: {audio.shape}")

    if sr != TARGET_SAMPLE_RATE:
        g = math.gcd(int(sr), TARGET_SAMPLE_RATE)
        audio = signal.resample_poly(audio, TARGET_SAMPLE_RATE // g, int(sr) // g).astype(np.float32)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def create_app(
    database_url: str | None = None,
    onnx_path: str = "artifacts/spectraguard_int8.onnx",
    pt_path: str = "artifacts/spectraguard.pt",
) -> FastAPI:
    """Application factory creating a configured FastAPI instance."""
    ledger = ProvenanceLedger(database_url=database_url)
    engine = SpectraGuardEngine(onnx_path=onnx_path, pt_path=pt_path, ledger=ledger)

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        fastapi_app.state.ledger = ledger
        fastapi_app.state.engine = engine
        yield

    application = FastAPI(
        title="SpectraGuard API",
        description="Real-Time Audio Provenance & Adversarial Deepfake Detection Engine",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.ledger = ledger
    application.state.engine = engine

    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard UI not found")
        return FileResponse(index_path)

    @application.get("/v1/demo-signal")
    def demo_signal(
        kind: str = Query("genuine", pattern="^(genuine|spoof)$"),
        corrupt: bool = Query(False),
        seed: int = Query(42),
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        clip = _generate_spoof_clip(rng) if kind == "spoof" else _generate_genuine_clip(rng)
        if corrupt:
            clip = add_gaussian_noise(telephony_bandpass(clip, low_hz=300.0, high_hz=3400.0), snr_db=20.0, seed=seed)
        return {
            "kind": kind,
            "corrupt": corrupt,
            "sample_rate": TARGET_SAMPLE_RATE,
            "samples": clip.tolist(),
        }

    @application.get("/health")
    def health() -> dict[str, Any]:
        chain_status = application.state.ledger.verify_chain()
        return {
            "status": "ok",
            "model_loaded": True,
            "onnx_path": str(application.state.engine.onnx_path),
            "onnx_provider": application.state.engine.onnx_provider,
            "audit_chain": chain_status,
        }

    @application.post("/v1/verify-audio")
    def verify_audio(payload: VerifyAudioRequest) -> dict[str, Any]:
        samples = np.asarray(payload.samples, dtype=np.float32)
        try:
            return application.state.engine.analyze(
                samples,
                embed_watermark=payload.embed_watermark,
                watermark_hex=payload.watermark_hex,
                include_gradcam=payload.include_gradcam,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.post("/v1/verify-wav")
    async def verify_wav(
        file: UploadFile = File(...),
        embed_watermark: bool = Form(False),
        watermark_hex: str | None = Form(None),
        include_gradcam: bool = Form(False),
    ) -> dict[str, Any]:
        raw = await file.read()
        audio = _decode_wav_to_16k_mono(raw)
        try:
            return application.state.engine.analyze(
                audio,
                embed_watermark=embed_watermark,
                watermark_hex=watermark_hex,
                include_gradcam=include_gradcam,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.websocket("/v1/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        ring_buffer: deque[float] = deque(maxlen=WINDOW_SAMPLES)
        samples_since_eval = 0
        first_eval_done = False

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                embed_wm = False
                wm_hex: str | None = None
                if "bytes" in message and message["bytes"] is not None:
                    chunk = np.frombuffer(message["bytes"], dtype=np.float32)
                elif "text" in message and message["text"] is not None:
                    import json

                    data = json.loads(message["text"])
                    chunk = np.asarray(data.get("samples", []), dtype=np.float32)
                    embed_wm = bool(data.get("embed_watermark", False))
                    wm_hex = data.get("watermark_hex")
                else:
                    continue

                if chunk.size == 0:
                    continue

                ring_buffer.extend(chunk.tolist())
                samples_since_eval += int(chunk.size)

                if len(ring_buffer) == WINDOW_SAMPLES and (
                    not first_eval_done or samples_since_eval >= HOP_SAMPLES
                ):
                    first_eval_done = True
                    samples_since_eval = 0
                    window = np.array(ring_buffer, dtype=np.float32)
                    result = application.state.engine.analyze(
                        window,
                        embed_watermark=embed_wm,
                        watermark_hex=wm_hex,
                        include_gradcam=False,
                    )
                    await websocket.send_json(result)
        except WebSocketDisconnect:
            return

    @application.get("/v1/audit/verify")
    def verify_audit_chain() -> dict[str, Any]:
        return application.state.ledger.verify_chain()

    @application.get("/v1/audit/logs")
    def get_audit_logs(limit: int = Query(25, ge=1, le=500)) -> dict[str, Any]:
        return {"records": application.state.ledger.get_recent_logs(limit=limit)}

    return application


app = create_app()
