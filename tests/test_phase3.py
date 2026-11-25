"""Phase 3 tests for audit ledger, INT8 ONNX latency, abstention gate, REST, and WebSocket."""

from __future__ import annotations

import io
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

from app.inference import SpectraGuardEngine, apply_abstention_gate
from app.main import create_app
from ml.model import compute_log_mel
from ml.train_export import _generate_genuine_clip, _generate_spoof_clip


@pytest.fixture()
def isolated_app_and_db(tmp_path: Path):
    db_path = tmp_path / "test_audit.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    application = create_app(database_url=db_url)
    return application, db_path


def test_abstention_gate_thresholds() -> None:
    assert apply_abstention_gate(0.15) == "AUTHENTIC"
    assert apply_abstention_gate(0.399) == "AUTHENTIC"
    assert apply_abstention_gate(0.40) == "STEP_UP_OTP"
    assert apply_abstention_gate(0.50) == "STEP_UP_OTP"
    assert apply_abstention_gate(0.60) == "STEP_UP_OTP"
    assert apply_abstention_gate(0.601) == "SPOOF_DETECTED"
    assert apply_abstention_gate(0.95) == "SPOOF_DETECTED"


def test_verify_audio_genuine_vs_spoof_and_watermark(isolated_app_and_db) -> None:
    application, _ = isolated_app_and_db
    rng = np.random.default_rng(101)
    genuine = _generate_genuine_clip(rng)
    spoof = _generate_spoof_clip(rng)

    with TestClient(application) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["audit_chain"]["chain_intact"] is True

        # Genuine clip with watermark embedding + Grad-CAM disabled
        resp_gen = client.post(
            "/v1/verify-audio",
            json={
                "samples": genuine.tolist(),
                "embed_watermark": True,
                "watermark_hex": "deadbeef",
                "include_gradcam": False,
            },
        )
        assert resp_gen.status_code == 200
        data_gen = resp_gen.json()
        assert data_gen["verdict"] == "AUTHENTIC"
        assert data_gen["spoof_probability"] < 0.40
        assert data_gen["watermark_hex"] == "deadbeef"
        assert data_gen["embed_snr_db"] > 32.0
        assert data_gen["watermark_confidence"] > 0.0

        # Spoof clip with Grad-CAM enabled
        resp_spoof = client.post(
            "/v1/verify-audio",
            json={
                "samples": spoof.tolist(),
                "embed_watermark": False,
                "include_gradcam": True,
            },
        )
        assert resp_spoof.status_code == 200
        data_spoof = resp_spoof.json()
        assert data_spoof["verdict"] == "SPOOF_DETECTED"
        assert data_spoof["spoof_probability"] > 0.60
        assert "gradcam" in data_spoof
        assert len(data_spoof["gradcam"]["heatmap"]) == 64


def test_verify_wav_upload(isolated_app_and_db) -> None:
    application, _ = isolated_app_and_db
    rng = np.random.default_rng(202)
    spoof = _generate_spoof_clip(rng)
    pcm16 = (np.clip(spoof, -1.0, 1.0) * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, 16000, pcm16)
    buf.seek(0)

    with TestClient(application) as client:
        resp = client.post(
            "/v1/verify-wav",
            files={"file": ("spoof.wav", buf.read(), "audio/wav")},
            data={"embed_watermark": "false", "include_gradcam": "false"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["verdict"] == "SPOOF_DETECTED"
        assert body["spoof_probability"] > 0.60


def test_onnx_int8_50_consecutive_p99_latency() -> None:
    engine = SpectraGuardEngine(ledger=None)
    rng = np.random.default_rng(303)
    clip = _generate_genuine_clip(rng)
    log_mel_np = compute_log_mel(clip).numpy().astype(np.float32)

    # Warmup pass
    _ = engine.predict_onnx(log_mel_np)

    latencies_ms: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = engine.predict_onnx(log_mel_np)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p99_ms = float(np.percentile(latencies_ms, 99))
    assert p99_ms < 25.0, f"Expected p99 INT8 ONNX latency < 25.0 ms, got {p99_ms:.2f} ms"


def test_websocket_ring_buffer_streaming(isolated_app_and_db) -> None:
    application, _ = isolated_app_and_db
    rng = np.random.default_rng(404)
    spoof = _generate_spoof_clip(rng)
    chunk_size = 8000  # 4 chunks of 0.5s -> trigger at 32,000 samples and hop every 16,000 samples

    with TestClient(application) as client:
        with client.websocket_connect("/v1/ws/stream") as ws:
            # First 32,000 samples (4 x 8,000) triggers initial 2s window verdict
            for i in range(4):
                seg = spoof[i * chunk_size : (i + 1) * chunk_size]
                ws.send_json({"samples": seg.tolist()})
            first_verdict = ws.receive_json()
            assert first_verdict["verdict"] == "SPOOF_DETECTED"
            assert first_verdict["spoof_probability"] > 0.60

            # Next 16,000 samples (2 x 8,000) triggers second hop verdict
            for i in range(2):
                seg = spoof[i * chunk_size : (i + 1) * chunk_size]
                ws.send_json({"samples": seg.tolist()})
            second_verdict = ws.receive_json()
            assert second_verdict["verdict"] == "SPOOF_DETECTED"


def test_audit_ledger_tamper_detection(isolated_app_and_db) -> None:
    application, db_path = isolated_app_and_db
    rng = np.random.default_rng(505)
    genuine = _generate_genuine_clip(rng)
    spoof = _generate_spoof_clip(rng)

    with TestClient(application) as client:
        client.post("/v1/verify-audio", json={"samples": genuine.tolist()})
        client.post("/v1/verify-audio", json={"samples": spoof.tolist()})
        client.post("/v1/verify-audio", json={"samples": genuine.tolist()})

        logs_resp = client.get("/v1/audit/logs?limit=10")
        assert logs_resp.status_code == 200
        records = logs_resp.json()["records"]
        assert len(records) == 3

        ok_verify = client.get("/v1/audit/verify").json()
        assert ok_verify == {"chain_intact": True, "total_records": 3, "corrupted_id": None}

        # Directly tamper with row id=2 in SQLite
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("UPDATE audit_ledger SET verdict = 'AUTHENTIC' WHERE id = 2")
            conn.commit()
        finally:
            conn.close()

        tampered_verify = client.get("/v1/audit/verify").json()
        assert tampered_verify["chain_intact"] is False
        assert tampered_verify["total_records"] == 3
        assert tampered_verify["corrupted_id"] == 2
