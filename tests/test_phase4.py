"""Phase 4 tests for dashboard serving, demo signals, container/Vercel config, CI, and pure-NumPy fallback."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.audit_db import ProvenanceLedger
from app.inference import SpectraGuardEngine, _compute_numpy_saliency, compute_log_mel_numpy
from app.main import create_app
from ml.signals import _generate_genuine_clip, _generate_spoof_clip


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_dashboard_route_serves_html(tmp_path: Path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'phase4.db').as_posix()}"
    application = create_app(database_url=db_url)

    with TestClient(application) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        html = resp.text
        assert "SpectraGuard" in html
        assert "gradcamCanvas" in html
        assert "/v1/ws/stream" in html
        assert "innerHTML" not in html

        demo_resp = client.get("/v1/demo-signal?kind=spoof&corrupt=true")
        assert demo_resp.status_code == 200
        demo_data = demo_resp.json()
        assert demo_data["kind"] == "spoof"
        assert demo_data["corrupt"] is True
        assert len(demo_data["samples"]) == 32000


def test_deployment_and_ci_files_exist_and_are_valid() -> None:
    dockerignore = (ROOT_DIR / ".dockerignore").read_text(encoding="utf-8")
    assert ".venv" in dockerignore
    assert "*.db" in dockerignore

    dockerfile = (ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile
    assert "requirements-dev.txt" in dockerfile
    assert "python -m ml.train_export" in dockerfile
    assert "uvicorn" in dockerfile

    compose = (ROOT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
    assert "spectraguard-api" in compose
    assert "postgres:16-alpine" in compose
    assert "DATABASE_URL" in compose
    assert "healthcheck:" in compose

    ci_yml = (ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "requirements-dev.txt" in ci_yml
    assert "python -m ml.train_export" in ci_yml
    assert "pytest -v" in ci_yml

    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    assert "SpectraGuard" in readme
    assert "ROC-AUC" in readme
    assert "INT8" in readme

    # Slim root requirements.txt for Vercel (<60 MB)
    req_slim = (ROOT_DIR / "requirements.txt").read_text(encoding="utf-8")
    for heavy in ("torch", "torchaudio", "onnxruntime", "scipy", "sqlalchemy"):
        assert heavy not in req_slim
    assert "numpy" in req_slim
    assert "fastapi" in req_slim

    req_dev = (ROOT_DIR / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "torch" in req_dev
    assert "onnxruntime" in req_dev
    assert "scipy" in req_dev

    assert not (ROOT_DIR / "requirements-vercel.txt").exists()

    vercel_cfg = json.loads((ROOT_DIR / "vercel.json").read_text(encoding="utf-8"))
    assert "installCommand" not in vercel_cfg
    assert "builds" not in vercel_cfg
    assert vercel_cfg["functions"]["app/main.py"]["maxDuration"] == 60
    assert (ROOT_DIR / "artifacts" / "spectraguard_weights.npz").exists()


def test_numpy_backend_parity_and_sqlite3_fallback(tmp_path: Path, monkeypatch) -> None:
    # 1. Verify VERCEL=1 /tmp path + stdlib sqlite3 hash chain
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    db_file = tmp_path / "stdlib_audit.db"
    ledger = ProvenanceLedger(database_url=f"sqlite:///{db_file.as_posix()}", force_sqlite3=True)
    rec1 = ledger.append_event(
        audio_sha256="a" * 64,
        verdict="AUTHENTIC",
        spoof_prob=0.01,
        watermark_hex="53504543",
        watermark_confidence=0.95,
        inference_latency_ms=2.5,
    )
    rec2 = ledger.append_event(
        audio_sha256="b" * 64,
        verdict="SPOOF_DETECTED",
        spoof_prob=0.99,
        watermark_hex="00000000",
        watermark_confidence=0.1,
        inference_latency_ms=2.8,
    )
    assert rec2["prev_hash"] == rec1["row_hash"]
    assert ledger.verify_chain() == {"chain_intact": True, "total_records": 2, "corrupted_id": None}
    assert len(ledger.get_recent_logs(limit=10)) == 2

    # 2. Verify pure-NumPy engine matches ONNX verdicts & runs <5 ms
    onnx_engine = SpectraGuardEngine(ledger=None, use_numpy_backend=False)
    numpy_engine = SpectraGuardEngine(ledger=None, use_numpy_backend=True)
    assert numpy_engine.onnx_provider == "NumPyPureEngine"

    rng = np.random.default_rng(77)
    genuine = _generate_genuine_clip(rng)
    spoof = _generate_spoof_clip(rng)

    res_g_ort = onnx_engine.analyze(genuine, log_to_ledger=False)
    res_g_np = numpy_engine.analyze(genuine, log_to_ledger=False)
    res_s_ort = onnx_engine.analyze(spoof, log_to_ledger=False)
    res_s_np = numpy_engine.analyze(spoof, log_to_ledger=False)

    assert res_g_np["verdict"] == res_g_ort["verdict"] == "AUTHENTIC"
    assert res_s_np["verdict"] == res_s_ort["verdict"] == "SPOOF_DETECTED"
    assert abs(res_g_np["spoof_probability"] - res_g_ort["spoof_probability"]) < 1e-2
    assert abs(res_s_np["spoof_probability"] - res_s_ort["spoof_probability"]) < 1e-2

    lm_np = compute_log_mel_numpy(spoof)
    _ = numpy_engine.predict_onnx(lm_np)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = numpy_engine.predict_onnx(lm_np)
    mean_ms = ((time.perf_counter() - t0) * 1000.0) / 20.0
    assert mean_ms < 5.0, f"Expected pure-NumPy forward pass < 5.0 ms, got {mean_ms:.2f} ms"

    heatmap, summary = _compute_numpy_saliency(lm_np, res_s_np["spoof_probability"])
    assert heatmap.shape == (64, 126)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
    assert summary["saliency_mode"] == "spectral_energy_fallback"
