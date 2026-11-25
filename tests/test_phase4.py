"""Phase 4 tests for dashboard serving, demo signals, container/Vercel config, CI, and docs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.audit_db import ProvenanceLedger
from app.inference import _compute_numpy_saliency, compute_log_mel_numpy
from app.main import create_app
from ml.signals import _generate_spoof_clip


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
    assert "python -m ml.train_export" in dockerfile
    assert "uvicorn" in dockerfile

    compose = (ROOT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
    assert "spectraguard-api" in compose
    assert "postgres:16-alpine" in compose
    assert "DATABASE_URL" in compose
    assert "healthcheck:" in compose

    ci_yml = (ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python -m ml.train_export" in ci_yml
    assert "pytest -v" in ci_yml

    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    assert "SpectraGuard" in readme
    assert "ROC-AUC" in readme
    assert "INT8" in readme

    req_vercel = (ROOT_DIR / "requirements-vercel.txt").read_text(encoding="utf-8")
    assert "onnxruntime" in req_vercel
    assert "torch" not in req_vercel

    vercel_cfg = json.loads((ROOT_DIR / "vercel.json").read_text(encoding="utf-8"))
    assert vercel_cfg["installCommand"] == "pip install -r requirements-vercel.txt"
    assert vercel_cfg["functions"]["app/main.py"]["maxDuration"] == 60


def test_vercel_fallbacks_work_without_torch(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    # Avoid creating /tmp db on Windows by checking url selection cleanly if not writable,
    # or verifying on posix-style /tmp path via monkeypatched create_engine
    captured_urls: list[str] = []

    import app.audit_db as audit_db_mod

    orig_create_engine = audit_db_mod.create_engine

    def _fake_create_engine(url, **kwargs):
        captured_urls.append(url)
        return orig_create_engine("sqlite:///:memory:", **kwargs)

    monkeypatch.setattr(audit_db_mod, "create_engine", _fake_create_engine)
    _ = ProvenanceLedger()
    assert captured_urls == ["sqlite:////tmp/spectraguard_audit.db"]

    # Verify torch-free Log-Mel + spectral saliency fallback
    rng = np.random.default_rng(77)
    spoof = _generate_spoof_clip(rng)
    lm_np = compute_log_mel_numpy(spoof)
    assert lm_np.shape == (1, 1, 64, 126)
    heatmap, summary = _compute_numpy_saliency(lm_np, 0.99)
    assert heatmap.shape == (64, 126)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
    assert summary["saliency_mode"] == "spectral_energy_fallback"
