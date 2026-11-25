"""Phase 4 tests for dashboard serving, demo signals, container config, CI, and docs."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


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
