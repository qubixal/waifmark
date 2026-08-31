"""Smoke tests for the FastAPI replacement."""

from fastapi.testclient import TestClient

from api.app import app


def _client():
    return TestClient(app)


def test_health():
    c = _client()
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_config_roundtrip():
    c = _client()
    orig = c.get("/api/config").json()
    assert "config" in orig and "raw_yaml" in orig
    # PUT same should succeed
    r = c.put("/api/config", json={"yaml_text": orig["raw_yaml"]})
    assert r.status_code == 200


def test_server_status_shape():
    c = _client()
    r = c.get("/api/server/status")
    assert r.status_code == 200
    j = r.json()
    assert "status" in j and "ready" in j


def test_benchmark_status_shape():
    c = _client()
    r = c.get("/api/benchmark/status")
    assert r.status_code == 200
    j = r.json()
    assert "status" in j


def test_runs_and_leaderboard():
    c = _client()
    assert c.get("/api/runs").status_code == 200
    assert c.get("/api/leaderboard").status_code == 200
    assert c.get("/api/audit/flagged").status_code == 200


def test_frontend_serves():
    c = _client()
    r = c.get("/")
    assert r.status_code == 200
    assert "Waifmark" in r.text
