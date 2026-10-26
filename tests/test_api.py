"""API 테스트: 가짜 CIC 데이터로 작은 모델을 만들어 MODEL_PATH 로 주입한다."""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NORMAL = {"flow_duration_us": 2500000, "fwd_packets": 30, "bwd_packets": 28, "fwd_bytes": 4800, "bwd_bytes": 9600,
          "fwd_pkt_len_mean": 160, "bwd_pkt_len_mean": 342, "flow_bytes_per_s": 5760, "flow_pkts_per_s": 23.2,
          "flow_iat_mean_us": 43860, "fwd_iat_mean_us": 86200, "syn_flags": 0, "rst_flags": 0, "psh_flags": 1,
          "ack_flags": 1, "init_fwd_win_bytes": 65535, "init_bwd_win_bytes": 0, "fwd_seg_size_min": 20,
          "down_up_ratio": 0}


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    csv = d / "sample.csv"
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_sample.py"), str(csv)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "model.py"), "--csv", str(csv), "--out-dir", str(d)],
                   check=True, cwd=ROOT, capture_output=True)
    os.environ["MODEL_PATH"] = str(d / "model.joblib")
    import api
    importlib.reload(api)
    return TestClient(api.app), api


def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok"}


def test_ready_lists_features(client):
    c, _ = client
    r = c.get("/ready").json()
    assert r["ready"] is True and len(r["features"]) == 19


def test_predict_returns_decision(client):
    c, _ = client
    r = c.post("/predict", json=NORMAL)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in ("allow", "block") and 0 <= body["attack_probability"] <= 1


def test_negative_value_rejected(client):
    c, _ = client
    assert c.post("/predict", json={**NORMAL, "fwd_packets": -3}).status_code == 422


def test_window_minus_one_allowed(client):
    c, _ = client
    assert c.post("/predict", json={**NORMAL, "init_fwd_win_bytes": -1}).status_code == 200


def test_missing_field_rejected(client):
    c, _ = client
    body = dict(NORMAL)
    body.pop("syn_flags")
    assert c.post("/predict", json=body).status_code == 422


def test_string_number_rejected_or_coerced_consistently(client):
    c, _ = client
    assert c.post("/predict", json={**NORMAL, "fwd_packets": "abc"}).status_code == 422


def test_batch(client):
    c, _ = client
    r = c.post("/predict/batch", json={"flows": [NORMAL, NORMAL]}).json()
    assert r["count"] == 2 and len(r["results"]) == 2


def test_model_missing_gives_503(client, monkeypatch):
    c, api = client
    saved = dict(api.state)
    api.state.update(model=None, error="FileNotFoundError: test")
    try:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=NORMAL).status_code == 503
    finally:
        api.state.update(saved)
