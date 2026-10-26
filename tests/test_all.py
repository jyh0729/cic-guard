"""데이터·학습·API 테스트 (가짜 RBA 형식 데이터 사용)."""
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import FEATURES, FeatureState  # noqa: E402


def ev(**kw):
    base = {"user_id": "u1", "ip": "10.0.0.1", "country": "KR", "asn": 4766, "device_type": "desktop",
            "browser": "Chrome", "os": "Windows", "success": True}
    base.update(kw)
    return base


def test_features_use_only_past():
    s = FeatureState()
    x1, _ = s.transform({**ev(), "ts": 0})
    assert x1[FEATURES.index("user_prev_logins_log")] == 0          # 첫 로그인은 기록 없음
    x2, r2 = s.transform({**ev(country="RU", asn=1, ip="10.0.0.2"), "ts": 60})
    assert x2[FEATURES.index("user_new_country")] == 1 and "user_new_country" in r2


def test_ip_many_users_reason():
    s = FeatureState()
    for i in range(6):
        _, r = s.transform({**ev(user_id=f"u{i}", ip="185.0.0.1", success=False), "ts": i})
    assert "ip_many_users" in r and "ip_burst" in r


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("lg")
    csv = d / "rba.csv"
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_sample.py"), str(csv)], check=True, cwd=ROOT, capture_output=True)
    subprocess.run([sys.executable, str(ROOT / "model.py"), "--csv", str(csv), "--out-dir", str(d)], check=True, cwd=ROOT, capture_output=True)
    return d, csv


def test_metrics_written(trained):
    d, _ = trained
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    assert {"baselines", "test", "split", "warnings"} <= set(m)
    assert m["test"]["new_ips_only"] is not None


@pytest.fixture(scope="session")
def client(trained):
    d, csv = trained
    os.environ.update(MODEL_PATH=str(d / "model.joblib"), CSV=str(csv), POOL_ROWS="0", LOG_FILE=str(d / "pred.jsonl"))
    os.environ.pop("CLOUDWATCH_LOG_GROUP", None)
    import api
    importlib.reload(api)
    return TestClient(api.app), api, d


def test_health_ready(client):
    c, _, _ = client
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/ready").json()["ready"] is True


def test_predict_and_log(client):
    c, _, d = client
    r = c.post("/predict", json=ev())
    assert r.status_code == 200 and r.json()["decision"] in ("allow", "verify", "block")
    lines = (d / "pred.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[-1])["event"] == "prediction"


def test_credential_stuffing_raises_risk(client):
    c, _, _ = client
    first = c.post("/predict", json=ev(user_id="x0", ip="185.220.9.9", country="RU", success=False)).json()["risk_score"]
    last = None
    for i in range(1, 15):
        last = c.post("/predict", json=ev(user_id=f"x{i}", ip="185.220.9.9", country="RU", success=False)).json()
    assert last["risk_score"] > first and last["decision"] != "allow"
    assert "같은 IP에서 여러 계정 시도" in last["reasons"]


def test_validation(client):
    c, _, _ = client
    assert c.post("/predict", json=ev(ip="not-an-ip")).status_code == 422
    assert c.post("/predict", json=ev(asn=-1)).status_code == 422
    body = ev(); body.pop("success")
    assert c.post("/predict", json=body).status_code == 422
    assert c.post("/predict", json=ev(device_type="toaster")).status_code == 422


def test_feedback_logged(client):
    c, _, d = client
    rid = c.post("/predict", json=ev(user_id="fb")).json()["request_id"]
    assert c.post("/feedback", json={"request_id": rid, "actual": "normal"}).json()["ok"]
    assert json.loads((d / "pred.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])["event"] == "feedback"
    assert c.post("/feedback", json={"request_id": "nope", "actual": "normal"}).status_code == 404
    assert c.get("/api/state").json()["stats"]["feedback"] >= 1


def test_batch(client):
    c, _, _ = client
    r = c.post("/predict/batch", json={"events": [ev(user_id="b1"), ev(user_id="b2")]}).json()
    assert r["count"] == 2


def test_console_page_and_stream(client):
    c, api, _ = client
    assert "LoginGuard" in c.get("/").text
    c.post("/api/stream", json={"running": True, "interval": 0.2})
    for _ in range(50):
        if api.stream["pool"].startswith("ready"):
            break
        time.sleep(0.2)
    assert api.stream["pool"].startswith("ready"), api.stream
    time.sleep(1)
    c.post("/api/stream", json={"running": False})
    assert any(e["source"] == "stream" for e in c.get("/api/state").json()["events"])


def test_model_missing_503(client):
    c, api, _ = client
    saved = dict(api.M)
    api.M.update(model=None, error="test")
    try:
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=ev()).status_code == 503
    finally:
        api.M.update(saved)
