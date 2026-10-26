"""LoginGuard — 로그인 위험 판정 API + 보안 관제 화면 + 예측·정답 로그 수집.

서비스: 클라우드 서버를 운영하는 보안 관제 담당자가 로그인 시도를 실시간으로 감시할 때 쓴다.
        로그인 1건(계정 ID, IP, 국가, 통신사, 기기, 성공/실패)을 받아 공격자 접근일 위험 점수를 계산하고
        allow(허용) / verify(추가 인증·캡챠) / block(차단)을 돌려준다.

엔드포인트
  GET  /                 관제 화면 (사용자용)
  GET  /health, /ready   서버 생존 / 모델 적재
  POST /predict          로그인 1건 판정        POST /predict/batch  여러 건
  POST /feedback         담당자 정답 입력 (attack | normal)
  GET  /api/state        화면용 상태            POST /api/stream     데모 수신 재생 제어

로그: 판정(prediction)과 정답(feedback)을 logs/predictions.jsonl 에 JSON 한 줄씩 저장하고,
      CLOUDWATCH_LOG_GROUP 이 있으면 CloudWatch Logs 로도 보낸다.

실행: uvicorn api:app --port 8000
환경변수: MODEL_PATH, CSV, POOL_ROWS, CLOUDWATCH_LOG_GROUP, CLOUDWATCH_LOG_STREAM, AWS_DEFAULT_REGION
"""
from __future__ import annotations

import collections
import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, IPvAnyAddress

from features import FEATURES, REASONS, FeatureState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("loginguard")

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "models" / "model.joblib"))
CSV = os.getenv("CSV", "data/rba-dataset.csv")
POOL_ROWS = int(os.getenv("POOL_ROWS", 300_000))  # 0 = 전체
LOG_FILE = Path(os.getenv("LOG_FILE", "logs/predictions.jsonl"))


# ------------------------------------------------------------------ 모델
M: dict = {"model": None, "error": None}


def load_model():
    try:
        b = joblib.load(MODEL_PATH)
        if b["features"] != FEATURES:
            raise ValueError("모델의 피처 목록이 features.py와 다릅니다")
        M.update(model=b["model"], name=b["model_name"], verify=float(os.getenv("VERIFY_THRESHOLD", b["verify"])),
                 block=float(os.getenv("BLOCK_THRESHOLD", b["block"])), error=None)
        log.info("model_loaded %s verify=%s block=%s", b["model_name"], M["verify"], M["block"])
    except Exception as e:
        M.update(model=None, error=f"{type(e).__name__}: {e}")
        log.error("model_load_failed %s", M["error"])


load_model()


def decide(p: float) -> str:
    return "block" if p >= M["block"] else "verify" if p >= M["verify"] else "allow"


# ------------------------------------------------------------------ 로그 수집
class LogSink:
    def __init__(self):
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.group = os.getenv("CLOUDWATCH_LOG_GROUP")
        self.stream = os.getenv("CLOUDWATCH_LOG_STREAM", f"loginguard-{time.strftime('%Y%m%d-%H%M%S')}")
        self.cw, self.error, self.sent, self.q = None, None, 0, queue.Queue()
        self.flock = threading.Lock()
        if self.group:
            try:
                import boto3
                self.cw = boto3.client("logs")
                for fn, kw in ((self.cw.create_log_group, {"logGroupName": self.group}),
                               (self.cw.create_log_stream, {"logGroupName": self.group, "logStreamName": self.stream})):
                    try:
                        fn(**kw)
                    except self.cw.exceptions.ResourceAlreadyExistsException:
                        pass
                threading.Thread(target=self._flush, daemon=True).start()
                log.info("cloudwatch group=%s stream=%s", self.group, self.stream)
            except Exception as e:
                self.cw, self.error = None, f"{type(e).__name__}: {e}"
                log.error("cloudwatch_off %s", self.error)

    def write(self, rec: dict):
        line = json.dumps(rec, ensure_ascii=False)
        with self.flock, LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.cw:
            self.q.put({"timestamp": int(time.time() * 1000), "message": line})

    def _flush(self):
        while True:
            time.sleep(2)
            batch = []
            while not self.q.empty() and len(batch) < 500:
                batch.append(self.q.get())
            if batch:
                try:
                    self.cw.put_log_events(logGroupName=self.group, logStreamName=self.stream, logEvents=batch)
                    self.sent += len(batch); self.error = None
                except Exception as e:
                    self.error = f"{type(e).__name__}: {e}"
                    log.error("cloudwatch_put_failed %s", self.error)

    def status(self):
        return {"file": str(LOG_FILE), "cloudwatch": bool(self.cw), "group": self.group,
                "stream": self.stream if self.cw else None, "sent": self.sent, "error": self.error}


sink = LogSink()

# ------------------------------------------------------------------ 판정
live_state = FeatureState()          # API로 들어온 실제 요청의 계정·IP 기록 (서버 메모리)
state_lock = threading.Lock()
events: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
ev_lock = threading.Lock()


def score(x: list[float], reasons: list[str], ev: dict, source: str, dataset_label: str | None = None) -> dict:
    p = float(M["model"].predict_proba(np.asarray([x], dtype=float))[0, 1])
    out = {"request_id": uuid.uuid4().hex[:12], "time": time.strftime("%H:%M:%S"), "source": source,
           "user_id": str(ev["user_id"]), "ip": str(ev["ip"]), "country": ev.get("country"),
           "device_type": ev.get("device_type"), "success": bool(ev["success"]),
           "risk_score": round(p, 4), "decision": decide(p), "reasons": [REASONS[r] for r in reasons],
           "model_version": M["name"], "dataset_label": dataset_label, "feedback": None}
    with ev_lock:
        events[out["request_id"]] = out
        while len(events) > 500:
            events.popitem(last=False)
    sink.write({"event": "prediction", **{k: v for k, v in out.items() if k != "feedback"},
                "thresholds": {"verify": M["verify"], "block": M["block"]},
                "input": {k: ev.get(k) for k in ("user_id", "ip", "country", "asn", "device_type", "browser", "os", "success", "rtt_ms", "ts")},
                "features": dict(zip(FEATURES, map(float, x)))})
    return out


class LoginEvent(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128, description="계정 ID")
    ip: IPvAnyAddress
    country: str = Field("-", max_length=3, description="국가 코드 (예: KR)")
    asn: int = Field(0, ge=0, description="통신사 번호(ASN)")
    device_type: Literal["desktop", "mobile", "tablet", "bot", "unknown"] = "unknown"
    browser: str = Field("-", max_length=64)
    os: str = Field("-", max_length=64)
    success: bool = Field(..., description="비밀번호 확인 결과 (비밀번호 자체는 받지 않는다)")
    rtt_ms: float | None = Field(None, ge=0)
    timestamp: float | None = Field(None, ge=0, description="유닉스 시각(초). 없으면 서버 현재 시각")


class Decision(BaseModel):
    request_id: str
    risk_score: float
    decision: Literal["allow", "verify", "block"]
    reasons: list[str]
    model_version: str


def _require_model():
    if M["model"] is None:
        raise HTTPException(503, "모델이 준비되지 않았습니다")


def _predict_one(e: LoginEvent, source="api") -> dict:
    ev = e.model_dump()
    ev["ip"] = str(e.ip)
    ev["ts"] = e.timestamp if e.timestamp is not None else time.time()
    with state_lock:
        x, reasons = live_state.transform(ev)
    return score(x, reasons, ev, source)


app = FastAPI(title="LoginGuard", version="2.0.0",
              description="로그인 1건을 받아 공격자 접근 위험을 판정하는 보안 관제 API (allow / verify / block)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if M["model"] is None:
        raise HTTPException(503, {"ready": False, "error": M["error"]})
    return {"ready": True, "model_version": M["name"], "thresholds": {"verify": M["verify"], "block": M["block"]},
            "features": FEATURES, "log": sink.status()}


@app.post("/predict", response_model=Decision)
def predict(e: LoginEvent):
    _require_model()
    d = _predict_one(e)
    log.info("predict id=%s user=%s ip=%s decision=%s risk=%.3f", d["request_id"], d["user_id"], d["ip"], d["decision"], d["risk_score"])
    return d


class Batch(BaseModel):
    events: list[LoginEvent] = Field(..., min_length=1, max_length=1000)


@app.post("/predict/batch")
def predict_batch(b: Batch):
    _require_model()
    res = [_predict_one(e) for e in sorted(b.events, key=lambda e: e.timestamp or 0)]
    keep = ("request_id", "user_id", "ip", "risk_score", "decision", "reasons")
    return {"count": len(res), "decisions": dict(collections.Counter(r["decision"] for r in res)),
            "results": [{k: r[k] for k in keep} for r in res]}


class Feedback(BaseModel):
    request_id: str
    actual: Literal["attack", "normal"]


@app.post("/feedback")
def feedback(f: Feedback):
    with ev_lock:
        ev = events.get(f.request_id)
        if not ev:
            raise HTTPException(404, "최근 판정 목록에 없는 request_id 입니다")
        ev["feedback"] = f.actual
    correct = (ev["decision"] != "allow") == (f.actual == "attack")
    sink.write({"event": "feedback", "request_id": f.request_id, "time": time.strftime("%H:%M:%S"),
                "user_id": ev["user_id"], "ip": ev["ip"], "decision": ev["decision"], "risk_score": ev["risk_score"],
                "actual": f.actual, "correct": correct, "model_version": ev["model_version"]})
    return {"ok": True, "correct": correct}


# ------------------------------------------------------------------ 데모 수신 (RBA 실제 기록 재생)
stream = {"running": False, "interval": 1.5, "attack_ratio": 0.3, "pool": "empty", "pool_error": None}
pool: dict = {}


def build_pool():
    """CSV 앞부분을 시간순으로 읽어 계정·IP 기록을 쌓으면서 피처를 계산하고, 뒤쪽 30%를 재생 후보로 둔다."""
    try:
        stream["pool"] = "building"
        from data import load_events
        df, _ = load_events(CSV, POOL_ROWS or None)
        st = FeatureState()
        cols = ["ts", "user_id", "ip", "country", "asn", "device_type", "browser", "os", "success", "rtt_ms"]
        start = int(len(df) * 0.7)
        items = {"normal": [], "attack": []}
        for i, row in enumerate(df[cols + ["is_attack_ip"]].itertuples(index=False, name=None)):
            ev = dict(zip(cols, row[:-1]))
            x, reasons = st.transform(ev)
            if i >= start:
                items["attack" if row[-1] else "normal"].append((x, reasons, ev))
        pool.update(items)
        stream["pool"] = f"ready (정상 {len(items['normal'])} / 공격 {len(items['attack'])})"
    except Exception as e:
        stream["pool"], stream["pool_error"] = "error", f"{type(e).__name__}: {e}"
        stream["running"] = False
        log.error("pool_failed %s", stream["pool_error"])


def stream_loop():
    rng = np.random.default_rng()
    while True:
        if stream["running"] and pool and M["model"] is not None:
            kind = "attack" if (rng.random() < stream["attack_ratio"] and pool["attack"]) or not pool["normal"] else "normal"
            x, reasons, ev = pool[kind][rng.integers(len(pool[kind]))]
            score(x, reasons, ev, "stream", "공격 IP" if kind == "attack" else "정상")
        time.sleep(stream["interval"])


threading.Thread(target=stream_loop, daemon=True).start()


class StreamCtl(BaseModel):
    running: bool | None = None
    interval: float | None = None
    attack_ratio: float | None = None


@app.post("/api/stream")
def stream_ctl(c: StreamCtl):
    if c.running and stream["pool"] in ("empty", "error"):
        threading.Thread(target=build_pool, daemon=True).start()
    if c.running is not None:
        stream["running"] = c.running
    if c.interval is not None:
        stream["interval"] = min(max(c.interval, 0.2), 10)
    if c.attack_ratio is not None:
        stream["attack_ratio"] = min(max(c.attack_ratio, 0), 1)
    return stream


@app.get("/api/state")
def state():
    with ev_lock:
        evs = list(events.values())
    fb = [e for e in evs if e["feedback"]]
    flagged = lambda e: e["decision"] != "allow"
    return {"model_version": M.get("name"), "thresholds": {"verify": M.get("verify"), "block": M.get("block")},
            "stream": stream, "log": sink.status(),
            "stats": {"total": len(evs), "allow": sum(e["decision"] == "allow" for e in evs),
                      "verify": sum(e["decision"] == "verify" for e in evs), "block": sum(e["decision"] == "block" for e in evs),
                      "feedback": len(fb), "correct": sum(flagged(e) == (e["feedback"] == "attack") for e in fb),
                      "false_positive": sum(flagged(e) and e["feedback"] == "normal" for e in fb),
                      "false_negative": sum(not flagged(e) and e["feedback"] == "attack" for e in fb)},
            "events": evs[::-1][:100]}


@app.get("/", response_class=HTMLResponse)
def console():
    return (Path(__file__).parent / "console.html").read_text(encoding="utf-8")
