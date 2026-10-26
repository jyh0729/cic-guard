"""클라우드 서버 로그인 보안 관제 API.

네트워크 흐름(CICFlowMeter 피처) 1건을 받아 SSH·FTP 무차별 대입 공격 확률을 계산하고
차단(block) 또는 통과(allow)를 돌려준다. 모델은 model.py가 저장한 파일을 읽기만 한다
(서버 시작 때 다시 학습하지 않는다).

실행:  uvicorn api:app --port 8000      문서: http://localhost:8000/docs
환경변수
  MODEL_PATH       모델 파일 (기본 models/model.joblib)
  BLOCK_THRESHOLD  차단 기준 확률 (기본: 모델에 저장된 값, 보통 0.5)
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from features import ALLOW_NEGATIVE, API_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cic-guard")

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "models" / "model.joblib"))

state: dict = {"model": None, "error": None}


def load_model() -> None:
    try:
        b = joblib.load(MODEL_PATH)
        if b["api_features"] != API_FEATURES:
            raise ValueError("모델의 피처 목록이 features.py와 다릅니다")
        state.update(model=b["model"], name=b["model_name"],
                     threshold=float(os.getenv("BLOCK_THRESHOLD", b.get("threshold", 0.5))),
                     version=f"{b['model_name']}@{hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]}",
                     error=None)
        log.info("model_loaded path=%s version=%s", MODEL_PATH, state["version"])
    except Exception as e:  # 서버는 켜 두고 /ready 와 /predict 가 503 을 돌려준다
        state.update(model=None, error=f"{type(e).__name__}: {e}")
        log.error("model_load_failed path=%s error=%s", MODEL_PATH, state["error"])


load_model()
app = FastAPI(title="CIC Guard", version="1.0.0",
              description="네트워크 흐름으로 SSH·FTP 무차별 대입 공격을 판정해 차단 여부를 알려 주는 API")

# 피처마다 0 이상 검증 (Init Win Byts 는 '관측 안 됨' = -1 허용)
Flow = create_model("Flow", **{f: (float, Field(..., ge=-1 if f in ALLOW_NEGATIVE else 0)) for f in API_FEATURES})


class Decision(BaseModel):
    request_id: str
    attack_probability: float
    decision: Literal["allow", "block"]
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    flows: list[Flow] = Field(..., min_length=1, max_length=1000)  # type: ignore[valid-type]


def _require_model():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다")


def _predict(rows: list) -> np.ndarray:
    X = np.array([[getattr(r, f) for f in API_FEATURES] for r in rows], dtype=float)
    return state["model"].predict_proba(X)[:, 1]


def _decide(p: float) -> Decision:
    return Decision(request_id=uuid.uuid4().hex[:12], attack_probability=round(float(p), 4),
                    decision="block" if p >= state["threshold"] else "allow",
                    threshold=state["threshold"], model_version=state["version"])


@app.get("/health")
def health():
    """서버 코드가 응답하는지만 본다."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """검증된 모델이 메모리에 올라와 있는지 본다."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail={"ready": False, "error": state["error"]})
    return {"ready": True, "model_version": state["version"], "threshold": state["threshold"],
            "features": API_FEATURES}


@app.post("/predict", response_model=Decision)
def predict(flow: Flow):  # type: ignore[valid-type]
    _require_model()
    t0 = time.perf_counter()
    d = _decide(_predict([flow])[0])
    log.info("predict id=%s decision=%s p=%.4f ms=%.1f", d.request_id, d.decision, d.attack_probability,
             (time.perf_counter() - t0) * 1000)
    return d


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    _require_model()
    probs = _predict(req.flows)
    results = [_decide(p) for p in probs]
    return {"count": len(results), "blocked": sum(r.decision == "block" for r in results), "results": results}
