"""RBA 로그인 데이터셋 읽기 → 시간순으로 피처 계산.

데이터: Login Data Set for Risk-Based Authentication (Wiefling et al., 2022), CC BY 4.0
  https://zenodo.org/records/6782156  — 실제 대형 온라인 서비스 로그인 기록을 바탕으로 만든 3,300만여 건
정답(라벨): Is Attack IP  — 그 로그인이 공격자 IP(크리덴셜 스터핑 등)에서 왔는가

정리 규칙 (전후 건수를 report 로 남긴다)
  1. 시각·계정·IP·성공 여부가 없는 행 제거
  2. 시각 순으로 정렬 (피처가 '이전 기록'만 쓰도록)
  3. 완전히 같은 행 제거

사용법:  python data.py --csv rba-dataset.csv --nrows 2000000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from features import FEATURES, FeatureState

DATA_URL = "https://zenodo.org/records/6782156/files/rba-dataset.zip?download=1"
COLS = {
    "Login Timestamp": "ts", "User ID": "user_id", "Round-Trip Time [ms]": "rtt_ms", "IP Address": "ip",
    "Country": "country", "ASN": "asn", "Browser Name and Version": "browser", "OS Name and Version": "os",
    "Device Type": "device_type", "Login Successful": "success", "Is Attack IP": "is_attack_ip",
    "Is Account Takeover": "is_takeover",
}


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})


def load_events(path: str, nrows: int | None = None) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, usecols=list(COLS), nrows=nrows, dtype={"User ID": str, "IP Address": str}).rename(columns=COLS)
    report = {"raw_rows": len(df)}
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for c in ("success", "is_attack_ip", "is_takeover"):
        df[c] = _bool(df[c])
    df = df.dropna(subset=["ts", "user_id", "ip", "success", "is_attack_ip"])
    report["after_dropna"] = len(df)
    df = df.drop_duplicates()
    report["after_dedup"] = len(df)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["ts"] = (df["ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds()  # 해상도(ns/us)와 무관하게 초 단위
    df["is_takeover"] = df["is_takeover"].fillna(False)
    df["rtt_ms"] = pd.to_numeric(df["rtt_ms"], errors="coerce")
    report["period"] = [pd.Timestamp(df["ts"].iloc[0], unit="s").isoformat(), pd.Timestamp(df["ts"].iloc[-1], unit="s").isoformat()]
    report["users"] = int(df["user_id"].nunique())
    report["ips"] = int(df["ip"].nunique())
    report["success_ratio"] = round(float(df["success"].mean()), 4)
    report["attack_ip_ratio"] = round(float(df["is_attack_ip"].mean()), 4)
    report["takeover_rows"] = int(df["is_takeover"].sum())
    return df, report


def build_features(df: pd.DataFrame, state: FeatureState | None = None) -> np.ndarray:
    state = state or FeatureState()
    cols = ["ts", "user_id", "ip", "country", "asn", "device_type", "browser", "os", "success", "rtt_ms"]
    X = np.empty((len(df), len(FEATURES)), dtype=np.float32)
    for i, row in enumerate(df[cols].itertuples(index=False, name=None)):
        X[i], _ = state.transform(dict(zip(cols, row)))
    return X


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--nrows", type=int)
    a = ap.parse_args()
    _, rep = load_events(a.csv, a.nrows)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
