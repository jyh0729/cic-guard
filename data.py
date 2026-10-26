"""CSE-CIC-IDS2018 CSV 읽기·정리.

정리 규칙 (전후 건수를 report 로 남긴다)
  1. 파일 중간에 다시 나오는 헤더 행(Label == "Label") 제거
  2. 숫자로 바꿀 수 없는 값 → 결측
  3. inf (Flow Byts/s 등에서 0으로 나눈 값) → 결측
  4. 결측 행 제거 (0으로 채우지 않는다: '모름'이 '실제 0'이 되기 때문)
  5. 피처·라벨이 완전히 같은 행 제거 (중복이 학습·평가에 함께 들어가 점수가 부푸는 것 방지)

사용법:
  python data.py --csv data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from features import ATTACK_LABELS, BENIGN_LABEL, CSV_FEATURES, EXTRA_COLUMNS

DATA_URL = ("https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
            "Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
            "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv")


def load_clean(paths: list[str], nrows: int | None = None) -> tuple[pd.DataFrame, dict]:
    frames = [pd.read_csv(p, usecols=CSV_FEATURES + EXTRA_COLUMNS, dtype=str, nrows=nrows) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    report = {"raw_rows": len(df)}

    df = df[df["Label"] != "Label"]
    report["after_header_rows"] = len(df)

    df = df[df["Label"].isin(ATTACK_LABELS | {BENIGN_LABEL})].copy()
    report["after_label_filter"] = len(df)

    for c in CSV_FEATURES + ["Dst Port"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[CSV_FEATURES] = df[CSV_FEATURES].replace([np.inf, -np.inf], np.nan)
    report["rows_with_missing_or_inf"] = int(df[CSV_FEATURES + ["Dst Port"]].isna().any(axis=1).sum())
    df = df.dropna(subset=CSV_FEATURES + ["Dst Port"])
    report["after_dropna"] = len(df)

    df = df.drop_duplicates(subset=CSV_FEATURES + ["Label"])
    report["after_dedup"] = len(df)

    df["y"] = df["Label"].isin(ATTACK_LABELS).astype(int)
    df["ts"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    report["label_counts"] = {k: int(v) for k, v in df["Label"].value_counts().items()}
    report["attack_ratio"] = round(float(df["y"].mean()), 4)
    return df.reset_index(drop=True), report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--nrows", type=int)
    a = ap.parse_args()
    _, rep = load_clean(a.csv, a.nrows)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
