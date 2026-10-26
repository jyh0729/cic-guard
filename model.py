"""CSE-CIC-IDS2018 흐름 데이터로 무차별 대입(SSH·FTP) 공격 판정 모델 학습·평가·저장.

평가 설계
  - 정리(중복 제거 포함) 뒤 학습 60% / 검증 20% / 테스트 20%, 라벨 비율 유지(stratify)
  - 후보 선택은 검증 F1 로만 한다. 테스트는 선택이 끝난 뒤 한 번만 본다.
  - 기준선 2개와 비교: 다수 클래스(전부 정상), 포트 규칙(목적지 21·22번이면 공격)
  - 미지 공격 검증: FTP 공격만으로 학습한 모델이 처음 보는 SSH 공격을 얼마나 잡는가
  - 정확도가 0.97 이상이면 경고한다 (과제 기준: 90% 후반 → 피처 조정 검토)

사용법:
  python model.py --csv data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv [--max-per-class 200000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from data import load_clean
from features import API_FEATURES, CSV_FEATURES, signed_log1p

SEED = 42
TOO_HIGH = 0.97


def candidates():
    return {
        "logistic_regression": make_pipeline(FunctionTransformer(signed_log1p), StandardScaler(),
                                             LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
    }


def scores(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"accuracy": round(accuracy_score(y, p), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y, p), 4),
            "precision": round(precision_score(y, p, zero_division=0), 4),
            "recall": round(recall_score(y, p, zero_division=0), 4),
            "f1": round(f1_score(y, p, zero_division=0), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "n": int(len(y))}


def cap_per_class(df, cap):
    if not cap:
        return df
    parts = [g.sample(min(len(g), cap), random_state=SEED) for _, g in df.groupby("Label")]
    return pd.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--max-per-class", type=int, default=0, help="라벨별 최대 행 수 (메모리 부족 시)")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    df, data_report = load_clean(a.csv)
    df = cap_per_class(df, a.max_per_class).reset_index(drop=True)
    data_report["used_rows"] = len(df)
    print("데이터:", json.dumps(data_report, ensure_ascii=False))

    X, y = df[CSV_FEATURES].to_numpy(dtype=float), df["y"].to_numpy()
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=SEED)

    # 기준선
    port_rule = df["Dst Port"].isin([21, 22]).astype(int).to_numpy()
    maj = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
    baselines = {"majority_all_benign": {"validation": scores(y[va], maj.predict(X[va])),
                                         "test": scores(y[te], maj.predict(X[te]))},
                 "port_rule_21_22": {"validation": scores(y[va], port_rule[va]),
                                     "test": scores(y[te], port_rule[te])}}

    # 후보 학습 → 검증 F1 로 선택
    results, fitted = {}, {}
    for name, m in candidates().items():
        t0 = time.time()
        m.fit(X[tr], y[tr])
        results[name] = {"validation": scores(y[va], m.predict(X[va])), "train_seconds": round(time.time() - t0, 1)}
        fitted[name] = m
        print(f"{name:20s} 검증 F1={results[name]['validation']['f1']:.4f} "
              f"재현율={results[name]['validation']['recall']:.4f} ({results[name]['train_seconds']}초)")
    best = max(results, key=lambda n: results[n]["validation"]["f1"])
    model = fitted[best]

    # 선택 뒤 테스트 한 번
    proba = model.predict_proba(X[te])[:, 1]
    pred = (proba >= a.threshold).astype(int)
    test = scores(y[te], pred)
    test["train_accuracy"] = round(accuracy_score(y[tr], model.predict(X[tr])), 4)
    per_attack = {}
    for lab in sorted(df.loc[te, "Label"].unique()):
        mask = (df.loc[te, "Label"] == lab).to_numpy()
        hit = pred[mask].mean()
        per_attack[lab] = {"n": int(mask.sum()), ("detected_rate" if lab != "Benign" else "false_alarm_rate"): round(float(hit), 4)}
    test["per_label"] = per_attack
    # 같은 포트(21·22) 안에서만: 포트 번호로는 못 가르는 '정상 SSH·FTP 접속 vs 공격'을 구별하는가
    same_port = df.loc[te, "Dst Port"].isin([21, 22]).to_numpy()
    if same_port.any():
        test["login_ports_only"] = scores(y[te][same_port], pred[same_port])

    # 미지 공격 검증: SSH 공격을 학습에서 빼고, 테스트의 SSH 공격을 얼마나 잡는지
    unseen = {}
    if {"FTP-BruteForce", "SSH-Bruteforce"} <= set(df["Label"]):
        tr_no_ssh = tr[(df.loc[tr, "Label"] != "SSH-Bruteforce").to_numpy()]
        m2 = candidates()[best].fit(X[tr_no_ssh], y[tr_no_ssh])
        te_ssh = te[(df.loc[te, "Label"] == "SSH-Bruteforce").to_numpy()]
        te_ben = te[(df.loc[te, "Label"] == "Benign").to_numpy()]
        unseen = {"trained_without": "SSH-Bruteforce",
                  "ssh_detected_rate": round(float(m2.predict(X[te_ssh]).mean()), 4),
                  "benign_false_alarm_rate": round(float(m2.predict(X[te_ben]).mean()), 4)}

    print(f"\n선택: {best} | 테스트 정확도 {test['accuracy']:.4f} 정밀도 {test['precision']:.4f} "
          f"재현율 {test['recall']:.4f} F1 {test['f1']:.4f} (학습 정확도 {test['train_accuracy']:.4f})")
    print("기준선 테스트 정확도: 전부 정상", baselines["majority_all_benign"]["test"]["accuracy"],
          "/ 포트 규칙", baselines["port_rule_21_22"]["test"]["accuracy"])
    print("라벨별:", json.dumps(per_attack, ensure_ascii=False))
    if "login_ports_only" in test:
        lp = test["login_ports_only"]
        print(f"21·22번 포트만: n={lp['n']} 정확도 {lp['accuracy']} 정밀도 {lp['precision']} 재현율 {lp['recall']} "
              f"정상 오탐 {lp['confusion_matrix']['fp']}건")
    if unseen:
        print("미지 공격(SSH 제외 학습):", json.dumps(unseen, ensure_ascii=False))
    warnings = []
    if test["accuracy"] >= TOO_HIGH:
        warnings.append(f"테스트 정확도 {test['accuracy']} ≥ {TOO_HIGH}: 과도하게 높음. 피처 중요도·누수·미지 공격 결과를 확인할 것")
    if baselines["port_rule_21_22"]["test"]["accuracy"] >= test["accuracy"]:
        warnings.append("포트 규칙이 모델 이상으로 정확함: 모델의 추가 가치를 다시 검토할 것")
    for w in warnings:
        print("⚠", w)

    importance = {}
    if hasattr(model, "feature_importances_"):
        importance = {API_FEATURES[i]: round(float(v), 4)
                      for i, v in sorted(enumerate(model.feature_importances_), key=lambda t: -t[1])}

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "model_name": best, "csv_features": CSV_FEATURES, "api_features": API_FEATURES,
                 "threshold": a.threshold, "sklearn_version": sklearn.__version__}, out / "model.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps({
        "dataset": "CSE-CIC-IDS2018, Processed Traffic Data (CICFlowMeter)", "files": [Path(p).name for p in a.csv],
        "task": "네트워크 흐름 1건이 SSH·FTP 무차별 대입 공격인지 판정",
        "data_report": data_report, "split": {"train": len(tr), "validation": len(va), "test": len(te)},
        "baselines": baselines, "candidates": results, "best_model": best, "threshold": a.threshold,
        "test": test, "unseen_attack": unseen, "feature_importance": importance, "warnings": warnings,
        "sklearn_version": sklearn.__version__}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out/'model.joblib'}, {out/'metrics.json'}")


if __name__ == "__main__":
    main()
