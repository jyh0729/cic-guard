"""RBA 로그인 기록으로 '공격자 IP에서 온 로그인' 판정 모델 학습·평가·저장.

평가 설계
  - 시간순 분할: 앞 60% 학습 / 다음 20% 검증 / 마지막 20% 테스트 (과거로 배워 미래를 맞히는지)
  - 후보 선택은 검증 F1, 테스트는 선택 뒤 한 번만
  - 기준선: 전부 정상 / 규칙(같은 IP에서 계정 5개 이상 시도 또는 실패 50% 이상)
  - 새 IP 검증: 학습 기간에 한 번도 안 나온 IP의 로그인만 따로 평가 (IP를 외운 게 아닌지)
  - 계정 탈취(Is Account Takeover) 로그인 중 몇 %를 추가 인증 이상으로 잡는지
  - 테스트 정확도 0.97 이상이면 경고 (과제 기준: 90% 후반 → 피처 조정 검토)

판정 단계: 위험 점수 < verify → allow, verify ≤ 점수 < block → verify(추가 인증·캡챠), block 이상 → block

사용법:  python model.py --csv rba-dataset.csv --nrows 2000000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import build_features, load_events
from features import FEATURES

SEED = 42
TOO_HIGH = 0.97


def scores(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"accuracy": round(accuracy_score(y, p), 4), "balanced_accuracy": round(balanced_accuracy_score(y, p), 4),
            "precision": round(precision_score(y, p, zero_division=0), 4),
            "recall": round(recall_score(y, p, zero_division=0), 4), "f1": round(f1_score(y, p, zero_division=0), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}, "n": int(len(y))}


def candidates():
    return {"logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=5, class_weight="balanced",
                                                    n_jobs=-1, random_state=SEED)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--nrows", type=int, default=0, help="앞에서부터 읽을 로그인 수 (0 = 전체)")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--max-train-rows", type=int, default=0, help="학습 행이 너무 많으면 무작위로 줄임")
    ap.add_argument("--verify", type=float, default=0.5)
    ap.add_argument("--block", type=float, default=0.9)
    a = ap.parse_args()

    t0 = time.time()
    df, report = load_events(a.csv, a.nrows or None)
    X = build_features(df)
    y = df["is_attack_ip"].astype(int).to_numpy()
    print("데이터:", json.dumps(report, ensure_ascii=False), f"(피처 계산 {time.time() - t0:.0f}초)")

    n = len(df)
    tr, va, te = np.arange(0, int(n * .6)), np.arange(int(n * .6), int(n * .8)), np.arange(int(n * .8), n)
    tr_fit = tr
    if a.max_train_rows and len(tr) > a.max_train_rows:
        tr_fit = np.sort(np.random.default_rng(SEED).choice(tr, a.max_train_rows, replace=False))
    split = {"train": len(tr), "train_used": len(tr_fit), "validation": len(va), "test": len(te),
             "train_attack_ratio": round(float(y[tr].mean()), 4), "test_attack_ratio": round(float(y[te].mean()), 4)}

    F = {f: i for i, f in enumerate(FEATURES)}
    rule = ((np.expm1(X[:, F["ip_distinct_users_log"]]) >= 5) |
            ((np.expm1(X[:, F["ip_prev_logins_log"]]) >= 5) & (X[:, F["ip_prev_fail_ratio"]] >= 0.5))).astype(int)
    baselines = {"majority_all_normal": {"test": scores(y[te], np.zeros(len(te), int))},
                 "rule_many_users_or_fails": {"validation": scores(y[va], rule[va]), "test": scores(y[te], rule[te])}}

    results, fitted = {}, {}
    for name, m in candidates().items():
        t1 = time.time()
        m.fit(X[tr_fit], y[tr_fit])
        pv = (m.predict_proba(X[va])[:, 1] >= a.verify).astype(int)
        results[name] = {"validation": scores(y[va], pv), "train_seconds": round(time.time() - t1, 1)}
        fitted[name] = m
        print(f"{name:20s} 검증 F1={results[name]['validation']['f1']:.4f} 재현율={results[name]['validation']['recall']:.4f} "
              f"({results[name]['train_seconds']}초)")
    best = max(results, key=lambda k: results[k]["validation"]["f1"])
    model = fitted[best]

    proba = model.predict_proba(X[te])[:, 1]
    pred = (proba >= a.verify).astype(int)
    test = scores(y[te], pred)
    test["train_accuracy"] = round(accuracy_score(y[tr_fit], (model.predict_proba(X[tr_fit])[:, 1] >= a.verify).astype(int)), 4)
    test["decisions"] = {"allow": int((proba < a.verify).sum()), "verify": int(((proba >= a.verify) & (proba < a.block)).sum()),
                         "block": int((proba >= a.block).sum())}
    test["normal_blocked_rate"] = round(float((proba[y[te] == 0] >= a.block).mean()), 5)

    train_ips = set(df["ip"].iloc[tr])
    new_ip = ~df["ip"].iloc[te].isin(train_ips).to_numpy()
    test["new_ips_only"] = scores(y[te][new_ip], pred[new_ip]) if new_ip.any() else None
    tk = df["is_takeover"].iloc[te].to_numpy(bool)
    test["account_takeover"] = {"n": int(tk.sum()), "caught_rate": round(float(pred[tk].mean()), 4) if tk.any() else None}

    warnings = []
    if test["accuracy"] >= TOO_HIGH:
        warnings.append(f"테스트 정확도 {test['accuracy']} ≥ {TOO_HIGH}: 공격 비율이 낮으면 정확도가 부풀므로 재현율·balanced accuracy·새 IP 결과를 함께 볼 것")
    if baselines["rule_many_users_or_fails"]["test"]["f1"] >= test["f1"]:
        warnings.append("단순 규칙의 F1이 모델 이상: 모델의 추가 가치를 다시 검토할 것")

    importance = {}
    if hasattr(model, "feature_importances_"):
        importance = {FEATURES[i]: round(float(v), 4) for i, v in sorted(enumerate(model.feature_importances_), key=lambda t: -t[1])}

    print(f"\n선택: {best} | 테스트 정확도 {test['accuracy']} 정밀도 {test['precision']} 재현율 {test['recall']} "
          f"F1 {test['f1']} balanced {test['balanced_accuracy']} (학습 정확도 {test['train_accuracy']})")
    print("기준선 테스트: 전부 정상 정확도", baselines["majority_all_normal"]["test"]["accuracy"],
          "/ 규칙 F1", baselines["rule_many_users_or_fails"]["test"]["f1"])
    print("판정 분포:", test["decisions"], "| 정상인데 차단 비율", test["normal_blocked_rate"])
    if test["new_ips_only"]:
        print("새 IP만:", {k: test["new_ips_only"][k] for k in ("n", "accuracy", "precision", "recall", "f1")})
    print("계정 탈취:", test["account_takeover"])
    for w in warnings:
        print("⚠", w)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "model_name": best, "features": FEATURES, "verify": a.verify, "block": a.block,
                 "sklearn_version": sklearn.__version__}, out / "model.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps({
        "dataset": "RBA login dataset (Wiefling et al., 2022)", "rows_read": a.nrows or "all",
        "task": "로그인 1건이 공격자 IP에서 온 것인지 판정 (Is Attack IP)", "data_report": report, "split": split,
        "baselines": baselines, "candidates": results, "best_model": best,
        "thresholds": {"verify": a.verify, "block": a.block}, "test": test, "feature_importance": importance,
        "warnings": warnings, "sklearn_version": sklearn.__version__}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out/'model.joblib'}, {out/'metrics.json'}  (총 {time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
