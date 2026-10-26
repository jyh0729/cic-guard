"""테스트용 가짜 CIC-IDS2018 CSV 생성기 (실제 데이터 아님).

실제 'Processed Traffic Data' CSV와 같은 80개 컬럼 이름·순서·타임스탬프 형식을 쓰고,
실제 파일에 있는 문제(중간 헤더 행, inf, 중복 행)를 일부러 넣어 정리 코드를 시험한다.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

COLUMNS = ("Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,"
           "Fwd Pkt Len Max,Fwd Pkt Len Min,Fwd Pkt Len Mean,Fwd Pkt Len Std,Bwd Pkt Len Max,Bwd Pkt Len Min,"
           "Bwd Pkt Len Mean,Bwd Pkt Len Std,Flow Byts/s,Flow Pkts/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,"
           "Flow IAT Min,Fwd IAT Tot,Fwd IAT Mean,Fwd IAT Std,Fwd IAT Max,Fwd IAT Min,Bwd IAT Tot,Bwd IAT Mean,"
           "Bwd IAT Std,Bwd IAT Max,Bwd IAT Min,Fwd PSH Flags,Bwd PSH Flags,Fwd URG Flags,Bwd URG Flags,"
           "Fwd Header Len,Bwd Header Len,Fwd Pkts/s,Bwd Pkts/s,Pkt Len Min,Pkt Len Max,Pkt Len Mean,Pkt Len Std,"
           "Pkt Len Var,FIN Flag Cnt,SYN Flag Cnt,RST Flag Cnt,PSH Flag Cnt,ACK Flag Cnt,URG Flag Cnt,"
           "CWE Flag Count,ECE Flag Cnt,Down/Up Ratio,Pkt Size Avg,Fwd Seg Size Avg,Bwd Seg Size Avg,"
           "Fwd Byts/b Avg,Fwd Pkts/b Avg,Fwd Blk Rate Avg,Bwd Byts/b Avg,Bwd Pkts/b Avg,Bwd Blk Rate Avg,"
           "Subflow Fwd Pkts,Subflow Fwd Byts,Subflow Bwd Pkts,Subflow Bwd Byts,Init Fwd Win Byts,"
           "Init Bwd Win Byts,Fwd Act Data Pkts,Fwd Seg Size Min,Active Mean,Active Std,Active Max,Active Min,"
           "Idle Mean,Idle Std,Idle Max,Idle Min,Label").split(",")


def make(n_benign: int = 6000, n_ftp: int = 1500, n_ssh: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def flow(label, port, hour_lo, hour_hi, attack):
        dur = rng.lognormal(8 if attack else 12, 1.0)
        fp = rng.integers(2, 8) if attack else rng.integers(1, 60)
        bp = rng.integers(1, 6) if attack else rng.integers(0, 60)
        fb = fp * (rng.normal(40, 8) if attack else rng.lognormal(5, 1))
        bb = bp * (rng.normal(60, 10) if attack else rng.lognormal(5.5, 1.2))
        r = dict.fromkeys(COLUMNS, 0)
        h, m, s = rng.integers(hour_lo, hour_hi), rng.integers(0, 60), rng.integers(0, 60)
        r.update({
            "Dst Port": port, "Protocol": 6, "Timestamp": f"14/02/2018 {h:02d}:{m:02d}:{s:02d}",
            "Flow Duration": int(dur), "Tot Fwd Pkts": int(fp), "Tot Bwd Pkts": int(bp),
            "TotLen Fwd Pkts": max(0.0, fb), "TotLen Bwd Pkts": max(0.0, bb),
            "Fwd Pkt Len Mean": max(0.0, fb) / fp, "Bwd Pkt Len Mean": (max(0.0, bb) / bp) if bp else 0,
            "Flow Byts/s": (fb + bb) / dur * 1e6, "Flow Pkts/s": (fp + bp) / dur * 1e6,
            "Flow IAT Mean": dur / max(fp + bp - 1, 1), "Fwd IAT Mean": dur / max(fp - 1, 1),
            "SYN Flag Cnt": int(rng.random() < (0.1 if attack else 0.05)),
            "RST Flag Cnt": int(rng.random() < (0.6 if attack else 0.05)),
            "PSH Flag Cnt": int(rng.random() < (0.8 if attack else 0.4)),
            "ACK Flag Cnt": int(rng.random() < 0.5),
            "Init Fwd Win Byts": int(rng.choice([26883, 29200, 65535, -1], p=[0.5, 0.3, 0.1, 0.1] if attack else [0.1, 0.2, 0.5, 0.2])),
            "Init Bwd Win Byts": int(rng.choice([227, 26847, 0, -1])),
            "Fwd Seg Size Min": int(rng.choice([32, 20])), "Down/Up Ratio": int(bp // max(fp, 1)),
            "Label": label,
        })
        rows.append(r)

    for _ in range(n_benign):
        flow("Benign", int(rng.choice([443, 80, 53, 3389, 22, 21], p=[0.5, 0.25, 0.15, 0.06, 0.03, 0.01])), 8, 18, False)
    for _ in range(n_ftp):
        flow("FTP-BruteForce", 21, 10, 12, True)
    for _ in range(n_ssh):
        flow("SSH-Bruteforce", 22, 14, 16, True)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.loc[df.sample(30, random_state=seed).index, "Flow Byts/s"] = np.inf     # 0으로 나눈 값
    df = pd.concat([df, df.sample(200, random_state=seed)], ignore_index=True)  # 중복 행
    header = pd.DataFrame([dict(zip(COLUMNS, COLUMNS))])                       # 중간 헤더 행
    df = pd.concat([df.iloc[:4000], header, df.iloc[4000:]], ignore_index=True)
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample.csv"
    make().to_csv(out, index=False)
    print(f"saved {out}")
