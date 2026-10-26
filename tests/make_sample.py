"""테스트용 가짜 RBA CSV (실제 데이터 아님). 실제 파일과 같은 컬럼 이름·형식을 쓴다."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

COLUMNS = ["index", "Login Timestamp", "User ID", "Round-Trip Time [ms]", "IP Address", "Country", "Region", "City",
           "ASN", "User Agent String", "Browser Name and Version", "OS Name and Version", "Device Type",
           "Login Successful", "Is Attack IP", "Is Account Takeover"]


def make(n_users: int = 800, n_normal: int = 12000, n_attack_ips: int = 25, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2020-02-03").value / 1e9
    span = 30 * 86400
    users = [str(rng.integers(-9e18, 9e18)) for _ in range(n_users)]
    home = {u: (rng.choice(["NO", "US", "DE", "BR", "IN"]), int(rng.integers(1000, 60000)),
                f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
                rng.choice(["desktop", "mobile"])) for u in users}
    rows = []

    def add(ts, u, ip, country, asn, dev, ok, attack, takeover=False):
        browser = "Chrome 80.0" if dev == "desktop" else "Chrome Mobile 79.0"
        os_ = "Windows 10" if dev == "desktop" else "Android 9"
        rows.append([0, pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], u,
                     float(rng.integers(300, 900)) if rng.random() < 0.05 else np.nan, ip, country, "-", "-", asn,
                     "Mozilla/5.0", browser, os_, dev, ok, attack, takeover])

    for _ in range(n_normal):
        u = users[rng.integers(n_users)]
        c, asn, ip, dev = home[u]
        if rng.random() < 0.05:  # 여행·새 기기
            c, asn, ip = rng.choice(["FR", "JP", "GB"]), int(rng.integers(1000, 60000)), f"172.16.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
        add(t0 + rng.random() * span, u, ip, c, asn, dev, bool(rng.random() < 0.9), False)

    for k in range(n_attack_ips):  # 크리덴셜 스터핑: 한 IP가 짧은 시간에 여러 계정 시도
        ip, c, asn = f"185.220.{k}.{rng.integers(1, 255)}", rng.choice(["RU", "CN", "VN"]), int(rng.integers(1000, 60000))
        start = t0 + rng.random() * span
        for j in range(int(rng.integers(20, 80))):
            u = users[rng.integers(n_users)]
            ok = bool(rng.random() < 0.08)
            add(start + j * rng.uniform(1, 20), u, ip, c, asn, "desktop", ok, True, takeover=ok and rng.random() < 0.3)

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("Login Timestamp").reset_index(drop=True)
    df["index"] = range(len(df))
    return pd.concat([df, df.sample(50, random_state=seed)]).sort_values("Login Timestamp").reset_index(drop=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "rba-sample.csv"
    make().to_csv(out, index=False)
    print("saved", out)
