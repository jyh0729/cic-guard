"""로그인 1건 → 피처. 학습(data.py)과 서빙(api.py, console.py)이 같은 코드를 쓴다.

핵심: 피처는 '그 로그인 이전까지의 기록'만으로 계산한다 (미래 정보 누수 없음).
  계정 기록: 지금까지 로그인 횟수, 실패 비율, 처음 보는 국가/통신사(ASN)/기기인지, 마지막 로그인 후 경과 시간
  IP 기록  : 이 IP의 지금까지 시도 횟수, 실패 비율, 시도한 계정 수, 마지막 시도 후 경과 시간
  이번 로그인: 성공/실패, 시간대, 응답 지연(RTT), 모바일 여부
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

FEATURES = [
    "success",                    # 이번 로그인 비밀번호 확인 결과 (1 성공 / 0 실패)
    "hour",                       # 0~23 (UTC)
    "rtt_ms_log",                 # 응답 지연, 없으면 -1
    "is_mobile",
    "user_prev_logins_log",
    "user_prev_fail_ratio",
    "user_new_country",           # 이 계정이 처음 보는 국가
    "user_new_asn",               # 처음 보는 통신사(ASN)
    "user_new_device",            # 처음 보는 브라우저·OS·기기 조합
    "user_hours_since_last_log",  # 처음이면 -1
    "ip_prev_logins_log",
    "ip_prev_fail_ratio",
    "ip_distinct_users_log",      # 이 IP가 지금까지 시도한 서로 다른 계정 수
    "ip_seconds_since_last_log",  # 처음이면 -1
]

REASONS = {  # 화면·API 응답에 보여 줄 위험 신호
    "user_new_country": "처음 보는 국가",
    "user_new_asn": "처음 보는 통신사",
    "user_new_device": "처음 보는 기기",
    "ip_many_users": "같은 IP에서 여러 계정 시도",
    "ip_many_fails": "같은 IP에서 실패 반복",
    "ip_burst": "같은 IP에서 짧은 간격으로 반복 시도",
}


def _log(x: float) -> float:
    return math.log1p(x)


@dataclass
class _User:
    n: int = 0
    fails: int = 0
    last_ts: float | None = None
    countries: set = field(default_factory=set)
    asns: set = field(default_factory=set)
    devices: set = field(default_factory=set)


@dataclass
class _Ip:
    n: int = 0
    fails: int = 0
    last_ts: float | None = None
    users: set = field(default_factory=set)


class FeatureState:
    """계정·IP 기록을 들고 있다가 로그인이 올 때마다 피처를 만들고 기록을 갱신한다."""

    def __init__(self):
        self.users: dict[str, _User] = {}
        self.ips: dict[str, _Ip] = {}

    def transform(self, ev: dict) -> tuple[list[float], list[str]]:
        """ev: ts(초), user_id, ip, country, asn, device_type, browser, os, success(bool), rtt_ms(None 가능)"""
        u = self.users.setdefault(str(ev["user_id"]), _User())
        p = self.ips.setdefault(str(ev["ip"]), _Ip())
        ts = float(ev["ts"])
        device = f'{ev.get("browser")}|{ev.get("os")}|{ev.get("device_type")}'
        rtt = ev.get("rtt_ms")

        f = {
            "success": float(bool(ev["success"])),
            "hour": float(int(ts // 3600) % 24),
            "rtt_ms_log": _log(rtt) if rtt is not None and rtt == rtt and rtt >= 0 else -1.0,
            "is_mobile": float(ev.get("device_type") == "mobile"),
            "user_prev_logins_log": _log(u.n),
            "user_prev_fail_ratio": u.fails / u.n if u.n else 0.0,
            "user_new_country": float(u.n > 0 and ev.get("country") not in u.countries),
            "user_new_asn": float(u.n > 0 and ev.get("asn") not in u.asns),
            "user_new_device": float(u.n > 0 and device not in u.devices),
            "user_hours_since_last_log": _log(max(ts - u.last_ts, 0) / 3600) if u.last_ts is not None else -1.0,
            "ip_prev_logins_log": _log(p.n),
            "ip_prev_fail_ratio": p.fails / p.n if p.n else 0.0,
            "ip_distinct_users_log": _log(len(p.users)),
            "ip_seconds_since_last_log": _log(max(ts - p.last_ts, 0)) if p.last_ts is not None else -1.0,
        }
        reasons = [k for k in ("user_new_country", "user_new_asn", "user_new_device") if f[k]]
        if len(p.users) >= 5:
            reasons.append("ip_many_users")
        if p.n >= 5 and p.fails / p.n >= 0.5:
            reasons.append("ip_many_fails")
        if p.last_ts is not None and ts - p.last_ts < 10:
            reasons.append("ip_burst")

        # 기록 갱신
        ok = bool(ev["success"])
        u.n += 1; u.fails += not ok; u.last_ts = ts
        if ok:  # 성공한 로그인의 환경만 '아는 환경'으로 등록
            u.countries.add(ev.get("country")); u.asns.add(ev.get("asn")); u.devices.add(device)
        p.n += 1; p.fails += not ok; p.last_ts = ts
        if len(p.users) < 10_000:
            p.users.add(str(ev["user_id"]))
        return [f[k] for k in FEATURES], reasons
