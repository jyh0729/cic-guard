"""학습·서빙이 함께 쓰는 피처 정의.

CSE-CIC-IDS2018 'Processed Traffic Data for ML Algorithms' CSV(CICFlowMeter 출력)의
컬럼 이름 → API에서 받는 영문 snake_case 이름.

제외한 컬럼과 이유
  - Dst Port, Protocol : 공격이 21/22번 포트에만 몰려 있어 모델이 '포트 번호'만 외운다.
                          포트 규칙은 기준선으로 따로 비교한다.
  - Timestamp          : 공격 시간대를 외우게 된다(누수).
  - Label              : 정답.
"""

# CSV 컬럼 → API 필드
FEATURE_MAP = {
    "Flow Duration": "flow_duration_us",
    "Tot Fwd Pkts": "fwd_packets",
    "Tot Bwd Pkts": "bwd_packets",
    "TotLen Fwd Pkts": "fwd_bytes",
    "TotLen Bwd Pkts": "bwd_bytes",
    "Fwd Pkt Len Mean": "fwd_pkt_len_mean",
    "Bwd Pkt Len Mean": "bwd_pkt_len_mean",
    "Flow Byts/s": "flow_bytes_per_s",
    "Flow Pkts/s": "flow_pkts_per_s",
    "Flow IAT Mean": "flow_iat_mean_us",
    "Fwd IAT Mean": "fwd_iat_mean_us",
    "SYN Flag Cnt": "syn_flags",
    "RST Flag Cnt": "rst_flags",
    "PSH Flag Cnt": "psh_flags",
    "ACK Flag Cnt": "ack_flags",
    "Init Fwd Win Byts": "init_fwd_win_bytes",
    "Init Bwd Win Byts": "init_bwd_win_bytes",
    "Fwd Seg Size Min": "fwd_seg_size_min",
    "Down/Up Ratio": "down_up_ratio",
}
CSV_FEATURES = list(FEATURE_MAP)
API_FEATURES = list(FEATURE_MAP.values())

# 기준선 비교용으로만 읽는 컬럼
EXTRA_COLUMNS = ["Dst Port", "Timestamp", "Label"]

# 이 데이터에서 '공격'으로 보는 라벨 (2018-02-14 파일)
ATTACK_LABELS = {"FTP-BruteForce", "SSH-Bruteforce"}
BENIGN_LABEL = "Benign"

# Init Win Byts 는 '관측 안 됨'을 -1 로 적는다 → 음수 허용 피처
ALLOW_NEGATIVE = {"init_fwd_win_bytes", "init_bwd_win_bytes"}


def signed_log1p(x):
    """값의 범위가 0 ~ 수억까지 넓어 로그로 줄인다. 고정 변환이라 학습 자료에서 배우는 값이 없다.
    모델 파일에 함수 위치가 저장되므로 학습·API 양쪽에서 import 가능한 이 모듈에 둔다."""
    import numpy as np
    return np.sign(x) * np.log1p(np.abs(x))
