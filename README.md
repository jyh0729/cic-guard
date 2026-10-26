# CIC Guard — 로그인 무차별 대입 공격 차단 API

**서비스 정의**: 클라우드 서버 로그인 기록을 분석해, 비정상(해킹) 접근을 실시간으로 차단하는 보안 관제 AI 모델 및 서빙 API

서버로 들어오는 네트워크 흐름(flow) 1건의 통계값을 받아 SSH·FTP 무차별 대입 공격일 확률을 계산하고, 기준 확률 이상이면 `block`, 아니면 `allow`를 돌려줍니다.

## 이전 버전(PR #2)에서 바뀐 점

| 리뷰 지적 | 반영 |
|---|---|
| 서버 시작 때마다 KDD99를 받아 다시 학습 | `model.py`로 한 번 학습해 `models/model.joblib` 저장, API는 읽기만 함 |
| KDD99(1999년)는 오래되고 중복이 많음 | **CSE-CIC-IDS2018** 2018-02-14 파일(약 105만 행)로 교체, 중복 제거 후 분할 |
| 프로토콜을 정수로 받음, 음수 입력 허용 | 프로토콜·포트는 피처에서 제외, 모든 수치에 `ge=0` 검증 (윈도 크기만 −1 허용) |
| 응답 키가 한글 | `attack_probability`, `decision`, `threshold`, `model_version` |
| 차단 기준 80% 하드코딩 | 모델 파일에 저장 + `BLOCK_THRESHOLD` 환경변수로 변경 가능 |
| 정확도 하나만 출력 | 기준선 2개, 정밀도·재현율·F1·혼동행렬, 라벨별 탐지율, 미지 공격 검증 |
| 테스트·의존성 목록 없음 | `tests/` (pytest 11개), `requirements.txt` |
| 상태 확인 없음 | `/health`(서버 생존), `/ready`(모델 적재) 분리, 요청 로그 |

## 데이터

- 출처: Canadian Institute for Cybersecurity, **CSE-CIC-IDS2018** (Sharafaldin, Lashkari, Ghorbani, ICISSP 2018), AWS 공개 버킷
- 파일: `Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`
- 라벨: `Benign`(정상), `FTP-BruteForce`, `SSH-Bruteforce` → 공격 = 1
- 정리 규칙 (`data.py`, 건수는 `metrics.json`의 `data_report`에 기록)
  1. 파일 중간에 다시 나오는 헤더 행 제거
  2. 숫자가 아닌 값·`inf` → 결측 → **행 제거** (0으로 채우면 '모름'이 '실제 0'이 됨)
  3. 피처·라벨이 완전히 같은 행 제거 (같은 행이 학습·테스트에 함께 들어가 점수가 부풀지 않도록)

### 피처 (19개, `features.py`)

흐름 길이, 정방향·역방향 패킷 수와 바이트, 평균 패킷 길이, 초당 바이트·패킷, 패킷 간격 평균, SYN·RST·PSH·ACK 플래그 수, 초기 TCP 윈도 크기, 최소 세그먼트 크기, 다운/업 비율.

**뺀 것**: `Dst Port`(21·22번이면 공격이라는 지름길 → 대신 기준선으로 사용), `Protocol`, `Timestamp`(공격이 특정 시간대에만 있어 시간만 보고 맞힐 수 있음).

## 평가 설계 (`model.py`)

- 학습 60% / 검증 20% / 테스트 20%, 라벨 비율 유지
- 후보: 로지스틱 회귀(로그 변환+표준화), 랜덤 포레스트 → **검증 F1로 선택**, 테스트는 선택 후 한 번만
- 기준선: 전부 정상 / 포트 규칙(21·22번이면 공격)
- **21·22번 포트만 따로 평가**: 포트로는 못 가르는 '정상 SSH·FTP 접속 vs 공격'을 구별하는지
- **미지 공격 검증**: SSH 공격을 빼고 학습한 모델이 처음 보는 SSH 공격을 얼마나 잡는지
- 테스트 정확도 0.97 이상이면 경고 (과제 기준 4.1: 90% 후반이면 피처 조정 검토)

## 결과

> Colab에서 실제 파일로 돌린 뒤 `models/metrics.json` 값으로 채웁니다.

| | 정확도 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
| 전부 정상 (기준선) | | | | |
| 포트 규칙 (기준선) | | | | |
| 선택 모델 | | | | |
| 21·22번 포트만 | | | | |

미지 공격(SSH 제외 학습) SSH 탐지율: ___ / 정상 오탐률: ___

## 실행

```bash
pip install -r requirements.txt

# 1) 데이터 받기 (약 340MB)
mkdir -p data && curl -L -o data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv \
  "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"

# 2) 정리 결과 확인 / 학습
python data.py  --csv data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
python model.py --csv data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv   # 메모리 부족 시 --max-per-class 200000

# 3) API
uvicorn api:app --port 8000          # 문서: http://localhost:8000/docs

# 4) 테스트 (실제 데이터 없이 가짜 CIC 형식 데이터로)
pytest -q
```

### 요청 예시

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "flow_duration_us": 2500000, "fwd_packets": 30, "bwd_packets": 28, "fwd_bytes": 4800, "bwd_bytes": 9600,
  "fwd_pkt_len_mean": 160, "bwd_pkt_len_mean": 342, "flow_bytes_per_s": 5760, "flow_pkts_per_s": 23.2,
  "flow_iat_mean_us": 43860, "fwd_iat_mean_us": 86200, "syn_flags": 0, "rst_flags": 0, "psh_flags": 1,
  "ack_flags": 1, "init_fwd_win_bytes": 65535, "init_bwd_win_bytes": 0, "fwd_seg_size_min": 20, "down_up_ratio": 0}'
# → {"request_id":"…","attack_probability":0.01,"decision":"allow","threshold":0.5,"model_version":"random_forest@…"}
```

| 엔드포인트 | 설명 |
|---|---|
| `GET /health` | 서버 생존 |
| `GET /ready` | 모델 적재 여부, 버전, 피처 목록 (없으면 503) |
| `POST /predict` | 흐름 1건 판정 |
| `POST /predict/batch` | 최대 1000건 판정 |

## 한계

- 2018년 실험망 하루치 데이터, 공격 도구도 정해져 있음 → 실제 서버 트래픽에서는 다시 검증 필요
- 이 날짜의 무차별 대입 흐름은 정상과 쉽게 구별돼 점수가 매우 높게 나올 수 있음. 그래서 포트 규칙 기준선, 21·22번 포트만 평가, 미지 공격 검증을 함께 보고함
- 흐름 1건 단위 판정이라 "같은 IP가 1분에 50번 시도" 같은 누적 패턴은 보지 못함 (다음 단계: IP별 시간 창 집계)
- 실제 차단(방화벽 연동)은 하지 않고 판정만 돌려줌
