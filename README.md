# LoginGuard — 로그인 이상 접근 판정 API와 보안 관제 화면

## 1. 서비스 정의

- **누가**: 클라우드 서버를 운영하는 보안 관제 담당자
- **어떤 상황에**: 서버로 들어오는 로그인 시도를 실시간으로 감시할 때
- **무엇을 입력받아**: 로그인 1건 — 계정 ID, 접속 IP, 국가, 통신사(ASN), 기기·브라우저·OS, 비밀번호 확인 결과(성공/실패)
- **무엇을 예측하나**: 공격자 접근(크리덴셜 스터핑·무차별 대입 등)일 위험 점수와 처리 방법 — `allow`(허용) / `verify`(추가 인증·캡챠) / `block`(차단)

> 비밀번호 자체는 받지도 저장하지도 않습니다. 로그인 서버가 확인한 **결과(성공/실패)**만 씁니다.

## 2. 데이터

- **RBA 로그인 데이터셋** — *Login Data Set for Risk-Based Authentication* (Wiefling et al., 2022, CC BY 4.0), [Zenodo](https://zenodo.org/records/6782156)
- 실제 대형 온라인 서비스의 로그인 기록을 바탕으로 만든 **3,300만여 건**
- 사용 컬럼: 로그인 시각, User ID, IP, 국가, ASN, 브라우저, OS, 기기 종류, 로그인 성공 여부, RTT
- 정답: `Is Attack IP` (공격자 IP에서 온 로그인인가). `Is Account Takeover`(계정 탈취)는 추가 평가에 사용
- 정리: 필수값 결측 행 제거 → 완전 중복 제거 → 시각 순 정렬 (건수는 `metrics.json`의 `data_report`)

### 피처 (14개, `features.py`) — 모두 **그 로그인 이전 기록만** 사용

| 구분 | 피처 |
|---|---|
| 이번 로그인 | 성공/실패, 시간대, 응답 지연(RTT), 모바일 여부 |
| 계정 기록 | 이전 로그인 수, 실패 비율, 처음 보는 국가·통신사·기기인지, 마지막 로그인 후 경과 시간 |
| IP 기록 | 이 IP의 이전 시도 수, 실패 비율, 시도한 계정 수, 마지막 시도 후 경과 시간 |

학습과 서빙이 **같은 `FeatureState` 코드**로 피처를 계산하므로 학습·서빙 불일치가 없습니다.

## 3. 평가 설계 (`model.py`)

- **시간순 분할**: 앞 60% 학습 / 다음 20% 검증 / 마지막 20% 테스트 → 과거로 배워 미래를 맞히는지
- 후보: 로지스틱 회귀, 랜덤 포레스트 → **검증 F1로 선택**, 테스트는 한 번만
- 기준선: 전부 정상 / 규칙(같은 IP에서 계정 5개 이상 또는 실패 50% 이상)
- **새 IP 검증**: 학습 기간에 없던 IP의 로그인만 따로 평가 (IP를 외운 게 아닌지)
- 계정 탈취 로그인 중 몇 %를 추가 인증 이상으로 잡는지
- 테스트 정확도 0.97 이상이면 경고 (공격 비율이 낮으면 정확도가 부풀기 때문에 재현율·balanced accuracy를 함께 봄)

## 4. 결과

> Colab에서 실제 데이터로 실행한 뒤 노트북 결과 표를 붙여 넣습니다.

| | 정확도 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
| 전부 정상 (기준선) | | | | |
| 규칙 (기준선) | | | | |
| 선택 모델 | | | | |
| 새 IP만 | | | | |

## 5. API와 화면 (`api.py`)

```bash
uvicorn api:app --port 8000      # 화면: http://localhost:8000/   문서: /docs
```

| 엔드포인트 | 설명 |
|---|---|
| `GET /` | **보안 관제 화면** — 실시간 로그인 판정 표, 위험 신호, 정답 입력, 통계 |
| `GET /health` / `GET /ready` | 서버 생존 / 모델 적재 |
| `POST /predict` | 로그인 1건 판정 |
| `POST /predict/batch` | 최대 1000건 |
| `POST /feedback` | 담당자 정답 입력 (`attack` / `normal`) |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "user_id": "user-1001", "ip": "185.220.0.10", "country": "RU", "asn": 12345,
  "device_type": "desktop", "browser": "Chrome 80.0", "os": "Windows 10", "success": false}'
# → {"request_id":"…","risk_score":0.97,"decision":"block","reasons":["같은 IP에서 여러 계정 시도", …],"model_version":"random_forest"}
```

## 6. 로그 수집 (입력·예측값·정답)

모든 판정과 정답 입력을 JSON 한 줄로 `logs/predictions.jsonl`에 저장하고, `CLOUDWATCH_LOG_GROUP`을 설정하면 **AWS CloudWatch Logs**로도 보냅니다.

```json
{"event":"prediction","request_id":"5f1c…","user_id":"user-1001","ip":"185.220.0.10","risk_score":0.97,"decision":"block","input":{…},"features":{…}}
{"event":"feedback","request_id":"5f1c…","decision":"block","actual":"attack","correct":true}
```

`request_id`로 예측과 정답을 이어 붙이면, 실제 운영 데이터로 모델 성능을 다시 재고 재학습 데이터를 만들 수 있습니다.

## 7. 실행

```bash
pip install -r requirements.txt
wget -O rba.zip "https://zenodo.org/records/6782156/files/rba-dataset.zip?download=1" && unzip rba.zip -d data
python model.py --csv data/rba-dataset.csv --nrows 2000000 --max-train-rows 600000
CSV=data/rba-dataset.csv uvicorn api:app --port 8000
pytest -q          # 실제 데이터 없이 가짜 RBA 형식 데이터로 테스트
```

## 8. 한계

- RBA는 실제 기록을 바탕으로 **합성한** 데이터라 실제 서비스에서는 다시 검증이 필요
- API의 계정·IP 기록은 서버 메모리에 있어 재시작하면 초기화됨 (운영 시 Redis 등 외부 저장소 필요)
- 관제 화면의 실시간 수신은 데모용으로 데이터셋 기록을 재생함
