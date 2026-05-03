# 꿈틀매쓰 DKT 서버

학생의 풀이 시퀀스를 받아 강·약점 개념을 진단하는 FastAPI 서버. 사전 학습된 DKT (Deep Knowledge Tracing) 모델을 추론에 사용한다.

전체 시스템 동작 원리는 [`docs/dkt-integration.md`](../docs/dkt-integration.md) 참고

## 프로젝트 구조

```
ggumteul_math_dkt/
├── app/
│   ├── main.py          # FastAPI 부트스트랩 + lifespan
│   ├── config.py        # 환경변수 + 차원 상수
│   ├── schemas.py       # Pydantic 요청/응답
│   ├── model/           # 모델·매핑·인코딩 (인프라)
│   │   ├── inference.py
│   │   ├── mapping.py
│   │   └── encoding.py
│   └── api/             # 라우터
│       ├── health.py
│       └── predict.py
├── data/
│   ├── model.pb                  # frozen graph
│   └── knowledgeTag_skillID.txt  # 1865 entries
├── Procfile             # uvicorn app.main:app
├── requirements.txt
└── .python-version      # 3.11
```

## 사전 준비

- Python 3.10\~3.11 권장 (TensorFlow 2.13\~2.15 호환). Python 3.12+ 는 TF 2.13\~2.15 wheel 이 없어 빌드 실패
- `.python-version` 파일이 `3.11` 로 고정 — Railway/mise/pyenv 가 자동 인식
- `data/model.pb`, `data/knowledgeTag_skillID.txt` 가 존재하는지 확인

## 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

부팅 로그에 다음 두 줄이 보이면 정상:
```
✅ 매핑 로드 완료: 1865 entries
✅ AI 모델 로드 완료
```

## API 사용 예시

### `GET /health`
```bash
curl http://localhost:8000/health
# {"model_loaded": true, "mapping_entries": 1865}
```

### `POST /predict`

**요청**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "test-user",
    "knowledge_tags": [5485, 5485, 6646, 7618, 7912],
    "corrects":       [1,    1,    0,    1,    0],
    "restrict_to_tags": [5485, 6774, 7797],
    "top_k": 5
  }'
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `student_id` | string | ✓ | 학생 식별자 |
| `knowledge_tags` | int[] | ✓ | 푼 문제들의 concept knowledgeTag 시퀀스 (1개 이상) |
| `corrects` | int[] | ✓ | 각 문제 정답 여부 (1=정답, 0=오답). `knowledge_tags` 와 길이 일치 |
| `restrict_to_tags` | int[] | ✗ | 응답 후보를 이 tag 들로 제한. 미지정 시 모델 전체 1865 skill 에서 추출 |
| `top_k` | int | ✗ | 강·약점 각각 몇 개씩 반환할지 (1\~20, 기본 5) |

**응답**
```json
{
  "student_id": "test-user",
  "diagnosis": {
    "top_strong": [
      { "knowledge_tag": 5485, "skill_id": 9,  "probability": 0.89 }
    ],
    "bottom_weak": [
      { "knowledge_tag": 7912, "skill_id": 14, "probability": 0.01 }
    ]
  }
}
```

| 필드 | 설명 |
|------|------|
| `top_strong[]` | 모델이 잘 한다고 판단한 skill (확률 내림차순, 최대 `top_k` 개) |
| `bottom_weak[]` | 모델이 약하다고 판단한 skill (확률 오름차순, 최대 `top_k` 개) |
| `knowledge_tag` | 외부 매핑된 ID — `Concept.knowledgeTag` 와 일치 |
| `skill_id` | 모델 내부 인덱스 (1\~1865, 디버깅용) |
| `probability` | 다음 문제 정답 예측 확률 (0\~1, 소수점 4자리) |

**에러**
- `400` — `knowledge_tags` 와 `corrects` 길이 불일치
- `422` — 매핑 가능한 `knowledge_tag` 가 없음
- `422` — `restrict_to_tags` 에 매핑 가능한 tag 없음
- `422` — `top_k` 가 1\~20 범위 밖

## 배포 (Railway)

### 1. GitHub repo 생성 + 푸시
GitHub 에서 새 repo 만들기 (private 권장) → 로컬에서:
```bash
cd ggumteul_math_dkt
git remote add origin git@github.com:<유저명>/ggumteul_math_dkt.git
git push -u origin main
```

### 2. Railway 프로젝트 연결
1. [Railway 대시보드](https://railway.app/) → **New Project** → **Deploy from GitHub repo**
2. 위에서 만든 repo 선택 (Railway 가 GitHub 권한 요청하면 허용)
3. Railway 가 `Procfile` 자동 인식 → `uvicorn app.main:app --host 0.0.0.0 --port $PORT` 로 시작
4. 빌드 완료 후 **Settings → Networking → Generate Domain** 으로 public URL 발급
   - 예: `https://ggumteul-math-dkt-production.up.railway.app`

> 환경변수 추가 불필요 — `MODEL_PATH`, `MAPPING_PATH` 는 코드 기본값으로 동작. `PORT` 는 Railway 가 자동 주입.

### 3. 배포 검증
```bash
curl https://<railway-dkt-url>/health
# → {"model_loaded": true, "mapping_entries": 1865}

curl -X POST https://<railway-dkt-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"student_id":"test","knowledge_tags":[5485],"corrects":[1]}'
```

### 4. NestJS 측 환경변수 갱신
기존 NestJS Railway 프로젝트 → **Variables** 에 추가:
```
DKT_BASE_URL = https://<railway-dkt-url>
DKT_TIMEOUT_MS = 10000
```
저장 후 자동 redeploy.

> `data/model.pb` (7MB), `data/knowledgeTag_skillID.txt` (19KB) 는 git 에 포함되어 있으므로 별도 업로드 불필요. Railway 가 GitHub 에서 그대로 가져온다.

### 운영 메모
- **Cold start**: TF frozen graph 첫 로드 시 5\~15초 소요. Railway 가 무료 플랜에서 idle sleep 시키면 첫 호출 응답이 느릴 수 있음.
- **로그 확인**: Railway 대시보드 → 해당 서비스 → Deployments → 최신 빌드 → View Logs.

## 모델·매핑 갱신

- `data/model.pb`: 팀원이 재학습한 frozen graph 로 교체
- `data/knowledgeTag_skillID.txt`: 매핑 변경 시 교체. 형식 `knowledgeTag<TAB>skillID`
- 둘 다 갱신 후 서버 재시작 필요
