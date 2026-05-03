# 꿈틀매쓰 DKT 서버

학생의 풀이 시퀀스를 받아 강·약점 개념을 진단하는 FastAPI 서버. 사전 학습된 DKT (Deep Knowledge Tracing) 모델을 추론에 사용한다.

전체 시스템 동작 원리는 [`docs/dkt-integration.md`](../docs/dkt-integration.md) 참고.

## 사전 준비

- Python 3.10\~3.11 권장 (TensorFlow 2.13\~2.15 호환). Python 3.12+ 는 TF 2.13\~2.15 wheel 이 없어 빌드 실패
- `.python-version` 파일이 `3.11` 로 고정 — Railway/mise/pyenv 가 자동 인식
- `model.pb`, `knowledgeTag_skillID.txt` 가 프로젝트 루트에 있는지 확인

## 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
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
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "test-user",
    "knowledge_tags": [5485, 5485, 6646, 7618, 7912],
    "corrects":       [1,    1,    0,    1,    0],
    "restrict_to_tags": [5485, 6774, 7797]
  }'
```

`restrict_to_tags` 는 선택. 지정하면 응답의 강·약점을 그 tag 들로 제한한다. 없으면 모델 전체 1865개 skill 에서 추출되어 우리 커리큘럼 외 영역이 잡힐 수 있다.

응답:
```json
{
  "student_id": "test-user",
  "diagnosis": {
    "top_5_strong": [
      { "knowledge_tag": 5485, "skill_id": 9, "probability": 0.89 }
    ],
    "bottom_5_weak": [
      { "knowledge_tag": 7912, "skill_id": 14, "probability": 0.01 }
    ]
  }
}
```

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
3. Railway 가 `Procfile` 자동 인식 → `uvicorn main:app --host 0.0.0.0 --port $PORT` 로 시작
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

> `model.pb` (7MB), `knowledgeTag_skillID.txt` (19KB) 는 git 에 포함되어 있으므로 별도 업로드 불필요. Railway 가 GitHub 에서 그대로 가져온다.

### 운영 메모
- **Cold start**: TF frozen graph 첫 로드 시 5\~15초 소요. Railway 가 무료 플랜에서 idle sleep 시키면 첫 호출 응답이 느릴 수 있음.
- **로그 확인**: Railway 대시보드 → 해당 서비스 → Deployments → 최신 빌드 → View Logs.

## 모델·매핑 갱신

- `model.pb`: 팀원이 재학습한 frozen graph 로 교체
- `knowledgeTag_skillID.txt`: 매핑 변경 시 교체. 형식 `knowledgeTag<TAB>skillID`
- 둘 다 갱신 후 서버 재시작 필요
