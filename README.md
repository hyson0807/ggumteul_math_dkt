# 꿈틀매쓰 DKT 서버

학생의 풀이 시퀀스를 받아 강·약점 개념을 진단하는 FastAPI 서버. 사전 학습된 DKT (Deep Knowledge Tracing) 모델을 추론에 사용한다.

전체 시스템 동작 원리는 [`docs/dkt-integration.md`](../docs/dkt-integration.md) 참고.

## 사전 준비

- Python 3.10\~3.11 권장 (TensorFlow 2.13\~2.15 호환)
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

1. GitHub 에 push 후 Railway 에서 새 프로젝트 → 이 디렉토리 연결
2. Railway 가 `Procfile` 의 명령으로 자동 시작
3. 배포 후 `/health` 로 모델 로드 확인 → `ggumteul_math_server` 의 `DKT_BASE_URL` 환경변수 갱신

`model.pb` (265KB), `knowledgeTag_skillID.txt` (19KB) 는 git 에 포함되어 별도 업로드 불필요.

## 모델·매핑 갱신

- `model.pb`: 팀원이 재학습한 frozen graph 로 교체
- `knowledgeTag_skillID.txt`: 매핑 변경 시 교체. 형식 `knowledgeTag<TAB>skillID`
- 둘 다 갱신 후 서버 재시작 필요
