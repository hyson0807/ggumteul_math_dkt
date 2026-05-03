"""꿈틀매쓰 DKT 추론 서버 진입점.

학생의 풀이 시퀀스 (knowledge_tag + correct) 를 받아 강·약점을 진단한다.
모델 단위는 skill (= knowledgeTag) 이며, 외부에는 knowledgeTag 만 노출하고
내부에서 skill_id 로 변환하여 모델에 입력한다.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, predict
from app.config import MAPPING_PATH, MODEL_PATH
from app.model.inference import model
from app.model.mapping import mapping


@asynccontextmanager
async def lifespan(_app: FastAPI):
    mapping.load(MAPPING_PATH)
    print(f"✅ 매핑 로드 완료: {len(mapping)} entries")
    model.load(MODEL_PATH)
    print("✅ AI 모델 로드 완료")
    yield


app = FastAPI(title="꿈틀매쓰 DKT 서버", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
