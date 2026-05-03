"""환경변수 + 모델 차원 상수."""

import os

MODEL_PATH = os.getenv("MODEL_PATH", "./data/model.pb")
MAPPING_PATH = os.getenv("MAPPING_PATH", "./data/knowledgeTag_skillID.txt")

# 모델 학습 시 차원 — 매핑·인코딩과 일치해야 함
NUM_PROBLEMS = 1865
