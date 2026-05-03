"""GET /health — 모델 / 매핑 로드 상태 확인."""

from fastapi import APIRouter

from app.model.inference import model
from app.model.mapping import mapping


router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {
        "model_loaded": model.loaded,
        "mapping_entries": len(mapping),
    }
