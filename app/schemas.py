"""DKT 서버 요청·응답 스키마."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    student_id: str
    knowledge_tags: List[int] = Field(min_length=1)
    corrects: List[int] = Field(min_length=1)
    # 응답에 포함될 후보 tag 를 제한. None 이면 모델 전체 (1865) 에서 top/bottom 추출.
    # NestJS 가 우리 커리큘럼 (229개 concept tag) 으로 제한할 때 사용.
    restrict_to_tags: Optional[List[int]] = None
    # 강·약점 각각 몇 개씩 반환할지. 기본 5, 후보 수보다 크면 후보 수만큼만.
    top_k: int = Field(default=5, gt=0, le=20)


class SkillEntry(BaseModel):
    knowledge_tag: int
    skill_id: int
    probability: float


class Diagnosis(BaseModel):
    top_strong: List[SkillEntry]
    bottom_weak: List[SkillEntry]


class PredictResponse(BaseModel):
    student_id: str
    diagnosis: Diagnosis
