"""POST /predict — 풀이 시퀀스 → 강·약점 진단."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import NUM_PROBLEMS
from app.model.encoding import encode_sequence
from app.model.inference import model
from app.model.mapping import mapping
from app.schemas import (
    Diagnosis,
    PredictRequest,
    PredictResponse,
    SkillEntry,
)


router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not model.loaded:
        raise HTTPException(503, "모델이 로드되지 않았습니다.")

    if len(req.knowledge_tags) != len(req.corrects):
        raise HTTPException(
            400,
            "knowledge_tags 와 corrects 의 길이가 일치하지 않습니다.",
        )

    skill_ids: list[int] = []
    corrects: list[int] = []
    for tag, c in zip(req.knowledge_tags, req.corrects):
        skill = mapping.to_skill(tag)
        if skill is None:
            continue
        skill_ids.append(skill)
        corrects.append(1 if c == 1 else 0)

    if not skill_ids:
        raise HTTPException(422, "매핑 가능한 knowledge_tag 가 없습니다.")

    x = encode_sequence(skill_ids, corrects)
    final_prob = model.predict_last(x)

    if req.restrict_to_tags:
        candidate_indices: list[int] = []
        for tag in req.restrict_to_tags:
            skill = mapping.to_skill(tag)
            if skill is None:
                continue
            idx = skill - 1
            if 0 <= idx < NUM_PROBLEMS:
                candidate_indices.append(idx)

        if not candidate_indices:
            raise HTTPException(
                422,
                "restrict_to_tags 에 매핑 가능한 tag 가 없습니다.",
            )

        candidate = np.array(candidate_indices)
        sub_probs = final_prob[candidate]
        order = np.argsort(sub_probs)
        k = min(req.top_k, len(candidate))
        top_indices = candidate[order[-k:][::-1]]
        bottom_indices = candidate[order[:k]]
    else:
        k = min(req.top_k, NUM_PROBLEMS)
        top_indices = np.argsort(final_prob)[-k:][::-1]
        bottom_indices = np.argsort(final_prob)[:k]

    return PredictResponse(
        student_id=req.student_id,
        diagnosis=Diagnosis(
            top_strong=_to_entries(top_indices, final_prob),
            bottom_weak=_to_entries(bottom_indices, final_prob),
        ),
    )


def _to_entries(indices: np.ndarray, probs: np.ndarray) -> list[SkillEntry]:
    out: list[SkillEntry] = []
    for i in indices:
        skill_id = int(i + 1)
        tag = mapping.to_tag(skill_id)
        if tag is None:
            continue
        out.append(
            SkillEntry(
                knowledge_tag=tag,
                skill_id=skill_id,
                probability=round(float(probs[i]), 4),
            )
        )
    return out
