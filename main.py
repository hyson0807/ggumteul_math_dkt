"""꿈틀매쓰 DKT 추론 서버.

학생의 풀이 시퀀스(knowledge_tag + correct)를 받아 강·약점 진단을 반환한다.
모델 단위는 skill (= knowledgeTag)이며, 외부에는 knowledgeTag 만 노출하고
내부에서 skill_id 로 변환하여 모델에 입력한다.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import tensorflow.compat.v1 as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

tf.disable_v2_behavior()

MODEL_PATH = os.getenv("MODEL_PATH", "./model.pb")
MAPPING_PATH = os.getenv("MAPPING_PATH", "./knowledgeTag_skillID.txt")
NUM_PROBLEMS = 1865

app = FastAPI(title="꿈틀매쓰 DKT 서버")

_sess: tf.Session | None = None
_x_tensor = None
_keep_prob_tensor = None
_preds_tensor = None
_tag_to_skill: dict[int, int] = {}
_skill_to_tag: dict[int, int] = {}


@app.on_event("startup")
def _startup() -> None:
    _load_mapping()
    _load_model()


def _load_mapping() -> None:
    if not os.path.exists(MAPPING_PATH):
        raise RuntimeError(f"매핑 파일을 찾을 수 없습니다: {MAPPING_PATH}")

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            try:
                tag, skill = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            _tag_to_skill[tag] = skill
            _skill_to_tag[skill] = tag

    if not _tag_to_skill:
        raise RuntimeError(f"매핑 파일이 비어 있습니다: {MAPPING_PATH}")
    print(f"✅ 매핑 로드 완료: {len(_tag_to_skill)} entries")


def _load_model() -> None:
    global _sess, _x_tensor, _keep_prob_tensor, _preds_tensor

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    with tf.io.gfile.GFile(MODEL_PATH, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    _sess = tf.Session(graph=graph)
    _x_tensor = graph.get_tensor_by_name("X:0")
    _keep_prob_tensor = graph.get_tensor_by_name("keep_prob:0")
    _preds_tensor = graph.get_tensor_by_name("output_layer/preds:0")
    print("✅ AI 모델 로드 완료")


class PredictRequest(BaseModel):
    student_id: str
    knowledge_tags: List[int] = Field(min_length=1)
    corrects: List[int] = Field(min_length=1)
    # 응답에 포함될 후보 tag 를 제한. None 이면 모델 전체 (1865) 에서 top/bottom 추출.
    # NestJS 가 우리 커리큘럼 (229개 concept tag) 으로 제한할 때 사용.
    restrict_to_tags: Optional[List[int]] = None


class SkillEntry(BaseModel):
    knowledge_tag: int
    skill_id: int
    probability: float


class Diagnosis(BaseModel):
    top_5_strong: List[SkillEntry]
    bottom_5_weak: List[SkillEntry]


class PredictResponse(BaseModel):
    student_id: str
    diagnosis: Diagnosis


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _sess is None or _x_tensor is None:
        raise HTTPException(503, "모델이 로드되지 않았습니다.")

    if len(req.knowledge_tags) != len(req.corrects):
        raise HTTPException(
            400,
            "knowledge_tags 와 corrects 의 길이가 일치하지 않습니다.",
        )

    skill_ids: list[int] = []
    corrects: list[int] = []
    for tag, c in zip(req.knowledge_tags, req.corrects):
        skill = _tag_to_skill.get(tag)
        if skill is None:
            continue
        skill_ids.append(skill)
        corrects.append(1 if c == 1 else 0)

    if not skill_ids:
        raise HTTPException(
            422,
            "매핑 가능한 knowledge_tag 가 없습니다.",
        )

    # 학습 시 인코딩 (load_data.py:OriginalInputProcessor) 과 일치:
    #   첫 절반 (problem_oh): 푼 skill 위치에 항상 1
    #   둘째 절반 (correct_oh): 정답일 때만 같은 위치에 1
    seq_len = len(skill_ids)
    x = np.zeros((1, seq_len, 2 * NUM_PROBLEMS), dtype=np.float32)
    for t, (s, c) in enumerate(zip(skill_ids, corrects)):
        idx = s - 1
        if idx < 0 or idx >= NUM_PROBLEMS:
            continue
        x[0, t, idx] = 1.0
        if c == 1:
            x[0, t, idx + NUM_PROBLEMS] = 1.0

    pred = _sess.run(
        _preds_tensor,
        feed_dict={_x_tensor: x, _keep_prob_tensor: 1.0},
    )
    final_prob = pred[0, -1, :]

    if req.restrict_to_tags:
        candidate_indices: list[int] = []
        for tag in req.restrict_to_tags:
            skill = _tag_to_skill.get(tag)
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
        k = min(5, len(candidate))
        top_indices = candidate[order[-k:][::-1]]
        bottom_indices = candidate[order[:k]]
    else:
        top_indices = np.argsort(final_prob)[-5:][::-1]
        bottom_indices = np.argsort(final_prob)[:5]

    return PredictResponse(
        student_id=req.student_id,
        diagnosis=Diagnosis(
            top_5_strong=_to_entries(top_indices, final_prob),
            bottom_5_weak=_to_entries(bottom_indices, final_prob),
        ),
    )


def _to_entries(indices: np.ndarray, probs: np.ndarray) -> list[SkillEntry]:
    out: list[SkillEntry] = []
    for i in indices:
        skill_id = int(i + 1)
        tag = _skill_to_tag.get(skill_id)
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


@app.get("/health")
def health() -> dict:
    return {
        "model_loaded": _sess is not None,
        "mapping_entries": len(_tag_to_skill),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
