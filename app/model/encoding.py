"""학습 시점과 동일한 one-hot 인코딩.

i-Scream 데이터셋 학습 코드 (load_data.py:OriginalInputProcessor) 와 동일:
- 정답: 첫 절반 + 둘째 절반 모두 idx 위치에 1
- 오답: 첫 절반 idx 위치에만 1, 둘째 절반 = 0
"""

from __future__ import annotations

import numpy as np

from app.config import NUM_PROBLEMS


def encode_sequence(skill_ids: list[int], corrects: list[int]) -> np.ndarray:
    seq_len = len(skill_ids)
    x = np.zeros((1, seq_len, 2 * NUM_PROBLEMS), dtype=np.float32)
    for t, (s, c) in enumerate(zip(skill_ids, corrects)):
        idx = s - 1
        if idx < 0 or idx >= NUM_PROBLEMS:
            continue
        x[0, t, idx] = 1.0
        if c == 1:
            x[0, t, idx + NUM_PROBLEMS] = 1.0
    return x
