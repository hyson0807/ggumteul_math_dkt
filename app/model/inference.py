"""TensorFlow frozen graph 로드 + 추론."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import tensorflow.compat.v1 as tf  # type: ignore[import-untyped]


tf.disable_v2_behavior()


class DktModel:
    def __init__(self) -> None:
        self._sess: Optional[tf.Session] = None
        self._x_tensor = None
        self._keep_prob_tensor = None
        self._preds_tensor = None

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise RuntimeError(f"모델 파일을 찾을 수 없습니다: {path}")

        with tf.io.gfile.GFile(path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        self._sess = tf.Session(graph=graph)
        self._x_tensor = graph.get_tensor_by_name("X:0")
        self._keep_prob_tensor = graph.get_tensor_by_name("keep_prob:0")
        self._preds_tensor = graph.get_tensor_by_name("output_layer/preds:0")

    @property
    def loaded(self) -> bool:
        return self._sess is not None

    def predict_last(self, x: np.ndarray) -> np.ndarray:
        """마지막 타임스텝의 1865차원 확률 벡터를 반환."""
        if self._sess is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        pred = self._sess.run(
            self._preds_tensor,
            feed_dict={self._x_tensor: x, self._keep_prob_tensor: 1.0},
        )
        return pred[0, -1, :]


# 모듈 싱글톤 — main.py 의 lifespan 에서 load() 호출
model = DktModel()
