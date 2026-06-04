from pathlib import Path

import numpy as np


class RetinaFaceDetector:
    """Face detection via RetinaFace (ONNX). Scaffold — wire session in export step."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    def detect(self, image: np.ndarray) -> list[dict[str, object]]:
        raise NotImplementedError("Load ONNX session and implement detect()")
