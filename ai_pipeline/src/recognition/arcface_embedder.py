from pathlib import Path

import numpy as np


class ArcFaceEmbedder:
    """512-d face embeddings via ArcFace (ONNX)."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Load ONNX session and implement embed()")
