# ai_pipeline/src/recognition/arcface_embedder.py
import numpy as np
import cv2
import onnxruntime as ort
from ai_pipeline.configs.settings import settings
from ai_pipeline.src.utils.logger import logger


class ArcFaceEmbedder:
    """
    Runs pruned + INT8 quantized ArcFace ResNet50.
    Produces L2-normalized 512-d face embeddings.
    """

    def __init__(self) -> None:
        model_path = settings.model_dir / settings.arcface_model
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"ArcFace loaded: {model_path.name}")

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = (face.astype(np.float32) - 127.5) / 127.5   # normalize to [-1, 1]
        face = face.transpose(2, 0, 1)[np.newaxis]           # HWC → NCHW
        return face

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        blob = self.preprocess(face_bgr)
        emb = self.session.run(None, {self.input_name: blob})[0][0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)            # L2 normalize
        return emb.astype(np.float32)

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2))                     # both L2-normalized


def is_match(emb1: np.ndarray, emb2: np.ndarray) -> tuple[bool, float]:
    score = ArcFaceEmbedder.cosine_similarity(emb1, emb2)
    return score >= settings.similarity_threshold, score