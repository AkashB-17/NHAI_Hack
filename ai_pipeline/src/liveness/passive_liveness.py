# ai_pipeline/src/liveness/passive_liveness.py
import numpy as np
import cv2
import onnxruntime as ort
from ai_pipeline.configs.settings import settings
from ai_pipeline.src.utils.logger import logger


class PassiveLiveness:
    """
    MobileNetV2 binary classifier trained on CelebA-Spoof.
    Input: 112x112 aligned face crop.
    Output: probability of being a real face (not a spoof).
    """

    def __init__(self) -> None:
        model_path = settings.model_dir / settings.liveness_model
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"Passive liveness loaded: {model_path.name}")

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        face = (face - mean) / std
        return face.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def predict(self, face_bgr: np.ndarray) -> tuple[bool, float]:
        blob  = self.preprocess(face_bgr)
        logit = self.session.run(None, {self.input_name: blob})[0][0]
        prob  = float(1 / (1 + np.exp(-logit)))     # sigmoid
        is_real = prob >= settings.liveness_threshold
        return is_real, prob