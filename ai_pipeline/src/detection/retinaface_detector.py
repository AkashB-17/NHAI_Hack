# ai_pipeline/src/detection/retinaface_detector.py
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
from ai_pipeline.configs.settings import settings
from ai_pipeline.src.utils.logger import logger


class RetinaFaceDetector:
    """
    Runs RetinaFace-MobileNet0.25 INT8 ONNX model.
    Returns bounding boxes and 5-point landmarks.
    """

    def __init__(self) -> None:
        model_path = settings.model_dir / settings.retinaface_model
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2          # safe for mid-range mobile CPU
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.input_size  = settings.det_input_size   # (640, 640)
        self.conf_thresh = settings.det_confidence
        logger.info(f"RetinaFace loaded: {model_path.name}")

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame.shape[:2]
        scale_x = self.input_size[0] / w
        scale_y = self.input_size[1] / h
        resized = cv2.resize(frame, self.input_size)
        blob = resized.astype(np.float32)
        blob -= np.array([104.0, 117.0, 123.0], dtype=np.float32)  # mean subtraction
        blob = blob.transpose(2, 0, 1)[np.newaxis]                  # HWC → NCHW
        return blob, scale_x, scale_y

    def postprocess(
        self,
        loc: np.ndarray,
        conf: np.ndarray,
        landms: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> list[dict]:
        """Decode anchor offsets → absolute pixel boxes + landmarks."""
        detections = []
        scores = conf[0, :, 1]                                      # class=1 is face
        mask = scores > self.conf_thresh
        for i in np.where(mask)[0]:
            x1, y1, x2, y2 = loc[0, i] / np.array([scale_x, scale_y, scale_x, scale_y])
            pts = landms[0, i].reshape(5, 2) / np.array([scale_x, scale_y])
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(scores[i]),
                "landmarks": pts.astype(np.float32),
            })
        return sorted(detections, key=lambda d: d["score"], reverse=True)

    def detect(self, frame: np.ndarray) -> list[dict]:
        blob, sx, sy = self.preprocess(frame)
        loc, conf, landms = self.session.run(None, {self.input_name: blob})
        return self.postprocess(loc, conf, landms, sx, sy)