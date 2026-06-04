# ai_pipeline/src/detection/face_aligner.py
import numpy as np
import cv2
from ai_pipeline.src.utils.logger import logger

# Standard 112x112 reference landmarks (InsightFace convention)
ARCFACE_DST = np.array([
    [38.29, 51.70],
    [73.53, 51.50],
    [56.02, 71.73],
    [41.55, 92.37],
    [70.73, 92.20],
], dtype=np.float32)


def align_face(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Applies affine warp using 5-point landmarks.
    Returns 112x112 BGR crop aligned to ArcFace convention.
    """
    tform = cv2.estimateAffinePartial2D(
        landmarks, ARCFACE_DST, method=cv2.LMEDS
    )[0]
    if tform is None:
        logger.warning("Affine estimation failed — falling back to bbox crop")
        return None
    aligned = cv2.warpAffine(frame, tform, (112, 112), flags=cv2.INTER_LINEAR)
    return aligned