import numpy as np
import pytest

from detection.face_aligner import FaceAligner
from detection.retinaface_detector import RetinaFaceDetector


def test_retinaface_detector_not_implemented():
    detector = RetinaFaceDetector()
    with pytest.raises(NotImplementedError):
        detector.detect(np.zeros((640, 480, 3), dtype=np.uint8))


def test_face_aligner_not_implemented():
    aligner = FaceAligner()
    with pytest.raises(NotImplementedError):
        aligner.align(np.zeros((112, 112, 3), dtype=np.uint8), np.zeros((5, 2)))
