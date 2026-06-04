# ai_pipeline/tests/test_liveness.py
import numpy as np
from ai_pipeline.src.liveness.active_liveness import eye_aspect_ratio, head_yaw_degrees

def test_ear_closed_eye():
    # Flat eye — all points on same horizontal line
    pts = np.array([[0,0],[1,0],[2,0],[3,0],[2,0],[1,0]], dtype=np.float32)
    assert eye_aspect_ratio(pts) < 0.1

def test_head_yaw_frontal():
    # Symmetric landmarks → near-zero yaw
    lm = np.array([[30,50],[70,50],[50,70],[35,90],[65,90]], dtype=np.float32)
    yaw = head_yaw_degrees(lm)
    assert abs(yaw) < 10