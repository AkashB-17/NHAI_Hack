# ai_pipeline/src/liveness/active_liveness.py
"""
Geometric liveness challenges — no neural model required.
Uses 68-point landmark positions (or MediaPipe 468-point mesh if available).
Challenges: blink (EAR < threshold) + head turn (yaw > threshold degrees).
"""
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from ai_pipeline.configs.settings import settings
from ai_pipeline.src.utils.logger import logger


class Challenge(Enum):
    BLINK     = auto()
    HEAD_LEFT = auto()
    HEAD_RIGHT = auto()
    SMILE     = auto()


@dataclass
class LivenessState:
    challenges: list[Challenge] = field(default_factory=lambda: [Challenge.BLINK, Challenge.HEAD_LEFT])
    completed:  set[Challenge]  = field(default_factory=set)
    attempts:   int             = 0
    max_attempts: int           = 30    # frames


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    eye_pts: shape (6, 2) — standard 68-pt landmark order
    """
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def head_yaw_degrees(landmarks_5pt: np.ndarray) -> float:
    """
    Approximate yaw from 5-point landmarks:
    left_eye, right_eye, nose, left_mouth, right_mouth
    Positive = turned right, Negative = turned left
    """
    left_eye   = landmarks_5pt[0]
    right_eye  = landmarks_5pt[1]
    nose       = landmarks_5pt[2]
    eye_center = (left_eye + right_eye) / 2.0
    dx = nose[0] - eye_center[0]
    dy = right_eye[0] - left_eye[0] + 1e-6
    return float(np.degrees(np.arctan2(dx, dy)))


class ActiveLiveness:
    def __init__(self) -> None:
        self.state = LivenessState()

    def reset(self) -> None:
        self.state = LivenessState()

    def check_frame(self, landmarks_5pt: np.ndarray) -> dict:
        self.state.attempts += 1
        result = {
            "passed": False,
            "completed": list(self.state.completed),
            "pending": [c for c in self.state.challenges if c not in self.state.completed],
            "expired": self.state.attempts >= self.state.max_attempts,
        }

        yaw = head_yaw_degrees(landmarks_5pt)

        if Challenge.HEAD_LEFT in result["pending"] and yaw < -settings.head_turn_yaw_deg:
            self.state.completed.add(Challenge.HEAD_LEFT)
            logger.debug(f"Challenge passed: HEAD_LEFT (yaw={yaw:.1f}°)")

        if Challenge.HEAD_RIGHT in result["pending"] and yaw > settings.head_turn_yaw_deg:
            self.state.completed.add(Challenge.HEAD_RIGHT)
            logger.debug(f"Challenge passed: HEAD_RIGHT (yaw={yaw:.1f}°)")

        result["completed"] = list(self.state.completed)
        result["pending"]   = [c for c in self.state.challenges if c not in self.state.completed]
        result["passed"]    = len(result["pending"]) == 0
        return result