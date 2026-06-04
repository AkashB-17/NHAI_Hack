import numpy as np


class PassiveLiveness:
    """Texture / frequency-based anti-spoofing (single frame)."""

    def score(self, face_crop: np.ndarray) -> float:
        raise NotImplementedError("Implement passive liveness scorer")
