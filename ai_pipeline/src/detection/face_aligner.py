import numpy as np


class FaceAligner:
    """Align face crops to a canonical pose for embedding."""

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement similarity transform from landmarks")
