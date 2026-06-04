import numpy as np


class ActiveLiveness:
    """Challenge-response liveness (blink, head turn, etc.)."""

    def verify_challenge(self, frames: list[np.ndarray], challenge: str) -> bool:
        raise NotImplementedError("Implement challenge verification")
