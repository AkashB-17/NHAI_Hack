import numpy as np
from PIL import Image


def load_rgb(path: str) -> np.ndarray:
    """Load image as RGB uint8 array (H, W, 3)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))
