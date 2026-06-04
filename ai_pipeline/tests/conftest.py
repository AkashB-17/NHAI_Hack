import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

for path in (_SRC, _CONFIGS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def sample_rgb_array():
    import numpy as np

    return np.zeros((112, 112, 3), dtype=np.uint8)
