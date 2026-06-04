# ai_pipeline/tests/conftest.py
import pytest
import numpy as np

@pytest.fixture
def dummy_frame():
    """480x640 BGR frame with a white square simulating a face region."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:300, 200:440] = 255
    return frame

@pytest.fixture
def dummy_embedding():
    emb = np.random.rand(512).astype(np.float32)
    return emb / np.linalg.norm(emb)