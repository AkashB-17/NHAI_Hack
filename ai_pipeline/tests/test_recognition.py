import numpy as np
import pytest

from recognition.arcface_embedder import ArcFaceEmbedder
from recognition.embedding_store import EmbeddingStore


def test_arcface_embed_not_implemented(sample_rgb_array):
    embedder = ArcFaceEmbedder()
    with pytest.raises(NotImplementedError):
        embedder.embed(sample_rgb_array)


def test_embedding_store_not_implemented(tmp_path):
    store = EmbeddingStore(tmp_path / "embeddings.db")
    with pytest.raises(NotImplementedError):
        store.save("user-1", np.zeros(512, dtype=np.float32))
