# ai_pipeline/tests/test_recognition.py
import numpy as np
from ai_pipeline.src.recognition.arcface_embedder import ArcFaceEmbedder

def test_cosine_same_embedding():
    emb = np.random.rand(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    score = ArcFaceEmbedder.cosine_similarity(emb, emb)
    assert abs(score - 1.0) < 1e-5

def test_cosine_orthogonal():
    e1 = np.zeros(512, dtype=np.float32); e1[0] = 1.0
    e2 = np.zeros(512, dtype=np.float32); e2[1] = 1.0
    score = ArcFaceEmbedder.cosine_similarity(e1, e2)
    assert abs(score) < 1e-5