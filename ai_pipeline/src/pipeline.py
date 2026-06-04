# ai_pipeline/src/pipeline.py
"""
Top-level orchestrator: detect → align → liveness → embed → match.
This is what the React Native bridge calls via JNI / react-native-onnxruntime.
"""
import time
import numpy as np
from dataclasses import dataclass
from ai_pipeline.src.detection.retinaface_detector import RetinaFaceDetector
from ai_pipeline.src.detection.face_aligner        import align_face
from ai_pipeline.src.recognition.arcface_embedder  import ArcFaceEmbedder, is_match
from ai_pipeline.src.recognition.embedding_store   import EmbeddingStore
from ai_pipeline.src.liveness.passive_liveness     import PassiveLiveness
from ai_pipeline.src.liveness.active_liveness      import ActiveLiveness
from ai_pipeline.src.utils.logger import logger


@dataclass
class AuthResult:
    success:        bool
    user_id:        str | None
    user_name:      str | None
    similarity:     float
    liveness_score: float
    latency_ms:     float
    reason:         str


class FaceAuthPipeline:
    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.detector    = RetinaFaceDetector()
        self.embedder    = ArcFaceEmbedder()
        self.passive     = PassiveLiveness()
        self.active      = ActiveLiveness()
        self.store       = EmbeddingStore()
        logger.success(f"Pipeline loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

    def enroll(self, frame: np.ndarray, user_id: str, user_name: str) -> bool:
        faces = self.detector.detect(frame)
        if not faces:
            logger.warning("Enrollment failed: no face detected")
            return False
        aligned = align_face(frame, faces[0]["landmarks"])
        if aligned is None:
            return False
        emb = self.embedder.embed(aligned)
        self.store.enroll(user_id, user_name, emb)
        return True

    def authenticate(self, frame: np.ndarray) -> AuthResult:
        t0 = time.perf_counter()

        faces = self.detector.detect(frame)
        if not faces:
            return AuthResult(False, None, None, 0.0, 0.0, _ms(t0), "no_face_detected")

        best = faces[0]
        aligned = align_face(frame, best["landmarks"])
        if aligned is None:
            return AuthResult(False, None, None, 0.0, 0.0, _ms(t0), "alignment_failed")

        # Passive liveness check
        is_real, live_score = self.passive.predict(aligned)
        if not is_real:
            return AuthResult(False, None, None, 0.0, live_score, _ms(t0), "spoof_detected")

        # Embed and match
        emb        = self.embedder.embed(aligned)
        candidates = self.store.get_all()
        best_score = 0.0
        best_user  = None

        for candidate in candidates:
            matched, score = is_match(emb, candidate["embedding"])
            if score > best_score:
                best_score = score
                best_user  = candidate if matched else None

        latency = _ms(t0)
        logger.info(f"Auth: {'PASS' if best_user else 'FAIL'} | score={best_score:.3f} | {latency:.0f}ms")

        if best_user:
            return AuthResult(True, best_user["id"], best_user["name"], best_score, live_score, latency, "ok")
        return AuthResult(False, None, None, best_score, live_score, latency, "no_match")


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000