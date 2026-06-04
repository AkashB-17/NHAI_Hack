from pathlib import Path

import numpy as np


class EmbeddingStore:
    """Persist and query encrypted face embeddings."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def save(self, user_id: str, embedding: np.ndarray) -> None:
        raise NotImplementedError("Implement SQLite + encryption backend")

    def find_match(
        self, embedding: np.ndarray, threshold: float
    ) -> tuple[str | None, float]:
        raise NotImplementedError("Implement cosine similarity search")
