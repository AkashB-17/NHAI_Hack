# ai_pipeline/src/recognition/embedding_store.py
"""
SQLite-backed encrypted embedding store.
Embeddings are AES-256-GCM encrypted before write.
Schema: users(id, name, embedding_enc BLOB, created_at, synced BOOL)
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from ai_pipeline.src.utils.logger import logger

DB_PATH  = Path("ai_pipeline/data/embeddings.db")
KEY_PATH = Path("ai_pipeline/data/.aes_key")   # gitignored


def _load_or_create_key() -> bytes:
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if KEY_PATH.exists():
        return KEY_PATH.read_bytes()
    key = AESGCM.generate_key(bit_length=256)
    KEY_PATH.write_bytes(key)
    logger.warning("New AES-256 key generated — back it up securely")
    return key


class EmbeddingStore:
    def __init__(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.key    = _load_or_create_key()
        self.aesgcm = AESGCM(self.key)
        self.conn   = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  TEXT NOT NULL,
                synced      INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def _encrypt(self, emb: np.ndarray) -> bytes:
        nonce = AESGCM.generate_key(bit_length=96)[:12]     # 96-bit nonce
        data  = emb.tobytes()
        ct    = self.aesgcm.encrypt(nonce, data, None)
        return nonce + ct

    def _decrypt(self, blob: bytes) -> np.ndarray:
        nonce, ct = blob[:12], blob[12:]
        data = self.aesgcm.decrypt(nonce, ct, None)
        return np.frombuffer(data, dtype=np.float32)

    def enroll(self, user_id: str, name: str, embedding: np.ndarray) -> None:
        enc = self._encrypt(embedding)
        self.conn.execute(
            "INSERT OR REPLACE INTO users (id, name, embedding, created_at) VALUES (?,?,?,?)",
            (user_id, name, enc, datetime.utcnow().isoformat()),
        )
        self.conn.commit()
        logger.info(f"Enrolled: {name} ({user_id})")

    def get_all(self) -> list[dict]:
        rows = self.conn.execute("SELECT id, name, embedding FROM users").fetchall()
        return [
            {"id": r[0], "name": r[1], "embedding": self._decrypt(r[2])}
            for r in rows
        ]

    def get_unsynced(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, embedding FROM users WHERE synced=0"
        ).fetchall()
        return [{"id": r[0], "name": r[1]} for r in rows]

    def mark_synced(self, user_ids: list[str]) -> None:
        self.conn.executemany(
            "UPDATE users SET synced=1 WHERE id=?",
            [(uid,) for uid in user_ids],
        )
        self.conn.commit()

    def purge_synced(self) -> int:
        cur = self.conn.execute("DELETE FROM users WHERE synced=1")
        self.conn.commit()
        logger.info(f"Purged {cur.rowcount} synced records")
        return cur.rowcount