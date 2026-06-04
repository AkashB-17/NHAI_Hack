from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_DIR = _REPO_ROOT / "ai_pipeline" / "models" / "quantized"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_dir: Path = Field(default=_DEFAULT_MODEL_DIR, alias="MODEL_DIR")
    similarity_threshold: float = Field(default=0.60, alias="SIMILARITY_THRESHOLD")
    liveness_threshold: float = Field(default=0.70, alias="LIVENESS_THRESHOLD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
