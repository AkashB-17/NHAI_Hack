# ai_pipeline/configs/settings.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Paths
    model_dir: Path = Path("ai_pipeline/models/quantized")
    retinaface_model: str = "retinaface_mnet025_int8.onnx"
    arcface_model: str = "arcface_r50_pruned_int8.onnx"
    liveness_model: str = "liveness_mnetv2_int8.onnx"

    # Inference
    det_input_size: tuple[int, int] = (640, 640)
    rec_input_size: tuple[int, int] = (112, 112)
    liveness_input_size: tuple[int, int] = (224, 224)
    similarity_threshold: float = 0.60
    liveness_threshold: float = 0.70
    det_confidence: float = 0.80

    # Active liveness
    blink_ear_threshold: float = 0.25
    head_turn_yaw_deg: float = 15.0

    # Logging
    log_level: str = "INFO"


settings = Settings()