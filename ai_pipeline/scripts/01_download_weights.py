"""Download pretrained .pth weights into ai_pipeline/models/weights/."""

from pathlib import Path

from loguru import logger

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "models" / "weights"


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Weights directory: {}", WEIGHTS_DIR)
    logger.warning("Implement download URLs (RetinaFace, ArcFace, liveness)")


if __name__ == "__main__":
    main()
