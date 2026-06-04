"""Export PyTorch checkpoints to ONNX under ai_pipeline/models/onnx/."""

from pathlib import Path

from loguru import logger

ONNX_DIR = Path(__file__).resolve().parents[1] / "models" / "onnx"


def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("ONNX output directory: {}", ONNX_DIR)
    logger.warning("Implement torch.onnx.export for each model")


if __name__ == "__main__":
    main()
