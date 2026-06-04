"""INT8 dynamic quantization → ai_pipeline/models/quantized/."""

from pathlib import Path

from loguru import logger

QUANTIZED_DIR = Path(__file__).resolve().parents[1] / "models" / "quantized"


def main() -> None:
    QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Quantized output directory: {}", QUANTIZED_DIR)
    logger.warning("Implement onnxruntime quantization")


if __name__ == "__main__":
    main()
