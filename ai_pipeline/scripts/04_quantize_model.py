# ai_pipeline/scripts/04_quantize_model.py
"""
Applies dynamic INT8 quantization to all three ONNX models.
No calibration dataset needed (weight-only quantization).
Run: uv run python ai_pipeline/scripts/04_quantize_model.py
"""
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
from onnxsim import simplify
from ai_pipeline.src.utils.logger import logger

ONNX_DIR      = Path("ai_pipeline/models/onnx")
QUANTIZED_DIR = Path("ai_pipeline/models/quantized")
QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)


def simplify_and_quantize(src_name: str, dst_name: str) -> None:
    src = ONNX_DIR / src_name
    simplified_path = ONNX_DIR / f"simplified_{src_name}"
    dst = QUANTIZED_DIR / dst_name

    # Step 1: Simplify graph (fold constants, remove dead nodes)
    logger.info(f"Simplifying {src_name}...")
    model = onnx.load(str(src))
    simplified, check = simplify(model)
    assert check, "Simplification check failed"
    onnx.save(simplified, str(simplified_path))
    logger.info(f"Simplified: {simplified_path.stat().st_size / 1e6:.1f} MB")

    # Step 2: Dynamic INT8 quantization (weights only, fast on CPU)
    logger.info(f"Quantizing {src_name} → {dst_name}...")
    quantize_dynamic(
        model_input=str(simplified_path),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        optimize_model=True,
        per_channel=True,          # better accuracy than per-tensor
        reduce_range=True,         # needed for x86 VNNI; harmless on ARM
    )
    final_size = dst.stat().st_size / 1e6
    logger.success(f"Quantized model → {dst} ({final_size:.1f} MB)")


if __name__ == "__main__":
    models = [
        ("retinaface_mnet025.onnx",       "retinaface_mnet025_int8.onnx"),
        ("arcface_r50_pruned.onnx",        "arcface_r50_pruned_int8.onnx"),
        ("liveness_mnetv2.onnx",           "liveness_mnetv2_int8.onnx"),
    ]
    for src, dst in models:
        simplify_and_quantize(src, dst)

    # Print total bundle size
    total = sum(
        (QUANTIZED_DIR / dst).stat().st_size
        for _, dst in models
        if (QUANTIZED_DIR / dst).exists()
    )
    logger.info(f"Total quantized bundle: {total / 1e6:.1f} MB")