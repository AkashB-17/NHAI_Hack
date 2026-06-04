# ai_pipeline/scripts/02_export_to_onnx.py
"""
Exports PyTorch RetinaFace and ArcFace models to ONNX format.
Run: uv run python ai_pipeline/scripts/02_export_to_onnx.py
"""
import torch
from pathlib import Path
from ai_pipeline.src.utils.logger import logger

WEIGHTS_DIR = Path("ai_pipeline/models/weights")
ONNX_DIR    = Path("ai_pipeline/models/onnx")
ONNX_DIR.mkdir(parents=True, exist_ok=True)


def export_retinaface() -> Path:
    from retinaface_model import RetinaFace  # from local clone of Pytorch_Retinaface
    model = RetinaFace(cfg=None, phase="test")
    state = torch.load(WEIGHTS_DIR / "retinaface_mnet025_v2.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 640, 640)
    out_path = ONNX_DIR / "retinaface_mnet025.onnx"

    torch.onnx.export(
        model, dummy, out_path,
        opset_version=12,
        input_names=["input"],
        output_names=["loc", "conf", "landms"],
        dynamic_axes={"input": {0: "batch"}},
        do_constant_folding=True,
    )
    logger.success(f"RetinaFace exported → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def export_arcface() -> Path:
    from backbones import get_model  # from InsightFace
    model = get_model("r50", fp16=False)
    state = torch.load(WEIGHTS_DIR / "arcface_r50_ms1mv3.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 112, 112)
    out_path = ONNX_DIR / "arcface_r50.onnx"

    torch.onnx.export(
        model, dummy, out_path,
        opset_version=12,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}},
        do_constant_folding=True,
    )
    logger.success(f"ArcFace exported → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    logger.info("Exporting RetinaFace...")
    export_retinaface()
    logger.info("Exporting ArcFace ResNet50...")
    export_arcface()

