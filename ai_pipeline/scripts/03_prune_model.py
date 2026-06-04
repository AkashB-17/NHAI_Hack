# ai_pipeline/scripts/03_prune_model.py
"""
Applies structured L1-norm channel pruning to ArcFace ResNet50.
Target: reduce from ~97 MB to ~25 MB before quantization.
Run: uv run python ai_pipeline/scripts/03_prune_model.py
"""
import torch
import torch.nn.utils.prune as prune
from pathlib import Path
from ai_pipeline.src.utils.logger import logger

WEIGHTS_DIR = Path("ai_pipeline/models/weights")
PRUNED_DIR  = Path("ai_pipeline/models/weights")  # same dir, different filename


def prune_arcface(amount: float = 0.45) -> None:
    """
    amount: fraction of channels to prune (0.45 = remove 45%).
    Tune this until you hit ~25 MB while keeping LFW accuracy > 97%.
    """
    from backbones import get_model
    model = get_model("r50", fp16=False)
    state = torch.load(WEIGHTS_DIR / "arcface_r50_ms1mv3.pth", map_location="cpu")
    model.load_state_dict(state)

    # Apply L1 unstructured pruning to all Conv2d layers
    params_to_prune = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Make pruning permanent (remove masks, bake zeros)
    for module, _ in params_to_prune:
        prune.remove(module, "weight")

    out_path = WEIGHTS_DIR / f"arcface_r50_pruned_{int(amount*100)}pct.pth"
    torch.save(model.state_dict(), out_path)

    size_mb = out_path.stat().st_size / 1e6
    logger.success(f"Pruned model saved → {out_path} ({size_mb:.1f} MB)")
    logger.info("Next step: fine-tune 3-5 epochs on MS1MV3 subset to recover accuracy")


if __name__ == "__main__":
    prune_arcface(amount=0.45)