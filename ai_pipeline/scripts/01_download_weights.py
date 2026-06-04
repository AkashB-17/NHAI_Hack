# ai_pipeline/scripts/01_download_weights.py
"""
Downloads pretrained weights from InsightFace model zoo.
Run once: uv run python ai_pipeline/scripts/01_download_weights.py
"""
import hashlib
import urllib.request
from pathlib import Path
from ai_pipeline.src.utils.logger import logger

WEIGHTS_DIR = Path("ai_pipeline/models/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "retinaface_mnet025_v2.pth": {
        "url": "https://github.com/biubug6/Pytorch_Retinaface/releases/download/v1.0/mobilenet0.25_Final.pth",
        "sha256": None,  # fill after first download
    },
    "arcface_r50_ms1mv3.pth": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/glint360k_cosface_r50_fp16_0.1.zip",
        "sha256": None,
        "note": "Manual download from InsightFace model zoo — see README",
    },
}


def download(name: str, url: str, sha256: str | None = None) -> Path:
    dest = WEIGHTS_DIR / name
    if dest.exists():
        logger.info(f"Already exists: {name}")
        return dest
    logger.info(f"Downloading {name}...")
    urllib.request.urlretrieve(url, dest)
    if sha256:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        assert digest == sha256, f"Checksum mismatch for {name}"
        logger.success(f"Checksum verified: {name}")
    return dest


if __name__ == "__main__":
    for name, meta in MODELS.items():
        if "note" in meta:
            logger.warning(f"{name}: {meta['note']}")
            continue
        download(name, meta["url"], meta.get("sha256"))
    logger.success("Weight download complete.")