# NHAI Face Auth — AI Pipeline

Offline facial recognition and liveness detection (ONNX Route 2).

## Layout

```
ai_pipeline/
├── configs/          # Pydantic settings
├── models/
│   ├── weights/      # .pth (gitignored — download via script)
│   ├── onnx/         # exported ONNX (gitignored)
│   └── quantized/    # INT8 ONNX for device (committed)
├── src/              # detection, recognition, liveness, utils
├── scripts/          # download → export → prune → quantize → validate
├── tests/
├── notebooks/
└── benchmarks/
```

## Setup

```bash
uv sync --all-extras
cp .env.example .env
```

## Scripts

Run from repo root:

```bash
uv run python ai_pipeline/scripts/01_download_weights.py
uv run python ai_pipeline/scripts/02_export_to_onnx.py
```

## Tests

```bash
uv run pytest
```
