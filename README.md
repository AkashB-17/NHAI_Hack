# NHAI Face Auth — AI Pipeline

Offline facial recognition and liveness detection .


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
