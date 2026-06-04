# NHAI_Hack – ONNX Face Authentication Pipeline

Offline Facial Recognition + Liveness Detection Pipeline optimized for edge deployment using ONNX Runtime.

## Overview

This project implements a complete face authentication system designed for low-latency, offline execution on resource-constrained devices.

The pipeline combines:

* RetinaFace MobileNet0.25 for face detection
* ArcFace ResNet50 for face recognition
* Passive liveness detection using MobileNetV2
* Active liveness verification using geometric challenges
* AES-256 encrypted local embedding storage
* ONNX Runtime INT8 inference for production deployment

The system is intended for secure identity verification scenarios where:

* Internet connectivity may be unavailable
* User privacy is critical
* Authentication must occur locally on-device
* Low latency and small model size are required

---

## Architecture

```text
Camera Frame
      │
      ▼
RetinaFace Detection
      │
      ▼
Face Alignment (5-point landmarks)
      │
      ▼
Passive Liveness Check
      │
      ▼
Active Liveness Challenge
      │
      ▼
ArcFace Embedding Generation
      │
      ▼
Encrypted Embedding Matching
      │
      ▼
Authentication Result
```

---

## Key Features

### Face Detection

* RetinaFace MobileNet0.25
* ONNX Runtime inference
* 5-point facial landmark extraction
* CPU-optimized execution

### Face Recognition

* ArcFace ResNet50 embeddings
* 512-dimensional feature vectors
* L2 normalization
* Cosine similarity matching

### Liveness Detection

#### Passive Liveness

* MobileNetV2 binary classifier
* Spoof detection
* Print attack resistance
* Screen replay attack resistance

#### Active Liveness

* Eye blink detection
* Head movement verification
* Geometric challenge-response system
* No neural network required

### Security

* AES-256-GCM embedding encryption
* Local SQLite storage
* Offline authentication
* No biometric data transmitted externally

### Optimization

* ONNX export
* Model pruning
* Graph simplification
* INT8 quantization
* CPU-only deployment

---

## Project Structure

```text
NHAI_Hack/
│
├── ai_pipeline/
│   ├── configs/
│   ├── models/
│   │   ├── weights/
│   │   ├── onnx/
│   │   └── quantized/
│   │
│   ├── src/
│   │   ├── detection/
│   │   ├── recognition/
│   │   ├── liveness/
│   │   └── utils/
│   │
│   ├── scripts/
│   ├── tests/
│   ├── notebooks/
│   └── benchmarks/
│
├── .github/workflows/
├── pyproject.toml
├── requirements.lock
└── README.md
```

---

## Technology Stack

| Component        | Technology             |
| ---------------- | ---------------------- |
| Language         | Python 3.11            |
| Package Manager  | UV                     |
| Inference Engine | ONNX Runtime           |
| Detection        | RetinaFace             |
| Recognition      | ArcFace                |
| Liveness         | MobileNetV2 + Geometry |
| Storage          | SQLite                 |
| Encryption       | AES-256-GCM            |
| Testing          | PyTest                 |
| Linting          | Ruff                   |
| Type Checking    | MyPy                   |
| CI/CD            | GitHub Actions         |

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/AkashB-17/NHAI_Hack.git

cd NHAI_Hack
```

### 2. Create Development Branch

```bash
git checkout -b dev
```

### 3. Create Virtual Environment

```bash
uv venv --python 3.11
```

Activate:

Linux/macOS

```bash
source .venv/bin/activate
```

Windows PowerShell

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
uv pip install -e ".[dev]"
```

### 5. Verify Setup

```bash
python --version
uv --version
```

---

## Environment Configuration

Create a `.env` file:

```env
MODEL_DIR=ai_pipeline/models/quantized

SIMILARITY_THRESHOLD=0.60

LIVENESS_THRESHOLD=0.70

LOG_LEVEL=INFO
```

---

## Model Preparation Workflow

### Step 1 – Download Weights

```bash
uv run python ai_pipeline/scripts/01_download_weights.py
```

### Step 2 – Export to ONNX

```bash
uv run python ai_pipeline/scripts/02_export_to_onnx.py
```

### Step 3 – Prune Models

```bash
uv run python ai_pipeline/scripts/03_prune_model.py
```

### Step 4 – Quantize Models

```bash
uv run python ai_pipeline/scripts/04_quantize_model.py
```

### Step 5 – Validate Accuracy

```bash
uv run python ai_pipeline/scripts/05_validate_accuracy.py \
--lfw-dir /path/to/lfw
```

---

## Running the Pipeline

### Enrollment

```python
from ai_pipeline.src.pipeline import FaceAuthPipeline
import cv2

pipeline = FaceAuthPipeline()

frame = cv2.imread("user.jpg")

pipeline.enroll(
    frame=frame,
    user_id="EMP001",
    user_name="John Doe"
)
```

### Authentication

```python
result = pipeline.authenticate(frame)

print(result.success)
print(result.user_name)
print(result.similarity)
```

---

## Testing

Run all tests:

```bash
uv run pytest -v
```

Run with coverage:

```bash
uv run pytest -v --cov=ai_pipeline/src
```

---

## Code Quality

### Ruff

```bash
uv run ruff check ai_pipeline/
```

### MyPy

```bash
uv run mypy ai_pipeline/src/
```

---

## CI/CD

GitHub Actions automatically performs:

* Dependency installation
* Ruff linting
* MyPy type checking
* Unit testing
* Coverage reporting

Triggered on:

* Push to `dev`
* Push to `main`
* Pull Requests into `main`

---

## Branch Strategy

```text
main
 └── Stable production releases

dev
 └── Active integration branch

feature/*
 └── Individual feature development
```

### Development Flow

```text
feature branch
      │
      ▼
Pull Request
      │
      ▼
dev
      │
      ▼
main
```

---

## Performance Targets

| Metric                 | Target    |
| ---------------------- | --------- |
| Authentication Latency | < 500 ms  |
| Face Detection         | < 150 ms  |
| Recognition            | < 100 ms  |
| Liveness Check         | < 100 ms  |
| Total Model Bundle     | < 50 MB   |
| Recognition Accuracy   | > 97% LFW |
| Offline Capability     | 100%      |

---

## Security Considerations

* Embeddings stored locally only
* AES-256-GCM encryption
* No raw facial images persisted
* Offline-first architecture
* No third-party authentication dependency

---

## Future Roadmap

* Android ONNX Runtime integration
* React Native bridge
* Face anti-spoofing improvements
* Model distillation
* ARM-specific optimizations
* Edge TPU compatibility
* Federated enrollment sync



