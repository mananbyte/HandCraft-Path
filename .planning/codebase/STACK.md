---
document: STACK.md
focus: tech
mapped: 2026-05-25
---

# Technology Stack — HandCraft-Path (PanNuke Project)

## Language & Runtime

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11 | CPython; faster than 3.10 |
| CUDA | 12.x (12.5–12.9 verified) | Any modern NVIDIA GPU |
| Conda env name | `HandCraft-Path` | Managed via `environment.yml` |

## Conda Channels (priority order)

1. `rapidsai` — RAPIDS GPU stack
2. `conda-forge` — community packages
3. `nvidia` — CUDA drivers
4. ~~`defaults`~~ — **intentionally excluded** (conflicts with RAPIDS)

---

## Core ML Stack

| Package | Version | Role |
|---------|---------|------|
| **RAPIDS** | 25.12 | GPU metapackage — pulls cuML, cuDF, CuPy 13.x, RMM |
| `cuML` | 25.12 | GPU-accelerated ML (RandomForest, SVM) |
| `cuDF` | 25.12 | GPU dataframes (Arrow 18+) |
| `CuPy` | 13.x | GPU array computing (NumPy-compatible API) |
| `scikit-learn` | ≥1.5 | CPU ML fallback; full tag estimator API; RFE + ExtraTrees |
| `lightgbm` | ≥4.3 | GPU histogram trees; Stage 1 challenger model |
| `xgboost` | ≥2.1 (3.x bundled by RAPIDS) | GPU hist backend; Stage 2 challenger |
| `imbalanced-learn` | ≥0.12 | SMOTE for class imbalance |
| `optuna` | ≥3.5 | Bayesian HPO with TPE sampler |

## Image Processing Stack

| Package | Version | Role |
|---------|---------|------|
| `scikit-image` | ≥0.22 | LBP, Gabor, GLCM, structure tensor, SLIC, entropy |
| `opencv` | ≥4.8 | Image I/O, color conversions |
| `pillow` | ≥10.0 | Image utilities |
| `staintools` | 2.1.2 (pip) | Macenko stain normalization (SVD method; no SPAMS) |
| `pydensecrf` | maintained fork (pip) | Dense CRF post-processing for segmentation |

> **Note:** The project includes a self-contained `MacenkoNormalizer` in
> `src/preprocessing/stain_normalizer.py` using only NumPy — staintools is a
> secondary dependency.

## Feature Selection

| Package | Version | Role |
|---------|---------|------|
| `mrmr-selection` | 0.2.8 (pip) | Minimum Redundancy Maximum Relevance feature selection |

## Utilities

| Package | Role |
|---------|------|
| `matplotlib` ≥3.8 | Debug visualizations and paper figures |
| `tqdm` | Progress bars in training loops |
| `jupyterlab` ≥4.0 | Notebook environment for inspection |
| `psutil` | RAM/process memory monitoring (MemoryConfig) |
| `joblib` | Model serialization (save/load .joblib artifacts) |

## Inference Server (Optional)

| Package | Version | Status |
|---------|---------|--------|
| `fastapi` | 0.104.1 (pip) | REST API server — build last, after evaluation |
| `uvicorn` | 0.24.0 (pip) | ASGI server for FastAPI |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `environment.yml` | Reproducible conda environment (Python 3.11, RAPIDS 25.12, CUDA 12.x) |
| `.vscode/settings.json` | Editor settings |
| `.gitignore` | Excludes `.venv/`, `data/`, `__pycache__/`, model artifacts |

## Python Environment Setup

```bash
conda env create -f environment.yml --solver=libmamba
conda activate HandCraft-Path
```

## Key Dependency Constraints

- **numpy**: Owned by RAPIDS 25.12 — do NOT pin separately (gets NumPy 2.x)
- **scipy**: Same — let RAPIDS manage
- `python-spams` removed — conflicts with NumPy 2.x; Macenko uses SVD (no SPAMS needed)
- `pydensecrf 1.0rc3` broken on Py3.11 — uses maintained fork from `lucasb-eyer`

---

## Fixed Feature Counts (Never change after training)

| Feature Group | Count | Description |
|--------------|-------|-------------|
| Dense per pixel | 87 | Full-image features (OD, color, LBP, Gabor, gradients, etc.) |
| GLCM per sampled pixel | 6 | Texture regularity (computed only for sampled pixels) |
| **Total Stage 1** | **93** | Input to pixel classifier |
| Stage 2 nucleus region | 40 | Morphology + color + LBP + context ring |

## GPU Acceleration Pattern

The project uses a **hybrid GPU/CPU pattern**:
- GPU-accelerated: windowed color stats (54 features), LoG gradients, distance transform
- CPU-only: LBP, Gabor, SLIC superpixels (incompatible with CuPy API)
- Graceful fallback: all GPU paths have CPU fallback if CuPy unavailable or GPU fails

```python
# Pattern used throughout — detected at import time
try:
    import cupy as cp
    _ = cp.cuda.runtime.getDeviceCount()
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False
```
