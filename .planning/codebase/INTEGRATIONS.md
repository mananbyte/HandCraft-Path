---
document: INTEGRATIONS.md
focus: tech
mapped: 2026-05-25
---

# External Integrations — HandCraft-Path (PanNuke Project)

## Datasets

### PanNuke Dataset
- **Type**: Public histopathology dataset (H&E stained)
- **Source**: Downloaded manually; stored at `data/raw/`
- **Format**: NumPy `.npy` files — accessed via `mmap_mode='r'` (never fully loaded)
- **Structure**:
  ```
  data/raw/Fold1/images/fold1/images.npy   # shape: (N, 256, 256, 3) float32
  data/raw/Fold1/images/fold1/types.npy    # tissue type labels
  data/raw/Fold1/masks/fold1/masks.npy     # shape: (N, 256, 256, 6) float32
  ```
- **Content**: 256×256 RGB H&E patches from 19 tissue types
- **Mask channels**: 0-4 = nucleus type instances (neoplastic, inflammatory, connective, dead, epithelial); 5 = background
- **Split**: Fold1+Fold2 = train, Fold3 = **held out test** (never used until final evaluation)

## GPU / Compute Integrations

### RAPIDS (NVIDIA)
- **Package**: `rapids=25.12` via `rapidsai` conda channel
- **Components used**:
  - `cuML` — GPU Random Forest, SVM (cuML alternatives)
  - `cuDF` — GPU DataFrames
  - `CuPy` — GPU array ops (direct usage in feature extraction)
- **Integration pattern**: Optional — all GPU paths fall back to CPU if CUDA unavailable
- **VRAM detection**: `cp.cuda.runtime.memGetInfo()` per device

### CUDA Runtime
- **Version**: CUDA 12.x (12.5–12.9 verified)
- **Detection**: `cp.cuda.runtime.getDeviceCount()` at import time
- **Distance transform GPU**: `cupyx.scipy.ndimage.distance_transform_edt`
- **Gaussian ops GPU**: `cupyx.scipy.ndimage.uniform_filter`, `gaussian_laplace`

## Serialization / Persistence

### joblib
- **Role**: Save/load all fitted ML artifacts
- **Artifacts saved**:
  ```
  data/models/normalizer.joblib           # MacenkoNormalizer (fitted on Fold1+2)
  data/models/scaler_stage1_binary.joblib # StandardScaler
  data/models/selector_stage1_binary.joblib # Feature selector
  data/models/rf_stage1_binary.joblib     # Random Forest (base)
  data/models/rf_stage1_binary_tuned.joblib  # Random Forest (HPO tuned)
  data/models/lgbm_stage1_binary.joblib   # LightGBM
  data/models/optuna_study_binary.joblib  # Optuna study (HPO history)
  data/models/rf_stage2.joblib            # Stage 2 nucleus classifier
  data/models/svm_stage2.joblib           # Stage 2 SVM challenger
  ```
- **Convention**: filename must include experiment mode (e.g., `_binary`, `_3class`)

### NumPy memmap
- **Role**: Large dataset persistence (streaming write during feature extraction)
- **Files**: `data/processed/{split}_{mode}_X.npy`, `data/processed/{split}_{mode}_y.npy`
- **Available processed data**:
  ```
  data/processed/fold1_binary_X.npy
  data/processed/fold1_binary_y.npy
  data/processed/fold2_binary_X.npy
  data/processed/fold2_binary_y.npy
  ```
- **Access pattern**: `np.lib.format.open_memmap(path, mode='r+', dtype=np.float32, shape=(N, 93))`

### Checkpoint JSON
- **Pattern**: Atomic write (`tmp → os.replace()`) every 100 images
- **Location**: `data/processed/checkpoint_{split}_{mode}.json`
- **Purpose**: Crash recovery — resume from last completed image

## Hyperparameter Optimization

### Optuna
- **Version**: ≥3.5
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Storage**: In-memory (saved to `data/models/optuna_study_{mode}.joblib`)
- **Metric optimized**: Macro-F1 (Stage 1)

## Optional Inference Server

### FastAPI + Uvicorn
- **Status**: Declared in `environment.yml` but `src/inference/pipeline.py` is empty
- **Intent**: REST API endpoint for segmentation inference
- **Not yet implemented**

## External References / Papers

- Macenko et al., "A method for normalizing histology slides for quantitative analysis", ISBI 2009 — implemented in `src/preprocessing/stain_normalizer.py`
- `pydensecrf` — Lucas Beyer's maintained fork (from `git+https://github.com/lucasb-eyer/pydensecrf.git`)

## No Cloud/API Dependencies

- **No cloud storage** — all data is local
- **No external APIs** — fully offline pipeline
- **No database** — `.npy` + `.joblib` files only
- **No auth providers**
- **No message queues**
- **No webhooks**
