# HandCraft-Path Stack

## Environment
- Python 3.11
- conda env: `HandCraft-Path` (CUDA 12.x compatible)
- RAPIDS 25.12 (cuML, cuDF, CuPy 13.x)

## Core ML
- scikit-learn >=1.5
- lightgbm >=4.3
- xgboost >=2.1
- imbalanced-learn >=0.12
- optuna >=3.5

## Image processing
- scikit-image >=0.22
- opencv >=4.8
- staintools 2.1.2 (Macenko SVD normalization via NumPy)
- pydensecrf (from maintained fork)

## Feature selection
- mrmr-selection 0.2.8
- scikit-learn RFE (GPU-accelerated via cuML LogisticRegression)

## Data
- PanNuke dataset: Fold1, Fold2 (train), Fold3 (test — held out)
- 256×256 RGB H&E patches, 19 tissue types
- images.npy shape: (N, 256, 256, 3)
- masks.npy shape: (N, 256, 256, 6) — channels 0-4: nucleus types, 5: background

## Feature counts & Selection (Phase 3 locked)
- Total candidate Stage 1 features: 93
- Standardization: `HandCraftPathScaler` (group-specific multi-scaler)
- Selected features: **25 features** (method: `rfe` on GPU)
- Stage 2 nucleus region features: 40

## Key paths
- Raw data: `data/raw/`
- Processed arrays: `data/processed/`
- Saved models: `data/models/`
- Debug outputs: `data/debug_samples/`