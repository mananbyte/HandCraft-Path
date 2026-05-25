# HandCraft-Path Stack

## Environment
- Python 3.10
- conda env: pannuke-cuda122 (CUDA 12.2 compatible)
- RAPIDS 24.10 (cuML, cuDF, cuPy)

## Core ML
- scikit-learn 1.3.2
- lightgbm 4.1.0
- xgboost 2.0.2
- imbalanced-learn 0.11.0
- optuna 3.4.0

## Image processing
- scikit-image 0.22.0
- opencv 4.8.1
- staintools 2.1.2 (Macenko normalization)
- python-spams (staintools dependency)
- pydensecrf (from maintained fork)

## Feature selection
- mrmr-selection 0.2.8

## Data
- PanNuke dataset: Fold1, Fold2 (train), Fold3 (test — held out)
- 256×256 RGB H&E patches, 19 tissue types
- images.npy shape: (N, 256, 256, 3)
- masks.npy shape: (N, 256, 256, 6) — channels 0-4: nucleus types, 5: background

## Feature counts (fixed — never change after training)
- Dense features per pixel: 87
- GLCM per sampled pixel: 6
- Total Stage 1 features: 93
- Stage 2 nucleus region features: 40

## Key paths
- Raw data: data/raw/Fold/
- Processed arrays: data/processed/
- Saved models: data/models/
- Debug outputs: data/debug_samples/