---
document: ARCHITECTURE.md
focus: arch
mapped: 2026-05-25
---

# Architecture — HandCraft-Path (PanNuke Project)

## Project Goal

Research project investigating **hand-crafted feature ML** vs deep learning on H&E nucleus segmentation/typing:
1. How far do hand-crafted features go vs CNNs on PanNuke?
2. Does explicit boundary class (Exp B) improve over binary segmentation (Exp A)?
3. Does better segmentation improve downstream nucleus type classification?
4. Oracle gap: how much does Stage 1 error hurt Stage 2 accuracy?

---

## High-Level Architecture: Two-Stage Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1 — Pixel Classifier                      │
│                        (Nucleus Segmentation)                           │
├────────────────────────────────────────────────────────────────────────┤
│  256×256 RGB H&E image                                                 │
│         │                                                               │
│    [MacenkoNormalizer]  ← fitted on Fold1+2 subset                    │
│         │                                                               │
│    [ColorConverter]  → {rgb, lab, hsv, hed}  (4 color spaces)         │
│         │                                                               │
│    [PixelFeatureExtractor]  → 87 dense features per pixel (65536 rows) │
│         │                                                               │
│    [ActiveBoundaryMining]  → ~1200 sampled pixels (400/class)         │
│         │                                                               │
│    [GLCM]  → 6 features per sampled pixel                              │
│         │                                                               │
│    Concatenate → (N_sampled, 93) training rows                         │
│         │                                                               │
│    [RandomForest / LightGBM]  → pixel-level predictions                │
│                                                                         │
│  Experiment A output: 0=background, 1=nucleus                          │
│  Experiment B output: 0=background, 1=nucleus interior, 2=boundary     │
└────────────────────────────────────────────────────────────────────────┘
                              │
                 Stage 1 predictions (segmentation mask)
                              │
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2 — Region Classifier                      │
│                        (Nucleus Type Classification)                    │
├────────────────────────────────────────────────────────────────────────┤
│  One nucleus region (binary mask + image patch)                         │
│         │                                                               │
│    [NucleusFeatureExtractor]  → 40 features:                           │
│      • Morphology (10): area, perimeter, eccentricity, solidity, etc.  │
│      • Color stats (16): LAB + HED mean/std + H-channel percentiles    │
│      • LBP histogram (10): texture distribution within region          │
│      • Context ring (4): HED-H + LAB-L stats in 5px surrounding ring  │
│         │                                                               │
│    [SVM-RBF / XGBoost]  → class 0-4                                   │
│                                                                         │
│  Output: neoplastic / inflammatory / connective / dead / epithelial    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Fitted Artifact Dependency Chain

```
normalizer.joblib
  └── scaler_{binary|3class}.joblib
        └── selector_{binary|3class}.joblib  (feature selection)
              └── {rf|lgbm}_{binary|3class}_{base|tuned}.joblib
                    └── evaluation results on Fold3 (one-time, held-out)

Stage 2 (independent):
scaler_stage2.joblib
  └── {svm|xgb}_stage2_{base|tuned}.joblib
        └── stage2 evaluation (oracle + pipeline modes)
```

**Invariant**: Artifacts must be fit and saved in this order — never skip or reorder.

---

## Evaluation Modes

### Stage 1 Metrics
- **Primary HPO metric**: Macro-F1
- **Reporting**: Dice coefficient, IoU, per-tissue Dice (19 tissue types)

### Stage 2 Evaluation Modes
| Mode | Input to Stage 2 |
|------|-----------------|
| **Oracle** | Ground truth nucleus regions (ideal upper bound) |
| **Pipeline** | Stage 1 predicted regions (real-world performance) |
| **Gap** | Oracle macro-F1 − Pipeline macro-F1 (error propagation measure) |

---

## Experiments

| Experiment | Label Mode | Classes | Stage 1 Models |
|-----------|-----------|---------|---------------|
| **Exp A** | binary | 2 (background, nucleus) | RF base, RF tuned, LightGBM |
| **Exp B** | 3class | 3 (background, interior, boundary) | RF base, RF tuned, LightGBM |

---

## Data Flow: Training Dataset Build

```
data/raw/Fold{1,2}/
       │
  [Pass 1: count_total_samples]  ← labels only, ~2 min
       │ total_N
  [Allocate memmap: X=(N,93) y=(N,)]
       │
  [Pass 2: streaming_loader.build_dataset_memmap]
       │   batch of images → normalize → color convert
       │   → extract 87 dense features → sample pixels
       │   → compute GLCM for sampled pixels
       │   → concatenate 93 features → write to memmap
       │   → checkpoint every 100 images (atomic JSON)
       │
data/processed/{split}_{mode}_X.npy  ← float32 (N, 93)
data/processed/{split}_{mode}_y.npy  ← uint8 (N,)
```

**Peak RAM**: ~500 MB constant regardless of dataset size (streaming design).

---

## Inference Pipeline (Not Yet Implemented)

`src/inference/pipeline.py` is an **empty placeholder**. Intended flow:
```
new_image → [MacenkoNormalizer] → [ColorConverter] → [PixelFeatureExtractor]
         → [StandardScaler] → [FeatureSelector] → [RF/LightGBM] → segmentation mask
         → [instance extraction] → [NucleusFeatureExtractor] → [SVM/XGBoost] → class labels
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---------|--------|-----------|
| Streaming vs materialized | Streaming | 190 GB intermediate feature maps unfeasible on disk |
| GPU vs CPU features | Hybrid | GPU for windowed stats/LoG; CPU for LBP/Gabor/SLIC |
| Sampling strategy | Active Boundary Mining | 3× oversample near boundaries where errors matter most |
| GLCM scope | Per-sampled-pixel only | Computing for all 65k pixels per image is too slow |
| Train/test split | Fixed fold split | Fold3 touched exactly once — final evaluation only |
