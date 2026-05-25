# HandCraft-Path Architecture

## Research questions
1. How far do hand-crafted features go vs CNN architectures on H&E nucleus segmentation?
2. Does explicit boundary class (Exp B) improve over binary segmentation (Exp A)?
3. Does better segmentation improve downstream nucleus type classification?
4. Oracle gap: how much does Stage 1 error hurt Stage 2 accuracy?

## Two-stage pipeline

### Stage 1 — Pixel classifier (segmentation)
Input:  256×256 RGB H&E image
Output: 256×256 semantic mask

Experiment A: 0=background, 1=nucleus
Experiment B: 0=background, 1=nucleus interior, 2=nucleus boundary

Pipeline:
  image → Macenko normalize → color convert (LAB, HSV, HED)
        → extract 87 dense features (full image)
        → active boundary mining (sample ~1200 pixels per image)
        → compute GLCM for sampled pixels (6 features)
        → concatenate → (N_sampled, 93) training rows

Models: Random Forest (primary), LightGBM (challenger)
Metric: Macro-F1 (HPO), Dice + IoU + per-tissue Dice (reporting)

### Stage 2 — Region classifier (nucleus typing)
Input:  one nucleus region (binary mask + image)
Output: class label 0-4 (neoplastic, inflammatory, connective, dead, epithelial)

Features (40 total):
  morphology (10): area, perimeter, eccentricity, solidity, circularity...
  color stats (16): LAB + HED mean/std + H-channel percentiles
  LBP histogram (10): texture distribution within region
  context ring (4): HED-H + LAB-L stats in 5px surrounding ring

Evaluation modes:
  Oracle:   GT nucleus regions → Stage 2 classifier
  Pipeline: Stage 1 predicted regions → Stage 2 classifier
  Gap:      Oracle macro-F1 − Pipeline macro-F1

Models: SVM-RBF (primary), XGBoost (challenger)
Metric: Macro-F1 + per-class F1 + confusion matrix + oracle gap

## Fitted artifact dependency chain (Phase 3 Completed)
normalizer.joblib (Macenko stain normalization vector)
  └── data/models/scaler_fold1_binary.joblib (Fitted HandCraftPathScaler object)
        ├── data/processed/fold1_binary_X_scaled.npy (Scaled binary matrix, 2.07M rows × 93)
        ├── data/processed/fold1_3class_X_scaled.npy (Scaled 3-class matrix, 3.07M rows × 93)
        └── data/models/selected_features_rfe_25.json / .csv (Selected 25 features via GPU RFE)
              └── [Phase 4 Planned] {rf|lgbm}_{binary|3class}_{base|tuned}.joblib
                    └── evaluation results (Fold3, one-time)

Stage 2 (independent):
scaler_stage2.joblib
  └── {svm|xgb}_stage2_{base|tuned}.joblib
        └── stage2 evaluation results (oracle + pipeline)