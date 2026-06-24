# GSD Phase 4: Validation Audit (Nyquist Validation)

This document contains the official validation audit for **Phase 4: Experiment A (Binary) Segmentation Ensemble**, confirming the base models, custom soft-voting ensemble class, and persisted weights are fully verified.

---

## 🔍 Validation Audit Summary

- **Phase**: `04-experiment-a-binary-segmentation-training`
- **Locked Base Estimators**: `cuML RandomForestClassifier`, `LGBMClassifier`, `XGBClassifier`
- **Locked Blend Weights**: `RF=0.4, LGB=0.3, XGB=0.3`
- **Validation Verdict**: **PENDING**

---

## 🧪 Success Criteria & Verification Checklist

### 1. Custom SoftVotingEnsemble Integration (Wave 1)
- **Audit**: Verify `src/training/ensemble.py` implements soft voting and fits RF, LGBM, and XGBoost on GPU.
- **CPU Fallback Validation**: Verify that the class gracefully imports and switches to sklearn/CPU models if GPU libraries are missing.
- **Verification Command**:
  ```bash
  python -c "from src.training.ensemble import SoftVotingEnsemble; print(SoftVotingEnsemble)"
  ```

### 2. Base Classifier & Ensemble Training (Wave 1)
- **Audit**: Verify that `scripts/train_ensemble.py` executes under the memory boundary and saves serialized models.
- **Model Storage**: Verify that all 4 models are successfully written to `data/models/`:
  - `cuml_rf.joblib`
  - `lgbm.joblib`
  - `xgboost.joblib`
  - `binary_ensemble.joblib`
- **Verification Command**:
  ```bash
  ls -lh data/models/ | grep joblib
  ```

### 3. Visual Performance Evaluation & Diagnostics (Wave 1)
- **Audit**: Verify that the inspection notebook `08_inspect_binary_ensemble.ipynb` runs end-to-end.
- **Metrics**: Verify that ROC, PR, confusion matrices, and side-by-side feature importances are compiled.
- **Decision File**: Verify that [DECISION.md](./DECISION.md) is updated with final macro-F1 and per-class scores.

### 4. Unit Test Suite Execution
- **Audit**: Verify that the test suite executing on synthetic inputs passes completely.
- **Verification Command**:
  ```bash
  conda run -n HandCraft-Path pytest tests/test_ensemble.py -v
  ```

---
*Validation status: PENDING | Phase 4 GSD Initialization*
