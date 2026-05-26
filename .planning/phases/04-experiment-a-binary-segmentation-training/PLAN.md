# Phase 4: Experiment A (Binary) Segmentation Ensemble

**Status:** Context captured — ready for planning.

## Decisions (Locked from Context Discussion)

| Decision | Value | Rule |
|---|---|---|
| Training Data | Full Fold 1 (2.07M rows, 25 RFE features) | Train on everything — after 25-feature slice it's GPU-safe |
| Validation Set | Fold 2 (held-out, ~1.86M rows) | True cross-fold generalization — NOT used for any fitting |
| Fold 2 Scaling | In-memory batch transform via locked scaler_fold1_binary.joblib | No new .npy written to disk |
| HPO Strategy | Light Optuna sweep on 200K subsample, 20-30 trials, 30-min budget | Tune then refit on full Fold 1 |
| Optuna Studies | One per model: rf_study, lgbm_study, xgb_study | Independent tuning |
| Optuna Metric | Macro-F1 on subsample val split | Consistent with Phase 3 |
| Best Params | Both JSON (`data/models/rf_best_params.json`) and embedded in joblib | Human audit + self-contained loading |
| Ensemble Weights | Grid search on weight simplex (step=0.1, ~165 combos) | Data-driven, deterministic, auditable |
| Notebook | Full research document with Markdown prose + all 5 visualization types | Per discussion decisions D-11, D-12 |

## Infrastructure Already Built (Phase 3 Outputs)

| File | Purpose |
|---|---|
| `data/processed/fold1_binary_X_scaled.npy` | Scaled Fold 1 features (2.07M × 93) — slice to 25 RFE cols |
| `data/processed/fold1_binary_y.npy` | Fold 1 binary labels |
| `data/processed/fold2_binary_X.npy` | Raw Fold 2 features — scale in-memory before eval |
| `data/processed/fold2_binary_y.npy` | Fold 2 binary labels |
| `data/models/scaler_fold1_binary.joblib` | Fitted HandCraftPathScaler |
| `data/models/selected_features_rfe_25.json` | 25 RFE feature indices and names |
| `src/training/scaler.py` | FeatureScaler.load() for Fold 2 in-memory scaling |
| `src/utils/safe_loader.py` | safe_load_npy for memory-safe array loading |

## Plans (Sequential — Execute in Wave Order)

### Wave 1
- **[04-01-PLAN.md](./04-01-PLAN.md)** — Implement ensemble module, train script with Optuna HPO, inspection notebook, and unit tests.

## Final Artefacts (Phase 4 Complete When All Exist)

```
src/training/ensemble.py
scripts/train_ensemble.py
notebooks/08_inspect_binary_ensemble.ipynb
tests/test_ensemble.py
data/models/cuml_rf.joblib
data/models/lgbm.joblib
data/models/xgboost.joblib
data/models/binary_ensemble.joblib
data/models/rf_best_params.json
data/models/lgbm_best_params.json
data/models/xgb_best_params.json
.planning/phases/04-experiment-a-binary-segmentation-training/DECISION.md (filled)
```

## Success Criteria (from ROADMAP)

1. Random Forest (cuML), LightGBM (GPU), and XGBoost (GPU) trained on full Fold 1 with Optuna-tuned hyperparameters.
2. `SoftVotingEnsemble` wrapper with Fold-2-optimized blending weights serialized to `data/models/binary_ensemble.joblib`.
3. Notebook `08_inspect_binary_ensemble.ipynb` executes end-to-end with all 5 required visualization types + Markdown prose.
4. `pytest tests/test_ensemble.py` passes (CPU-only, <30s).
5. DECISION.md filled with Fold 2 Macro-F1 scores for all 3 base models and the ensemble.

---
*Plan updated: 2026-05-26 | Phase: 4 of 10 | Depends on: Phase 3 scaled matrices*
