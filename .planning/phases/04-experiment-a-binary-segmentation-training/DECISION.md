# Phase 4: Decisions

## Decision 1: Training Data & Validation Strategy

**Status:** CONFIRMED — locked in discuss-phase session.

| Aspect | Decision |
|---|---|
| Training data | Full Fold 1 — 2,073,571 rows, sliced to 25 RFE features (~200 MB GPU) |
| Validation data | Fold 2 held-out — ~1,859,513 rows (true cross-fold generalization) |
| Fold 2 scaling | In-memory batch transform via `FeatureScaler.load("data/models/scaler_fold1_binary.joblib")` — no new .npy written |

---

## Decision 2: Hyperparameter Optimization

**Status:** CONFIRMED — locked in discuss-phase session.

| Aspect | Decision |
|---|---|
| Strategy | Light Optuna sweep on 200K-row stratified subsample of Fold 1 |
| Study structure | One independent study per model: `rf_study`, `lgbm_study`, `xgb_study` |
| Trial budget | 20–30 trials, 30-minute wall-clock limit per study |
| Objective metric | Macro-F1 on subsample val split (consistent with Phase 3) |
| Post-sweep | Refit best hyperparams on **full Fold 1** before Fold 2 evaluation |
| Persistence | Best params saved as both JSON (`data/models/{rf,lgbm,xgb}_best_params.json`) and embedded in joblib |

---

## Decision 3: Ensemble Blending Weights

**Status:** CONFIRMED — locked in discuss-phase session.

| Aspect | Decision |
|---|---|
| Method | Grid search over weight simplex (step=0.1, ~165 deterministic combinations) |
| Objective | Maximize Macro-F1 on Fold 2 predictions |
| Persistence | Optimal weights stored inside `binary_ensemble.joblib` |

---

## Decision 4: Model Evaluation Table (Fold 2)

**Status:** PENDING — to be filled by `scripts/train_ensemble.py` + `notebooks/08_inspect_binary_ensemble.ipynb`

| Classifier | Fold 2 Macro-F1 | AUC-ROC | Optuna Best Trials | Training Time (s) |
|---|---|---|---|---|
| Random Forest (cuML) | PENDING | PENDING | PENDING | PENDING |
| LightGBM (GPU) | PENDING | PENDING | PENDING | PENDING |
| XGBoost (GPU) | PENDING | PENDING | PENDING | PENDING |
| **SoftVoting Ensemble** | **PENDING** | **PENDING** | N/A | PENDING |

**Optimal ensemble weights:** RF=PENDING, LGB=PENDING, XGB=PENDING  
**Recorded by:** `notebooks/08_inspect_binary_ensemble.ipynb`
