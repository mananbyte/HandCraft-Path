# Phase 4: Experiment A (Binary) Segmentation Ensemble - Context

**Gathered:** 2026-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 4 trains, tunes, and ensembles three GPU-accelerated pixel classifiers (cuML RandomForest, LightGBM GPU, XGBoost GPU) on the scaled binary feature matrix from Phase 3. It uses Fold 1 (2.07M rows × 25 RFE features) for training with per-model Optuna HPO sweeps, evaluates against Fold 2 as a held-out validation set (true cross-fold generalization), optimizes ensemble blending weights via grid search on the weight simplex, and serializes all model artifacts to `data/models/`. It concludes with a research-quality Jupyter notebook documenting performance metrics, feature importances, and biological interpretation.

</domain>

<decisions>
## Implementation Decisions

### Training Data Strategy
- **D-01:** Train on the **full Fold 1 matrix** (2,073,571 rows). After slicing to the 25 RFE features, this is ~200 MB GPU-resident — well within VRAM limits. No subsampling for final model training.
- **D-02:** Use **Fold 2 as the held-out validation set** (`data/processed/fold2_binary_X.npy` + `fold2_binary_y.npy`, ~1.86M rows) for true cross-fold generalization evaluation. Fold 2 is NOT used for any fitting.
- **D-03:** Fold 2 must be scaled using the **locked `scaler_fold1_binary.joblib`** from Phase 3 — applied in-memory per batch via `FeatureScaler.transform()` streaming. No new scaled `.npy` is written to disk for Fold 2.

### Hyperparameter Tuning (Optuna)
- **D-04:** Run a **light Optuna sweep** on a **200K-row stratified subsample of Fold 1** (same budget as Phase 3 scaler fitting — guaranteed memory-safe). Trial budget: 20–30 trials, ~30-minute wall-clock limit per model.
- **D-05:** Each model gets its **own independent Optuna study** — `rf_study`, `lgbm_study`, `xgb_study` — so hyperparameters are tuned independently without cross-contamination.
- **D-06:** Optuna objective metric: **Macro-F1 on the 200K subsample validation split** (consistent with Phase 3 proxy metric — rewards balance across both background and nucleus classes).
- **D-07:** After HPO, **refit the winning hyperparameters on the full Fold 1** before evaluating on Fold 2. The subsample is for search only.
- **D-08:** Best hyperparameters are persisted **both as JSON** (e.g., `data/models/rf_best_params.json`) for human auditing and **embedded inside the joblib model file** for self-contained loading.

### Ensemble Weight Policy
- **D-09:** Ensemble blending weights are **optimized on Fold 2 validation predictions** (not training data). Use a **grid search over the weight simplex with step=0.1** (~165 deterministic combinations for 3 models). Objective: maximize Macro-F1 on Fold 2.
- **D-10:** The optimized weights (e.g., RF=w₁, LGB=w₂, XGB=w₃, summing to 1.0) are serialized inside `binary_ensemble.joblib` and reported in DECISION.md.

### Notebook Scope (08_inspect_binary_ensemble.ipynb)
- **D-11:** The notebook is structured as a **research document with proper human-readable Markdown prose** — each section explains what is being shown and its biological/clinical significance (not just code + plots).
- **D-12:** Required visualizations beyond ROC/PR curves:
  1. **Feature importance comparison**: bar chart overlaying RF impurity, LGB gain, and XGB gain for all 25 RFE features side-by-side.
  2. **Per-tissue Macro-F1 breakdown**: 19 PanNuke tissue types — shows where the ensemble generalizes vs. struggles.
  3. **Probability calibration curves**: reliability diagrams — shows if model probabilities are trustworthy for soft voting.
  4. **Confusion matrix heatmap + full classification report** for each individual model and the ensemble.
  5. **Weight optimization surface**: heatmap of Macro-F1 over the simplex grid — shows why the optimal weights win.

### the agent's Discretion
- CLI output formatting, tqdm progress bar telemetry style, and temporary subsample handling are left to the agent's discretion, provided they follow established patterns from `standardize_and_select.py`.
- The specific Optuna sampler (TPE is the default and recommended) is left to the agent.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 3 Outputs (direct dependencies)
- `data/processed/fold1_binary_X_scaled.npy` — Scaled binary training features (2.07M rows × 93 cols, float32). Slice to 25 RFE columns before training.
- `data/processed/fold1_binary_y.npy` — Binary labels for Fold 1 training.
- `data/processed/fold2_binary_X.npy` — Raw Fold 2 features (must be scaled in-memory before eval).
- `data/processed/fold2_binary_y.npy` — Binary labels for Fold 2 evaluation.
- `data/models/scaler_fold1_binary.joblib` — Fitted `FeatureScaler` (HandCraftPathScaler). Apply to Fold 2 before prediction.
- `data/models/selected_features_rfe_25.json` — JSON registry of the 25 RFE feature indices and names.

### Infrastructure (reuse patterns from these)
- `src/training/scaler.py` — `FeatureScaler` class with `transform_memmap`, `save`, `load`. Reuse `load()` for Fold 2 scaling.
- `src/utils/safe_loader.py` — `safe_load_npy(path, mode="r")` for memory-safe numpy loading.
- `scripts/standardize_and_select.py` — CLI script pattern, tqdm telemetry, progress logging, memory reporting. Follow this style.
- `src/training/feature_selection.py` — `get_feature_names()` for mapping feature indices to names.

### Planning Documents
- `.planning/phases/03-dataset-standardization-mrmr-selection/DECISION.md` — Locked scaler group decisions, feature count n=25, method=rfe.
- `.planning/phases/03-dataset-standardization-mrmr-selection/SUMMARY.md` — Phase 3 lessons learned (cuML API parity, virtual memory vs. leak, quantile range for skewed backgrounds).
- `.planning/ROADMAP.md` — Phase 4 success criteria and dependencies.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FeatureScaler.load(path)` in `src/training/scaler.py`: Load the fitted Phase 3 scaler to transform Fold 2 in-memory without any refitting.
- `safe_load_npy(path, mode="r")` in `src/utils/safe_loader.py`: Memory-safe NumPy 2.x loader for reading large .npy files as memmaps.
- `get_feature_names()` in `src/training/feature_selection.py`: Returns the 93-element name list — slice with RFE indices from JSON to get 25-feature names.
- `_rss_mb()` in `src/training/scaler.py`: Private RSS reporter — copy this pattern for training memory telemetry.

### Established Patterns
- **No full matrix in RAM:** Use `mode="r"` mmap + index slicing for large arrays. For training, slice to 25 columns first — result fits in GPU VRAM.
- **random_state=42:** Every stochastic operation uses this seed. cuML models accept `random_state` in `RandomForestClassifier` (unlike `LogisticRegression` — see Phase 3 lesson).
- **Joblib with compress=3:** All model artifacts use `joblib.dump(..., compress=3)` per scaler.py pattern.
- **Logger + tqdm:** Use `logging.getLogger(__name__)` and tqdm progress bars with RSS/throughput telemetry.
- **Assert gates:** Shape assertions at every function boundary (e.g., `assert X.ndim == 2`, `assert X.shape[1] == 25`).

### Integration Points
- **Input:** `data/processed/fold1_binary_X_scaled.npy[:, rfe_indices]` (25 cols), `fold1_binary_y.npy`
- **Fold 2 eval:** `fold2_binary_X.npy` → in-memory batch transform via `FeatureScaler.load(...)._scaler.transform(chunk)` → `[:, rfe_indices]`
- **Output:** `data/models/{cuml_rf,lgbm,xgboost}.joblib`, `data/models/binary_ensemble.joblib`, `data/models/{rf,lgbm,xgb}_best_params.json`
- **Notebook input:** All joblib files + Fold 2 predictions — loaded at notebook startup

</code_context>

<specifics>
## Specific Ideas

- The weight optimization simplex grid search should output a **heatmap** in the notebook (2D slice with RF weight on x-axis, LGB weight on y-axis, XGB weight = 1 - RF - LGB) showing the F1 landscape — this directly justifies the chosen weights.
- Per-tissue F1 breakdown requires matching Fold 2 pixel rows back to their source tissue type. This metadata may need to be stored during the Fold 2 prediction step — the agent should check if tissue labels are available in the raw PanNuke data or need to be reconstructed.

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope.

</deferred>

---

*Phase: 4-Experiment A Binary Segmentation Ensemble*
*Context gathered: 2026-05-26*
