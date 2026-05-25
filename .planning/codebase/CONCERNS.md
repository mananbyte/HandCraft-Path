---
document: CONCERNS.md
focus: concerns
mapped: 2026-05-25
---

# Technical Concerns — HandCraft-Path (PanNuke Project)

## 🔴 Critical: Incomplete Implementation

### Empty Core Files
The following files are **entirely empty** (1 line, no code), meaning critical
pipeline stages cannot be executed:

| File | Missing Functionality | Impact |
|------|-----------------------|--------|
| `src/features/nucleus_feature_extractor.py` | Stage 2 region feature extraction | Stage 2 cannot run at all |
| `src/features/feature_selector.py` | MRMR feature selection | Cannot select features for training |
| `src/training/cuml_trainer.py` | cuML GPU training wrapper | Training pipeline blocked |
| `src/inference/pipeline.py` | End-to-end inference | No inference capability |
| `scripts/train_stage1.py` | Stage 1 model training | Cannot train Stage 1 from CLI |
| `scripts/train_stage2.py` | Stage 2 model training | Cannot train Stage 2 from CLI |
| `scripts/tune_stage1.py` | Hyperparameter optimization | Cannot run HPO from CLI |
| `scripts/evaluate_stage1.py` | Stage 1 evaluation on Fold3 | Cannot evaluate from CLI |
| `scripts/evaluate_stage2.py` | Stage 2 evaluation (oracle + pipeline) | Cannot evaluate from CLI |
| `scripts/build_stage2_dataset.py` | Build Stage 2 training data | Stage 2 dataset missing |
| `scripts/fit_normalizer.py` | Fit normalizer standalone | Minor — covered by build_dataset.py |
| `scripts/train_streaming.py` | Streaming training | Minor — covered by other scripts |

> **Note**: Some trained model `.joblib` files exist in `data/models/` (RF, LightGBM, SVM,
> optuna studies), suggesting training was done at some point — but the training scripts
> themselves were either deleted, moved, or never created as standalone files.

---

## 🟠 High: Processed Data Incomplete

### Only Binary Labels Processed
Currently only Fold1+Fold2 binary data exists in `data/processed/`:
```
fold1_binary_X.npy  ← Fold 1 features (binary labels)
fold1_binary_y.npy
fold2_binary_X.npy  ← Fold 2 features (binary labels)
fold2_binary_y.npy
```

**Missing**:
- `train_binary_X.npy` / `train_binary_y.npy` — combined Fold1+2 training set
- `train_3class_X.npy` / `train_3class_y.npy` — 3-class experiment data
- `test_binary_X.npy` / `test_binary_y.npy` — Fold3 test set features
- `test_3class_X.npy` / `test_3class_y.npy` — Fold3 3-class test features

This means Experiment B (3-class with boundary) hasn't been run, and Fold3 evaluation
(the primary research output) hasn't been performed.

---

## 🟠 High: Stage 2 Data & Training Gap

`data/models/rf_stage2.joblib` and `svm_stage2.joblib` exist, but:
- `scripts/build_stage2_dataset.py` is **empty** — how was Stage 2 data built?
- `src/features/nucleus_feature_extractor.py` is **empty** — how were Stage 2 features extracted?
- This suggests Stage 2 was done via notebook or outside the recorded scripts

**Risk**: Stage 2 training is not reproducible from scripts alone.

---

## 🟡 Medium: Stack.md vs environment.yml Mismatch

`Stack.md` documents outdated stack (committed earlier):
```
Stack.md says:       environment.yml says:
Python 3.10          Python 3.11
RAPIDS 24.10         RAPIDS 25.12
scikit-learn 1.3.2   scikit-learn ≥1.5
lightgbm 4.1.0       lightgbm ≥4.3
```

`Stack.md` should be updated to reflect the current environment.

---

## 🟡 Medium: GPU Runtime Initialization Fragility

Evidence of recurring GPU init issues:
- `scripts/diagnose_gpu_init.py` (235 lines) — dedicated GPU diagnostic tool
- `docs/memory-issues.txt`, `docs/ram-issues.txt` — extensive RAM/GPU troubleshooting docs
- `PANNUKE_FORCE_CPU_EDT` env variable — escape hatch for GPU EDT failures
- `src/sampling/pixel_sampler.py` has complex GPU init state management

The GPU init check at module import time (`cp.cuda.runtime.getDeviceCount()`)
can silently report `_HAS_GPU = False` even when GPU is present, falling back
to CPU without warning in `pixel_feature_extractor.py`.

---

## 🟡 Medium: No Formal Test Suite

- No `pytest`, no `unittest`, no CI/CD
- Test scripts are standalone (`test_stain_normalizer.py`, `test_feature_pipeline.py`)
- No automated regression tests for model performance
- If a refactor breaks the 87-feature count or pipeline ordering, no test will catch it

---

## 🟡 Medium: GLCM Is Computationally Expensive

In `src/features/pixel_feature_extractor.py`, GLCM is computed per sampled pixel with a Python loop:
```python
for i in range(n):  # ~1200 iterations per image
    result[i] = compute_glcm_for_patch(patch)  # pure Python + skimage
```
This is explicitly not GPU-accelerated and is the slowest step per image.
No batching or parallelization.

---

## 🟡 Medium: Superpixel Features Are Slow

```python
segments = slic(img_uint8, n_segments=200, compactness=10, sigma=1, start_label=0)
for seg_id in np.unique(segments):  # Python loop over ~200 segments
    mask = (segments == seg_id)
    ...
```
SLIC cannot use GPU (CuPy). Per-segment Python loop is O(n_segments) per image.

---

## 🟢 Low: Color Space Stain Matrix Is Hardcoded

```python
# src/preprocessing/color_converter.py
HE_MATRIX = np.array([
    [0.644211, 0.716556, 0.266844],
    [0.092789, 0.954111, 0.283111],
    [0.63718,  0.00000,  0.000103],
])
```
This is a fixed H&E stain matrix — not learned from data. Appropriate for
this dataset but not adaptable to other staining protocols without code change.

---

## 🟢 Low: `app/` Directory Is Empty

`app/` exists but contains nothing — presumably a placeholder for the FastAPI app.
`src/inference/pipeline.py` is also empty. The inference server is entirely absent.

---

## 🟢 Low: Notebook Checkpoints Not Gitignored

`notebooks/.ipynb_checkpoints/` contains checkpoint files that should be in `.gitignore`.
They appear to be present in the working directory.

---

## Summary Table

| Severity | Issue | Count |
|---------|-------|-------|
| 🔴 Critical | Empty core implementation files | 12 files |
| 🟠 High | Incomplete processed data / Stage 2 reproducibility | 2 issues |
| 🟡 Medium | Stack.md mismatch, GPU fragility, no test suite, GLCM/SLIC speed | 4 issues |
| 🟢 Low | Hardcoded stain matrix, empty app dir, notebook checkpoints | 3 issues |
