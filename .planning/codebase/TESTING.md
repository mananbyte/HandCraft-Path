---
document: TESTING.md
focus: quality
mapped: 2026-05-25
---

# Testing — HandCraft-Path (PanNuke Project)

## Testing Philosophy

This project is **research/ML code** — testing is primarily:
1. **Unit validation scripts** (not a formal test framework)
2. **Jupyter notebooks** for visual/manual inspection
3. **Assertion guards** embedded in pipeline functions

There is **no pytest/unittest framework** configured. Tests are standalone scripts in `scripts/`.

---

## Test Scripts

### `scripts/test_stain_normalizer.py` (49 lines)
- Validates `MacenkoNormalizer` on a small sample
- Checks fit/transform cycle, output shape, and dtype
- Run: `python scripts/test_stain_normalizer.py`

### `scripts/test_feature_pipeline.py` (64 lines)
- End-to-end feature extraction test on one image
- Validates `convert_image()` → `extract_dense_features_cpu()` → shapes
- Checks no NaN/Inf in output features
- Run: `python scripts/test_feature_pipeline.py`

### `scripts/diagnose_gpu_init.py` (235 lines)
- Comprehensive GPU diagnostic tool
- Checks CuPy import, CUDA device count, VRAM, driver version
- Tests `distance_transform_edt` on GPU
- Helps debug GPU initialization failures
- Run: `python scripts/diagnose_gpu_init.py`

### `scripts/demo_memory_config.py` (128 lines)
- Validates `MemoryConfig` auto-detection and batch size computation
- Demonstrates different memory configuration options
- Run: `python scripts/demo_memory_config.py`

---

## Inspection Notebooks

| Notebook | What It Validates |
|----------|-------------------|
| `notebooks/00_inspect_gpu_init.ipynb` | GPU/CuPy environment check |
| `notebooks/01_inspect_raw_data.ipynb` | Data shapes, value ranges, tissue type distribution |
| `notebooks/02_inspect_labels.ipynb` | Binary and 3-class label generation correctness |
| `notebooks/03_inspect_stain_processing.ipynb` | Macenko normalization visual validation |
| `notebooks/04_inspect_sampling.ipynb` | Active boundary mining strategy validation |
| `notebooks/05_inspect_features.ipynb` | Feature extraction correctness |
| `notebooks/06_evaluate_stage1.ipynb` | Stage 1 model metrics on held-out data |
| `notebooks/07_evaluate_stage2.ipynb` | Stage 2 model metrics (oracle + pipeline modes) |
| `notebooks/08_ablation_study.ipynb` | Feature ablation analysis |

---

## Embedded Assertion Guards

Pipeline functions include inline assertions as invariant checks:

### Label Generator (`src/data/label_generator.py`)
```python
# After generate_binary_labels():
assert nucleus_binary.shape == (256, 256)
assert nucleus_binary.dtype == np.uint8
assert set(np.unique(nucleus_binary)).issubset({0, 1})

# After generate_3class_labels():
assert labels.max() <= 2
assert labels.min() >= 0
assert (boundary & ~nucleus_binary).sum() == 0  # no boundary outside nucleus
```

### Feature Extractor (`src/features/pixel_feature_extractor.py`)
```python
# Post-assembly:
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
# Implicit: shape check via concatenation (fails loudly if counts wrong)
assert all_feats.shape == (256, 256, 87)  # DENSE_FEATURES
```

---

## What's Tested vs. What's Not

### ✅ Validated (by notebooks or test scripts)
- Stain normalization (Macenko) — correctness and stability
- Color conversion (LAB, HSV, HED) — shape and value range
- Label generation (binary + 3-class) — via assertions and visual inspection
- Active boundary mining — distribution checks in notebook 04
- Dense feature extraction (87 features) — shape, NaN checks
- GLCM feature extraction — visual inspection
- Memory config — demonstrated via demo script
- GPU init diagnostics

### ❌ Not Tested / Missing
- `src/features/nucleus_feature_extractor.py` — **file is empty**
- `src/features/feature_selector.py` — **file is empty**
- `src/training/cuml_trainer.py` — **file is empty**
- `src/inference/pipeline.py` — **file is empty**
- `scripts/train_stage1.py` — **file is empty**
- `scripts/train_stage2.py` — **file is empty**
- `scripts/evaluate_stage1.py` — **file is empty**
- `scripts/evaluate_stage2.py` — **file is empty**
- Integration test for full pipeline (normalize → train → evaluate)
- No regression tests for model performance
- No CI/CD pipeline

---

## Running Manual Validation

```bash
# Activate environment
conda activate HandCraft-Path

# Validate stain normalizer
python scripts/test_stain_normalizer.py

# Validate feature pipeline
python scripts/test_feature_pipeline.py

# Check GPU setup
python scripts/diagnose_gpu_init.py

# Check memory config
python scripts/demo_memory_config.py

# Build training data (runs streaming pipeline — biggest integration test)
python scripts/build_dataset.py --label-mode binary --folds 1,2
```

---

## Coverage Summary

| Component | Coverage Level |
|-----------|---------------|
| Preprocessing (stain norm, color convert) | Medium — scripts + notebooks |
| Feature extraction (87 dense + GLCM) | Medium — notebooks + assertions |
| Label generation | High — assertions + visual notebooks |
| Pixel sampling | Medium — notebook inspection |
| Memory config | Medium — demo script |
| Dataset builder (streaming) | Low — no automated test; run manually |
| Model training (Stage 1) | **None** — scripts empty |
| Model training (Stage 2) | **None** — scripts empty |
| Evaluation | **None** — scripts empty |
| Inference pipeline | **None** — file empty |
