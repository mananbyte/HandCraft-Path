---
document: CONVENTIONS.md
focus: quality
mapped: 2026-05-25
---

# Coding Conventions — HandCraft-Path (PanNuke Project)

## Language & Style

- **Python only** — no pipeline code in notebooks (notebooks = inspection only)
- **No magic numbers** — every threshold, radius, or count gets a named constant at module top
- **Files under 300 LOC** — split into focused modules when growing beyond this
- **Type hints in docstrings** — not inline annotations (but docstrings describe shapes and types)
- **Every script logs**: Python version, library versions, start time, args used

## Docstring Convention

Module-level docstrings document feature groups with explicit counts:
```python
"""
Pixel-level feature extraction for Stage 1 segmentation.

Features: 87 dense (per-pixel) + 6 GLCM (per-sampled-pixel) = 93 total.

Feature groups (from feature_research.md):
  1. Optical Density (OD)           : 3   — Beer-Lambert DNA content
  ...
     Dense subtotal                 : 87
 11. GLCM (per sampled pixel only)  : 6
     Total per sampled pixel        : 93
"""
```

Function docstrings include Parameters, Returns, and shape information:
```python
def extract_dense_features_cpu(spaces):
    """
    Extract 87 dense features for one image (CPU path).

    Parameters
    ----------
    spaces : dict from convert_image()

    Returns
    -------
    features : np.ndarray, (65536, 87), float32
    """
```

## Constants Pattern

Feature counts are defined as module-level named constants:
```python
# src/features/pixel_feature_extractor.py
FEATURE_GROUPS = {
    'od':         3,
    'color':     54,
    'lbp':        3,
    'gabor':     12,
    'gradient':   5,
    'structure':  3,
    'dog':        3,
    'superpixel': 2,
    'entropy':    1,
    'edge_dist':  1,
}
DENSE_FEATURES = sum(FEATURE_GROUPS.values())  # 87
GLCM_FEATURES = 6
TOTAL_FEATURES = DENSE_FEATURES + GLCM_FEATURES  # 93
```

## Input Validation Pattern

Every function that touches data includes shape assertions:
```python
def generate_binary_labels(mask):
    nucleus_binary = (mask[:, :, :5].sum(axis=2) > 0).astype(np.uint8)
    assert nucleus_binary.shape == (256, 256)
    assert nucleus_binary.dtype == np.uint8
    assert set(np.unique(nucleus_binary)).issubset({0, 1})
    return nucleus_binary
```

## GPU/CPU Fallback Pattern

All GPU-accelerated paths use a consistent try/import/fallback pattern:
```python
try:
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter as gpu_uniform_filter
    _ = cp.cuda.runtime.getDeviceCount()
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False
```

Function naming convention for GPU/CPU variants:
- `compute_X_batch_gpu(...)` — GPU batched version
- `compute_X_single_cpu(...)` / `compute_X_cpu(...)` — CPU fallback

## NumPy Array Rules

- **No in-place mutation** without explicit comment explaining why
- Always use `np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)` after feature assembly
- Arrays are `float32` throughout — explicit cast at return: `.astype(np.float32)`
- Labels are `uint8`
- Flat pixel indices are `int32`

## Data Access Rules

- **Never** `np.load(path)` without `mmap_mode='r'` for `.npy` files (prevents RAM blowup)
- **Never** write to `data/raw/` — it is permanently read-only
- **Never** fit normalizer, scaler, or selector on Fold3 data

## Reproducibility

```python
# Every script that uses random operations:
np.random.seed(42)
# Every sklearn/cuML model:
random_state=42
```

## Checkpoint/Flush Order

Critical ordering invariant (documented in `docs/ram-issues.txt`):
```python
X_mm.flush()   # FIRST — flush data to disk
y_mm.flush()
save_checkpoint(...)  # SECOND — checkpoint after flush (safe against crash)
```
Atomic checkpoint write:
```python
tmp_path = path + ".tmp"
with open(tmp_path, 'w') as f:
    json.dump(data, f)
os.replace(tmp_path, path)  # atomic on Linux
```

## Pipeline Order (Never Violate)

```
normalize → color convert → extract dense features → sample → GLCM → concatenate
```

```
fit scaler → fit selector  (never reverse)
```

```
Stage 1 and Stage 2 are independent — never mix features or labels
```

## Artifact Naming Convention

```
scaler_binary.joblib         ← experiment mode in filename
rf_3class_tuned.joblib       ← model + mode + tuning status
optuna_study_binary.joblib   ← HPO study with mode
```

Rule: **never overwrite a tuned model with an untuned one**.

## Error Handling Pattern

GPU operations use nested try/except with fallback and memory cleanup:
```python
try:
    result = gpu_operation(data)
except Exception as exc:
    _GPU_DISABLED_RUNTIME = True
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    result = cpu_fallback(data)
```

## Memory Management Pattern (Streaming Loops)

```python
# After each batch:
del dense_features, spaces, labels, indices, sampled_labels, gray, glcm_feats, pixel_rows
gc.collect()
if _HAS_MALLOC_TRIM:
    _LIBC.malloc_trim(0)  # release glibc heap back to OS
```

## Feature Group Comments

Feature assembly sections use visual separators:
```python
# ═════════════════════════════════════════════════════════════════════════
# GROUP 2 — Color statistics at 3 scales (54 features)
# ═════════════════════════════════════════════════════════════════════════
```
