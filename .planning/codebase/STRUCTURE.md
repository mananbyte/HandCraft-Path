---
document: STRUCTURE.md
focus: arch
mapped: 2026-05-25
---

# Directory Structure — HandCraft-Path (PanNuke Project)

## Root Layout

```
Pannuke-project/
├── .planning/              ← GSD planning (codebase maps, project state)
│   └── codebase/           ← These 7 documents
├── src/                    ← Core library (importable Python package)
├── scripts/                ← Runnable pipeline scripts (CLI entry points)
├── notebooks/              ← Jupyter notebooks for inspection only
├── data/                   ← All data (read-only raw, mutable processed/models)
├── docs/                   ← Technical documentation and guides
├── app/                    ← (Empty/placeholder — FastAPI app dir?)
├── environment.yml         ← Conda environment spec
├── Architecture.md         ← High-level architecture overview
├── Stack.md                ← Technology stack reference
├── Rules.md                ← Project coding and data rules
├── README.md               ← Full project documentation (~500 lines)
├── IMPLEMENTATION_SUMMARY.md  ← Summary of memory config implementation
└── QUICK_REFERENCE.md      ← Quick reference card for common commands
```

---

## `src/` — Core Library

```
src/
├── __init__.py
├── preprocessing/
│   ├── __init__.py
│   ├── stain_normalizer.py     ← MacenkoNormalizer class (208 lines)
│   ├── color_converter.py      ← convert_image() → {rgb, lab, hsv, hed} (40 lines)
│   └── stain_separation.py     ← (separate stain utility)
├── data/
│   ├── __init__.py
│   ├── label_generator.py      ← generate_binary_labels(), generate_3class_labels(),
│   │                              extract_nucleus_instances() (134 lines)
│   ├── streaming_loader.py     ← build_dataset_memmap() — 2-pass memmap streaming (601 lines)
│   ├── batch_loader.py         ← (batch loading utilities)
│   └── debug_sample_manager.py ← Debug sample helpers
├── features/
│   ├── __init__.py
│   ├── pixel_feature_extractor.py   ← 87-feature dense extractor, GLCM, GPU/CPU paths (503 lines)
│   ├── nucleus_feature_extractor.py ← Stage 2 region feature extractor (EMPTY)
│   ├── batch_feature_extractor.py   ← Batch extraction helpers
│   └── feature_selector.py          ← MRMR feature selection (EMPTY)
├── sampling/
│   ├── __init__.py
│   └── pixel_sampler.py        ← active_boundary_mining() — GPU/CPU EDT (138 lines)
├── training/
│   ├── __init__.py
│   └── cuml_trainer.py         ← cuML GPU training wrapper (EMPTY)
├── inference/
│   ├── __init__.py
│   └── pipeline.py             ← End-to-end inference pipeline (EMPTY)
└── utils/
    ├── __init__.py
    └── memory_config.py        ← MemoryConfig class — dynamic batch sizing (436 lines)
```

---

## `scripts/` — CLI Entry Points

```
scripts/
├── build_dataset.py        ← Build Stage 1 training data (229 lines) ✅ COMPLETE
├── build_stage2_dataset.py ← Build Stage 2 training data (EMPTY)
├── build_training_data.py  ← (EMPTY — superseded by build_dataset.py?)
├── fit_normalizer.py       ← Fit stain normalizer standalone (EMPTY)
├── train_stage1.py         ← Train Stage 1 classifier (EMPTY)
├── train_stage2.py         ← Train Stage 2 classifier (EMPTY)
├── train_streaming.py      ← Train with streaming (EMPTY)
├── tune_stage1.py          ← HPO for Stage 1 (EMPTY)
├── evaluate_stage1.py      ← Stage 1 evaluation on Fold3 (EMPTY)
├── evaluate_stage2.py      ← Stage 2 evaluation — oracle + pipeline modes (EMPTY)
├── demo_memory_config.py   ← Memory config demonstration (128 lines) ✅
├── demo_streaming_pipeline.py ← Streaming demo (EMPTY)
├── diagnose_gpu_init.py    ← GPU diagnostics (235 lines) ✅
├── test_feature_pipeline.py ← Feature pipeline test (64 lines) ✅
└── test_stain_normalizer.py ← Stain normalizer test (49 lines) ✅
```

**Legend**: ✅ = has content | (EMPTY) = placeholder file only

---

## `data/` — Data Directory

```
data/
├── raw/                    ← READ-ONLY — never write here
│   ├── Fold1/
│   │   ├── images/fold1/images.npy   # (N, 256, 256, 3) float32
│   │   ├── images/fold1/types.npy    # tissue type strings
│   │   └── masks/fold1/masks.npy     # (N, 256, 256, 6) float32
│   ├── Fold2/              ← same structure
│   └── Fold3/              ← same structure — HELD OUT (touch only at final eval)
├── processed/              ← Generated training matrices
│   ├── fold1_binary_X.npy  # (N, 93) float32
│   ├── fold1_binary_y.npy  # (N,) uint8
│   ├── fold2_binary_X.npy
│   └── fold2_binary_y.npy
└── models/                 ← Fitted artifacts
    ├── normalizer.joblib
    ├── normalizer_test.joblib
    ├── rf_stage1_binary.joblib
    ├── rf_stage1_binary_tuned.joblib
    ├── rf_stage1_3class.joblib
    ├── rf_stage1_3class_tuned.joblib
    ├── lgbm_stage1_binary.joblib
    ├── lgbm_stage1_3class.joblib
    ├── selector_stage1_binary.joblib
    ├── selector_stage1_3class.joblib
    ├── optuna_study_binary.joblib
    ├── optuna_study_3class.joblib
    ├── rf_stage2.joblib
    └── svm_stage2.joblib
```

---

## `notebooks/` — Inspection Notebooks

```
notebooks/
├── 00_inspect_gpu_init.ipynb        ← GPU detection and CuPy validation
├── 01_inspect_raw_data.ipynb        ← Raw data shape/content inspection
├── 02_inspect_labels.ipynb          ← Label generation validation
├── 03_inspect_stain_processing.ipynb ← Macenko normalization visual check
├── 04_inspect_sampling.ipynb        ← Pixel sampling strategy validation
├── 05_inspect_features.ipynb        ← Feature extraction inspection
├── 06_evaluate_stage1.ipynb         ← Stage 1 model evaluation
├── 07_evaluate_stage2.ipynb         ← Stage 2 model evaluation
└── 08_ablation_study.ipynb          ← Ablation study analysis
```

**Rule**: Notebooks are for inspection only — no pipeline code in notebooks.

---

## `docs/` — Technical Documentation

```
docs/
├── feature_research.md           ← Feature group rationale and analysis
├── handcraft_path_agent_workflow.md ← Agent workflow documentation
├── MEMORY_CONFIG_GUIDE.md        ← Dynamic memory allocation guide
├── memory-issues.txt             ← Memory analysis and streaming design rationale
├── pannuke_implementation_guide.md ← Full implementation guide
├── project_structure.md          ← Project structure reference
├── ram-issues.txt                ← RAM analysis and strategies
├── run_commits.sh                ← Commit helper script
└── training_guide.md             ← Training pipeline guide
```

---

## Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `_binary` suffix | Experiment A (2-class labels) |
| `_3class` suffix | Experiment B (3-class labels) |
| `_base` suffix | Un-tuned baseline model |
| `_tuned` suffix | HPO-optimized model |
| `_stage1` / `_stage2` | Which pipeline stage |
| `fold{N}` in filename | Which data fold |
| `X.npy` / `y.npy` | Features matrix / labels vector |

## Import Path Convention

All scripts add the project root to `sys.path`:
```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
```

Then import as:
```python
from src.preprocessing.stain_normalizer import MacenkoNormalizer
from src.data.streaming_loader import build_dataset_memmap
from src.features.pixel_feature_extractor import extract_dense_features_cpu
```
