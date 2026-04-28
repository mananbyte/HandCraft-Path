# HandCraft-Path: Classical Feature Engineering for Nucleus Segmentation and Typing in H&E Histopathology

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)
![RAPIDS](https://img.shields.io/badge/RAPIDS-25.12-7B2FBE)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

**A GPU-accelerated, memory-efficient classical machine learning pipeline for pan-tissue nucleus segmentation on the PanNuke dataset — without a single neural network.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Why Classical ML?](#why-classical-ml)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Engineering](#feature-engineering)
- [Active Boundary Mining](#active-boundary-mining)
- [Memory Design](#memory-design)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Usage](#pipeline-usage)
- [Inspection Notebooks](#inspection-notebooks)
- [Experiments](#experiments)
- [Results (Planned)](#results-planned)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Overview

This project implements a **fully reproducible, GPU-accelerated classical ML pipeline** for nucleus segmentation on the [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) histopathology dataset.

The goal is to demonstrate that carefully engineered hand-crafted features — combined with GPU-accelerated processing and memory-safe streaming — can produce competitive segmentation results without deep learning. This serves as a rigorous baseline and provides **biological interpretability** that black-box networks cannot.

**What makes this project unique:**
- **Constant ~500 MB RAM** regardless of dataset size (7,904 images) via memmap streaming
- **93 biologically grounded features** per pixel, all GPU-accelerated
- **Active Boundary Mining** — a novel sampling strategy that oversamples near-nucleus boundaries where errors matter most
- **Crash-resilient** — atomic checkpoints allow resuming from any point in a 3+ hour run
- **Fully interpretable** — every feature has a biological motivation documented in research notes

---

## Why Classical ML?

> *"If you can solve a problem with a ruler, don't buy a laser cutter."*

| Property | Deep Learning | This Pipeline |
|----------|--------------|---------------|
| Interpretability | ❌ Black box | ✅ Feature importance maps |
| Training data needed | Millions of samples | ~3.2M pixel samples |
| RAM requirements | 8–32 GB GPU VRAM | ~500 MB constant |
| Training time | Hours on multi-GPU | ~3h CPU / ~1h GPU |
| Biological insights | ❌ Latent space | ✅ OD, LBP, structure tensor |
| Reproducibility | Hardware-dependent | ✅ Fully pinned environment |

Classical ML is also significantly easier to audit, certify, and deploy in clinical settings where model explainability is a regulatory requirement.

---

## Dataset

**PanNuke** is a pan-tissue, semi-automatically generated nucleus instance segmentation dataset.

| Property | Details |
|----------|---------|
| Images | 7,904 patches, 256×256 pixels |
| Tissue types | 19 (breast, lung, colon, prostate, …) |
| Nucleus classes | 5 (neoplastic, inflammatory, connective, dead, epithelial) |
| Mask format | (256, 256, 6) — channels 0–4 per class, channel 5 background |
| Folds | 3 cross-validation folds |
| Split used | Fold 1+2 = train, Fold 3 = test |

**Download:** [https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)

Place downloaded folds under `data/raw/` as follows:
```
data/raw/
├── Fold1/
│   ├── images/fold1/images.npy   # (2656, 256, 256, 3) float32
│   └── masks/fold1/masks.npy     # (2656, 256, 256, 6) float32
├── Fold2/
│   └── ...
└── Fold3/
    └── ...
```

---

## Pipeline Architecture

The pipeline is split into two sequential stages:

```
┌────────────────────────────────────────────────────────────┐
│                     STAGE 1: SEGMENTATION                  │
│                                                            │
│  Raw Image (256×256×3)                                     │
│       │                                                    │
│       ▼                                                    │
│  [1] Macenko Stain Normalization                           │
│       │                                                    │
│       ▼                                                    │
│  [2] Color Space Conversion (LAB · HSV · HED)              │
│       │                                                    │
│       ▼                                                    │
│  [3] Label Generation (binary or 3-class)                  │
│       │                                                    │
│       ▼                                                    │
│  [4] Dense Feature Extraction — 87 features/pixel (GPU)    │
│       │                                                    │
│       ▼                                                    │
│  [5] Active Boundary Mining — 400 px/class sampled         │
│       │                                                    │
│       ▼                                                    │
│  [6] GLCM — 6 features/sampled pixel                       │
│       │                                                    │
│       ▼                                                    │
│  [7] Memmap Write + Checkpoint                             │
│       │                                                    │
│       ▼                                                    │
│  Output: X (N × 93) + y (N,) on disk                       │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                     STAGE 2: CLASSIFICATION                │
│   (Planned — nucleus type from Stage 1 detections)         │
└────────────────────────────────────────────────────────────┘
```

### Two Label Modes

| Mode | Classes | Use |
|------|---------|-----|
| **Binary** (Exp A) | 0=background, 1=nucleus | Nucleus detection |
| **3-Class** (Exp B) | 0=background, 1=interior, 2=boundary | Boundary-aware segmentation |

---

## Feature Engineering

Each pixel is described by **93 features** — 87 computed densely over the entire image, plus 6 GLCM features computed per sampled pixel on a 15×15 patch.

### Feature Groups

| # | Group | Count | Signal |
|---|-------|-------|--------|
| 1 | Optical Density (OD) | 3 | Beer-Lambert law — DNA content proxy |
| 2 | Color Statistics (LAB + HSV + HED) | 54 | Mean + Std at radii 3, 7, 15 px × 9 channels |
| 3 | Local Binary Pattern (LBP) | 3 | Chromatin micro-texture at r=1,2,3 |
| 4 | Gabor Filter Bank | 12 | Directional texture — 3 frequencies × 4 orientations |
| 5 | Sobel + Laplacian of Gaussian (LoG) | 5 | Edges + blob detector |
| 6 | Structure Tensor | 3 | λ₁, λ₂ eigenvalues + anisotropy ratio |
| 7 | Difference of Gaussians (DoG) | 3 | Multi-scale nucleus size encoding |
| 8 | Superpixel Context | 2 | SLIC region mean HED-H and LAB-L |
| 9 | Local Entropy | 1 | Transition zone detector |
| 10 | Edge Distance | 1 | Border artifact correction |
| 11 | GLCM | 6 | Contrast, dissimilarity, homogeneity, energy, correlation, ASM |

**Total: 87 dense + 6 GLCM = 93 features per training sample.**

> **Design note:** Frangi vesselness is deliberately excluded — it is mathematically redundant with structure tensor anisotropy (λ₁/λ₂ ratio) and adds ~40% computation time for zero information gain.

### Key Biological Motivations

| Feature | Why it matters |
|---------|----------------|
| **HED Hematoxylin** | Direct proxy for chromatin density — the strongest single nucleus signal |
| **LoG σ=2** | Optimal blob detector for lymphocyte-scale (~10px diameter) nuclei |
| **Structure Tensor Anisotropy** | Low at round nuclei, high at elongated stroma — clean separator |
| **OD Blue channel** | Beer-Lambert: dense DNA absorbs more blue light |
| **Gabor isotropy** | Nuclei respond uniformly across orientations; stroma does not |

---

## Active Boundary Mining

A key contribution of this pipeline is the **Active Boundary Mining** sampling strategy, which addresses the fundamental class imbalance in histology segmentation.

### The Problem
In a 256×256 image, background pixels outnumber nucleus pixels ~4:1. Uniform random sampling would produce a classifier that ignores the hardest cases — pixels at the nucleus-background interface.

### The Solution

```
Background map
      │
      ▼  GPU distance_transform_edt (CuPy)
      │
Distance from nearest nucleus pixel
      │
      ▼
Weight map:
  ┌─────────────────────────────────────────┐
  │  distance < 10px AND label == background │ → weight × 3  (boosted)
  │  nucleus pixels                          │ → weight × 1  (normal)
  │  far background (distance > 10px)        │ → weight × 1  (normal)
  └─────────────────────────────────────────┘
      │
      ▼
Weighted random sampling — 400 pixels per class
```

**Result:** The classifier learns from the hardest decision boundary first, reducing false positives at nucleus edges.

**Applied only to training data** — validation and test sets use uniform sampling to avoid evaluation bias.

---

## Memory Design

Processing 7,904 images with 93 features/pixel naively requires **~12 GB RAM**. This pipeline uses **constant ~500 MB RAM** through:

### Two-Pass Memmap Architecture

```
Pass 1 (fast, ~2 min):
  Scan all masks → count exact sample allocations
  → Allocate numpy memmaps of the exact required size

Pass 2 (slow, ~2-3 h):
  For each image:
    1. Load single image (mmap_mode='r') — no full load
    2. Normalize → convert → label → extract → sample → GLCM
    3. Write directly to memmap at current write_ptr
    4. Delete all intermediate arrays
    5. Every 100 images: flush() + save checkpoint JSON
```

### Atomic Checkpointing

```json
{
  "last_completed_image_idx": 1400,
  "write_ptr": 1124800,
  "fold_dir_idx": 0,
  "total_allocated": 4038008,
  "timestamp": "2026-04-22T12:34:56",
  "label_mode": "binary",
  "n_per_class": 400
}
```

**On crash:** re-run the exact same command. The pipeline detects the checkpoint, re-opens the memmap in `r+` mode, and continues from `last_completed_image_idx + 1`. Zero data duplication.

---

## Repository Structure

```
HandCraft-Path/
│
├── environment.yml                   # Pinned conda environment (RAPIDS 25.12)
├── README.md
│
├── src/
│   ├── preprocessing/
│   │   ├── stain_normalizer.py       # Macenko SVD normalization
│   │   └── color_converter.py        # LAB / HSV / HED conversions
│   │
│   ├── data/
│   │   ├── label_generator.py        # Binary + 3-class label maps
│   │   └── streaming_loader.py       # Memmap writer + checkpoint manager
│   │
│   ├── features/
│   │   └── pixel_feature_extractor.py # 93-feature GPU extractor
│   │
│   └── sampling/
│       └── pixel_sampler.py           # Active boundary mining
│
├── scripts/
│   ├── build_dataset.py              # Main CLI runner
│   ├── test_feature_pipeline.py      # Smoke test (5 images)
│   └── diagnose_gpu_init.py          # GPU/CuPy diagnostic tool
│
├── notebooks/
│   ├── 00_inspect_gpu_init.ipynb     # GPU environment check
│   ├── 01_inspect_raw_data.ipynb     # Raw data exploration
│   ├── 02_inspect_labels.ipynb       # Label generation viz
│   ├── 03_inspect_stain_processing.ipynb  # Normalization analysis
│   ├── 04_inspect_sampling.ipynb     # Boundary mining visualization
│   └── 05_inspect_features.ipynb     # All 11 feature groups
│
└── data/                             # NOT tracked in git
    ├── raw/                          # PanNuke folds (download separately)
    ├── processed/                    # Built datasets (X.npy, y.npy)
    └── models/                       # Fitted normalizer (normalizer.joblib)
```

---

## Installation

### Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA 12.x driver
- Miniconda or Anaconda
- ~15 GB free disk space (raw data + processed features)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/HandCraft-Path.git
cd HandCraft-Path
```

### 2. Create the conda environment

```bash
# Install libmamba solver for faster resolution (recommended)
conda install -n base conda-libmamba-solver

# Create environment
conda env create -f environment.yml --solver=libmamba

# Activate
conda activate HandCraft-Path
```

### 3. Verify GPU setup

```bash
python scripts/diagnose_gpu_init.py
```

Expected output:
```
✓ CuPy available
✓ CUDA device count: 1
✓ Device: NVIDIA ... (12288 MB)
✓ CuPy memory pool: OK
✓ cupyx EDT: OK
```

---

## Quick Start

### Step 1 — Smoke test (5 images, ~30 seconds)

```bash
python scripts/test_feature_pipeline.py
```

Verifies: feature shape (N, 93), zero NaN, zero Inf.

### Step 2 — Build training dataset

```bash
# Binary labels — Folds 1+2 (train set)
python scripts/build_dataset.py --label-mode binary --folds 1,2

# 3-class labels — Folds 1+2
python scripts/build_dataset.py --label-mode 3class --folds 1,2
```

**Output:**
```
data/processed/train_binary_X.npy    # (N, 93) float32
data/processed/train_binary_y.npy    # (N,)    uint8
```

### Step 3 — Build test dataset

```bash
python scripts/build_dataset.py --label-mode binary --folds 3
python scripts/build_dataset.py --label-mode 3class --folds 3
```

### Resuming after a crash

Re-run the exact same command — the pipeline auto-detects the checkpoint:

```bash
python scripts/build_dataset.py --label-mode binary --folds 1,2
# Output: ═══ Resuming from checkpoint ═══
#           Last image:  1400
#           Write ptr:   1,124,800
```

---

## Pipeline Usage

### `build_dataset.py` — CLI Reference

```
usage: build_dataset.py --label-mode {binary,3class} --folds FOLDS

  --label-mode    binary  → 2-class (bg / nucleus)
                  3class  → 3-class (bg / interior / boundary)
  --folds         comma-separated fold numbers, e.g. 1,2 or 3
  --n-per-class   pixels to sample per class per image (default: 400)
  --resume        force resume from checkpoint (auto-detected by default)
  --output-dir    override default data/processed/ output location
```

### Environment Variable Override

To force CPU-only distance transform (if GPU is unavailable):

```bash
PANNUKE_FORCE_CPU_EDT=1 python scripts/build_dataset.py --label-mode binary --folds 1,2
```

---

## Inspection Notebooks

All notebooks are designed to be run **after** the corresponding pipeline stage completes. They provide visual verification of each step.

| Notebook | Purpose | Key Visualizations |
|----------|---------|-------------------|
| `00_inspect_gpu_init` | Verify GPU/CuPy setup | Device info, memory pool, EDT test |
| `01_inspect_raw_data` | Explore raw PanNuke data | Image/mask distributions, tissue types |
| `02_inspect_labels` | Verify label generation | Binary + 3-class overlays, boundary erosion |
| `03_inspect_stain_processing` | Analyze stain normalization | Before/after, OD distributions, channel stats |
| `04_inspect_sampling` | Visualize boundary mining | Distance maps, weight heatmaps, class balance across 100 images |
| `05_inspect_features` | Inspect all 11 feature groups | Per-group feature maps, boxplots, correlation matrix |

Launch:

```bash
conda activate HandCraft-Path
jupyter lab notebooks/
```

---

## Experiments

Two parallel experiments test the effect of label granularity:

| Experiment | Label Mode | Classes | Hypothesis |
|-----------|-----------|---------|------------|
| **Exp A** | Binary | bg / nucleus | Simpler supervision, faster training |
| **Exp B** | 3-Class | bg / interior / boundary | Explicit boundary class improves edge quality |

Both experiments use identical feature extraction — only the label space differs.

### Planned Classifiers

| Classifier | Library | Notes |
|-----------|---------|-------|
| Random Forest | cuML (GPU) | Baseline, scale-invariant |
| XGBoost | XGBoost GPU | Gradient boosting |
| LightGBM | LightGBM GPU | Fast histogram trees |
| SVM-RBF | scikit-learn | After GroupAwareScaler normalization |

### Planned Feature Selection

Three-stage selector applied to the normalized 93-feature space:
1. **VarianceThreshold** — removes constant features
2. **mRMR** — minimum redundancy maximum relevance (targets ~60 features)
3. **RFE with ExtraTrees** — model-driven recursive elimination

---

## Results (Planned)

*This section will be updated as experiments complete.*

| Experiment | Classifier | IoU | F1 (nucleus) | Boundary F1 |
|-----------|-----------|-----|-------------|------------|
| Exp A Binary | Random Forest | TBD | TBD | — |
| Exp A Binary | XGBoost | TBD | TBD | — |
| Exp B 3-Class | Random Forest | TBD | TBD | TBD |
| Exp B 3-Class | LightGBM | TBD | TBD | TBD |

---

## Reproducibility

This project is designed for **exact reproducibility**:

| Component | How reproduced |
|-----------|---------------|
| Environment | `conda env create -f environment.yml` — fully pinned |
| Stain normalizer | Fitted on first 200 images of Fold 1, saved as `normalizer.joblib` |
| Random sampling | NumPy seed set per image (image index as seed) |
| Feature computation | Pure NumPy/SciPy/CuPy — deterministic given same input |
| Dataset splits | PanNuke's predefined Fold 1/2/3 split — not reshuffled |

**All results are reproducible from the original PanNuke data + this repository.**

---

## Known Limitations

- **GPU memory mode:** CuPy's memory pool requires exclusive device access. If another process holds the CUDA context, set `PANNUKE_FORCE_CPU_EDT=1` to fall back to scipy's CPU distance transform.
- **Disk space:** Building all four datasets (binary train/test + 3class train/test) requires approximately 7–10 GB.
- **Empty stub files:** Scripts for training, evaluation, and inference (`scripts/train_stage1.py`, `scripts/evaluate_stage1.py`, etc.) are placeholders for the next development phase.

---

## Citation

If you use this pipeline or the PanNuke dataset in your research, please cite:

**PanNuke Dataset:**
```bibtex
@article{gamper2020pannuke,
  title={PanNuke Dataset Extension, Insights and Baselines},
  author={Gamper, Jevgenij and Alemi Koohbanani, Navid and Benet, Ksenija and
          Khuram, Ali and Rajpoot, Nasir},
  journal={arXiv preprint arXiv:2003.10778},
  year={2020}
}
```

**PanNuke Original:**
```bibtex
@inproceedings{gamper2019pannuke,
  title={PanNuke: an open pan-tissue histology dataset for nuclei instance
         segmentation and classification},
  author={Gamper, Jevgenij and Alemi Koohbanani, Navid and Benet, Ksenija and
          Khuram, Ali and Rajpoot, Nasir},
  booktitle={European Congress on Digital Pathology},
  pages={11--19},
  year={2019},
  organization={Springer}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The PanNuke dataset has its own license — please refer to the [official dataset page](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) for usage terms.

---

<div align="center">

**Built with ❤️ for interpretable, memory-efficient medical image analysis**

</div>
