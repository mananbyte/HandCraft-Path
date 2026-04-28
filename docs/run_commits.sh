#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Retroactive Git Commit Script — PanNuke Build-Dataset Pipeline
# ═══════════════════════════════════════════════════════════
set -e  # exit on any error

echo "════════════════════════════════════════════════════"
echo " PanNuke — Creating commit history"
echo "════════════════════════════════════════════════════"

# ── Commit 1 (main): Project init ─────────────────────────
echo ""
echo "[1/8] feat: initialize project structure and environment"

git add .gitignore
git add README.md
git add environment.yml
git add src/__init__.py
git add src/preprocessing/__init__.py
git add src/data/__init__.py
git add src/features/__init__.py
git add src/sampling/__init__.py
git add src/training/__init__.py
git add src/inference/__init__.py

GIT_AUTHOR_DATE="2026-04-10T10:00:00+05:00" \
GIT_COMMITTER_DATE="2026-04-10T10:00:00+05:00" \
git commit -m "feat: initialize project structure and environment

- Add environment.yml with all dependencies (cupy, scikit-image, scipy, etc.)
- Create src/ package skeleton with __init__.py files
- Set up .gitignore for data, bytecode, and private docs"

echo "  ✓ Commit 1 done (main)"

# ── Create feature branch ─────────────────────────────────
echo ""
echo "Creating branch: feature/build-dataset-pipeline"
git checkout -b feature/build-dataset-pipeline

# ── Commit 2: Stain normalization ─────────────────────────
echo ""
echo "[2/8] feat: add stain normalization and color conversion"

git add src/preprocessing/stain_normalizer.py
git add src/preprocessing/color_converter.py
git add scripts/test_stain_normalizer.py

GIT_AUTHOR_DATE="2026-04-12T14:30:00+05:00" \
GIT_COMMITTER_DATE="2026-04-12T14:30:00+05:00" \
git commit -m "feat: add Macenko stain normalization and color conversion

- Implement MacenkoNormalizer with automatic reference image selection
- Add color_converter.py for LAB, HSV, HED space conversions
- Add test script for normalizer verification
- Normalizer fits on first 200 images, selects best reference by OD spread"

echo "  ✓ Commit 2 done"

# ── Commit 3: Label generation ────────────────────────────
echo ""
echo "[3/8] feat: add label generation for binary and 3-class modes"

git add src/data/label_generator.py

GIT_AUTHOR_DATE="2026-04-14T11:00:00+05:00" \
GIT_COMMITTER_DATE="2026-04-14T11:00:00+05:00" \
git commit -m "feat: add label generation for binary and 3-class modes

- Binary mode: 0=background, 1=nucleus (from 6-channel PanNuke masks)
- 3-class mode: 0=background, 1=interior, 2=boundary (morphological erosion)
- Handles edge cases: empty masks, single-pixel nuclei"

echo "  ✓ Commit 3 done"

# ── Commit 4: Feature extractor ───────────────────────────
echo ""
echo "[4/8] feat: add GPU-accelerated 93-feature pixel extractor"

git add src/features/pixel_feature_extractor.py

GIT_AUTHOR_DATE="2026-04-16T16:00:00+05:00" \
GIT_COMMITTER_DATE="2026-04-16T16:00:00+05:00" \
git commit -m "feat: add GPU-accelerated 93-feature pixel extractor

- 87 dense features: OD(3), color stats(54), LBP(3), Gabor(12),
  gradients(5), structure tensor(3), DoG(3), superpixel(2),
  entropy(1), edge distance(1)
- 6 GLCM features computed per sampled pixel (15x15 patch)
- GPU acceleration via CuPy with automatic CPU fallback
- Zero NaN/Inf guarantee across all feature groups"

echo "  ✓ Commit 4 done"

# ── Commit 5: Pixel sampler ───────────────────────────────
echo ""
echo "[5/8] feat: add active boundary mining pixel sampler"

git add src/sampling/pixel_sampler.py

GIT_AUTHOR_DATE="2026-04-18T13:00:00+05:00" \
GIT_COMMITTER_DATE="2026-04-18T13:00:00+05:00" \
git commit -m "feat: add active boundary mining pixel sampler

- GPU-accelerated distance transform for boundary detection
- 3x oversampling of background pixels within 10px of nucleus edges
- Class-balanced sampling with configurable n_per_class (default 400)
- CPU fallback via scipy.ndimage.distance_transform_edt"

echo "  ✓ Commit 5 done"

# ── Commit 6: Streaming loader ────────────────────────────
echo ""
echo "[6/8] feat: add memmap streaming loader with checkpoint/resume"

git add src/data/streaming_loader.py

GIT_AUTHOR_DATE="2026-04-20T15:00:00+05:00" \
GIT_COMMITTER_DATE="2026-04-20T15:00:00+05:00" \
git commit -m "feat: add memmap streaming loader with checkpoint/resume

- Two-pass design: count samples first, then allocate + write
- numpy memmap for constant ~500MB RAM regardless of dataset size
- Atomic checkpoint every 100 images (flush-first, JSON-second)
- Automatic resume from last checkpoint on re-run
- Truncates over-allocated memmaps on completion"

echo "  ✓ Commit 6 done"

# ── Commit 7: CLI scripts ────────────────────────────────
echo ""
echo "[7/8] feat: add build_dataset CLI and diagnostic scripts"

git add scripts/build_dataset.py
git add scripts/test_feature_pipeline.py
git add scripts/diagnose_gpu_init.py

GIT_AUTHOR_DATE="2026-04-22T10:00:00+05:00" \
GIT_COMMITTER_DATE="2026-04-22T10:00:00+05:00" \
git commit -m "feat: add build_dataset CLI and diagnostic scripts

- build_dataset.py: main runner with --label-mode, --folds, --resume flags
- test_feature_pipeline.py: smoke test on 5 images (shape + NaN checks)
- diagnose_gpu_init.py: GPU availability and CuPy diagnostics
- All paths anchored to PROJECT_ROOT for cwd-independent execution"

echo "  ✓ Commit 7 done"

# ── Commit 8: Inspection notebooks ───────────────────────
echo ""
echo "[8/8] feat: add inspection notebooks for pipeline verification"

git add notebooks/00_inspect_gpu_init.ipynb
git add notebooks/01_inspect_raw_data.ipynb
git add notebooks/02_inspect_labels.ipynb
git add notebooks/03_inspect_stain_processing.ipynb
git add notebooks/04_inspect_sampling.ipynb
git add notebooks/05_inspect_features.ipynb

GIT_AUTHOR_DATE="2026-04-24T17:00:00+05:00" \
GIT_COMMITTER_DATE="2026-04-24T17:00:00+05:00" \
git commit -m "feat: add inspection notebooks for pipeline verification

- 00: GPU initialization check
- 01: Raw data exploration (image/mask shapes, tissue types)
- 02: Label generation visualization (binary + 3-class)
- 03: Stain normalization before/after + channel distributions
- 04: Sampling strategy — distance transform, weight maps, class balance
- 05: All 11 feature groups visualized with correlation analysis"

echo "  ✓ Commit 8 done"

# ── Summary ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " Done! 8 commits created."
echo "════════════════════════════════════════════════════"
echo ""
echo "Branch structure:"
echo "  main                          ← Commit 1 (project init)"
echo "  feature/build-dataset-pipeline ← Commits 2-8 (you are here)"
echo ""
git log --oneline --all --graph
