"""
Build training/test datasets for Stage 1 segmentation.

Usage:
  python scripts/build_dataset.py --label-mode binary --folds 1,2
  python scripts/build_dataset.py --label-mode 3class --folds 1,2
  python scripts/build_dataset.py --label-mode binary --folds 3
  python scripts/build_dataset.py --label-mode 3class --folds 3
  python scripts/build_dataset.py --resume  # resume last interrupted run

Each run:
  1. Loads or fits+saves the stain normalizer
  2. Streams all images via memmap (constant ~500 MB RAM)
  3. Writes features directly to disk with checkpoints every 100 images
  4. Outputs: data/processed/{split}_{mode}_X.npy and _y.npy
"""

import argparse
import sys
import os
import numpy as np

# Add project root to path (anchored to this script's location)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.stain_normalizer import MacenkoNormalizer


# ── Data paths (anchored to project root, safe from any cwd) ─────────────

DATA_ROOT     = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "data", "models")
NORMALIZER_PATH = os.path.join(MODELS_DIR, "normalizer.joblib")


def get_fold_paths(fold_num):
    """Return (images_path, masks_path) for a fold number."""
    images_path = os.path.join(
        DATA_ROOT, f"Fold{fold_num}", "images", f"fold{fold_num}", "images.npy"
    )
    masks_path = os.path.join(
        DATA_ROOT, f"Fold{fold_num}", "masks", f"fold{fold_num}", "masks.npy"
    )
    return images_path, masks_path


def fit_or_load_normalizer(fold_nums):
    """
    Load existing normalizer or fit on training folds and save.

    Fits on ALL images from the specified folds (using mmap_mode='r'
    to avoid loading everything into RAM at once).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(NORMALIZER_PATH):
        print(f"Loading existing normalizer from {NORMALIZER_PATH}")
        return MacenkoNormalizer.load(NORMALIZER_PATH)

    print("Fitting normalizer on training images...")
    # Collect a subset for fitting (first 200 images from first fold)
    # Using all images is overkill — median OD stabilizes after ~100
    first_fold = fold_nums[0]
    images_path = get_fold_paths(first_fold)[0]
    images = np.load(images_path, mmap_mode='r')

    # Use first 200 images for fitting (fast, stable reference selection)
    fit_subset = images[:200]

    normalizer = MacenkoNormalizer()
    normalizer.fit(fit_subset)
    normalizer.save(NORMALIZER_PATH)
    return normalizer


def main():
    parser = argparse.ArgumentParser(description="Build Stage 1 datasets")
    parser.add_argument(
        "--label-mode", choices=["binary", "3class"], required=True,
        help="Label generation mode"
    )
    parser.add_argument(
        "--folds", type=str, required=True,
        help="Comma-separated fold numbers (e.g., '1,2' or '3')"
    )
    parser.add_argument(
        "--n-per-class", type=int, default=400,
        help="Pixels to sample per class per image (default: 400)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint if available"
    )
    parser.add_argument(
        "--output-dir", type=str, default=PROCESSED_DIR,
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--force-cpu-edt", action="store_true",
        help="Force CPU distance transform in pixel sampler (disables GPU EDT)"
    )
    parser.add_argument(
        "--feature-backend", choices=["auto", "cpu", "gpu"], default="auto",
        help="Dense feature extraction backend (default: auto)"
    )
    parser.add_argument(
        "--feature-batch-size", type=int, default=None,
        help="Dense feature extraction batch size (default: auto-computed from available memory)"
    )
    
    # ── NEW: Memory configuration arguments ──────────────────────────────────
    parser.add_argument(
        "--max-memory", type=float, default=None,
        help="Maximum memory to use in GB (e.g., 8). Overrides auto-detection."
    )
    parser.add_argument(
        "--memory-percent", type=float, default=None,
        help="Use only this percentage of available memory (0-100). Useful for shared systems."
    )
    parser.add_argument(
        "--memory-backend", choices=["auto", "cpu", "gpu"], default="auto",
        help="Memory backend for size calculation (default: auto, GPU if available)"
    )
    parser.add_argument(
        "--memory-safety-margin", type=float, default=25,
        help="Safety margin as %% of detected memory (default: 25). Higher = more conservative."
    )
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="Skip auto-calibration of per-image memory (faster but less accurate)"
    )

    args = parser.parse_args()

    if args.force_cpu_edt:
        os.environ["PANNUKE_FORCE_CPU_EDT"] = "1"
        print("Forcing CPU EDT in pixel sampler (PANNUKE_FORCE_CPU_EDT=1)")

    # Import after optional env flags are set.
    from src.data.streaming_loader import build_dataset_memmap
    from src.utils.memory_config import MemoryConfig

    fold_nums = [int(f.strip()) for f in args.folds.split(",")]

    # Determine split name
    if set(fold_nums) == {1, 2}:
        split = "train"
    elif set(fold_nums) == {3}:
        split = "test"
    else:
        split = f"fold{'_'.join(map(str, fold_nums))}"

    print(f"{'═'*60}")
    print(f"Building {split}_{args.label_mode} dataset")
    print(f"  Folds: {fold_nums}")
    print(f"  Label mode: {args.label_mode}")
    print(f"  Pixels per class: {args.n_per_class}")
    print(f"{'═'*60}\n")

    # Output paths
    os.makedirs(args.output_dir, exist_ok=True)
    output_X = os.path.join(args.output_dir, f"{split}_{args.label_mode}_X.npy")
    output_y = os.path.join(args.output_dir, f"{split}_{args.label_mode}_y.npy")
    ckpt_path = os.path.join(args.output_dir, f"checkpoint_{split}_{args.label_mode}.json")

    # Check for existing output
    if os.path.exists(output_X) and not args.resume:
        if not os.path.exists(ckpt_path):
            print(f"Output already exists: {output_X}")
            print("Use --resume to continue, or delete the file to restart.")
            return

    # ── NEW: Initialize memory configuration ──────────────────────────────
    memory_config = MemoryConfig(
        max_memory_gb=args.max_memory,
        memory_percent=args.memory_percent,
        safety_margin_pct=args.memory_safety_margin,
        force_backend=args.memory_backend,
        verbose=True,
    )

    # Fit/load normalizer
    normalizer = fit_or_load_normalizer(fold_nums)

    # Build fold paths
    fold_dirs = [get_fold_paths(f) for f in fold_nums]

    # Verify all files exist
    for img_path, mask_path in fold_dirs:
        assert os.path.exists(img_path), f"Missing: {img_path}"
        assert os.path.exists(mask_path), f"Missing: {mask_path}"

    # ── NEW: Auto-calibrate per-image memory (if not disabled) ────────────
    if not args.no_calibration:
        try:
            memory_config.calibrate_from_images(fold_dirs, n_calibrate_images=5)
        except Exception as e:
            print(f"⚠ Auto-calibration failed: {e}. Using conservative estimate.")
    
    # ── Compute optimal batch size ────────────────────────────────────────
    feature_batch_size = memory_config.compute_batch_size(
        feature_batch_size_override=args.feature_batch_size
    )

    # Print memory summary
    memory_config.print_summary()

    print(f"  Feature backend: {args.feature_backend}")
    print(f"  Computed batch size: {feature_batch_size}\n")

    # Build dataset
    build_dataset_memmap(
        fold_dirs=fold_dirs,
        normalizer=normalizer,
        label_mode=args.label_mode,
        output_X_path=output_X,
        output_y_path=output_y,
        checkpoint_path=ckpt_path,
        n_per_class=args.n_per_class,
        feature_backend=args.feature_backend,
        feature_batch_size=feature_batch_size,
        memory_config=memory_config,
    )


if __name__ == "__main__":
    main()
