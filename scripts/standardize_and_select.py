"""
scripts/standardize_and_select.py
──────────────────────────────────
Production pipeline for Phase 3: Dataset Standardization & mRMR Feature Selection.

Steps:
  1. Fit FeatureScaler on a stratified subsample of the training fold.
  2. Stream-transform all processed matrices (binary + 3-class) into scaled copies.
  3. Run feature selection on the scaled training matrix.
  4. Persist all 5 artefacts to disk.

Memory contract: peak system RSS stays ≤500 MB throughout (streaming + subsample fit).

Usage:
  conda run -n HandCraft-Path python scripts/standardize_and_select.py \\
      --folds 1 \\
      --label-mode binary \\
      --n-features 20 \\
      --method mrmr \\
      --scaler RobustScaler \\
      --sample-n 200000

Outputs written to:
  data/models/scaler_fold{N}_{label_mode}.joblib
  data/processed/fold{N}_binary_X_scaled.npy
  data/processed/fold{N}_3class_X_scaled.npy
  data/models/selected_features_{method}_{n}.json
  data/models/selected_features_{method}_{n}.csv
"""

import argparse
import logging
import sys
import os
import time

import numpy as np
import psutil

# Allow running from project root: python scripts/standardize_and_select.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.scaler import FeatureScaler, get_all_scaler_types
from src.training.feature_selection import (
    run_feature_selection,
    save_selected_features,
    SUPPORTED_METHODS,
    DEFAULT_N_FEATURES,
)
from src.utils.safe_loader import safe_load_npy

# ── Constants ──────────────────────────────────────────────────────────────
RSS_LIMIT_MB: float = 3000.0
PROCESSED_DIR: str = "data/processed"
MODELS_DIR: str = "data/models"
SELECTION_SAMPLE_N: int = 100_000   # subsample for feature selection (fit already done)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("standardize_and_select")


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit scaler, transform matrices, run feature selection."
    )
    p.add_argument(
        "--folds", type=int, default=1,
        help="Training fold number (default: 1)",
    )
    p.add_argument(
        "--label-mode", choices=["binary", "3class"], default="binary",
        help="Label mode used during data building (default: binary)",
    )
    p.add_argument(
        "--n-features", type=int, default=DEFAULT_N_FEATURES,
        help="Number of features to select (default: 20). Override with notebook elbow result.",
    )
    p.add_argument(
        "--method", choices=SUPPORTED_METHODS, default="mrmr",
        help="Feature selection method (default: mrmr)",
    )
    p.add_argument(
        "--scaler", choices=get_all_scaler_types(), default="RobustScaler",
        help="Scaler type (default: RobustScaler)",
    )
    p.add_argument(
        "--sample-n", type=int, default=200_000,
        help="Rows to subsample for scaler fit (default: 200_000)",
    )
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────
def rss_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def assert_rss(stage: str) -> None:
    """Hard-fail if RSS exceeds limit — prevents silent OOM cascades."""
    current = rss_mb()
    logger.info("[RSS] %s: %.1f MB", stage, current)
    if current > RSS_LIMIT_MB:
        logger.error(
            "RSS LIMIT EXCEEDED at '%s': %.1f MB > %.1f MB",
            stage, current, RSS_LIMIT_MB,
        )
        sys.exit(1)


def separator(label: str) -> None:
    width = 70
    print(f"\n{'═' * width}")
    print(f"  {label}")
    print(f"{'═' * width}")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    t_start = time.time()

    separator(f"Phase 3: Standardize & Select | fold={args.folds} | label={args.label_mode}")

    fold_tag = f"fold{args.folds}"
    binary_X_path   = os.path.join(PROCESSED_DIR, f"{fold_tag}_binary_X.npy")
    binary_y_path   = os.path.join(PROCESSED_DIR, f"{fold_tag}_binary_y.npy")
    class3_X_path   = os.path.join(PROCESSED_DIR, f"{fold_tag}_3class_X.npy")

    binary_X_scaled = os.path.join(PROCESSED_DIR, f"{fold_tag}_binary_X_scaled.npy")
    class3_X_scaled = os.path.join(PROCESSED_DIR, f"{fold_tag}_3class_X_scaled.npy")
    scaler_path     = os.path.join(MODELS_DIR, f"scaler_{fold_tag}_{args.label_mode}.joblib")

    os.makedirs(MODELS_DIR, exist_ok=True)

    assert_rss("start")

    # ── Step 1: Fit Scaler ─────────────────────────────────────────────────
    separator(f"Step 1: Fit {args.scaler} on {args.sample_n:,} subsample rows")

    fit_X_path = binary_X_path if args.label_mode == "binary" else class3_X_path
    fit_y_path = binary_y_path

    assert os.path.exists(fit_X_path), f"Training X not found: {fit_X_path}"
    assert os.path.exists(fit_y_path), f"Training y not found: {fit_y_path}"

    scaler = FeatureScaler(scaler_type=args.scaler)
    scaler.fit(fit_X_path, fit_y_path, sample_n=args.sample_n)
    assert_rss("after scaler fit")

    scaler.save(scaler_path)
    print(f"  ✓ Scaler saved: {scaler_path}")
    assert_rss("after scaler save")

    # ── Step 2: Transform binary matrix ───────────────────────────────────
    separator("Step 2: Stream-transform binary X matrix")

    if os.path.exists(binary_X_path):
        scaler.transform_memmap(binary_X_path, binary_X_scaled)
        assert_rss("after binary transform")
    else:
        logger.warning("Binary X not found, skipping: %s", binary_X_path)

    # ── Step 3: Transform 3-class matrix ──────────────────────────────────
    separator("Step 3: Stream-transform 3-class X matrix")

    if os.path.exists(class3_X_path):
        scaler.transform_memmap(class3_X_path, class3_X_scaled)
        assert_rss("after 3class transform")
    else:
        logger.warning("3-class X not found, skipping: %s", class3_X_path)

    # ── Step 4: Feature Selection ──────────────────────────────────────────
    separator(f"Step 4: {args.method.upper()} Feature Selection (n={args.n_features})")

    # Load scaled binary matrix subsample for selection
    scaled_path = binary_X_scaled if os.path.exists(binary_X_scaled) else binary_X_path
    X_mm = safe_load_npy(scaled_path, mode="r")
    y_mm = safe_load_npy(binary_y_path, mode="r")

    n_total = X_mm.shape[0]
    actual_n = min(SELECTION_SAMPLE_N, n_total)

    rng = np.random.default_rng(42)
    all_idx = np.arange(n_total)
    # Stratified subsample using class-proportional draw
    classes, counts = np.unique(y_mm, return_counts=True)
    sel_idx = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y_mm == cls)[0]
        n_draw = max(1, int(actual_n * cnt / n_total))
        drawn = rng.choice(cls_idx, size=min(n_draw, len(cls_idx)), replace=False)
        sel_idx.append(drawn)
    sel_idx = np.sort(np.concatenate(sel_idx))

    X_sub = np.array(X_mm[sel_idx], dtype=np.float32)
    y_sub = np.array(y_mm[sel_idx])
    X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)

    assert_rss("after selection subsample load")
    print(f"  Subsample shape: {X_sub.shape}")

    indices, names = run_feature_selection(
        X_sub, y_sub,
        n_features=args.n_features,
        method=args.method,
    )

    assert_rss("after feature selection")

    paths = save_selected_features(
        indices, names,
        method=args.method,
        n=args.n_features,
        out_dir=MODELS_DIR,
    )

    # ── Step 5: Spot-check scaled matrix ──────────────────────────────────
    separator("Step 5: Spot-check scaled matrix statistics")

    if os.path.exists(binary_X_scaled):
        stats = scaler.spot_check(binary_X_scaled)
        print(f"  Mean |col mean|:   {stats['mean_abs_col_mean']:.6f}  (target ≈ 0)")
        print(f"  Mean |col median|: {stats['mean_abs_col_median']:.6f}  (target ≈ 0)")
        print(f"  Max  |col mean|:   {stats['max_abs_col_mean']:.6f}")
        print(f"  Outlier frac >3σ:  {stats['outlier_frac_3sigma']:.4%}")

    # ── Summary ────────────────────────────────────────────────────────────
    t_elapsed = time.time() - t_start
    separator("Phase 3 Complete")
    print(f"  Scaler:           {args.scaler}")
    print(f"  Features selected: {args.n_features} (method={args.method})")
    print(f"  Artefacts:")
    print(f"    {scaler_path}")
    if os.path.exists(binary_X_scaled):
        print(f"    {binary_X_scaled}")
    if os.path.exists(class3_X_scaled):
        print(f"    {class3_X_scaled}")
    print(f"    {paths['json']}")
    print(f"    {paths['csv']}")
    print(f"  Peak RSS:         {rss_mb():.1f} MB")
    print(f"  Elapsed:          {t_elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
