#!/usr/bin/env python3
"""
scripts/train_ensemble.py
─────────────────────────
Train, tune (Optuna HPO), and ensemble binary pixel classifiers.

Flow:
  1. Load fold1_binary_X_scaled.npy → slice to 25 RFE features
  2. Run per-model Optuna HPO studies on 200K stratified subsample
  3. Refit best params on full Fold 1
  4. Scale Fold 2 in-memory → slice to 25 features
  5. Grid search weight simplex on Fold 2 predictions
  6. Save all model artefacts

Memory contract: peak RSS stays within OS limits by using memmap reads and
streaming Fold 2 through the loaded scaler in 50K-row chunks.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import joblib
import numpy as np
import optuna
import psutil
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.ensemble import SoftVotingEnsemble, build_default_estimators
from src.training.scaler import FeatureScaler
from src.utils.safe_loader import safe_load_npy

# ── Constants ──────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
FOLD2_CHUNK_SIZE: int = 50_000      # rows per Fold-2 scaling chunk
RSS_REPORT_STAGES: list = []        # track stages for telemetry
PROCESSED_DIR: str = "data/processed"
MODELS_DIR: str = "data/models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Silence Optuna's verbose trial-level output
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("train_ensemble")


# ── Helpers ────────────────────────────────────────────────────────────────
def rss_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def separator(label: str) -> None:
    width = 72
    print(f"\n{'═' * width}")
    print(f"  {label}")
    print(f"{'═' * width}")


def print_rss(stage: str) -> None:
    logger.info("[RSS] %s: %.1f MB", stage, rss_mb())


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train, tune, and ensemble binary pixel classifiers (Phase 4)."
    )
    p.add_argument("--fold", type=int, default=1, help="Training fold number (default: 1)")
    p.add_argument(
        "--selected-features",
        type=str,
        default="data/models/selected_features_rfe_25.json",
        help="Path to selected_features JSON (default: data/models/selected_features_rfe_25.json)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/models/",
        help="Output directory for model artefacts (default: data/models/)",
    )
    p.add_argument("--n-trials", type=int, default=25, help="Optuna trials per model (default: 25)")
    p.add_argument(
        "--hpo-sample-n",
        type=int,
        default=200_000,
        help="Subsample size for Optuna HPO (default: 200000)",
    )
    p.add_argument(
        "--optuna-timeout",
        type=int,
        default=1800,
        help="Wall-clock timeout per Optuna study in seconds (default: 1800)",
    )
    return p.parse_args()


# ── Optuna objectives ──────────────────────────────────────────────────────
def _rf_objective(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    y_vl: np.ndarray,
) -> float:
    """RF Optuna objective: maximise Macro-F1 on 20% HPO val split."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 8, 30),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }
    from src.training.ensemble import CuMLRF, HAS_CUML  # local import to get updated HAS_CUML
    rf = CuMLRF(random_state=RANDOM_STATE, **params)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_vl)
    return float(f1_score(y_vl, np.array(y_pred), average="macro", zero_division=0))


def _lgbm_objective(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    y_vl: np.ndarray,
) -> float:
    """LGBM Optuna objective."""
    import lightgbm as lgb
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
    }
    clf = lgb.LGBMClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params
    )
    clf.fit(X_tr, y_tr)
    return float(f1_score(y_vl, clf.predict(X_vl), average="macro", zero_division=0))


def _xgb_objective(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    y_vl: np.ndarray,
) -> float:
    """XGBoost Optuna objective."""
    import xgboost as xgb
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    clf = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False,
        **params,
    )
    clf.fit(X_tr, y_tr)
    return float(f1_score(y_vl, clf.predict(X_vl), average="macro", zero_division=0))


# ── Subsample helper ───────────────────────────────────────────────────────
def draw_subsample(
    X_mm: np.ndarray,
    y_mm: np.ndarray,
    n: int,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw a stratified subsample of n rows from memmap arrays.
    Returns dense float32 arrays safe to use with GPU estimators.
    """
    n_total = X_mm.shape[0]
    actual_n = min(n, n_total)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=actual_n / n_total, random_state=random_state
    )
    _, idx = next(sss.split(np.zeros(n_total), y_mm))
    idx = np.sort(idx)
    X_sub = np.array(X_mm[idx], dtype=np.float32)
    y_sub = np.array(y_mm[idx], dtype=np.int32)
    X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
    return X_sub, y_sub


# ── Scale Fold 2 in-memory ─────────────────────────────────────────────────
def scale_fold2_inMemory(
    X2_path: str,
    scaler: FeatureScaler,
    rfe_indices: list,
) -> np.ndarray:
    """
    Stream Fold 2 through the Phase-3 scaler in 50K-row chunks and slice
    to the 25 RFE feature columns.  No new .npy file is written to disk.

    Parameters
    ----------
    X2_path    : path to fold2_binary_X.npy (raw, unscaled)
    scaler     : loaded FeatureScaler from Phase 3
    rfe_indices: list of 25 column indices

    Returns
    -------
    np.ndarray (n_rows, 25) float32 — fully in RAM
    """
    X_mm = safe_load_npy(X2_path, mode="r")
    n_total, n_cols = X_mm.shape
    n_chunks = (n_total + FOLD2_CHUNK_SIZE - 1) // FOLD2_CHUNK_SIZE
    n_rfe = len(rfe_indices)

    # Pre-allocate output (fully in RAM — Fold 2 25-col ≈ 177 MB)
    X_out = np.empty((n_total, n_rfe), dtype=np.float32)

    print_rss("scale_fold2_inMemory start")
    for i in tqdm(range(n_chunks), desc="  Scaling Fold-2", unit="chunk"):
        start = i * FOLD2_CHUNK_SIZE
        end = min(start + FOLD2_CHUNK_SIZE, n_total)
        chunk = np.array(X_mm[start:end], dtype=np.float32)
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        # Apply scaler group transform on full 93-col chunk, then slice
        scaled_chunk = scaler._scaler.transform(chunk).astype(np.float32)
        X_out[start:end] = scaled_chunk[:, rfe_indices]

    print_rss("scale_fold2_inMemory done")
    return X_out


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    fold_tag = f"fold{args.fold}"
    separator(f"Phase 4-01: Binary Ensemble Training | fold={args.fold}")
    print_rss("start")

    # ── Step 1: Load RFE feature indices ──────────────────────────────────
    separator("Step 1: Load RFE feature indices")
    with open(args.selected_features) as fh:
        feat_json = json.load(fh)
    rfe_indices: list = feat_json["indices"]
    rfe_names: list = feat_json["names"]
    assert len(rfe_indices) == 25, f"Expected 25 RFE indices, got {len(rfe_indices)}"
    print(f"  RFE features ({len(rfe_indices)}): {rfe_names[:5]}…")

    # ── Step 2: Load Fold 1 training data ─────────────────────────────────
    separator("Step 2: Load Fold 1 training matrix (memmap → slice to 25 cols)")
    X_path = os.path.join(PROCESSED_DIR, f"{fold_tag}_binary_X_scaled.npy")
    y_path = os.path.join(PROCESSED_DIR, f"{fold_tag}_binary_y.npy")

    assert os.path.exists(X_path), f"Missing: {X_path}"
    assert os.path.exists(y_path), f"Missing: {y_path}"

    X_mm = safe_load_npy(X_path, mode="r")
    y_mm = safe_load_npy(y_path, mode="r")
    n_total = X_mm.shape[0]

    print(f"  Fold 1 shape: X={X_mm.shape}, y={y_mm.shape}")
    print_rss("after memmap open")

    # Slice to 25 RFE columns — result fits in VRAM / RAM
    print("  Slicing to 25 RFE columns…")
    idx_rfe = np.array(rfe_indices)
    X_fold1 = np.array(X_mm[:, idx_rfe], dtype=np.float32)
    y_fold1 = np.array(y_mm, dtype=np.int32)
    X_fold1 = np.nan_to_num(X_fold1, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Fold 1 training matrix: {X_fold1.shape}")
    print_rss("after fold1 slice")

    # ── Step 3: HPO subsample ─────────────────────────────────────────────
    separator(f"Step 3: Stratified subsample for Optuna HPO (n={args.hpo_sample_n:,})")
    X_sub, y_sub = draw_subsample(
        X_mm=X_fold1,
        y_mm=y_fold1,
        n=args.hpo_sample_n,
    )
    # NOTE: X_fold1 already in RAM — wrap it for draw_subsample
    actual_n_sub = len(y_sub)
    print(f"  Subsample shape: {X_sub.shape}")

    # 80/20 HPO split
    sss_hpo = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, vl_idx = next(sss_hpo.split(np.zeros(actual_n_sub), y_sub))
    X_hpo_tr, X_hpo_vl = X_sub[tr_idx], X_sub[vl_idx]
    y_hpo_tr, y_hpo_vl = y_sub[tr_idx], y_sub[vl_idx]
    print(f"  HPO train: {X_hpo_tr.shape} | val: {X_hpo_vl.shape}")
    print_rss("after hpo subsample")

    # ── Step 4: RF Optuna study ────────────────────────────────────────────
    separator(f"Step 4: RF Optuna study ({args.n_trials} trials, {args.optuna_timeout}s)")
    rf_study = optuna.create_study(direction="maximize", study_name="rf_study")
    rf_study.optimize(
        lambda trial: _rf_objective(trial, X_hpo_tr, y_hpo_tr, X_hpo_vl, y_hpo_vl),
        n_trials=args.n_trials,
        timeout=args.optuna_timeout,
        show_progress_bar=True,
    )
    rf_best = rf_study.best_params
    print(f"  RF best params: {rf_best}  | F1={rf_study.best_value:.4f}")
    rf_params_path = os.path.join(args.out_dir, "rf_best_params.json")
    with open(rf_params_path, "w") as fh:
        json.dump(rf_best, fh, indent=2)
    print(f"  ✓ Saved: {rf_params_path}")
    print_rss("after rf HPO")

    # ── Step 5: LGBM Optuna study ──────────────────────────────────────────
    separator(f"Step 5: LGBM Optuna study ({args.n_trials} trials, {args.optuna_timeout}s)")
    lgbm_study = optuna.create_study(direction="maximize", study_name="lgbm_study")
    lgbm_study.optimize(
        lambda trial: _lgbm_objective(trial, X_hpo_tr, y_hpo_tr, X_hpo_vl, y_hpo_vl),
        n_trials=args.n_trials,
        timeout=args.optuna_timeout,
        show_progress_bar=True,
    )
    lgbm_best = lgbm_study.best_params
    print(f"  LGBM best params: {lgbm_best}  | F1={lgbm_study.best_value:.4f}")
    lgbm_params_path = os.path.join(args.out_dir, "lgbm_best_params.json")
    with open(lgbm_params_path, "w") as fh:
        json.dump(lgbm_best, fh, indent=2)
    print(f"  ✓ Saved: {lgbm_params_path}")
    print_rss("after lgbm HPO")

    # ── Step 6: XGB Optuna study ───────────────────────────────────────────
    separator(f"Step 6: XGB Optuna study ({args.n_trials} trials, {args.optuna_timeout}s)")
    xgb_study = optuna.create_study(direction="maximize", study_name="xgb_study")
    xgb_study.optimize(
        lambda trial: _xgb_objective(trial, X_hpo_tr, y_hpo_tr, X_hpo_vl, y_hpo_vl),
        n_trials=args.n_trials,
        timeout=args.optuna_timeout,
        show_progress_bar=True,
    )
    xgb_best = xgb_study.best_params
    print(f"  XGB best params: {xgb_best}  | F1={xgb_study.best_value:.4f}")
    xgb_params_path = os.path.join(args.out_dir, "xgb_best_params.json")
    with open(xgb_params_path, "w") as fh:
        json.dump(xgb_best, fh, indent=2)
    print(f"  ✓ Saved: {xgb_params_path}")
    print_rss("after xgb HPO")

    # ── Step 7: Refit on full Fold 1 ──────────────────────────────────────
    separator("Step 7: Refit all models on full Fold 1")
    estimators = build_default_estimators(rf_best, lgbm_best, xgb_best)
    ensemble = SoftVotingEnsemble(estimators=estimators)

    print(f"  Fitting ensemble on {X_fold1.shape[0]:,} rows × {X_fold1.shape[1]} features…")
    ensemble.fit(X_fold1, y_fold1)
    print_rss("after full-fold1 refit")

    # ── Step 8: Scale Fold 2 in-memory ────────────────────────────────────
    separator("Step 8: Scale Fold 2 in-memory via Phase-3 scaler")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_fold{args.fold}_binary.joblib")
    assert os.path.exists(scaler_path), f"Missing scaler: {scaler_path}"
    scaler = FeatureScaler.load(scaler_path)

    X2_path = os.path.join(PROCESSED_DIR, "fold2_binary_X.npy")
    y2_path = os.path.join(PROCESSED_DIR, "fold2_binary_y.npy")
    assert os.path.exists(X2_path), f"Missing: {X2_path}"
    assert os.path.exists(y2_path), f"Missing: {y2_path}"

    X_fold2 = scale_fold2_inMemory(X2_path, scaler, rfe_indices)
    y_fold2 = np.array(safe_load_npy(y2_path, mode="r"), dtype=np.int32)
    print(f"  Fold 2: {X_fold2.shape}, y: {y_fold2.shape}")
    print_rss("after fold2 scale")

    # ── Step 9: Optimise ensemble weights on Fold 2 ────────────────────────
    separator("Step 9: Optimise ensemble weights via weight simplex grid search")
    best_weights = ensemble.optimize_weights(X_fold2, y_fold2)
    print(f"  Optimal weights: {best_weights}")
    print_rss("after weight optimisation")

    # ── Step 10: Save all model artefacts ──────────────────────────────────
    separator("Step 10: Save model artefacts")

    rf_path = os.path.join(args.out_dir, "cuml_rf.joblib")
    lgbm_path = os.path.join(args.out_dir, "lgbm.joblib")
    xgb_path = os.path.join(args.out_dir, "xgboost.joblib")
    ensemble_path = os.path.join(args.out_dir, "binary_ensemble.joblib")

    joblib.dump(
        {"model": estimators["rf"], "best_params": rf_best}, rf_path, compress=3
    )
    joblib.dump(
        {"model": estimators["lgbm"], "best_params": lgbm_best}, lgbm_path, compress=3
    )
    joblib.dump(
        {"model": estimators["xgb"], "best_params": xgb_best}, xgb_path, compress=3
    )
    ensemble.save(ensemble_path)

    for p in [rf_path, lgbm_path, xgb_path, ensemble_path]:
        sz = os.path.getsize(p) / 1024 / 1024
        print(f"  ✓ {p} ({sz:.1f} MB)")

    # ── Step 11: Fold 2 evaluation summary ────────────────────────────────
    separator("=== Fold 2 Evaluation Results ===")

    rf_pred = np.array(estimators["rf"].predict(X_fold2))
    lgbm_pred = estimators["lgbm"].predict(X_fold2)
    xgb_pred = estimators["xgb"].predict(X_fold2)
    ens_pred = ensemble.predict(X_fold2)

    rf_f1   = f1_score(y_fold2, rf_pred,   average="macro", zero_division=0)
    lgbm_f1 = f1_score(y_fold2, lgbm_pred, average="macro", zero_division=0)
    xgb_f1  = f1_score(y_fold2, xgb_pred,  average="macro", zero_division=0)
    ens_f1  = f1_score(y_fold2, ens_pred,  average="macro", zero_division=0)

    w = best_weights
    est_keys = list(estimators.keys())
    print(f"  Random Forest   Macro-F1: {rf_f1:.4f}")
    print(f"  LightGBM        Macro-F1: {lgbm_f1:.4f}")
    print(f"  XGBoost         Macro-F1: {xgb_f1:.4f}")
    print(
        f"  Ensemble        Macro-F1: {ens_f1:.4f} "
        f"(weights: RF={w.get('rf', 0):.2f}, "
        f"LGB={w.get('lgbm', 0):.2f}, "
        f"XGB={w.get('xgb', 0):.2f})"
    )

    t_elapsed = time.time() - t_start
    print(f"\n  Peak RSS: {rss_mb():.1f} MB | Elapsed: {t_elapsed:.1f}s")
    separator("Phase 4-01 Complete")


if __name__ == "__main__":
    main()
