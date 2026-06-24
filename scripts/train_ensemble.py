#!/usr/bin/env python3
"""
scripts/train_ensemble.py
─────────────────────────
Train, tune (Optuna HPO), and ensemble binary pixel classifiers.

Memory-safe two-pass architecture:
  PASS 1 (RF only):   Fold1 + RF in RAM → stream Fold2 from disk for RF probas → delete RF
  PASS 2 (LGBM/XGB): Free Fold1 → load Fold2 fully → fit LGBM → fit XGB → probas → delete
  PASS 3 (weights):   Load 3×31 MB .npy proba files → grid search → save ensemble

Peak RAM: ~7.5 GB (RF pass) → ~3 GB (LGBM/XGB pass) → <2 GB (weight search)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import rmm
rmm.reinitialize(managed_memory=True)

import joblib
import numpy as np
import optuna
import psutil
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.ensemble import (
    SoftVotingEnsemble,
    build_default_estimators,
    HAS_GPU,
    HAS_LGBM_GPU,
    _cuml_rf_proba_cpu,
)
from src.training.scaler import FeatureScaler
from src.utils.safe_loader import safe_load_npy

# ── Constants ──────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
FOLD2_CHUNK_SIZE: int = 50_000
PROCESSED_DIR: str = "data/processed"
MODELS_DIR: str = "data/models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("train_ensemble")


# ── Helpers ────────────────────────────────────────────────────────────────
def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def separator(label: str) -> None:
    width = 72
    print(f"\n{'═' * width}")
    print(f"  {label}")
    print(f"{'═' * width}")


def print_rss(stage: str) -> None:
    logger.info("[RSS] %s: %.1f MB", stage, rss_mb())


def _free_gpu() -> None:
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train, tune, and ensemble binary pixel classifiers (Phase 4)."
    )
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--selected-features", type=str,
                   default="data/models/selected_features_rfe_25.json")
    p.add_argument("--out-dir", type=str, default="data/models/")
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--hpo-sample-n", type=int, default=200_000)
    p.add_argument("--optuna-timeout", type=int, default=1800)
    p.add_argument("--skip-hpo", action="store_true",
                   help="Load pre-existing HPO params instead of running Optuna.")
    return p.parse_args()


# ── Optuna objectives ──────────────────────────────────────────────────────
def _rf_objective(trial, X_tr, y_tr, X_vl, y_vl) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 8, 30),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }
    from src.training.ensemble import CuMLRF
    rf = CuMLRF(random_state=RANDOM_STATE, **params)
    rf.fit(X_tr, y_tr)
    return float(f1_score(y_vl, np.array(rf.predict(X_vl)), average="macro", zero_division=0))


def _lgbm_objective(trial, X_tr, y_tr, X_vl, y_vl) -> float:
    import lightgbm as lgb
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
    }
    clf = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1 if HAS_LGBM_GPU else -1,
                              verbose=-1, device="gpu" if HAS_LGBM_GPU else "cpu", **params)
    clf.fit(X_tr, y_tr)
    return float(f1_score(y_vl, clf.predict(X_vl), average="macro", zero_division=0))


def _xgb_objective(trial, X_tr, y_tr, X_vl, y_vl) -> float:
    import xgboost as xgb
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    clf = xgb.XGBClassifier(random_state=RANDOM_STATE, device="cuda" if HAS_GPU else "cpu",
                             eval_metric="logloss", use_label_encoder=False, **params)
    clf.fit(X_tr, y_tr)
    return float(f1_score(y_vl, clf.predict(X_vl), average="macro", zero_division=0))


# ── Subsample helper ───────────────────────────────────────────────────────
def draw_subsample(X_mm, y_mm, n: int, random_state: int = RANDOM_STATE):
    n_total = X_mm.shape[0]
    actual_n = min(n, n_total)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=actual_n / n_total,
                                 random_state=random_state)
    _, idx = next(sss.split(np.zeros(n_total), y_mm))
    idx = np.sort(idx)
    X_sub = np.nan_to_num(np.array(X_mm[idx], dtype=np.float32), nan=0.0)
    y_sub = np.array(y_mm[idx], dtype=np.int32)
    return X_sub, y_sub


# ── Scale Fold 2 in-memory ─────────────────────────────────────────────────
def scale_fold2_inMemory(X2_path: str, scaler: FeatureScaler, rfe_indices: list) -> np.ndarray:
    """Stream Fold 2 through scaler in 50K chunks, return (n_rows, 25) float32."""
    X_mm = safe_load_npy(X2_path, mode="r")
    n_total, _ = X_mm.shape
    n_chunks = (n_total + FOLD2_CHUNK_SIZE - 1) // FOLD2_CHUNK_SIZE
    X_out = np.empty((n_total, len(rfe_indices)), dtype=np.float32)
    print_rss("scale_fold2_inMemory start")
    for i in tqdm(range(n_chunks), desc="  Scaling Fold-2", unit="chunk"):
        s = i * FOLD2_CHUNK_SIZE
        e = min(s + FOLD2_CHUNK_SIZE, n_total)
        chunk = np.nan_to_num(np.array(X_mm[s:e], dtype=np.float32), nan=0.0)
        X_out[s:e] = scaler._scaler.transform(chunk).astype(np.float32)[:, rfe_indices]
    print_rss("scale_fold2_inMemory done")
    del X_mm
    gc.collect()
    return X_out


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    separator(f"Phase 4-01: Binary Ensemble Training | fold={args.fold}")
    print_rss("start")

    # ── Step 1: Load RFE indices ───────────────────────────────────────────
    separator("Step 1: Load RFE feature indices")
    with open(args.selected_features) as fh:
        feat_json = json.load(fh)
    rfe_indices: list = feat_json["indices"]
    rfe_names: list = feat_json["names"]
    assert len(rfe_indices) == 25, f"Expected 25 RFE indices, got {len(rfe_indices)}"
    print(f"  RFE features ({len(rfe_indices)}): {rfe_names[:5]}…")

    # ── Step 2: Load Fold 1 ────────────────────────────────────────────────
    separator("Step 2: Load Fold 1 training matrix (memmap → slice to 25 cols)")
    fold_tag = f"fold{args.fold}"
    X_path = os.path.join(PROCESSED_DIR, f"{fold_tag}_binary_X_scaled.npy")
    y_path = os.path.join(PROCESSED_DIR, f"{fold_tag}_binary_y.npy")
    assert os.path.exists(X_path), f"Missing: {X_path}"
    assert os.path.exists(y_path), f"Missing: {y_path}"

    X_mm = safe_load_npy(X_path, mode="r")
    y_mm = safe_load_npy(y_path, mode="r")
    print(f"  Fold 1 shape: X={X_mm.shape}, y={y_mm.shape}")
    print_rss("after memmap open")

    print("  Slicing to 25 RFE columns…")
    idx_rfe = np.array(rfe_indices)
    X_fold1 = np.nan_to_num(np.array(X_mm[:, idx_rfe], dtype=np.float32), nan=0.0)
    y_fold1 = np.array(y_mm, dtype=np.int32)
    print(f"  Fold 1 training matrix: {X_fold1.shape}")
    print_rss("after fold1 slice")

    # ── HPO or load pre-existing params ───────────────────────────────────
    rf_params_path   = os.path.join(args.out_dir, "rf_best_params.json")
    lgbm_params_path = os.path.join(args.out_dir, "lgbm_best_params.json")
    xgb_params_path  = os.path.join(args.out_dir, "xgb_best_params.json")

    has_pre = (os.path.exists(rf_params_path) and os.path.exists(lgbm_params_path)
               and os.path.exists(xgb_params_path))

    if args.skip_hpo and has_pre:
        separator("Bypassing HPO: Loading pre-existing best parameters")
        with open(rf_params_path)   as fh: rf_best   = json.load(fh)
        with open(lgbm_params_path) as fh: lgbm_best = json.load(fh)
        with open(xgb_params_path)  as fh: xgb_best  = json.load(fh)
        print(f"  ✓ Loaded RF best params: {rf_best}")
        print(f"  ✓ Loaded LGBM best params: {lgbm_best}")
        print(f"  ✓ Loaded XGB best params: {xgb_best}")
    else:
        separator(f"Step 3: Stratified subsample for HPO (n={args.hpo_sample_n:,})")
        X_sub, y_sub = draw_subsample(X_fold1, y_fold1, n=args.hpo_sample_n)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        tr_idx, vl_idx = next(sss.split(np.zeros(len(y_sub)), y_sub))
        X_hpo_tr, X_hpo_vl = X_sub[tr_idx], X_sub[vl_idx]
        y_hpo_tr, y_hpo_vl = y_sub[tr_idx], y_sub[vl_idx]
        print_rss("after hpo subsample")

        separator(f"Step 4: RF Optuna study ({args.n_trials} trials)")
        rf_study = optuna.create_study(direction="maximize", study_name="rf_study")
        rf_study.optimize(
            lambda trial: _rf_objective(trial, X_hpo_tr, y_hpo_tr, X_hpo_vl, y_hpo_vl),
            n_trials=args.n_trials, timeout=args.optuna_timeout, show_progress_bar=True,
        )
        rf_best = rf_study.best_params
        with open(rf_params_path, "w") as fh: json.dump(rf_best, fh, indent=2)
        print(f"  RF best: {rf_best}  F1={rf_study.best_value:.4f}")

        separator(f"Step 5: LGBM Optuna study ({args.n_trials} trials)")
        lgbm_study = optuna.create_study(direction="maximize", study_name="lgbm_study")
        lgbm_study.optimize(
            lambda trial: _lgbm_objective(trial, X_hpo_tr, y_hpo_tr, X_hpo_vl, y_hpo_vl),
            n_trials=args.n_trials, timeout=args.optuna_timeout, show_progress_bar=True,
        )
        lgbm_best = lgbm_study.best_params
        with open(lgbm_params_path, "w") as fh: json.dump(lgbm_best, fh, indent=2)
        print(f"  LGBM best: {lgbm_best}  F1={lgbm_study.best_value:.4f}")

        separator(f"Step 6: XGB Optuna study ({args.n_trials} trials)")
        xgb_study = optuna.create_study(direction="maximize", study_name="xgb_study")
        xgb_study.optimize(
            lambda trial: _xgb_objective(trial, X_hpo_tr, y_hpo_tr, X_hpo_vl, y_hpo_vl),
            n_trials=args.n_trials, timeout=args.optuna_timeout, show_progress_bar=True,
        )
        xgb_best = xgb_study.best_params
        with open(xgb_params_path, "w") as fh: json.dump(xgb_best, fh, indent=2)
        print(f"  XGB best: {xgb_best}  F1={xgb_study.best_value:.4f}")

    # ── Output paths ───────────────────────────────────────────────────────
    rf_path         = os.path.join(args.out_dir, "cuml_rf.joblib")
    lgbm_path       = os.path.join(args.out_dir, "lgbm.joblib")
    xgb_path        = os.path.join(args.out_dir, "xgboost.joblib")
    ensemble_path   = os.path.join(args.out_dir, "binary_ensemble.joblib")
    rf_proba_path   = os.path.join(args.out_dir, "rf_fold2_probas.npy")
    lgbm_proba_path = os.path.join(args.out_dir, "lgbm_fold2_probas.npy")
    xgb_proba_path  = os.path.join(args.out_dir, "xgb_fold2_probas.npy")

    scaler_path = os.path.join(MODELS_DIR, f"scaler_fold{args.fold}_binary.joblib")
    X2_path = os.path.join(PROCESSED_DIR, "fold2_binary_X.npy")
    y2_path = os.path.join(PROCESSED_DIR, "fold2_binary_y.npy")
    assert os.path.exists(scaler_path), f"Missing scaler: {scaler_path}"
    assert os.path.exists(X2_path),     f"Missing: {X2_path}"
    assert os.path.exists(y2_path),     f"Missing: {y2_path}"

    chunk_size = 50_000

    # ══════════════════════════════════════════════════════════════════════
    # PASS 1 — RF only (Fold 2 is NEVER loaded into RAM here)
    # Fold2 is streamed chunk-by-chunk from raw disk through the scaler.
    # Peak RAM: Fold1 (1.6 GB) + RF (~7 GB) = ~8.7 GB  (no Fold2 = -177 MB)
    # ══════════════════════════════════════════════════════════════════════
    separator("Step 7a: Fit cuML RF alone (no Fold 2 in RAM)")
    single_est = build_default_estimators(rf_best, {"n_estimators": 1}, {"n_estimators": 1})
    rf_est = single_est["rf"]
    print(f"  Fitting rf on {X_fold1.shape[0]:,} rows…")
    rf_est.fit(X_fold1, y_fold1)
    print_rss("after rf fit")

    joblib.dump({"model": rf_est, "best_params": rf_best}, rf_path, compress=3)
    print(f"  ✓ Saved rf → {rf_path} ({os.path.getsize(rf_path)/1024/1024:.1f} MB)")

    # Stream Fold-2 from raw disk through scaler — never allocate the full scaled matrix
    print("  Computing RF Fold-2 probas (streaming raw Fold-2 from disk)…")
    scaler_rf = FeatureScaler.load(scaler_path)
    X2_mm = safe_load_npy(X2_path, mode="r")
    n_fold2 = X2_mm.shape[0]
    rf_probas_out = np.empty((n_fold2, 2), dtype=np.float64)
    for start in tqdm(range(0, n_fold2, chunk_size), desc="  RF proba chunks", unit="chunk"):
        end = min(start + chunk_size, n_fold2)
        raw = np.nan_to_num(np.array(X2_mm[start:end], dtype=np.float32), nan=0.0)
        scaled = scaler_rf._scaler.transform(raw).astype(np.float32)[:, idx_rfe]
        rf_probas_out[start:end] = _cuml_rf_proba_cpu(rf_est, scaled)
    np.save(rf_proba_path, rf_probas_out)
    print(f"  ✓ Saved RF probas → {rf_proba_path} ({os.path.getsize(rf_proba_path)/1024/1024:.1f} MB)")

    del rf_est, single_est, scaler_rf, X2_mm, rf_probas_out
    gc.collect()
    _free_gpu()
    print_rss("after rf deleted from RAM")

    # ══════════════════════════════════════════════════════════════════════
    # PASS 2 — Free Fold 1, load Fold 2 fully, fit LGBM then XGB
    # Peak RAM: Fold2 (177 MB) + one model (~200 MB for LGBM/XGB) = ~2-3 GB
    # NOTE: LGBM and XGB are fitted on Fold 1 (reloaded briefly).
    # ══════════════════════════════════════════════════════════════════════
    print("  [Memory] Freeing Fold 1 data…")
    del X_fold1, y_fold1, X_mm, y_mm
    gc.collect()
    print_rss("after freeing fold1 data")

    separator("Step 8: Scale Fold 2 in-memory via Phase-3 scaler")
    scaler = FeatureScaler.load(scaler_path)
    X_fold2 = scale_fold2_inMemory(X2_path, scaler, rfe_indices)
    y_fold2 = np.array(safe_load_npy(y2_path, mode="r"), dtype=np.int32)
    del scaler
    print(f"  Fold 2: {X_fold2.shape}, y: {y_fold2.shape}")
    print_rss("after fold2 scale")

    # LGBM and XGB are fitted on Fold 1 data — reload it momentarily
    # Fold 1 scaled file is a memmap; slicing it costs ~196 MB extra
    for name, best_params, model_path, proba_path in [
        ("lgbm", lgbm_best, lgbm_path, lgbm_proba_path),
        ("xgb",  xgb_best,  xgb_path,  xgb_proba_path),
    ]:
        separator(f"  Fitting {name} on Fold 1 (brief reload)")
        X_mm2   = safe_load_npy(X_path, mode="r")
        y_mm2   = safe_load_npy(y_path, mode="r")
        X_f1_tmp = np.nan_to_num(np.array(X_mm2[:, idx_rfe], dtype=np.float32), nan=0.0)
        y_f1_tmp = np.array(y_mm2, dtype=np.int32)
        del X_mm2, y_mm2

        single_est2 = build_default_estimators(
            {"n_estimators": 1},
            lgbm_best if name == "lgbm" else {"n_estimators": 1},
            xgb_best  if name == "xgb"  else {"n_estimators": 1},
        )
        est = single_est2[name]
        print(f"  Fitting {name} on {X_f1_tmp.shape[0]:,} rows…")
        est.fit(X_f1_tmp, y_f1_tmp)
        del X_f1_tmp, y_f1_tmp, single_est2
        gc.collect()
        print_rss(f"after {name} fit")

        joblib.dump({"model": est, "best_params": best_params}, model_path, compress=3)
        print(f"  ✓ Saved {name} → {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)")

        chunks = []
        for start in range(0, X_fold2.shape[0], chunk_size):
            end = min(start + chunk_size, X_fold2.shape[0])
            chunks.append(np.array(est.predict_proba(X_fold2[start:end]), dtype=np.float64))
        probas = np.vstack(chunks)
        np.save(proba_path, probas)
        print(f"  ✓ Saved {name} probas → {proba_path} ({os.path.getsize(proba_path)/1024/1024:.1f} MB)")

        del est, chunks, probas
        gc.collect()
        _free_gpu()
        print_rss(f"after {name} deleted from RAM")

    # ══════════════════════════════════════════════════════════════════════
    # PASS 3 — Grid search weights using 3 × 31 MB proba .npy files only
    # Peak RAM: Fold2 (177 MB) + 3 proba arrays (94 MB) = ~2 GB
    # ══════════════════════════════════════════════════════════════════════
    separator("Step 9: Optimise ensemble weights via weight simplex grid search")
    probas_dict = {
        "rf":   np.load(rf_proba_path),
        "lgbm": np.load(lgbm_proba_path),
        "xgb":  np.load(xgb_proba_path),
    }
    print_rss("after loading all 3 proba arrays")

    from sklearn.metrics import f1_score as _f1
    grid = np.arange(0.0, 1.0 + 0.1, 0.1)
    best_f1: float = -1.0
    est_order = ["rf", "lgbm", "xgb"]
    best_weights: dict = {n: 1 / 3 for n in est_order}

    for w0 in grid:
        for w1 in grid:
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9:
                continue
            w2 = max(0.0, w2)
            if abs(w0 + w1 + w2 - 1.0) > 1e-9:
                continue
            blended = (w0 * probas_dict["rf"]
                       + w1 * probas_dict["lgbm"]
                       + w2 * probas_dict["xgb"])
            f1 = _f1(y_fold2, np.argmax(blended, axis=1), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = {"rf": w0, "lgbm": w1, "xgb": w2}

    print(f"  Optimal weights: {best_weights}  | best Macro-F1={best_f1:.4f}")
    print_rss("after weight optimisation")

    # ── Step 10: Save ensemble ─────────────────────────────────────────────
    separator("Step 10: Save ensemble with optimal weights")
    estimators_reload = {
        "rf":   joblib.load(rf_path)["model"],
        "lgbm": joblib.load(lgbm_path)["model"],
        "xgb":  joblib.load(xgb_path)["model"],
    }
    ensemble = SoftVotingEnsemble(estimators=estimators_reload, weights=best_weights)
    ensemble._fitted = True
    ensemble.save(ensemble_path)
    print(f"  ✓ Saved ensemble → {ensemble_path} ({os.path.getsize(ensemble_path)/1024/1024:.1f} MB)")
    print_rss("after saving ensemble")

    # ── Step 11: Fold 2 evaluation ─────────────────────────────────────────
    separator("=== Fold 2 Evaluation Results ===")
    w0, w1, w2 = best_weights["rf"], best_weights["lgbm"], best_weights["xgb"]
    rf_pred   = np.argmax(probas_dict["rf"],   axis=1).astype(np.int32)
    lgbm_pred = np.argmax(probas_dict["lgbm"], axis=1).astype(np.int32)
    xgb_pred  = np.argmax(probas_dict["xgb"],  axis=1).astype(np.int32)
    blended_ens = (w0 * probas_dict["rf"]
                   + w1 * probas_dict["lgbm"]
                   + w2 * probas_dict["xgb"])
    ens_pred = np.argmax(blended_ens, axis=1).astype(np.int32)

    rf_f1   = _f1(y_fold2, rf_pred,   average="macro", zero_division=0)
    lgbm_f1 = _f1(y_fold2, lgbm_pred, average="macro", zero_division=0)
    xgb_f1  = _f1(y_fold2, xgb_pred,  average="macro", zero_division=0)
    ens_f1  = _f1(y_fold2, ens_pred,  average="macro", zero_division=0)

    print(f"  Random Forest   Macro-F1: {rf_f1:.4f}")
    print(f"  LightGBM        Macro-F1: {lgbm_f1:.4f}")
    print(f"  XGBoost         Macro-F1: {xgb_f1:.4f}")
    print(
        f"  Ensemble        Macro-F1: {ens_f1:.4f} "
        f"(weights: RF={w0:.2f}, LGB={w1:.2f}, XGB={w2:.2f})"
    )

    t_elapsed = time.time() - t_start
    print(f"\n  Peak RSS: {rss_mb():.1f} MB | Elapsed: {t_elapsed:.1f}s")
    separator("Phase 4-01 Complete")


if __name__ == "__main__":
    main()
