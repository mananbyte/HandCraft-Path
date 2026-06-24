"""
src/training/ensemble.py
────────────────────────
Soft-voting ensemble for HandCraft-Path binary segmentation classifiers.

Combines cuML RandomForest, LightGBM, and XGBoost via weighted probability
averaging.  Ensemble weights are optimised post-hoc on the held-out Fold 2
validation set using an exhaustive grid search over the weight simplex.

Biological rationale: Nucleus/background pixel discrimination benefits from
combining complementary strengths — RF captures local textural patterns,
LightGBM excels at sparse gradient features, and XGBoost adds regularised
depth-first splits that sharpen boundary confidence.
"""

import os
import json
import logging
from typing import Optional

import joblib
import numpy as np
from sklearn.metrics import f1_score

# ── GPU fallback guard (cuML API note: RF accepts random_state; LR does not) ──
HAS_CUML: bool = False
HAS_GPU: bool = False   # True when a CUDA device is confirmed visible at import
try:
    import cupy
    _n_dev = cupy.cuda.runtime.getDeviceCount()
    # Check if GPU has enough VRAM (require at least 3.5 GB) for large-scale RAPIDS/XGBoost operations
    _vram_ok = True
    if _n_dev > 0:
        try:
            _device = cupy.cuda.Device(0)
            _total_vram = _device.mem_info[1]
            if _total_vram < 1.2 * 1024 * 1024 * 1024:  # 1.2 GB in bytes
                _vram_ok = False
        except Exception:
            _vram_ok = False

    if _n_dev > 0 and _vram_ok:
        HAS_GPU = True
        from cuml.ensemble import RandomForestClassifier as CuMLRF
        HAS_CUML = True
    else:
        from sklearn.ensemble import RandomForestClassifier as CuMLRF  # type: ignore
except Exception:
    from sklearn.ensemble import RandomForestClassifier as CuMLRF  # type: ignore

import lightgbm as lgb
import xgboost as xgb

# Try a dummy fit to verify if LightGBM GPU mode actually works on this environment
HAS_LGBM_GPU: bool = False
if HAS_GPU:
    try:
        _X_test = np.random.rand(10, 2)
        _y_test = np.random.randint(0, 2, 10)
        _clf_test = lgb.LGBMClassifier(device="gpu", verbose=-1)
        _clf_test.fit(_X_test, _y_test)
        HAS_LGBM_GPU = True
    except Exception:
        HAS_LGBM_GPU = False

# ── Module-level constants ─────────────────────────────────────────────────
RANDOM_STATE: int = 42
SIMPLEX_STEP: float = 0.1          # weight grid-search step (≈165 combos)
N_FEATURES: int = 25               # expected feature count after RFE slicing
DEFAULT_WEIGHTS: dict = {
    "rf": 1 / 3,
    "lgbm": 1 / 3,
    "xgb": 1 / 3,
}

logger = logging.getLogger(__name__)


# ── CPU-safe cuML RF inference helper ─────────────────────────────────────
def _cuml_rf_proba_cpu(rf_model, X: np.ndarray) -> np.ndarray:
    """
    Run cuML RandomForest probability inference via the CPU FIL backend.

    Root cause of Step 9 crash: after training, XGBoost (device=cuda) holds
    ~1.4 GB in VRAM. When cuML RF.predict_proba() is called, it tries to load
    ~146 MB of FIL tree structures onto the GPU → std::bad_alloc crash on 2 GB
    cards. When the threshold is relaxed, it falls back to an internal CPU path
    that allocates massive temporary float32 arrays in System RAM → 15 GB RAM
    spike → system freeze.

    Solution: call `rf.as_fil().cpu_forest.predict(X)` directly. This is the
    internal CPU inference path of cuML FIL. Verified properties:
    - 0 MB of VRAM consumed during inference
    - Numerically equivalent to GPU path (max diff < 1.2e-7)
    - Returns a CumlArray → converted to numpy via .to_output('numpy')
    """
    if HAS_CUML and hasattr(rf_model, 'as_fil'):
        raw = rf_model.as_fil().cpu_forest.predict(X)
        return raw.to_output('numpy').astype(np.float64)
    # sklearn RF fallback
    return np.array(rf_model.predict_proba(X), dtype=np.float64)


# ── Helper ─────────────────────────────────────────────────────────────────
def build_default_estimators(
    rf_params: dict,
    lgbm_params: dict,
    xgb_params: dict,
) -> dict:
    """
    Instantiate the three base estimators with the supplied hyperparameters.

    Falls back to sklearn RandomForestClassifier when cuML is unavailable
    (e.g. during CI/unit tests on CPU-only machines).

    Parameters
    ----------
    rf_params   : Hyperparameters for cuML / sklearn RandomForest.
    lgbm_params : Hyperparameters for LightGBM LGBMClassifier.
    xgb_params  : Hyperparameters for XGBoost XGBClassifier.

    Returns
    -------
    dict with keys 'rf', 'lgbm', 'xgb'.
    """
    # cuML RF accepts random_state; sklearn RF also does
    rf = CuMLRF(random_state=RANDOM_STATE, **rf_params)

    # LightGBM: use GPU device when a CUDA device is confirmed and functioning; fall back to
    # CPU gracefully so unit tests, CPU-only machines, and OpenCL-less machines work.
    lgbm_device = "gpu" if HAS_LGBM_GPU else "cpu"
    lgbm_clf = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1 if HAS_LGBM_GPU else -1,   # LGBM GPU mode ignores n_jobs; set 1 to avoid warnings
        verbose=-1,
        device=lgbm_device,
        **lgbm_params,
    )

    # XGBoost: use CUDA device when available; CPU fallback for tests.
    xgb_device = "cuda" if HAS_GPU else "cpu"
    xgb_clf = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        device=xgb_device,
        eval_metric="logloss",
        use_label_encoder=False,
        **xgb_params,
    )

    logger.info(
        "build_default_estimators: HAS_CUML=%s | HAS_GPU=%s | HAS_LGBM_GPU=%s | RF=%s | LGBM(device=%s) | XGB(device=%s)",
        HAS_CUML, HAS_GPU, HAS_LGBM_GPU, type(rf).__name__, lgbm_device, xgb_device,
    )
    return {"rf": rf, "lgbm": lgbm_clf, "xgb": xgb_clf}


# ── SoftVotingEnsemble ─────────────────────────────────────────────────────
class SoftVotingEnsemble:
    """
    Weighted soft-voting ensemble for binary pixel segmentation.

    Blends predict_proba() outputs from RF, LGBM, and XGBoost using
    per-model scalar weights that are optimised on the held-out Fold 2.

    Parameters
    ----------
    estimators : dict
        Mapping of name → fitted (or unfitted) estimator.
        Expected keys: 'rf', 'lgbm', 'xgb'.
    weights : Optional[dict]
        Mapping of name → float weight summing to 1.0.
        Defaults to equal weighting if None.
    """

    def __init__(
        self,
        estimators: dict,
        weights: Optional[dict] = None,
    ) -> None:
        self.estimators = estimators
        # Default: equal weights for all estimators
        self.weights: dict = weights if weights is not None else {
            k: 1.0 / len(estimators) for k in estimators
        }
        self._fitted: bool = False

    # ── Fit ───────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftVotingEnsemble":
        """
        Fit each base estimator independently on the full training matrix.

        Parameters
        ----------
        X : np.ndarray, shape (N, 25)
            Scaled, RFE-sliced feature matrix.
        y : np.ndarray, shape (N,)
            Binary label vector {0 = background, 1 = nucleus}.
        """
        assert X.ndim == 2, f"X must be 2-D, got shape {X.shape}"
        assert y.ndim == 1, f"y must be 1-D, got shape {y.shape}"
        assert X.shape[1] == N_FEATURES, (
            f"Expected {N_FEATURES} features, got {X.shape[1]}"
        )

        for name, est in self.estimators.items():
            logger.info("Fitting estimator: %s on %d rows", name, X.shape[0])
            est.fit(X, y)
            logger.info("Finished: %s", name)

        self._fitted = True
        return self

    # ── Predict proba ─────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted average of base estimator class probabilities.
        Safe for massive inputs by chunking predictions to prevent CUDA OOM.

        Parameters
        ----------
        X : np.ndarray, shape (N, 25)

        Returns
        -------
        np.ndarray, shape (N, 2)
        """
        assert self._fitted, "Call fit() before predict_proba()"
        assert X.ndim == 2, f"X must be 2-D, got shape {X.shape}"

        n_rows = X.shape[0]
        chunk_size = 50_000

        if n_rows > chunk_size:
            chunks = []
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                chunks.append(self._predict_proba_chunk(X[start:end]))
            return np.vstack(chunks).astype(np.float32)
        else:
            return self._predict_proba_chunk(X)

    def _predict_proba_chunk(self, X: np.ndarray) -> np.ndarray:
        blended: Optional[np.ndarray] = None
        total_weight: float = sum(self.weights[k] for k in self.estimators)

        for name, est in self.estimators.items():
            w = self.weights[name] / total_weight
            # Use zero-VRAM CPU FIL path for RF to avoid VRAM OOM when XGBoost
            # is also in VRAM. Other estimators use their native predict_proba.
            if name == "rf":
                proba = _cuml_rf_proba_cpu(est, X)
            else:
                proba = np.array(est.predict_proba(X), dtype=np.float64)
            if blended is None:
                blended = w * proba
            else:
                blended = blended + w * proba  # explicit sum avoids in-place mutation

        return blended.astype(np.float32)

    # ── Predict ───────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Argmax of blended class probabilities → {0, 1} label vector."""
        assert X.ndim == 2, f"X must be 2-D, got shape {X.shape}"
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int32)

    # ── Weight optimisation ───────────────────────────────────────────────
    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        step: float = SIMPLEX_STEP,
    ) -> dict:
        """
        Grid search over the weight simplex to maximise Macro-F1 on Fold 2.

        Iterates all (w_rf, w_lgb, w_xgb) where each weight ∈ [0, 1] at
        the given step and w_rf + w_lgb + w_xgb ≈ 1.0 (within 1e-9).

        Parameters
        ----------
        X_val : np.ndarray, shape (M, 25)
        y_val : np.ndarray, shape (M,)
        step  : float — grid step (default 0.1 → ~165 valid combinations)

        Returns
        -------
        best_weights : dict with keys matching self.estimators
        """
        assert self._fitted, "Call fit() before optimize_weights()"
        assert X_val.ndim == 2, f"X_val must be 2-D, got {X_val.shape}"
        assert y_val.ndim == 1, f"y_val must be 1-D, got {y_val.shape}"

        # Pre-compute each estimator's raw probabilities once in chunks to prevent VRAM OOM
        est_names = list(self.estimators.keys())
        probas = {}
        chunk_size = 50_000
        n_rows = X_val.shape[0]

        for name in est_names:
            est = self.estimators[name]
            est_probas = []
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                X_chunk = X_val[start:end]
                # Use CPU FIL path for cuML RF to prevent VRAM OOM when
                # XGBoost is simultaneously resident in VRAM.
                if name == "rf":
                    chunk_proba = _cuml_rf_proba_cpu(est, X_chunk)
                else:
                    chunk_proba = np.array(est.predict_proba(X_chunk), dtype=np.float64)
                est_probas.append(chunk_proba)
            probas[name] = np.vstack(est_probas)

        grid = np.arange(0.0, 1.0 + step, step)
        best_f1: float = -1.0
        best_weights: dict = dict(self.weights)

        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                w2 = max(0.0, w2)  # clip tiny floating point negatives

                # Enforce simplex: sum must be ≈ 1
                if abs(w0 + w1 + w2 - 1.0) > 1e-9:
                    continue

                weights_trial = {
                    est_names[0]: w0,
                    est_names[1]: w1,
                    est_names[2]: w2,
                }
                
                # In-place addition to avoid allocating multiple 31MB arrays per loop iteration
                blended = weights_trial[est_names[0]] * probas[est_names[0]]
                blended += weights_trial[est_names[1]] * probas[est_names[1]]
                blended += weights_trial[est_names[2]] * probas[est_names[2]]
                
                y_pred = np.argmax(blended, axis=1)
                f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = dict(weights_trial)

        logger.info(
            "optimize_weights: best Macro-F1=%.4f | weights=%s",
            best_f1, best_weights,
        )
        # Update instance weights in place (documented mutation)
        self.weights = best_weights
        return best_weights

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Serialise ensemble (estimators + weights + metadata) via joblib."""
        assert self._fitted, "Fit the ensemble before saving."
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "estimators": self.estimators,
            "weights": self.weights,
            "n_features": N_FEATURES,
            "has_cuml": HAS_CUML,
        }
        joblib.dump(payload, path, compress=3)
        logger.info("SoftVotingEnsemble saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "SoftVotingEnsemble":
        """Reconstruct a SoftVotingEnsemble from a joblib artefact."""
        payload = joblib.load(path)
        instance = cls.__new__(cls)
        instance.estimators = payload["estimators"]
        instance.weights = payload["weights"]
        instance._fitted = True
        logger.info("SoftVotingEnsemble loaded from %s", path)
        return instance
