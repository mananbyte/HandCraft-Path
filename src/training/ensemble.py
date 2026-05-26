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
try:
    from cuml.ensemble import RandomForestClassifier as CuMLRF
    HAS_CUML: bool = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as CuMLRF  # type: ignore
    HAS_CUML: bool = False

import lightgbm as lgb
import xgboost as xgb

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

    lgbm_clf = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        **lgbm_params,
    )

    xgb_clf = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False,
        **xgb_params,
    )

    logger.info(
        "build_default_estimators: HAS_CUML=%s | RF=%s | LGBM=%s | XGB=%s",
        HAS_CUML, type(rf).__name__,
        type(lgbm_clf).__name__,
        type(xgb_clf).__name__,
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

        Parameters
        ----------
        X : np.ndarray, shape (N, 25)

        Returns
        -------
        np.ndarray, shape (N, 2)
        """
        assert self._fitted, "Call fit() before predict_proba()"
        assert X.ndim == 2, f"X must be 2-D, got shape {X.shape}"

        blended: Optional[np.ndarray] = None
        total_weight: float = sum(self.weights[k] for k in self.estimators)

        for name, est in self.estimators.items():
            w = self.weights[name] / total_weight
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

        # Pre-compute each estimator's raw probabilities once
        est_names = list(self.estimators.keys())
        probas = {
            name: np.array(self.estimators[name].predict_proba(X_val), dtype=np.float64)
            for name in est_names
        }

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
                blended = sum(
                    weights_trial[n] * probas[n] for n in est_names
                )
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
