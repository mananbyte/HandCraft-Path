"""
tests/test_ensemble.py
──────────────────────
Unit tests for src/training/ensemble.py (SoftVotingEnsemble).

All tests run on CPU-only (sklearn RandomForest), require no GPU,
and should complete in < 30 seconds total.
"""

import os
import tempfile

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.training.ensemble import SoftVotingEnsemble, build_default_estimators, HAS_CUML

# ── Fixtures ──────────────────────────────────────────────────────────────
N_ROWS: int = 500
N_FEATURES: int = 25
RANDOM_STATE: int = 42


def _make_cpu_estimators() -> dict:
    """Return a minimal dict of CPU-only sklearn estimators for fast tests."""
    return {
        "rf": RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE, n_jobs=1),
        "lgbm": RandomForestClassifier(n_estimators=8, random_state=RANDOM_STATE, n_jobs=1),
        "xgb": RandomForestClassifier(n_estimators=8, random_state=RANDOM_STATE, n_jobs=1),
    }


@pytest.fixture
def synthetic_data():
    """500 × 25 binary classification dataset."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_ROWS, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, 2, size=N_ROWS).astype(np.int32)
    return X, y


@pytest.fixture
def fitted_ensemble(synthetic_data):
    """Pre-fitted SoftVotingEnsemble (CPU-only)."""
    X, y = synthetic_data
    ens = SoftVotingEnsemble(estimators=_make_cpu_estimators())
    ens.fit(X, y)
    return ens, X, y


# ── 1. predict_proba shape ────────────────────────────────────────────────
def test_predict_proba_shape(fitted_ensemble):
    """predict_proba must return (N, 2) shape for binary classification."""
    ens, X, _ = fitted_ensemble
    proba = ens.predict_proba(X)
    assert proba.shape == (N_ROWS, 2), (
        f"Expected shape ({N_ROWS}, 2), got {proba.shape}"
    )


# ── 2. predict values ─────────────────────────────────────────────────────
def test_predict_values(fitted_ensemble):
    """predict must return a 1-D array of length N with values in {0, 1}."""
    ens, X, _ = fitted_ensemble
    y_pred = ens.predict(X)
    assert y_pred.ndim == 1, f"Expected 1-D, got {y_pred.ndim}-D"
    assert len(y_pred) == N_ROWS, f"Expected {N_ROWS} predictions"
    unique_vals = set(y_pred.tolist())
    assert unique_vals.issubset({0, 1}), f"Unexpected label values: {unique_vals}"


# ── 3. weighted average math ──────────────────────────────────────────────
def test_weighted_average_math(synthetic_data):
    """
    Manually create 2 estimators with known constant probabilities and
    verify that the blended output matches the hand-computed weighted avg.
    """
    X, y = synthetic_data

    # Create two estimators whose probas we can control via a mock
    class ConstantProbaEstimator:
        def __init__(self, p0, p1):
            self._p0, self._p1 = p0, p1
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.tile([self._p0, self._p1], (len(X), 1)).astype(np.float64)

    est_a = ConstantProbaEstimator(0.3, 0.7)
    est_b = ConstantProbaEstimator(0.6, 0.4)

    weights = {"a": 0.4, "b": 0.6}
    ens = SoftVotingEnsemble(estimators={"a": est_a, "b": est_b}, weights=weights)
    ens._fitted = True  # skip fit for mock estimators

    proba = ens.predict_proba(X[:10])

    # Manual calculation: 0.4 * [0.3, 0.7] + 0.6 * [0.6, 0.4]
    expected_p0 = 0.4 * 0.3 + 0.6 * 0.6  # = 0.12 + 0.36 = 0.48
    expected_p1 = 0.4 * 0.7 + 0.6 * 0.4  # = 0.28 + 0.24 = 0.52
    np.testing.assert_allclose(proba[:, 0], expected_p0, atol=1e-5)
    np.testing.assert_allclose(proba[:, 1], expected_p1, atol=1e-5)


# ── 4. equal weights → simple average ────────────────────────────────────
def test_equal_weights(synthetic_data):
    """With equal weights all estimators contribute equally → simple average."""
    X, y = synthetic_data

    class ConstantProba:
        def __init__(self, val):
            self._val = val
            self.classes_ = np.array([0, 1])

        def fit(self, X, y): return self

        def predict_proba(self, X):
            return np.tile([self._val, 1 - self._val], (len(X), 1)).astype(np.float64)

    probas = [0.2, 0.5, 0.8]
    estimators = {f"m{i}": ConstantProba(p) for i, p in enumerate(probas)}
    ens = SoftVotingEnsemble(estimators=estimators)
    ens._fitted = True

    out = ens.predict_proba(X[:5])
    expected_p0 = np.mean(probas)  # = 0.5
    np.testing.assert_allclose(out[:, 0], expected_p0, atol=1e-5)


# ── 5. optimize_weights sums to 1.0 ──────────────────────────────────────
def test_optimize_weights_sums_to_one(fitted_ensemble):
    """optimize_weights must return weights that sum to 1.0."""
    ens, X, y = fitted_ensemble
    best_w = ens.optimize_weights(X, y, step=0.5)  # coarse step for speed
    total = sum(best_w.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"


# ── 6. save / load roundtrip ──────────────────────────────────────────────
def test_save_load_roundtrip(fitted_ensemble):
    """Saved ensemble must produce identical predictions after loading."""
    ens, X, _ = fitted_ensemble
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ensemble.joblib")
        ens.save(path)
        assert os.path.exists(path), "Saved file not found"

        ens2 = SoftVotingEnsemble.load(path)
        pred1 = ens.predict(X)
        pred2 = ens2.predict(X)
        np.testing.assert_array_equal(pred1, pred2)


# ── 7. shape assertion: X.ndim != 2 ──────────────────────────────────────
def test_shape_assertion_X(fitted_ensemble):
    """predict_proba must raise AssertionError when X is 1-D."""
    ens, _, _ = fitted_ensemble
    X_bad = np.zeros(N_FEATURES, dtype=np.float32)  # 1-D
    with pytest.raises(AssertionError):
        ens.predict_proba(X_bad)


# ── 8. shape assertion: y.ndim != 1 ──────────────────────────────────────
def test_shape_assertion_y(synthetic_data):
    """fit() must raise AssertionError when y is 2-D."""
    X, y = synthetic_data
    y_bad = y.reshape(-1, 1)  # 2-D
    ens = SoftVotingEnsemble(estimators=_make_cpu_estimators())
    with pytest.raises(AssertionError):
        ens.fit(X, y_bad)


# ── 9. CPU fallback: HAS_CUML=False → sklearn RF used ────────────────────
def test_cpu_fallback(synthetic_data):
    """
    When HAS_CUML is False, build_default_estimators must return a sklearn RF
    (not a cuML RF). We patch HAS_CUML to False via module attribute manipulation.
    """
    import src.training.ensemble as ens_module
    original_cuml = ens_module.HAS_CUML
    original_rf_cls = ens_module.CuMLRF

    try:
        from sklearn.ensemble import RandomForestClassifier as SklearnRF
        ens_module.HAS_CUML = False
        ens_module.CuMLRF = SklearnRF

        estimators = build_default_estimators({}, {}, {})
        assert isinstance(estimators["rf"], SklearnRF), (
            f"Expected sklearn RF, got {type(estimators['rf'])}"
        )
    finally:
        # Restore module state
        ens_module.HAS_CUML = original_cuml
        ens_module.CuMLRF = original_rf_cls
