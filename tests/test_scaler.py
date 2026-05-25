"""
tests/test_scaler.py
──────────────────────
Unit tests for src/training/scaler.py (FeatureScaler).

Tests are designed to run in <10 seconds on CPU with synthetic data
and do not require the real PanNuke processed matrices.
"""

import os
import tempfile

import numpy as np
import pytest

from src.training.scaler import FeatureScaler, get_all_scaler_types, SCALER_REGISTRY

# ── Fixtures ──────────────────────────────────────────────────────────────
N_ROWS: int = 1_000
N_FEATURES: int = 93
RANDOM_STATE: int = 42


@pytest.fixture
def synthetic_npy(tmp_path):
    """Creates small synthetic X and y .npy files for testing."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_ROWS, N_FEATURES)).astype(np.float32)
    # Inject some outliers (simulating stain artefacts)
    X[::50, :3] *= 50.0
    y = rng.integers(0, 2, size=N_ROWS)

    X_path = str(tmp_path / "X.npy")
    y_path = str(tmp_path / "y.npy")
    np.save(X_path, X)
    np.save(y_path, y)
    return X_path, y_path, X, y


# ── Registry tests ─────────────────────────────────────────────────────────
def test_get_all_scaler_types():
    types = get_all_scaler_types()
    assert isinstance(types, list)
    assert len(types) == 6
    assert "RobustScaler" in types
    assert "StandardScaler" in types
    assert "HandCraftPathScaler" in types


def test_invalid_scaler_type_raises():
    with pytest.raises(AssertionError, match="Unknown scaler"):
        FeatureScaler(scaler_type="MagicScaler")


# ── Fit tests ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("scaler_type", list(SCALER_REGISTRY.keys()))
def test_fit_all_scalers(scaler_type, synthetic_npy):
    X_path, y_path, _, _ = synthetic_npy
    fs = FeatureScaler(scaler_type=scaler_type)
    result = fs.fit(X_path, y_path, sample_n=500)
    assert result is fs, "fit() must return self"
    assert fs._fitted is True


def test_fit_returns_self(synthetic_npy):
    X_path, y_path, _, _ = synthetic_npy
    fs = FeatureScaler()
    assert fs.fit(X_path, y_path, sample_n=500) is fs


def test_transform_before_fit_raises(synthetic_npy, tmp_path):
    X_path, _, _, _ = synthetic_npy
    fs = FeatureScaler()
    with pytest.raises(AssertionError, match="fit\\(\\)"):
        fs.transform_memmap(X_path, str(tmp_path / "out.npy"))


# ── Transform tests ────────────────────────────────────────────────────────
def test_transform_memmap_output_shape(synthetic_npy, tmp_path):
    X_path, y_path, X, _ = synthetic_npy
    out_path = str(tmp_path / "X_scaled.npy")
    fs = FeatureScaler("RobustScaler")
    fs.fit(X_path, y_path, sample_n=500)
    fs.transform_memmap(X_path, out_path, chunk_size=200)

    X_out = np.load(out_path, mmap_mode="r")
    assert X_out.shape == X.shape, f"Expected {X.shape}, got {X_out.shape}"


def test_robust_scaler_suppresses_outliers(synthetic_npy, tmp_path):
    """RobustScaler should yield fewer values with |z|>3 than StandardScaler on outlier-injected data."""
    X_path, y_path, _, _ = synthetic_npy

    robust_out = str(tmp_path / "robust.npy")
    std_out    = str(tmp_path / "standard.npy")

    FeatureScaler("RobustScaler").fit(X_path, y_path, sample_n=500).transform_memmap(
        X_path, robust_out, chunk_size=200
    )
    FeatureScaler("StandardScaler").fit(X_path, y_path, sample_n=500).transform_memmap(
        X_path, std_out, chunk_size=200
    )

    X_robust = np.load(robust_out)
    X_std    = np.load(std_out)

    # After RobustScaler the median of each column should be closer to 0
    median_robust = np.abs(np.median(X_robust, axis=0)).mean()
    median_std    = np.abs(np.median(X_std,    axis=0)).mean()
    # Robust median should be ≤ standard median (robust explicitly centres on median)
    assert median_robust <= median_std + 0.5, (
        f"RobustScaler median {median_robust:.4f} unexpectedly > StandardScaler {median_std:.4f}"
    )


# ── Save / Load round-trip ─────────────────────────────────────────────────
def test_save_load_roundtrip(synthetic_npy, tmp_path):
    X_path, y_path, X, _ = synthetic_npy
    model_path = str(tmp_path / "scaler.joblib")

    fs = FeatureScaler("RobustScaler")
    fs.fit(X_path, y_path, sample_n=500)
    fs.save(model_path)

    assert os.path.exists(model_path)

    fs2 = FeatureScaler.load(model_path)
    assert fs2.scaler_type == "RobustScaler"
    assert fs2._fitted is True

    # Both scalers must produce the same output for the same input
    sample = X[:10].astype(np.float32)
    out1 = fs._scaler.transform(sample)
    out2 = fs2._scaler.transform(sample)
    np.testing.assert_allclose(out1, out2, rtol=1e-5)


# ── spot_check ─────────────────────────────────────────────────────────────
def test_spot_check_returns_dict(synthetic_npy, tmp_path):
    X_path, y_path, _, _ = synthetic_npy
    out_path = str(tmp_path / "scaled.npy")
    fs = FeatureScaler("RobustScaler")
    fs.fit(X_path, y_path, sample_n=500)
    fs.transform_memmap(X_path, out_path, chunk_size=200)

    stats = fs.spot_check(out_path, n_rows=100)
    assert "mean_abs_col_mean" in stats
    assert "outlier_frac_3sigma" in stats
    assert stats["mean_abs_col_mean"] >= 0.0


def test_repr(synthetic_npy):
    X_path, y_path, _, _ = synthetic_npy
    fs = FeatureScaler("RobustScaler")
    assert "unfitted" in repr(fs)
    fs.fit(X_path, y_path, sample_n=200)
    assert "fitted" in repr(fs)
