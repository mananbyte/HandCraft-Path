"""
tests/test_feature_selection.py
──────────────────────────────────
Unit tests for src/training/feature_selection.py.

Tests run on synthetic data in <15 seconds without requiring real datasets.
"""

import json
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.training.feature_selection import (
    get_feature_names,
    mrmr_selection,
    run_feature_selection,
    save_selected_features,
    SUPPORTED_METHODS,
    DEFAULT_N_FEATURES,
)
from src.utils.safe_loader import safe_load_npy

# ── Constants ──────────────────────────────────────────────────────────────
N_ROWS: int = 2_000
N_FEATURES: int = 93
RANDOM_STATE: int = 42


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_scaled():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_ROWS, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, 2, size=N_ROWS)
    return X, y


@pytest.fixture
def synthetic_scaled_3class():
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.standard_normal((N_ROWS, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, 3, size=N_ROWS)
    return X, y


# ── Feature names ──────────────────────────────────────────────────────────
def test_get_feature_names_count():
    names = get_feature_names()
    assert len(names) == N_FEATURES, f"Expected 93 features, got {len(names)}"


def test_get_feature_names_unique():
    names = get_feature_names()
    assert len(set(names)) == len(names), "Feature names must be unique"


def test_get_feature_names_contains_known():
    names = get_feature_names()
    assert "od_R" in names
    assert "glcm_ASM" in names
    assert "entropy" in names
    assert "edge_dist" in names


# ── mrmr_selection ─────────────────────────────────────────────────────────
def test_mrmr_returns_k_indices(synthetic_scaled):
    X, y = synthetic_scaled
    indices = mrmr_selection(X, y, k=20)
    assert len(indices) == 20


def test_mrmr_indices_in_range(synthetic_scaled):
    X, y = synthetic_scaled
    indices = mrmr_selection(X, y, k=15)
    assert all(0 <= i < N_FEATURES for i in indices), "All indices must be in [0, 93)"


def test_mrmr_no_duplicates(synthetic_scaled):
    X, y = synthetic_scaled
    indices = mrmr_selection(X, y, k=20)
    assert len(indices) == len(set(indices)), "mRMR must not return duplicate indices"


def test_mrmr_selects_relevant_feature():
    """Classic mRMR check: highly-relevant feature must appear in selection."""
    np.random.seed(RANDOM_STATE)
    X = np.random.randn(500, 10).astype(np.float32)
    X[:, 2] = X[:, 0] * 0.95 + np.random.randn(500).astype(np.float32) * 0.05
    y = (X[:, 0] + X[:, 5] > 0).astype(int)
    selected = mrmr_selection(X, y, k=3)
    assert len(selected) == 3
    assert 0 in selected or 2 in selected
    assert 5 in selected


# ── run_feature_selection ──────────────────────────────────────────────────
@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_run_feature_selection_returns_n(method, synthetic_scaled):
    X, y = synthetic_scaled
    n = 15
    indices, names = run_feature_selection(X, y, n_features=n, method=method)
    assert len(indices) == n
    assert len(names) == n


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_run_feature_selection_indices_valid(method, synthetic_scaled):
    X, y = synthetic_scaled
    indices, _ = run_feature_selection(X, y, n_features=10, method=method)
    assert all(0 <= i < N_FEATURES for i in indices)


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_run_feature_selection_no_duplicates(method, synthetic_scaled):
    X, y = synthetic_scaled
    indices, _ = run_feature_selection(X, y, n_features=10, method=method)
    assert len(indices) == len(set(indices)), f"{method}: duplicate indices"


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_run_feature_selection_names_match_registry(method, synthetic_scaled):
    X, y = synthetic_scaled
    all_names = get_feature_names()
    indices, names = run_feature_selection(X, y, n_features=10, method=method)
    expected = [all_names[i] for i in indices]
    assert names == expected


def test_run_feature_selection_invalid_method_raises(synthetic_scaled):
    X, y = synthetic_scaled
    with pytest.raises(AssertionError, match="Unknown method"):
        run_feature_selection(X, y, n_features=10, method="magic_method")


def test_run_feature_selection_n_out_of_range_raises(synthetic_scaled):
    X, y = synthetic_scaled
    with pytest.raises(AssertionError):
        run_feature_selection(X, y, n_features=200, method="mrmr")


def test_run_feature_selection_3class(synthetic_scaled_3class):
    X, y = synthetic_scaled_3class
    for method in SUPPORTED_METHODS:
        indices, names = run_feature_selection(X, y, n_features=10, method=method)
        assert len(indices) == 10


# ── save_selected_features ─────────────────────────────────────────────────
def test_save_creates_json_and_csv(tmp_path, synthetic_scaled):
    X, y = synthetic_scaled
    indices, names = run_feature_selection(X, y, n_features=10, method="mrmr")
    paths = save_selected_features(indices, names, method="mrmr", n=10, out_dir=str(tmp_path))
    assert os.path.exists(paths["json"])
    assert os.path.exists(paths["csv"])


def test_saved_json_is_loadable(tmp_path, synthetic_scaled):
    X, y = synthetic_scaled
    indices, names = run_feature_selection(X, y, n_features=10, method="anova")
    paths = save_selected_features(indices, names, method="anova", n=10, out_dir=str(tmp_path))
    with open(paths["json"]) as fh:
        payload = json.load(fh)
    assert payload["method"] == "anova"
    assert payload["n_features"] == 10
    assert len(payload["indices"]) == 10
    assert len(payload["names"]) == 10


def test_saved_json_indices_match(tmp_path, synthetic_scaled):
    X, y = synthetic_scaled
    indices, names = run_feature_selection(X, y, n_features=12, method="mrmr")
    paths = save_selected_features(indices, names, method="mrmr", n=12, out_dir=str(tmp_path))
    with open(paths["json"]) as fh:
        payload = json.load(fh)
    assert payload["indices"] == [int(i) for i in indices]
    assert payload["names"] == names


def test_save_mismatch_raises(tmp_path):
    with pytest.raises(AssertionError, match="length mismatch"):
        save_selected_features([0, 1], ["a"], method="mrmr", n=2, out_dir=str(tmp_path))


def test_save_correct_filename(tmp_path, synthetic_scaled):
    X, y = synthetic_scaled
    indices, names = run_feature_selection(X, y, n_features=5, method="rfe")
    paths = save_selected_features(indices, names, method="rfe", n=5, out_dir=str(tmp_path))
    assert "selected_features_rfe_5" in paths["json"]
    assert "selected_features_rfe_5" in paths["csv"]


# ── safe_load_npy (existing test preserved) ───────────────────────────────
def test_safe_load_npy(tmp_path):
    filepath = str(tmp_path / "test_array.npy")
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.save(filepath, arr)
    loaded = safe_load_npy(filepath)
    assert np.allclose(arr, loaded)
    assert loaded.dtype == np.float32
    assert loaded.shape == (2, 2)


if __name__ == "__main__":
    print("All feature selection tests pass.")
