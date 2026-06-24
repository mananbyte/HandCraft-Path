import os
import tempfile
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.training.ensemble import SoftVotingEnsemble, build_default_estimators


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = np.random.rand(500, 25).astype(np.float32)
    y = np.random.randint(0, 2, size=500).astype(np.int32)
    return X, y


@pytest.fixture
def fitted_ensemble(dummy_data):
    X, y = dummy_data
    # Use small scikit-learn RFs for fast testing
    estimators = {
        "rf": RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42),
        "lgbm": RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42),
        "xgb": RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42),
    }
    ens = SoftVotingEnsemble(estimators=estimators)
    ens.fit(X, y)
    return ens


def test_soft_voting_proba_shape(fitted_ensemble, dummy_data):
    X, _ = dummy_data
    probas = fitted_ensemble.predict_proba(X)
    assert probas.shape == (500, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_soft_voting_predict(fitted_ensemble, dummy_data):
    X, _ = dummy_data
    preds = fitted_ensemble.predict(X)
    assert preds.ndim == 1
    assert preds.shape == (500,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_weighted_average():
    ens = SoftVotingEnsemble(estimators={}, weights={"rf": 0.5, "lgbm": 0.5})
    # Mock probas
    probas = {
        "rf": np.array([[0.2, 0.8], [0.9, 0.1]]),
        "lgbm": np.array([[0.4, 0.6], [0.7, 0.3]]),
    }
    # Weighted avg manually:
    # row 0: 0.5*0.2 + 0.5*0.4 = 0.3, 0.5*0.8 + 0.5*0.6 = 0.7
    # row 1: 0.5*0.9 + 0.5*0.7 = 0.8, 0.5*0.1 + 0.5*0.3 = 0.2
    
    class MockEst:
        def __init__(self, probas):
            self.p = probas
        def predict_proba(self, X):
            return self.p

    ens.estimators = {"rf": MockEst(probas["rf"]), "lgbm": MockEst(probas["lgbm"])}
    ens._fitted = True
    
    # Dummy X just for length
    X = np.zeros((2, 5))
    res = ens.predict_proba(X)
    expected = np.array([[0.3, 0.7], [0.8, 0.2]])
    np.testing.assert_allclose(res, expected)


def test_equal_weights_default():
    estimators = {"rf": None, "lgbm": None, "xgb": None}
    ens = SoftVotingEnsemble(estimators)
    assert ens.weights == {"rf": 1/3, "lgbm": 1/3, "xgb": 1/3}


def test_optimize_weights_grid(fitted_ensemble, dummy_data):
    X, y = dummy_data
    best_weights = fitted_ensemble.optimize_weights(X, y, step=0.5)
    assert isinstance(best_weights, dict)
    assert "rf" in best_weights and "lgbm" in best_weights and "xgb" in best_weights
    total_weight = sum(best_weights.values())
    assert np.isclose(total_weight, 1.0)
    assert fitted_ensemble.weights == best_weights


def test_save_load_roundtrip(fitted_ensemble, dummy_data):
    X, _ = dummy_data
    original_preds = fitted_ensemble.predict(X)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ensemble.joblib")
        fitted_ensemble.save(path)
        
        loaded_ensemble = SoftVotingEnsemble.load(path)
        loaded_preds = loaded_ensemble.predict(X)
        
        np.testing.assert_array_equal(original_preds, loaded_preds)
        assert loaded_ensemble.weights == fitted_ensemble.weights


def test_shape_assertion_X(fitted_ensemble):
    with pytest.raises(AssertionError):
        fitted_ensemble.fit(np.zeros((5, 5, 5)), np.zeros(5))


def test_shape_assertion_y(fitted_ensemble):
    with pytest.raises(AssertionError):
        fitted_ensemble.fit(np.zeros((5, 5)), np.zeros((5, 2)))


def test_cpu_fallback():
    # If we fake HAS_CUML = False and CuMLRF to sklearn's RF, it should work
    import src.training.ensemble as ens_module
    from sklearn.ensemble import RandomForestClassifier
    old_cumlrf = ens_module.CuMLRF
    ens_module.CuMLRF = RandomForestClassifier
    
    try:
        ests = ens_module.build_default_estimators({}, {}, {})
        assert isinstance(ests["rf"], RandomForestClassifier)
    finally:
        ens_module.CuMLRF = old_cumlrf
