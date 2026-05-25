import os
import sys
import numpy as np
import tempfile

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.training.feature_selection import get_feature_names, mrmr_selection, safe_load_npy

def test_get_feature_names():
    names = get_feature_names()
    assert len(names) == 93
    assert "od_R" in names
    assert "glcm_ASM" in names
    assert "entropy" in names
    assert "edge_dist" in names
    print("test_get_feature_names passed!")

def test_mrmr_selection():
    # Create simple mock data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    # Make feature 2 highly correlated with feature 0
    X[:, 2] = X[:, 0] * 0.95 + np.random.randn(100) * 0.05
    # Target depends on feature 0 and 5
    y = (X[:, 0] + X[:, 5] > 0).astype(int)
    
    selected = mrmr_selection(X, y, k=3)
    assert len(selected) == 3
    # Feature 0 or 2 should be selected since they are highly relevant
    assert 0 in selected or 2 in selected
    # Feature 5 should be selected because it's relevant and not redundant
    assert 5 in selected
    print("test_mrmr_selection passed!")

def test_safe_load_npy():
    # Test safe load npy with standard numpy file
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = os.path.join(tmp_dir, "test_array.npy")
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.save(filepath, arr)
        
        loaded = safe_load_npy(filepath)
        assert np.allclose(arr, loaded)
        assert loaded.dtype == np.float32
        assert loaded.shape == (2, 2)
    print("test_safe_load_npy passed!")

if __name__ == "__main__":
    test_get_feature_names()
    test_mrmr_selection()
    test_safe_load_npy()
    print("All tests passed successfully!")
