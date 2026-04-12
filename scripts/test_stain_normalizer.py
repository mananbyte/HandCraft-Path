"""
Quick smoke-test for the fixed MacenkoNormalizer.
Run with: python scripts/test_stain_normalizer.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import MagicMock
# Staintools imports Vahadane by default; avoid spams dependency for Macenko-only use.
sys.modules['spams'] = MagicMock()

import numpy as np
from src.preprocessing.stain_normalizer import MacenkoNormalizer

BASE = "data/raw/Fold1/images/fold1/images.npy"

print("=" * 60)
print("Stain Normalizer Smoke Test")
print("=" * 60)

# Load a small subset
print(f"\nLoading {BASE} ...")
images = np.load(BASE, mmap_mode='r')
print(f"Loaded: {images.shape}, dtype={images.dtype}")

# Test on first 50 images
subset = images[:50]

print("\n--- Fitting normalizer on 50 images ---")
normalizer = MacenkoNormalizer()
normalizer.fit(subset)

print(f"\nFit succeeded! Reference index: {normalizer.reference_index}")

# Test transform on 5 images
print("\n--- Testing transform on 5 images ---")
for i in [0, 10, 20, 30, 40]:
    img = images[i].astype(np.uint8)
    result = normalizer.transform(img)
    assert result.shape == (256, 256, 3), f"Wrong output shape: {result.shape}"
    assert result.dtype == np.uint8,       f"Wrong output dtype: {result.dtype}"
    print(f"  Image {i:>3}: input range [{img.min()}, {img.max()}]  "
          f"→ normalized range [{result.min()}, {result.max()}]  ✅")

print("\nAll tests passed ✅")

# Save for later use
os.makedirs("data/models", exist_ok=True)
normalizer.save("data/models/normalizer_test.joblib")
print("Test normalizer saved to data/models/normalizer_test.joblib")
