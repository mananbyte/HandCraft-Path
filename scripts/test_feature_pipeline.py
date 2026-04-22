"""
Quick smoke test: extract features for 5 images, verify shapes and no NaN.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.preprocessing.stain_normalizer import MacenkoNormalizer
from src.preprocessing.color_converter import convert_image
from src.data.label_generator import generate_binary_labels, generate_3class_labels
from src.features.pixel_feature_extractor import (
    extract_dense_features_cpu, compute_glcm_for_samples,
    DENSE_FEATURES, GLCM_FEATURES, TOTAL_FEATURES,
)
from src.sampling.pixel_sampler import active_boundary_mining

print(f"Feature constants: DENSE={DENSE_FEATURES}, GLCM={GLCM_FEATURES}, TOTAL={TOTAL_FEATURES}")

# Load a small subset
images = np.load("data/raw/Fold1/images/fold1/images.npy", mmap_mode='r')
masks  = np.load("data/raw/Fold1/masks/fold1/masks.npy",  mmap_mode='r')

print(f"Loaded: images={images.shape}, masks={masks.shape}")

# Fit normalizer on first 50 images
normalizer = MacenkoNormalizer()
normalizer.fit(images[:50])

# Test 5 images
for idx in [0, 10, 20, 30, 40]:
    t0 = time.time()

    img = normalizer.transform(np.clip(images[idx], 0, 255).astype(np.uint8))
    spaces = convert_image(img)

    # Dense features
    dense = extract_dense_features_cpu(spaces)
    assert dense.shape == (65536, DENSE_FEATURES), f"Bad dense shape: {dense.shape}"
    assert not np.isnan(dense).any(), "NaN in dense features!"

    # Labels + sampling
    labels_bin = generate_binary_labels(masks[idx])
    indices, sampled_labels = active_boundary_mining(labels_bin, n_per_class=400)

    # GLCM
    gray = spaces['lab'][:, :, 0].astype(np.float32)
    glcm = compute_glcm_for_samples(gray, indices)
    assert glcm.shape == (len(indices), GLCM_FEATURES), f"Bad GLCM shape: {glcm.shape}"

    # Final assembly
    full = np.concatenate([dense[indices], glcm], axis=1)
    assert full.shape[1] == TOTAL_FEATURES, f"Bad total: {full.shape[1]}"
    assert not np.isnan(full).any(), "NaN in final features!"

    dt = time.time() - t0
    print(f"  Image {idx:3d}: dense={dense.shape}, "
          f"sampled={len(indices)}, "
          f"final=({len(indices)}, {full.shape[1]}), "
          f"time={dt:.1f}s ✓")

print(f"\n✅ All 5 images passed. Feature pipeline is correct.")
print(f"   Dense: {DENSE_FEATURES} features per pixel")
print(f"   GLCM:  {GLCM_FEATURES} features per sampled pixel")
print(f"   Total: {TOTAL_FEATURES} features per sampled pixel")
