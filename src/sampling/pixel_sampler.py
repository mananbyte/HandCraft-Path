"""
Active Boundary Mining — pixel sampling strategy for Stage 1.

Oversamples pixels near nucleus boundaries where classification
errors are most consequential. Uses GPU distance transform when
available, CPU fallback otherwise.

From feature_research.md and ram-issues.txt:
  - n_per_class=400 → ~1,200 pixels per image (3 classes for Exp B)
  - boundary_boost=3 → 3× oversampling within 10px of nucleus edge
  - Training only — validation/test use uniform sampling
"""

import numpy as np
import os

try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as gpu_edt
    _GPU_EDT = True
except ImportError:
    _GPU_EDT = False


_GPU_DISABLED_RUNTIME = False
_GPU_INIT_DONE = False
_FORCE_CPU_EDT = os.getenv("PANNUKE_FORCE_CPU_EDT", "0").lower() in {
    "1", "true", "yes", "y"
}


def _initialize_gpu_runtime():
    """Initialize CuPy once and clear stale memory pools."""
    global _GPU_DISABLED_RUNTIME, _GPU_INIT_DONE
    if _GPU_INIT_DONE or not _GPU_EDT:
        return

    if _FORCE_CPU_EDT:
        _GPU_DISABLED_RUNTIME = True
        _GPU_INIT_DONE = True
        print("[pixel_sampler] PANNUKE_FORCE_CPU_EDT=1 -> using CPU EDT")
        return

    try:
        _ = cp.cuda.runtime.getDevice()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()
        _GPU_INIT_DONE = True
    except Exception as exc:
        _GPU_DISABLED_RUNTIME = True
        _GPU_INIT_DONE = True
        print(f"[pixel_sampler] GPU init failed, using CPU EDT: {exc}")


def _distance_transform(binary_mask_cpu):
    """GPU-accelerated distance transform with CPU fallback."""
    global _GPU_DISABLED_RUNTIME
    _initialize_gpu_runtime()

    if _GPU_EDT and not _GPU_DISABLED_RUNTIME:
        try:
            mask_gpu = cp.asarray(binary_mask_cpu.astype(np.float32))
            dist_gpu = gpu_edt(mask_gpu)
            return cp.asnumpy(dist_gpu)
        except Exception as exc:
            _GPU_DISABLED_RUNTIME = True
            print(f"[pixel_sampler] GPU EDT unavailable, falling back to CPU: {exc}")
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    from scipy.ndimage import distance_transform_edt as cpu_edt
    return cpu_edt(binary_mask_cpu)


def active_boundary_mining(labels, n_per_class=400, boundary_boost=3):
    """
    Sample pixel indices with elevated probability near boundaries.

    Strategy:
    1. All nucleus pixels are eligible for sampling
    2. Background pixels within 10px of a nucleus are sampled
       at boundary_boost × higher rate than far-background
    3. Strictly enforces n_per_class per label class

    Parameters
    ----------
    labels : np.ndarray, shape (256, 256), values in {0,1} or {0,1,2}
    n_per_class : int — target samples per class
    boundary_boost : int — oversampling factor for near-boundary pixels

    Returns
    -------
    sampled_indices : np.ndarray, shape (N,) — flat pixel indices (int32)
    sampled_labels  : np.ndarray, shape (N,) — corresponding labels (uint8)
    """
    H, W = labels.shape
    flat_labels = labels.flatten()

    # Distance transform from nucleus pixels
    nucleus_mask = (labels > 0).astype(np.float32)
    dist_from_nucleus = _distance_transform(1 - nucleus_mask)

    # Sampling weight map
    weights = np.ones(H * W, dtype=np.float32)
    near_boundary = (dist_from_nucleus.flatten() < 10) & (flat_labels == 0)
    weights[near_boundary] *= boundary_boost

    all_indices = []
    all_labels = []

    unique_labels = np.unique(flat_labels)

    for lbl in unique_labels:
        lbl_mask = (flat_labels == lbl)
        lbl_weights = weights * lbl_mask.astype(np.float32)
        lbl_weights_sum = lbl_weights.sum()

        if lbl_weights_sum == 0:
            continue

        lbl_probs = lbl_weights / lbl_weights_sum
        n_available = lbl_mask.sum()
        n_sample = min(n_per_class, n_available)

        chosen = np.random.choice(
            H * W, size=n_sample, replace=False, p=lbl_probs
        )
        all_indices.append(chosen)
        all_labels.append(np.full(n_sample, lbl, dtype=np.uint8))

    indices = np.concatenate(all_indices).astype(np.int32)
    sampled_labels = np.concatenate(all_labels)
    return indices, sampled_labels
