"""
Pixel-level feature extraction for Stage 1 segmentation.

Features: 87 dense (per-pixel) + 6 GLCM (per-sampled-pixel) = 93 total.

Feature groups (from feature_research.md):
  1. Optical Density (OD)           : 3   — Beer-Lambert DNA content
  2. Color stats (LAB+HSV+HED)      : 54  — mean+std × 3 scales × 9 channels
  3. LBP (r=1,2,3)                   : 3   — chromatin micro-texture
  4. Gabor bank (3 freq × 4 orient)  : 12  — directional texture
  5. Sobel + LoG                     : 5   — edges + blob detection
  6. Structure tensor                 : 3   — isotropy vs anisotropy
  7. DoG scale space                  : 3   — nucleus size encoding
  8. Superpixel context               : 2   — spatial coherence
  9. Local entropy                    : 1   — transition zone detector
 10. Edge distance                    : 1   — border artifact correction
     ─────────────────────────────────────
     Dense subtotal                   : 87
 11. GLCM (per sampled pixel only)   : 6   — texture regularity
     ─────────────────────────────────────
     Total per sampled pixel          : 93

NOTE: Frangi vesselness is deliberately excluded — redundant with
structure tensor anisotropy (see feature_research.md Part D).
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Try GPU imports; fall back gracefully ─────────────────────────────────
try:
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter as gpu_uniform_filter
    from cupyx.scipy.ndimage import gaussian_laplace as gpu_gaussian_laplace
    # CuPy import can succeed even when runtime init fails (driver mismatch).
    _ = cp.cuda.runtime.getDeviceCount()
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

from scipy.ndimage import uniform_filter as cpu_uniform_filter
from scipy.ndimage import gaussian_laplace as cpu_gaussian_laplace
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import sobel_h, sobel_v, gaussian
from skimage.segmentation import slic
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk as morph_disk

# ── Constants ─────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    'od':         3,   # Optical density R, G, B
    'color':     54,   # 9 channels × 2 stats × 3 scales
    'lbp':        3,   # LBP at r=1, 2, 3
    'gabor':     12,   # 3 freq × 4 orient
    'gradient':   5,   # Sobel X, Y, mag + LoG σ=2,4
    'structure':  3,   # λ1, λ2, anisotropy
    'dog':        3,   # DoG at 3 scale pairs
    'superpixel': 2,   # Superpixel mean HED-H, LAB-L
    'entropy':    1,   # Local entropy window=5
    'edge_dist':  1,   # Normalized distance to border
}
DENSE_FEATURES = sum(FEATURE_GROUPS.values())  # 87
GLCM_FEATURES = 6
TOTAL_FEATURES = DENSE_FEATURES + GLCM_FEATURES  # 93


# ═════════════════════════════════════════════════════════════════════════
# GROUP 1 — Optical Density (3 features)
# ═════════════════════════════════════════════════════════════════════════

def compute_od_features(rgb_float):
    """OD = -log(I/255 + 1e-6) per RGB channel. Shape: (H, W, 3)."""
    od = -np.log(rgb_float / 255.0 + 1e-6)
    return od.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════
# GROUP 2 — Color statistics at 3 scales (54 features)
# ═════════════════════════════════════════════════════════════════════════

def compute_windowed_stats_batch_gpu(channels_batch, radii=(3, 7, 15)):
    """
    GPU-accelerated windowed mean+std for a batch of single-channel images.

    Parameters
    ----------
    channels_batch : np.ndarray, shape (B, H, W)
    radii : tuple of int

    Returns
    -------
    features : np.ndarray, shape (B, H, W, 2*len(radii))  — back on CPU
    """
    channels_gpu = cp.asarray(channels_batch.astype(np.float32))
    features = []
    for r in radii:
        size = (1, 2*r+1, 2*r+1)
        mean_map = gpu_uniform_filter(channels_gpu, size=size)
        mean_sq  = gpu_uniform_filter(channels_gpu ** 2, size=size)
        std_map  = cp.sqrt(cp.maximum(mean_sq - mean_map**2, 0))
        features.append(mean_map)
        features.append(std_map)
    result = cp.stack(features, axis=-1)
    return cp.asnumpy(result)


def compute_windowed_stats_single_cpu(channel, radii=(3, 7, 15)):
    """CPU fallback for a single channel."""
    features = []
    for r in radii:
        size = 2 * r + 1
        ch = channel.astype(np.float32)
        mean_map = cpu_uniform_filter(ch, size=size)
        mean_sq  = cpu_uniform_filter(ch ** 2, size=size)
        std_map  = np.sqrt(np.maximum(mean_sq - mean_map**2, 0))
        features.extend([mean_map, std_map])
    return np.stack(features, axis=-1)


def compute_color_features_batch_gpu(spaces_list):
    """
    GPU-batch color feature extraction for multiple images.
    Returns list of np.ndarray, each shape (256, 256, 54).
    """
    B = len(spaces_list)
    H, W = 256, 256
    all_feats = [np.empty((H, W, 0), dtype=np.float32) for _ in range(B)]

    for space_name in ['lab', 'hsv', 'hed']:
        for c in range(3):
            batch = np.stack([s[space_name][:, :, c] for s in spaces_list])
            stats = compute_windowed_stats_batch_gpu(batch)  # (B,H,W,6)
            for i in range(B):
                all_feats[i] = np.concatenate(
                    [all_feats[i], stats[i]], axis=-1
                )
    return all_feats  # each (H, W, 54)


def compute_color_features_cpu(spaces):
    """Single-image CPU path. Returns (H, W, 54)."""
    feats = []
    for space_name in ['lab', 'hsv', 'hed']:
        for c in range(3):
            channel_feats = compute_windowed_stats_single_cpu(
                spaces[space_name][:, :, c]
            )
            feats.append(channel_feats)
    return np.concatenate(feats, axis=-1)


# ═════════════════════════════════════════════════════════════════════════
# GROUP 3 — LBP (3 features)
# ═════════════════════════════════════════════════════════════════════════

def compute_lbp_features(gray, radii=(1, 2, 3)):
    """Uniform rotation-invariant LBP at multiple radii. (H, W, 3)."""
    feats = []
    for r in radii:
        P = 8 * r
        lbp = local_binary_pattern(gray, P=P, R=r, method='uniform')
        lbp_norm = lbp / (P + 2)  # normalize to 0-1
        feats.append(lbp_norm.astype(np.float32))
    return np.stack(feats, axis=-1)


# ═════════════════════════════════════════════════════════════════════════
# GROUP 4 — Gabor bank (12 features)
# ═════════════════════════════════════════════════════════════════════════

def compute_gabor_features(gray):
    """3 frequencies × 4 orientations. (H, W, 12)."""
    from skimage.filters import gabor
    feats = []
    for frequency in [0.1, 0.2, 0.4]:
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            real, _ = gabor(gray, frequency=frequency, theta=theta)
            feats.append(real.astype(np.float32))
    return np.stack(feats, axis=-1)


# ═════════════════════════════════════════════════════════════════════════
# GROUP 5 — Gradient features: Sobel + LoG (5 features)
# ═════════════════════════════════════════════════════════════════════════

def compute_gradient_features_batch_gpu(gray_batch):
    """
    GPU-accelerated LoG + CPU Sobel for a batch.
    Returns list of np.ndarray, each (H, W, 5).
    """
    B = gray_batch.shape[0]
    gray_gpu = cp.asarray(gray_batch)

    log2_gpu = gpu_gaussian_laplace(gray_gpu, sigma=(0, 2, 2))
    log4_gpu = gpu_gaussian_laplace(gray_gpu, sigma=(0, 4, 4))
    log2_all = cp.asnumpy(log2_gpu)
    log4_all = cp.asnumpy(log4_gpu)

    results = []
    for i in range(B):
        gray = gray_batch[i]
        sx = sobel_h(gray).astype(np.float32)
        sy = sobel_v(gray).astype(np.float32)
        grad_mag = np.sqrt(sx**2 + sy**2)
        feats = np.stack([sx, sy, grad_mag,
                          log2_all[i].astype(np.float32),
                          log4_all[i].astype(np.float32)], axis=-1)
        results.append(feats)
    return results


def compute_gradient_features_cpu(gray):
    """Single-image CPU path. (H, W, 5)."""
    sx = sobel_h(gray).astype(np.float32)
    sy = sobel_v(gray).astype(np.float32)
    grad_mag = np.sqrt(sx**2 + sy**2)
    log2 = cpu_gaussian_laplace(gray, sigma=2).astype(np.float32)
    log4 = cpu_gaussian_laplace(gray, sigma=4).astype(np.float32)
    return np.stack([sx, sy, grad_mag, log2, log4], axis=-1)


# ═════════════════════════════════════════════════════════════════════════
# GROUP 6 — Structure tensor (3 features)
# ═════════════════════════════════════════════════════════════════════════

def compute_structure_tensor_features(gray):
    """λ1, λ2, anisotropy. (H, W, 3)."""
    Axx, Axy, Ayy = structure_tensor(gray, sigma=1)
    eig = structure_tensor_eigenvalues(np.array([Axx, Axy, Ayy]))
    ev1 = eig[0].astype(np.float32)
    ev2 = eig[1].astype(np.float32)
    anisotropy = (ev1 - ev2) / (ev1 + ev2 + 1e-8)
    return np.stack([ev1, ev2, anisotropy], axis=-1)


# ═════════════════════════════════════════════════════════════════════════
# GROUP 7 — Difference of Gaussians (3 features)
# ═════════════════════════════════════════════════════════════════════════

def compute_dog_features(gray):
    """DoG at 3 scale pairs. (H, W, 3)."""
    feats = []
    for s1, s2 in [(1, 2), (2, 4), (4, 8)]:
        g1 = gaussian(gray, sigma=s1)
        g2 = gaussian(gray, sigma=s2)
        feats.append((g1 - g2).astype(np.float32))
    return np.stack(feats, axis=-1)


# ═════════════════════════════════════════════════════════════════════════
# GROUP 8 — Superpixel context (2 features)
# ═════════════════════════════════════════════════════════════════════════

def compute_superpixel_features(spaces):
    """Mean HED-H and LAB-L per SLIC superpixel. (H, W, 2)."""
    h_channel = spaces['hed'][:, :, 0].astype(np.float32)
    gray = spaces['lab'][:, :, 0].astype(np.float32)
    img_uint8 = np.clip(spaces['rgb'], 0, 255).astype(np.uint8)

    segments = slic(img_uint8, n_segments=200, compactness=10,
                    sigma=1, start_label=0)

    sp_h_mean = np.zeros_like(h_channel)
    sp_l_mean = np.zeros_like(gray)
    for seg_id in np.unique(segments):
        mask = (segments == seg_id)
        sp_h_mean[mask] = h_channel[mask].mean()
        sp_l_mean[mask] = gray[mask].mean()

    return np.stack([sp_h_mean, sp_l_mean], axis=-1)


# ═════════════════════════════════════════════════════════════════════════
# GROUP 9 — Local entropy (1 feature)
# ═════════════════════════════════════════════════════════════════════════

def compute_entropy_feature(gray):
    """Shannon entropy in a window of radius 5. (H, W, 1)."""
    gray_uint = np.clip(gray, 0, 255).astype(np.uint8)
    local_ent = rank_entropy(gray_uint, morph_disk(5)).astype(np.float32)
    local_ent /= (local_ent.max() + 1e-8)
    return local_ent[:, :, np.newaxis]


# ═════════════════════════════════════════════════════════════════════════
# GROUP 10 — Edge distance (1 feature)
# ═════════════════════════════════════════════════════════════════════════

def compute_edge_distance(H=256, W=256):
    """Normalized distance to nearest image border. (H, W, 1)."""
    row_dist = np.minimum(np.arange(H), H - 1 - np.arange(H)) / H
    col_dist = np.minimum(np.arange(W), W - 1 - np.arange(W)) / W
    edge_dist = np.minimum(
        row_dist[:, np.newaxis], col_dist[np.newaxis, :]
    ).astype(np.float32)
    return edge_dist[:, :, np.newaxis]


# ═════════════════════════════════════════════════════════════════════════
# GROUP 11 — GLCM (6 features, per-sampled-pixel only)
# ═════════════════════════════════════════════════════════════════════════

def compute_glcm_for_patch(gray_patch):
    """
    GLCM statistics for a single 15×15 patch.
    Returns 1D array of 6 features (mean across 4 angles).
    """
    if gray_patch.size < 9:
        return np.zeros(6, dtype=np.float32)

    if gray_patch.max() <= 1.0:
        gray_uint = (gray_patch * 255).astype(np.uint8)
    else:
        gray_uint = gray_patch.astype(np.uint8)

    gray_32 = (gray_uint // 8).astype(np.uint8)  # downsample to 32 levels
    gcm = graycomatrix(
        gray_32, distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=32, symmetric=True, normed=True
    )

    props = ['contrast', 'dissimilarity', 'homogeneity',
             'energy', 'correlation', 'ASM']
    feats = []
    for p in props:
        vals = graycoprops(gcm, p).flatten()
        feats.append(vals.mean())

    return np.array(feats, dtype=np.float32)


def compute_glcm_for_samples(gray_image, flat_indices, patch_radius=7):
    """
    Compute GLCM features for a set of sampled pixel indices in parallel.

    Parameters
    ----------
    gray_image : np.ndarray, (H, W), float32
    flat_indices : np.ndarray, (N,), int — flat pixel indices
    patch_radius : int — half-size of patch (15×15 → radius=7)

    Returns
    -------
    glcm_features : np.ndarray, (N, 6), float32
    """
    H, W = gray_image.shape
    rows = flat_indices // W
    cols = flat_indices % W
    n = len(flat_indices)

    # Pre-extract patch slices
    patches = []
    for i in range(n):
        r, c = rows[i], cols[i]
        r0 = max(0, r - patch_radius)
        r1 = min(H, r + patch_radius + 1)
        c0 = max(0, c - patch_radius)
        c1 = min(W, c + patch_radius + 1)
        patch = gray_image[r0:r1, c0:c1]
        patches.append(patch)

    from joblib import Parallel, delayed
    # Run GLCM computations in parallel across all CPU cores using thread pool (Cython releases GIL)
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(compute_glcm_for_patch)(p) for p in patches
    )

    return np.array(results, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════
# ASSEMBLY — Single-image CPU path
# ═════════════════════════════════════════════════════════════════════════

def extract_dense_features_cpu(spaces):
    """
    Extract 87 dense features for one image (CPU path).

    Parameters
    ----------
    spaces : dict from convert_image()

    Returns
    -------
    features : np.ndarray, (65536, 87), float32
    """
    gray = spaces['lab'][:, :, 0].astype(np.float32)

    # Assemble all groups in the order defined in feature_research.md Part F
    groups = [
        compute_od_features(spaces['rgb']),          # (H,W,3)
        compute_color_features_cpu(spaces),           # (H,W,54)
        compute_lbp_features(gray),                   # (H,W,3)
        compute_gabor_features(gray),                 # (H,W,12)
        compute_gradient_features_cpu(gray),           # (H,W,5)
        compute_structure_tensor_features(gray),       # (H,W,3)
        compute_dog_features(gray),                    # (H,W,3)
        compute_superpixel_features(spaces),           # (H,W,2)
        compute_entropy_feature(gray),                 # (H,W,1)
        compute_edge_distance(),                       # (H,W,1)
    ]

    all_feats = np.concatenate(groups, axis=-1)  # (256, 256, 87)
    features = all_feats.reshape(-1, DENSE_FEATURES)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


def extract_all_features(image_uint8):
    """
    Extract full 87 dense features for one image (CPU path, for inference).
    GLCM is NOT included — it is computed separately per sampled pixel.

    Parameters
    ----------
    image_uint8 : np.ndarray, (256,256,3), uint8

    Returns
    -------
    features : np.ndarray, (65536, 87), float32
    """
    from src.preprocessing.color_converter import convert_image
    spaces = convert_image(image_uint8)
    return extract_dense_features_cpu(spaces)


# ═════════════════════════════════════════════════════════════════════════
# ASSEMBLY — Batch GPU path (for training)
# ═════════════════════════════════════════════════════════════════════════

def _extract_cpu_features_single(spaces, color_feats, gradient_feats):
    """Compute CPU-intensive dense features for a single image and assemble."""
    gray = spaces['lab'][:, :, 0].astype(np.float32)

    od_feats    = compute_od_features(spaces['rgb'])   # (H,W,3)
    lbp_feats   = compute_lbp_features(gray)                   # (H,W,3)
    gabor_feats = compute_gabor_features(gray)                 # (H,W,12)
    struct_feats = compute_structure_tensor_features(gray)      # (H,W,3)
    dog_feats   = compute_dog_features(gray)                   # (H,W,3)
    sp_feats    = compute_superpixel_features(spaces)          # (H,W,2)
    ent_feats   = compute_entropy_feature(gray)                # (H,W,1)
    edge_feats  = compute_edge_distance()                      # (H,W,1)

    all_feats = np.concatenate([
        od_feats,                     # 3
        color_feats,                  # 54
        lbp_feats,                    # 3
        gabor_feats,                  # 12
        gradient_feats,               # 5
        struct_feats,                 # 3
        dog_feats,                    # 3
        sp_feats,                     # 2
        ent_feats,                    # 1
        edge_feats,                   # 1
    ], axis=-1)                       # (256, 256, 87)

    features = all_feats.reshape(-1, DENSE_FEATURES)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════
# ASSEMBLY — Batch GPU path (for training)
# ═════════════════════════════════════════════════════════════════════════

def extract_features_batch_gpu(images_uint8, batch_size=32, log_progress=True):
    """
    GPU-accelerated feature extraction for a batch of images.

    Batches GPU-friendly ops (windowed stats, LoG) while running
    CPU-only ops (LBP, Gabor, SLIC) per-image in parallel.

    Parameters
    ----------
    images_uint8 : np.ndarray, (N, 256, 256, 3)
    batch_size : int
    log_progress : bool

    Returns
    -------
    all_features : list of np.ndarray, each (65536, 87)
    """
    from src.preprocessing.color_converter import convert_image

    if not _HAS_GPU:
        # Fall back to CPU for each image
        return [extract_all_features(img) for img in images_uint8]

    N = len(images_uint8)
    all_features = []

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_imgs = images_uint8[batch_start:batch_end]
        B = len(batch_imgs)

        # CPU: per-image color conversion
        spaces_list = [convert_image(img) for img in batch_imgs]

        # GPU batch: windowed stats (54 features)
        color_feats_list = compute_color_features_batch_gpu(spaces_list)

        # GPU batch: LoG (part of 5 gradient features)
        gray_batch = np.stack(
            [s['lab'][:, :, 0].astype(np.float32) for s in spaces_list]
        )
        gradient_feats_list = compute_gradient_features_batch_gpu(gray_batch)

        # CPU per-image: remaining features (in parallel across available cores!)
        from joblib import Parallel, delayed
        batch_features = Parallel(n_jobs=-1, backend="threading")(
            delayed(_extract_cpu_features_single)(
                spaces_list[i], color_feats_list[i], gradient_feats_list[i]
            ) for i in range(B)
        )
        all_features.extend(batch_features)

        if log_progress and (batch_end % 100 == 0 or batch_end == N):
            print(f"  Feature extraction: {batch_end}/{N} images")

    return all_features
