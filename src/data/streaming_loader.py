"""
Memory-safe streaming dataset builder using memmap + checkpoints.

Uses Option A from ram-issues.txt:
  Pass 1: Count total samples (labels only, ~2 min)
  Pass 2: Extract features + write directly to memmap

Peak RAM: ~500 MB constant regardless of dataset size.
Checkpoint every 100 images for crash recovery.

Output: .npy memmap files ready for model training.
"""

import numpy as np
import json
import os
import gc
import ctypes
from datetime import datetime

try:
    from tqdm.auto import tqdm
except Exception:
    class _NoOpTqdm:
        def __init__(self, *args, **kwargs):
            self.n = kwargs.get("initial", 0)

        def update(self, n=1):
            self.n += n

        def set_postfix_str(self, *args, **kwargs):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def tqdm(*args, **kwargs):
        return _NoOpTqdm(*args, **kwargs)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from src.data.label_generator import generate_binary_labels, generate_3class_labels
from src.preprocessing.color_converter import convert_image
from src.features.pixel_feature_extractor import (
    extract_dense_features_cpu, extract_features_batch_gpu,
    compute_glcm_for_samples, DENSE_FEATURES, GLCM_FEATURES, TOTAL_FEATURES,
    _HAS_GPU,
)
from src.sampling.pixel_sampler import active_boundary_mining


CHECKPOINT_EVERY = 100

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

try:
    _LIBC = ctypes.CDLL("libc.so.6")
    _HAS_MALLOC_TRIM = hasattr(_LIBC, "malloc_trim")
except Exception:
    _LIBC = None
    _HAS_MALLOC_TRIM = False


# ── Checkpoint I/O ────────────────────────────────────────────────────────

def save_checkpoint(path, fold_dir_idx, image_idx, write_ptr,
                    total_allocated, label_mode, n_per_class):
    """Atomic checkpoint write — safe against mid-write crashes."""
    data = {
        "last_completed_image_idx": image_idx,
        "write_ptr": int(write_ptr),
        "fold_dir_idx": fold_dir_idx,
        "total_allocated": int(total_allocated),
        "timestamp": datetime.now().isoformat(),
        "label_mode": label_mode,
        "n_per_class": n_per_class,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)  # atomic on Linux


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def _log_ram(label=""):
    if _HAS_PSUTIL:
        ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
        return f"{ram_mb:.0f}MB"
    return "N/A"


def _runtime_telemetry(start_time, images_processed, rows_written):
    """Compact runtime telemetry string for logs/progress bars."""
    elapsed = max((datetime.now() - start_time).total_seconds(), 1e-6)
    img_s = images_processed / elapsed
    row_s = rows_written / elapsed

    parts = [
        f"elapsed={elapsed/60:.1f}m",
        f"img/s={img_s:.2f}",
        f"row/s={row_s:.0f}",
        f"ram={_log_ram()}",
    ]

    if _HAS_PSUTIL:
        try:
            swap = psutil.swap_memory()
            parts.append(f"swap={swap.percent:.0f}%")
        except Exception:
            pass

    if _HAS_CUPY:
        try:
            free_b, total_b = cp.cuda.runtime.memGetInfo()
            used_b = total_b - free_b
            parts.append(f"gpu={used_b/1e9:.1f}/{total_b/1e9:.1f}GB")
        except Exception:
            # GPU may be unavailable mid-run; telemetry should never crash.
            pass

    return " | ".join(parts)


def _cleanup_runtime(trim_heap=False):
    """Best-effort runtime cleanup for long streaming loops."""
    gc.collect()
    if _HAS_CUPY:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    if trim_heap and _HAS_MALLOC_TRIM:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


# ── Pass 1: count samples ────────────────────────────────────────────────

def _count_total_samples(fold_dirs, label_mode, n_per_class):
    """
    Quick pass through all masks to count exact sample allocation.
    Does NOT compute features — only generates labels and counts classes.
    """
    total = 0
    n_classes = 2 if label_mode == 'binary' else 3

    total_images = sum(
        len(np.load(images_path, mmap_mode='r'))
        for images_path, _ in fold_dirs
    )

    pbar = tqdm(
        total=total_images,
        desc="Pass1 count",
        unit="img",
        dynamic_ncols=True,
        leave=True,
    )

    processed = 0

    try:
        for images_path, masks_path in fold_dirs:
            masks = np.load(masks_path, mmap_mode='r')
            for i in range(len(masks)):
                if label_mode == 'binary':
                    labels = generate_binary_labels(masks[i])
                else:
                    labels = generate_3class_labels(masks[i])
                for cls in range(n_classes):
                    total += min(n_per_class, (labels == cls).sum())

                processed += 1
                pbar.update(1)
                if processed % 50 == 0:
                    pbar.set_postfix_str(f"samples={total:,}")
    finally:
        pbar.close()

    return total


# ── Pass 2: extract + write ──────────────────────────────────────────────

def build_dataset_memmap(
        fold_dirs, normalizer, label_mode,
        output_X_path, output_y_path,
        checkpoint_path, n_per_class=400,
    checkpoint_every=CHECKPOINT_EVERY,
    feature_backend="auto",
    feature_batch_size=100,
    memory_config=None):
    """
    Streaming feature extraction with memmap + checkpoint/resume.

    Parameters
    ----------
    fold_dirs : list of (images_path, masks_path) tuples
    normalizer : MacenkoNormalizer (fitted)
    label_mode : 'binary' or '3class'
    output_X_path : str — path for features memmap (.npy)
    output_y_path : str — path for labels memmap (.npy)
    checkpoint_path : str — path for checkpoint JSON
    n_per_class : int — pixels per class per image
    checkpoint_every : int — save checkpoint every N images
    feature_backend : str — 'auto' | 'cpu' | 'gpu'
    feature_batch_size : int — images per dense-feature extraction batch
    memory_config : MemoryConfig, optional
        Memory configuration object (if not provided, fallback to feature_batch_size)

    Returns
    -------
    X_mm, y_mm : np.ndarray (memmap views, possibly truncated)
    """
    # ── Log memory configuration (if provided) ────────────────────────
    if memory_config is not None:
        print(f"\nMemory configuration:")
        print(f"  Backend: {memory_config.get_backend().upper()}")
        print(f"  Usable memory: {memory_config.usable_memory_gb:.2f} GB")
        if memory_config._calibrated:
            print(f"  Per-image (calibrated): {memory_config.actual_per_image_mb:.1f} MB")
        print()

    # ── Check for existing checkpoint ─────────────────────────────────
    ckpt = load_checkpoint(checkpoint_path)

    if ckpt is not None:
        print(f"═══ Resuming from checkpoint ═══")
        print(f"  Last image:  {ckpt['last_completed_image_idx']}")
        print(f"  Write ptr:   {ckpt['write_ptr']:,}")
        print(f"  Timestamp:   {ckpt['timestamp']}")

        total_allocated = ckpt['total_allocated']
        resume_fold_idx = ckpt['fold_dir_idx']
        resume_image_idx = ckpt['last_completed_image_idx'] + 1
        write_ptr = ckpt['write_ptr']

        X_mm = np.lib.format.open_memmap(
            output_X_path, mode='r+', dtype=np.float32,
            shape=(total_allocated, TOTAL_FEATURES)
        )
        y_mm = np.lib.format.open_memmap(
            output_y_path, mode='r+', dtype=np.uint8,
            shape=(total_allocated,)
        )

    else:
        print("═══ Starting fresh ═══")
        resume_fold_idx = 0
        resume_image_idx = 0
        write_ptr = 0

        # Pass 1: count
        print("\nPass 1: counting samples...")
        total_allocated = _count_total_samples(
            fold_dirs, label_mode, n_per_class
        )
        print(f"  Total samples: {total_allocated:,}")
        print(f"  X size on disk: {total_allocated * TOTAL_FEATURES * 4 / 1e9:.2f} GB")

        os.makedirs(os.path.dirname(output_X_path), exist_ok=True)

        X_mm = np.lib.format.open_memmap(
            output_X_path, mode='w+', dtype=np.float32,
            shape=(total_allocated, TOTAL_FEATURES)
        )
        y_mm = np.lib.format.open_memmap(
            output_y_path, mode='w+', dtype=np.uint8,
            shape=(total_allocated,)
        )

    # ── Pass 2: extract + write ───────────────────────────────────────
    feature_backend = str(feature_backend).lower().strip()
    if feature_backend not in {"auto", "cpu", "gpu"}:
        raise ValueError(
            "feature_backend must be one of: 'auto', 'cpu', 'gpu'"
        )
    if feature_batch_size < 1:
        raise ValueError("feature_batch_size must be >= 1")
    if feature_backend == "gpu" and not _HAS_GPU:
        raise RuntimeError(
            "feature_backend='gpu' requested but GPU extractor is unavailable"
        )

    resolved_feature_backend = (
        "gpu" if (feature_backend == "gpu" or (feature_backend == "auto" and _HAS_GPU))
        else "cpu"
    )
    use_gpu_features = resolved_feature_backend == "gpu"

    print(f"\nPass 2: extracting {TOTAL_FEATURES} features per pixel...")
    print(f"  Feature backend (requested): {feature_backend}")
    print(f"  Feature backend (resolved): {resolved_feature_backend.upper()}")
    print(f"  Feature batch size: {feature_batch_size}")
    print(f"  Checkpoint every: {checkpoint_every} images\n")

    images_processed = 0
    start_time = datetime.now()
    last_fold_idx = resume_fold_idx
    last_image_idx = max(resume_image_idx - 1, 0)

    total_images_all = sum(
        len(np.load(images_path, mmap_mode='r'))
        for images_path, _ in fold_dirs
    )
    overall_pbar = tqdm(
        total=total_images_all,
        desc="Pass2 overall",
        unit="img",
        dynamic_ncols=True,
        leave=True,
    )
    fold_pbar = None

    try:
        for fold_idx, (images_path, masks_path) in enumerate(fold_dirs):

            # Skip completed folds
            if fold_idx < resume_fold_idx:
                n_skip = len(np.load(images_path, mmap_mode='r'))
                images_processed += n_skip
                overall_pbar.update(n_skip)
                print(f"Fold {fold_idx+1}: skipped ({n_skip} images, already done)")
                continue

            images = np.load(images_path, mmap_mode='r')
            masks  = np.load(masks_path,  mmap_mode='r')
            n_images = len(images)
            start_i = resume_image_idx if fold_idx == resume_fold_idx else 0

            print(f"Fold {fold_idx+1}: {n_images} images "
                  f"(starting from {start_i})")

            fold_pbar = tqdm(
                total=n_images,
                initial=start_i,
                desc=f"Fold {fold_idx+1}",
                unit="img",
                dynamic_ncols=True,
                leave=False,
            )

            dataset_full = False
            for batch_start in range(start_i, n_images, feature_batch_size):
                if dataset_full:
                    break

                batch_end = min(batch_start + feature_batch_size, n_images)
                batch_range = range(batch_start, batch_end)

                batch_norm = []
                batch_spaces = []
                batch_labels = []

                for i in batch_range:
                    last_fold_idx = fold_idx
                    last_image_idx = i

                    img_norm = normalizer.transform(
                        np.clip(images[i], 0, 255).astype(np.uint8)
                    )
                    spaces = convert_image(img_norm)

                    if label_mode == 'binary':
                        labels = generate_binary_labels(masks[i])
                    else:
                        labels = generate_3class_labels(masks[i])

                    batch_norm.append(img_norm)
                    batch_spaces.append(spaces)
                    batch_labels.append(labels)

                if use_gpu_features:
                    try:
                        dense_features_list = extract_features_batch_gpu(
                            np.stack(batch_norm, axis=0),
                            batch_size=feature_batch_size,
                            log_progress=False,
                        )
                    except Exception as gpu_exc:
                        print(
                            f"GPU dense extraction failed ({type(gpu_exc).__name__}: "
                            f"{gpu_exc}). Falling back to CPU dense extraction."
                        )
                        use_gpu_features = False
                        dense_features_list = [
                            extract_dense_features_cpu(spaces)
                            for spaces in batch_spaces
                        ]
                else:
                    dense_features_list = [
                        extract_dense_features_cpu(spaces)
                        for spaces in batch_spaces
                    ]

                for local_idx, i in enumerate(batch_range):
                    dense_features = dense_features_list[local_idx]
                    spaces = batch_spaces[local_idx]
                    labels = batch_labels[local_idx]

                    # Sample pixels
                    indices, sampled_labels = active_boundary_mining(
                        labels, n_per_class=n_per_class
                    )
                    n_sampled = len(indices)

                    if n_sampled == 0:
                        del dense_features, spaces, labels, indices, sampled_labels
                        images_processed += 1
                        fold_pbar.update(1)
                        overall_pbar.update(1)
                        if images_processed % 10 == 0:
                            telemetry = _runtime_telemetry(start_time, images_processed, write_ptr)
                            fold_pbar.set_postfix_str(telemetry)
                            overall_pbar.set_postfix_str(telemetry)
                        if (i + 1) % 25 == 0:
                            _cleanup_runtime(trim_heap=False)
                        continue

                    # GLCM per sampled pixel (6)
                    gray = spaces['lab'][:, :, 0].astype(np.float32)
                    glcm_feats = compute_glcm_for_samples(gray, indices)

                    # Assemble: 87 dense + 6 GLCM = 93
                    pixel_rows = np.concatenate([
                        dense_features[indices],  # (n, 87)
                        glcm_feats               # (n, 6)
                    ], axis=1)                   # (n, 93)

                    # Write to memmap
                    end_ptr = write_ptr + n_sampled
                    if end_ptr > total_allocated:
                        # Safety: don't write past allocation
                        n_sampled = total_allocated - write_ptr
                        if n_sampled <= 0:
                            dataset_full = True
                            break
                        pixel_rows = pixel_rows[:n_sampled]
                        sampled_labels = sampled_labels[:n_sampled]
                        end_ptr = write_ptr + n_sampled

                    X_mm[write_ptr:end_ptr] = pixel_rows
                    y_mm[write_ptr:end_ptr] = sampled_labels
                    write_ptr = end_ptr

                    # Cleanup
                    del dense_features, spaces, labels
                    del indices, sampled_labels, gray, glcm_feats, pixel_rows
                    images_processed += 1
                    fold_pbar.update(1)
                    overall_pbar.update(1)
                    if images_processed % 10 == 0:
                        telemetry = _runtime_telemetry(start_time, images_processed, write_ptr)
                        fold_pbar.set_postfix_str(telemetry)
                        overall_pbar.set_postfix_str(telemetry)
                    if (i + 1) % 25 == 0:
                        _cleanup_runtime(trim_heap=False)

                    # ── Checkpoint ────────────────────────────────────────
                    if (i + 1) % checkpoint_every == 0:
                        # Flush FIRST, checkpoint SECOND (see ram-issues.txt)
                        X_mm.flush()
                        y_mm.flush()

                        save_checkpoint(
                            checkpoint_path,
                            fold_dir_idx=fold_idx,
                            image_idx=i,
                            write_ptr=write_ptr,
                            total_allocated=total_allocated,
                            label_mode=label_mode,
                            n_per_class=n_per_class,
                        )

                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = images_processed / max(elapsed, 1)
                        remaining = sum(
                            len(np.load(fp, mmap_mode='r'))
                            for fp, _ in fold_dirs
                        ) - images_processed
                        eta_h = (remaining / max(rate, 0.01)) / 3600

                        print(
                            f"  [{fold_idx+1}/{len(fold_dirs)}] "
                            f"img {i+1}/{n_images} | "
                            f"rows: {write_ptr:,} | "
                            f"{rate:.1f} img/s | "
                            f"ETA: {eta_h:.1f}h | "
                            f"{_runtime_telemetry(start_time, images_processed, write_ptr)}"
                        )
                        _cleanup_runtime(trim_heap=True)

                del dense_features_list, batch_norm, batch_spaces, batch_labels

            if dataset_full:
                print("Reached pre-allocated sample capacity; stopping extraction early.")
                del images, masks
                fold_pbar.close()
                _cleanup_runtime(trim_heap=True)
                break

            # Reset resume pointer for next fold
            resume_image_idx = 0
            del images, masks
            fold_pbar.close()
            _cleanup_runtime(trim_heap=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Flushing memmaps and saving checkpoint...")
        if fold_pbar is not None:
            fold_pbar.close()
        overall_pbar.close()
        X_mm.flush()
        y_mm.flush()
        save_checkpoint(
            checkpoint_path,
            fold_dir_idx=last_fold_idx,
            image_idx=last_image_idx,
            write_ptr=write_ptr,
            total_allocated=total_allocated,
            label_mode=label_mode,
            n_per_class=n_per_class,
        )
        _cleanup_runtime(trim_heap=True)
        raise
    except Exception:
        print("\nError during dataset build. Flushing memmaps and saving checkpoint...")
        if fold_pbar is not None:
            fold_pbar.close()
        overall_pbar.close()
        X_mm.flush()
        y_mm.flush()
        save_checkpoint(
            checkpoint_path,
            fold_dir_idx=last_fold_idx,
            image_idx=last_image_idx,
            write_ptr=write_ptr,
            total_allocated=total_allocated,
            label_mode=label_mode,
            n_per_class=n_per_class,
        )
        _cleanup_runtime(trim_heap=True)
        raise

    overall_pbar.close()

    # ── Final flush ───────────────────────────────────────────────────
    X_mm.flush()
    y_mm.flush()

    actual_rows = write_ptr

    # Truncate if over-allocated
    if actual_rows < total_allocated:
        print(f"\nTruncating {total_allocated:,} → {actual_rows:,} rows")
        X_final = np.array(X_mm[:actual_rows])
        y_final = np.array(y_mm[:actual_rows])
        np.save(output_X_path, X_final)
        np.save(output_y_path, y_final)
        del X_final, y_final

    # Delete checkpoint on success
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint deleted — run completed successfully")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'═'*60}")
    print(f"Done. {label_mode} dataset: "
          f"X=({actual_rows:,}, {TOTAL_FEATURES}) | "
          f"y=({actual_rows:,},)")
    print(f"Time: {elapsed/3600:.1f}h | "
          f"Rate: {images_processed/max(elapsed,1):.1f} img/s")
    print(f"Saved: {output_X_path}")
    print(f"       {output_y_path}")

    return actual_rows
