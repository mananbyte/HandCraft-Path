"""
Dynamic memory configuration and resource detection for PanNuke pipeline.

Detects available CPU RAM and GPU VRAM, auto-calibrates per-image memory usage,
and computes optimal batch sizes for efficient processing.

Features:
  - Auto-detects CPU RAM + GPU VRAM (multi-GPU support)
  - Auto-calibrates actual memory footprint from first 5 images
  - Supports user overrides: --max-memory, --memory-percent, --feature-batch-size
  - Applies configurable safety margin (default: 25%)
  - Fallback to CPU if GPU unavailable
  - Detailed logging of memory decisions
"""

import os
import sys
import numpy as np
from datetime import datetime


try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False


class MemoryConfig:
    """
    Automatically detect and configure memory allocation for dataset pipeline.
    
    Parameters
    ----------
    max_memory_gb : float, optional
        Explicit maximum memory limit (GB). Overrides auto-detection.
    memory_percent : float, optional
        Use only this percentage of available memory (0-100).
    safety_margin_pct : float, default=25
        Safety margin as percentage of detected memory (prevents OOM).
    force_backend : str, default='auto'
        'auto' → GPU if available, else CPU
        'gpu' → force GPU (fail if unavailable)
        'cpu' → force CPU-only
    per_image_estimate_mb : float, default=60
        Conservative per-image estimate (MB) for pre-calibration.
        Used only if auto-calibration is skipped.
    min_batch_size : int, default=5
        Minimum batch size (prevents oversplitting).
    max_batch_size : int, default=500
        Maximum batch size (prevents memory explosion).
    verbose : bool, default=True
        Print diagnostics and decisions.
    """
    
    def __init__(
        self,
        max_memory_gb=None,
        memory_percent=None,
        safety_margin_pct=25,
        force_backend='auto',
        per_image_estimate_mb=60,
        min_batch_size=5,
        max_batch_size=500,
        verbose=True,
    ):
        self.max_memory_gb = max_memory_gb
        self.memory_percent = memory_percent
        self.safety_margin_pct = safety_margin_pct
        self.force_backend = force_backend.lower()
        self.per_image_estimate_mb = per_image_estimate_mb
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        
        # Computed values
        self.cpu_ram_gb = None
        self.gpu_vram_gb = None
        self.gpu_device_count = 0
        self.backend = None
        self.usable_memory_gb = None
        self.actual_per_image_mb = None
        self.optimal_batch_size = None
        self._calibrated = False
        
        # Run detection
        self._detect_resources()
    
    # ── Resource Detection ───────────────────────────────────────────────────
    
    def _detect_resources(self):
        """Detect available CPU RAM and GPU VRAM."""
        if self.verbose:
            print("\n" + "="*72)
            print("MEMORY CONFIGURATION: Detecting available resources")
            print("="*72)
        
        # Detect CPU RAM
        if _HAS_PSUTIL:
            try:
                ram_bytes = psutil.virtual_memory().available
                self.cpu_ram_gb = ram_bytes / 1e9
                if self.verbose:
                    print(f"✓ CPU RAM available: {self.cpu_ram_gb:.2f} GB")
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Could not detect CPU RAM: {e}")
                self.cpu_ram_gb = None
        else:
            if self.verbose:
                print("⚠ psutil not available; CPU RAM auto-detection disabled")
            self.cpu_ram_gb = None
        
        # Detect GPU VRAM
        self.gpu_vram_gb = self._detect_gpu_memory()
        
        # Choose backend
        self._select_backend()
        
        # Calculate usable memory
        self._calculate_usable_memory()
        
        if self.verbose:
            print(f"✓ Backend selected: {self.backend.upper()}")
            print(f"✓ Usable memory: {self.usable_memory_gb:.2f} GB "
                  f"(safety margin: {self.safety_margin_pct}%)")
            print()
    
    def _detect_gpu_memory(self):
        """Detect GPU VRAM from all visible CUDA devices."""
        if not _HAS_CUPY:
            if self.verbose:
                print("⚠ CuPy not available; GPU detection disabled")
            return 0.0
        
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            self.gpu_device_count = device_count
            
            if device_count == 0:
                if self.verbose:
                    print("⚠ No CUDA devices detected")
                return 0.0
            
            total_vram_gb = 0.0
            for dev_id in range(device_count):
                try:
                    with cp.cuda.Device(dev_id):
                        free_b, total_b = cp.cuda.runtime.memGetInfo()
                        vram_gb = total_b / 1e9
                        total_vram_gb += vram_gb
                        if self.verbose:
                            print(f"✓ GPU {dev_id} VRAM: {vram_gb:.2f} GB "
                                  f"(free: {free_b/1e9:.2f} GB)")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠ Could not query GPU {dev_id}: {e}")
            
            return total_vram_gb
        
        except Exception as e:
            if self.verbose:
                print(f"⚠ GPU memory detection failed: {e}")
            return 0.0
    
    def _select_backend(self):
        """Select processing backend (GPU or CPU)."""
        if self.force_backend == 'gpu':
            if self.gpu_vram_gb > 0:
                self.backend = 'gpu'
            else:
                raise RuntimeError(
                    "Backend 'gpu' requested but no GPU available. "
                    "Install CuPy or use --feature-backend auto/cpu"
                )
        elif self.force_backend == 'cpu':
            self.backend = 'cpu'
        else:  # auto
            if self.gpu_vram_gb > 0:
                self.backend = 'gpu'
            elif self.cpu_ram_gb is not None:
                self.backend = 'cpu'
            else:
                raise RuntimeError(
                    "No memory detection available. "
                    "Install psutil or CuPy, or manually set --max-memory"
                )
    
    def _calculate_usable_memory(self):
        """Calculate usable memory based on backend and user overrides."""
        if self.max_memory_gb is not None:
            # Explicit user limit
            detected_memory = self.max_memory_gb
            source = "user --max-memory"
        elif self.backend == 'gpu':
            detected_memory = self.gpu_vram_gb
            source = f"GPU VRAM (sum of {self.gpu_device_count} device(s))"
        else:
            detected_memory = self.cpu_ram_gb
            source = "system RAM"
        
        # Apply safety margin
        margin_fraction = self.safety_margin_pct / 100.0
        self.usable_memory_gb = detected_memory * (1.0 - margin_fraction)
        
        # Apply user percentage if provided
        if self.memory_percent is not None:
            self.usable_memory_gb *= (self.memory_percent / 100.0)
            source += f" × {self.memory_percent}%"
        
        if self.verbose:
            print(f"  Detected memory ({source}): {detected_memory:.2f} GB")
            print(f"  After {self.safety_margin_pct}% safety margin: "
                  f"{self.usable_memory_gb:.2f} GB")
    
    # ── Auto-Calibration ────────────────────────────────────────────────────
    
    def calibrate_from_images(self, images_paths_and_masks, n_calibrate_images=5):
        """
        Auto-calibrate per-image memory by processing first N images.
        
        This is the most accurate way to determine actual memory footprint
        including all preprocessing, feature extraction, and sampling overhead.
        
        Parameters
        ----------
        images_paths_and_masks : list of (images_path_str, masks_path_str)
            Fold directory tuples from build_dataset.py
        n_calibrate_images : int, default=5
            Number of images to process for calibration
        
        Returns
        -------
        actual_per_image_mb : float
            Measured peak memory per image (MB)
        """
        if not _HAS_PSUTIL:
            if self.verbose:
                print("⚠ psutil not available; skipping auto-calibration")
            self.actual_per_image_mb = self.per_image_estimate_mb
            self._calibrated = False
            return self.actual_per_image_mb
        
        if self.verbose:
            print("\n" + "="*72)
            print("MEMORY AUTO-CALIBRATION: Processing first 5 images")
            print("="*72)
        
        try:
            import numpy as np
            from src.data.label_generator import generate_binary_labels
            from src.preprocessing.color_converter import convert_image
            from src.features.pixel_feature_extractor import (
                extract_dense_features_cpu, TOTAL_FEATURES
            )
            from src.sampling.pixel_sampler import active_boundary_mining
        except Exception as e:
            if self.verbose:
                print(f"⚠ Could not import pipeline modules: {e}")
            self.actual_per_image_mb = self.per_image_estimate_mb
            self._calibrated = False
            return self.actual_per_image_mb
        
        peak_memory_mb = 0.0
        process = psutil.Process(os.getpid())
        baseline_mb = process.memory_info().rss / 1e6
        
        try:
            images_processed = 0
            for img_path, mask_path in images_paths_and_masks:
                images = np.load(img_path, mmap_mode='r')
                masks = np.load(mask_path, mmap_mode='r')
                
                for i in range(min(n_calibrate_images - images_processed, len(images))):
                    # Simulate pipeline: normalize → convert → label → extract → sample
                    img = np.clip(images[i], 0, 255).astype(np.uint8)
                    
                    # Simple stain pass (without full Macenko to save time)
                    # In actual pipeline, normalizer would be used
                    normalized = img.astype(np.float32)
                    
                    spaces = convert_image(normalized.astype(np.uint8))
                    labels = generate_binary_labels(masks[i])
                    dense_feats = extract_dense_features_cpu(spaces)
                    indices, _ = active_boundary_mining(labels, n_per_class=400)
                    
                    # Measure memory after processing
                    current_mb = process.memory_info().rss / 1e6
                    used_mb = current_mb - baseline_mb
                    peak_memory_mb = max(peak_memory_mb, used_mb)
                    
                    # Cleanup
                    del img, normalized, spaces, labels, dense_feats, indices
                    
                    images_processed += 1
                    if images_processed >= n_calibrate_images:
                        break
                
                if images_processed >= n_calibrate_images:
                    break
        
        except Exception as e:
            if self.verbose:
                print(f"⚠ Calibration failed: {e}. Using estimate.")
            self.actual_per_image_mb = self.per_image_estimate_mb
            self._calibrated = False
            return self.actual_per_image_mb
        
        # Average over calibration images
        self.actual_per_image_mb = max(peak_memory_mb, 30)  # min 30 MB safety
        self._calibrated = True
        
        if self.verbose:
            print(f"✓ Calibration complete: {self.actual_per_image_mb:.1f} MB/image "
                  f"(measured from {images_processed} images)")
            print()
        
        return self.actual_per_image_mb
    
    # ── Batch Size Calculation ──────────────────────────────────────────────
    
    def compute_batch_size(self, feature_batch_size_override=None):
        """
        Compute optimal batch size based on available memory.
        
        Parameters
        ----------
        feature_batch_size_override : int, optional
            Explicit batch size override (highest priority)
        
        Returns
        -------
        batch_size : int
            Recommended batch size
        """
        if feature_batch_size_override is not None:
            self.optimal_batch_size = feature_batch_size_override
            if self.verbose:
                print(f"✓ Using explicit batch size: {self.optimal_batch_size} images")
            return self.optimal_batch_size
        
        # Use actual per-image memory if calibrated, else estimate
        per_image_mb = (
            self.actual_per_image_mb 
            if self._calibrated 
            else self.per_image_estimate_mb
        )
        
        usable_memory_mb = self.usable_memory_gb * 1000
        batch_size = int(usable_memory_mb / per_image_mb)
        batch_size = max(self.min_batch_size, min(batch_size, self.max_batch_size))
        
        self.optimal_batch_size = batch_size
        
        if self.verbose:
            calibration_note = " (measured)" if self._calibrated else " (estimated)"
            print(f"✓ Computed batch size: {self.optimal_batch_size} images")
            print(f"  ({usable_memory_mb:.0f} MB ÷ {per_image_mb:.1f} MB/image{calibration_note})")
        
        return self.optimal_batch_size
    
    # ── Public Interface ────────────────────────────────────────────────────
    
    def get_batch_size(self):
        """Return computed batch size (or compute if not done yet)."""
        if self.optimal_batch_size is None:
            self.compute_batch_size()
        return self.optimal_batch_size
    
    def get_backend(self):
        """Return selected backend ('gpu' or 'cpu')."""
        return self.backend
    
    def get_memory_summary(self):
        """
        Return human-readable summary of memory configuration.
        Useful for logging at start of pipeline.
        """
        calibration_note = " (auto-calibrated)" if self._calibrated else " (estimated)"
        
        lines = [
            "",
            "="*72,
            "MEMORY CONFIGURATION SUMMARY",
            "="*72,
            f"Backend:               {self.backend.upper()}",
            f"Total usable memory:   {self.usable_memory_gb:.2f} GB",
            f"Per-image footprint:   {self.actual_per_image_mb or self.per_image_estimate_mb:.1f} MB{calibration_note}",
            f"Batch size:            {self.optimal_batch_size or 'not computed'} images",
            f"Safety margin:         {self.safety_margin_pct}%",
            f"GPU devices:           {self.gpu_device_count}",
            "="*72,
            "",
        ]
        
        if self._calibrated:
            est_total_mb = (self.actual_per_image_mb or 0) * (self.optimal_batch_size or 0)
            lines.insert(-2, f"Est. batch memory use: ~{est_total_mb/1000:.1f} GB")
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print memory configuration summary to stdout."""
        print(self.get_memory_summary())


# ── Helper function for CLI ─────────────────────────────────────────────────

def create_memory_config_from_args(args):
    """
    Create MemoryConfig instance from argparse arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    
    Returns
    -------
    MemoryConfig instance
    """
    return MemoryConfig(
        max_memory_gb=args.max_memory if hasattr(args, 'max_memory') and args.max_memory else None,
        memory_percent=args.memory_percent if hasattr(args, 'memory_percent') and args.memory_percent else None,
        safety_margin_pct=args.memory_safety_margin if hasattr(args, 'memory_safety_margin') else 25,
        force_backend=args.memory_backend if hasattr(args, 'memory_backend') else 'auto',
        per_image_estimate_mb=60,
        verbose=True,
    )
