"""
Feature scaler module for HandCraft-Path pixel classification.

Default scaler: RobustScaler (IQR-based).
Biological rationale: H&E stain-artefact spikes in optical density channels
create heavy tails that shift z-score statistics. RobustScaler uses the median
and IQR, making it insensitive to these outliers without discarding them.

Fits on a stratified subsample (sample_n=200_000) to maintain <500 MB RSS.
Transforms full memmap matrices in streaming chunk_size=50_000 row batches.
"""

import os
import json
import logging
from typing import Optional

import joblib
import numpy as np
import psutil
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.safe_loader import safe_load_npy

# ── Module-level constants ─────────────────────────────────────────────────
DEFAULT_SCALER_TYPE: str = "RobustScaler"
DEFAULT_SAMPLE_N: int = 200_000
DEFAULT_CHUNK_SIZE: int = 50_000
RANDOM_STATE: int = 42

# Map string keys → sklearn classes (keeps config readable in CLI args)
SCALER_REGISTRY: dict = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "QuantileTransformer_uniform": lambda: QuantileTransformer(
        output_distribution="uniform", random_state=RANDOM_STATE, n_quantiles=1000
    ),
    "QuantileTransformer_normal": lambda: QuantileTransformer(
        output_distribution="normal", random_state=RANDOM_STATE, n_quantiles=1000
    ),
    "PowerTransformer": PowerTransformer,  # Yeo-Johnson by default; handles negatives
}

logger = logging.getLogger(__name__)


# ── FeatureScaler ─────────────────────────────────────────────────────────
class FeatureScaler:
    """
    Production pixel-feature scaler for HandCraft-Path.

    Memory-safe: fits on a small stratified subsample, then streams
    the full matrix through in fixed-size chunks using numpy.memmap slicing.

    Parameters
    ----------
    scaler_type : str
        Key from SCALER_REGISTRY. Defaults to 'RobustScaler'.
    """

    def __init__(self, scaler_type: str = DEFAULT_SCALER_TYPE) -> None:
        assert scaler_type in SCALER_REGISTRY, (
            f"Unknown scaler '{scaler_type}'. "
            f"Choose from: {list(SCALER_REGISTRY.keys())}"
        )
        self.scaler_type: str = scaler_type
        factory = SCALER_REGISTRY[scaler_type]
        # callable factory (lambda) vs class — both work with ()
        self._scaler = factory() if callable(factory) else factory()
        self._fitted: bool = False

    # ── Fit ───────────────────────────────────────────────────────────────
    def fit(
        self,
        X_path: str,
        y_path: str,
        sample_n: int = DEFAULT_SAMPLE_N,
        random_state: int = RANDOM_STATE,
    ) -> "FeatureScaler":
        """
        Fits the scaler on a stratified subsample to stay ≤500 MB RSS.

        Parameters
        ----------
        X_path : str
            Path to the full feature memmap (.npy), shape (N, 93).
        y_path : str
            Path to the label memmap (.npy), shape (N,).
        sample_n : int
            Number of rows to subsample for fitting (default 200_000).
        random_state : int
            Reproducibility seed.

        Returns
        -------
        self
        """
        rss_before = _rss_mb()
        logger.info(
            "FeatureScaler.fit() — RSS before load: %.1f MB", rss_before
        )

        X_mm = safe_load_npy(X_path, mode="r")
        y_mm = safe_load_npy(y_path, mode="r")
        n_total = X_mm.shape[0]

        assert X_mm.ndim == 2, f"Expected 2-D X array, got shape {X_mm.shape}"
        assert y_mm.ndim == 1, f"Expected 1-D y array, got shape {y_mm.shape}"
        assert len(y_mm) == n_total, "X and y row counts must match"

        # Draw a stratified subsample — avoids loading the whole matrix
        actual_n = min(sample_n, n_total)
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=actual_n / n_total, random_state=random_state
        )
        _, sample_idx = next(sss.split(np.zeros(n_total), y_mm))

        # Load only the subsample rows into RAM
        X_sample = np.array(X_mm[np.sort(sample_idx)], dtype=np.float32)
        X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=0.0, neginf=0.0)

        rss_after_load = _rss_mb()
        logger.info(
            "Loaded %d subsample rows — RSS: %.1f MB (+%.1f MB)",
            actual_n,
            rss_after_load,
            rss_after_load - rss_before,
        )

        self._scaler.fit(X_sample)
        self._fitted = True

        logger.info(
            "FeatureScaler.fit() complete — scaler=%s, RSS: %.1f MB",
            self.scaler_type,
            _rss_mb(),
        )
        return self

    # ── Transform memmap in chunks ─────────────────────────────────────────
    def transform_memmap(
        self,
        X_path: str,
        out_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """
        Streams the full feature matrix through the fitted scaler, writing
        a new float32 memmap at out_path. Never loads the full matrix at once.

        Parameters
        ----------
        X_path : str
            Path to raw feature memmap (.npy), shape (N, 93).
        out_path : str
            Output path for scaled memmap (.npy), same shape.
        chunk_size : int
            Rows per streaming chunk. Default 50_000 ≈ 18 MB per chunk.
        """
        assert self._fitted, "Call fit() before transform_memmap()"

        X_mm = safe_load_npy(X_path, mode="r")
        n_total, n_features = X_mm.shape

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # Pre-allocate output memmap on disk (no RAM allocation for full matrix)
        X_out = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32, shape=(n_total, n_features)
        )

        n_chunks = (n_total + chunk_size - 1) // chunk_size
        rss_peak = _rss_mb()

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_total)
            chunk = np.array(X_mm[start:end], dtype=np.float32)
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            X_out[start:end] = self._scaler.transform(chunk).astype(np.float32)
            rss_now = _rss_mb()
            if rss_now > rss_peak:
                rss_peak = rss_now

        X_out.flush()
        logger.info(
            "transform_memmap complete: %s → %s | rows=%d | peak RSS=%.1f MB",
            X_path,
            out_path,
            n_total,
            rss_peak,
        )
        print(f"  ✓ Scaled matrix written: {out_path} (peak RSS {rss_peak:.1f} MB)")

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Serialises the fitted scaler object to disk via joblib."""
        assert self._fitted, "Scaler must be fitted before saving."
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        meta = {"scaler_type": self.scaler_type}
        joblib.dump({"scaler": self._scaler, "meta": meta}, path, compress=3)
        logger.info("FeatureScaler saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Loads a persisted FeatureScaler from disk."""
        payload = joblib.load(path)
        meta = payload["meta"]
        instance = cls.__new__(cls)
        instance.scaler_type = meta["scaler_type"]
        instance._scaler = payload["scaler"]
        instance._fitted = True
        logger.info("FeatureScaler loaded from %s (type=%s)", path, instance.scaler_type)
        return instance

    # ── Convenience: spot-check a scaled matrix ───────────────────────────
    def spot_check(self, scaled_path: str, n_rows: int = 5_000) -> dict:
        """
        Loads a small slice of a scaled matrix and returns basic statistics.
        Used for post-hoc sanity assertions in the production script.
        """
        X_mm = safe_load_npy(scaled_path, mode="r")
        idx = np.random.default_rng(RANDOM_STATE).integers(0, X_mm.shape[0], size=n_rows)
        sample = np.array(X_mm[np.sort(idx)], dtype=np.float32)
        return {
            "mean_abs_col_mean": float(np.abs(sample.mean(axis=0)).mean()),
            "mean_abs_col_median": float(np.abs(np.median(sample, axis=0)).mean()),
            "max_abs_col_mean": float(np.abs(sample.mean(axis=0)).max()),
            "outlier_frac_3sigma": float(
                np.mean(np.abs(sample) > 3.0)
            ),
        }

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"FeatureScaler(type={self.scaler_type}, status={status})"


# ── Internal helpers ──────────────────────────────────────────────────────
def _rss_mb() -> float:
    """Returns current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def get_all_scaler_types() -> list:
    """Returns list of all available scaler type keys."""
    return list(SCALER_REGISTRY.keys())
