"""
Macenko stain normalization — zero external dependencies.

Self-contained implementation using only numpy + skimage.
No staintools, no spams required.

Reference:
    Macenko et al., "A method for normalizing histology slides for
    quantitative analysis", ISBI 2009.
"""

import numpy as np
import joblib


class MacenkoNormalizer:
    """
    Macenko SVD-based stain normalization for H&E images.
    Works with any NumPy version (1.x or 2.x).
    """

    BETA = 0.15  # OD threshold for tissue pixels

    def __init__(self):
        self.stain_matrix_target = None
        self.maxC_target = None
        self.reference_image = None
        self.reference_index = None
        self.is_fitted = False

    # ── colour conversions ────────────────────────────────────────────────────

    @staticmethod
    def _to_uint8(image):
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _rgb_to_od(img_uint8):
        """RGB uint8 → optical density. Always returns finite values."""
        img = np.clip(img_uint8.astype(np.float64), 1.0, 255.0)
        return -np.log(img / 255.0)

    @staticmethod
    def _od_to_rgb(od):
        """Optical density → RGB uint8."""
        return np.clip(255.0 * np.exp(-od), 0, 255).astype(np.uint8)

    # ── luminosity standardisation ────────────────────────────────────────────

    @staticmethod
    def _standardize_luminosity(img_uint8):
        """
        Standardize brightness to match staintools behavior exactly.
        img_uint8: (H, W, 3)
        """
        I = img_uint8.astype(np.float64)
        # Convert to optical density space to find tissue
        OD = -np.log(np.clip(I / 255.0, 1e-6, 1.0))
        tissue_mask = np.any(OD > 0.15, axis=2)
        
        # If there's enough tissue, standardize the 95th percentile to 255
        if np.sum(tissue_mask) > 10:
            percentile_95 = np.percentile(I[tissue_mask], 95)
            if percentile_95 > 0:
                I = I * (255.0 / percentile_95)
                
        return np.clip(I, 0, 255).astype(np.uint8)

    # ── Macenko core ──────────────────────────────────────────────────────────

    @classmethod
    def _get_stain_matrix(cls, img_uint8):
        """
        Extract 2×3 stain matrix [H; E] via Macenko algorithm.
        """
        OD = cls._rgb_to_od(img_uint8).reshape(-1, 3)

        # Keep tissue pixels: ANY channel above threshold (matches staintools)
        mask = np.any(OD > cls.BETA, axis=1)
        OD_hat = OD[mask]
        if OD_hat.shape[0] < 10:
            raise ValueError(f"Only {OD_hat.shape[0]} tissue pixels — too few")

        # Use covariance matrix eigenvectors for principal components
        cov = np.cov(OD_hat, rowvar=False)
        w, v = np.linalg.eigh(cov)
        
        # Top 2 eigenvectors (eigh returns ascending, so last two are largest)
        top2 = v[:, [2, 1]]  # Shape: (3, 2)
        
        # Ensure vectors point in the positive OD direction
        if top2[0, 0] < 0: top2[:, 0] *= -1
        if top2[0, 1] < 0: top2[:, 1] *= -1

        # Project tissue OD onto the 2D plane
        proj = OD_hat @ top2                     # (M, 2)
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        # Extreme angle directions (robust percentiles)
        min_a = np.percentile(angles, 1)
        max_a = np.percentile(angles, 99)

        v_min = top2 @ np.array([np.cos(min_a), np.sin(min_a)])  # (3,)
        v_max = top2 @ np.array([np.cos(max_a), np.sin(max_a)])  # (3,)

        # Normalize to unit length
        v_min /= np.linalg.norm(v_min)
        v_max /= np.linalg.norm(v_max)

        # Convention: Hematoxylin has a LARGER first OD component than Eosin
        if v_min[0] > v_max[0]:
            return np.array([v_min, v_max])  # H first
        else:
            return np.array([v_max, v_min])  # H first

    @classmethod
    def _get_concentrations(cls, img_uint8, stain_matrix):
        """Solve for per-pixel stain concentrations."""
        OD = cls._rgb_to_od(img_uint8).reshape(-1, 3)
        # Solve C * stain_matrix = OD  --> C = OD * pinv(stain_matrix)
        C, _, _, _ = np.linalg.lstsq(stain_matrix.T, OD.T, rcond=None)
        
        # Enforce non-negativity (prevents black background artifacts)
        C = np.maximum(C.T, 0)
        return C  # (N_pixels, 2)

    # ── tissue quality ────────────────────────────────────────────────────────

    @staticmethod
    def _tissue_fraction(img_uint8, white_thresh=245):
        return (img_uint8 < white_thresh).any(axis=2).mean()

    # ── fit ────────────────────────────────────────────────────────────────────

    def select_reference_image(self, images):
        """Rank images by closeness to median OD, preferring tissue-rich."""
        print(f"Selecting reference from {len(images)} candidates...")
        od_means, tissue_fracs = [], []
        for img in images:
            u8 = self._to_uint8(img)
            od_means.append(self._rgb_to_od(u8).mean())
            tissue_fracs.append(self._tissue_fraction(u8))

        od_means = np.array(od_means)
        tissue_fracs = np.array(tissue_fracs)
        median_od = np.median(od_means)

        order = list(np.argsort(np.abs(od_means - median_od)))
        good = [i for i in order if tissue_fracs[i] >= 0.10]
        return (good if good else order), od_means, tissue_fracs

    def fit(self, images):
        """Fit normalizer on the best available reference image."""
        order, od_means, tissue_fracs = self.select_reference_image(images)

        last_err = None
        for idx in order[:30]:
            cand = self._standardize_luminosity(self._to_uint8(images[idx]))
            try:
                sm = self._get_stain_matrix(cand)
                C  = self._get_concentrations(cand, sm)
                maxC = np.percentile(C, 99, axis=0).reshape(1, 2)

                self.stain_matrix_target = sm
                self.maxC_target = maxC
                self.reference_image = cand
                self.reference_index = idx
                self.is_fitted = True
                print(f"Fit OK — image {idx}  "
                      f"(OD={od_means[idx]:.4f}, tissue={tissue_fracs[idx]:.3f})")
                return self
            except Exception as e:
                print(f"  skip {idx}: {e}")
                last_err = e

        raise RuntimeError(f"fit() failed. Last error: {last_err}") from last_err

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, image):
        """Normalize one image to match the reference stain appearance."""
        assert self.is_fitted, "Call fit() first"
        img = self._standardize_luminosity(self._to_uint8(image))

        try:
            sm_src = self._get_stain_matrix(img)
            C_src  = self._get_concentrations(img, sm_src)
            maxC_src = np.percentile(C_src, 99, axis=0).reshape(1, 2)

            # Scale source concentrations to match target
            C_src *= (self.maxC_target / (maxC_src + 1e-6))

            # Reconstruct with target stain matrix
            od_norm = C_src @ self.stain_matrix_target
            return self._od_to_rgb(od_norm.reshape(img.shape))
        except Exception:
            return img  # fallback: luminosity-standardised original

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path):
        joblib.dump(self, path)
        print(f"Saved to {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)
