# PanNuke Classical ML Segmentation — Complete Implementation Guide

**Project:** LedgerLand-adjacent research · Nucleus segmentation via hand-crafted features  
**Scope:** Stage 1 semantic segmentation (binary + 3-class) → Stage 2 nucleus type classification  
**Target:** 3–4 days, GPU machine with RAPIDS 25.12, CUDA 12.x (12.5+ recommended)

---

## How to use this guide

Every section follows this pattern:

- **What** — the exact deliverable for this step
- **Why** — the scientific or engineering rationale
- **Impact if skipped** — what breaks downstream
- **Instructions** — exact code and commands
- **Verify** — how to confirm it worked before moving on

Do not proceed past any step without completing its **Verify** check. A broken step silently corrupts everything downstream.

---

## Part 0 — Project understanding (read before touching code)

### 0.1 What you are building

You are building two sequential, independent machine learning models:

**Stage 1 — Pixel classifier (segmentation)**  
Input: a 256×256 RGB H&E image patch  
Output: a 256×256 integer mask where each pixel is labeled  
- Experiment A: `0=background, 1=nucleus`  
- Experiment B: `0=background, 1=nucleus interior, 2=nucleus boundary`

**Stage 2 — Region classifier (nucleus typing)**  
Input: a single segmented nucleus region (binary mask + image crop)  
Output: one integer label `0=neoplastic, 1=inflammatory, 2=connective, 3=dead, 4=epithelial`

These two models are trained on different labels, with different feature sets, evaluated with different metrics. They share only the image data and the preprocessing pipeline.

### 0.2 What the PanNuke dataset contains

- 7,904 image patches, each 256×256 RGB
- 19 tissue types
- `images.npy` — shape `(N, 256, 256, 3)`, dtype float32, values 0–255
- `masks.npy` — shape `(N, 256, 256, 6)`, dtype float32
  - Channels 0–4: one instance per nucleus type (neoplastic, inflammatory, connective, dead, epithelial)
  - Channel 5: background
  - Each channel contains unique integer IDs per nucleus instance, 0 elsewhere
- `types.npy` — shape `(N,)`, tissue type string per image
- Split into Fold1, Fold2, Fold3 — use Fold1+2 for training, Fold3 for test throughout

### 0.3 Research questions this project answers

1. How well can hand-crafted features match CNN-based semantic segmentation on H&E pathology images?
2. Does adding an explicit boundary class (Exp B) improve segmentation quality versus binary (Exp A)?
3. Does better segmentation (Exp A vs B) improve downstream nucleus type classification accuracy?
4. What is the performance gap between oracle-fed Stage 2 (perfect segmentation input) versus pipeline-fed Stage 2 (real Stage 1 output)?

### 0.4 Evaluation metrics — fixed, do not change

| Stage | Metric | Why |
|---|---|---|
| Stage 1 Exp A | Binary Dice, binary IoU | Standard binary segmentation metrics |
| Stage 1 Exp B | Per-class Dice (3 classes), binary Dice after collapsing 1+2→nucleus | Measures boundary quality separately |
| Stage 2 (oracle) | Per-class F1, macro-F1, confusion matrix | Oracle = ground truth regions fed to classifier |
| Stage 2 (pipeline) | Same metrics | Real Stage 1 output fed to classifier |
| End-to-end | Oracle gap = pipeline macro-F1 − oracle macro-F1 | Measures how much Stage 1 errors hurt Stage 2 |

### 0.5 GPU acceleration strategy

**What:** This project uses NVIDIA RAPIDS 25.12 to GPU-accelerate all compute-heavy operations. Understanding which operations run on GPU vs CPU is essential for debugging and performance tuning.

**GPU-accelerated (RAPIDS / CuPy):**
- **Model training:** cuML RandomForest, cuML SVM, XGBoost (`device='cuda'`), LightGBM (`device='gpu'`)
- **Feature matrices:** CuPy arrays for large batch operations (windowed statistics, gradient maps)
- **Feature selection:** cuDF DataFrames, cuML-based feature importance ranking
- **Prediction:** cuML `.predict()` runs entirely on GPU — no CPU round-trip

**CPU-bound (no GPU equivalent exists):**
- Color space conversion: `skimage` (`rgb2lab`, `rgb2hsv`, `separate_stains`)
- Texture features: `skimage` LBP, Gabor filters, GLCM
- Stain normalization: `staintools` (Macenko SVD)
- Superpixel segmentation: `skimage` SLIC
- Label generation: `skimage` morphology operations

**Design pattern — CPU/GPU boundary:**

Per-image preprocessing (color conversion, stain normalization, texture) runs on CPU because `skimage` has no GPU backend. Feature computation **batches** images to GPU for windowed statistics and gradient features — the heaviest per-image workload. The resulting feature matrices stay on GPU for training and prediction. Transfer between CPU↔GPU happens at well-defined boundaries to minimise overhead.

```python
# Pattern used throughout this project:
import cupy as cp
import numpy as np

# CPU: per-image preprocessing (skimage — no GPU version)
spaces = convert_image(img_uint8)

# GPU: batch windowed statistics (send N images at once)
channels_gpu = cp.asarray(channels_batch)            # CPU → GPU
stats_gpu = cupyx_uniform_filter(channels_gpu, ...)   # GPU compute
stats_cpu = stats_gpu.get()                           # GPU → CPU

# GPU: model training (data stays on GPU end-to-end)
X_gpu = cp.asarray(X_train)
model = cuml.ensemble.RandomForestClassifier(...)
model.fit(X_gpu, y_gpu)   # trains entirely on GPU
preds = model.predict(X_test_gpu)   # predicts on GPU
```

**Batch size selection:** The batch size for GPU feature extraction depends on your VRAM. Each 256×256 float32 image with 9 channels for windowed stats uses ~2.4 MB on GPU. A safe starting point:

| VRAM | Recommended batch size |
|---|---|
| 8 GB | 16–32 images |
| 16 GB | 64–128 images |
| 24 GB+ | 128–256 images |

Monitor with `nvidia-smi` during the first batch. If you see OOM errors, halve the batch size.

---

## Part 1 — Environment setup

### Step 1.1 — Verify hardware and CUDA

**What:** Confirm your GPU is visible and CUDA version is compatible with RAPIDS 25.12.

**Why:** RAPIDS 25.12 requires CUDA 12.x (12.5+ recommended). Running on the wrong CUDA version causes silent import errors or wrong numerical results.

**Instructions:**

```bash
nvidia-smi
# Expected: shows GPU name, CUDA Version: 12.x

python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
# Expected: 12050 or higher (CUDA 12.5+)

# Check available VRAM — determines batch sizes for GPU feature extraction
python -c "import cupy; print(f'VRAM: {cupy.cuda.Device().mem_info[1] / 1e9:.1f} GB')"
```

**Verify:** All commands complete without error. Note your GPU name and VRAM. If VRAM < 16 GB, you will need to reduce batch sizes in later steps (see Section 0.5 table) — note this now.

---

### Step 1.2 — Create pinned environment file

**What:** A reproducible `environment.yml` that captures every dependency with exact versions.

**Why:** RAPIDS packages have strict inter-dependency requirements. An unpinned install will silently install incompatible versions and fail at runtime, not at install time.

**Impact if skipped:** Your environment cannot be reproduced. Collaborators or future you cannot replicate results.

**Instructions:**

Create `environment.yml` in your project root:

```yaml
name: pannuke-ml

channels:
  - rapidsai
  - conda-forge
  - nvidia
  # 'defaults' intentionally omitted — conflicts with rapidsai packages

dependencies:
  # ── Python & CUDA ────────────────────────────────────────────────────────────
  - python=3.11
  - cuda-version=12.*            # any CUDA 12.x; RAPIDS 25.12 validated on 12.5–12.9

  # ── RAPIDS metapackage ───────────────────────────────────────────────────────
  # Installs: cuML 25.12, cuDF 25.12, CuPy 13.x, RMM, Dask-CUDA
  # DO NOT pin numpy/scipy separately — let RAPIDS own these versions
  - rapids=25.12

  # ── Core ML (let conda pick compatible versions with RAPIDS numpy) ───────────
  - scikit-learn>=1.5     # full tag estimator API; RFE + ExtraTrees
  - lightgbm>=4.3         # GPU histogram; Python 3.11 wheels available
  - xgboost>=2.1          # GPU hist backend (3.x bundled by RAPIDS anyway)
  - imbalanced-learn>=0.12 # SMOTE; compatible with sklearn >=1.5
  - optuna>=3.5           # Bayesian HPO with TPE sampler

  # ── Image processing ─────────────────────────────────────────────────────────
  - scikit-image>=0.22
  - opencv>=4.8
  - pillow>=10.0

  # ── Utilities ────────────────────────────────────────────────────────────────
  - matplotlib>=3.8       # debug visualizations and paper figures
  - tqdm                  # progress bars in training loops
  - jupyterlab>=4.0       # notebook environment

  # ── Pip packages (numpy-2.x compatible, no conda equivalent) ─────────────────
  - pip
  - pip:
      # Stain normalization — Macenko SVD method (no SPAMS required)
      - staintools==2.1.2

      # Minimum Redundancy Maximum Relevance feature selection
      - mrmr-selection==0.2.8

      # Dense CRF post-processing (pydensecrf 1.0rc3 broken on Py3.11;
      # maintained fork with identical API)
      - git+https://github.com/lucasb-eyer/pydensecrf.git

      # Optional FastAPI inference server
      # Build this LAST — only after all evaluation is complete
      - fastapi==0.104.1
      - uvicorn==0.24.0
```

```bash
conda env create -f environment.yml --solver=libmamba   # recommended solver
conda activate pannuke-ml
#
#   If libmamba is not installed:
conda install -n base conda-libmamba-solver
conda env create -f environment.yml --solver=libmamba
```

**Verify:**

```python
import sys
from unittest.mock import MagicMock

# Create a fake 'spams' module so staintools doesn't crash on import.
# staintools 2.1.2 unconditionally imports VahadaneStainExtractor which
# requires spams.  We only use Macenko (SVD-based) — no spams needed.
# python-spams requires numpy <2.0, conflicting with RAPIDS 25.12.
sys.modules['spams'] = MagicMock()

import cuml, cudf, cupy
import sklearn, lightgbm, optuna
import skimage, cv2, staintools
print("All imports OK")
print("cuML version:", cuml.__version__)       # should print 25.12.x
print("CuPy CUDA:", cupy.cuda.runtime.runtimeGetVersion())  # 12050+
print("VRAM:", f"{cupy.cuda.Device().mem_info[1]/1e9:.1f} GB")
```

---

### Step 1.3 — Verify data structure

**What:** Confirm PanNuke files are present and shaped correctly before writing any code that depends on them.

**Why:** If masks.npy has the wrong shape or dtype, every label generation function will produce silently wrong results.

**Instructions:**

```python
import numpy as np
import os

base = "data/raw/Fold"

for fold in [1, 2, 3]:
    fold_dir = f"{base}/Fold{fold}/images/fold{fold}"
    images = np.load(f"{fold_dir}/images.npy")
    masks  = np.load(f"{base}/Fold{fold}/masks/fold{fold}/masks.npy")
    types  = np.load(f"{base}/Fold{fold}/images/fold{fold}/types.npy")
    
    print(f"\nFold {fold}:")
    print(f"  images: {images.shape} dtype={images.dtype}")
    print(f"  masks:  {masks.shape}  dtype={masks.dtype}")
    print(f"  types:  {types.shape}")
    
    assert images.shape[1:] == (256, 256, 3), "Wrong image shape"
    assert masks.shape[1:] == (256, 256, 6),  "Wrong mask shape — check channel order"
    assert images.shape[0] == masks.shape[0], "Image/mask count mismatch"
    
    # Confirm channel semantics
    # Channels 0-4 should have sparse non-zero values (nucleus instances)
    # Channel 5 should be densely non-zero (background)
    for c in range(5):
        nonzero_frac = (masks[:5, :, :, c] > 0).mean()
        print(f"  channel {c} nonzero fraction (first 5 imgs): {nonzero_frac:.3f}")
    bg_frac = (masks[:5, :, :, 5] > 0).mean()
    print(f"  channel 5 (background) nonzero fraction: {bg_frac:.3f}")
```

**Verify:** Background channel (5) has a much higher nonzero fraction than nucleus channels (0–4). If channel 5 is sparse, your channel ordering is wrong — swap it and re-read the PanNuke README.

---

## Part 2 — Label generation

This is the most critical section. Wrong labels make everything downstream meaningless. Read every line.

### Step 2.1 — Stage 1 Experiment A labels (binary)

**What:** Generate per-pixel binary masks: `0=background, 1=nucleus`.

**Why:** Experiment A is the simplest possible segmentation formulation. It answers: can hand-crafted features distinguish nucleus from background at all? It is also the baseline against which Experiment B is compared.

**Impact if wrong:** The model will learn the wrong mapping. There is no runtime error — it will train, produce predictions, and give you meaningless numbers.

**Instructions:**

Create `src/data/label_generator.py`:

```python
import numpy as np
from skimage.morphology import binary_erosion, disk


def generate_binary_labels(mask):
    """
    Generate binary segmentation labels from PanNuke instance mask.
    
    Parameters
    ----------
    mask : np.ndarray, shape (256, 256, 6)
        PanNuke mask tensor. Channels 0-4 are nucleus types,
        channel 5 is background. Each channel contains unique
        integer IDs per nucleus instance, 0 elsewhere.
    
    Returns
    -------
    labels : np.ndarray, shape (256, 256), dtype uint8
        0 = background, 1 = nucleus (any type)
    """
    # A pixel is nucleus if ANY of channels 0-4 is nonzero
    nucleus_binary = (mask[:, :, :5].sum(axis=2) > 0).astype(np.uint8)
    
    # Sanity checks
    assert nucleus_binary.shape == (256, 256)
    assert nucleus_binary.dtype == np.uint8
    assert set(np.unique(nucleus_binary)).issubset({0, 1})
    
    return nucleus_binary


def generate_3class_labels(mask, erosion_radius=2):
    """
    Generate 3-class segmentation labels with explicit boundary class.
    
    Classes: 0=background, 1=nucleus interior, 2=nucleus boundary
    
    Boundary is generated by eroding EACH nucleus individually, then
    subtracting the eroded mask from the original binary mask.
    Eroding individually prevents touching nuclei from sharing boundaries.
    
    Parameters
    ----------
    mask : np.ndarray, shape (256, 256, 6)
    erosion_radius : int
        Radius of disk structuring element for erosion.
        Recommended range: 1-3. At 40x magnification, radius=2
        gives ~2-pixel boundaries which are visually meaningful.
    
    Returns
    -------
    labels : np.ndarray, shape (256, 256), dtype uint8
        0=background, 1=nucleus interior, 2=nucleus boundary
    """
    selem = disk(erosion_radius)
    
    # Combine all nucleus channels into a single instance map
    # Use argmax across channels 0-4 to get the dominant channel,
    # then use the raw sum to get a binary mask
    nucleus_binary = (mask[:, :, :5].sum(axis=2) > 0)
    
    # Build eroded mask by eroding each nucleus instance separately
    # This is critical: eroding the combined mask merges touching nuclei
    # and produces incorrect boundary pixels at the contact zone
    eroded = np.zeros((256, 256), dtype=bool)
    
    for c in range(5):  # for each nucleus type channel
        channel = mask[:, :, c]
        instance_ids = np.unique(channel)
        instance_ids = instance_ids[instance_ids > 0]  # exclude 0 (no nucleus)
        
        for inst_id in instance_ids:
            single_nucleus = (channel == inst_id)
            eroded_nucleus = binary_erosion(single_nucleus, selem)
            eroded |= eroded_nucleus
    
    # Boundary = nucleus pixels that were removed by erosion
    boundary = nucleus_binary & (~eroded)
    
    labels = np.zeros((256, 256), dtype=np.uint8)
    labels[eroded] = 1       # interior
    labels[boundary] = 2     # boundary
    # background stays 0
    
    # Verify no pixel is assigned multiple classes
    assert labels.max() <= 2
    assert labels.min() >= 0
    # Verify boundary pixels are a subset of nucleus pixels
    assert (boundary & ~nucleus_binary).sum() == 0
    
    return labels
```

**Verify:**

```python
# Load one image and its mask
mask_sample = np.load("data/raw/Fold/Fold1/masks/fold1/masks.npy")[0]

binary_lbl = generate_binary_labels(mask_sample)
three_lbl   = generate_3class_labels(mask_sample, erosion_radius=2)

print("Binary unique values:", np.unique(binary_lbl))         # {0, 1}
print("3-class unique values:", np.unique(three_lbl))         # {0, 1, 2}
print("Binary nucleus fraction:", binary_lbl.mean())          # expect 0.15–0.40
print("Interior fraction:", (three_lbl == 1).mean())
print("Boundary fraction:", (three_lbl == 2).mean())
print("Background fraction:", (three_lbl == 0).mean())

# Visual check — save overlay images
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
img = np.load("data/raw/Fold/Fold1/images/fold1/images.npy")[0].astype(np.uint8)
axes[0].imshow(img); axes[0].set_title("Original")
axes[1].imshow(binary_lbl, cmap='gray'); axes[1].set_title("Binary labels")
axes[2].imshow(three_lbl, cmap='viridis'); axes[2].set_title("3-class labels")
plt.savefig("data/debug_samples/label_verification.png", dpi=150, bbox_inches='tight')
print("Saved to data/debug_samples/label_verification.png")
```

Open the saved image. Verify: boundary pixels (class 2) form thin rings around each nucleus, touching nuclei have separated boundary rings, no background pixels are labeled as nucleus.

---

### Step 2.2 — Stage 2 nucleus type labels

**What:** For each nucleus instance in the dataset, extract its class label (0–4) and the pixel coordinates of its region.

**Why:** Stage 2 trains on these labels. Generating them incorrectly — e.g. assigning the wrong class to an instance — will cause the classifier to learn a corrupted mapping with no error signal during training.

**Instructions:**

Add to `src/data/label_generator.py`:

```python
def extract_nucleus_instances(image, mask):
    """
    Extract all nucleus instances from one image for Stage 2 training.
    
    Parameters
    ----------
    image : np.ndarray, shape (256, 256, 3) — RGB, float32 0-255
    mask  : np.ndarray, shape (256, 256, 6)
    
    Returns
    -------
    list of dicts, one per nucleus instance:
        {
          'class': int (0-4),
          'class_name': str,
          'region_mask': np.ndarray (256,256) bool — True at nucleus pixels,
          'bbox': (min_row, min_col, max_row, max_col),
          'image': np.ndarray (256,256,3) — full image for context
        }
    """
    CLASS_NAMES = ['neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial']
    instances = []
    
    for class_idx in range(5):
        channel = mask[:, :, class_idx]
        inst_ids = np.unique(channel)
        inst_ids = inst_ids[inst_ids > 0]
        
        for inst_id in inst_ids:
            region = (channel == inst_id)
            area = region.sum()
            
            # Skip unrealistically small regions (likely annotation artifacts)
            if area < 10:
                continue
            
            rows = np.where(region.any(axis=1))[0]
            cols = np.where(region.any(axis=0))[0]
            bbox = (rows[0], cols[0], rows[-1]+1, cols[-1]+1)
            
            instances.append({
                'class': class_idx,
                'class_name': CLASS_NAMES[class_idx],
                'region_mask': region,
                'bbox': bbox,
                'image': image,
            })
    
    return instances
```

**Verify:**

```python
img  = np.load("data/raw/Fold/Fold1/images/fold1/images.npy")[0]
mask = np.load("data/raw/Fold/Fold1/masks/fold1/masks.npy")[0]

instances = extract_nucleus_instances(img, mask)
print(f"Total instances in image 0: {len(instances)}")

# Check class distribution
from collections import Counter
dist = Counter(i['class_name'] for i in instances)
print("Class distribution:", dict(dist))
# Expected: neoplastic and epithelial tend to dominate; dead is rarest

# Verify no instance has class outside 0-4
assert all(0 <= i['class'] <= 4 for i in instances)
# Verify all regions are non-empty
assert all(i['region_mask'].sum() >= 10 for i in instances)
```

---

## Part 3 — Preprocessing pipeline

### Step 3.1 — Macenko stain normalization

**What:** Normalize the hematoxylin and eosin stain appearance across all images to a common reference.

**Why:** H&E staining varies between labs, scanners, and preparation protocols. A feature like "mean pixel value in red channel" will mean different things in a dark-stained image versus a light-stained one. Without normalization, your model learns staining artifacts rather than biological structure, and generalizes poorly to new slides.

**Impact if skipped:** Your model will learn stain-specific rather than structure-specific features. Performance will be inflated on the training distribution and collapse on new data.

**Instructions:**

Create `src/preprocessing/stain_normalizer.py`:

```python
import numpy as np
import staintools
import joblib


class MacenkoNormalizer:
    """
    Macenko SVD-based stain normalization for H&E images.
    
    The Macenko method (2009) decomposes each image into a stain
    matrix via SVD on the optical density (OD) space, then
    re-expresses the image using a reference stain matrix.
    This removes inter-scanner and inter-lab staining variability
    while preserving biological structure information.
    """
    
    def __init__(self):
        self.normalizer = staintools.StainNormalizer(method='macenko')
        self.reference_image = None
        self.is_fitted = False
    
    def select_reference_image(self, images, method='median_od'):
        """
        Select the most representative image as normalization target.
        
        Uses the image whose optical density matrix is closest to
        the median OD across all training images. Using a random
        image as reference introduces arbitrary bias.
        
        Parameters
        ----------
        images : np.ndarray, shape (N, 256, 256, 3), float32 0-255
        method : str — currently only 'median_od' supported
        """
        print(f"Selecting reference image from {len(images)} candidates...")
        
        # Compute mean OD per image (scalar summary)
        od_means = []
        valid_indices = []
        
        for i, img in enumerate(images):
            img_uint8 = np.clip(img, 1, 255).astype(np.uint8)
            od = -np.log(img_uint8.astype(np.float32) / 255.0 + 1e-6)
            od_means.append(od.mean())
            valid_indices.append(i)
        
        od_means = np.array(od_means)
        median_od = np.median(od_means)
        
        # Pick image closest to median
        ref_idx = valid_indices[np.argmin(np.abs(od_means - median_od))]
        self.reference_image = np.clip(images[ref_idx], 0, 255).astype(np.uint8)
        
        print(f"Selected reference image index: {ref_idx}")
        print(f"Reference image OD mean: {od_means[ref_idx]:.4f}")
        return ref_idx
    
    def fit(self, images):
        """Fit normalizer to reference image selected from images."""
        ref_idx = self.select_reference_image(images)
        self.normalizer.fit(self.reference_image)
        self.is_fitted = True
        return self
    
    def transform(self, image):
        """
        Normalize one image.
        
        Parameters
        ----------
        image : np.ndarray, shape (256, 256, 3), any numeric dtype
        
        Returns
        -------
        normalized : np.ndarray, shape (256, 256, 3), uint8
        """
        assert self.is_fitted, "Call fit() before transform()"
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        try:
            normalized = self.normalizer.transform(img_uint8)
            return normalized
        except Exception:
            # Staintools can fail on near-white or near-black images
            # Return original image in that case
            return img_uint8
    
    def save(self, path):
        joblib.dump(self, path)
        print(f"Normalizer saved to {path}")
    
    @staticmethod
    def load(path):
        return joblib.load(path)
```

**Verify:**

```python
from src.preprocessing.stain_normalizer import MacenkoNormalizer
import numpy as np
import matplotlib.pyplot as plt

images_fold1 = np.load("data/raw/Fold/Fold1/images/fold1/images.npy")

normalizer = MacenkoNormalizer()
normalizer.fit(images_fold1[:100])   # fit on first 100 images — fast

# Test on 5 images
fig, axes = plt.subplots(5, 2, figsize=(8, 20))
for i in range(5):
    original = images_fold1[i * 50].astype(np.uint8)
    normalized = normalizer.transform(original)
    axes[i, 0].imshow(original);    axes[i, 0].set_title("Original")
    axes[i, 1].imshow(normalized);  axes[i, 1].set_title("Normalized")
plt.savefig("data/debug_samples/stain_norm_verification.png", dpi=150, bbox_inches='tight')
```

Open the saved image. Verify: all normalized images have similar overall color tone. The purple/blue of hematoxylin and pink of eosin should be consistent across rows. If normalized images look identical (lost all variation), something is wrong with the reference selection.

---

### Step 3.2 — Color space conversion

**What:** Convert each normalized image from RGB into LAB, HSV, and HED (Hematoxylin-Eosin-DAB) color spaces. These become the source for color features in Part 4.

**Why:**
- LAB separates luminance (L) from color (A=green-red, B=blue-yellow). L is perceptually uniform — brightness differences in L correlate with actual tissue density differences.
- HSV separates hue from saturation, useful for stain purity features.
- HED deconvolves the actual biological stains. The H channel is the purest single-channel representation of nuclear chromatin content — more discriminative than any RGB channel for nucleus detection.

**Instructions:**

Create `src/preprocessing/color_converter.py`:

```python
import numpy as np
from skimage.color import rgb2lab, rgb2hsv, separate_stains, hdx_from_rgb

# Ruifrok & Johnston (2001) H&E deconvolution matrix
HE_MATRIX = np.array([
    [0.644211, 0.716556, 0.266844],
    [0.092789, 0.954111, 0.283111],
    [0.63718,  0.00000,  0.000103]
])


def convert_image(image_uint8):
    """
    Convert RGB image to LAB, HSV, and HED color spaces.
    
    Parameters
    ----------
    image_uint8 : np.ndarray, shape (256,256,3), dtype uint8
    
    Returns
    -------
    dict with keys: 'rgb', 'lab', 'hsv', 'hed'
    Each value is np.ndarray shape (256,256,3), float32
    """
    img_float = image_uint8.astype(np.float32)
    img_01 = img_float / 255.0  # scikit-image expects 0-1 for lab/hsv
    
    lab = rgb2lab(img_01).astype(np.float32)
    hsv = rgb2hsv(img_01).astype(np.float32)
    
    # HED deconvolution — stain separation
    # Result channels: H=hematoxylin (nuclei), E=eosin (cytoplasm), D=DAB
    hed = separate_stains(img_01, HE_MATRIX).astype(np.float32)
    
    return {
        'rgb': img_float,
        'lab': lab,
        'hsv': hsv,
        'hed': hed,
    }
```

**Verify:**

```python
from src.preprocessing.color_converter import convert_image

img = np.load("data/raw/Fold/Fold1/images/fold1/images.npy")[0].astype(np.uint8)
spaces = convert_image(img)

for name, arr in spaces.items():
    print(f"{name}: shape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}")
    assert not np.isnan(arr).any(), f"NaN in {name}"
    assert not np.isinf(arr).any(), f"Inf in {name}"

# HED H-channel should be highest at nucleus locations
binary_lbl = generate_binary_labels(
    np.load("data/raw/Fold/Fold1/masks/fold1/masks.npy")[0]
)
h_nucleus = spaces['hed'][:,:,0][binary_lbl == 1].mean()
h_background = spaces['hed'][:,:,0][binary_lbl == 0].mean()
print(f"H channel mean at nucleus: {h_nucleus:.4f}")
print(f"H channel mean at background: {h_background:.4f}")
# Nucleus should have higher H value — confirms HED is working correctly
```

---

## Part 4 — Feature extraction for Stage 1 (pixel-level)

The goal is a feature vector of ~200–300 dimensions per pixel, capturing color, texture, gradient, and spatial context. Every feature must be computed from a local neighborhood around the pixel — not from the whole image — because at inference time you classify pixels one at a time.

### Step 4.1 — Multi-scale windowed statistics (color features)

**What:** For each pixel, compute mean and std of each color channel in windows of radius 3, 7, and 15 pixels centered on that pixel.

**Why:** A pixel inside a nucleus has different mean H-channel values at different scales: at scale 3 it reflects the chromatin density; at scale 15 it reflects whether it is surrounded by other nuclei or by stroma. The combination of scales gives the classifier multi-resolution context without requiring a CNN.

**Instructions:**

Create `src/features/pixel_feature_extractor.py`:

```python
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import uniform_filter as gpu_uniform_filter
from scipy.ndimage import uniform_filter as cpu_uniform_filter
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel_h, sobel_v, laplace, gaussian
from skimage.filters import frangi
import warnings
warnings.filterwarnings('ignore')


# ── GPU batch windowed statistics ────────────────────────────────────────────
# This is the heaviest per-image computation (9 channels × 3 scales = 27
# filter passes).  Batching N images to GPU gives ~5-10× speedup.

def compute_windowed_stats_batch_gpu(channels_batch, radii=(3, 7, 15)):
    """
    GPU-accelerated windowed mean+std for a batch of single-channel images.
    
    Parameters
    ----------
    channels_batch : np.ndarray, shape (B, H, W)
        Batch of single-channel images on CPU.
    radii : tuple of int
    
    Returns
    -------
    features : np.ndarray, shape (B, H, W, 2*len(radii))  — back on CPU
    """
    channels_gpu = cp.asarray(channels_batch.astype(np.float32))
    features = []
    for r in radii:
        size = (1, 2*r+1, 2*r+1)   # batch dim=1, spatial dims=window
        mean_map = gpu_uniform_filter(channels_gpu, size=size)
        mean_sq  = gpu_uniform_filter(channels_gpu ** 2, size=size)
        std_map  = cp.sqrt(cp.maximum(mean_sq - mean_map**2, 0))
        features.append(mean_map)
        features.append(std_map)
    result = cp.stack(features, axis=-1)   # (B, H, W, 2*len(radii))
    return cp.asnumpy(result)


def compute_windowed_stats_single_cpu(channel, radii=(3, 7, 15)):
    """
    CPU fallback for a single channel — used when GPU is unavailable
    or for single-image inference.
    """
    features = []
    for r in radii:
        size = 2 * r + 1
        mean_map = cpu_uniform_filter(channel.astype(np.float32), size=size)
        mean_sq  = cpu_uniform_filter(channel.astype(np.float32)**2, size=size)
        std_map  = np.sqrt(np.maximum(mean_sq - mean_map**2, 0))
        features.extend([mean_map, std_map])
    return np.stack(features, axis=-1)


def compute_color_features_batch_gpu(spaces_list):
    """
    GPU-batch color feature extraction for multiple images.
    
    Parameters
    ----------
    spaces_list : list of dict — output of convert_image() per image
    
    Returns
    -------
    list of np.ndarray, each shape (256, 256, 57)
    """
    B = len(spaces_list)
    H, W = 256, 256
    all_feats = [np.empty((H, W, 0), dtype=np.float32) for _ in range(B)]
    
    # Batch windowed stats: 9 channels (3 spaces × 3 channels each)
    for space_name in ['lab', 'hsv', 'hed']:
        for c in range(3):
            # Collect this channel across all images → (B, H, W)
            batch = np.stack([s[space_name][:, :, c] for s in spaces_list])
            # GPU batch computation
            stats = compute_windowed_stats_batch_gpu(batch)  # (B,H,W,6)
            for i in range(B):
                all_feats[i] = np.concatenate(
                    [all_feats[i], stats[i]], axis=-1
                )
    
    # Optical Density: OD = -log(I/255 + 1e-6) per RGB channel — light compute
    for i in range(B):
        rgb = spaces_list[i]['rgb']
        od_feats = []
        for c in range(3):
            od = -np.log(rgb[:, :, c] / 255.0 + 1e-6)
            od_feats.append(od[:, :, np.newaxis])
        all_feats[i] = np.concatenate(
            [all_feats[i]] + od_feats, axis=-1
        )   # (256, 256, 57)
    
    return all_feats


def compute_color_features(spaces):
    """
    Single-image color feature extraction (CPU fallback / inference).
    For batch training, use compute_color_features_batch_gpu() instead.
    """
    feats = []
    for space_name in ['lab', 'hsv', 'hed']:
        arr = spaces[space_name]
        for c in range(3):
            channel_feats = compute_windowed_stats_single_cpu(arr[:, :, c])
            feats.append(channel_feats)
    rgb = spaces['rgb']
    for c in range(3):
        od = -np.log(rgb[:, :, c] / 255.0 + 1e-6)
        feats.append(od[:, :, np.newaxis])
    return np.concatenate(feats, axis=-1)  # (256, 256, 57)
```

---

### Step 4.2 — Texture features (LBP, GLCM, Gabor)

**What:** Extract LBP histograms, GLCM statistics, and Gabor filter responses per pixel neighborhood.

**Why:**
- LBP captures micro-texture patterns in a rotation-invariant way. Nuclear chromatin (coarse, granular) has a different LBP signature than smooth cytoplasm or fibrous stroma.
- GLCM captures spatial gray-level relationships. Neoplastic nuclei often show heterogeneous chromatin (high GLCM contrast, low homogeneity) compared to normal nuclei.
- Gabor filters capture oriented texture at specific scales. The elongated structure of stromal tissue responds differently to horizontal vs vertical Gabor filters than round nuclei.

**Important:** GLCM computed pixel-by-pixel over the full image is computationally infeasible. The strategy here is to compute GLCM on a patch around each sampled pixel during training, not as a dense feature map. For the dense feature map needed at inference, use only the LBP and Gabor maps.

Add to `src/features/pixel_feature_extractor.py`:

```python
def compute_lbp_features(gray_image, radii=(1, 2, 3)):
    """
    Compute LBP feature maps at multiple radii.
    Uses uniform rotation-invariant LBP: captures texture edges,
    corners, and flat regions robustly to image rotation.
    
    Returns (H, W, len(radii)) array.
    """
    feats = []
    for r in radii:
        P = 8 * r   # number of sampling points
        lbp = local_binary_pattern(gray_image, P=P, R=r, method='uniform')
        lbp_norm = lbp / (P + 2)  # normalize to 0-1
        feats.append(lbp_norm)
    return np.stack(feats, axis=-1)   # (H, W, 3)


def compute_gabor_features(gray_image):
    """
    Convolve with Gabor filters at 4 orientations × 3 scales.
    Gabor filters are band-pass filters tuned to specific
    spatial frequencies and orientations. Responses reveal
    directional texture structure.
    
    Returns (H, W, 12) array.
    """
    from skimage.filters import gabor
    feats = []
    for frequency in [0.1, 0.2, 0.4]:
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            real, _ = gabor(gray_image, frequency=frequency, theta=theta)
            feats.append(real.astype(np.float32))
    return np.stack(feats, axis=-1)   # (H, W, 12)


def compute_glcm_for_patch(gray_patch):
    """
    Compute GLCM statistics for a single image patch.
    Used for sampled pixels during training only — not dense maps.
    
    Returns 1D array of 16 features:
    contrast, dissimilarity, homogeneity, energy, correlation, ASM
    at 4 angles (0,45,90,135) aggregated as mean across angles.
    Plus 4 individual-angle contrast values.
    """
    gray_uint = (gray_patch * 255).astype(np.uint8) if gray_patch.max() <= 1 else gray_patch.astype(np.uint8)
    
    # 4 angles, 1 distance, 32 gray levels (downsampled for speed)
    gray_32 = (gray_uint // 8).astype(np.uint8)
    gcm = graycomatrix(gray_32, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)
    
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feats = []
    for p in props:
        vals = graycoprops(gcm, p).flatten()   # shape (1,4) → flatten to (4,)
        feats.append(vals.mean())              # mean across angles
    
    return np.array(feats)   # 6 features


def compute_texture_features(spaces):
    """
    Compute dense texture feature maps for pixel-level classification.
    Uses LBP and Gabor only (GLCM is computed per-sample separately).
    
    Returns (H, W, 15) array:  3 LBP + 12 Gabor
    """
    gray = spaces['lab'][:, :, 0]   # L channel as grayscale proxy
    
    lbp  = compute_lbp_features(gray)      # (H,W,3)
    gabor = compute_gabor_features(gray)   # (H,W,12)
    
    return np.concatenate([lbp, gabor], axis=-1)   # (H,W,15)
```

---

### Step 4.3 — Gradient and edge features

**What:** Compute Sobel gradients, Laplacian of Gaussian (LoG), and structure tensor eigenvalues.

**Why:**
- Sobel X and Y gradients detect intensity transitions. Nucleus boundaries produce strong gradient responses.
- LoG is a blob detector tuned to the scale of nuclei. At σ=2–4 it responds maximally at nucleus-sized circular structures — a direct feature for nucleus presence.
- Structure tensor eigenvalues measure local orientation coherence. Round nuclei give isotropic responses (eigenvalues equal); elongated stromal fibers give anisotropic responses. This is a powerful discriminator between nucleus types.

Add to `src/features/pixel_feature_extractor.py`:

```python
def compute_gradient_features_batch_gpu(gray_batch):
    """
    GPU-accelerated LoG for a batch of grayscale images.
    Sobel, structure tensor, and Frangi remain CPU (skimage-only).
    
    Parameters
    ----------
    gray_batch : np.ndarray, shape (B, H, W), float32
    
    Returns
    -------
    list of np.ndarray, each shape (H, W, 9)
    """
    from cupyx.scipy.ndimage import gaussian_laplace as gpu_gaussian_laplace
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    
    B = gray_batch.shape[0]
    gray_gpu = cp.asarray(gray_batch)
    
    # GPU: batch LoG at two scales
    log2_gpu = gpu_gaussian_laplace(gray_gpu, sigma=(0, 2, 2))  # no blur along batch
    log4_gpu = gpu_gaussian_laplace(gray_gpu, sigma=(0, 4, 4))
    log2_all = cp.asnumpy(log2_gpu)
    log4_all = cp.asnumpy(log4_gpu)
    
    results = []
    for i in range(B):
        gray = gray_batch[i]
        
        # CPU: Sobel (skimage — no GPU equivalent)
        sx = sobel_h(gray)
        sy = sobel_v(gray)
        grad_mag = np.sqrt(sx**2 + sy**2)
        
        # GPU LoG results (already computed)
        log2 = log2_all[i]
        log4 = log4_all[i]
        
        # CPU: Structure tensor (skimage)
        Axx, Axy, Ayy = structure_tensor(gray, sigma=1)
        eig = structure_tensor_eigenvalues(np.array([Axx, Axy, Ayy]))
        ev1, ev2 = eig[0], eig[1]
        anisotropy = (ev1 - ev2) / (ev1 + ev2 + 1e-8)
        
        # CPU: Frangi vesselness (skimage)
        frangi_map = frangi(gray).astype(np.float32)
        
        feats = [sx, sy, grad_mag, log2, log4, ev1, ev2, anisotropy, frangi_map]
        results.append(np.stack(feats, axis=-1))
    
    return results


def compute_gradient_features(spaces):
    """
    Single-image gradient features (CPU fallback / inference).
    For batch training, use compute_gradient_features_batch_gpu() instead.
    """
    from scipy.ndimage import gaussian_laplace
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    
    gray = spaces['lab'][:, :, 0].astype(np.float32)
    sx = sobel_h(gray)
    sy = sobel_v(gray)
    grad_mag = np.sqrt(sx**2 + sy**2)
    log2 = gaussian_laplace(gray, sigma=2)
    log4 = gaussian_laplace(gray, sigma=4)
    Axx, Axy, Ayy = structure_tensor(gray, sigma=1)
    eig = structure_tensor_eigenvalues(np.array([Axx, Axy, Ayy]))
    ev1, ev2 = eig[0], eig[1]
    anisotropy = (ev1 - ev2) / (ev1 + ev2 + 1e-8)
    frangi_map = frangi(gray).astype(np.float32)
    return np.stack([sx, sy, grad_mag, log2, log4,
                     ev1, ev2, anisotropy, frangi_map], axis=-1)
```

---

### Step 4.4 — Spatial context features

**What:** Compute Difference of Gaussians (DoG) and SLIC superpixel statistics.

**Why:**
- DoG approximates the LoG and is fast. At multiple scales it builds a spatial scale-space representation, encoding whether the pixel sits at the center of a nucleus-sized blob or at a larger/smaller structure.
- Superpixel statistics enforce spatial coherence cheaply. All pixels in the same superpixel receive the same "region average" features. This implicitly regularizes predictions — pixels in the same homogeneous region will predict the same class — without requiring a CRF.

Add to `src/features/pixel_feature_extractor.py`:

```python
def compute_spatial_features(spaces):
    """
    Compute spatial context feature maps.
    
    Returns (H, W, 10) array:
    - DoG at 3 scale pairs: 6 features
    - SLIC superpixel mean H-channel: 1 feature
    - SLIC superpixel mean L-channel: 1 feature
    - Distance to nearest image edge (normalized): 1 feature
    - Local entropy (window=5): 1 feature
    """
    from skimage.segmentation import slic
    from skimage.filters import rank
    from skimage.morphology import disk as morph_disk
    from skimage.measure import shannon_entropy
    
    gray = spaces['lab'][:, :, 0].astype(np.float32)
    h_channel = spaces['hed'][:, :, 0].astype(np.float32)
    
    # Difference of Gaussians
    dog_feats = []
    for s1, s2 in [(1, 2), (2, 4), (4, 8)]:
        g1 = gaussian(gray, sigma=s1)
        g2 = gaussian(gray, sigma=s2)
        dog_feats.append((g1 - g2).astype(np.float32))
    
    # SLIC superpixels (fast, color-aware)
    from skimage.color import rgb2lab as _rgb2lab
    img_uint8 = np.clip(spaces['rgb'], 0, 255).astype(np.uint8)
    segments = slic(img_uint8, n_segments=200, compactness=10,
                    sigma=1, start_label=0)
    
    sp_h_mean = np.zeros_like(h_channel)
    sp_l_mean = np.zeros_like(gray)
    for seg_id in np.unique(segments):
        mask = (segments == seg_id)
        sp_h_mean[mask] = h_channel[mask].mean()
        sp_l_mean[mask] = gray[mask].mean()
    
    # Normalized distance to nearest edge
    H, W = gray.shape
    row_dist = np.minimum(np.arange(H), H - 1 - np.arange(H)) / H
    col_dist = np.minimum(np.arange(W), W - 1 - np.arange(W)) / W
    edge_dist = np.minimum(row_dist[:, np.newaxis], col_dist[np.newaxis, :])
    
    # Local entropy (window 5)
    from skimage.filters.rank import entropy as rank_entropy
    gray_uint = (np.clip(gray, 0, 255)).astype(np.uint8)
    local_ent = rank_entropy(gray_uint, morph_disk(5)).astype(np.float32)
    local_ent /= local_ent.max() + 1e-8
    
    all_feats = dog_feats + [sp_h_mean, sp_l_mean, edge_dist.astype(np.float32), local_ent]
    return np.stack(all_feats, axis=-1)   # (H, W, 10)
```

---

### Step 4.5 — Master feature assembly

**What:** Combine all feature maps into one function that takes an image and returns the full feature matrix.

**Why:** Having a single entry point for feature extraction ensures that training and inference use identical feature computation. Any discrepancy (a feature computed differently at training vs inference) will cause a distribution shift that silently degrades predictions.

Add to `src/features/pixel_feature_extractor.py`:

```python
METADATA_COLS = ['image_idx', 'pixel_row', 'pixel_col']

FEATURE_GROUPS = {
    'color': 57,      # 9 channels × 6 stats (mean+std × 3 scales) + 3 OD
    'texture': 15,    # 3 LBP + 12 Gabor
    'gradient': 9,    # Sobel + LoG + structure tensor + Frangi
    'spatial': 10,    # DoG + superpixel + edge dist + entropy
}
TOTAL_FEATURES = sum(FEATURE_GROUPS.values())  # 91


def extract_all_features(image_uint8):
    """
    Extract full pixel-level feature matrix for ONE image (CPU path).
    Used for single-image inference.  For batch training, use
    extract_features_batch_gpu() below.
    
    Parameters
    ----------
    image_uint8 : np.ndarray, shape (256,256,3), dtype uint8
    
    Returns
    -------
    features : np.ndarray, shape (256*256, TOTAL_FEATURES), float32
    """
    from src.preprocessing.color_converter import convert_image
    
    spaces = convert_image(image_uint8)
    
    color_feats    = compute_color_features(spaces)     # (256,256,57) CPU
    texture_feats  = compute_texture_features(spaces)   # (256,256,15) CPU
    gradient_feats = compute_gradient_features(spaces)  # (256,256,9)  CPU
    spatial_feats  = compute_spatial_features(spaces)    # (256,256,10) CPU
    
    all_feats = np.concatenate([
        color_feats, texture_feats, gradient_feats, spatial_feats
    ], axis=-1)
    features = all_feats.reshape(-1, TOTAL_FEATURES)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


def extract_features_batch_gpu(images_uint8, batch_size=32):
    """
    GPU-accelerated feature extraction for a batch of images.
    
    Batches the GPU-friendly operations (windowed stats, LoG) while
    running CPU-only operations (LBP, Gabor, SLIC) per-image.
    
    Parameters
    ----------
    images_uint8 : np.ndarray, shape (N, 256, 256, 3)
    batch_size : int — number of images per GPU batch.
        Tune based on VRAM (see Section 0.5 table).
    
    Returns
    -------
    all_features : list of np.ndarray, each shape (65536, 91)
    """
    from src.preprocessing.color_converter import convert_image
    N = len(images_uint8)
    all_features = []
    
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_imgs = images_uint8[batch_start:batch_end]
        B = len(batch_imgs)
        
        # CPU: per-image color conversion (skimage — no GPU version)
        spaces_list = [convert_image(img) for img in batch_imgs]
        
        # ── GPU batch: windowed stats (54 features) + OD (3) = 57 ──────
        color_feats_list = compute_color_features_batch_gpu(spaces_list)
        
        # ── GPU batch: gradient LoG (9 features) ──────────────────────
        gray_batch = np.stack(
            [s['lab'][:, :, 0].astype(np.float32) for s in spaces_list]
        )
        gradient_feats_list = compute_gradient_features_batch_gpu(gray_batch)
        
        # ── CPU per-image: texture (15) + spatial (10) ────────────────
        for i in range(B):
            texture_feats  = compute_texture_features(spaces_list[i])
            spatial_feats  = compute_spatial_features(spaces_list[i])
            
            all_feats = np.concatenate([
                color_feats_list[i],
                texture_feats,
                gradient_feats_list[i],
                spatial_feats
            ], axis=-1)   # (256, 256, 91)
            
            features = all_feats.reshape(-1, TOTAL_FEATURES)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            all_features.append(features.astype(np.float32))
        
        if batch_end % 100 == 0 or batch_end == N:
            print(f"  Feature extraction: {batch_end}/{N} images")
    
    return all_features
```

**Verify:**

```python
from src.features.pixel_feature_extractor import (
    extract_all_features, extract_features_batch_gpu, TOTAL_FEATURES
)
import time
import numpy as np

images = np.load("data/raw/Fold/Fold1/images/fold1/images.npy")[:10]
images_uint8 = images.astype(np.uint8)

# Single-image CPU path (for inference)
t0 = time.time()
feats_single = extract_all_features(images_uint8[0])
print(f"Single-image CPU: {time.time()-t0:.2f}s, shape={feats_single.shape}")

# Batch GPU path (for training)
t0 = time.time()
feats_batch = extract_features_batch_gpu(images_uint8, batch_size=5)
print(f"Batch GPU (10 imgs, bs=5): {time.time()-t0:.2f}s")
print(f"Per-image shape: {feats_batch[0].shape}")   # (65536, 91)

# Verify consistency: single vs batch should produce identical features
assert np.allclose(feats_single, feats_batch[0], atol=1e-5), \
    "Single and batch features differ!"
print("Single/batch consistency: PASSED")
print(f"NaN count: {sum(np.isnan(f).sum() for f in feats_batch)}")  # must be 0
```

---

## Part 5 — Pixel sampling strategy

You cannot use all 65,536 pixels per image for training — this is computationally infeasible and introduces extreme class imbalance (background dominates). You need a principled sampling strategy.

### Step 5.1 — Active Boundary Mining (your existing strategy, corrected)

**What:** Sample pixels with higher probability near nucleus boundaries and lower probability in large background regions.

**Why:** Boundary pixels are where the classifier makes the most consequential errors. A boundary pixel misclassified as background removes a nucleus edge. A background pixel near a boundary misclassified as nucleus inflates predicted nucleus size. Oversampling boundaries directly addresses the hardest classification cases.

**Instructions:**

Create `src/sampling/pixel_sampler.py`:

```python
import numpy as np
import cupy as cp

try:
    from cupyx.scipy.ndimage import distance_transform_edt as gpu_edt
    _GPU_EDT = True
except ImportError:
    from scipy.ndimage import distance_transform_edt as gpu_edt
    _GPU_EDT = False


def _distance_transform(binary_mask_cpu):
    """GPU-accelerated distance transform with CPU fallback."""
    if _GPU_EDT:
        mask_gpu = cp.asarray(binary_mask_cpu.astype(np.float32))
        dist_gpu = gpu_edt(mask_gpu)
        return cp.asnumpy(dist_gpu)
    else:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(binary_mask_cpu)


def active_boundary_mining(labels, n_per_class=300, boundary_boost=3):
    """
    Sample pixel indices with elevated probability near boundaries.
    
    Strategy:
    1. All nucleus pixels are eligible for sampling
    2. Background pixels within distance D of a nucleus are sampled
       at `boundary_boost` × higher rate than far-background
    3. Strictly enforces n_per_class per label class to prevent imbalance
    
    Parameters
    ----------
    labels : np.ndarray, shape (256,256), values in {0,1} or {0,1,2}
    n_per_class : int — target samples per class
    boundary_boost : int — oversampling factor for near-boundary pixels
    
    Returns
    -------
    sampled_indices : np.ndarray, shape (N,) — flat pixel indices
    sampled_labels  : np.ndarray, shape (N,) — corresponding labels
    """
    H, W = labels.shape
    flat_labels = labels.flatten()
    
    # Distance transform from nucleus pixels (distance=0 at nucleus edge)
    nucleus_mask = (labels > 0).astype(np.float32)
    dist_from_nucleus = _distance_transform(1 - nucleus_mask)
    
    # Sampling weight map
    weights = np.ones(H * W, dtype=np.float32)
    near_boundary = (dist_from_nucleus.flatten() < 10) & (flat_labels == 0)
    weights[near_boundary] *= boundary_boost
    
    sampled_rows, sampled_cols = [], []
    sampled_lbls = []
    
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
        
        chosen = np.random.choice(H * W, size=n_sample,
                                   replace=False, p=lbl_probs)
        sampled_rows.extend(chosen // W)
        sampled_cols.extend(chosen % W)
        sampled_lbls.extend([lbl] * n_sample)
    
    indices = np.array(sampled_rows) * W + np.array(sampled_cols)
    return indices.astype(np.int32), np.array(sampled_lbls, dtype=np.uint8)
```

---

## Part 6 — Feature selection

### Step 6.1 — Three-stage selection pipeline

**What:** Remove uninformative and redundant features in three sequential steps: variance threshold → mRMR → RFE.

**Why:**
- Variance threshold removes constant or near-constant features. These carry zero information and add noise to distance-based models like SVM.
- mRMR (Minimum Redundancy Maximum Relevance) selects features that are jointly maximally informative — it penalizes selecting two features that are highly correlated with each other, even if both are correlated with the label. This is critical here because color features across LAB/HSV/HED channels are highly correlated.
- RFE (Recursive Feature Elimination) uses a model's own feature importances to iteratively discard the least useful features. This gives a model-aware selection that the filter methods above cannot provide.

**Impact if skipped:** SVM performance degrades significantly with correlated features. RF trains slowly on redundant features and gives less reliable importance scores.

**Instructions:**

Create `src/features/feature_selector.py`:

```python
import numpy as np
import cupy as cp
import cudf
import joblib
from sklearn.feature_selection import VarianceThreshold, RFE
from cuml.ensemble import RandomForestClassifier as cuRF_selector
from cuml.preprocessing import StandardScaler as cuScaler


class ThreeStageSelector:
    """
    Sequential 3-stage feature selector for pixel-level classification.
    GPU-accelerated via cuML / cuDF where possible.
    
    Stage 1: VarianceThreshold (sklearn — fast on CPU, negligible overhead)
    Stage 2: mRMR via cuDF DataFrames (GPU-accelerated tabular ops)
    Stage 3: cuML RF feature importances + top-K selection
             (fallback: sklearn RFE if cuML fails after retries)
    Stage 4: cuML StandardScaler (GPU)
    
    Must be fit on a subsample of training data only, never on test data.
    """
    
    def __init__(self, variance_threshold=0.01, mrmr_k=80, rfe_k=60,
                 cuml_retries=3):
        self.variance_threshold = variance_threshold
        self.mrmr_k = mrmr_k
        self.rfe_k = rfe_k
        self.cuml_retries = cuml_retries
        
        self.vt = VarianceThreshold(threshold=variance_threshold)
        self.mrmr_indices = None
        self.importance_indices = None   # cuML RF top-K indices
        self.rfe = None                  # sklearn RFE fallback
        self.scaler = None               # fitted cuML or sklearn scaler
        self.is_fitted = False
        self._used_cuml_stage3 = False
    
    def _cuml_importance_selection(self, X2, y_sample):
        """
        Stage 3 primary path: cuML RF feature importances + manual top-K.
        Retries up to self.cuml_retries times on failure.
        """
        for attempt in range(1, self.cuml_retries + 1):
            try:
                print(f"  cuML RF importance selection (attempt {attempt})...")
                X_gpu = cp.asarray(X2.astype(np.float32))
                y_gpu = cp.asarray(y_sample.astype(np.int32))
                rf = cuRF_selector(
                    n_estimators=100, max_depth=10,
                    max_features=0.5, random_state=42 + attempt
                )
                rf.fit(X_gpu, y_gpu)
                importances = rf.feature_importances_
                if hasattr(importances, 'get'):
                    importances = importances.get()
                importances = np.asarray(importances).flatten()
                top_k = np.argsort(importances)[::-1][:self.rfe_k]
                self.importance_indices = np.sort(top_k)
                self._used_cuml_stage3 = True
                print(f"  cuML top-{self.rfe_k} selection OK")
                return X2[:, self.importance_indices]
            except Exception as e:
                print(f"  cuML attempt {attempt} failed: {e}")
        return None   # signal to use fallback
    
    def _sklearn_rfe_fallback(self, X2, y_sample):
        """Stage 3 fallback: sklearn RFE on CPU."""
        from sklearn.ensemble import ExtraTreesClassifier
        print("  Falling back to sklearn RFE (CPU)...")
        base = ExtraTreesClassifier(
            n_estimators=100, n_jobs=-1,
            random_state=42, max_depth=10
        )
        self.rfe = RFE(base, n_features_to_select=self.rfe_k, step=5)
        self.rfe.fit(X2, y_sample)
        self._used_cuml_stage3 = False
        return self.rfe.transform(X2)
    
    def fit(self, X_sample, y_sample):
        """
        Fit selector on a subsample (recommended: 300,000–500,000 pixels).
        """
        print(f"Input features: {X_sample.shape[1]}")
        
        # Stage 1: Variance threshold (CPU — negligible cost)
        X1 = self.vt.fit_transform(X_sample)
        print(f"After variance threshold: {X1.shape[1]} features")
        
        # Stage 2: mRMR (cuDF for GPU-accelerated tabular ops)
        try:
            from mrmr import mrmr_classif
            df = cudf.DataFrame(X1)
            selected = mrmr_classif(df, cudf.Series(y_sample), K=self.mrmr_k)
            self.mrmr_indices = np.array(selected)
        except Exception:
            try:
                # Fallback to pandas if cuDF mRMR fails
                import pandas as pd
                from mrmr import mrmr_classif
                print("  cuDF mRMR failed, falling back to pandas...")
                df = pd.DataFrame(X1)
                selected = mrmr_classif(df, pd.Series(y_sample), K=self.mrmr_k)
                self.mrmr_indices = np.array(selected)
            except ImportError:
                print("  mrmr not installed, skipping Stage 2")
                self.mrmr_indices = np.arange(X1.shape[1])
        
        X2 = X1[:, self.mrmr_indices]
        print(f"After mRMR: {X2.shape[1]} features")
        
        # Stage 3: cuML RF importance (primary) → sklearn RFE (fallback)
        X3 = self._cuml_importance_selection(X2, y_sample)
        if X3 is None:
            X3 = self._sklearn_rfe_fallback(X2, y_sample)
        print(f"After Stage 3: {X3.shape[1]} features")
        
        # Stage 4: cuML StandardScaler (GPU)
        try:
            X3_gpu = cp.asarray(X3.astype(np.float32))
            self.scaler = cuScaler()
            self.scaler.fit(X3_gpu)
        except Exception:
            from sklearn.preprocessing import StandardScaler
            print("  cuML scaler failed, using sklearn StandardScaler")
            self.scaler = StandardScaler()
            self.scaler.fit(X3)
        
        self.is_fitted = True
        print(f"Final feature count: {X3.shape[1]}")
        return self
    
    def transform(self, X):
        assert self.is_fitted, "Call fit() first"
        X1 = self.vt.transform(X)
        X2 = X1[:, self.mrmr_indices]
        
        if self._used_cuml_stage3:
            X3 = X2[:, self.importance_indices]
        else:
            X3 = self.rfe.transform(X2)
        
        # cuML scaler accepts numpy — auto-converts internally
        X_scaled = self.scaler.transform(X3)
        if hasattr(X_scaled, 'get'):
            X_scaled = X_scaled.get()
        return np.asarray(X_scaled, dtype=np.float32)
    
    def fit_transform(self, X_sample, y_sample):
        self.fit(X_sample, y_sample)
        return self.transform(X_sample)
    
    def save(self, path):
        joblib.dump(self, path)
        print(f"Selector saved to {path}")
    
    @staticmethod
    def load(path):
        return joblib.load(path)
```

---

## Part 7 — Stage 1 model training

### Step 7.1 — Build training dataset

**What:** Iterate over all training images (Fold1 + Fold2), apply preprocessing, extract features, sample pixels, and accumulate the training matrix.

**Why:** This is where all previous steps come together. The order — normalize, then extract features, then sample — must be strictly maintained. Sampling before feature extraction would lose spatial context. Extracting features before normalization would make color features scanner-dependent.

**Instructions:**

Create `scripts/build_training_data.py`:

```python
import numpy as np
import os
import sys
sys.path.insert(0, '.')

from src.preprocessing.stain_normalizer import MacenkoNormalizer
from src.preprocessing.color_converter import convert_image
from src.features.pixel_feature_extractor import extract_features_batch_gpu
from src.data.label_generator import generate_binary_labels, generate_3class_labels
from src.sampling.pixel_sampler import active_boundary_mining


def build_stage1_dataset(fold_dirs, normalizer, label_mode='binary',
                          n_per_class=400, max_images=None,
                          gpu_batch_size=32):
    """
    Build full training matrix for Stage 1 segmentation.
    Uses GPU-batched feature extraction (see Section 0.5).
    
    Parameters
    ----------
    fold_dirs : list of (images_path, masks_path)
    normalizer : fitted MacenkoNormalizer
    label_mode : 'binary' or '3class'
    n_per_class : int — pixels sampled per class per image
    max_images : int or None — cap for debugging
    gpu_batch_size : int — images per GPU batch (tune for your VRAM)
    
    Returns
    -------
    X : np.ndarray, shape (N_total, F)
    y : np.ndarray, shape (N_total,)
    """
    X_list, y_list = [], []
    total_images = 0
    
    for images_path, masks_path in fold_dirs:
        images = np.load(images_path)
        masks  = np.load(masks_path)
        n = len(images) if max_images is None else min(len(images), max_images)
        
        # Process in GPU batches
        for batch_start in range(0, n, gpu_batch_size):
            batch_end = min(batch_start + gpu_batch_size, n)
            
            # Step 1: Stain-normalize the batch (CPU — per-image)
            batch_normalized = np.stack([
                normalizer.transform(images[i])
                for i in range(batch_start, batch_end)
            ])
            
            # Step 2: GPU-batch feature extraction
            features_list = extract_features_batch_gpu(
                batch_normalized, batch_size=gpu_batch_size
            )
            
            # Step 3: Per-image label generation + pixel sampling (CPU)
            for j, i in enumerate(range(batch_start, batch_end)):
                if label_mode == 'binary':
                    labels = generate_binary_labels(masks[i])
                else:
                    labels = generate_3class_labels(masks[i], erosion_radius=2)
                
                indices, sample_labels = active_boundary_mining(
                    labels, n_per_class=n_per_class
                )
                
                X_list.append(features_list[j][indices])
                y_list.append(sample_labels)
            
            print(f"  Processed {batch_end}/{n} images from {images_path}")
        
        total_images += n
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"\nTotal pixels: {len(X):,}")
    print(f"From {total_images} images")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y


if __name__ == '__main__':
    base = "data/raw/Fold"
    fold_dirs = [
        (f"{base}/Fold1/images/fold1/images.npy", f"{base}/Fold1/masks/fold1/masks.npy"),
        (f"{base}/Fold2/images/fold2/images.npy", f"{base}/Fold2/masks/fold2/masks.npy"),
    ]
    
    # Load normalizer (fitted in Part 3)
    normalizer = MacenkoNormalizer.load("data/models/normalizer.joblib")
    
    for mode in ['binary', '3class']:
        print(f"\n=== Building {mode} dataset ===")
        X, y = build_stage1_dataset(fold_dirs, normalizer, label_mode=mode)
        np.save(f"data/processed/X_stage1_{mode}.npy", X)
        np.save(f"data/processed/y_stage1_{mode}.npy", y)
        print(f"Saved X shape: {X.shape}, y shape: {y.shape}")
```

---

### Step 7.2 — Train GPU-accelerated models (RF, XGBoost, LightGBM)

**What:** Train three classifiers on the pixel feature matrix, all GPU-accelerated: cuML RandomForest, XGBoost with `device='cuda'`, and LightGBM with `device='gpu'`.

**Why:** Training multiple models enables comparison in the results tables. cuML RF is the primary model (interpretable via feature importances). XGBoost GPU histogram trees are typically the strongest tree ensemble. LightGBM GPU provides a fast alternative with leaf-wise growth that often excels on imbalanced data.

**Instructions:**

Create `scripts/train_stage1.py`:

```python
import numpy as np
import cupy as cp
import joblib
import sys
sys.path.insert(0, '.')

from src.features.feature_selector import ThreeStageSelector
from cuml.ensemble import RandomForestClassifier as cuRF
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


def _smote_balance(X, y):
    """Apply SMOTE on CPU, return numpy arrays."""
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_bal, y_bal = sm.fit_resample(X, y)
    print(f"After SMOTE: {X_bal.shape[0]:,} samples")
    print(f"Class counts: {dict(zip(*np.unique(y_bal, return_counts=True)))}")
    return X_bal, y_bal


def train_cuml_rf(X_bal, y_bal, params=None):
    """cuML Random Forest — GPU end-to-end."""
    if params is None:
        params = {
            'n_estimators': 500,
            'max_depth': 20,
            'max_features': 0.3,
            'random_state': 42,
        }
    print(f"\n[cuML RF] Training with {params}")
    X_gpu = cp.asarray(X_bal.astype(np.float32))
    y_gpu = cp.asarray(y_bal.astype(np.int32))
    rf = cuRF(**params)
    rf.fit(X_gpu, y_gpu)
    print("[cuML RF] Training complete.")
    return rf


def train_xgboost_gpu(X_bal, y_bal, n_classes, params=None):
    """XGBoost with GPU histogram backend."""
    if params is None:
        params = {
            'n_estimators': 500,
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'device': 'cuda',         # GPU hist backend
            'tree_method': 'hist',
            'random_state': 42,
            'eval_metric': 'mlogloss',
        }
    if n_classes == 2:
        params['objective'] = 'binary:logistic'
        params.pop('eval_metric', None)
    else:
        params['objective'] = 'multi:softmax'
        params['num_class'] = n_classes
    
    print(f"\n[XGBoost GPU] Training with {params}")
    xgb = XGBClassifier(**params)
    xgb.fit(X_bal, y_bal)
    print("[XGBoost GPU] Training complete.")
    return xgb


def train_lightgbm_gpu(X_bal, y_bal, n_classes, params=None):
    """LightGBM with GPU histogram backend."""
    if params is None:
        params = {
            'n_estimators': 500,
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'device': 'gpu',
            'random_state': 42,
            'verbose': -1,
        }
    if n_classes == 2:
        params['objective'] = 'binary'
    else:
        params['objective'] = 'multiclass'
        params['num_class'] = n_classes
    
    print(f"\n[LightGBM GPU] Training with {params}")
    lgb = LGBMClassifier(**params)
    lgb.fit(X_bal, y_bal)
    print("[LightGBM GPU] Training complete.")
    return lgb


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['binary', '3class'], default='binary')
    args = p.parse_args()
    
    print(f"=== Training Stage 1 [{args.mode}] ===")
    
    X = np.load(f"data/processed/X_stage1_{args.mode}.npy")
    y = np.load(f"data/processed/y_stage1_{args.mode}.npy")
    n_classes = len(np.unique(y))
    
    # Fit feature selector on subsample
    print("\nFitting feature selector on 300k subsample...")
    idx = np.random.choice(len(X), size=min(300_000, len(X)), replace=False)
    selector = ThreeStageSelector(mrmr_k=80, rfe_k=60)
    selector.fit(X[idx], y[idx])
    selector.save(f"data/models/selector_stage1_{args.mode}.joblib")
    
    # Apply feature selection + SMOTE
    X_sel = selector.transform(X)
    X_bal, y_bal = _smote_balance(X_sel, y)
    
    # ── Train all three models ──────────────────────────────────────────
    rf  = train_cuml_rf(X_bal, y_bal)
    xgb = train_xgboost_gpu(X_bal, y_bal, n_classes)
    lgb = train_lightgbm_gpu(X_bal, y_bal, n_classes)
    
    # Save models
    joblib.dump(rf,  f"data/models/rf_stage1_{args.mode}.joblib")
    joblib.dump(xgb, f"data/models/xgb_stage1_{args.mode}.joblib")
    joblib.dump(lgb, f"data/models/lgb_stage1_{args.mode}.joblib")
    print(f"\nAll 3 models saved to data/models/")
```

---

## Part 8 — Stage 1 evaluation

### Step 8.1 — Evaluate on Fold 3 test set

**What:** Run the trained model on held-out Fold 3 images and compute all evaluation metrics.

**Why:** Fold 3 was never seen during training or feature selector fitting. Evaluating on it gives an unbiased estimate of generalization performance. Do not evaluate on training folds — this will give optimistic numbers that cannot be compared to external baselines.

**Instructions:**

Create `scripts/evaluate_stage1.py`:

```python
import numpy as np
import cupy as cp
import joblib
import sys
sys.path.insert(0, '.')

from sklearn.metrics import f1_score, jaccard_score
from src.preprocessing.stain_normalizer import MacenkoNormalizer
from src.features.pixel_feature_extractor import extract_features_batch_gpu
from src.data.label_generator import generate_binary_labels, generate_3class_labels


def _to_numpy(arr):
    """Convert cuDF / CuPy / pandas arrays to plain numpy."""
    if hasattr(arr, 'to_numpy'):   # cuDF Series / DataFrame
        return arr.to_numpy()
    if hasattr(arr, 'get'):        # CuPy ndarray
        return arr.get()
    return np.asarray(arr)


def dice_coefficient(y_true, y_pred, label):
    """Binary Dice for one class label."""
    tp = ((y_true == label) & (y_pred == label)).sum()
    fp = ((y_true != label) & (y_pred == label)).sum()
    fn = ((y_true == label) & (y_pred != label)).sum()
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def evaluate_stage1(model, selector, normalizer, images_path,
                     masks_path, label_mode, types_path=None,
                     gpu_batch_size=32):
    """
    Full evaluation of Stage 1 on a test fold.
    Uses GPU batch feature extraction and handles cuML model outputs.
    Computes metrics on all pixels (not sampled) for unbiased evaluation.
    """
    images = np.load(images_path)
    masks  = np.load(masks_path)
    types  = np.load(types_path, allow_pickle=True) if types_path else None

    all_true, all_pred = [], []
    tissue_results = {}
    N = len(images)

    for batch_start in range(0, N, gpu_batch_size):
        batch_end = min(batch_start + gpu_batch_size, N)

        # Stain normalise (CPU per-image)
        batch_norm = np.stack([
            normalizer.transform(images[i])
            for i in range(batch_start, batch_end)
        ])

        # GPU batch feature extraction
        features_list = extract_features_batch_gpu(batch_norm,
                                                    batch_size=gpu_batch_size)

        for j, i in enumerate(range(batch_start, batch_end)):
            # Feature selection — returns float32 numpy (selector.transform
            # already does .get() internally)
            features_sel = selector.transform(features_list[j])

            # GPU inference — transfer to GPU, predict, bring back
            X_gpu = cp.asarray(features_sel)
            raw_preds = model.predict(X_gpu)
            preds = _to_numpy(raw_preds).flatten().astype(np.uint8)

            if label_mode == 'binary':
                labels = generate_binary_labels(masks[i])
            else:
                labels = generate_3class_labels(masks[i])

            all_true.append(labels.flatten())
            all_pred.append(preds)

            if types is not None:
                tissue = str(types[i])
                tissue_results.setdefault(tissue, {'true': [], 'pred': []})
                tissue_results[tissue]['true'].append(labels.flatten())
                tissue_results[tissue]['pred'].append(preds)

        if batch_end % 200 == 0 or batch_end == N:
            print(f"  Evaluated {batch_end}/{N} images")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred).astype(y_true.dtype)

    # Global metrics
    print(f"\n=== Stage 1 Evaluation [{label_mode}] ===")
    print(f"Total pixels evaluated: {len(y_true):,}")

    if label_mode == 'binary':
        nucleus_dice = dice_coefficient(y_true, y_pred, label=1)
        nucleus_iou  = jaccard_score(y_true, y_pred, pos_label=1)
        macro_f1     = f1_score(y_true, y_pred, average='macro')
        print(f"Nucleus Dice:  {nucleus_dice:.4f}")
        print(f"Nucleus IoU:   {nucleus_iou:.4f}")
        print(f"Macro F1:      {macro_f1:.4f}")
    else:
        for lbl, name in [(0,'background'), (1,'interior'), (2,'boundary')]:
            d = dice_coefficient(y_true, y_pred, label=lbl)
            print(f"Dice [{name}]: {d:.4f}")
        y_true_bin = (y_true > 0).astype(np.uint8)
        y_pred_bin = (y_pred > 0).astype(np.uint8)
        print(f"Collapsed nucleus Dice: {dice_coefficient(y_true_bin, y_pred_bin, 1):.4f}")
        print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")

    # Tissue-stratified results
    if tissue_results:
        print("\n--- Per-tissue Dice (nucleus) ---")
        tissue_dices = {}
        for tissue, data in tissue_results.items():
            yt = np.concatenate(data['true'])
            yp = np.concatenate(data['pred']).astype(yt.dtype)
            yt_b = (yt > 0).astype(np.uint8)
            yp_b = (yp > 0).astype(np.uint8)
            tissue_dices[tissue] = dice_coefficient(yt_b, yp_b, label=1)
        for t, d in sorted(tissue_dices.items(), key=lambda x: x[1]):
            print(f"  {t:<20}: {d:.4f}")

    return {'y_true': y_true, 'y_pred': y_pred}
```

---

## Part 9 — Stage 2 feature extraction (region-level)

### Step 9.1 — Extract nucleus region features

**What:** For each nucleus instance, compute a single feature vector capturing morphology, color, texture, and context of the entire nucleus region.

**Why:** Region-level features are different in kind from pixel-level features. A pixel feature describes "what is happening at this exact location." A region feature describes "what kind of object is this nucleus." Morphological features like eccentricity and solidity are only meaningful at the region level.

**Instructions:**

Create `src/features/nucleus_feature_extractor.py`:

```python
import numpy as np
from skimage.measure import regionprops
from skimage.feature import local_binary_pattern
from skimage.color import rgb2lab
from scipy.ndimage import distance_transform_edt


def extract_nucleus_features(region_mask, image_lab, image_hed):
    """
    Extract region-level features for one nucleus instance.
    
    Parameters
    ----------
    region_mask : np.ndarray, shape (256,256), bool — True at nucleus pixels
    image_lab   : np.ndarray, shape (256,256,3) — LAB color space
    image_hed   : np.ndarray, shape (256,256,3) — HED stain channels
    
    Returns
    -------
    features : np.ndarray, shape (N_NUCLEUS_FEATURES,) ≈ 50 features
    """
    
    # --- Morphological features (12) ---
    label_img = region_mask.astype(np.uint8)
    props = regionprops(label_img)[0]
    
    morph = np.array([
        props.area,
        props.perimeter if props.perimeter > 0 else 0,
        props.eccentricity,
        props.solidity,
        props.extent,
        props.major_axis_length,
        props.minor_axis_length,
        props.major_axis_length / (props.minor_axis_length + 1e-6),  # elongation
        props.convex_area / (props.area + 1e-6),  # convexity
        props.perimeter**2 / (4 * np.pi * props.area + 1e-6),  # circularity
        props.euler_number,
        np.sqrt(props.area / np.pi),   # equivalent radius
    ], dtype=np.float32)
    
    # --- Color statistics within nucleus (24) ---
    # LAB: mean + std per channel = 6 features
    # HED: mean + std per channel = 6 features
    # Additional percentiles on H channel = 4 features
    color_feats = []
    for c in range(3):
        vals = image_lab[:,:,c][region_mask]
        color_feats.extend([vals.mean(), vals.std()])
    for c in range(3):
        vals = image_hed[:,:,c][region_mask]
        color_feats.extend([vals.mean(), vals.std()])
    
    h_vals = image_hed[:,:,0][region_mask]
    color_feats.extend([
        np.percentile(h_vals, 10),
        np.percentile(h_vals, 25),
        np.percentile(h_vals, 75),
        np.percentile(h_vals, 90),
    ])
    color_feats = np.array(color_feats, dtype=np.float32)
    
    # --- Texture within nucleus bounding box (10) ---
    r0, c0, r1, c1 = props.bbox
    patch = image_lab[r0:r1, c0:c1, 0]  # L channel crop
    if patch.size > 0:
        gray_uint = np.clip(patch, 0, 255).astype(np.uint8)
        lbp = local_binary_pattern(gray_uint, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0,10), density=True)
        texture_feats = lbp_hist.astype(np.float32)
    else:
        texture_feats = np.zeros(10, dtype=np.float32)
    
    # --- Context: ring around nucleus (4) ---
    # The 5px ring immediately outside the nucleus
    dilated = distance_transform_edt(~region_mask) <= 5
    ring_mask = dilated & ~region_mask
    
    ring_feats = np.zeros(4, dtype=np.float32)
    if ring_mask.sum() > 0:
        ring_l = image_lab[:,:,0][ring_mask]
        ring_h = image_hed[:,:,0][ring_mask]
        ring_feats = np.array([
            ring_l.mean(), ring_l.std(),
            ring_h.mean(), ring_h.std(),
        ], dtype=np.float32)
    
    # Combine all
    features = np.concatenate([morph, color_feats, texture_feats, ring_feats])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


N_NUCLEUS_FEATURES = 12 + 16 + 4 + 10 + 4  # = 46
```

---

## Part 10 — Stage 2 model training

### Step 10.1 — Build Stage 2 training dataset

**What:** Extract region features and class labels for all nucleus instances across training folds.

**Why:** Stage 2 trains on oracle data — ground truth nucleus regions, not Stage 1 predictions. This separates the two sources of error (segmentation quality vs classification ability) and gives a clean upper bound on what Stage 2 can achieve.

**Instructions:**

Create `scripts/build_stage2_dataset.py`:

```python
import numpy as np
import sys
sys.path.insert(0, '.')

from src.preprocessing.stain_normalizer import MacenkoNormalizer
from src.preprocessing.color_converter import convert_image
from src.data.label_generator import extract_nucleus_instances
from src.features.nucleus_feature_extractor import extract_nucleus_features


def build_stage2_dataset(fold_dirs, normalizer):
    X_list, y_list = [], []
    
    for images_path, masks_path in fold_dirs:
        images = np.load(images_path)
        masks  = np.load(masks_path)
        
        for i in range(len(images)):
            img_norm = normalizer.transform(images[i])
            spaces = convert_image(img_norm)
            
            instances = extract_nucleus_instances(images[i], masks[i])
            
            for inst in instances:
                feat = extract_nucleus_features(
                    inst['region_mask'],
                    spaces['lab'],
                    spaces['hed']
                )
                X_list.append(feat)
                y_list.append(inst['class'])
            
            if (i + 1) % 100 == 0:
                print(f"  Image {i+1}: {len(instances)} nuclei, "
                      f"total so far: {len(X_list):,}")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.uint8)
    
    print(f"\nStage 2 dataset: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    class_names = ['neoplastic','inflammatory','connective','dead','epithelial']
    for u, c in zip(unique, counts):
        print(f"  {class_names[u]}: {c:,} instances ({100*c/len(y):.1f}%)")
    
    return X, y
```

### Step 10.2 — Train Stage 2 classifier

**What:** Train an SVM with RBF kernel and a Random Forest on the nucleus region features.

**Why:** SVM with RBF kernel is the strongest classical classifier for the feature scale here (~200,000 instances × 46 features). At this size it is computationally feasible. The nucleus feature space is likely to have non-linear class boundaries (neoplastic vs epithelial differ in texture and morphology, not just linearly in color), making RBF kernel appropriate.

**Instructions:**

Create `scripts/train_stage2.py`:

```python
import numpy as np
import cupy as cp
import joblib
import sys
sys.path.insert(0, '.')

from cuml.svm import SVC as cuSVC
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.preprocessing import StandardScaler as cuScaler
from imblearn.over_sampling import SMOTE


CLASS_NAMES = ['neoplastic','inflammatory','connective','dead','epithelial']


def train_stage2(X, y):
    # Normalize features on GPU — critical for SVM
    X_gpu = cp.asarray(X.astype(np.float32))
    scaler = cuScaler()
    X_sc = scaler.fit_transform(X_gpu)
    X_sc_cpu = cp.asnumpy(X_sc)
    
    # Handle class imbalance on CPU — dead nuclei are rare
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_sc_cpu, y)
    print(f"After SMOTE: {X_bal.shape}")
    
    # Transfer balanced data to GPU
    X_bal_gpu = cp.asarray(X_bal.astype(np.float32))
    y_bal_gpu = cp.asarray(y_bal.astype(np.int32))
    
    # cuML SVM with RBF kernel (primary) — massive speedup over sklearn
    print("\nTraining cuML SVM-RBF (GPU)...")
    svm = cuSVC(C=10.0, gamma='scale', kernel='rbf',
                probability=True, random_state=42)
    svm.fit(X_bal_gpu, y_bal_gpu)
    
    # cuML Random Forest (secondary — for feature importance)
    print("Training cuML Random Forest (GPU)...")
    rf = cuRF(n_estimators=500, max_depth=20,
              max_features=0.5, random_state=42)
    rf.fit(X_bal_gpu, y_bal_gpu)
    
    return svm, rf, scaler


if __name__ == '__main__':
    X = np.load("data/processed/X_stage2.npy")
    y = np.load("data/processed/y_stage2.npy")
    
    svm, rf, scaler = train_stage2(X, y)
    
    joblib.dump({'model': svm, 'scaler': scaler}, "data/models/svm_stage2.joblib")
    joblib.dump({'model': rf, 'scaler': scaler},  "data/models/rf_stage2.joblib")
    print("\nModels saved.")
```

---

## Part 11 — Hyperparameter tuning

### Step 11.1 — Optuna study for Stage 1 (cuML RF + XGBoost GPU)

**What:** Run Bayesian hyperparameter optimization using Optuna TPE sampler on Stage 1 models.

**Why:** The default hyperparameters are reasonable starting points but not optimal. Optuna's TPE sampler explores the parameter space efficiently. We tune both cuML RF and XGBoost GPU — the two strongest candidates — and pick the best overall.

**Note:** cuML models do not implement the sklearn estimator API required by `cross_val_score`, so we use a manual K-fold loop with CuPy arrays.

**Instructions:**

Create `scripts/tune_stage1.py`:

```python
import numpy as np
import cupy as cp
import optuna
import joblib
import sys
sys.path.insert(0, '.')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from cuml.ensemble import RandomForestClassifier as cuRF
from xgboost import XGBClassifier


def cuml_kfold_f1(model_cls, params, X, y, n_splits=3):
    """Manual K-fold CV for cuML / XGBoost models."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        model = model_cls(**params)
        
        if model_cls == cuRF:
            model.fit(cp.asarray(X_tr), cp.asarray(y_tr.astype(np.int32)))
            preds = model.predict(cp.asarray(X_va))
            preds = cp.asnumpy(preds).astype(int)
        else:
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
        
        scores.append(f1_score(y_va, preds, average='macro'))
    return np.mean(scores)


def create_objective(X, y, selector, model_type='rf'):
    X_sel = selector.transform(X).astype(np.float32)
    
    def objective(trial):
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'max_features': trial.suggest_float('max_features', 0.1, 0.8),
                'random_state': 42,
            }
            return cuml_kfold_f1(cuRF, params, X_sel, y)
        
        else:  # xgboost
            n_classes = len(np.unique(y))
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample', 0.3, 1.0),
                'device': 'cuda',
                'tree_method': 'hist',
                'random_state': 42,
                'eval_metric': 'mlogloss',
                'objective': 'multi:softmax' if n_classes > 2 else 'binary:logistic',
            }
            if n_classes > 2:
                params['num_class'] = n_classes
            return cuml_kfold_f1(XGBClassifier, params, X_sel, y)
    
    return objective


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['binary', '3class'], default='binary')
    p.add_argument('--model', choices=['rf', 'xgboost'], default='rf')
    p.add_argument('--trials', type=int, default=50)
    args = p.parse_args()
    
    X = np.load(f"data/processed/X_stage1_{args.mode}.npy")
    y = np.load(f"data/processed/y_stage1_{args.mode}.npy")
    selector = joblib.load(f"data/models/selector_stage1_{args.mode}.joblib")
    
    # Subsample for speed
    idx = np.random.choice(len(X), size=min(200_000, len(X)), replace=False)
    X_sub, y_sub = X[idx], y[idx]
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f'stage1_{args.model}_{args.mode}'
    )
    
    objective = create_objective(X_sub, y_sub, selector, model_type=args.model)
    study.optimize(objective, n_trials=args.trials, timeout=7200)
    
    print("\nBest trial:")
    print(f"  Value (macro F1): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    joblib.dump(study, f"data/models/optuna_study_{args.model}_{args.mode}.joblib")
```

---

## Part 12 — End-to-end inference and production export

### Step 12.1 — Full inference pipeline

**What:** A single function that takes a new image and returns the semantic mask + nucleus type predictions.

**Instructions:**

Create `src/inference/pipeline.py`:

```python
import numpy as np
import cupy as cp
import joblib
from scipy.ndimage import label as nd_label

from src.preprocessing.stain_normalizer import MacenkoNormalizer
from src.preprocessing.color_converter import convert_image
from src.features.pixel_feature_extractor import extract_all_features
from src.features.nucleus_feature_extractor import extract_nucleus_features


CLASS_NAMES = ['neoplastic','inflammatory','connective','dead','epithelial']


def _to_numpy(arr):
    """Convert cuDF / CuPy outputs to plain numpy."""
    if hasattr(arr, 'to_numpy'):
        return arr.to_numpy()
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


class LedgerLandInferencePipeline:

    def __init__(self, normalizer_path, selector_path, seg_model_path,
                 cls_model_path, seg_mode='binary'):
        self.normalizer = MacenkoNormalizer.load(normalizer_path)
        self.selector   = joblib.load(selector_path)
        seg = joblib.load(seg_model_path)
        cls = joblib.load(cls_model_path)
        self.seg_model  = seg['model'] if isinstance(seg, dict) else seg
        self.cls_model  = cls['model']
        self.cls_scaler = cls['scaler']
        self.seg_mode   = seg_mode

    def predict(self, image_rgb):
        """
        Run full two-stage prediction on one 256×256 image.

        Feature extraction uses the CPU path (single-image inference).
        Both cuML seg and cls models receive CuPy arrays and return
        outputs that are converted to numpy via _to_numpy().

        Returns
        -------
        dict:
            'semantic_mask': (256,256) uint8 — 0=bg, 1=nucleus [,2=boundary]
            'instance_mask': (256,256) int   — unique int per nucleus
            'nucleus_classes':    dict instance_id → class_name
            'nucleus_class_ids':  dict instance_id → class_id (0–4)
        """
        # Stage 1: semantic segmentation (CPU feature extraction)
        img_norm = self.normalizer.transform(image_rgb)
        features = extract_all_features(img_norm)          # CPU path
        features_sel = self.selector.transform(features)   # float32 numpy

        # GPU inference
        X_gpu = cp.asarray(features_sel)
        raw_preds = self.seg_model.predict(X_gpu)
        semantic_mask = _to_numpy(raw_preds).reshape(256, 256).astype(np.uint8)

        # Extract nucleus instances via connected components
        nucleus_binary = (semantic_mask > 0).astype(np.uint8)
        instance_mask, n_instances = nd_label(nucleus_binary)

        # Stage 2: classify each nucleus
        spaces = convert_image(img_norm)
        nucleus_classes = {}
        nucleus_class_ids = {}

        for inst_id in range(1, n_instances + 1):
            region = (instance_mask == inst_id)
            if region.sum() < 10:
                continue

            feat = extract_nucleus_features(region, spaces['lab'], spaces['hed'])
            feat_gpu = cp.asarray(feat.reshape(1, -1).astype(np.float32))

            # Scale on GPU then predict
            feat_sc = self.cls_scaler.transform(feat_gpu)
            raw_cls = self.cls_model.predict(feat_sc)
            cls_id = int(_to_numpy(raw_cls)[0])

            nucleus_classes[inst_id] = CLASS_NAMES[cls_id]
            nucleus_class_ids[inst_id] = cls_id

        return {
            'semantic_mask':    semantic_mask,
            'instance_mask':    instance_mask,
            'nucleus_classes':  nucleus_classes,
            'nucleus_class_ids': nucleus_class_ids,
        }
```

---

## Part 13 — Reporting your results

This is the minimum results table your report must include:

### Table 1 — Stage 1 segmentation comparison

> [!NOTE]
> All three Stage 1 models are trained on the same GPU-accelerated pipeline (RAPIDS 25.12). Report all three to demonstrate the value of each approach.

| Model | Backend | Exp | Binary Dice | Binary IoU | Macro F1 |
|---|---|---|---|---|---|
| cuML Random Forest | GPU (RAPIDS) | A (binary) | ? | ? | ? |
| XGBoost | GPU (`device='cuda'`) | A (binary) | ? | ? | ? |
| LightGBM | GPU (`device='gpu'`) | A (binary) | ? | ? | ? |
| cuML Random Forest | GPU (RAPIDS) | B (3-class) | ? | ? | ? |
| XGBoost | GPU (`device='cuda'`) | B (3-class) | ? | ? | ? |
| LightGBM | GPU (`device='gpu'`) | B (3-class) | ? | ? | ? |
| U-Net baseline | CPU/GPU (literature) | A (binary) | (from literature or your run) | | |

### Table 2 — Stage 2 nucleus typing

> [!NOTE]
> Oracle = ground truth nucleus regions fed to classifier (upper bound). Pipeline = real Stage 1 output fed to classifier (real-world performance).

| Model | Backend | Input source | Macro F1 | Neoplastic F1 | Dead F1 |
|---|---|---|---|---|---|
| cuML SVM-RBF | GPU (RAPIDS) | Oracle (GT masks) | ? | ? | ? |
| cuML SVM-RBF | GPU (RAPIDS) | Stage 1 cuML RF Exp A | ? | ? | ? |
| cuML SVM-RBF | GPU (RAPIDS) | Stage 1 XGBoost Exp A | ? | ? | ? |
| cuML SVM-RBF | GPU (RAPIDS) | Stage 1 cuML RF Exp B | ? | ? | ? |
| cuML SVM-RBF | GPU (RAPIDS) | Stage 1 XGBoost Exp B | ? | ? | ? |
| cuML RF | GPU (RAPIDS) | Oracle (GT masks) | ? | ? | ? |

### Table 3 — Oracle gap

Oracle gap = Oracle macro F1 − Pipeline macro F1

This number tells you how much Stage 1 segmentation errors hurt Stage 2. If the gap is large (> 10%), improving the segmentation stage is more valuable than improving the classifier. If the gap is small, the bottleneck is the classifier's intrinsic difficulty distinguishing nucleus types from features.

Report the oracle gap for each Stage 1 model (RF Exp A, XGBoost Exp A, RF Exp B, XGBoost Exp B) to show whether better Stage 1 Dice scores translate into smaller oracle gaps.

### Table 4 — Tissue-stratified Stage 1 Dice (all 19 tissues)

Report this table in an appendix for the best-performing Stage 1 model. It is required to demonstrate that your model does not have catastrophic failure on specific tissue types.

---

## Appendix A — Directory structure (final state)

```
project/
├── environment.yml
├── data/
│   ├── raw/Fold/{Fold1,Fold2,Fold3}/
│   ├── debug_samples/
│   ├── processed/
│   │   ├── X_stage1_binary.npy
│   │   ├── y_stage1_binary.npy
│   │   ├── X_stage1_3class.npy
│   │   ├── y_stage1_3class.npy
│   │   ├── X_stage2.npy
│   │   └── y_stage2.npy
│   └── models/
│       ├── normalizer.joblib
│       ├── selector_stage1_binary.joblib
│       ├── selector_stage1_3class.joblib
│       ├── rf_stage1_binary.joblib        ← cuML RF
│       ├── xgb_stage1_binary.joblib       ← XGBoost GPU
│       ├── lgb_stage1_binary.joblib       ← LightGBM GPU
│       ├── rf_stage1_3class.joblib
│       ├── xgb_stage1_3class.joblib
│       ├── lgb_stage1_3class.joblib
│       ├── svm_stage2.joblib              ← cuML SVM-RBF
│       ├── rf_stage2.joblib               ← cuML RF
│       ├── optuna_study_rf_binary.joblib
│       └── optuna_study_xgboost_binary.joblib
├── src/
│   ├── data/label_generator.py
│   ├── preprocessing/{stain_normalizer,color_converter}.py
│   ├── sampling/pixel_sampler.py
│   ├── features/{pixel_feature_extractor,nucleus_feature_extractor,feature_selector}.py
│   └── inference/pipeline.py
└── scripts/
    ├── build_training_data.py
    ├── train_stage1.py
    ├── evaluate_stage1.py
    ├── build_stage2_dataset.py
    ├── train_stage2.py
    └── tune_stage1.py
```

---

## Appendix B — What to cut if you fall behind schedule

| Time pressure | Safe to cut | Must not cut |
|---|---|---|
| Mild | Optuna tuning | Feature selector (run it, don't tune it) |
| Moderate | pydensecrf, Frangi filter | Macenko normalization |
| Severe | LightGBM secondary model | Stage 1 label fix, Stage 2 oracle eval |
| Critical | FastAPI endpoint | All evaluation metrics |

The oracle evaluation (Stage 2 with ground truth regions as input) is not optional even under time pressure — it is the scientific contribution that separates your two-stage analysis from a single-model result.

---

*End of implementation guide — version 2.0 (GPU-accelerated with RAPIDS 25.12)*
