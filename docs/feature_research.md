# Feature Set Research Document
## PanNuke Classical ML Nucleus Segmentation

**Purpose:** Research-backed rationale for every feature group used in Stage 1 (pixel classifier)
and Stage 2 (nucleus type classifier). Every feature is justified by its biological meaning
in H&E histology and its computational discriminative power.

**How to use this document:** Before implementing any feature, read its section completely.
Understand what biological or structural signal it captures, why that signal matters for
distinguishing nucleus from background (Stage 1) or one nucleus type from another (Stage 2),
and what its known failure modes are.

---

## Part A — The Biology You Are Measuring

Before discussing features, it is essential to understand what H&E staining reveals and
what you are actually trying to measure computationally.

### What H&E staining does

Hematoxylin binds to negatively charged molecules — primarily DNA and RNA — staining
nuclei blue-purple. Eosin binds to positively charged proteins in the cytoplasm and
extracellular matrix, staining them pink-red.

This means:

- **Dark blue-purple regions = nuclei** (high DNA content)
- **Pink regions = cytoplasm, collagen, muscle** (high protein content)
- **White/light regions = empty space, fat vacuoles, lumen**

Every feature you build should ultimately trace back to measuring one of these three
tissue compartments and their properties.

### What differs between nucleus types

| Type | Visual characteristics in H&E |
|---|---|
| Neoplastic | Large, irregular shape, coarse/clumped chromatin (very dark), prominent nucleoli, variable size |
| Inflammatory | Small, round, dense (lymphocytes: very dark compact nucleus), or multi-lobed (neutrophils) |
| Connective | Elongated/spindle-shaped, pale chromatin, embedded in fibrous extracellular matrix |
| Dead | Fragmented, pyknotic (shrunken, very dark), karyorrhectic (fragmented), or ghost-like |
| Epithelial | Medium size, round-oval, organized in sheets, pale-moderate chromatin |

Your Stage 2 classifier must distinguish these from region-level features.
Your Stage 1 classifier must only distinguish nucleus (any type) from background.

---

## Part B — Stage 1 Features (Pixel-Level)

For Stage 1, every feature is computed per pixel, using a local neighborhood
(window) centered on that pixel. The goal is: given these numbers for one pixel,
can the classifier decide whether this pixel belongs to a nucleus or not?

---

### Feature Group 1 — Optical Density (OD)

**What it is:**
Optical density is the negative log of the transmitted light intensity relative
to a blank (pure white) reference:

```
OD = -log(I / 255 + 1e-6)
```

Computed per RGB channel, giving three values per pixel.

**Biological basis:**
Beer-Lambert law states that absorbance (OD) is proportional to the concentration
of the absorbing substance. In histology, hematoxylin concentration is proportional
to DNA content. A pixel over a nucleus has high OD in the blue channel because the
dense DNA-hematoxylin complex absorbs light strongly. Background tissue (connective
tissue, fat) has much lower OD.

**Why it discriminates nucleus from background:**
OD is a more physically meaningful measure than raw pixel intensity. Two images
of the same tissue stained differently (different stain concentration, different
scanner) will have different raw RGB values but similar OD values — stain normalization
operates in OD space for this reason. After Macenko normalization, OD features are
more stable across your 19 tissue types than raw RGB statistics.

**Key signal:** High OD in the blue channel → high probability of nucleus.

**Known failure mode:** Very dark tissue structures that are not nuclei (e.g., red
blood cells in some staining protocols, melanin in skin sections) also have high OD.
This is why OD alone is insufficient — it needs to be combined with texture features
that capture the spatial structure of chromatin.

**Features extracted:** OD for R, G, B channels = 3 features.

---

### Feature Group 2 — Color Space Statistics (LAB, HSV, HED)

**What they are:**
Multi-scale windowed statistics (mean and standard deviation) of each channel in
three color spaces, computed at window radii of 3, 7, and 15 pixels.

**Why multiple color spaces:**
No single color space captures all diagnostically relevant color variation.
Each space separates color information differently:

#### LAB

CIE LAB separates luminance (L) from chrominance (A = green-red axis, B = blue-yellow axis).
The key property is that LAB is perceptually uniform — equal distances in LAB space
correspond to equal perceived color differences.

- **L channel:** Nuclei are darker than background (lower L value) because they absorb more
  light. The L channel is the single most discriminative raw channel for nucleus vs background.
- **A channel:** Hematoxylin-stained nuclei are shifted toward purple-blue, which in LAB
  corresponds to negative A (green direction). Eosin-stained cytoplasm is shifted toward red
  (positive A).
- **B channel:** Captures blue-yellow variation. Nuclei are shifted toward blue (negative B).

LAB is preferred over RGB for color statistics because its channels are less correlated
with each other than RGB channels, making each LAB feature more independently informative.

#### HSV

HSV separates hue (H), saturation (S), and value (V = brightness).

- **H channel (Hue):** Nuclei stained with hematoxylin have a characteristic purple hue
  (~240-280° in HSV). Eosin-stained cytoplasm is pink (~0-30°). Background is near white
  (undefined hue, high saturation = 0).
- **S channel (Saturation):** Highly stained regions (both nuclei and cytoplasm) have high
  saturation. White background has saturation near 0. This is a strong background detector.
- **V channel (Value):** Dark nuclei have low V. This is similar to L in LAB but not
  perceptually corrected.

HSV is useful because hue and saturation are separated from brightness, allowing the
classifier to learn color identity (what stain is present) separately from stain
intensity (how much stain is present).

#### HED (Hematoxylin-Eosin-DAB deconvolution)

HED is not a color space in the traditional sense — it is a stain unmixing operation
using the known absorption spectra of hematoxylin and eosin. The Ruifrok and Johnston
(2001) method uses a fixed deconvolution matrix to separate the contributions of each
stain into independent channels:

- **H channel:** Pure hematoxylin content. This is the single most directly meaningful
  channel for nucleus detection — it measures the DNA-binding stain directly.
- **E channel:** Pure eosin content. High E values indicate cytoplasm or extracellular matrix.
- **D channel (DAB):** Used in immunohistochemistry, not typically informative in standard H&E.

The H channel of HED is the most biologically grounded single feature for nucleus
detection. A pixel with high H-channel value is, by definition, covered by hematoxylin,
which means it is covering DNA-rich material.

**Why multi-scale windows matter:**
A single pixel's color value is noisy. A 3×3 window (radius 1) around a pixel captures
local color — but if the pixel sits at a nucleus boundary, the window mixes nucleus and
background color. A 15×15 window captures the neighborhood context — a pixel surrounded
by other nucleus pixels will show a high HED-H mean in a large window, reinforcing the
classification. Using three window sizes simultaneously allows the classifier to learn
both local color identity and neighborhood context.

**Features extracted:**
- 9 channels (LAB×3, HSV×3, HED×3) × 2 stats (mean, std) × 3 scales (r=3,7,15)
- = 9 × 2 × 3 = **54 features**
- Plus 3 OD features from Group 1
- **Total color features: 57**

---

### Feature Group 3 — Local Binary Patterns (LBP)

**What it is:**
LBP encodes the local texture around a pixel by comparing each pixel in a circular
neighborhood to the center pixel. Each neighbor is assigned 1 if brighter than center,
0 if darker. The resulting binary code encodes the local micro-pattern.

Uniform LBP restricts patterns to those with at most 2 bitwise transitions
(e.g., 00011110 has 2 transitions = uniform). This gives P+2 possible values for
P sampling points (P values for uniform patterns + 1 for non-uniform + 1 for all-same).
These are more robust to noise than full LBP.

**Biological basis:**
Nuclear chromatin has a characteristic granular texture — dense clumps of heterochromatin
surrounded by lighter euchromatin. This creates a specific micro-texture pattern that
is visible at the scale of individual pixels. LBP captures this because it detects
the arrangement of light and dark sub-pixel structures relative to each other.

**Why it discriminates nucleus from background:**
Background connective tissue has elongated fiber texture patterns (many transitions in
one direction). Nuclear chromatin has a roughly isotropic granular pattern (many
transitions in all directions). Empty space has uniform patterns (no transitions).
These three produce distinctly different LBP histograms.

**Why multiple radii (r=1, 2, 3):**
- r=1: Captures chromatin granularity at sub-nuclear scale (~2μm patches at 40×).
  Sensitive to individual chromatin clumps.
- r=2: Captures nuclear interior texture at intermediate scale.
- r=3: Captures the nucleus boundary pattern — the edge of a nucleus creates a specific
  LBP code (bright inside, dark outside creates a distinct half-and-half pattern).

Using three radii simultaneously gives the classifier a scale-space texture representation.

**Known failure mode:** LBP is sensitive to noise. Low-signal-to-noise images (e.g.,
thin sections) may produce noisy LBP values. The uniform variant mitigates this but
does not eliminate it. This is why LBP is used as one feature among many, not alone.

**Features extracted:** 3 radii × 1 LBP map per radius = **3 features** (the raw LBP
value per pixel at each radius, not histograms — histograms are used for Stage 2).

---

### Feature Group 4 — Gabor Filter Bank

**What it is:**
A Gabor filter is a Gaussian envelope modulated by a sinusoidal wave. Convolving an
image with a Gabor filter produces a response map that is high wherever the image
has intensity variation at a specific spatial frequency and orientation.

A Gabor filter bank uses multiple frequencies and orientations to build a full
scale-orientation decomposition.

**Parameters used:**
- Frequencies: 0.1, 0.2, 0.4 (cycles per pixel) — covers nucleus-scale to sub-nuclear-scale
- Orientations: 0°, 45°, 90°, 135° — covers all major directions
- Total: 3 × 4 = 12 filter responses per pixel

**Biological basis:**
Nuclei, stroma, and smooth muscle differ in their directional texture:
- **Nuclei:** Roughly isotropic (round) — respond similarly to all 4 orientations.
  The ratio of max-to-min orientation response is close to 1.
- **Connective tissue/stroma:** Highly anisotropic — collagen fibers are elongated.
  One or two orientations produce much stronger responses than the others.
- **Smooth muscle:** Elongated cells with strong orientation response in one direction.
- **Empty space:** Near-zero response at all orientations.

**Why it discriminates nucleus from background:**
The combination of orientation responses creates an "orientation fingerprint." A pixel
in connective tissue will have one dominant orientation response (aligned with fiber
direction). A pixel in a nucleus will have similar responses at all orientations.
The classifier learns to recognize these fingerprints.

**Why multiple frequencies:**
- f=0.1 (low frequency): Sensitive to large-scale gradients — detects the transition
  zone between tissue compartments.
- f=0.2 (mid frequency): Sensitive to chromatin-scale texture within nuclei.
- f=0.4 (high frequency): Sensitive to fine-grain texture — chromatin clumps,
  membrane details.

**Known failure mode:** Gabor responses are sensitive to image blur. Scanner defocus
or thick sections reduce high-frequency Gabor responses. Since you apply Macenko
normalization but not deblurring, this is a potential variability source across
PanNuke's 19 tissue types, which were scanned with different protocols.

**Features extracted:** 4 orientations × 3 frequencies = **12 features**.

---

### Feature Group 5 — GLCM (Gray-Level Co-occurrence Matrix)

**What it is:**
GLCM captures the spatial relationship between pixel pairs in a neighborhood.
For a given displacement (distance d, angle θ), the GLCM counts how often intensity
level i occurs adjacent to intensity level j in direction θ at distance d.
Statistical properties (contrast, homogeneity, energy, correlation, dissimilarity,
ASM) are then computed from this matrix.

**Why it is computed per-sampled-pixel (not as a dense map):**
GLCM for a full 256×256 image at one angle takes ~10-50ms. For 65,536 pixels per
image this is infeasible. Instead, for each sampled pixel, extract a 15×15 patch
centered on it and compute GLCM on that patch. At ~1,200 sampled pixels per image,
this is ~1,200 × 50ms = 60 seconds per image — still slow. Use 32 gray levels
(reduce 256→32 by integer division) to speed this up to ~5ms per patch.

**Biological basis:**
GLCM captures chromatin texture properties that LBP and Gabor miss:

- **Contrast** (GLCM): Measures intensity differences between neighboring pixels.
  Coarsely textured chromatin (neoplastic nuclei) has high contrast. Fine-grained
  chromatin (normal epithelial nuclei) has lower contrast.
- **Homogeneity:** High in regions with uniform intensity (smooth cytoplasm, background).
  Low in regions with variable intensity (heterogeneous chromatin).
- **Energy (Angular Second Moment):** High in highly regular textures (empty space,
  uniform eosin). Low in irregular, complex textures.
- **Correlation:** Measures linear dependency of gray levels. High in fibrous structures
  (stroma). Lower in isotropic nuclear chromatin.

**Why GLCM adds value over LBP:**
LBP captures the binary pattern of which neighbors are brighter/darker than the center.
GLCM captures the actual intensity difference magnitude and direction of that difference.
They are complementary — LBP is better at detecting texture type, GLCM is better at
measuring texture regularity and intensity statistics.

**Features extracted per sampled pixel:** 6 properties × 1 (mean across 4 angles) = **6 features**.

---

### Feature Group 6 — Gradient Features (Sobel, Laplacian of Gaussian)

**What it is:**

**Sobel X and Y:** First-order derivative filters that detect horizontal and vertical
intensity gradients respectively. Gradient magnitude = sqrt(Sx² + Sy²) detects edges
regardless of orientation.

**Laplacian of Gaussian (LoG):** Second-order derivative after Gaussian smoothing.
Responds maximally at the center of blob-like structures at the scale matching the
Gaussian sigma. Zero-crossings of LoG mark edges.

At σ=2: responds to blobs of radius ~4-6 pixels (~10-15μm at 40×).
At σ=4: responds to blobs of radius ~8-12 pixels (~20-30μm at 40×).

This scale range covers most nucleus sizes in PanNuke.

**Biological basis:**
Nucleus boundaries are the sharpest intensity transitions in H&E images. The boundary
between dark hematoxylin-stained chromatin and the lighter surrounding cytoplasm or
connective tissue creates a strong gradient. Both Sobel and LoG respond to this.

**Why LoG is particularly valuable:**
LoG is a blob detector, not just an edge detector. It responds positively at the
interior of a blob and negatively just outside it (or vice versa, depending on
sign convention). This means:
- Pixels inside a nucleus get a positive LoG response at σ=2-4
- Pixels just outside a nucleus get a negative LoG response
- Pixels in flat background or in large structures get a near-zero response

This three-zone response (positive inside, negative at boundary+surround, zero
in flat regions) is one of the most discriminative single features for nucleus detection.

**Why not HOG (Histogram of Oriented Gradients):**
HOG computes a histogram of gradient orientations over a cell of pixels.
For pixel-wise classification, true HOG requires computing a histogram per pixel's
neighborhood — this is computationally expensive and the resulting feature adds
limited value over Sobel + LoG combined. HOG is more useful for detecting specific
oriented shapes (human bodies, cars) than for detecting isotropic blobs like nuclei.
We exclude it to keep the feature set lean.

**Features extracted:**
- Sobel X, Sobel Y, gradient magnitude: 3 features
- LoG at σ=2: 1 feature
- LoG at σ=4: 1 feature
- **Total: 5 features**

---

### Feature Group 7 — Structure Tensor (Isotropy / Anisotropy)

**What it is:**
The structure tensor is a 2×2 matrix built from the gradient components at each pixel:

```
S = [[Ix*Ix, Ix*Iy],
     [Ix*Iy, Iy*Iy]]
```

averaged over a local window. Its eigenvalues λ1 ≥ λ2 describe the dominant
gradient magnitudes in the two principal directions.

Derived features:
- **λ1:** Largest eigenvalue — magnitude of dominant gradient direction.
- **λ2:** Smallest eigenvalue — magnitude of gradient in perpendicular direction.
- **Anisotropy:** (λ1 - λ2) / (λ1 + λ2 + ε) — ranges from 0 (perfectly isotropic)
  to 1 (perfectly anisotropic/directional).

**Biological basis:**
This is the single best classical feature for distinguishing round isotropic nuclei
from elongated anisotropic structures:

- **Nuclei:** Round shape → gradients point outward in all directions → λ1 ≈ λ2 →
  anisotropy ≈ 0. Low anisotropy.
- **Collagen fibers/stromal cells:** Elongated → gradients mainly perpendicular to
  fiber direction → λ1 >> λ2 → anisotropy ≈ 1. High anisotropy.
- **Empty space:** No gradient in any direction → λ1 ≈ λ2 ≈ 0. Low eigenvalues.

**Why this matters specifically for PanNuke:**
PanNuke has 19 tissue types including many with dense stroma (connective tissue,
fibrous tumors). Stromal fibroblasts and collagen bundles are strongly anisotropic.
Inflammatory cells (lymphocytes) and epithelial cells are isotropic. The structure
tensor anisotropy cleanly separates these classes even when color features overlap.

**Features extracted:**
- λ1 (largest eigenvalue): 1 feature
- λ2 (smallest eigenvalue): 1 feature
- Anisotropy index: 1 feature
- **Total: 3 features**

---

### Feature Group 8 — Difference of Gaussians (DoG) Scale Space

**What it is:**
DoG approximates the Laplacian of Gaussian by subtracting two Gaussian-blurred versions
of the image at different scales:

```
DoG(σ1, σ2) = Gaussian(σ1) - Gaussian(σ2)
```

Computed at three scale pairs: (σ=1,2), (σ=2,4), (σ=4,8).

**Why it works as a blob detector:**
DoG responds positively to circular blobs whose radius matches the scale pair.
At (σ=1,2): sensitive to small blobs (~4-8px, lymphocyte-sized nuclei).
At (σ=2,4): sensitive to medium blobs (~8-16px, epithelial/neoplastic nuclei).
At (σ=4,8): sensitive to large blobs (~16-32px, large neoplastic nuclei or cell clusters).

**Relationship to LoG:**
DoG is computationally faster than LoG and approximates it closely for scale ratios
near √2. We use both because they complement each other: LoG gives a cleaner blob
response at two fixed scales (σ=2,4), while DoG gives a scale-space representation
across three scale levels that helps the classifier understand what size structure
a pixel belongs to.

**Biological basis:**
Different nucleus types have different size distributions:
- Lymphocytes (inflammatory): small (~5-8μm diameter at 40× = ~8-12px)
- Epithelial: medium (~10-15μm = ~16-24px)
- Neoplastic: variable and often large (~15-25μm = ~24-40px)

The three DoG scale pairs effectively segment the size space. A pixel responding
strongly to DoG(σ=1,2) but weakly to DoG(σ=4,8) is more likely to be in a small
nucleus (inflammatory type). This size information, combined with color and texture,
substantially improves discrimination.

**Features extracted:** 3 scale pairs = **3 features**.

---

### Feature Group 9 — Superpixel Context Features

**What it is:**
SLIC (Simple Linear Iterative Clustering) superpixels segment the image into
~200 compact, color-homogeneous regions. For each superpixel region, compute
the mean value of the HED H-channel and LAB L-channel. All pixels within the
same superpixel are assigned the same superpixel mean values.

**Why this is needed:**
Pixel classifiers make independent decisions for each pixel. This leads to
"salt and pepper" noise in predictions — isolated pixels classified differently
from their neighbors, even when the image region is clearly uniform. Superpixel
features impose soft spatial coherence without requiring a CRF:

If a pixel is surrounded by other pixels that all belong to the same
SLIC superpixel, and that superpixel has a high mean HED-H value, the pixel
is almost certainly in a nucleus region. This context feature reinforces the
per-pixel color features with region-level evidence.

**Why not full CRF refinement at the feature stage:**
CRF operates on predictions, not features — it is a post-processing step that
adjusts the final probability map. Superpixel features bring spatial context
into the classifier itself, allowing it to learn region-level patterns that
CRF would only correct after the fact.

**Features extracted:**
- Superpixel mean HED H-channel: 1 feature
- Superpixel mean LAB L-channel: 1 feature
- **Total: 2 features**

---

### Feature Group 10 — Local Entropy

**What it is:**
Shannon entropy computed over a 5×5 pixel neighborhood using the gray-level
intensity histogram of that window.

**Biological basis:**
Entropy measures information content = complexity = unpredictability of intensity values.

- **Nucleus interior:** High entropy — chromatin is heterogeneous, many different
  intensity levels within a small window.
- **Empty background:** Very low entropy — near-uniform white, almost no variation.
- **Cytoplasm:** Medium entropy — more uniform than nucleus but less so than background.
- **Nucleus boundary:** Highest entropy — the window straddles high-intensity (nucleus)
  and low-intensity (background), maximizing the histogram spread.

**Why this helps for the 3-class experiment (Experiment B):**
In Experiment B you have a boundary class (class 2). Local entropy is the single most
direct proxy for "is this pixel at an intensity transition?" The boundary class is, by
definition, the highest-entropy zone in the image. A dedicated entropy feature gives
the classifier a direct signal for boundary pixels without relying on indirect evidence
from gradient or texture features.

**Features extracted:** 1 local entropy map = **1 feature**.

---

### Feature Group 11 — Edge Distance Features

**What it is:**
Normalized distance from each pixel to the nearest image border.
Computed as: min(row, H-1-row, col, W-1-col) / max(H, W).

**Why this is included:**
PanNuke patches are 256×256 crops from larger whole-slide images. Nuclei near the
patch border may be partially cropped — the mask still labels them as nucleus
pixels, but the nucleus is incomplete. A classifier that does not know whether
a pixel is near the border cannot account for this boundary artifact.

Additionally, some tissue types show systematic differences in nucleus density
near patch borders (where tissue sections were cut). The distance-to-border
feature lets the classifier learn a correction for these edge effects if they
exist in the training data.

This feature has low information content for most pixels (the majority are far
from the border) but has non-trivial value for pixels in the outer ~20 pixels
of each patch.

**Features extracted:** 1 distance map = **1 feature**.

---

### Summary: Stage 1 Feature Count

| Group | Features | Primary signal captured |
|---|---|---|
| Optical Density | 3 | DNA content via Beer-Lambert law |
| Color statistics (LAB, HSV, HED) | 54 | Stain identity and intensity at 3 scales |
| LBP (3 radii) | 3 | Chromatin micro-texture pattern |
| Gabor bank (3 freq × 4 orient) | 12 | Directional texture at multiple scales |
| GLCM (per sampled pixel) | 6 | Texture regularity and spatial statistics |
| Sobel + LoG | 5 | Nucleus boundaries and blob detection |
| Structure tensor | 3 | Isotropy vs anisotropy (nucleus vs stroma) |
| DoG scale space | 3 | Nucleus size encoding |
| Superpixel context | 2 | Spatial coherence with region context |
| Local entropy | 1 | Transition zone detector |
| Edge distance | 1 | Border artifact correction |
| **TOTAL** | **93** | |

---

## Part C — Stage 2 Features (Region-Level)

For Stage 2, each nucleus instance is one sample. You extract one feature vector
per nucleus, not per pixel. The goal is: given these numbers for one nucleus,
can the classifier decide whether it is neoplastic, inflammatory, connective,
dead, or epithelial?

The features here are fundamentally different from Stage 1 — they describe
the shape, size, color, and texture of an entire region, not a local neighborhood.

---

### Region Feature Group 1 — Morphological Shape Features

**What they measure:**
Properties of the nucleus boundary and shape computed from `skimage.measure.regionprops`.

| Feature | Definition | Biological interpretation |
|---|---|---|
| Area | Pixel count of region | Nucleus size — neoplastic nuclei are larger on average |
| Perimeter | Length of boundary | Irregular boundaries = higher perimeter for same area |
| Eccentricity | 0=circle, 1=line | Lymphocytes are round (low); fibroblasts are elongated (high) |
| Solidity | area / convex hull area | Lobulated nuclei (neutrophils) have low solidity |
| Extent | area / bounding box area | Irregular shapes have low extent |
| Major/minor axis ratio | Elongation index | Connective tissue nuclei are spindle-shaped (high ratio) |
| Circularity | 4π×area / perimeter² | Round nuclei = high; irregular = low |
| Convexity | area / convex area | Lobulated or fragmented nuclei have low convexity |
| Euler number | topological connectivity | Lobulated nuclei (neutrophils) have Euler < 1 |
| Equivalent radius | sqrt(area/π) | Direct size measure in pixels |

**Why morphology is the strongest Stage 2 discriminator:**
Nucleus type is largely determined by its shape. Pathologists use shape as the
primary visual cue: lymphocytes are round and small, fibroblasts are spindle-shaped,
neoplastic cells are large and irregular. These shape differences are directly
captured by morphological features in a rotation-invariant, scale-normalized way.

**Critical note on scale normalization:**
PanNuke's 256×256 patches come from variable magnifications across different tissue
sources. Area and perimeter are not directly comparable across magnifications. You must
check whether PanNuke's images are normalized to a standard magnification. If not,
include the raw pixel area but also include derived ratios (elongation, circularity,
solidity) which are scale-invariant.

**Features extracted: 10 morphological features**

---

### Region Feature Group 2 — Color Statistics Within Nucleus

**What they measure:**
Statistical summaries (mean, std, percentiles) of pixel values within the nucleus
region, measured in LAB and HED color spaces.

| Feature | Biological meaning |
|---|---|
| HED H-channel mean | Average hematoxylin content — hyperchromatic nuclei (neoplastic, dead) have higher mean |
| HED H-channel std | Chromatin heterogeneity — irregular chromatin (neoplastic) has higher std |
| HED H-channel 10th/90th percentile | Captures the extremes of chromatin density distribution |
| LAB L-channel mean | Overall nucleus darkness — dead (pyknotic) nuclei are very dark |
| LAB A-channel mean | Hue identity — confirms nucleus staining vs non-specific |
| HED E-channel mean | Eosin within nucleus boundary — epithelial cells may show more cytoplasmic eosin |

**Why color matters for type classification:**
Different nucleus types have different chromatin organization:
- **Neoplastic:** Hyperchromatic (darker, higher OD), coarsely granular chromatin.
  High HED H-channel mean and std.
- **Inflammatory (lymphocytes):** Homogeneous, very dark compact chromatin.
  High mean HED H, very low std (homogeneous).
- **Dead (pyknotic):** Extremely high OD, very dark. Highest LAB L-channel (darkest).
- **Epithelial:** Pale, vesicular chromatin. Lower HED H-channel mean than neoplastic.
- **Connective:** Pale, elongated chromatin. Low HED H-channel, low std.

**These patterns are why the H-channel mean alone is not enough:**
Lymphocytes and dead nuclei are both very dark, but dead nuclei are fragmented
(low solidity) while lymphocytes are compact (high solidity). Color features must
be combined with morphological features for correct type separation.

**Features extracted: 16 color statistical features**
(mean+std for L,A,B,H,E channels = 10, plus 4th/10th/75th/90th percentile of H = 6)

---

### Region Feature Group 3 — LBP Texture Histogram (Region-Level)

**What it is:**
Unlike Stage 1 where we used the raw LBP value at each pixel, for Stage 2 we
compute the full LBP histogram over all pixels in the nucleus region's bounding box.
This gives a distribution of texture patterns within the nucleus.

**Why a histogram instead of a single value:**
A histogram captures the entire distribution of micro-texture patterns within a nucleus.
Two nuclei can have the same mean LBP value but very different distributions:
a neoplastic nucleus might have a wide, flat distribution (many different texture patterns)
while a lymphocyte has a narrow, peaked distribution (one dominant compact texture pattern).

**Biological basis:**
Chromatin texture varies systematically with nucleus type:
- **Neoplastic:** Coarsely clumped chromatin → high variance in LBP codes, many
  non-uniform patterns (high LBP outlier count).
- **Inflammatory (lymphocytes):** Smooth, compact chromatin → predominantly uniform
  LBP patterns, narrow histogram.
- **Dead:** Fragmented chromatin → very irregular, high proportion of non-uniform
  LBP codes.
- **Epithelial:** Vesicular (open) chromatin → mixture of smooth and granular, medium spread.

**Implementation note:**
Compute LBP at radius=1, P=8 with uniform method. Bin into 10 bins (8 uniform patterns
+ 1 non-uniform + 1 flat) and L1-normalize. This gives a rotation-invariant, compact
texture fingerprint per nucleus.

**Features extracted: 10 LBP histogram bins**

---

### Region Feature Group 4 — Context Ring Features

**What it is:**
Statistics of pixels in a 5-pixel-wide ring immediately surrounding the nucleus region
(the annular zone just outside the nucleus boundary).

**Biological basis:**
The tissue immediately surrounding a nucleus is diagnostic of its type:

- **Neoplastic nuclei in carcinomas:** Surrounded by other tumor cells or minimal stroma.
  Ring has high HED H-channel (nearby nuclei) or high eosin (dense cytoplasm).
- **Inflammatory cells (lymphocytes):** Surrounded by stromal matrix or other lymphocytes.
  Ring has high eosin (fibrous stroma) or similar compact nuclear texture.
- **Connective tissue nuclei (fibroblasts):** Embedded in collagen matrix. Ring has
  high eosin, low HED H (no surrounding nuclei — fibrous stroma).
- **Epithelial cells:** Part of epithelial sheets. Ring has cytoplasmic eosin from
  adjacent cells, moderate HED H from nearby nuclei in the sheet.
- **Dead nuclei:** Often surrounded by inflammatory cells or necrotic debris.
  Ring may show both nuclear (HED H) and heterogeneous signals.

**Why this feature matters for the oracle gap analysis:**
Context ring features are only correctly computable from the true nucleus region
(oracle input). When Stage 1 predictions are fed to Stage 2, the predicted nucleus
region may be slightly larger or smaller than the true nucleus, shifting the ring
position and corrupting context features. The oracle gap (oracle accuracy − pipeline
accuracy) will partially reflect this context feature degradation.

**Features extracted:**
- Ring HED H-channel: mean, std = 2 features
- Ring LAB L-channel: mean, std = 2 features
- **Total: 4 features**

---

### Stage 2 Feature Count Summary

| Group | Features | Key discriminator for |
|---|---|---|
| Morphological shape | 10 | Lymphocyte vs fibroblast (size, elongation) |
| Color statistics (LAB + HED) | 16 | Neoplastic vs epithelial (hyperchromasia), dead (pyknosis) |
| LBP histogram | 10 | Chromatin texture regularity |
| Context ring | 4 | Connective tissue (stromal context) |
| **TOTAL** | **40** | |

---

## Part D — Features Deliberately Excluded and Why

Being explicit about what you are NOT using is as important as what you are using.

### Frangi Vesselness — Excluded

Frangi vesselness detects elongated tubular structures. It was in the earlier draft.
We exclude it because:
1. Tubular structures (blood vessels, ducts) appear at scales much larger than
   individual nuclei. At the 256×256 patch scale, partial vessels appear at the edges.
2. The structure tensor anisotropy already captures elongation information with fewer
   assumptions about shape.
3. Frangi is computationally expensive relative to its marginal benefit for this task.

### HOG (Histogram of Oriented Gradients) — Excluded

HOG was designed for detecting oriented, elongated shapes (people, cars) in natural
images. For nucleus detection, nuclei are round and roughly isotropic — HOG's
orientation histograms add complexity without the shape-matching benefit that makes
HOG powerful in its intended domain. The Gabor bank + structure tensor combination
provides equivalent orientation-sensitive features with clearer biological interpretation.

### Hu Moments — Excluded from Stage 1, Optional for Stage 2

Hu moments are rotation-invariant shape descriptors. They are meaningful at the
region level (Stage 2) where you have a complete nucleus shape to describe. At the
pixel level (Stage 1), there is no meaningful "shape" to compute moments for — the
local neighborhood is not a nucleus shape, just a texture patch. For Stage 2, Hu
moments can be added as an optional extension if the 10 morphological features
prove insufficient for shape discrimination.

### Zernike Moments — Excluded

Zernike moments provide a rich rotation-invariant shape description. They are
excluded because they require a clean binary region of roughly circular shape to
compute accurately. In the pipeline-fed Stage 2 (where Stage 1 predictions drive
the regions), nucleus boundaries are noisy enough that Zernike moments become
unreliable. Simpler morphological features (eccentricity, solidity, circularity)
capture the same shape information more robustly under noisy boundary conditions.

### Deep Features (CNN activations) — Excluded by design

This entire project's research question is how far hand-crafted features can go.
CNN features are excluded by research design, not limitation.

---

## Part E — Feature Interactions: What the Classifier Actually Learns

Individual features do not make decisions — the classifier learns combinations.
Here are the most important feature interactions to understand:

**Interaction 1: LoG + HED H-channel (nucleus vs empty space)**
A pixel with high LoG response at σ=2 AND high HED H-channel mean is almost
certainly inside a nucleus. Either feature alone can be fooled (LoG by any dark
circular structure, HED by staining artifacts), but together they are highly specific.

**Interaction 2: Anisotropy + HED H-channel (nucleus vs stroma)**
A pixel with low anisotropy (isotropic structure) AND high HED H-channel
is inside a nucleus. A pixel with high anisotropy AND moderate HED H-channel
is likely in fibrous stroma with some nuclear contamination in the window.

**Interaction 3: LBP entropy + local entropy (nucleus interior vs boundary)**
This interaction matters for Experiment B (3-class). Boundary pixels have high local
entropy (intensity transition) AND specific LBP codes (the half-bright, half-dark
patterns at radius 1-2 that appear at nucleus edges). Interior pixels have moderate
entropy but different LBP codes (more uniform, all-bright patterns).

**Interaction 4: Superpixel mean HED + DoG (nucleus vs background at superpixel scale)**
Even if a single pixel is ambiguous, if its superpixel has a high mean HED H-value
AND the DoG at the appropriate scale is positive, the classifier has strong contextual
evidence that the pixel is in a nucleus-dense region.

---

## Part F — Feature Computation Order and Dependencies

The implementation must follow this exact order because later features depend on earlier ones:

```
1. Load raw image [images[i]]                    → raw uint8 (256,256,3)
2. Macenko normalization                          → normalized uint8 (256,256,3)
3. Color conversion                               → spaces dict {lab, hsv, hed, rgb}
4. Group 1: Optical density                       → from spaces['rgb']
5. Group 2: Color statistics (windowed)           → from spaces['lab'], ['hsv'], ['hed']
6. Group 3: LBP                                   → from spaces['lab'][:,:,0] (gray)
7. Group 4: Gabor                                 → from spaces['lab'][:,:,0] (gray)
8. Group 5: LoG, Sobel                            → from spaces['lab'][:,:,0] (gray)
9. Group 6: Structure tensor                      → from spaces['lab'][:,:,0] (gray)
10. Group 7: DoG                                  → from spaces['lab'][:,:,0] (gray)
11. Group 8: Superpixel context                   → from spaces['hed'][:,:,0] + spaces['lab']
12. Group 9: Local entropy                        → from spaces['lab'][:,:,0] (gray as uint8)
13. Group 10: Edge distance                       → from image shape only (no pixel values)
14. Concatenate all → (256,256,93) feature tensor
15. Flatten → (65536, 93) pixel matrix
16. Sample pixels using active boundary mining
17. [Per sampled pixel] compute GLCM on 15×15 patch → appended to each sampled row
    Final per-pixel feature vector: 93 + 6 = 99 features
```

The reason gray (LAB L-channel) is used as the grayscale proxy rather than
`rgb2gray` is consistency: LAB is already computed in step 3, so using its
L-channel avoids a redundant color conversion and ensures all gradient and
texture features operate on the same grayscale representation.

---

## Part G — Expected Feature Importance Ranking

Based on established histopathology literature, the expected importance ranking
from highest to lowest for Stage 1 (nucleus vs background):

1. HED H-channel statistics (Groups 1, 2) — most direct measurement of nuclear stain
2. LAB L-channel statistics (Group 2) — darkness is the dominant visual cue
3. LoG at σ=2-4 (Group 5) — blob detection at nucleus scale
4. Structure tensor anisotropy (Group 6) — nucleus vs stroma discrimination
5. DoG scale space (Group 7) — size context
6. LBP at r=2-3 (Group 3) — chromatin texture
7. Gabor responses (Group 4) — directional texture context
8. Superpixel context (Group 8) — spatial regularization
9. GLCM (Group 5) — chromatin regularity
10. Local entropy (Group 9) — boundary detection (more useful for Exp B)
11. OD features (Group 1) — partially redundant with HED H after normalization
12. Edge distance (Group 10) — low importance except near patch borders

**Important:** This is an expected ranking based on domain knowledge. The actual
importance from your trained Random Forest may differ, especially for features
that are correlated with each other (OD and HED H-channel carry similar information).
Always verify against the RF feature importances from your trained model and compare
to this expected ranking. Large disagreements indicate either a feature implementation
error or an unexpected biological phenomenon in your specific tissue types.

---

*End of Feature Set Research Document*
*Version 1.0 — to be updated after feature selection results are obtained from the trained model*
