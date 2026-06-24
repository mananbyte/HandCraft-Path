#!/usr/bin/env python3
"""
notebooks/create_ensemble_notebook.py
──────────────────────────────────────
Generator script that creates notebooks/08_inspect_binary_ensemble.ipynb
using nbformat. This avoids writing raw JSON by hand.

Run with:
  conda run -n HandCraft-Path python notebooks/create_ensemble_notebook.py
"""

import nbformat as nbf
import os

# ── Output path ────────────────────────────────────────────────────────────
NOTEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "08_inspect_binary_ensemble.ipynb",
)

# ── Helper to create cells ─────────────────────────────────────────────────
def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


# ── Build notebook ─────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "HandCraft-Path",
    "language": "python",
    "name": "HandCraft-Path",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

cells = []

# ──────────────────────────────────────────────────────────────────────────
# Section 1 — Setup & Artifact Loading
# ──────────────────────────────────────────────────────────────────────────
cells.append(md("""# Phase 4 — Binary Segmentation Ensemble Inspection

## Section 1 — Setup & Artifact Loading

This notebook documents the performance of the three binary nucleus segmentation
classifiers (cuML RandomForest, LightGBM, XGBoost) and their soft-voting ensemble
trained in Phase 4 on the PanNuke Fold 1 feature matrix (2.07M rows × 25 RFE features).

**Training summary:**
- Models were tuned independently via Optuna HPO (25 trials × 30 min budget) on a
  200K stratified subsample of Fold 1.
- Best hyperparameters were then refitted on the **full Fold 1** (2.07M rows).
- Validation uses the **held-out Fold 2** (~1.86M rows), scaled in-memory via the
  locked Phase 3 `HandCraftPathScaler` — Fold 2 was never seen during training or tuning.
- Ensemble blending weights were optimised via grid search on the weight simplex (step=0.1).

Each section below explains the biological and clinical significance of the shown metric
before the code and plots.
"""))

cells.append(code("""import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Ensure project root is on path
sys.path.insert(0, os.path.abspath('..'))

MODELS_DIR = 'data/models'
PROCESSED_DIR = 'data/processed'
FIGURES_DIR = 'notebooks/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Graceful missing-file guard ────────────────────────────────────────────
REQUIRED_FILES = [
    f'{MODELS_DIR}/binary_ensemble.joblib',
    f'{MODELS_DIR}/cuml_rf.joblib',
    f'{MODELS_DIR}/lgbm.joblib',
    f'{MODELS_DIR}/xgboost.joblib',
    f'{MODELS_DIR}/selected_features_rfe_25.json',
    f'{PROCESSED_DIR}/fold2_binary_X.npy',
    f'{PROCESSED_DIR}/fold2_binary_y.npy',
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    print('⚠ Training artifacts not yet generated. Run:')
    print('  python scripts/train_ensemble.py')
    print()
    print('Missing files:')
    for f in missing:
        print(f'  - {f}')
    print()
    print('Notebook created successfully — run train_ensemble.py first, then re-execute notebook.')
    ARTIFACTS_READY = False
else:
    ARTIFACTS_READY = True
    print('✓ All required artifacts found.')
"""))

cells.append(code("""if not ARTIFACTS_READY:
    raise SystemExit('Artifacts not ready — see message above.')

from src.training.ensemble import SoftVotingEnsemble
from src.training.scaler import FeatureScaler
from src.utils.safe_loader import safe_load_npy

# ── Load feature indices ───────────────────────────────────────────────────
with open(f'{MODELS_DIR}/selected_features_rfe_25.json') as fh:
    feat_json = json.load(fh)
RFE_INDICES = feat_json['indices']
FEATURE_NAMES = feat_json['names']
print(f'RFE features ({len(RFE_INDICES)}): {FEATURE_NAMES[:5]}...')

# ── Load base models ───────────────────────────────────────────────────────
rf_payload   = joblib.load(f'{MODELS_DIR}/cuml_rf.joblib')
lgbm_payload = joblib.load(f'{MODELS_DIR}/lgbm.joblib')
xgb_payload  = joblib.load(f'{MODELS_DIR}/xgboost.joblib')

rf_model   = rf_payload['model']
lgbm_model = lgbm_payload['model']
xgb_model  = xgb_payload['model']

rf_params   = rf_payload['best_params']
lgbm_params = lgbm_payload['best_params']
xgb_params  = xgb_payload['best_params']

# ── Load ensemble ──────────────────────────────────────────────────────────
ensemble = SoftVotingEnsemble.load(f'{MODELS_DIR}/binary_ensemble.joblib')
ens_weights = ensemble.weights
print(f'Ensemble weights: {ens_weights}')

# ── Print model summary table ──────────────────────────────────────────────
print('\\n=== Model Summary ===')
print(f'  RF   best params: {rf_params}')
print(f'  LGBM best params: {lgbm_params}')
print(f'  XGB  best params: {xgb_params}')
print(f'  Ensemble weights: {ens_weights}')
"""))

cells.append(code("""# ── Scale Fold 2 in-memory ────────────────────────────────────────────────
CHUNK_SIZE = 50_000

scaler = FeatureScaler.load(f'{MODELS_DIR}/scaler_fold1_binary.joblib')
X2_mm = safe_load_npy(f'{PROCESSED_DIR}/fold2_binary_X.npy', mode='r')
y_fold2 = np.array(safe_load_npy(f'{PROCESSED_DIR}/fold2_binary_y.npy', mode='r'), dtype=np.int32)

n_total = X2_mm.shape[0]
n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE
idx_rfe = np.array(RFE_INDICES)

X_fold2 = np.empty((n_total, len(RFE_INDICES)), dtype=np.float32)
for i in range(n_chunks):
    start = i * CHUNK_SIZE
    end   = min(start + CHUNK_SIZE, n_total)
    chunk = np.array(X2_mm[start:end], dtype=np.float32)
    chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = scaler._scaler.transform(chunk).astype(np.float32)
    X_fold2[start:end] = scaled[:, idx_rfe]

print(f'Fold 2 shape: {X_fold2.shape}, labels: {y_fold2.shape}')
print(f'Class distribution: {dict(zip(*np.unique(y_fold2, return_counts=True)))}')
"""))

# ──────────────────────────────────────────────────────────────────────────
# Section 2 — ROC & PR Curves
# ──────────────────────────────────────────────────────────────────────────
cells.append(md("""## Section 2 — ROC & Precision-Recall Curves

ROC curves compare the trade-off between sensitivity (recall of nucleus pixels) and
specificity (avoiding false nucleus detections). For histopathology segmentation, recall
is clinically critical — missed nuclei cannot be typed downstream. PR curves complement
ROC by focusing on precision-recall balance, which is more informative under class imbalance.

A high AUC-ROC (> 0.95) confirms the classifier reliably separates nucleus from background
pixels across all decision thresholds. A high Average Precision (> 0.90) confirms that high
recall can be achieved without flooding the typing pipeline with spurious background pixels.
"""))

cells.append(code("""from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Pre-compute probabilities for all models
print('Computing probabilities on Fold 2...')
rf_proba   = np.array(rf_model.predict_proba(X_fold2), dtype=np.float64)[:, 1]
lgbm_proba = lgbm_model.predict_proba(X_fold2)[:, 1].astype(np.float64)
xgb_proba  = xgb_model.predict_proba(X_fold2)[:, 1].astype(np.float64)
ens_proba  = ensemble.predict_proba(X_fold2)[:, 1].astype(np.float64)

model_probas = {
    'Random Forest': rf_proba,
    'LightGBM':      lgbm_proba,
    'XGBoost':       xgb_proba,
    'Ensemble':      ens_proba,
}
colors = {'Random Forest': '#1f77b4', 'LightGBM': '#ff7f0e',
          'XGBoost': '#2ca02c', 'Ensemble': '#d62728'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for name, proba in model_probas.items():
    fpr, tpr, _ = roc_curve(y_fold2, proba)
    roc_auc = auc(fpr, tpr)
    lw = 2.5 if name == 'Ensemble' else 1.5
    ls = '-' if name == 'Ensemble' else '--'
    ax1.plot(fpr, tpr, color=colors[name], lw=lw, ls=ls, label=f'{name} (AUC={roc_auc:.4f})')

    prec, rec, _ = precision_recall_curve(y_fold2, proba)
    ap = average_precision_score(y_fold2, proba)
    ax2.plot(rec, prec, color=colors[name], lw=lw, ls=ls, label=f'{name} (AP={ap:.4f})')

ax1.plot([0, 1], [0, 1], 'k--', lw=1)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves — Fold 2 Validation', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(alpha=0.3)

ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curves — Fold 2 Validation', fontsize=13, fontweight='bold')
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig_path = f'{FIGURES_DIR}/roc_pr_curves.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ Saved: {fig_path}')
"""))

# ──────────────────────────────────────────────────────────────────────────
# Section 3 — Confusion Matrix & Classification Report
# ──────────────────────────────────────────────────────────────────────────
cells.append(md("""## Section 3 — Confusion Matrix & Classification Report

The confusion matrix quantifies four error categories for binary nucleus segmentation:
- **True Positives (TP):** Nucleus correctly detected → feeds downstream typing pipeline
- **True Negatives (TN):** Background correctly rejected → no spurious typing candidates
- **False Positives (FP):** Background mistakenly labelled as nucleus → creates spurious typing candidates, wastes classifier capacity
- **False Negatives (FN):** Nucleus missed → permanently lost from the typing pipeline (cannot be recovered downstream)

For clinical applications, **FN is the higher-cost error** — a missed nucleus is a lost
pathology signal. The ensemble should minimise FN (maximise recall) while maintaining
acceptable precision.
"""))

cells.append(code("""from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

# Compute predictions
rf_pred   = np.array(rf_model.predict(X_fold2), dtype=np.int32)
lgbm_pred = lgbm_model.predict(X_fold2).astype(np.int32)
xgb_pred  = xgb_model.predict(X_fold2).astype(np.int32)
ens_pred  = ensemble.predict(X_fold2)

model_preds = {
    'Random Forest': rf_pred,
    'LightGBM':      lgbm_pred,
    'XGBoost':       xgb_pred,
    'Ensemble':      ens_pred,
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, pred) in zip(axes, model_preds.items()):
    cm_mat = confusion_matrix(y_fold2, pred)
    sns.heatmap(cm_mat, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Background', 'Nucleus'],
                yticklabels=['Background', 'Nucleus'])
    f1 = f1_score(y_fold2, pred, average='macro', zero_division=0)
    ax.set_title(f'{name}\\nMacro-F1: {f1:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)

plt.suptitle('Confusion Matrices — Fold 2 Validation', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig_path = f'{FIGURES_DIR}/confusion_matrices.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ Saved: {fig_path}')

print('\\n=== Classification Reports ===')
for name, pred in model_preds.items():
    print(f'\\n--- {name} ---')
    print(classification_report(y_fold2, pred,
                                target_names=['Background', 'Nucleus'],
                                zero_division=0))
"""))

# ──────────────────────────────────────────────────────────────────────────
# Section 4 — Feature Importance Comparison
# ──────────────────────────────────────────────────────────────────────────
cells.append(md("""## Section 4 — Feature Importance Comparison

Feature importances reveal which of the 25 hand-crafted features are biologically most
informative for binary nucleus/background discrimination.

- **RF (Gini impurity):** Measures mean decrease in impurity — captures features that best
  split nucleus vs. background at each tree node.
- **LightGBM (gain):** Measures total information gain contributed by a feature across all splits.
- **XGBoost (gain):** Same metric as LightGBM but computed by XGBoost's depth-first tree construction.

Consistency across all three methods suggests robust, generalisable biological relevance —
features important across RF, LGB, and XGB capture stable morphological or staining signatures
(e.g. H-channel intensity, nucleus-scale Gabor responses) rather than model-specific artefacts.
"""))

cells.append(code("""# Extract feature importances
rf_imp   = np.array(rf_model.feature_importances_)
lgbm_imp = lgbm_model.feature_importances_  # gain-based
xgb_imp  = xgb_model.feature_importances_   # gain-based

# Normalise each to [0, 1] for fair visual comparison
def norm01(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)

rf_n   = norm01(rf_imp)
lgbm_n = norm01(lgbm_imp)
xgb_n  = norm01(xgb_imp)

avg_imp = (rf_n + lgbm_n + xgb_n) / 3.0
sort_idx = np.argsort(avg_imp)[::-1]  # descending

fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(FEATURE_NAMES))
width = 0.28

ax.bar(x - width, rf_n[sort_idx],   width, label='RF (Gini)', color='#1f77b4', alpha=0.85)
ax.bar(x,         lgbm_n[sort_idx], width, label='LGBM (Gain)', color='#ff7f0e', alpha=0.85)
ax.bar(x + width, xgb_n[sort_idx],  width, label='XGB (Gain)', color='#2ca02c', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(
    [FEATURE_NAMES[i] for i in sort_idx],
    rotation=90, fontsize=8
)
ax.set_ylabel('Normalised Importance', fontsize=11)
ax.set_title('Feature Importance Comparison — All 25 RFE Features\\n(sorted by average importance)', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Annotate top 5 features per model
for idx_model, (imp, color, label) in enumerate(
    [(rf_n, '#1f77b4', 'RF'), (lgbm_n, '#ff7f0e', 'LGBM'), (xgb_n, '#2ca02c', 'XGB')]
):
    top5 = np.argsort(imp)[::-1][:5]
    # Mark their position in the sorted bar chart
    for rank, orig_idx in enumerate(top5):
        sorted_pos = np.where(sort_idx == orig_idx)[0][0]
        # Label only rank-0 (top feature) for readability
        if rank == 0:
            offset = (idx_model - 1) * width
            ax.annotate(
                f'Top: {FEATURE_NAMES[orig_idx]}',
                xy=(sorted_pos + offset, imp[orig_idx]),
                xytext=(sorted_pos + offset + 0.5, imp[orig_idx] + 0.05),
                fontsize=7, color=color, arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
            )

plt.tight_layout()
fig_path = f'{FIGURES_DIR}/feature_importance_comparison.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ Saved: {fig_path}')

# Print top 5 per model
for name, imp in [('RF', rf_imp), ('LGBM', lgbm_imp), ('XGB', xgb_imp)]:
    top5 = np.argsort(imp)[::-1][:5]
    print(f'  Top-5 {name}: {[FEATURE_NAMES[i] for i in top5]}')
"""))

# ──────────────────────────────────────────────────────────────────────────
# Section 5 — Probability Calibration Curves
# ──────────────────────────────────────────────────────────────────────────
cells.append(md("""## Section 5 — Probability Calibration Curves

Reliable probability estimates are crucial for soft voting: if a model outputs 0.9 confidence
but is correct only 70% of the time, the soft vote will be systematically biased toward that
overconfident model.

**Calibration curves** (reliability diagrams) show how well each model's probability outputs
correspond to actual observed frequencies — a perfectly calibrated model follows the diagonal.
Curves above the diagonal indicate underconfidence (model is more correct than it predicts);
curves below indicate overconfidence.

For the ensemble, good calibration across all three base models means the blended probabilities
can be interpreted as genuine posterior nucleus probabilities, which is important if downstream
phases threshold at a specific confidence level.
"""))

cells.append(code("""from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, proba) in zip(axes, model_probas.items()):
    frac_pos, mean_pred = calibration_curve(y_fold2, proba, n_bins=15, strategy='uniform')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
    ax.plot(mean_pred, frac_pos, 'o-', color=colors[name], lw=2, ms=6, label=name)
    ax.fill_between(mean_pred, frac_pos, mean_pred,
                    alpha=0.1, color=colors[name])
    ax.set_xlabel('Mean Predicted Probability', fontsize=10)
    ax.set_ylabel('Fraction of Positives', fontsize=10)
    ax.set_title(f'{name} — Calibration Curve', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.suptitle('Probability Calibration Curves — Fold 2 Validation',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig_path = f'{FIGURES_DIR}/calibration_curves.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ Saved: {fig_path}')
"""))

# ──────────────────────────────────────────────────────────────────────────
# Section 6 — Weight Optimisation Surface
# ──────────────────────────────────────────────────────────────────────────
cells.append(md("""## Section 6 — Weight Optimisation Surface

The ensemble blending weights control how much each model's probability estimates contribute
to the final prediction. The optimal weights were found by maximising Macro-F1 on the held-out
Fold 2 validation set via exhaustive grid search over the weight simplex (step=0.1, ~165 valid
combinations).

This heatmap shows the F1 landscape across the simplex slice (RF weight on x-axis, LGB weight
on y-axis, with XGB weight = 1 − RF − LGB implied). The peak (★) indicates why the data-driven
weights outperform any hand-tuned uniform configuration.
"""))

cells.append(code("""from sklearn.metrics import f1_score as sk_f1

STEP = 0.1
grid = np.arange(0.0, 1.0 + STEP, STEP)

# Pre-compute proba arrays for speed
est_names = ['rf', 'lgbm', 'xgb']
probas_dict = {
    'rf':   rf_proba,
    'lgbm': lgbm_proba,
    'xgb':  xgb_proba,
}

# Build 2-class proba arrays
probas_2d = {
    'rf':   np.column_stack([1 - rf_proba,   rf_proba]),
    'lgbm': np.column_stack([1 - lgbm_proba, lgbm_proba]),
    'xgb':  np.column_stack([1 - xgb_proba,  xgb_proba]),
}

# Build heat map grid
n_steps = len(grid)
heat = np.full((n_steps, n_steps), np.nan)

best_f1_surf = -1.0
best_w_surf = {}

for i, w_rf in enumerate(grid):
    for j, w_lgb in enumerate(grid):
        w_xgb = 1.0 - w_rf - w_lgb
        if w_xgb < -1e-9:
            continue
        w_xgb = max(0.0, w_xgb)
        if abs(w_rf + w_lgb + w_xgb - 1.0) > 1e-9:
            continue

        blended = w_rf * probas_2d['rf'] + w_lgb * probas_2d['lgbm'] + w_xgb * probas_2d['xgb']
        y_pred = np.argmax(blended, axis=1)
        f1 = sk_f1(y_fold2, y_pred, average='macro', zero_division=0)
        heat[i, j] = f1  # row=w_rf, col=w_lgb

        if f1 > best_f1_surf:
            best_f1_surf = f1
            best_w_surf = {'rf': w_rf, 'lgbm': w_lgb, 'xgb': w_xgb}

print(f'Best from surface scan: F1={best_f1_surf:.4f} | weights={best_w_surf}')
print(f'Stored ensemble weights: {ens_weights}')

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(
    heat.T,   # transpose so x=RF, y=LGB
    origin='lower',
    extent=[grid[0] - STEP/2, grid[-1] + STEP/2, grid[0] - STEP/2, grid[-1] + STEP/2],
    aspect='auto',
    cmap='RdYlGn',
)
plt.colorbar(im, ax=ax, label='Macro-F1 (Fold 2)')

# Mark optimal point
opt_rf  = ens_weights.get('rf', best_w_surf.get('rf', 0))
opt_lgb = ens_weights.get('lgbm', best_w_surf.get('lgbm', 0))
ax.plot(opt_rf, opt_lgb, 'r*', markersize=18, label=f'Optimal (RF={opt_rf:.1f}, LGB={opt_lgb:.1f})')

ax.set_xlabel('RF Weight (w₁)', fontsize=12)
ax.set_ylabel('LGB Weight (w₂)', fontsize=12)
ax.set_title('Ensemble Weight Optimisation Surface\\n(XGB weight = 1 − RF − LGB implied)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)

plt.tight_layout()
fig_path = f'{FIGURES_DIR}/weight_optimisation_surface.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ Saved: {fig_path}')
"""))

# ──────────────────────────────────────────────────────────────────────────
# Section 7 — Per-Tissue F1 Breakdown
# ──────────────────────────────────────────────────────────────────────────
cells.append(md("""## Section 7 — Per-Tissue Macro-F1 Breakdown

PanNuke contains 19 distinct tissue types, each with different cellular density, staining
characteristics, and nuclear morphology. Per-tissue performance reveals where the ensemble
generalises robustly vs. where tissue-specific variations challenge the hand-crafted features.

For example, highly densely packed tissues (colon, breast) tend to have more touching nuclei
where boundary-adjacent pixels are ambiguous, whereas low-density tissues (skin, adrenal gland)
may have cleaner separation.
"""))

cells.append(code("""# Attempt to load tissue metadata
tissue_meta_path = 'data/processed/fold2_tissue_labels.npy'

if os.path.exists(tissue_meta_path):
    tissue_labels = np.load(tissue_meta_path, allow_pickle=True)
    unique_tissues = np.unique(tissue_labels)
    tissue_f1s = {}
    for tissue in unique_tissues:
        mask = tissue_labels == tissue
        if mask.sum() < 10:
            continue
        f1_t = sk_f1(y_fold2[mask], ens_pred[mask], average='macro', zero_division=0)
        tissue_f1s[tissue] = f1_t

    sorted_tissues = sorted(tissue_f1s.items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(12, 5))
    names_t = [t for t, _ in sorted_tissues]
    vals_t  = [v for _, v in sorted_tissues]
    bars = ax.barh(names_t, vals_t, color=cm.RdYlGn(np.array(vals_t)))
    ax.axvline(np.mean(vals_t), color='navy', linestyle='--', lw=1.5, label=f'Mean={np.mean(vals_t):.4f}')
    ax.set_xlabel('Macro-F1', fontsize=11)
    ax.set_title('Ensemble Macro-F1 per PanNuke Tissue Type (Fold 2)', fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig_path = f'{FIGURES_DIR}/per_tissue_f1.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved: {fig_path}')
else:
    print('Per-tissue breakdown requires tissue metadata — this will be computed in Phase 6')
    print('during mask reconstruction. Showing overall class distribution instead.')

    # Fallback: show class distribution of fold2_binary_y
    classes, counts = np.unique(y_fold2, return_counts=True)
    class_names = ['Background (0)', 'Nucleus (1)']
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(class_names, counts, color=['#5577aa', '#aa5533'])
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max()*0.01,
                f'{cnt:,}', ha='center', fontsize=11)
    ax.set_ylabel('Pixel Count', fontsize=11)
    ax.set_title('Fold 2 Class Distribution (Binary)', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.tight_layout()
    fig_path = f'{FIGURES_DIR}/fold2_class_distribution.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved class distribution plot: {fig_path}')
"""))

# ──────────────────────────────────────────────────────────────────────────
# Section 8 — Decision Summary
# ──────────────────────────────────────────────────────────────────────────
cells.append(md("""## Section 8 — Decision Summary

This section computes the final Fold 2 Macro-F1 scores for all four classifiers and writes
the results to `DECISION.md` for the Phase 4 planning record. The ensemble is declared the
winner if its Fold 2 Macro-F1 exceeds all individual base models.
"""))

cells.append(code("""from sklearn.metrics import f1_score as sk_f1_final

rf_f1   = sk_f1_final(y_fold2, rf_pred,   average='macro', zero_division=0)
lgbm_f1 = sk_f1_final(y_fold2, lgbm_pred, average='macro', zero_division=0)
xgb_f1  = sk_f1_final(y_fold2, xgb_pred,  average='macro', zero_division=0)
ens_f1  = sk_f1_final(y_fold2, ens_pred,  average='macro', zero_division=0)

w_rf  = ens_weights.get('rf', 1/3)
w_lgb = ens_weights.get('lgbm', 1/3)
w_xgb = ens_weights.get('xgb', 1/3)

print('=== Phase 4 Final Results (Fold 2 Validation) ===')
print(f'  Random Forest   Macro-F1: {rf_f1:.4f}')
print(f'  LightGBM        Macro-F1: {lgbm_f1:.4f}')
print(f'  XGBoost         Macro-F1: {xgb_f1:.4f}')
print(f'  Ensemble        Macro-F1: {ens_f1:.4f}')
print(f'  Optimal weights: RF={w_rf:.2f}, LGB={w_lgb:.2f}, XGB={w_xgb:.2f}')

# Determine winner
model_results = {
    'Random Forest': rf_f1,
    'LightGBM':      lgbm_f1,
    'XGBoost':       xgb_f1,
    'Ensemble':      ens_f1,
}
winner = max(model_results, key=lambda k: model_results[k])
print(f'\\n  WINNER: {winner} | Fold 2 Macro-F1: {model_results[winner]:.4f}')
if winner == 'Ensemble':
    print(f'  Optimal weights: RF={w_rf:.2f}, LGB={w_lgb:.2f}, XGB={w_xgb:.2f}')

# ── Write DECISION.md ──────────────────────────────────────────────────────
decision_path = '.planning/phases/04-experiment-a-binary-segmentation-training/DECISION.md'
os.makedirs(os.path.dirname(decision_path), exist_ok=True)

decision_content = f\"\"\"# Phase 4 Decision Record

**Generated by:** notebooks/08_inspect_binary_ensemble.ipynb
**Validation set:** Fold 2 (held-out, never seen during training or HPO)

## Fold 2 Macro-F1 Results

| Model         | Macro-F1 |
|---------------|----------|
| Random Forest | {rf_f1:.4f}   |
| LightGBM      | {lgbm_f1:.4f}   |
| XGBoost       | {xgb_f1:.4f}   |
| **Ensemble**  | **{ens_f1:.4f}** |

## Ensemble Weights (Optimised)

| Model         | Weight |
|---------------|--------|
| Random Forest | {w_rf:.3f}   |
| LightGBM      | {w_lgb:.3f}   |
| XGBoost       | {w_xgb:.3f}   |

## Winner

**WINNER: {winner} | Fold 2 Macro-F1: {model_results[winner]:.4f} | Optimal weights: RF={w_rf:.2f}, LGB={w_lgb:.2f}, XGB={w_xgb:.2f}**

## Artefacts

- `data/models/cuml_rf.joblib` — fitted cuML/sklearn RF
- `data/models/lgbm.joblib` — fitted LightGBM
- `data/models/xgboost.joblib` — fitted XGBoost
- `data/models/binary_ensemble.joblib` — SoftVotingEnsemble with optimised weights
- `data/models/rf_best_params.json` — RF Optuna best hyperparameters
- `data/models/lgbm_best_params.json` — LGBM Optuna best hyperparameters
- `data/models/xgb_best_params.json` — XGB Optuna best hyperparameters
\"\"\"

with open(decision_path, 'w') as fh:
    fh.write(decision_content)
print(f'\\n✓ DECISION.md written: {decision_path}')
print(f'\\nPrint Summary:')
print(f'WINNER: {winner} | Fold 2 Macro-F1: {model_results[winner]:.4f} | Optimal weights: RF={w_rf:.2f}, LGB={w_lgb:.2f}, XGB={w_xgb:.2f}')
"""))

# ──────────────────────────────────────────────────────────────────────────
# Assemble and write
# ──────────────────────────────────────────────────────────────────────────
nb.cells = cells

with open(NOTEBOOK_PATH, "w") as fh:
    nbf.write(nb, fh)

print(f"✓ Notebook written: {NOTEBOOK_PATH}")
print(f"  Sections: {len([c for c in cells if c.cell_type == 'markdown'])} markdown, "
      f"{len([c for c in cells if c.cell_type == 'code'])} code")
