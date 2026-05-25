"""
Script to programmatically create the 06_inspect_feature_scaler.ipynb notebook
using nbformat, then save it ready for execution.
"""

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import os

cells = []

# ── Cell 0: Setup & imports ──────────────────────────────────────────────────
cells.append(new_code_cell(
    "import sys, os\n"
    "import json\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "PROJECT_ROOT = os.path.abspath('..')\n"
    "if PROJECT_ROOT not in sys.path:\n"
    "    sys.path.insert(0, PROJECT_ROOT)\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import psutil\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.metrics import f1_score\n"
    "\n"
    "from src.utils.safe_loader import safe_load_npy\n"
    "from src.training.scaler import FeatureScaler, SCALER_REGISTRY\n"
    "from src.training.feature_selection import get_feature_names\n"
    "\n"
    "FIGURES_DIR = 'figures'\n"
    "os.makedirs(FIGURES_DIR, exist_ok=True)\n"
    "\n"
    "PHASE_DIR = os.path.join(\n"
    "    PROJECT_ROOT,\n"
    "    '.planning', 'phases', '03-dataset-standardization-mrmr-selection'\n"
    ")\n"
    "\n"
    "print('Setup complete.')\n"
    "print(f'Project root: {PROJECT_ROOT}')\n"
    "print(f'CWD: {os.getcwd()}')\n"
))

# ── Cell 1: Load data ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## Section 1 — Data Loading & Stratified Subsample"))

cells.append(new_code_cell(
    "X_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'fold1_binary_X.npy')\n"
    "Y_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'fold1_binary_y.npy')\n"
    "\n"
    "print('Loading feature matrix (mmap, read-only)...')\n"
    "X_mm = safe_load_npy(X_PATH, mode='r')\n"
    "y_mm = safe_load_npy(Y_PATH, mode='r')\n"
    "\n"
    "print(f'X shape: {X_mm.shape}, dtype: {X_mm.dtype}')\n"
    "print(f'y shape: {y_mm.shape}, dtype: {y_mm.dtype}')\n"
    "\n"
    "classes, counts = np.unique(y_mm, return_counts=True)\n"
    "print('Class distribution:')\n"
    "for c, n in zip(classes, counts):\n"
    "    print(f'  class {c}: {n:,}  ({n/len(y_mm)*100:.1f}%)')\n"
))

cells.append(new_code_cell(
    "SAMPLE_N = 50_000\n"
    "RANDOM_STATE = 42\n"
    "\n"
    "print(f'Drawing stratified subsample of {SAMPLE_N:,} rows...')\n"
    "_, sub_idx = train_test_split(\n"
    "    np.arange(len(y_mm)),\n"
    "    test_size=SAMPLE_N,\n"
    "    stratify=np.array(y_mm),\n"
    "    random_state=RANDOM_STATE,\n"
    ")\n"
    "sub_idx = np.sort(sub_idx)\n"
    "\n"
    "X_sub = np.array(X_mm[sub_idx], dtype=np.float32)\n"
    "y_sub = np.array(y_mm[sub_idx], dtype=np.int32)\n"
    "\n"
    "X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)\n"
    "\n"
    "print(f'Subsample: X={X_sub.shape}, y={y_sub.shape}')\n"
    "classes_sub, counts_sub = np.unique(y_sub, return_counts=True)\n"
    "print('Subsample class distribution:')\n"
    "for c, n in zip(classes_sub, counts_sub):\n"
    "    print(f'  class {c}: {n:,}  ({n/len(y_sub)*100:.1f}%)')\n"
    "\n"
    "rss_after_load = psutil.Process().memory_info().rss / 1024 / 1024\n"
    "print(f'RSS after loading subsample: {rss_after_load:.1f} MB')\n"
))

# ── Cell 2: Feature group index map ──────────────────────────────────────────
cells.append(new_markdown_cell("## Section 2 — Feature Group Index Map"))

cells.append(new_code_cell(
    "feature_names = get_feature_names()\n"
    "print(f'Total features: {len(feature_names)}')\n"
    "\n"
    "FEATURE_GROUPS = {\n"
    "    'OD (3)':             list(range(0, 3)),\n"
    "    'Color stats (54)':   list(range(3, 57)),\n"
    "    'LBP (3)':            list(range(57, 60)),\n"
    "    'Gabor (12)':         list(range(60, 72)),\n"
    "    'Gradient (5)':       list(range(72, 77)),\n"
    "    'Struct tensor (3)':  list(range(77, 80)),\n"
    "    'DoG (3)':            list(range(80, 83)),\n"
    "    'Superpixel (2)':     list(range(83, 85)),\n"
    "    'Entropy (1)':        [85],\n"
    "    'Edge dist (1)':      [86],\n"
    "    'GLCM (6)':           list(range(87, 93)),\n"
    "}\n"
    "\n"
    "all_cols = sorted([c for cols in FEATURE_GROUPS.values() for c in cols])\n"
    "assert all_cols == list(range(93)), f'Mismatch! Got {len(all_cols)} indices.'\n"
    "print('Feature group map verified — all 93 indices covered.')\n"
    "\n"
    "rows = [(grp, cols[0], cols[-1], len(cols)) for grp, cols in FEATURE_GROUPS.items()]\n"
    "df_groups = pd.DataFrame(rows, columns=['Group', 'Start', 'End', 'Count'])\n"
    "print(df_groups.to_string(index=False))\n"
))

# ── Cell 3: Fit all 5 scalers ─────────────────────────────────────────────────
cells.append(new_markdown_cell("## Section 3 — Fit All 5 Scalers on Subsample"))

cells.append(new_code_cell(
    "SCALER_NAMES = [\n"
    "    'StandardScaler',\n"
    "    'RobustScaler',\n"
    "    'QuantileTransformer_uniform',\n"
    "    'QuantileTransformer_normal',\n"
    "    'PowerTransformer',\n"
    "]\n"
    "\n"
    "results = {}\n"
    "\n"
    "for sname in SCALER_NAMES:\n"
    "    rss_before = psutil.Process().memory_info().rss / 1024 / 1024\n"
    "    fs = FeatureScaler(scaler_type=sname)\n"
    "    fs._scaler.fit(X_sub)\n"
    "    rss_after = psutil.Process().memory_info().rss / 1024 / 1024\n"
    "    X_scaled = fs._scaler.transform(X_sub).astype(np.float32)\n"
    "    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)\n"
    "    rss_delta = rss_after - rss_before\n"
    "    results[sname] = {\n"
    "        'scaler': fs,\n"
    "        'X_scaled': X_scaled,\n"
    "        'rss_before': rss_before,\n"
    "        'rss_after': rss_after,\n"
    "        'rss_delta': rss_delta,\n"
    "    }\n"
    "    print(f'[{sname}] RSS: {rss_before:.1f} -> {rss_after:.1f} MB  (delta {rss_delta:+.1f} MB)')\n"
    "\n"
    "print('All 5 scalers fitted.')\n"
))

# ── Cell 4: Distribution histograms ──────────────────────────────────────────
cells.append(new_markdown_cell("## Section 4 — Distribution Histograms (Representative Features)"))

cells.append(new_code_cell(
    "REP_FEATURES = {\n"
    "    'OD[0] (od_R)':             0,\n"
    "    'GLCM[0] (contrast)':       87,\n"
    "    'LBP[0] (lbp_r1)':          57,\n"
    "    'Gabor[0] (gabor_f0.1_t0)': 60,\n"
    "}\n"
    "\n"
    "COLORS = ['steelblue', 'darkorange', 'green', 'crimson', 'purple']\n"
    "\n"
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
    "axes = axes.ravel()\n"
    "\n"
    "for ax_idx, (feat_label, col_idx) in enumerate(REP_FEATURES.items()):\n"
    "    ax = axes[ax_idx]\n"
    "    for s_idx, sname in enumerate(SCALER_NAMES):\n"
    "        vals = results[sname]['X_scaled'][:, col_idx]\n"
    "        lo, hi = np.percentile(vals, [0.5, 99.5])\n"
    "        vals_clipped = np.clip(vals, lo, hi)\n"
    "        sns.kdeplot(vals_clipped, ax=ax, label=sname, color=COLORS[s_idx], linewidth=1.5)\n"
    "    ax.set_title(feat_label, fontsize=11)\n"
    "    ax.set_xlabel('Scaled value')\n"
    "    ax.set_ylabel('Density')\n"
    "    ax.legend(fontsize=7, loc='upper right')\n"
    "\n"
    "plt.suptitle('Feature Distributions After Scaling (50k subsample)', fontsize=13, y=1.01)\n"
    "plt.tight_layout()\n"
    "fig_path = os.path.join(FIGURES_DIR, 'scaler_distributions.png')\n"
    "plt.savefig(fig_path, dpi=120, bbox_inches='tight')\n"
    "plt.close()\n"
    "print(f'Distribution plot saved -> {fig_path}')\n"
))

# ── Cell 5: Outlier fraction table ────────────────────────────────────────────
cells.append(new_markdown_cell("## Section 5 — Outlier Fraction Table (|z| > 3)"))

cells.append(new_code_cell(
    "outlier_rows = []\n"
    "\n"
    "for sname in SCALER_NAMES:\n"
    "    X_scaled = results[sname]['X_scaled']\n"
    "    row = {'Scaler': sname}\n"
    "    for grp, cols in FEATURE_GROUPS.items():\n"
    "        frac = float(np.mean(np.abs(X_scaled[:, cols]) > 3.0))\n"
    "        row[grp] = frac\n"
    "    row['Global'] = float(np.mean(np.abs(X_scaled) > 3.0))\n"
    "    outlier_rows.append(row)\n"
    "\n"
    "df_outlier = pd.DataFrame(outlier_rows).set_index('Scaler')\n"
    "print('Outlier Fraction (|z| > 3):')\n"
    "print(df_outlier.map(lambda x: f'{x:.4f}').to_string())\n"
    "\n"
    "results_outlier = {sname: df_outlier.loc[sname, 'Global'] for sname in SCALER_NAMES}\n"
    "print('\\nGlobal outlier fractions:')\n"
    "for sname, frac in results_outlier.items():\n"
    "    print(f'  {sname}: {frac:.4f}  ({frac*100:.2f}%)')\n"
))

# ── Cell 6: Proxy classifier F1 ──────────────────────────────────────────────
cells.append(new_markdown_cell("## Section 6 — Proxy Classifier F1 (LogReg 80/20 Split)"))

cells.append(new_code_cell(
    "X_tr, X_val_split, y_tr, y_val_split = train_test_split(\n"
    "    X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=RANDOM_STATE\n"
    ")\n"
    "\n"
    "f1_scores = {}\n"
    "comparison_rows = []\n"
    "\n"
    "for sname in SCALER_NAMES:\n"
    "    fs = results[sname]['scaler']\n"
    "    X_tr_s  = fs._scaler.transform(X_tr).astype(np.float32)\n"
    "    X_val_s = fs._scaler.transform(X_val_split).astype(np.float32)\n"
    "    X_tr_s  = np.nan_to_num(X_tr_s,  nan=0.0, posinf=0.0, neginf=0.0)\n"
    "    X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)\n"
    "\n"
    "    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE, n_jobs=-1)\n"
    "    lr.fit(X_tr_s, y_tr)\n"
    "    y_pred = lr.predict(X_val_s)\n"
    "    macro_f1 = float(f1_score(y_val_split, y_pred, average='macro'))\n"
    "    f1_scores[sname] = macro_f1\n"
    "\n"
    "    global_outlier = results_outlier[sname]\n"
    "    rss_delta = results[sname]['rss_delta']\n"
    "    comparison_rows.append({\n"
    "        'Scaler': sname,\n"
    "        'Macro-F1': macro_f1,\n"
    "        'Outlier_pct': global_outlier * 100,\n"
    "        'RSS_delta_MB': rss_delta,\n"
    "    })\n"
    "    print(f'[{sname}] Macro-F1={macro_f1:.4f}  Outlier={global_outlier*100:.3f}%  RSS_delta={rss_delta:+.1f} MB')\n"
    "\n"
    "df_comparison = pd.DataFrame(comparison_rows)\n"
    "print('\\nComparison Table:')\n"
    "print(df_comparison.to_string(index=False))\n"
))

# ── Cell 7: Decision ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## Section 7 — Decision"))

cells.append(new_code_cell(
    "ROBUST_NAME = 'RobustScaler'\n"
    "MARGIN = 0.005\n"
    "\n"
    "best_f1_name = max(f1_scores, key=lambda k: f1_scores[k])\n"
    "best_f1_val  = f1_scores[best_f1_name]\n"
    "robust_f1    = f1_scores[ROBUST_NAME]\n"
    "\n"
    "best_outlier_name = min(results_outlier, key=lambda k: results_outlier[k])\n"
    "best_outlier_val  = results_outlier[best_outlier_name]\n"
    "robust_outlier    = results_outlier[ROBUST_NAME]\n"
    "\n"
    "print(f'RobustScaler F1:      {robust_f1:.4f}')\n"
    "print(f'Best F1 scaler:       {best_f1_name} ({best_f1_val:.4f})')\n"
    "print(f'RobustScaler outlier: {robust_outlier:.4f}')\n"
    "print(f'Best outlier scaler:  {best_outlier_name} ({best_outlier_val:.4f})')\n"
    "\n"
    "f1_margin = best_f1_val - robust_f1\n"
    "if (best_f1_name != ROBUST_NAME\n"
    "        and f1_margin >= MARGIN\n"
    "        and results_outlier[best_f1_name] < robust_outlier):\n"
    "    winner = best_f1_name\n"
    "    override_reason = (\n"
    "        f'Override: {winner} beats RobustScaler on both F1 '\n"
    "        f'(margin={f1_margin:.4f}>=0.005) and outlier fraction.'\n"
    "    )\n"
    "    print(f'\\n*** OVERRIDE: Winner = {winner} ***')\n"
    "    print(override_reason)\n"
    "else:\n"
    "    winner = ROBUST_NAME\n"
    "    override_reason = (\n"
    "        f'RobustScaler retained as default. '\n"
    "        f'Best alternative F1 margin = {f1_margin:.4f} (threshold 0.005). '\n"
    "        f'No challenger meets both override conditions.'\n"
    "    )\n"
    "    print(f'\\n*** DEFAULT RETAINED: Winner = {winner} ***')\n"
    "    print(override_reason)\n"
    "\n"
    "winner_f1      = f1_scores[winner]\n"
    "winner_outlier = results_outlier[winner]\n"
    "winner_rss     = results[winner]['rss_delta']\n"
    "\n"
    "summary_lines = [\n"
    "    f'WINNER: {winner}',\n"
    "    f'  Macro-F1:               {winner_f1:.4f}',\n"
    "    f'  Global outlier fraction: {winner_outlier:.4f}  ({winner_outlier*100:.2f}%)',\n"
    "    f'  RSS delta during fit:   {winner_rss:.1f} MB',\n"
    "]\n"
    "print('\\n' + '\\n'.join(summary_lines))\n"
))

cells.append(new_code_cell(
    "import re\n"
    "\n"
    "decision_md_path = os.path.join(PHASE_DIR, 'DECISION.md')\n"
    "with open(decision_md_path, 'r') as fh:\n"
    "    content = fh.read()\n"
    "\n"
    "# Build table rows\n"
    "table_rows = ''\n"
    "for sname in SCALER_NAMES:\n"
    "    outlier_pct = f'{results_outlier[sname]*100:.3f}%'\n"
    "    f1_val = f'{f1_scores[sname]:.4f}'\n"
    "    if sname == winner:\n"
    "        is_winner = 'YES'\n"
    "        table_rows += f'| **{sname}** | **{outlier_pct}** | **{f1_val}** | **{is_winner}** |\\n'\n"
    "    elif sname == 'RobustScaler':\n"
    "        is_winner = 'DEFAULT'\n"
    "        table_rows += f'| {sname} | {outlier_pct} | {f1_val} | {is_winner} |\\n'\n"
    "    else:\n"
    "        is_winner = 'no'\n"
    "        table_rows += f'| {sname} | {outlier_pct} | {f1_val} | {is_winner} |\\n'\n"
    "\n"
    "new_table = (\n"
    "    '| Scaler | Outlier Frac | Proxy F1 | Winner? |\\n'\n"
    "    '|---|---|---|---|\\n'\n"
    "    + table_rows.rstrip('\\n')\n"
    ")\n"
    "\n"
    "# Replace candidates table block\n"
    "old_table_pattern = r'(\\*\\*Candidates evaluated:\\*\\*\\n)\\| Scaler.*?(?=\\n\\n)'\n"
    "replacement_block = r'\\g<1>' + new_table\n"
    "content = re.sub(old_table_pattern, replacement_block, content, flags=re.DOTALL)\n"
    "\n"
    "# Replace status line\n"
    "content = content.replace(\n"
    "    '**Status:** PENDING — to be filled after `notebooks/06_inspect_feature_scaler.ipynb` runs.',\n"
    "    '**Status:** CONFIRMED — filled by `notebooks/06_inspect_feature_scaler.ipynb`.'\n"
    ")\n"
    "\n"
    "# Replace final choice placeholder\n"
    "content = content.replace(\n"
    "    '**Final choice:** `[TO BE FILLED BY NOTEBOOK]`',\n"
    "    f'**Final choice:** `{winner}`\\n\\n**Override note:** {override_reason}'\n"
    ")\n"
    "\n"
    "with open(decision_md_path, 'w') as fh:\n"
    "    fh.write(content)\n"
    "\n"
    "print(f'DECISION.md updated at: {decision_md_path}')\n"
    "# Print Decision 1 section\n"
    "lines = content.split('\\n')\n"
    "in_d1 = False\n"
    "for line in lines:\n"
    "    if '## Decision 1' in line:\n"
    "        in_d1 = True\n"
    "    if in_d1:\n"
    "        if '## Decision 2' in line:\n"
    "            break\n"
    "        print(line)\n"
))

cells.append(new_code_cell(
    "results_json = {\n"
    "    'winner': winner,\n"
    "    'override_reason': override_reason,\n"
    "    'f1_scores': {k: round(v, 6) for k, v in f1_scores.items()},\n"
    "    'global_outlier_fractions': {k: round(v, 6) for k, v in results_outlier.items()},\n"
    "    'rss_delta_mb': {k: round(results[k]['rss_delta'], 2) for k in SCALER_NAMES},\n"
    "}\n"
    "\n"
    "json_path = os.path.join(PHASE_DIR, 'scaler_results.json')\n"
    "with open(json_path, 'w') as fh:\n"
    "    json.dump(results_json, fh, indent=2)\n"
    "\n"
    "print(f'scaler_results.json written -> {json_path}')\n"
    "print(json.dumps(results_json, indent=2))\n"
))

# ── Build and write the notebook ──────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0",
    }
})

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '06_inspect_feature_scaler.ipynb')
with open(out_path, 'w') as fh:
    nbformat.write(nb, fh)

print(f"Notebook written: {out_path}")
