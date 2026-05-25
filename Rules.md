# HandCraft-Path Rules

## Code
- Python only — no notebooks for pipeline code (notebooks for inspection only)
- Type hints on every function signature
- No magic numbers — every threshold, radius, or count gets a named constant
- Keep files under 300 LOC — split into focused modules
- No in-place mutation of numpy arrays without explicit comment explaining why
- Every function that touches data must have an assert on input shape

## Data integrity
- Never load full .npy files without mmap_mode='r'
- Never write to data/raw/ — it is read-only
- Never fit any scaler, selector, or normalizer on Fold3 data
- Fold3 is touched exactly once: final evaluation only
- Always flush memmap before writing checkpoint

## Pipeline order (never violate)
- normalize → color convert → extract dense features → sample → GLCM → concatenate
- Scaler must be fit before selector — never the reverse
- Inference order must match training order exactly

## Saving artifacts
- Every fitted object (scaler, selector, normalizer, model) is saved with joblib immediately after fitting
- Filename must include experiment mode: e.g. scaler_binary.joblib, rf_3class_tuned.joblib
- Never overwrite a tuned model file with an untuned one

## Experiment discipline
- Experiment A = binary labels (0=bg, 1=nucleus)
- Experiment B = 3-class labels (0=bg, 1=interior, 2=boundary)
- Stage 1 and Stage 2 are independent — never mix their features or labels
- Oracle evaluation uses GT masks — pipeline evaluation uses Stage 1 predictions

## Reproducibility
- Every random operation uses random_state=42
- numpy random seed set at top of every script: np.random.seed(42)
- Every script logs: Python version, library versions, start time, args used

## Git & Version Control
- **Branch Hygiene**: Never work or commit directly on the `main` branch. All development must occur on focused, task-specific feature branches (e.g. `feature/glcm-optimization`).
- **Atomic & Meaningful Commits**: Every commit must represent a single, self-contained logical change (e.g. one feature, one bugfix, one optimization).
- **Descriptive Imperative Messages**: Commits must use clean, standardized prefixes and imperatively-phrased messages:
  - `feat:` for new capabilities (e.g. `feat: parallelize GLCM textures via thread pool`)
  - `fix:` for bug fixes (e.g. `fix: handle NumPy 2.x header serialization in checkpoint load`)
  - `perf:` for performance optimizations
  - `docs:` for documentation updates
  - `test:` for adding or modifying unit tests
- **Zero Repo Litter (Strictly Enforced)**: 
  - Never stage or commit large binary arrays (`.npy`), model weights (`.joblib`), temporary JSON checkpoints, local Conda environments, or debug logs.
  - Double check `git status` before running `git commit` to ensure no ignored local files are tracked.
- **Pristine History**: Remove all print debug statements, scratch scripts, and temporary scratch files before staging changes. Keep the repository history clean, lean, and production-ready.