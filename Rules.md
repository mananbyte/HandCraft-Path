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