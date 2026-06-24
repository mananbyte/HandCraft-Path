import re

with open("scripts/train_ensemble.py", "r") as f:
    code = f.read()

# Add RMM managed memory and skip-hpo arg
code = code.replace(
    'import cupy',
    'import rmm\nrmm.reinitialize(managed_memory=True)\nimport cupy'
)

code = code.replace(
    '    p.add_argument("--optuna-timeout", type=int, default=1800)\n    return p.parse_args()',
    '    p.add_argument("--optuna-timeout", type=int, default=1800)\n    p.add_argument("--skip-hpo", action="store_true")\n    return p.parse_args()'
)

# Add skip-hpo logic
old_hpo = '    # ── Step 3: HPO subsample ─────────────────────────────────────────────\n    separator(f"Step 3: Stratified subsample for Optuna HPO (n={args.hpo_sample_n:,})")\n    X_sub, y_sub = draw_subsample('

new_hpo = '''    # ── Step 3: HPO subsample ─────────────────────────────────────────────
    separator(f"Step 3: Stratified subsample for Optuna HPO (n={args.hpo_sample_n:,})")
    
    rf_params_path = os.path.join(args.out_dir, "rf_best_params.json")
    lgbm_params_path = os.path.join(args.out_dir, "lgbm_best_params.json")
    xgb_params_path = os.path.join(args.out_dir, "xgb_best_params.json")

    if args.skip_hpo and os.path.exists(rf_params_path) and os.path.exists(lgbm_params_path) and os.path.exists(xgb_params_path):
        import json
        print("  --skip-hpo flag detected and best param JSONs found. Skipping HPO...")
        with open(rf_params_path, "r") as f: rf_best = json.load(f)
        with open(lgbm_params_path, "r") as f: lgbm_best = json.load(f)
        with open(xgb_params_path, "r") as f: xgb_best = json.load(f)
        print("  ✓ Loaded RF best params:", rf_best)
        print("  ✓ Loaded LGBM best params:", lgbm_best)
        print("  ✓ Loaded XGB best params:", xgb_best)
    else:
        X_sub, y_sub = draw_subsample('''

code = code.replace(old_hpo, new_hpo)

# Indent the HPO section
lines = code.split('\n')
out = []
in_else = False
for i, line in enumerate(lines):
    if 'X_sub, y_sub = draw_subsample(' in line:
        in_else = True
        
    if in_else:
        if line.startswith('    # ── Step 7: Refit on full Fold 1 ──────────────────────────────────────'):
            in_else = False
            out.append(line)
        elif line.strip():
            out.append('    ' + line)
        else:
            out.append(line)
    else:
        out.append(line)

code = '\n'.join(out)

# Replace Step 7-11 with out-of-core execution
old_step7_11 = code[code.find('    # ── Step 7: Refit on full Fold 1'):code.find('    t_elapsed = time.time() - t_start')]

new_step7_11 = '''    # ── Step 7: Refit and Predict sequentially (Out-of-Core memory fix) ───
    separator("Step 7: Refit all models on full Fold 1 and evaluate sequentially")
    
    estimators = build_default_estimators(rf_best, lgbm_best, xgb_best)
    
    print("  [Memory] Scaling Fold 2 in-memory...")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_fold{args.fold}_binary.joblib")
    scaler = FeatureScaler.load(scaler_path)
    X2_path = os.path.join(PROCESSED_DIR, "fold2_binary_X.npy")
    y2_path = os.path.join(PROCESSED_DIR, "fold2_binary_y.npy")
    X_fold2 = scale_fold2_inMemory(X2_path, scaler, rfe_indices)
    y_fold2 = np.array(safe_load_npy(y2_path, mode="r"), dtype=np.int32)
    
    fold2_probas = {}
    
    for name, est in estimators.items():
        print(f"\\n05:29:42 | INFO     | src.training.ensemble | Fitting estimator: {name} on {X_fold1.shape[0]} rows")
        est.fit(X_fold1, y_fold1)
        print_rss(f"after fitting {name}")
        
        print(f"  Predicting Fold 2 with {name}...")
        chunk_size = 50_000
        n_rows = X_fold2.shape[0]
        preds = []
        
        import gc
        gc.collect()
        if name == "rf":
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            preds.append(np.array(est.predict_proba(X_fold2[start:end]), dtype=np.float64))
            
        fold2_probas[name] = np.vstack(preds)
        
        model_path = os.path.join(args.out_dir, f"{name}_temp.joblib")
        import joblib
        joblib.dump(est, model_path, compress=1)
        print(f"  Saved {name} temporary model. Freeing memory...")
        
        estimators[name] = None
        gc.collect()
        if name == "rf":
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        print_rss(f"after freeing {name}")

    print("  [Memory] Freeing Fold 1 training data...")
    del X_fold1
    del y_fold1
    try:
        del X_mm
        del y_mm
    except Exception:
        pass
    import gc
    gc.collect()
    print_rss("after freeing fold1 data")

    # ── Step 9: Optimise ensemble weights on Fold 2 ────────────────────────
    separator("Step 9: Optimise ensemble weights via weight simplex grid search")
    
    ensemble = SoftVotingEnsemble(estimators={}, weights=None)
    best_weights = ensemble.optimize_weights_precomputed(fold2_probas, y_fold2)
    print(f"  Optimal weights: {best_weights}")
    
    ensemble.weights = best_weights
    ensemble._fitted = True

    # ── Step 10: Save full models ──────────────────────────────────────────
    separator("Step 10: Save trained base models and ensemble")
    for name in ["rf", "lgbm", "xgb"]:
        temp_path = os.path.join(args.out_dir, f"{name}_temp.joblib")
        estimators[name] = joblib.load(temp_path)
        os.remove(temp_path)
    
    ensemble.estimators = estimators

    rf_path = os.path.join(args.out_dir, "cuml_rf.joblib")
    lgbm_path = os.path.join(args.out_dir, "lgbm.joblib")
    xgb_path = os.path.join(args.out_dir, "xgboost.joblib")
    ensemble_path = os.path.join(args.out_dir, "binary_ensemble.joblib")

    joblib.dump({"model": estimators["rf"], "best_params": rf_best}, rf_path, compress=3)
    joblib.dump({"model": estimators["lgbm"], "best_params": lgbm_best}, lgbm_path, compress=3)
    joblib.dump({"model": estimators["xgb"], "best_params": xgb_best}, xgb_path, compress=3)
    ensemble.save(ensemble_path)

    for p in [rf_path, lgbm_path, xgb_path, ensemble_path]:
        sz = os.path.getsize(p) / 1024 / 1024
        print(f"  ✓ {p} ({sz:.1f} MB)")

    # ── Step 11: Fold 2 evaluation summary ────────────────────────────────
    separator("=== Fold 2 Evaluation Results ===")

    rf_pred = np.argmax(fold2_probas["rf"], axis=1)
    lgbm_pred = np.argmax(fold2_probas["lgbm"], axis=1)
    xgb_pred = np.argmax(fold2_probas["xgb"], axis=1)
    
    ens_probas = best_weights["rf"] * fold2_probas["rf"]
    ens_probas += best_weights["lgbm"] * fold2_probas["lgbm"]
    ens_probas += best_weights["xgb"] * fold2_probas["xgb"]
    ens_pred = np.argmax(ens_probas, axis=1)

    from sklearn.metrics import f1_score
    rf_f1   = f1_score(y_fold2, rf_pred,   average="macro", zero_division=0)
    lgbm_f1 = f1_score(y_fold2, lgbm_pred, average="macro", zero_division=0)
    xgb_f1  = f1_score(y_fold2, xgb_pred,  average="macro", zero_division=0)
    ens_f1  = f1_score(y_fold2, ens_pred,  average="macro", zero_division=0)

    w = best_weights
    print(f"  Random Forest   Macro-F1: {rf_f1:.4f}")
    print(f"  LightGBM        Macro-F1: {lgbm_f1:.4f}")
    print(f"  XGBoost         Macro-F1: {xgb_f1:.4f}")
    print(f"  Ensemble        Macro-F1: {ens_f1:.4f} (weights: RF={w.get('rf', 0):.2f}, LGB={w.get('lgbm', 0):.2f}, XGB={w.get('xgb', 0):.2f})")

'''

code = code.replace(old_step7_11, new_step7_11)

with open("scripts/train_ensemble.py", "w") as f:
    f.write(code)

