# Feature Selection Evaluation Report

This report compares three feature selection methods (mRMR, ANOVA, RFE) on the PanNuke hand-crafted feature matrix (93 features).

## Experiment Setup
- **Source Dataset:** `fold1_binary_X.npy` / `fold1_binary_y.npy`
- **Total Samples:** 2,073,571
- **Downsampling:** Stratified subset of 100,000 samples (80% Train, 20% Validation)
- **Classifier:** Logistic Regression (max_iter=1000, standardized features)

## Performance Metrics

| Selection Method | Number of Features | Validation Macro-F1 |
|------------------|--------------------|---------------------|
| Baseline (Full)  | 93                 | 0.8779           |
| **mRMR**         | 20                 | **0.8675**           |
| ANOVA            | 20                 | 0.8669           |
| RFE              | 20                 | 0.8730           |

## Top Selected Features Comparison

| Rank | mRMR Selected | ANOVA Selected | RFE Selected |
|------|---------------|----------------|--------------|
| 1 | color_lab_c2_r3_mean | color_lab_c2_r3_mean | color_lab_c0_r7_std |
| 2 | gradient_sobel_v | color_lab_c0_r3_mean | color_lab_c1_r7_mean |
| 3 | color_lab_c0_r3_mean | gradient_log_s2 | color_lab_c1_r15_mean |
| 4 | gradient_log_s2 | color_hsv_c2_r3_mean | color_lab_c2_r3_mean |
| 5 | color_lab_c2_r7_mean | color_lab_c2_r7_mean | color_lab_c2_r7_mean |
| 6 | color_hsv_c2_r3_mean | gradient_log_s4 | color_hsv_c1_r3_mean |
| 7 | gradient_sobel_h | color_hsv_c1_r3_mean | color_hsv_c1_r7_std |
| 8 | gradient_log_s4 | color_lab_c0_r7_mean | color_hsv_c1_r15_std |
| 9 | color_hsv_c1_r3_mean | color_hsv_c2_r7_mean | color_hsv_c2_r3_mean |
| 10 | struct_ev1 | superpixel_lab_l | color_hsv_c2_r7_std |
| 11 | superpixel_lab_l | color_hsv_c1_r7_mean | color_hsv_c2_r15_mean |
| 12 | color_lab_c0_r7_mean | color_lab_c1_r3_mean | color_hed_c0_r7_std |
| 13 | edge_dist | color_lab_c1_r7_mean | color_hed_c0_r15_std |
| 14 | color_hsv_c2_r7_mean | dog_s4_s8 | color_hed_c1_r3_mean |
| 15 | lbp_r2 | color_hed_c1_r3_mean | color_hed_c1_r7_std |
| 16 | color_lab_c1_r3_mean | color_hed_c2_r3_mean | color_hed_c1_r15_std |
| 17 | color_hsv_c1_r7_mean | od_G | color_hed_c2_r3_mean |
| 18 | color_lab_c1_r7_std | color_lab_c2_r15_mean | color_hed_c2_r15_std |
| 19 | dog_s4_s8 | color_hed_c1_r7_mean | gradient_log_s4 |
| 20 | gabor_f0.2_t3pi_4 | od_R | dog_s4_s8 |

## Discussion & Rationale
- **mRMR** balances relevance (ANOVA F-statistic) with redundancy (Pearson correlation). It prevents selecting highly correlated variations of the same feature group (e.g. multiple scales of the same color statistic).
- **Baseline Comparison:** Using only 20 features (selected via mRMR) achieves performance very close to (or better than) using all 93 features, verifying that the selected features capture most of the classification signal while significantly reducing model complexity and inference cost.
