# Final Morgan-RF Model Selection Summary

## What this document is

This is the closing document of the controlled hyperparameter study for the
EGFR Morgan-RF model. It summarizes the validation-based selection process,
locks the final configuration, and reports the single final evaluation on
the standard-scaffold test set.

Important distinction:

* model selection was performed on the **validation split only**;
* the **standard-scaffold test set was not used for model selection during
  the controlled hyperparameter study**;
* after the final test evaluation, no further parameter was changed.

## Validation-based studies (model selection)

### 1. Morgan radius (1, 2, 3)

Validation performance was essentially flat across radii (Spearman
0.769-0.771). Radius 2 was retained because it is the established
reference and there was no material difference.

### 2. Morgan nBits (512, 1024, 2048, 4096)

Validation Spearman ranged only from 0.768 to 0.775. The representation
collision count barely changed (287 groups at 512 bits vs 286 at
1024-4096). nBits = 2048 was retained.

### 3. Tanimoto = 1 audit

27 validation/train pairs had Tanimoto exactly 1.0 but always different
canonical SMILES, InChIKeys, and scaffolds. These are representation
collisions, not data leakage. No split change was made.

### 4. RandomForest min_samples_leaf (1, 2, 4, 8, 16)

Validation Spearman decreased monotonically from 0.7689 (leaf 1) to 0.6923
(leaf 16). The train-validation gap shrank, but only because both train
and validation performance fell. min_samples_leaf = 1 was retained.

### 5. RandomForest max_features (0.2, 0.4, 0.6, 0.8, 1.0)

Validation performance improved monotonically as max_features decreased:
Spearman 0.7689 -> 0.7792 and R2 0.5956 -> 0.6135. Feature-importance mass
became less concentrated (top-10 fraction 0.326 -> 0.226), consistent with
greater tree diversity. Tree complexity barely changed.

### 6. max_features multi-seed confirmation (1.0 / 0.20 / 0.15 / 0.10 x 5 seeds)

The benefit was robust: all 20 configurations were trained on the same
fixed split, and every low max_features value beat 1.0 on all five seeds
for all metrics. Paired Spearman improvement was +0.009 to +0.012 with 5/5
seeds positive; 0.10-0.20 formed a stable region. max_features = 0.20 was
locked as the concrete reference.

### 7. RandomForest max_depth (20, 40, 60, 80, None)

Depth 20 strongly underfit (validation R2 0.582); depth 40 was still
capacity-limited; 60-80-None formed a high-depth plateau (validation R2
0.612-0.614, Spearman 0.779-0.780). The 80 limit was only binding for 63%
of trees. max_depth = None was retained because the finite-depth advantage
was below seed-to-seed noise.

## Locked final configuration

```text
Morgan: radius = 2, nBits = 2048
RandomForestRegressor:
  n_estimators = 300
  random_state = 42
  n_jobs = -1
  min_samples_leaf = 1
  max_features = 0.20
  max_depth = None
  all other parameters = sklearn defaults
```

Split: standard 80/10/10 Bemis-Murcko scaffold split, seed 42 (train 8,128
/ validation 1,015 / test 1,018).

Original baseline for comparison: identical except `max_features = 1.0`.

## Final test evaluation (locked models)

| metric | baseline (mf=1.0) | tuned (mf=0.20) | tuned - baseline | 95% CI of difference |
|---|---:|---:|---:|---:|
| MAE | 0.6494 | 0.6492 | -0.0002 | [-0.0091, 0.0084] |
| RMSE | 0.8832 | 0.8633 | -0.0199 | [-0.0325, -0.0079] |
| R2 | 0.5728 | 0.5918 | +0.0191 | [0.0075, 0.0316] |
| Spearman | 0.7587 | 0.7663 | +0.0076 | [0.0009, 0.0138] |
| ROC-AUC (active >= 7) | 0.8795 | 0.8818 | +0.0023 | [-0.0023, 0.0067] |
| PR-AUC (active >= 7) | 0.8853 | 0.8906 | +0.0052 | [-0.0001, 0.0109] |

Paired bootstrap: 1,000 resamples of the 1,018 test molecules, same seed 42
for resampling, no model retraining.

## What the test result means

The validation benefit transferred to the test set for the core ranking and
regression metrics: RMSE, R2, and Spearman all improved with 95% bootstrap
confidence intervals excluding zero. MAE was unchanged (CI crosses zero),
and the two screening metrics moved in the same direction but were not
clearly significant at the 95% level (PR-AUC lower bound -0.0001). This is
best described as **B. Partial transfer**: the direction of the benefit is
real and consistent with validation, but it is modest and not visible in
every metric.

Important benchmark note: the ROC-AUC / PR-AUC numbers above are on the
standard-scaffold test split and must not be mixed with the M9
retrospective screening benchmark, which uses a different external
screening set.

## Key methodological finding

Feature subsampling robustly improved Random Forest scaffold-held-out
generalization (validation and test), whereas changing fingerprint
scale/dimensionality (radius, nBits) or directly reducing tree complexity
(min_samples_leaf, max_depth) did not. The controlled study therefore
identifies tree diversity as the modeling decision that matters most for
this Morgan-RF setup.

No claim is made that tuning dramatically transformed the model: the
improvement is consistent but small (test R2 +0.019, Spearman +0.008).

## Files

```text
src/models/final_rf_confirmation.py
results/final_rf_configuration.json
results/final_rf_test_results.json
results/final_rf_test_predictions.csv
```

The prediction CSV contains molecule identifier, canonical SMILES, true
pIC50, baseline/tuned predictions, absolute errors, and
`max_train_tanimoto` for all 1,018 test molecules.
