# Random Forest max_features Study

## Purpose and protocol

This controlled study changes exactly one modeling decision: the RandomForest
`max_features` fraction (0.2, 0.4, 0.6, 0.8, 1.0).

Everything else is fixed:

* Morgan radius = 2, nBits = 2048, same `GetMorganGenerator` implementation as M2/M3;
* standard 80/10/10 Bemis-Murcko scaffold split, seed 42;
* training set 8,128 molecules; validation set 1,015 molecules;
* **the standard-scaffold test set (1,018 molecules) is never used**;
* n_estimators = 300, random_state = 42, n_jobs = -1, min_samples_leaf = 1,
  max_depth = None;
* all other RandomForest parameters at sklearn defaults.

Primary model-selection metric: validation Spearman. Secondary: validation
RMSE, MAE, R2. Training metrics and gaps are diagnostic only.

## Full effective RandomForestRegressor parameters

The reference configuration (max_features = 1.0) has these effective
parameters:

```text
bootstrap: True
ccp_alpha: 0.0
criterion: squared_error
max_depth: None
max_features: 1.0
max_leaf_nodes: None
max_samples: None
min_impurity_decrease: 0.0
min_samples_leaf: 1
min_samples_split: 2
min_weight_fraction_leaf: 0.0
monotonic_cst: None
n_estimators: 300
n_jobs: -1
oob_score: False
random_state: 42
verbose: 0
warm_start: False
```

Only `max_features` changes across runs.

## Results (validation split only)

| max_features | train MAE | train RMSE | train R2 | train Spearman | val MAE | val RMSE | val R2 | val Spearman | R2 gap | Spearman gap | RMSE gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.1943 | 0.2728 | 0.9595 | 0.9830 | 0.5865 | 0.7612 | 0.6135 | **0.7792** | 0.3461 | 0.2038 | 0.4884 |
| 0.4 | 0.1944 | 0.2733 | 0.9594 | 0.9829 | 0.5834 | 0.7610 | **0.6136** | 0.7788 | 0.3457 | 0.2041 | 0.4877 |
| 0.6 | 0.1951 | 0.2740 | 0.9592 | 0.9827 | 0.5859 | 0.7651 | 0.6094 | 0.7778 | 0.3498 | 0.2048 | 0.4911 |
| 0.8 | 0.1956 | 0.2747 | 0.9590 | 0.9825 | 0.5894 | 0.7679 | 0.6066 | 0.7760 | 0.3524 | 0.2065 | 0.4932 |
| 1.0 | 0.1962 | 0.2761 | 0.9586 | 0.9822 | 0.5947 | 0.7785 | 0.5956 | 0.7689 | 0.3629 | 0.2133 | 0.5025 |

Validation metrics improve monotonically as `max_features` decreases from 1.0
to 0.2; 0.2 and 0.4 are effectively tied on all four validation metrics.

## Tree complexity diagnostics

| max_features | mean depth | median depth | max depth | mean leaves | median leaves | max leaves |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 86.6 | 84.5 | 138 | 4,725 | 4,729 | 4,799 |
| 0.4 | 86.3 | 84.0 | 156 | 4,696 | 4,696 | 4,780 |
| 0.6 | 85.7 | 82.5 | 131 | 4,684 | 4,686 | 4,758 |
| 0.8 | 84.3 | 82.5 | 144 | 4,679 | 4,682 | 4,757 |
| 1.0 | 84.1 | 81.0 | 136 | 4,674 | 4,675 | 4,755 |

Feature subsampling leaves tree size almost unchanged. Trees at 0.2 are only
about 2.4 levels deeper and have about 51 more leaves on average than at 1.0.
The effect on model complexity is therefore small.

## Feature-importance diagnostics

| max_features | top-10 importance | top-50 importance | non-zero features |
|---:|---:|---:|---:|
| 0.2 | 0.2258 | 0.3758 | 2,048 |
| 0.4 | 0.2633 | 0.4122 | 2,048 |
| 0.6 | 0.2847 | 0.4326 | 2,048 |
| 0.8 | 0.2992 | 0.4494 | 2,048 |
| 1.0 | 0.3253 | 0.4736 | 2,048 |

At max_features = 1.0 the top 10 bits carry 32.5% of aggregate importance;
at 0.2 they carry only 22.6%. Stronger feature subsampling spreads importance
mass across more bits. All 2,048 bits have non-zero aggregate importance in
every setting, so the fingerprint is not reduced to a small active subset.

## What max_features controls

`max_features` is the number of fingerprint bits each split may consider.
With max_features = 1.0, every split can evaluate all 2,048 bits; with 0.2,
each split randomly samples about 409 bits. Lower values add randomness to
each tree and are expected to make trees less correlated with one another.

## Why feature subsampling could help

Random Forest prediction error depends on both individual-tree accuracy and
correlation between trees. If all trees repeatedly choose the same strong
bits, they can become similar and the ensemble behaves more like one large
tree. Feature subsampling forces trees to use different bit subsets, which
can add diversity. Too much subsampling can instead prevent trees from
accessing informative bits and cause underfitting.

## Training performance

Training metrics are almost flat across the range: train R2 goes from 0.9586
to 0.9595 and train Spearman from 0.9822 to 0.9830. The forest still fits
the training set essentially the same way at every setting, so the
validation differences are not explained by a loss of training fit.

## Validation performance

Validation performance improves modestly and monotonically as feature
subsampling increases:

* validation Spearman: 0.7689 -> 0.7792;
* validation R2: 0.5956 -> 0.6136;
* validation MAE: 0.5947 -> 0.5834;
* validation RMSE: 0.7785 -> 0.7610.

0.2 and 0.4 are nearly identical, with 0.4 marginally better on MAE/RMSE/R2
and 0.2 marginally better on Spearman.

## Generalization gaps

The train-validation gap shrinks modestly as feature subsampling increases:

* Spearman gap: 0.2133 -> 0.2038;
* R2 gap: 0.3629 -> 0.3461;
* RMSE gap: 0.5025 -> 0.4884.

Unlike the min_samples_leaf study, the gap here shrinks while absolute
validation performance improves, which is the direction that would be
expected from added tree diversity.

## Does feature subsampling change tree complexity?

Almost not. Mean depth changes from 84.1 (1.0) to 86.6 (0.2) and mean leaves
from 4,674 to 4,725. The small complexity increase is a secondary effect of
fewer available features per split; it does not explain the validation gain.

## Do regression and ranking metrics agree?

Yes, with one trivial tie. Validation R2 ranks `0.4 > 0.2 > 0.6 > 0.8 > 1.0`
while validation Spearman ranks `0.2 > 0.4 > 0.6 > 0.8 > 1.0`; both put the
two low-fraction settings clearly ahead of 1.0 and agree that 1.0 is worst.
MAE and RMSE also improve monotonically from 1.0 down to 0.2/0.4.

## Effect size

The improvement is real but small: validation Spearman +0.010 and R2 +0.018
at 0.2/0.4 relative to 1.0, with MAE about -0.008 and RMSE about -0.017.
This is a single-seed, single-split comparison, so the difference is not
large enough by itself to justify changing the final model.

## Decision classification

**D. Lower max_features continues improving.** The best ranking performance
occurs at the lowest tested boundary (0.2), and validation metrics improve
monotonically as max_features falls. The trend suggests 0.05-0.15 may be
worth a separate, validation-only follow-up experiment.

## Recommended next action

Do not change the final model or README headline metrics yet. If a follow-up
is desired, run one controlled validation-only study at max_features =
0.05 / 0.10 / 0.15 using the same split, fingerprints, and other RF
parameters, and check whether the improvement continues or plateaus before
changing the reference configuration.

## Files

```text
src/models/max_features_tuning_study.py
results/max_features_tuning_results.csv
results/max_features_tuning_results.json
results/figures/max_features_train_val.png
results/figures/max_features_validation.png
```

## Reproducibility

```bash
conda activate egfr-aidd
python src/models/max_features_tuning_study.py
```

The JSON records the split, the full effective RF parameter dictionary,
per-setting tree complexity, feature-importance diagnostics, package
versions, and all train/validation metrics. The standard-scaffold test split
was not evaluated.
