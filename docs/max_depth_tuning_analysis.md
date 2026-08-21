# Random Forest max_depth Study

## Purpose and protocol

This is the final RF hyperparameter study before final model confirmation.
Exactly one modeling decision changes: `max_depth` (20, 40, 60, 80, None).

Everything else is fixed:

* Morgan radius = 2, nBits = 2048, same `GetMorganGenerator` implementation as M2/M3;
* standard 80/10/10 Bemis-Murcko scaffold split, seed 42;
* training set 8,128 molecules; validation set 1,015 molecules;
* **the standard-scaffold test set (1,018 molecules) is never used**;
* n_estimators = 300, random_state = 42, n_jobs = -1, min_samples_leaf = 1,
  max_features = 0.20;
* all other RandomForest parameters at sklearn defaults.

Primary metric: validation Spearman. Secondary: validation R2, RMSE, MAE.

## Full effective RandomForestRegressor parameters

The reference configuration (max_depth = None) has these effective
parameters:

```text
bootstrap: True
ccp_alpha: 0.0
criterion: squared_error
max_depth: None
max_features: 0.2
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

Only `max_depth` changes across runs.

## Results (validation split only)

| max_depth | train MAE | train RMSE | train R2 | train Spearman | val MAE | val RMSE | val R2 | val Spearman | R2 gap | Spearman gap | RMSE gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.3897 | 0.5109 | 0.8581 | 0.9355 | 0.6178 | 0.7915 | 0.5820 | 0.7597 | 0.2760 | 0.1758 | 0.2806 |
| 40 | 0.2281 | 0.3103 | 0.9476 | 0.9775 | 0.5872 | 0.7621 | 0.6125 | 0.7781 | 0.3351 | 0.1994 | 0.4518 |
| 60 | 0.1992 | 0.2774 | 0.9582 | 0.9823 | **0.5864** | **0.7606** | **0.6140** | **0.7800** | 0.3441 | 0.2023 | 0.4832 |
| 80 | 0.1950 | 0.2732 | 0.9594 | 0.9830 | 0.5869 | 0.7628 | 0.6118 | 0.7789 | 0.3476 | 0.2040 | 0.4896 |
| None | 0.1943 | 0.2728 | 0.9595 | 0.9830 | 0.5865 | 0.7612 | 0.6135 | 0.7792 | 0.3461 | 0.2038 | 0.4884 |

Validation performance rises sharply from depth 20 to 40, is still slightly
lower at 40 than at 60+, and is effectively identical for 60, 80, and None.

## Tree complexity and depth-limit saturation

| max_depth | mean depth | median depth | max depth | mean leaves | median leaves | mean samples/leaf | trees hitting limit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20.0 | 20 | 20 | 1,877 | 1,871 | 4.37 | 100% |
| 40 | 40.0 | 40 | 40 | 3,981 | 3,996 | 2.05 | 100% |
| 60 | 60.0 | 60 | 60 | 4,570 | 4,600 | 1.78 | 100% |
| 80 | 77.8 | 80 | 80 | 4,696 | 4,708 | 1.73 | 63.3% |
| None | 86.6 | 84.5 | 138 | 4,725 | 4,729 | 1.72 | n/a |

The 20/40/60 limits are binding for every tree. The 80 limit is only weakly
binding: 36.7% of trees finish before depth 80, and the forest's natural
mean depth is 86.6, so max_depth = 80 removes very little capacity.

## What max_depth controls

`max_depth` caps how many sequential splitting levels a tree may use. Depth
20 trees can combine at most 20 Morgan-bit decisions along any path; deeper
trees can combine more specific bit sequences. It is a tree-capacity limit
that directly constrains each individual tree.

## Difference from min_samples_leaf

`min_samples_leaf` stops growth by data mass: a leaf must keep at least a
minimum number of training molecules. `max_depth` stops growth by path
length, independent of leaf size. The two parameters constrain different
aspects of tree shape. Here min_samples_leaf = 1 is retained, so leaves may
still be tiny; the only variation is how many levels trees may traverse.

## Training performance

Training fit rises monotonically with depth: train R2 is 0.858 at depth 20,
0.948 at 40, and 0.958-0.960 at 60-80/None. Depth 20 is strongly
capacity-limited; by 60 the forest has nearly the same training fit as
unrestricted depth.

## Validation performance

Validation follows training fit until the plateau: depth 20 is clearly worst
(Spearman 0.7597, R2 0.5820), depth 40 is better but still below the top
(0.7781 / 0.6125), and 60/80/None all sit at 0.7789-0.7800 Spearman and
0.6118-0.6140 R2. The best numerical value is 60, but its advantage over
None is tiny (Spearman +0.0008, R2 +0.0005), well inside the seed-to-seed
variation observed in the multi-seed max_features study.

## Generalization gaps

Gaps are not a useful signal here: depth 20 has the smallest gap only
because both train and validation performance are low. From depth 40 upward
the gaps are similar (Spearman 0.199-0.204, R2 0.335-0.348), and the small
differences do not track validation ranking.

## Overfitting / underfitting / plateau evidence

* Depth 20: strong underfitting. Both train and validation metrics are
  substantially below the other configurations.
* Depth 40: transitional. Every tree hits the limit and validation is
  slightly but consistently below 60/None.
* Depths 60-80 and None: high-depth plateau. Validation metrics are
  effectively identical, and 80 is already non-binding for over a third of
  trees.

## Do regression and ranking metrics agree?

Yes. MAE, RMSE, R2, and Spearman all rank 60 first, None second, 80 third,
40 fourth, and 20 last. The differences among 60/80/None are tiny.

## Is the improvement large enough to change max_depth = None?

No. max_depth = 60 has the best validation numbers, but the advantage over
None is 0.0008 in Spearman and 0.0005 in R2, far below the multi-seed
variation of the same model family (Spearman SD ~0.0014-0.0018, R2 SD
~0.0022-0.0023). Choosing 60 over None would be selecting a single-seed
noise-level difference.

## Decision classification

**B. High-depth plateau.** Performance becomes effectively identical beyond
about depth 60. Restricting to 40 loses a little capacity; restricting to 20
clearly underfits.

## Recommendation

Retain `max_depth = None`. Unlimited depth is not measurably worse than any
finite candidate, it is the simpler configuration, and it matches the
reference used throughout M2/M3 and the earlier tuning studies. No README
headline metric is changed.

## Files

```text
src/models/max_depth_tuning_study.py
results/max_depth_tuning_results.csv
results/max_depth_tuning_results.json
results/figures/max_depth_train_val.png
results/figures/max_depth_complexity.png
```

## Reproducibility

```bash
conda activate egfr-aidd
python src/models/max_depth_tuning_study.py
```

The JSON records the split, the full effective RF parameter dictionary,
per-setting tree complexity and depth-limit saturation, package versions,
and all train/validation metrics. The standard-scaffold test split was not
evaluated.
