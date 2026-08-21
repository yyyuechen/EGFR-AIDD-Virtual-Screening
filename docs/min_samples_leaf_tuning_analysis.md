# Random Forest min_samples_leaf Study

## Purpose and protocol

This controlled study changes exactly one modeling decision: the RandomForest
`min_samples_leaf` (1, 2, 4, 8, 16).

Everything else is fixed:

* Morgan radius = 2, nBits = 2048, same `GetMorganGenerator` implementation as M2/M3;
* standard 80/10/10 Bemis-Murcko scaffold split, seed 42;
* training set 8,128 molecules; validation set 1,015 molecules;
* **the standard-scaffold test set (1,018 molecules) is never used**;
* n_estimators = 300, random_state = 42, n_jobs = -1;
* all other RandomForest parameters at sklearn defaults.

Primary model-selection metric: validation Spearman. Secondary: validation
RMSE, MAE, R2. Training metrics and gaps are diagnostic only.

## Full effective RandomForestRegressor parameters

The reference configuration (min_samples_leaf = 1) has these effective
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

Only `min_samples_leaf` changes across runs.

## Results (validation split only)

| min_samples_leaf | train MAE | train RMSE | train R2 | train Spearman | val MAE | val RMSE | val R2 | val Spearman | R2 gap | Spearman gap | RMSE gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1962 | 0.2761 | 0.9586 | 0.9822 | 0.5947 | 0.7785 | **0.5956** | **0.7689** | 0.3629 | 0.2133 | 0.5025 |
| 2 | 0.2634 | 0.3689 | 0.9260 | 0.9672 | 0.6029 | 0.7852 | 0.5886 | 0.7647 | 0.3374 | 0.2025 | 0.4164 |
| 4 | 0.3646 | 0.4975 | 0.8654 | 0.9365 | 0.6167 | 0.7972 | 0.5760 | 0.7571 | 0.2894 | 0.1794 | 0.2997 |
| 8 | 0.4653 | 0.6198 | 0.7911 | 0.8964 | 0.6476 | 0.8340 | 0.5359 | 0.7321 | 0.2552 | 0.1643 | 0.2142 |
| 16 | 0.5617 | 0.7337 | 0.7072 | 0.8490 | 0.6879 | 0.8852 | 0.4772 | 0.6923 | 0.2300 | 0.1568 | 0.1514 |

All differences in the table are monotonic: every train and validation metric
gets worse as `min_samples_leaf` increases, while every train-validation gap
gets smaller.

## Tree complexity diagnostics

| min_samples_leaf | mean depth | median depth | max depth | mean leaves | median leaves | mean samples per leaf |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.1 | 81.0 | 136 | 4,674 | 4,675 | 1.7 |
| 2 | 69.7 | 67.5 | 116 | 2,177 | 2,177 | 3.7 |
| 4 | 53.0 | 52.0 | 89 | 1,027 | 1,027 | 7.9 |
| 8 | 38.7 | 38.0 | 65 | 497 | 497 | 16.3 |
| 16 | 28.3 | 28.0 | 50 | 245 | 245 | 33.2 |

Increasing `min_samples_leaf` reduces complexity in a clean, monotonic way:
trees become shallower, have fewer leaves, and each leaf is supported by more
training molecules.

## What min_samples_leaf controls

`min_samples_leaf` is the minimum number of training molecules that must
remain in any terminal leaf of a decision tree. At leaf = 1 a tree may create
a leaf supported by a single training molecule; at leaf = 4 every terminal
leaf must contain at least four training molecules. Larger values forbid
very local splits, so trees cannot memorize narrow training-chemistry
patterns.

## Why increasing it regularizes Random Forest trees

Each tree is trained by recursively splitting the training set. A smaller
minimum leaf size allows the recursion to continue until nearly every
training molecule has its own prediction. This gives excellent in-sample fit
but makes individual leaves more sensitive to individual molecules. A larger
minimum leaf size forces splits to cover broader groups, reducing variance
at the cost of some bias.

## Training performance

Training performance drops monotonically: R2 falls from 0.9586 (leaf 1) to
0.7072 (leaf 16) and train Spearman falls from 0.9822 to 0.8490. At leaf 1
the forest is close to memorizing the training labels; larger leaf sizes
progressively trade that memorization away.

## Validation performance

Validation performance also drops monotonically: validation Spearman falls
from 0.7689 (leaf 1) to 0.6923 (leaf 16) and validation R2 falls from 0.5956
to 0.4772. On this scaffold-held-out validation split, the least regularized
model is also the best model.

## Generalization gaps

The gaps shrink monotonically:

* Spearman gap: 0.2133 -> 0.1568
* R2 gap: 0.3629 -> 0.2300
* RMSE gap (val - train): 0.5025 -> 0.1514

So the model becomes less overfit in the relative sense. But the shrinkage is
achieved by reducing absolute validation performance, not by improving it.
The gap is smaller at leaf 16 because both train and validation performance
are worse.

## Overfitting / regularization / underfitting evidence

* Leaf 1 is overfitting in the diagnostic sense: train R2 0.959 vs validation
  R2 0.596 and train Spearman 0.982 vs validation 0.769. It is still the best
  validation model.
* There is no intermediate region that improves validation: every value
  between 2 and 16 is worse than the default.
* Leaf 16 shows underfitting: both train and validation metrics fall well
  below leaf 1, and the validation ranking correlation drops to 0.69.

## Do regression and ranking metrics agree?

Yes. For every metric (MAE, RMSE, R2, Spearman) the validation ordering is
identical:

```text
1 > 2 > 4 > 8 > 16
```

There is no conflict between calibration and ranking in this study.

## Is the improvement large enough to change the default?

No. Regularization only hurts validation performance here; leaf = 1 retains
the highest validation Spearman (0.7689) and R2 (0.5956). The gap-based
diagnostic shows overfitting at leaf 1, but the alternative leaf sizes do not
convert that diagnostic into better scaffold-held-out predictions.

## Decision classification

**B. Default remains best.** leaf = 1 gives the strongest validation results
on every metric; added regularization only reduces absolute performance, even
though it narrows the train-validation gap.

## Recommendation

Retain `min_samples_leaf = 1` (the current default). No project model or
README headline metric is changed. If future work targets external ranking,
a separate external-validation study is the right place to test this
parameter again.

## Files

```text
src/models/min_samples_leaf_tuning_study.py
results/min_samples_leaf_tuning_results.csv
results/min_samples_leaf_tuning_results.json
results/figures/min_samples_leaf_train_val.png
results/figures/min_samples_leaf_complexity.png
```

## Reproducibility

```bash
conda activate egfr-aidd
python src/models/min_samples_leaf_tuning_study.py
```

The JSON records the split, the full effective RF parameter dictionary,
per-setting tree complexity, package versions, and all train/validation
metrics. The standard-scaffold test split was not evaluated.
