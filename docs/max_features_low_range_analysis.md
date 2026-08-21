# Random Forest max_features Low-Range Study

## Purpose and protocol

The previous `max_features` study (0.2-1.0) found that validation performance
improved as feature subsampling increased, with the best tested value at the
lower boundary 0.2. This study extends the boundary downward to answer
whether performance keeps improving or eventually degrades:

```text
max_features = 0.05, 0.10, 0.15, 0.20
```

Everything else is fixed:

* Morgan radius = 2, nBits = 2048, same `GetMorganGenerator` implementation as M2/M3;
* standard 80/10/10 Bemis-Murcko scaffold split, seed 42;
* training set 8,128 molecules; validation set 1,015 molecules;
* **the standard-scaffold test set (1,018 molecules) is never used**;
* n_estimators = 300, random_state = 42, n_jobs = -1, min_samples_leaf = 1,
  max_depth = None;
* all other RandomForest parameters at sklearn defaults.

Primary metric: validation Spearman. Secondary: validation RMSE, MAE, R2.

## Results (validation split only)

| max_features | approx bits/split | train MAE | train RMSE | train R2 | train Spearman | val MAE | val RMSE | val R2 | val Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 102 | 0.1945 | 0.2714 | 0.9600 | 0.9835 | 0.5875 | 0.7643 | 0.6102 | 0.7744 |
| 0.10 | 205 | 0.1947 | 0.2726 | 0.9596 | 0.9832 | 0.5832 | 0.7582 | 0.6165 | 0.7803 |
| 0.15 | 307 | 0.1948 | 0.2726 | 0.9596 | 0.9831 | **0.5807** | **0.7563** | **0.6183** | **0.7811** |
| 0.20 | 410 | 0.1943 | 0.2728 | 0.9595 | 0.9830 | 0.5865 | 0.7612 | 0.6135 | 0.7792 |

Validation performance peaks at 0.15 on every metric. The 0.10-0.20 region
is a shallow plateau; 0.05 is clearly worse than 0.10-0.20.

Context from the prior study:

| max_features | val R2 | val Spearman |
|---:|---:|---:|
| 0.2 | 0.6135 | 0.7792 |
| 0.4 | 0.6136 | 0.7788 |
| 0.6 | 0.6094 | 0.7778 |
| 0.8 | 0.6066 | 0.7760 |
| 1.0 | 0.5956 | 0.7689 |

## Train-validation gaps

| max_features | Spearman gap | R2 gap | RMSE gap |
|---:|---:|---:|---:|
| 0.05 | 0.2092 | 0.3497 | 0.4930 |
| 0.10 | 0.2029 | 0.3431 | 0.4856 |
| 0.15 | 0.2020 | 0.3413 | 0.4838 |
| 0.20 | 0.2038 | 0.3461 | 0.4884 |

The smallest gaps occur at 0.15, matching the best validation metrics. At
0.05 the gap opens up again, so the lower boundary is not just worse in
absolute terms, it also generalizes slightly worse.

## Tree complexity

| max_features | mean depth | median depth | max depth | mean leaves | median leaves |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 83.6 | 82 | 134 | 4,824 | 4,826 |
| 0.10 | 83.5 | 82 | 135 | 4,772 | 4,774 |
| 0.15 | 86.7 | 84 | 141 | 4,743 | 4,745 |
| 0.20 | 86.6 | 84.5 | 138 | 4,725 | 4,729 |

Complexity is not the driver of the low-range trend. Trees at 0.15 are about
3 levels deeper than at 0.05, but they have about 80 fewer leaves. No simple
monotone relationship with validation performance is visible.

## Feature-importance concentration

| max_features | top-10 fraction | top-50 fraction | non-zero features |
|---:|---:|---:|---:|
| 0.05 | 0.1376 | 0.2991 | 2,047 |
| 0.10 | 0.1740 | 0.3332 | 2,047 |
| 0.15 | 0.2057 | 0.3551 | 2,048 |
| 0.20 | 0.2258 | 0.3758 | 2,048 |

Stronger feature subsampling spreads importance mass further: top-10
importance falls from 0.226 at 0.20 to 0.138 at 0.05. At 0.05 and 0.10 one
of the 2,048 bits is never selected by any split. The concentration trend is
monotone and is the expected mechanical consequence of more aggressive
feature sampling.

## What the low-range results mean

Feature subsampling helps up to a point. From 1.0 down to 0.15, validation
performance improves monotonically (Spearman 0.7689 -> 0.7811, R2 0.5956 ->
0.6183). Below 0.15 the trend reverses: at 0.05 validation Spearman falls to
0.7744 and R2 to 0.6102, which is worse than 0.10 and 0.20. This is
consistent with the diversity hypothesis up to the optimum, followed by the
underfitting side of the tradeoff where individual trees no longer see
enough informative fingerprint bits.

## Do regression and ranking metrics agree?

Yes. On all four validation metrics (MAE, RMSE, R2, Spearman) 0.15 is best,
0.05 is worst in this low range, and 0.10/0.20 sit in between. Ranking and
calibration tell the same story.

## Effect size

The differences inside 0.10-0.20 are small: Spearman range 0.7803-0.7811 and
R2 range 0.6135-0.6183. The 0.05 dip is larger (Spearman -0.007, R2 -0.008
relative to 0.15) but still small in absolute terms. This is a single-seed,
single-split comparison, so the numeric peak should not be over-interpreted.

## Decision classification

**A. Interior optimum.** Validation performance peaks near 0.15 and declines
at the lower boundary 0.05. The peak region 0.10-0.20 is a shallow plateau,
so the optimum is broad rather than sharp.

## Recommendation

The evidence does not support keeping 0.2 as "clearly better" nor does it
support investigating even lower values: 0.05 already degrades performance,
and 0.15 is only marginally better than 0.2 (Spearman +0.002, R2 +0.005).
If the project adopts a change, 0.15 is the most plausible interior optimum
and should be confirmed with multiple seeds before touching the final model;
otherwise retaining 0.2 is acceptable because the practical difference is
negligible. No README headline metric is changed in this study.

## Files

```text
src/models/max_features_low_range_study.py
results/max_features_low_range_results.csv
results/max_features_low_range_results.json
results/figures/max_features_low_range_train_val.png
results/figures/max_features_low_range_validation.png
```

The original `max_features` study files were not overwritten.

## Reproducibility

```bash
conda activate egfr-aidd
python src/models/max_features_low_range_study.py
```

The JSON records the split, the full effective RF parameter dictionary,
per-setting tree complexity, feature-importance diagnostics, package
versions, and all train/validation metrics. The standard-scaffold test split
was not evaluated.
