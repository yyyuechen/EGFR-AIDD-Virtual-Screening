# Morgan Fingerprint Radius Study

## Purpose and protocol

This is a controlled, one-parameter-at-a-time hyperparameter study. The only
modeling decision changed is the **Morgan fingerprint radius** (1, 2, 3).

Everything else is fixed:

* standard 80/10/10 Bemis-Murcko scaffold split, seed 42;
* training set: 8,128 molecules; validation set: 1,015 molecules;
* **the standard-scaffold test set (1,018 molecules) is never used**;
* nBits = 2048;
* RandomForest from the M2/M3 settings: 300 trees, random_state 42,
  n_jobs = -1;
* preprocessing, endpoint definition, and all other settings unchanged.

**Primary model-selection metric:** validation Spearman.
**Secondary metrics:** validation RMSE, MAE, R2.

## What Morgan radius means chemically

Morgan fingerprints encode circular atom environments. Radius 1 captures
each atom plus its direct neighbors (bonds of length 1); radius 2 extends
the environment to two bonds; radius 3 captures larger local substructure
contexts up to three bonds away. A larger radius therefore represents more
of the surrounding molecular graph in each fingerprint bit pattern.

## Results (validation split only)

| radius | MAE | RMSE | R2 | Spearman | mean on-bits | mean max train Tanimoto |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.583 | 0.771 | 0.604 | **0.776** | 36.4 | 0.801 |
| 2 | 0.595 | 0.779 | 0.596 | 0.769 | 62.2 | 0.720 |
| 3 | 0.587 | 0.772 | 0.602 | 0.773 | 86.0 | 0.665 |

Full fingerprint and similarity statistics:

| radius | median on-bits | density | mean max sim | median max sim | min max sim | max max sim |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37.0 | 1.78% | 0.801 | 0.824 | 0.265 | 1.000 |
| 2 | 64.0 | 3.04% | 0.720 | 0.750 | 0.165 | 1.000 |
| 3 | 88.0 | 4.20% | 0.665 | 0.700 | 0.121 | 1.000 |

## How representation sparsity changes with radius

Fingerprint density increases with radius: mean on-bits per molecule grows
from 36.4 (radius 1) to 62.2 (radius 2) to 86.0 (radius 3), and density
grows from 1.78% to 4.20%. Larger radii capture more structural context but
produce denser, more overlapping representations.

## How chemical-space similarity changes with radius

`max_train_tanimoto` on the validation set decreases as radius increases:

* mean: 0.801 -> 0.720 -> 0.665;
* median: 0.824 -> 0.750 -> 0.700;
* minimum: 0.265 -> 0.165 -> 0.121.

This is expected: with more bits, two molecules' fingerprints share a
smaller fraction of the total pattern. The applicability-domain statistic is
therefore radius-dependent; a "similarity of 0.7" means different things at
different radii.

The maximum is 1.0 at all radii, meaning at least one validation molecule
has a fingerprint identical to a training molecule even though scaffolds
are disjoint. This is a reminder that fingerprints are not unique structure
identifiers.

## Does a larger neighborhood improve EGFR activity prediction?

On this validation split, **no**; the largest radius did not clearly improve
prediction. Radius 1 has the highest validation Spearman (0.776) and the
lowest MAE / RMSE / R2 ordering, radius 3 is close behind, and radius 2 is
slightly lowest on all four metrics.

Important caveat: the differences are small (Spearman range 0.007; R2 range
0.009). The result is a mild preference for radius 1 under this exact
protocol, not a decisive recommendation. The project's default radius 2
remains a reasonable, chemically standard choice, and a multi-seed
repetition would be needed before changing it.

## Do regression and ranking metrics agree?

Yes, in this study. The metric ordering is identical for all four metrics:

```text
radius 1 > radius 3 > radius 2
```

for validation Spearman, R2, RMSE, and MAE. The regression and ranking views
therefore point in the same direction, but again with small margins.

## Files

```text
src/models/radius_tuning_study.py
results/radius_tuning_results.csv
results/radius_tuning_results.json
```

## Reproducibility

```bash
conda activate egfr-aidd
python src/models/radius_tuning_study.py
```

The JSON records the split definition and actual counts, RandomForest
parameters, package versions, and all per-radius statistics. The test split
was not evaluated; the headline README results were not modified.
