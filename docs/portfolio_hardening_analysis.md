# Portfolio hardening: scaffold-aware validation and applicability domain

This document adds two focused analyses to the existing EGFR AIDD project:

1. a conventional 80/10/10 Bemis-Murcko scaffold split, compared against the
   existing random split and the existing hard scaffold OOD split;
2. a chemical applicability-domain analysis based on the maximum Morgan
   Tanimoto similarity between each standard-scaffold test molecule and any
   training molecule.

No new model family was introduced. The models are the existing
Morgan + RandomForest and Morgan + XGBoost baselines with the exact M2/M3
settings (RF 300 trees, XGB 300 trees, lr=0.05, depth=6, seed=42).

---

## 1. Why a random split can be optimistic

A random row-level split can place close chemical neighbours in both the
training and test sets. Morgan fingerprints of similar molecules overlap
heavily, so the model can appear to generalise well even when it has in
effect memorised the local chemistry around the test molecule. Random split
metrics are therefore an upper-bound estimate of performance.

## 2. What a Bemis-Murcko scaffold split tests

A Bemis-Murcko scaffold is the ring system left after removing side chains.
Molecules sharing the same scaffold are kept in the same split, so the model
must predict activity for scaffolds it never saw during training. This tests
generalisation to new ring systems rather than interpolation within familiar
chemistry.

## 3. Why the existing hard OOD split is more difficult

The existing M3 scaffold split is built by sorting scaffold groups
largest-first and greedily filling the training side until
`int(0.8 * 10,161) = 8,128` molecules. This pushes every small group into the
test set. Verified from the data:

| Property | Hard scaffold OOD test |
|---|---:|
| Test molecules | 2,033 (20.0%) |
| Unique test scaffolds | 2,033 |
| Singleton scaffolds | 2,033 / 2,033 (100%) |
| Largest test scaffold group | 1 |
| Train / test scaffold overlap | 0 |

Every test molecule comes from a scaffold seen exactly once in the whole
dataset, and no validation split exists. This is why we label it the **hard
scaffold OOD split**: it is an extreme single-scaffold extrapolation test,
not a typical scaffold-aware validation.

## 4. Standard scaffold split results

The new standard split keeps complete scaffold groups together and targets
80/10/10 with a deterministic seed (42):

| Split | Molecules | % | Scaffolds | Largest group | Singletons | pIC50 mean |
|---|---:|---:|---:|---:|---:|---:|
| Train | 8,128 | 80.0 | 2,858 | 560 | 1,879 | 6.93 |
| Validation | 1,015 | 10.0 | 444 | 35 | 292 | 7.01 |
| Test | 1,018 | 10.0 | 339 | 57 | 227 | 6.99 |

Leakage checks: `train ∩ val = 0`, `train ∩ test = 0`,
`val ∩ test = 0` scaffolds.

The test set is still scaffold-unseen, but it contains multi-molecule
scaffolds (largest group 57) and a much smaller singleton fraction (227/339
scaffolds), so it is a more typical scaffold-aware benchmark than the hard
OOD split.

## 5. Hard scaffold OOD results

For completeness, the existing hard split was re-evaluated with the same
models and settings. The regression and screening metrics are shown in the
comparison table below.

## 6. Split comparison (Morgan-RF and Morgan-XGB)

Regression (test set):

| Split | Model | MAE | RMSE | R2 | Spearman |
|---|---|---:|---:|---:|---:|
| Random | RandomForest | 0.536 | 0.738 | 0.701 | 0.837 |
| Random | XGBoost | 0.624 | 0.812 | 0.634 | 0.803 |
| Standard scaffold | RandomForest | 0.649 | 0.852 | 0.573 | 0.759 |
| Standard scaffold | XGBoost | 0.738 | 0.974 | 0.509 | 0.713 |
| Hard scaffold OOD | RandomForest | 0.759 | 0.998 | 0.398 | 0.651 |
| Hard scaffold OOD | XGBoost | 0.812 | 1.086 | 0.324 | 0.605 |

Screening (active pIC50 >= 7.0, inactive pIC50 < 6.0, middle band excluded):

| Split | Model | ROC-AUC | PR-AUC | EF1% | EF5% |
|---|---|---:|---:|---:|---:|
| Random | RandomForest | 0.963 | 0.979 | 1.489 | 1.489 |
| Random | XGBoost | 0.953 | 0.974 | 1.489 | 1.470 |
| Standard scaffold | RandomForest | 0.928 | 0.959 | 1.481 | 1.481 |
| Standard scaffold | XGBoost | 0.918 | 0.955 | 1.481 | 1.481 |
| Hard scaffold OOD | RandomForest | 0.884 | 0.916 | 1.713 | 1.713 |
| Hard scaffold OOD | XGBoost | 0.850 | 0.866 | 1.713 | 1.570 |

Interpretation:

* Random split is clearly the most optimistic: RF R2 0.701, Spearman 0.837.
* Standard scaffold split is harder but still strong: RF R2 0.573,
  Spearman 0.759, ROC-AUC 0.928.
* Hard scaffold OOD is the most difficult: RF R2 0.398, Spearman 0.651,
  ROC-AUC 0.884.
* Screening ranking remains useful even on unseen scaffolds: every scaffold
  split still achieves ROC-AUC > 0.85 and EF1% > 1.4.

## 7. Applicability-domain analysis

For every molecule in the standard scaffold test set, `max_train_tanimoto`
is the maximum Morgan Tanimoto similarity (radius=2, 2048 bits) to any
training molecule:

* Tanimoto close to 1 means the fingerprints are very similar.
* Tanimoto close to 0 means low fingerprint overlap.
* Tanimoto is fingerprint-dependent and is not a perfect definition of
  chemical similarity.

Distribution on the 1,018 test molecules:

| Statistic | Value |
|---|---:|
| Mean | 0.705 |
| Median | 0.730 |
| Min | 0.197 |
| Max | 1.000 |

Per-molecule predictions, errors and similarities are saved in
`results/applicability_domain_predictions.csv`.

### Similarity bins

The `<0.2` and `0.2-<0.4` bins contained too few molecules, so they were
merged into `<0.4` (n=45) and documented:

| Bin | n | Mean similarity | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|
| <0.4 | 45 | 0.334 | 1.068 | 1.427 | -0.068 |
| 0.4-<0.6 | 173 | 0.521 | 0.815 | 1.061 | 0.420 |
| 0.6-<0.8 | 527 | 0.716 | 0.633 | 0.840 | 0.569 |
| >=0.8 | 273 | 0.862 | 0.507 | 0.705 | 0.697 |

MAE decreases monotonically from 1.068 to 0.507 as similarity to the
training set increases. The smallest bin also has the worst regression fit
(R2 below zero), so predictions there should be treated as unreliable.

## 8. Relationship between similarity and error

`Spearman(max_train_tanimoto, absolute_error) = -0.232 (p = 7.3e-14)`.

This is a small-to-moderate negative correlation: on average, molecules more
similar to the training chemistry have lower absolute pIC50 error. The
monotone MAE trend across bins is the more intuitive summary. The correlation
is not strong, and high similarity does not guarantee accuracy, so the
relationship should be used as a reliability hint rather than a hard rule.

### Ranking by similarity group (optional)

The standard-scaffold test was split into high (>=0.6) and low (<0.6)
similarity groups:

| Group | n | Spearman | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| High similarity (>=0.6) | 800 | 0.783 | 0.944 | 0.975 |
| Low similarity (<0.6) | 218 | 0.617 | 0.858 | 0.795 |

Ranking quality remains clearly positive even in the low-similarity group
(ROC-AUC 0.858), but is weaker than in the high-similarity group (0.944).

## 9. Implications for virtual screening

The hard scaffold OOD split overstates how difficult scaffold-unseen
screening is for this pipeline. The standard scaffold split is a more
realistic evaluation: performance drops from random split but stays strong,
with RF ROC-AUC 0.928 and EF1% 1.481. Screening rank is still useful for
chemically distant molecules, though confidence should decrease as
`max_train_tanimoto` decreases.

## 10. Limitations

* Morgan Tanimoto is representation-dependent; a different fingerprint or
  embedding would give different similarities.
* Structurally similar compounds can still show activity cliffs.
* Low similarity does not automatically mean a prediction is wrong, and high
  similarity does not guarantee accuracy.
* Assay heterogeneity affects the activity labels.
* This analysis is retrospective, not prospective validation.
* The standard split is one deterministic realisation; a different seed would
  give slightly different splits and metrics.

## Files

```text
src/models/portfolio_hardening_analysis.py
results/scaffold_split_comparison.json
results/applicability_domain_predictions.csv
results/figures/portfolio_split_comparison.png
results/figures/portfolio_applicability_scatter.png
results/figures/portfolio_applicability_bins.png
```

## How to run

```bash
conda activate egfr-aidd
python src/models/portfolio_hardening_analysis.py
```
