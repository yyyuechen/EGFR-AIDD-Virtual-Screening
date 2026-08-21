# max_features Multi-Seed Reproducibility Study

## Purpose and protocol

The single-seed max_features studies found a broad validation optimum around
0.10-0.20. This study asks whether that benefit is reproducible across
Random Forest random states, not whether another hyperparameter value is
better.

Fixed settings:

* standard 80/10/10 Bemis-Murcko scaffold split, seed 42 (split is fixed and
  identical for every RF seed);
* Morgan radius = 2, nBits = 2048, same implementation as M2/M3;
* pIC50 endpoint and preprocessing unchanged;
* n_estimators = 300, min_samples_leaf = 1, max_depth = None, n_jobs = -1;
* all other RandomForest parameters at sklearn defaults.

Compared: max_features = 1.0, 0.20, 0.15, 0.10, each trained with RF
random_state = 42, 7, 123, 2024, 777. **The standard-scaffold test set is
never used.**

Primary metric: validation Spearman. Secondary: R2, RMSE, MAE.

## Seed-level results

| max_features | seed | val MAE | val RMSE | val R2 | val Spearman |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 42 | 0.5947 | 0.7785 | 0.5956 | 0.7689 |
| 1.0 | 7 | 0.5951 | 0.7808 | 0.5933 | 0.7685 |
| 1.0 | 123 | 0.5986 | 0.7820 | 0.5920 | 0.7677 |
| 1.0 | 2024 | 0.5943 | 0.7778 | 0.5964 | 0.7700 |
| 1.0 | 777 | 0.5967 | 0.7810 | 0.5931 | 0.7690 |
| 0.20 | 42 | 0.5865 | 0.7612 | 0.6135 | 0.7792 |
| 0.20 | 7 | 0.5832 | 0.7595 | 0.6151 | 0.7789 |
| 0.20 | 123 | 0.5820 | 0.7585 | 0.6161 | 0.7806 |
| 0.20 | 2024 | 0.5805 | 0.7567 | 0.6180 | 0.7810 |
| 0.20 | 777 | 0.5800 | 0.7553 | 0.6193 | 0.7834 |
| 0.15 | 42 | 0.5807 | 0.7563 | 0.6183 | 0.7811 |
| 0.15 | 7 | 0.5836 | 0.7609 | 0.6137 | 0.7789 |
| 0.15 | 123 | 0.5816 | 0.7573 | 0.6174 | 0.7824 |
| 0.15 | 2024 | 0.5817 | 0.7576 | 0.6171 | 0.7799 |
| 0.15 | 777 | 0.5842 | 0.7609 | 0.6138 | 0.7796 |
| 0.10 | 42 | 0.5832 | 0.7582 | 0.6165 | 0.7803 |
| 0.10 | 7 | 0.5851 | 0.7615 | 0.6131 | 0.7763 |
| 0.10 | 123 | 0.5860 | 0.7632 | 0.6114 | 0.7766 |
| 0.10 | 2024 | 0.5839 | 0.7615 | 0.6131 | 0.7788 |
| 0.10 | 777 | 0.5830 | 0.7596 | 0.6151 | 0.7795 |

## Mean +/- SD across five seeds

| max_features | val MAE | val RMSE | val R2 | val Spearman |
|---:|---:|---:|---:|---:|
| 1.0 | 0.5959 +/- 0.0018 | 0.7800 +/- 0.0018 | 0.5941 +/- 0.0018 | 0.7688 +/- 0.0008 |
| 0.20 | 0.5824 +/- 0.0026 | 0.7582 +/- 0.0023 | 0.6164 +/- 0.0023 | 0.7806 +/- 0.0018 |
| 0.15 | 0.5824 +/- 0.0015 | 0.7586 +/- 0.0021 | 0.6161 +/- 0.0022 | 0.7804 +/- 0.0014 |
| 0.10 | 0.5842 +/- 0.0013 | 0.7608 +/- 0.0019 | 0.6138 +/- 0.0020 | 0.7783 +/- 0.0018 |

## Paired improvement vs max_features = 1.0

| max_features | metric | mean improvement | SD of improvement | seeds better |
|---:|---:|---:|---:|---:|
| 0.20 | Spearman | +0.0118 | 0.0018 | 5/5 |
| 0.15 | Spearman | +0.0116 | 0.0019 | 5/5 |
| 0.10 | Spearman | +0.0094 | 0.0014 | 5/5 |
| 0.20 | R2 | +0.0223 | 0.0032 | 5/5 |
| 0.15 | R2 | +0.0220 | 0.0021 | 5/5 |
| 0.10 | R2 | +0.0198 | 0.0020 | 5/5 |
| 0.20 | MAE | -0.0135 | 0.0036 | 5/5 lower |
| 0.15 | MAE | -0.0135 | 0.0022 | 5/5 lower |
| 0.10 | MAE | -0.0116 | 0.0015 | 5/5 lower |
| 0.20 | RMSE | -0.0218 | 0.0031 | 5/5 lower |
| 0.15 | RMSE | -0.0214 | 0.0021 | 5/5 lower |
| 0.10 | RMSE | -0.0192 | 0.0019 | 5/5 lower |

Every low max_features value beats 1.0 on every metric for all five seeds.
The mean Spearman improvement (+0.009 to +0.012) is roughly 6-8 times the
within-configuration seed SD, so the benefit is not driven by one lucky
seed.

## Prediction stability across seeds

| max_features | mean pred SD | median pred SD | p95 pred SD |
|---:|---:|---:|---:|
| 1.0 | 0.0433 | 0.0397 | 0.0840 |
| 0.20 | 0.0434 | 0.0401 | 0.0853 |
| 0.15 | 0.0434 | 0.0407 | 0.0849 |
| 0.10 | 0.0429 | 0.0407 | 0.0815 |

Across-seed prediction variability is almost identical for every
configuration (mean per-molecule SD around 0.043 pIC50 units). Feature
subsampling does not make the ensemble's validation predictions more noisy
at the seed level.

## Interpretation

The feature-subsampling benefit is reproducible. The earlier single-seed
conclusion is not an artifact of random_state = 42. Within the 0.10-0.20
region, no single value is uniquely best: 0.20 and 0.15 are statistically
equivalent (mean Spearman 0.7806 vs 0.7804), and 0.10 is only slightly lower
(0.7783). All three remain clearly above 1.0.

## Decision classification

**A. Robust feature-subsampling benefit.** max_features = 0.10-0.20
consistently outperforms 1.0 on all five seeds, and the improvement is
several times larger than seed-to-seed variability.

## Recommendation

Do not claim that 0.15 is uniquely optimal; the evidence supports a broad
0.10-0.20 region. If a concrete reference is needed, 0.20 is the most
defensible practical choice: its mean Spearman (0.7806) and R2 (0.6164) are
at least as good as 0.15 across seeds and it is a round, interpretable
fraction. The original 1.0 baseline is the worst of the four configurations
in this study. No README headline metric is changed; these are
validation-only findings.

## Files

```text
src/models/max_features_multiseed_study.py
results/max_features_multiseed_results.csv
results/max_features_multiseed_summary.json
results/figures/max_features_multiseed_spearman.png
```

## Reproducibility

```bash
conda activate egfr-aidd
python src/models/max_features_multiseed_study.py
```

The JSON records the split, full effective RF parameters, all 20 seed-level
results, mean/SD/min/max per configuration, paired improvements, prediction
stability, and package versions. The standard-scaffold test split was not
evaluated.
