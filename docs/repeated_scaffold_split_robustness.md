# Repeated Scaffold-Split Robustness Study

## Purpose and protocol

This study tests whether the validated benefit of RF feature subsampling
(`max_features = 0.20` vs `1.0`) is robust to **chemical scaffold partition
variation**, not RF stochasticity.

Frozen configurations:

* baseline: `max_features = 1.0`;
* tuned: `max_features = 0.20`;
* identical otherwise: Morgan radius 2 / 2048 bits, 300 trees,
  `min_samples_leaf = 1`, `max_depth = None`, RF `random_state = 42`.

Five deterministic 80/10/10 whole-Murcko-scaffold partitions were generated
with the same splitting algorithm used by the canonical standard split, using
split seeds 7, 21, 42, 123, and 2024. Seed 42 reproduces the canonical split
(8,128 / 1,015 / 1,018). No parameter was changed based on any result.

This answers a different question from the earlier multi-seed study:

* RF-seed robustness: same chemical split, different RF stochasticity;
* this study: different chemical partitions, fixed RF stochasticity.

## Results (test set, one row per scaffold partition)

| split seed | train | val | test | test scaffolds | mean test pIC50 | active frac | mean max sim | baseline R2 | tuned R2 | baseline Sp | tuned Sp | dR2 | dSp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 8,128 | 1,018 | 1,015 | 446 | 6.986 | 0.518 | 0.725 | 0.6081 | 0.6111 | 0.7907 | 0.7923 | +0.0030 | +0.0016 |
| 21 | 8,129 | 1,015 | 1,017 | 407 | 6.966 | 0.509 | 0.730 | 0.6274 | 0.6363 | 0.7967 | 0.8014 | +0.0089 | +0.0047 |
| 42 | 8,128 | 1,015 | 1,018 | 339 | 6.985 | 0.535 | 0.705 | 0.5728 | 0.5918 | 0.7587 | 0.7663 | +0.0190 | +0.0076 |
| 123 | 8,116 | 1,002 | 1,043 | 294 | 6.768 | 0.439 | 0.698 | 0.6038 | 0.6214 | 0.7728 | 0.7847 | +0.0175 | +0.0119 |
| 2024 | 8,126 | 1,021 | 1,014 | 325 | 7.154 | 0.580 | 0.709 | 0.6582 | 0.6731 | 0.7992 | 0.8041 | +0.0149 | +0.0049 |

All splits have zero scaffold overlap between train/validation/test.

## Paired improvement summary

| metric | mean | SD | median | min | max | splits improved |
|---|---:|---:|---:|---:|---:|---:|
| ΔSpearman | +0.0061 | 0.0039 | +0.0049 | +0.0016 | +0.0119 | 5/5 |
| ΔR2 | +0.0127 | 0.0066 | +0.0149 | +0.0030 | +0.0190 | 5/5 |
| ΔRMSE | -0.0137 | 0.0070 | -0.0166 | -0.0199 | -0.0033 | 5/5 lower |
| ΔMAE | -0.0035 | 0.0054 | -0.0019 | -0.0112 | +0.0023 | 4/5 lower |

The tuned model beat the baseline on Spearman and R2 in every partition.
RMSE improved in 5/5; MAE improved in 4/5 (seed 7 was 0.002 worse).

## Split-difficulty diagnostics

Mean `max_train_tanimoto` ranged from 0.698 (seed 123) to 0.730 (seed 21);
the canonical seed-42 test set was among the harder partitions (0.705).
Seed 42 also had the lowest baseline performance, consistent with lower
train-test chemical similarity. The exploratory correlation (n=5, no strong
claim) was positive for Spearman (rho 0.6 for both models) and positive for
baseline R2 (rho 0.6) but weaker for tuned R2 (rho 0.3).

## Interpretation

**A. Robust across scaffold partitions.** The tuned configuration improves
Spearman and R2 in 5/5 alternative scaffold partitions, with positive mean
paired improvements for both. The effect is larger on harder splits
(seed 42/123) and smaller on the easiest split (seed 7), but it never
reverses. This is partition robustness evidence, not a replacement for the
canonical seed-42 headline numbers.

## Files

```text
src/models/repeated_scaffold_split_study.py
results/repeated_scaffold_split_results.csv
results/repeated_scaffold_split_summary.json
results/figures/repeated_scaffold_spearman.png
results/figures/repeated_scaffold_r2.png
results/figures/repeated_scaffold_similarity.png
```

## Reproducibility

```bash
conda activate egfr-aidd
python src/models/repeated_scaffold_split_study.py
```

The summary JSON records all split counts/scaffold counts, test metrics,
paired deltas, difficulty diagnostics, and package versions. No model
configuration was changed after seeing these results; the locked canonical
model remains `max_features = 0.20`.
