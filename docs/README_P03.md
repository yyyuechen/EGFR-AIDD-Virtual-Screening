# P0-3 - External label audit and multi-seed reproducibility

## Why this milestone exists

The M6.5 external validation result (+0.286 Spearman, p=0.028) had two known
weaknesses:

1. The 59 external labels are heterogeneous ChEMBL IC50 records in non-nM
   units, and the original builder kept the **most potent record** per
   molecule, which is an optimistic choice.
2. The final model was trained once with seed 42, so the p-value ignored
   seed-to-seed variability.

P0-3 fixes both: it audits and cleans the external labels, then retrains the
final screening models across five seeds and reports the spread.

---

## P0-3a - External label audit

### Records

| Check | Value |
|---|---:|
| Raw non-nM IC50 records | 61 |
| Unique molecules | 59 |
| Exact records (`standard_relation = "="`) | 61 / 61 |
| Units | 54 ug.mL-1, 6 /uM, 1 10^3 uM |
| Publication year range | 1992 - 2024 |
| Molecules with duplicate records | 2 |
| Ambiguous 10^3 uM records | 1 |

All 61 records are exact IC50 binding measurements (assay type B), so no
censored records had to be dropped. The only real ambiguity is the single
`10^3 uM` record, which is kept but flagged.

### Cleaning rule

The original external builder took the **most potent** record per molecule.
P0-3 switches to the **median pIC50** across records, which matches the M1
molecule-level aggregation rule and avoids the optimistic
"best record wins" choice.

### Sensitivity: max vs median labels

Only 2 of 59 molecules have more than one record, so the cleaning rule
changes only 2 labels:

| Label version | External Spearman | p-value | MAE |
|---|---:|---:|---:|
| Max (old rule) | +0.286 | 0.028 | 0.977 |
| Median (clean rule) | +0.291 | 0.025 | 0.964 |

The label cleanup barely changes the result and slightly improves MAE. The
external Spearman conclusion is stable to this choice.

---

## P0-3b - Multi-seed reproducibility

The final screening models were retrained on the **clean** external labels
with the same deterministic candidate-like split across seeds
`{42, 7, 123, 2024, 777}`:

| Seed | RF | XGB | Transformer | Equal RF+TF ensemble |
|---:|---:|---:|---:|---:|
| 42 | +0.233 | -0.044 | +0.250 | +0.291 |
| 7 | +0.153 | -0.033 | -0.308 | -0.239 |
| 123 | +0.237 | -0.233 | -0.019 | +0.113 |
| 2024 | +0.224 | +0.025 | -0.117 | +0.033 |
| 777 | +0.185 | +0.074 | -0.138 | -0.077 |

Summary across seeds:

| Model | Mean +/- SD | Min | Max | Positive seeds |
|---|---:|---:|---:|---:|
| RandomForest | +0.206 +/- 0.036 | +0.153 | +0.237 | 5 / 5 |
| XGBoost | -0.042 +/- 0.117 | -0.233 | +0.074 | 2 / 5 |
| Transformer | -0.066 +/- 0.205 | -0.308 | +0.250 | 1 / 5 |
| Equal RF+TF ensemble | +0.024 +/- 0.199 | -0.239 | +0.291 | 3 / 5 |

Additional checks on the equal-weight ensemble:

| Check | Value |
|---|---:|
| Bootstrap 95% CI (seed 42) | [0.047, 0.519] |
| Permutation p-value (seed 42) | 0.013 |
| Top-20 overlap with seed 42 | 12 - 20 of 20 |

### How to read this

* **RandomForest is the only consistently positive model.** Its external
  Spearman is +0.15 to +0.24 in every seed, with a small SD. This is the
  defensible screening signal.
* **Transformer and the equal-weight ensemble are seed-sensitive.** The
  equal-weight ensemble is positive in only 3/5 seeds and has a mean close
  to zero (+0.024). The seed-42 result of +0.286 was a favorable draw, not a
  reproducible property.
* **Controls still rank 1-4 in every seed**, which confirms the earlier
  review: that check is necessary but has no discriminative power.
* The bootstrap CI [0.047, 0.519] and permutation p=0.013 describe only the
  seed-42 draw; they should not be read as proof of a stable ranking.

### Recommendation

For any prospective screening claim, the final model should be
**RandomForest only** (or a multi-seed RF average), not the equal-weight
RF + Transformer ensemble. The Transformer is useful as an educational
comparison, but its external ranking is not reproducible enough to be the
basis of the final shortlist.

---

## Files

```text
src/models/p03_external_label_audit.py
src/models/p03_multiseed_reproducibility.py

results/p03_external_label_audit.csv
results/p03_external_clean_labels.csv
results/p03_external_label_audit.json
results/p03_multiseed_reproducibility.json
results/p03_multiseed_per_seed.csv
results/figures/p03_multiseed_external_rho.png
```

## How to run

```bash
conda activate egfr-aidd
python src/models/p03_external_label_audit.py
python src/models/p03_multiseed_reproducibility.py --seeds 42,7,123,2024,777
```

The multi-seed script retrains the Transformer five times and takes roughly
40-60 minutes.

## Caveats

* The 59 external molecules are still a small, heterogeneous, single-source
  test set; multi-seed retraining reduces model randomness but not label
  noise.
* `ug.mL-1` conversion uses RDKit molecular weight and `10^3 uM` is treated
  literally as 1 mM with a flag.
* GCN / GIN remain single-seed educational baselines; P0-3 covers the final
  screening models.
* The candidate-like validation split is deterministic (Morgan-based), so
  seed variation only affects model fitting.
