# M2-M6 result analysis and external validation

## 1. What the benchmark actually measures

All models use the same Murcko scaffold split: 8,128 train / 2,033 test.
The test set is deliberately extreme:

- 2,033 / 2,033 test scaffolds are singleton scaffolds.
- Scaffold test pIC50 mean is 6.676 vs 7.010 in train (label shift).

So M3-M5 measure extrapolation to brand-new chemistry, not generalization
inside a known scaffold family.

| Milestone | Model / split | R2 | RMSE | MAE | Spearman |
|---|---:|---:|---:|---:|
| M2 | Morgan RF, random split | 0.701 | 0.738 | 0.536 | 0.837 |
| M2 | Morgan XGB, random split | 0.634 | 0.816 | 0.624 | 0.803 |
| M3 | Morgan RF, scaffold split | 0.398 | 0.998 | 0.759 | 0.651 |
| M3 | Morgan XGB, scaffold split | 0.324 | 1.058 | 0.812 | 0.605 |
| M4 | GCN, scaffold split | 0.136 | 1.196 | 0.965 | 0.387 |
| M4 | GIN, scaffold split | 0.218 | 1.138 | 0.914 | 0.451 |
| M5 | SMILES Transformer, scaffold split | 0.044 | 1.258 | 0.987 | 0.469 |

Reading: the random-split baselines are reasonable. The scaffold-split drop is
expected and was already documented, but the absolute numbers are weak enough
that these models cannot be presented as reliable predictors of new scaffolds.

## 2. External validation of the M6 candidate ranking

The 60 M6 candidates were excluded from M1 only because their raw IC50 units
are non-standard (54 ug/mL, 6 uM, 1 "10^3 uM", and 1 missing value). The raw
ChEMBL records can still be converted to pIC50, which gives a crude external
test set of 59 molecules that the models never trained on.

Run:

```bash
conda activate egfr-aidd
python src/models/external_candidate_validation.py
```

Outputs:

```text
results/m6_external_validation.json
results/m6_external_validation.csv
```

Summary (n = 59, external pIC50 mean 4.72, model training mean 6.94):

| Prediction column | Pearson | Spearman | MAE | RMSE |
|---|---:|---:|---:|---:|
| rf_pred | +0.07 | +0.03 | 1.00 | 1.22 |
| xgb_pred | +0.10 | +0.11 | 1.00 | 1.18 |
| gcn_pred | -0.03 | -0.01 | 0.96 | 1.18 |
| gin_pred | -0.06 | -0.08 | 0.98 | 1.20 |
| transformer_pred | -0.32 | -0.23 | 1.09 | 1.53 |
| ensemble_pic50 | -0.19 | -0.26 | 0.99 | 1.25 |

Examples of the failure mode:

- CHEMBL2316916 is ranked 1 by the ensemble (predicted 7.23), but its raw
  IC50 converts to pIC50 4.97 (~10.6 uM).
- CHEMBL483830 is the most potent external compound (pIC50 7.60, 0.01 ug/mL)
  but is ranked 48 by the ensemble (predicted 5.14).

Caveats:

- These are heterogeneous, mostly old IC50 measurements in ug/mL; they are a
  crude sanity check, not a gold standard.
- The candidate set is biased toward weak compounds (external mean 4.72 vs
  training mean 6.94), which is exactly the extrapolation regime where the
  models are weakest.
- If "10^3 uM" is instead interpreted as plain uM, the ensemble Spearman is
  -0.18, so the conclusion is robust to that unit ambiguity.

## 3. Verdict

The M2-M6 numbers are internally reproducible, but they do not currently
support a trustworthy virtual-screening hit list:

1. M2 shows the representation and pipeline are sane on a random split.
2. M3-M5 show that all five models struggle on singleton-scaffold
   extrapolation, with the deep models worse than Morgan baselines.
3. The M6 external check is not just weak, it is negative: ranking a
   candidate higher does not correspond to a stronger measured IC50.

Recommended checkpoint before M7/M8:

- Option A (recommended): pause and improve the modeling first. Add the 59
  external molecules as a permanent external test set, recalibrate or tune
  the models (especially RF/XGB, which are the strongest), and only re-run
  virtual screening once external Spearman is clearly positive.
- Option B: continue to M7/M8 as an educational workflow demonstration, but
  explicitly state that the current shortlist is not validated and docking
  must not be presented as confirming the ensemble ranking.

## 4. M6.5 improvement loop (external gate passed)

After the negative external check, an improvement loop was run without using
the external labels for tuning:

- The 59 raw-IC50 molecules stayed fixed as the external test set.
- Four approved EGFR inhibitors available in the local export (erlotinib,
  gefitinib, afatinib, lapatinib) were added as positive controls and removed
  from training.
- Hyperparameters were selected on a candidate-like internal validation split
  (the 25% of M1 molecules most similar to the screening library).
- The final screening model is RandomForest only, because XGB, GCN/GIN and
  the Transformer did not transfer to the external domain.

Result:

| Metric | M6 original ensemble | M6.5 RF (calibrated) |
|---|---:|---:|
| External Spearman | -0.26 | +0.233 |
| External Pearson | -0.19 | +0.151 |
| External MAE | 0.99 | 1.06 |

Positive-control ranks in the improved 64-molecule shortlist:

| Control | Rank |
|---|---:|
| Afatinib | 1 |
| Erlotinib | 2 |
| Gefitinib | 3 |
| Lapatinib | 4 |

Interpretation: the external Spearman is now positive but still modest
(p=0.076, n=59); the positive-control check is the stronger evidence that the
ranking is meaningful. Osimertinib was not included because the local ChEMBL
export has no SMILES / IC50 record for it.

Reproducible files:

```text
src/models/m65_model_improvement.py
results/m65_model_improvement.json
results/m65_improved_shortlist.csv
docs/README_M65.md
```

The external gate is considered passed for M7/M8, with the caveat that the
M6.5 RF shortlist is a screening hypothesis, not a confirmed hit list.

### Fair comparison after retraining all five models

The first M6.5 loop only retrained RF / XGB, so the comparison with the deep
models was not apples-to-apples. A follow-up run retrained GCN / GIN /
Transformer under the same candidate-like split and the same selection
principle (best-epoch selection by validation Spearman).

| Model | External Spearman | External Pearson | MAE | Control ranks |
|---|---:|---:|---:|---:|
| RandomForest | +0.233 | +0.151 | 1.087 | 1, 2, 3, 4 |
| XGBoost | -0.042 | -0.095 | 0.968 | 1, 2, 3, 4 |
| GCN | -0.252 | -0.210 | 1.030 | 1, 2, 3, 4 |
| GIN | +0.149 | +0.082 | 0.993 | 1, 2, 3, 4 |
| Transformer | +0.244 | +0.123 | 0.945 | 2, 4, 6, 7 |

Key correction to the earlier narrative: after the fair retrain, the SMILES
Transformer is competitive with RF on the external ranking (+0.244 vs
+0.233); its previous -0.233 came from the M6 full-data training protocol,
not from the representation itself. RF remains the safest single choice, but
"deep models are all worse" is no longer the correct statement.

Reproducible files:

```text
src/models/m65_fair_comparison.py
results/m65_fair_comparison.json
results/m65_fair_comparison_predictions.csv
results/figures/m65_fair_comparison.png
```

### RF + Transformer weighted ensemble

Since RF and Transformer were the two best external ranking models, a
weighted mean was validated under the same protocol. The weight was selected
by candidate-like validation Spearman; the external 59 molecules were used
only once as the final check.

| Setup | External Spearman | External Pearson | MAE | Control ranks |
|---|---:|---:|---:|---:|
| RF only | +0.233 | +0.151 | 1.087 | 1, 2, 3, 4 |
| Transformer only | +0.244 | +0.123 | 0.945 | 2, 4, 6, 7 |
| Equal weight 0.5 / 0.5 | +0.286 | +0.157 | 0.977 | 1, 2, 3, 4 |
| Validation-selected w_rf=0.85 | +0.271 | +0.164 | 1.048 | 1, 2, 3, 4 |

The equal-weight ensemble is the simplest and has the best external Spearman
(+0.286, p=0.028); the validation-selected weight also passes (+0.271,
p=0.038). Both improve over the single models and keep all four positive
controls in the top 4.

Reproducible files:

```text
src/models/m65_rf_transformer_ensemble.py
results/m65_rf_transformer_ensemble.json
results/m65_rf_transformer_ensemble_predictions.csv
results/figures/m65_rf_transformer_ensemble.png
```
