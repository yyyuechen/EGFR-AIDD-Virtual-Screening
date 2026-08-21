# M6.5 - Model improvement loop

## What this milestone does

M6's original ensemble was ranked by the mean of Morgan / graph / SMILES
Transformer predictions. A crude external check using the 59 candidate
molecules with non-nM ChEMBL IC50 records gave a **negative** Spearman
(-0.26), so the shortlist could not be trusted. M6.5 runs one improvement
loop without retouching the external labels:

1. The 59 raw-IC50 molecules are fixed as a permanent external test set.
2. Four approved EGFR inhibitors that exist in the local ChEMBL export are
   added as positive controls and removed from the training pool.
3. A candidate-like validation split is built from the M1 molecules most
   similar to the screening library (Morgan Tanimoto), so hyperparameters
   are selected in the screening domain without using external labels.
4. A small RF / XGB grid is scored on that internal validation split.
5. Linear and isotonic calibration are fit on internal validation predictions
   and applied to the external predictions.
6. The improved shortlist is re-ranked and the positive-control ranks are
   checked.

## Result

| Model / setup | External Spearman | External Pearson | MAE |
|---|---:|---:|---:|
| M6 original 5-model ensemble | -0.259 | -0.194 | 0.987 |
| M6.5 RF (uncalibrated) | +0.233 | +0.151 | 1.087 |
| M6.5 RF (linear calibration) | +0.233 | +0.151 | 1.064 |
| M6.5 RF (isotonic calibration) | +0.213 | +0.140 | 1.034 |
| M6.5 XGB (best internal variant, for reference) | -0.042 | -0.095 | 0.968 |

External Spearman p-value for the calibrated RF is 0.076 with n=59, so the
correlation is a positive trend but not significant by itself. The
positive-control check is a necessary but not sufficient sanity check: the
controls are far more active than the weak candidate library (reference pIC50
7.5-9.0 vs external mean 4.72), so high control ranks are expected even for
models with poor external ranking. The primary evidence remains the external
Spearman.

| Control | Rank in improved 64-molecule shortlist |
|---|---:|
| Afatinib | 1 |
| Erlotinib | 2 |
| Gefitinib | 3 |
| Lapatinib | 4 |

All four known strong EGFR inhibitors are ranked in the top 4, and the first
non-control candidate is rank 5.

Note: this check has low discriminative power. In the fair comparison below,
even models with negative or near-zero external Spearman (XGB -0.04,
GCN -0.25) place all four controls in the top 4, because the controls are
much more active than the candidate library. High control ranks confirm the
pipeline is not broken; they do not prove that the ranking signal is strong.

## Why the model changed

- XGB did not transfer to the external domain in this loop (best internal
  XGB variant had external Spearman -0.04), so the final screening ranking
  uses RandomForest only.
- The original graph / Transformer models remain useful as an educational
  comparison, but they are no longer part of the screening ensemble.
- Calibration improves absolute error but, being monotonic, does not change
  the ranking much; the ranking improvement comes from the RF retrain on a
  screening-like internal split and from removing the poorly transferring
  models.

## Caveats

- The 59 external labels come from heterogeneous, mostly old ChEMBL IC50
  records in ug/mL; they are a sanity check, not a gold standard.
- The candidate library is dominated by weak compounds (external mean 4.72 vs
  training mean 6.94), so this test measures extrapolation to weak / novel
  chemistry.
- Osimertinib is not in the local ChEMBL export, so it could not be added as a
  verified local control without a source for its SMILES and IC50.
- The M6.5 result is a single held-out estimate, not a repeated cross-
  validation; treating +0.233 as a guaranteed ranking quality would overstate
  the evidence.

## Run

```bash
conda activate egfr-aidd
python src/models/m65_model_improvement.py
```

## Outputs

```text
results/m65_model_improvement.json
results/m65_improved_shortlist.csv
results/figures/m65_external_validation.png
```

The isotonic calibration segment mapping is saved under
`isotonic_calibration` (`X_thresholds` / `y_thresholds`) in
`results/m65_model_improvement.json`; the calibrated metrics are under
`external_metrics.isotonic_calibrated_RF`.

## Fair comparison across all five models

To make the comparison fair, GCN / GIN / Transformer were retrained under the
same M6.5 protocol: same training pool, same candidate-like validation split,
same seed, same external 59-molecule test set, and the same selection
principle (best candidate-like validation Spearman; trees via grid, deep
models via best-epoch selection over 30 epochs).

Run:

```bash
python src/models/m65_fair_comparison.py
```

| Model | Representation | External Spearman | External Pearson | MAE | Controls in top 20 |
|---|---:|---:|---:|---:|---:|
| RandomForest | Morgan (2048 bit) | +0.233 | +0.151 | 1.087 | 1, 2, 3, 4 |
| XGBoost | Morgan (2048 bit) | -0.042 | -0.095 | 0.968 | 1, 2, 3, 4 |
| GCN | molecular graph | -0.252 | -0.210 | 1.030 | 1, 2, 3, 4 |
| GIN | graph + bond features | +0.149 | +0.082 | 0.993 | 1, 2, 3, 4 |
| Transformer | SMILES tokens | +0.244 | +0.123 | 0.945 | 2, 4, 6, 7 |

Interpretation after the fair retrain:

- RF and Transformer are now comparable on external ranking (+0.233 vs
  +0.244), both as positive trends that do not individually reach p<0.05.
  The Transformer's original -0.233 came from the old M6 full-data training
  protocol, not from the architecture being unusable.
- GIN transfers moderately; XGB and GCN still fail on external ranking.
- Every model ranks all four positive controls in the top 7, but this is a
  necessary sanity check, not proof of ranking quality, because the controls
  are much more active than the candidate library.

Outputs:

```text
results/m65_fair_comparison.json
results/m65_fair_comparison_predictions.csv
results/figures/m65_fair_comparison.png
```

## RF + Transformer weighted ensemble

Because RF and Transformer are the two best external ranking models, a
weighted mean of their predictions was validated under the same protocol.
The weight is selected by candidate-like validation Spearman; the external
59 molecules are used only as the final held-out check.

Run:

```bash
python src/models/m65_rf_transformer_ensemble.py
```

| Setup | External Spearman | External Pearson | MAE | Controls in top 20 |
|---|---:|---:|---:|---:|
| RF only | +0.233 | +0.151 | 1.087 | 1, 2, 3, 4 |
| Transformer only | +0.244 | +0.123 | 0.945 | 2, 4, 6, 7 |
| Equal weight 0.5 / 0.5 | +0.286 | +0.157 | 0.977 | 1, 2, 3, 4 |
| Validation-selected w_rf=0.85 | +0.271 | +0.164 | 1.048 | 1, 2, 3, 4 |

The equal-weight ensemble is the simplest and has the best external Spearman
(+0.286, p=0.028); the validation-selected weight also gives a positive trend
(+0.271, p=0.038). Both are better than either single model. Important
caveat: the equal-weight choice was made after inspecting the external
results, so +0.286 is an exploratory post-selection optimum rather than a
pre-specified comparison. The single models are only positive trends (RF
+0.233, p=0.076; Transformer +0.244, p=0.062); only the equal-weight ensemble
reaches p<0.05 on this single external set.

A later P0-3 multi-seed audit (`docs/README_P03.md`) retrained these models
on the cleaned external labels across five seeds. RandomForest is the only
consistently positive model (mean +0.206, 5/5 seeds), while the equal-weight
ensemble is seed-sensitive (mean +0.024, 3/5 seeds); the RF-only model is
therefore the more defensible final screening model.

Outputs:

```text
results/m65_rf_transformer_ensemble.json
results/m65_rf_transformer_ensemble_predictions.csv
results/figures/m65_rf_transformer_ensemble.png
```
