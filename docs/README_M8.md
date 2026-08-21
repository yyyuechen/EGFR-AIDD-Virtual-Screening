# M8 - Final analysis report and error analysis

## Project summary

This educational project built a complete EGFR inhibitor discovery-style
pipeline:

```text
M1  ChEMBL EGFR pIC50 dataset (CHEMBL203)
M2  Morgan fingerprint RF / XGB baseline (random split)
M3  Morgan baseline under random vs Murcko scaffold split
M4  GCN / GIN graph baselines under the scaffold split
M5  SMILES Transformer baseline under the scaffold split
M6  Ensemble virtual screening on 60 candidates
M6.5 External validation + model improvement + fair retrain
M7  ADMET descriptors + AutoDock Vina docking
M8  Final analysis and error analysis (this report)
```

The M6.5 final screening model is an equal-weight RF + SMILES Transformer
ensemble. The equal-weight choice was made after inspecting the external IC50
results (external Spearman +0.286, p=0.028), so this should be read as an
exploratory post-selection optimum rather than a pre-specified result. The
single-model alternatives are positive trends that do not individually reach
p<0.05.

## Milestone regression metrics

| Milestone | Split | Model | R2 | MAE | Spearman |
|---|---|---|---:|---:|---:|
| M2 | random | RandomForest | 0.701 | 0.536 | 0.837 |
| M2 | random | XGBoost | 0.634 | 0.624 | 0.803 |
| M3 | random | RandomForest | 0.701 | 0.536 | 0.837 |
| M3 | random | XGBoost | 0.634 | 0.624 | 0.803 |
| M3 | scaffold | RandomForest | 0.398 | 0.759 | 0.651 |
| M3 | scaffold | XGBoost | 0.324 | 0.812 | 0.605 |
| M4 | scaffold | GCN | 0.136 | 0.965 | 0.387 |
| M4 | scaffold | GIN | 0.218 | 0.914 | 0.451 |
| M5 | scaffold | Transformer | 0.044 | 0.987 | 0.469 |

The scaffold split is deliberately hard: all 2,033 test molecules are
singleton scaffolds, and the test pIC50 mean is 6.68 vs 7.01 in training.

## External validation history

| Setup | External Spearman | p-value | MAE |
|---|---:|---:|---:|
| M6 original 5-model ensemble | -0.259 | 0.048 | 0.987 |
| M6.5 RF (calibrated) | +0.233 | 0.076 | 1.064 |
| M6.5 fair retrain: GCN | -0.252 | 0.054 | 1.030 |
| M6.5 fair retrain: XGBoost | -0.042 | 0.751 | 0.968 |
| M6.5 fair retrain: GIN | +0.149 | 0.261 | 0.993 |
| M6.5 fair retrain: RandomForest | +0.233 | 0.076 | 1.087 |
| M6.5 fair retrain: Transformer | +0.244 | 0.062 | 0.945 |
| RF + Transformer equal weight | **+0.286** | **0.028** | **0.977** |

The external test is the 59 molecules with non-nM ChEMBL IC50 records. The
positive controls (afatinib, erlotinib, gefitinib, lapatinib) rank in the top
4 of the improved shortlist.

This control check is necessary but not sufficient: the controls are far more
active than the weak candidate library (pIC50 7.5-9.0 vs external mean 4.72),
so high control ranks are expected even for models with poor external ranking
(the fair-comparison table above shows this explicitly).

## Model agreement on the 64-molecule scoring set

Spearman correlations between fair-comparison predictions:

| | RF | XGB | GCN | GIN | TF |
|---|---:|---:|---:|---:|---:|
| RF | 1.00 | 0.65 | 0.34 | 0.50 | 0.44 |
| XGB | 0.65 | 1.00 | 0.55 | 0.44 | 0.27 |
| GCN | 0.34 | 0.55 | 1.00 | 0.57 | 0.48 |
| GIN | 0.50 | 0.44 | 0.57 | 1.00 | 0.45 |
| TF | 0.44 | 0.27 | 0.48 | 0.45 | 1.00 |

RF-Transformer consensus on the scoring set:

- Mean absolute prediction difference: 0.63 pIC50 units
- Max absolute difference: 2.13
- Molecules with difference > 1.0: 9 / 64

The two models chosen for the final ensemble are complementary (0.44
correlation), which is one reason the weighted mean improves external
ranking.

## External error analysis

The original M6 models systematically overestimate the weak external
candidates:

| Model prediction | Mean bias | MAE | RMSE | External Spearman |
|---|---:|---:|---:|---:|
| RF | +0.79 | 1.00 | 1.22 | +0.03 |
| XGB | +0.78 | 1.00 | 1.18 | +0.11 |
| GCN | +0.63 | 0.96 | 1.18 | -0.01 |
| GIN | +0.69 | 0.98 | 1.20 | -0.08 |
| Transformer | +0.67 | 1.09 | 1.53 | -0.23 |
| M6 ensemble | +0.70 | 0.99 | 1.25 | -0.26 |

This is the expected out-of-distribution behavior for a candidate library
whose measured activity is systematically weaker than the training set
(external mean 4.72 vs training mean 6.94). It is documented rather than
hidden, and the M6.5 improvement corrected the ranking signal while
calibration reduced absolute error.

## Final M7 shortlist and confidence tiers

20 molecules were docked successfully. Confidence tier counts:

| Check | Count |
|---|---:|
| Docked | 20 |
| ADMET_ok | 12 |
| Docking hit (<= -7.0 kcal/mol) | 17 |
| High confidence (4/4 signals) | 3 |
| Mid confidence (>=2 signals) | 17 |
| Low confidence | 0 |

Top 10:

| Rank | ChEMBL ID | Name | pIC50 | Affinity | ADMET | Tier |
|---:|---|---|---:|---:|---:|---|
| 1 | CHEMBL1173655 | afatinib | 7.95 | -8.42 | True | high |
| 2 | CHEMBL554 | lapatinib | 7.76 | -8.82 | False | mid |
| 3 | CHEMBL939 | gefitinib | 7.59 | -8.19 | True | high |
| 4 | CHEMBL553 | erlotinib | 7.70 | -7.01 | True | high |
| 5 | CHEMBL243664 | - | 6.92 | -8.64 | True | mid |
| 6 | CHEMBL521470 | - | 6.54 | -9.52 | True | mid |
| 7 | CHEMBL13983 | - | 6.39 | -9.22 | True | mid |
| 8 | CHEMBL5564149 | - | 6.31 | -9.08 | True | mid |
| 9 | CHEMBL554993 | - | 6.16 | -9.94 | False | mid |
| 10 | CHEMBL2316916 | - | 7.01 | -7.74 | False | mid |

The four approved EGFR inhibitor controls occupy ranks 1-4. This is a
necessary but not sufficient sanity check; the external Spearman and the M9
retrospective benchmark are the primary evidence of ranking quality.

## Files

Analysis script and outputs:

```text
src/models/m8_final_report.py
results/m8_final_analysis.json
results/m8_final_candidates.csv
results/figures/m8_model_agreement.png
results/figures/m8_external_error.png
results/figures/m8_final_shortlist.png
```

All earlier milestone results are referenced by
`results/m8_final_analysis.json`, which contains the full metric history.

## Caveats

- The scaffold split is an extreme extrapolation benchmark; low R2 on that
  split is expected and not a claim of model failure.
- The 59-molecule external IC50 set is heterogeneous and crude; it is a
  sanity check, not a gold standard.
- AutoDock Vina was used with a rigid receptor and a box defined by the
  erlotinib co-crystal; docking scores are screening approximations.
- The composite score weights (0.5 / 0.3 / 0.2) and confidence tiers are
  educational modeling choices, not experimentally validated.
- Osimertinib is not included because the local ChEMBL export has no
  SMILES / IC50 record for it.
- The equal-weight RF + Transformer ensemble was selected after inspecting
  the external results, so the +0.286 p-value carries post-selection optimism
  and needs prospective confirmation.
- The positive-control ranks are a sanity check only; they do not distinguish
  good from bad ranking models because the controls are much more active than
  the candidate library.
- The later P0-3 multi-seed audit (`docs/README_P03.md`) shows the
  equal-weight ensemble is seed-sensitive (mean external Spearman +0.024,
  only 3/5 seeds positive), while RandomForest alone is consistently positive
  (+0.206, 5/5 seeds). The M8 ensemble ranking should be treated as
  exploratory; RF-only is the more defensible final screening model.
