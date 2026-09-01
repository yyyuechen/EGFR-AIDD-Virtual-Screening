# Project Summary

## Objective

Build an end-to-end computational pipeline that predicts EGFR inhibitor
activity from molecular structure, evaluates how well the models generalize
to new chemistry, characterizes where predictions become unreliable, and
prioritizes candidate molecules through virtual screening, ADMET filtering,
and molecular docking.

## Dataset

* 10,161 curated EGFR IC50 molecules from ChEMBL (UniProt P00533 /
  CHEMBL203).
* Continuous pIC50 endpoint; exact IC50 measurements only, normalized to nM.
* RDKit validation/standardization and median aggregation for repeated
  measurements.
* Final dataset: `data/processed/egfr_activity_final.csv`; processing
  provenance: `results/data_processing_summary.json`.

## Workflow

```text
ChEMBL EGFR -> data curation -> 10,161 molecules
-> molecular representations (Morgan / GNN / SMILES Transformer)
-> scaffold-aware evaluation -> applicability-domain analysis
-> virtual screening -> ADMET filtering -> EGFR docking
-> candidate prioritization
```

## Representation Methods

* Morgan fingerprints (radius=2, 2048 bits) with RandomForest / XGBoost.
* Molecular graphs with GCN / GIN (PyTorch Geometric).
* SMILES token sequences with a small Transformer encoder.

## Validation Strategy

* Random split (optimistic reference).
* Standard Bemis-Murcko scaffold split (80/10/10, seed 42): the primary
  scaffold-aware benchmark.
* Hard scaffold OOD split: extreme stress test where all 2,033 test
  scaffolds are singletons.
* External set: 59 heterogeneous ChEMBL IC50 records as a transferability
  stress test.
* Multi-seed audit across {42, 7, 123, 2024, 777}.

## Key Numerical Results

| Result | Value |
|---|---:|
| Dataset size | 10,161 molecules |
| Morgan-RF standard scaffold R2 | 0.573 (baseline) / 0.592 (final tuned) |
| Morgan-RF standard scaffold Spearman | 0.759 / 0.766 |
| Morgan-RF hard scaffold OOD R2 | 0.398 |
| Morgan-RF hard scaffold OOD Spearman | 0.651 |
| Retrospective screening ROC-AUC | 0.885 |
| Retrospective screening PR-AUC | 0.918 |
| Retrospective screening EF1% | 1.713 |
| External RF Spearman (5 seeds) | +0.206 mean, positive 5/5 |
| Best erlotinib redocking RMSD | 1.238 A |
| Vina models with RMSD < 2 A | 6 / 9 |

## Applicability-Domain Findings

* `Spearman(max_train_tanimoto, absolute_error) = -0.232 (p < 1e-10)`.
* MAE increases monotonically from 0.507 (similarity >= 0.8) to 1.068
  (similarity < 0.4).
* Ranking remains useful in lower-similarity regions: ROC-AUC 0.858 for
  similarity < 0.6 vs 0.944 for similarity >= 0.6.

## Main Conclusions

* Random splitting is optimistic relative to scaffold-aware evaluation.
* Morgan-RF is the strongest and most robust baseline in this project.
* More complex learned representations did not consistently outperform
  Morgan fingerprints under this dataset and training regime.
* Prediction reliability decreases with chemical novelty, but ranking stays
  useful even at lower similarity.
* External transferability is weaker than internal scaffold-aware
  performance; multi-seed analysis favored RandomForest.
* Docking geometry was validated through erlotinib co-crystal redocking.

See [METHODS_AND_DECISIONS.md](METHODS_AND_DECISIONS.md) for protocol
details, [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for how to rerun the
pipeline, and [LIMITATIONS.md](LIMITATIONS.md) for caveats.
