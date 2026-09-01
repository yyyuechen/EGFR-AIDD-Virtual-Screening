# Final Project Summary

## Project objective

Build an end-to-end, portfolio-quality computational pipeline that predicts
EGFR inhibitor activity from molecular structure, evaluates how well the
models generalize to new chemistry, characterizes where predictions become
unreliable, and prioritizes candidates through virtual screening, ADMET
filtering, and molecular docking.

## Dataset

* 10,161 curated EGFR IC50 molecules from ChEMBL (UniProt P00533 /
  CHEMBL203).
* Continuous pIC50 endpoint; exact IC50 measurements only, normalized to nM.
* RDKit validation/standardization and median aggregation for repeated
  measurements.

## Workflow

```text
ChEMBL EGFR -> data curation -> 10,161 molecules
-> molecular representations (Morgan / GNN / SMILES Transformer)
-> scaffold-aware evaluation -> applicability-domain analysis
-> virtual screening -> ADMET filtering -> EGFR docking
-> candidate prioritization
```

## Representation methods

* Morgan fingerprints (radius=2, 2048 bits) with RandomForest / XGBoost.
* Molecular graphs with GCN / GIN (PyTorch Geometric).
* SMILES token sequences with a small Transformer encoder.

## Validation strategy

* Random split (optimistic reference).
* Standard Bemis-Murcko scaffold split (80/10/10, seed 42): the primary
  scaffold-aware benchmark.
* Hard scaffold OOD split: extreme stress test where all 2,033 test
  scaffolds are singletons.
* External set: 59 heterogeneous ChEMBL IC50 records as a transferability
  stress test.
* Multi-seed audit across {42, 7, 123, 2024, 777}.

## Key numerical results

| Result | Value |
|---|---:|
| Dataset size | 10,161 molecules |
| Morgan-RF standard scaffold R2 | 0.573 |
| Morgan-RF standard scaffold Spearman | 0.759 |
| Morgan-RF hard scaffold OOD R2 | 0.398 |
| Morgan-RF hard scaffold OOD Spearman | 0.651 |
| Retrospective screening ROC-AUC | 0.885 |
| Retrospective screening PR-AUC | 0.918 |
| Retrospective screening EF1% | 1.713 |
| Best erlotinib redocking RMSD | 1.238 A |
| Vina models with RMSD < 2 A | 6 / 9 |

## Applicability-domain findings

* `Spearman(max_train_tanimoto, absolute_error) = -0.232 (p < 1e-10)`.
* MAE increases monotonically from 0.507 (similarity >=0.8) to 1.068
  (similarity <0.4).
* Ranking remains useful in lower-similarity regions: ROC-AUC 0.858 for
  similarity <0.6 vs 0.944 for similarity >=0.6.

## External validation findings

* RandomForest external Spearman mean +0.206, positive in 5/5 seeds.
* Equal-weight RF+Transformer ensemble mean +0.024, positive in 3/5 seeds.
* Single-seed ensemble gains were not reproducible; external transferability
  is substantially weaker than internal scaffold-aware performance.

## Docking validation

AutoDock Vina docking of 20/20 shortlist molecules against EGFR 1M17 was
validated by redocking the erlotinib co-crystal ligand: best RMSD 1.238 A
and 6/9 poses within 2 A of the crystal pose.

## Main conclusions

* Random splitting is optimistic relative to scaffold-aware evaluation.
* Morgan-RF is the strongest and most robust baseline in this project.
* More complex learned representations did not consistently outperform
  Morgan fingerprints under this dataset/training regime.
* Prediction reliability decreases with chemical novelty, but ranking stays
  useful even at lower similarity.
* External transferability is weak; multi-seed analysis favors RandomForest.
* Docking geometry is trustworthy enough for screening-level interpretation.

## Limitations

* Retrospective only; no prospective or wet-lab validation.
* Morgan Tanimoto is representation-dependent and activity cliffs remain
  possible.
* Assay heterogeneity affects activity labels.
* Vina scores and ADMET flags are screening approximations, not
  experimental measurements.
* The standard scaffold split is one deterministic realisation.
