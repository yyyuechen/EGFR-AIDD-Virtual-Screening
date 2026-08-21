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

## Suggested CV description

> Built an end-to-end AI-driven drug discovery pipeline for EGFR
> inhibitors: curated 10,161 ChEMBL IC50 molecules, benchmarked Morgan
> fingerprints, GNNs, and a SMILES Transformer, and showed that Morgan
> fingerprints with Random Forest were the most robust representation.
> Designed scaffold-aware and applicability-domain validation (standard
> scaffold R2 0.573, Spearman 0.759; error increases with chemical novelty),
> ran retrospective screening (ROC-AUC 0.885), multi-seed external
> validation, ADMET filtering, and AutoDock Vina docking validated by
> erlotinib redocking (RMSD 1.238 A).

## 60-90 second interview explanation

"I built a computational drug discovery pipeline for EGFR inhibitors,
starting from 10,161 ChEMBL IC50 measurements. I curated the data into a
clean molecule-level pIC50 dataset, then compared three molecular
representations: Morgan fingerprints, graph neural networks, and a SMILES
Transformer. The key scientific question was generalization to new
chemistry, so I built two scaffold-aware benchmarks: a standard scaffold
split, where the model must predict activity for unseen ring systems, and a
hard out-of-distribution split where every test scaffold is unique. Morgan
fingerprints with Random Forest were the strongest and most robust model,
with R2 0.573 on the standard scaffold split and ROC-AUC 0.885 in a
retrospective screening benchmark. I also added an applicability-domain
analysis showing that predictions become less reliable as molecules become
less similar to the training chemistry, which is useful for knowing when to
trust the model. A five-seed audit showed that Random Forest was
reproducibly positive externally, while a previously attractive ensemble was
not. Finally, I filtered candidates with ADMET descriptors, docked them with
AutoDock Vina, and validated the docking setup by redocking the erlotinib
co-crystal to 1.24 Angstrom. The main takeaways are that scaffold-aware
validation is essential, simpler representations can beat deep models in
this regime, and honest uncertainty analysis matters for real screening."
