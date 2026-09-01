# Methods and Decisions

This document records the modeling protocol and the decisions behind the
final locked configuration. Numbers are the frozen results from the
corresponding result files.

## 1. Data Curation

* Download EGFR activity records and molecule SMILES from ChEMBL
  (`src/data/download_chembl.py`).
* Keep exact IC50 measurements only, normalize all units to nM, and build a
  continuous pIC50 endpoint.
* Validate and standardize SMILES with RDKit (largest fragment + canonical
  SMILES), then aggregate repeated measurements by median pIC50
  (`src/data/preprocess_egfr.py`).
* The final curated set contains 10,161 molecules.

## 2. Molecular Representations

All models are compared under identical splits:

| Representation | Models |
|---|---|
| Morgan fingerprints (radius 2, 2048 bits) | RandomForest / XGBoost |
| Molecular graphs | GCN / GIN |
| SMILES token sequences | Transformer encoder |

The learned graph/sequence models did not consistently outperform the
Morgan fingerprint baseline under this dataset and training regime.

## 3. Evaluation Splits

* **Random 80/20**: optimistic in-distribution reference.
* **Standard scaffold 80/10/10, seed 42**: Bemis-Murcko scaffold groups are
  kept intact and assigned deterministically to train/validation/test.
  Scaffold leakage is zero; this is the primary QSAR benchmark.
* **Hard scaffold OOD 80/20**: extreme stress test in which all 2,033 test
  scaffolds are singletons and no validation split exists.
* **External set**: 59 heterogeneous ChEMBL IC50 records with non-nM units
  as a transferability stress test, not a gold standard.

## 4. Final Morgan-RF Model Selection

Model selection used only the standard-scaffold validation split; the test
set was untouched until the final locked evaluation.

* Morgan radius 1-3 and nBits 512-4096 were broad performance plateaus;
  radius 2 and 2048 bits were retained.
* `min_samples_leaf` above 1 reduced validation performance; `max_depth`
  plateaus above 60; defaults were retained.
* Moderate feature subsampling (`max_features = 0.10-0.20`) was the only
  change that robustly improved scaffold-held-out validation, confirmed
  across five RF seeds.
* Locked configuration: Morgan radius 2 / 2048 bits, 300 trees,
  `min_samples_leaf = 1`, `max_depth = None`, `max_features = 0.20`,
  seed 42. The original default (`max_features = 1.0`) remains the explicit
  baseline.
* Final single evaluation on the held-out scaffold test set:
  RMSE 0.863 vs 0.883, R2 0.592 vs 0.573, Spearman 0.766 vs 0.759
  (paired bootstrap 95% CI excludes zero for RMSE, R2, and Spearman).

Results: `results/final_rf_configuration.json` and
`results/final_rf_test_results.json`.

## 5. Applicability Domain

For every standard-scaffold test molecule, `max_train_tanimoto` is the
maximum Morgan Tanimoto similarity to any training molecule.

* `Spearman(max_train_tanimoto, absolute_error) = -0.232 (p < 1e-10)`.
* MAE increases monotonically from 0.507 (similarity >= 0.8) to 1.068
  (similarity < 0.4).
* Ranking remains useful even at lower similarity: ROC-AUC 0.944 for
  similarity >= 0.6 vs 0.858 for similarity < 0.6.

## 6. Virtual Screening and Known-Inhibitor Sanity Check

* The candidate library contains ChEMBL-derived molecules excluded from the
  modeling dataset (`data/candidates/egfr_candidate_library.csv`).
* The final ranking set contains 64 molecules: 60 candidate-like molecules
  plus 4 known positive controls (afatinib, erlotinib, gefitinib,
  lapatinib).
* Under the locked Morgan-RF model, the four known inhibitors rank 1-4 and
  are recovered 4/4 in the top 5, top 10, and top 20
  (`results/final_known_inhibitor_summary.json`). This is a necessary but
  not sufficient sanity check.
* The retrospective screening benchmark on unseen scaffolds reports
  ROC-AUC 0.885, PR-AUC 0.918, and EF1% 1.713
  (`results/m9_retrospective_benchmark.json`).

## 7. External Robustness

Across five seeds on 59 heterogeneous external records
(`results/p03_multiseed_reproducibility.json`):

* RandomForest external Spearman mean +0.206, positive in 5/5 seeds.
* Equal-weight RF + Transformer ensemble mean +0.024, positive in 3/5
  seeds.
* Single-seed ensemble gains were not reproducible; external performance is
  substantially weaker than internal scaffold-aware performance.

## 8. Final Candidate Prioritization

The closing candidate study considered only non-training candidates (60
eligible molecules after excluding known controls) and prioritized them by
convergent evidence:

* canonical locked Morgan-RF prediction and rank;
* 25 frozen-model stability runs (5 scaffold partitions x 5 RF seeds);
* applicability-domain support with historical error context;
* physicochemical descriptors (Lipinski, Veber, ESOL, QED);
* AutoDock Vina docking outcome.

Tier rules are explicit and not a weighted score. Tier 1:
**CHEMBL13983** (pred pIC50 5.93, rank 4, median rank 9.0, top-10 frequency
0.56, max train similarity 0.64, ADMET pass, supportive docking -9.22).
Tier 2: CHEMBL5564149, CHEMBL131921, CHEMBL2315700, CHEMBL243664.
These are hypotheses for experimental validation, not confirmed inhibitors.

Final artifacts: `results/final_candidate_shortlist.csv`,
`results/final_candidate_evidence_matrix.csv`, and
`results/final_candidate_prioritization_summary.json`.

## 9. Docking Protocol

* Receptor: EGFR kinase domain PDB 1M17, prepared as a rigid receptor PDBQT
  (`data/docking/1m17_receptor.pdbqt`).
* Pocket: centered on the co-crystallized erlotinib (AQ4) via
  `data/docking/vina_box.json`.
* Ligands: 20/20 shortlist molecules prepared from SMILES with RDKit and
  Meeko; AutoDock Vina 1.2.x installed separately.
* Pose validation: redocking erlotinib gives best RMSD 1.238 A and 6/9
  Vina models within 2 A of the crystal pose
  (`results/m9_pose_validation.json`).
