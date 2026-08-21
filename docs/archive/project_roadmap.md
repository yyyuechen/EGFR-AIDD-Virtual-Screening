# EGFR AIDD Project Roadmap

This project is a practical extension of the AIDD course
(`AIDD人工智能药物设计与深度学习蛋白质预测`). Each milestone is intentionally
small enough to review before moving to the next.

```text
M1. EGFR activity dataset construction   [COMPLETE]
M2. Morgan fingerprint baseline          [COMPLETE]
M3. Model evaluation (random vs scaffold split)   [COMPLETE]
M4. Molecular graph models (GCN / MPNN)  [COMPLETE]
M5. SMILES sequence model (Transformer)  [COMPLETE]
M6. Virtual screening                      [COMPLETE]
M7. ADMET + docking                        [COMPLETE]
M8. Candidate ranking + final analysis     [COMPLETE]
M9. Evidence milestones                    [COMPLETE]
P0-3. External label audit + multi-seed    [COMPLETE]
```

---

## M1 - EGFR activity dataset construction (implemented)

* Objective: build a clean, molecule-level EGFR IC50 dataset.
* Data: ChEMBL, UniProt P00533.
* Course connection: Day01 ChEMBL download + Day02 RDKit / pIC50.
* Outputs: `data/processed/egfr_activity_final.csv`,
  `results/data_processing_summary.json`, QC notebook.

## M2 - Morgan fingerprint baseline

* Objective: first ML model on the M1 dataset.
* Data: `canonical_smiles` -> RDKit Mol -> Morgan fingerprint.
* Model: RandomForest (and optionally XGBoost).
* Course connection: Day02 `00-rdkit.ipynb`, `04-基于ML的生物活性预测.ipynb`.

## M3 - Model evaluation

* Objective: compare random split vs scaffold split.
* Why: scaffold split better mimics "new chemistry" generalization.
* Course connection: Day02 train/test/validation + model evaluation.
* Implemented with the same Morgan baseline from M2:
  `src/models/evaluate_splits.py`, `notebooks/03_morgan_split_comparison.ipynb`.
* Outputs: `results/m3_morgan_split_comparison_results.json`,
  `results/figures/m3_random_vs_scaffold_split.png`.

## M4 - Molecular graph models

* Objective: GCN / MPNN on molecular graphs.
* Data: RDKit Mol -> PyTorch Geometric `Data`.
* Output: learned graph embedding + pIC50 prediction.
* Course connection: Day03 GNN / molecular graphs.
* Implemented with GCN and GIN under the same M3 scaffold split:
  `src/models/graph_baseline.py`,
  `notebooks/04_molecular_graph_baseline.ipynb`.
* Outputs: `results/m4_graph_baseline_results.json`,
  `results/figures/m4_graph_baseline_scaffold_comparison.png`.

## M5 - SMILES sequence model

* Objective: tokenize SMILES and train a Transformer encoder.
* Output: learned sequence embedding + pIC50 prediction.
* Course connection: Day04 NLP / Transformer.
* Implemented as a small SMILES Transformer encoder under the same M3 scaffold
  split: `src/models/smiles_transformer.py`,
  `notebooks/05_smiles_transformer.ipynb`.
* Outputs: `results/m5_smiles_transformer_results.json`,
  `results/figures/m5_smiles_transformer_scaffold_comparison.png`.

## M6 - Virtual screening

* Objective: ensemble Morgan + GNN + Transformer predictions.
* Output: shortlist of candidate EGFR inhibitors.
* Course connection: Day02 virtual screening concept.
* Implemented as an ensemble of Morgan / graph / Transformer predictions:
  `src/models/virtual_screening.py`,
  `notebooks/06_virtual_screening.ipynb`.
* Candidate library: `data/candidates/egfr_candidate_library.csv`.
* Outputs: `results/m6_virtual_screening_results.json`,
  `results/m6_virtual_screening_shortlist.csv`,
  `results/figures/m6_virtual_screening_shortlist.png`,
  trained models under `results/models/`.
* Result checkpoint: external IC50 validation of the candidate ranking
  (`src/models/external_candidate_validation.py`,
  `results/m6_external_validation.json`,
  `../ANALYSIS_M2_M6.md`).

## M6.5 - Model improvement loop

* Objective: fix the external validation set and make the screening ranking
  trustworthy before M7.
* Approach: candidate-like internal validation, RF / XGB grid, calibration,
  positive controls, Morgan-only final ranking.
* Outputs: `src/models/m65_model_improvement.py`,
  `results/m65_model_improvement.json`,
  `results/m65_improved_shortlist.csv`,
  `results/figures/m65_external_validation.png`,
  `docs/README_M65.md`.
* Gate result: external Spearman +0.233 (was -0.26); all four local positive
  controls also rank 1-4, which is a necessary but not sufficient sanity
  check because the controls are far more active than the candidate library.
* Fair comparison: all five models retrained under the same candidate-like
  split (`src/models/m65_fair_comparison.py`,
  `results/m65_fair_comparison.json`,
  `results/m65_fair_comparison_predictions.csv`,
  `results/figures/m65_fair_comparison.png`).
* Final screening ensemble: RF + Transformer weighted mean
  (`src/models/m65_rf_transformer_ensemble.py`,
  `results/m65_rf_transformer_ensemble.json`,
  `results/m65_rf_transformer_ensemble_predictions.csv`,
  `results/figures/m65_rf_transformer_ensemble.png`); external Spearman
  +0.286 (equal weight) and controls ranked 1-4.

## M7 - ADMET + docking

* Objective: filter candidates by ADMET and structural docking. **Complete.**
* ADMET: RDKit Lipinski / Veber / ESOL / QED descriptors
  (`src/models/m7_admet.py`, `results/m7_admet_descriptors.csv`,
  `results/m7_admet_summary.json`); 52/64 molecules pass the combined rule.
* Docking: AutoDock Vina 1.2.7 against EGFR 1M17 with the erlotinib pocket
  (`src/models/m7_docking.py`, `results/m7_docking_results.csv`,
  `results/m7_docking_summary.json`, `data/docking/`);
  20/20 shortlist molecules docked successfully.
* Final ranking: 0.5*z(pIC50) + 0.3*z(-affinity) + 0.2*ADMET_ok
  (`src/models/m7_ranking.py`, `results/m7_final_ranking.csv`,
  `results/m7_final_ranking.json`, `results/figures/m7_final_ranking.png`).
* Sanity check: the four approved EGFR inhibitor controls occupy ranks 1-4.
* Course connection: Day02 ADME, Day06 docking / protein-ligand.

## M8 - Candidate ranking + final analysis

* Objective: rank candidates with activity + ADMET + docking + uncertainty.
  **Complete.**
* Final analysis: `src/models/m8_final_report.py`,
  `results/m8_final_analysis.json`,
  `results/m8_final_candidates.csv`,
  `results/figures/m8_model_agreement.png`,
  `results/figures/m8_external_error.png`,
  `results/figures/m8_final_shortlist.png`,
  `docs/README_M8.md`.
* Key result: equal-weight RF + Transformer ensemble reaches external
  Spearman +0.286 (p=0.028), interpreted as an exploratory post-selection
  optimum; four approved EGFR inhibitor controls rank 1-4 as a
  necessary-but-not-sufficient sanity check.

## M9 - Evidence milestones (screening benchmark + pose validation)

* Objective: turn the project from "models trained and compared" into "does
  the pipeline actually enrich actives and does the docking geometry make
  sense?". **Complete.**
* P0-1 retrospective screening benchmark:
  `src/models/m9_retrospective_benchmark.py`,
  `results/m9_retrospective_benchmark.json`,
  `results/m9_retrospective_predictions.csv`,
  `results/figures/m9_retrospective_roc.png`.
  Binary actives/inactives (pIC50 >= 7.0 / < 6.0) on the same M3 scaffold
  split; RandomForest reaches ROC-AUC 0.885, PR-AUC 0.918, EF1% 1.713,
  Spearman 0.652.
* P0-2 docking pose validation:
  `src/models/m9_docking_pose_validation.py`,
  `results/m9_pose_validation.json`,
  `results/m9_pose_validation.csv`,
  `results/figures/m9_pose_validation.png`.
  Redocked the erlotinib co-crystal ligand (AQ4) in EGFR 1M17; 6/9 Vina
  models have heavy-atom RMSD < 2.0 A and the best sampled pose is 1.238 A.
* Report: `docs/README_M9.md`.

## P0-3 - External label audit + multi-seed reproducibility

* Objective: fix the two remaining weaknesses of the external validation
  claim - crude, most-potent-selected labels and a single seed. **Complete.**
* Label audit: `src/models/p03_external_label_audit.py`,
  `results/p03_external_label_audit.json`,
  `results/p03_external_clean_labels.csv`.
  61 records / 59 molecules, all exact IC50; clean rule is the median pIC50
  across records (duplicates and ambiguous 10^3 uM flagged). Only 2 labels
  change, and ensemble Spearman stays essentially the same (0.286 -> 0.291).
* Multi-seed: `src/models/p03_multiseed_reproducibility.py`,
  `results/p03_multiseed_reproducibility.json`,
  `results/p03_multiseed_per_seed.csv`,
  `results/figures/p03_multiseed_external_rho.png`.
  Five seeds retrained on the clean labels. RandomForest is the only
  consistently positive model (mean external Spearman +0.206, 5/5 positive);
  the equal-weight RF + Transformer ensemble is seed-sensitive (mean +0.024,
  3/5 positive), so RF-only (or multi-seed RF) is recommended as the final
  screening model.
* Report: `docs/README_P03.md`.

## Future optional P0 steps (not yet started)

* P0-4: stacking and calibrated uncertainty for the final ensemble.
* P0-5: interpretability, e.g. Met793 hinge hydrogen bonds in the docked
  poses.

---

## Guiding principles

* Scientific correctness before software sophistication.
* Every major transformation must be auditable.
* Course materials are read-only reference; code is written fresh in this
  project and differences are documented in `README.md`.
