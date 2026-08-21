# Development History

This document preserves the milestone-oriented history of the project. The
main README is now organized around the scientific workflow; the milestones
below are kept as an audit trail. Each milestone has a dedicated report in
`docs/README_M*.md`.

## Milestones

### M1 - EGFR activity dataset construction

Downloaded ChEMBL EGFR IC50 data, normalized units to nM, validated and
standardized SMILES with RDKit, aggregated repeated measurements by median
pIC50, and produced a 10,161-molecule dataset.
Report: `docs/README_M1.md` content is summarized in the main README data
curation section and `results/data_processing_summary.json`.

### M2 - Morgan fingerprint baseline

RandomForest and XGBoost on Morgan fingerprints (radius=2, 2048 bits) under
a random split.
Report: `docs/README_M2.md`, results: `results/m2_morgan_baseline_results.json`.

### M3 - Random vs scaffold split

Same Morgan models under a random split and a Murcko scaffold split. The
scaffold split keeps complete scaffolds together and produces an extreme
unseen-scaffold test set.
Report: `docs/README_M3.md`, results: `results/m3_morgan_split_comparison_results.json`.

### M4 - Molecular graph models

GCN and GIN trained on the same scaffold split with best-epoch selection.
Report: `docs/README_M4.md`, results: `results/m4_graph_baseline_results.json`.

### M5 - SMILES Transformer

A small SMILES Transformer encoder trained on the same scaffold split with
best-epoch selection.
Report: `docs/README_M5.md`, results: `results/m5_smiles_transformer_results.json`.

### M6 - Virtual screening

Ensemble virtual screening of the candidate library.
Report: `docs/README_M6.md`, results: `results/m6_virtual_screening_results.json`.

### M6.5 - Model improvement loop

External validation exposed a negative correlation; the loop fixed the
external set, tuned models on a candidate-like split, and improved ranking.
Reports: `../README_M65.md`, `../ANALYSIS_M2_M6.md`.

### M7 - ADMET + docking

ADMET descriptors and AutoDock Vina docking against EGFR 1M17; 20/20
molecules docked successfully.
Report: `docs/README_M7.md`.

### M8 - Final analysis and error report

Consolidated metrics, external history, model agreement, error analysis, and
the final candidate shortlist.
Report: `docs/README_M8.md`.

### M9 - Evidence milestones (P0-1 + P0-2)

P0-1: retrospective screening benchmark (ROC-AUC, PR-AUC, EF).
P0-2: docking pose validation by erlotinib redocking.
Report: `docs/README_M9.md`.

### P0-3 - External label audit + multi-seed reproducibility

Cleaned external labels (median aggregation with provenance flags) and
retrained the final models across five seeds. RandomForest was the only
consistently positive model.
Report: `docs/README_P03.md`.

### Portfolio hardening - standard scaffold split + applicability domain

Added a conventional 80/10/10 scaffold split and a Morgan Tanimoto
applicability-domain analysis.
Report: `docs/portfolio_hardening_analysis.md`.

## Notes

* Random seeds: 42 for splits and models; {42, 7, 123, 2024, 777} for the
  multi-seed audit.
* All results JSONs record package versions, model parameters, split
  indices, and relevant statistics.
* Negative findings are preserved: learned representations did not
  consistently beat Morgan fingerprints, external transfer is weak, and the
  equal-weight ensemble was not seed-reproducible.
