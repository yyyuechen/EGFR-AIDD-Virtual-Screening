# GitHub Pre-Publication Audit

Audit date: 2026-08-21. This is a read-only audit; no files were deleted,
moved, or modified except this report.

## 1. Executive Summary

The repository is a compact, well-structured educational EGFR AIDD project
(274 MB total, 281 files). It contains no secrets, no personal absolute
paths, no private contact information, and no broken markdown links. All 17
checked README numerical claims match the canonical result files. The only
oversized artifact is a 241 MB RandomForest joblib under `results/models/`,
which is already covered by `.gitignore`.

Overall status: **YES AFTER FIXES**. The fixes are mostly hygiene:
document the platform-specific Vina binary, add data attribution/license
notes, declare `requests`, add a few `.gitignore` patterns, and remove
`.DS_Store` before the first commit.

## 2. Critical Issues Before Publication

1. **Platform-specific Vina binary** (`data/docking/vina`, Mach-O arm64):
   must not be committed to a public cross-platform repository. Document
   how to install AutoDock Vina 1.2.7 or commit a small download script.
2. **ChEMBL data redistribution**: raw and interim ChEMBL-derived files are
   already ignored. The compact processed dataset is derived from ChEMBL;
   add an explicit ChEMBL attribution/licensing note (ChEMBL data is
   typically CC BY-SA; verify before publishing).
3. **No LICENSE / CITATION / CHANGELOG**: optional but recommended for a
   public portfolio repository.
4. **`requests` is imported directly** in `src/data/download_chembl.py` but
   is not explicitly declared in `environment.yml`.
5. **`.DS_Store` exists at the repository root** and must be removed before
   the first commit (not by gitignore alone).
6. No `.git` repository exists; `git init` was not performed per audit
   instructions.

## 3. Secrets / Privacy

Scanned for API keys, tokens, passwords, bearer credentials, private keys,
SSH keys, emails, and phone-like patterns.

| finding | severity | action |
|---|---|---|
| None | - | No secrets, credentials, personal emails, phones, or private keys found. Matches for "secret/token" were SMILES tokenizer text and "secretion" biology terms only. |

No `.env` files exist.

## 4. Large Files

| path | size | recommendation |
|---|---:|---|
| `results/models/rf_egfr.joblib` | 241 MB | IGNORE (regenerate via `src/models/morgan_baseline.py` or pipeline scripts); already covered by `.gitignore` |

No other file exceeds 10 MB. All remaining files are below 2 MB.

## 5. Data Publication

| path | size | source | role | recommendation |
|---|---:|---|---|---|
| `data/raw/*` | 5.2 MB | ChEMBL downloads | raw third-party source | IGNORE / REGENERATE VIA SCRIPT |
| `data/interim/*` | 12 MB | derived preprocessing | intermediate | IGNORE |
| `data/processed/egfr_activity_final.csv` | 1.7 MB | derived from ChEMBL | compact final dataset | KEEP with ChEMBL attribution note |
| `data/candidates/egfr_candidate_library.csv` | 8 KB | derived | screening candidates | KEEP |
| `data/docking/1M17.pdb`, `4TKS.pdb`, `4WRG.pdb`, `vina_box.json`, `ligands/*.pdbqt` | ~2.4 MB | public PDB / derived | docking inputs | KEEP |
| `data/docking/vina` | 1.1 MB | platform binary | AutoDock Vina executable | DO NOT COMMIT (Mach-O arm64; manual review) |

No legal claims are made here; verify ChEMBL and PDB terms before release.

## 6. Portability

| file | issue | severity | suggested fix |
|---|---|---|---|
| `data/docking/vina` | Mach-O arm64 executable | HIGH | Do not commit; document installation or add download script |
| `src/models/graph_baseline.py`, `smiles_transformer.py`, `virtual_screening.py` | macOS `KMP_DUPLICATE_LIB_OK` env workaround | LOW | Harmless; keep, or guard with `platform.system()` |
| `environment.yml` | `requests` not declared | LOW/MEDIUM | Add `requests` to dependencies |
| `src/models/m7_docking.py` | Vina path CLI default `data/docking/vina` | LOW | Keep; document binary requirement |
| `src/models/m9_docking_pose_validation.py` | obabel resolved via PATH | LOW | Already portable; keep |

No hardcoded absolute machine paths were found.

## 7. README Consistency

All 17 checked claims PASS:

* dataset N = 10,161;
* tuned final RF: R2 0.5918, Spearman 0.7663, ROC-AUC 0.8818, PR-AUC 0.8906;
* baseline RF: R2 0.5728, Spearman 0.7587;
* repeated scaffold split: dSpearman +0.0061 +/- 0.0039, dR2 +0.0127 +/-
  0.0066, 5/5 improvement;
* Level 3 recovery: 4/4 in top 5%, pool N=475, all max sim < 0.60;
* docking pose validation: best RMSD 1.238 A, 6/9 below 2 A;
* final candidate tiers: CHEMBL13983 Tier 1, CHEMBL5564149 Tier 2.

No stale or conflicting headline claims were found.

## 8. Broken Links

Checked all relative markdown links in README and 32 docs files:

**0 broken links.**

## 9. Results Canonicalization

### A. FINAL PUBLIC RESULTS (recommended to commit)

```text
README.md
environment.yml
environment-docking.yml
.gitignore
src/
notebooks/
data/processed/
data/candidates/
data/docking/*.pdb, vina_box.json, ligands/
docs/final_rf_model_selection.md
docs/repeated_scaffold_split_robustness.md
docs/withheld_known_inhibitor_recovery_level{1,2,3}.md
docs/level3_low_similarity_feasibility.md
docs/final_candidate_prioritization.md
docs/FINAL_PROJECT_SUMMARY.md
results/final_rf_configuration.json
results/final_rf_test_results.json
results/final_rf_test_predictions.csv
results/repeated_scaffold_split_results.csv
results/repeated_scaffold_split_summary.json
results/withheld_known_inhibitor_*.csv|.json
results/scaffold_withheld_known_inhibitor_*.csv|.json
results/low_similarity_withheld_known_inhibitor_rankings.csv
results/withheld_known_inhibitor_level3_summary.json
results/level1_level2_level3_*.csv
results/final_known_inhibitor_ranking.csv
results/final_candidate_*.csv|.json
results/final_candidate_stability_25runs.csv
results/figures/final_*.png
results/figures/level1_level2_level3_*.png
results/figures/repeated_scaffold_*.png
results/figures/final_workflow.png
results/figures/final_key_results.png
results/m9_pose_validation.json
results/m9_pose_validation.png
results/m7_docking_results.csv
results/m7_docking_summary.json
```

### B. SCIENTIFIC SUPPORT / ABLATION RESULTS (optional commit)

```text
results/radius_tuning_*.csv|.json
results/nbits_tuning_*.csv|.json
results/min_samples_leaf_tuning_*.csv|.json
results/max_features_*.csv|.json
results/max_depth_tuning_*.csv|.json
results/tanimoto_one_audit.csv|.json
results/m65_*.csv|.json
results/m6_*.csv|.json
results/m7_admet_descriptors.csv, m7_final_ranking.*
results/m8_final_*.csv|.json
results/m9_retrospective_*.csv|.json
results/p03_*.csv|.json
results/applicability_domain_predictions.csv
results/scaffold_split_comparison.json
results/m2_morgan_baseline_results.json
results/m3_morgan_split_comparison_results.json
results/m4_graph_baseline_results.json
results/m5_smiles_transformer_results.json
results/data_processing_summary.json
docs/README_M*.md
docs/radius_tuning_analysis.md
docs/nbits_tuning_analysis.md
docs/min_samples_leaf_tuning_analysis.md
docs/max_features_*.md
docs/max_depth_tuning_analysis.md
docs/tanimoto_one_audit.md
docs/portfolio_hardening_analysis.md
docs/README_P03.md
```

### C. REDUNDANT / DEV-HISTORY (review or archive)

```text
docs/archive/HANDOFF_M1_TO_M2.md
docs/archive/ANALYSIS_M2_M6.md
docs/archive/project_roadmap.md
docs/archive/development_history.md
```

These are historically informative but not needed for the public
canonical story; consider moving under `docs/archive/` or keeping with a
"development history" label.

### D. LARGE MODEL ARTIFACTS (do not commit)

```text
results/models/  (rf_egfr.joblib 241 MB, gcn_egfr.pt, gin_egfr.pt,
                  transformer_egfr.pt, xgb_egfr.joblib, smiles_vocab.json)
```

### E. DOCKING OUTPUTS (small; optional commit)

```text
results/m7_docking_out/*.pdbqt  (628 KB total, used by M9)
```

## 10. .gitignore Review

Current rules correctly ignore Python caches, virtual environments,
Jupyter checkpoints, `data/raw/*`, `data/interim/*`, `results/models/`,
logs/tmp files, and `.DS_Store`.

Recommended additions:

```text
.env
.env.*
.pytest_cache/
.mypy_cache/
*.egg-info/
.idea/
.vscode/
data/docking/vina
```

Do not ignore scientific CSV/JSON result files. Remove the existing
`.DS_Store` file before the first commit.

## 11. COMMIT List

Group A from Section 9: README, environment files, `.gitignore`, `src/`,
`notebooks/`, `data/processed/`, `data/candidates/`, docking inputs (except
Vina binary), selected docs, selected final/support results, and
`results/figures/`.

## 12. DO NOT COMMIT List

```text
results/models/
data/raw/*
data/interim/*
data/docking/vina
__pycache__/ and *.py[cod]
.DS_Store
.env / .env.*
*.log, *.tmp
.ipynb_checkpoints/
```

## 13. MANUAL REVIEW List

```text
data/processed/egfr_activity_final.csv      -> ChEMBL attribution/license
data/docking/vina                           -> platform binary / licensing
docs/archive/HANDOFF_M1_TO_M2.md           -> archive or keep as history
docs/archive/ANALYSIS_M2_M6.md             -> archive or keep as history
docs/archive/project_roadmap.md            -> may be outdated
docs/archive/development_history.md        -> may be outdated
results/m7_docking_out/*.pdbqt             -> keep or ignore (small)
results/m6_virtual_screening_*.csv|.json   -> superseded by M6.5/final shortlist
results/applicability_domain_predictions.csv -> derived, optional
```

## 14. Suggested Public Repository Structure

```text
egfr-aidd-virtual-screening/
├── README.md
├── LICENSE                       (add)
├── CITATION.cff                  (optional, add)
├── environment.yml
├── environment-docking.yml
├── .gitignore
├── data/
│   ├── processed/
│   ├── candidates/
│   └── docking/                  (PDB, box; Vina binary excluded)
├── src/
│   ├── data/
│   └── models/
├── notebooks/
├── docs/
│   ├── final_rf_model_selection.md
│   ├── repeated_scaffold_split_robustness.md
│   ├── withheld_known_inhibitor_recovery_level{1,2,3}.md
│   └── final_candidate_prioritization.md
└── results/
    ├── figures/
    ├── final_*.json|.csv
    ├── repeated_scaffold_split_*.csv|.json
    └── robustness_*.csv|.json
```

## 15. Optional Improvements

* Add `LICENSE` (e.g., MIT for code) and `CITATION.cff`.
* Add a compact `QUICKSTART` section with two commands:
  `python src/models/final_rf_confirmation.py` (final RF result) and
  `python src/models/final_candidate_prioritization.py` (candidate
  shortlist).
* Add a ChEMBL/PDB attribution note in `data/processed/README` or
  `data/README.md`.
* Add `requests` to `environment.yml`.
* Consider a repository badge (e.g., license) after publication.
