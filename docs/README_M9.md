# M9 - Retrospective screening benchmark and docking pose validation

## Why M9 exists

M1-M8 built the complete pipeline, but the milestone metrics only showed
"can the model rank a held-out test set". A screening project should answer
the stronger question: **does the pipeline actually enrich active EGFR
inhibitors when we pretend the test set is a candidate library?**

M9 adds two cheap, high-value evidence milestones:

```text
P0-1  Retrospective screening benchmark
      Binary active/inactive labels on the scaffold test set
      ROC-AUC, PR-AUC, enrichment factor, Spearman

P0-2  Docking pose validation
      Redock the co-crystal erlotinib (AQ4) with AutoDock Vina
      Heavy-atom RMSD against the crystal pose
```

Both are fully reproducible with the same split, the same models, and the
same receptor used earlier in the project.

---

## P0-1 Retrospective screening benchmark

### Protocol

* Split: the same M3 Murcko scaffold split (8,128 train / 2,033 test).
* Models are retrained on the scaffold **train** split only.
* The Transformer uses 10% of the training molecules as an internal
  validation subset for best-epoch selection; the test split is never seen
  during training or model selection.
* Binary labels on the scaffold test set:
  * active: pIC50 >= 7.0 (849 molecules)
  * inactive: pIC50 < 6.0 (605 molecules)
  * excluded middle band: 6.0 <= pIC50 < 7.0
* 1,454 test molecules enter the binary screening benchmark.

### Results

| Model | ROC-AUC | PR-AUC | EF1% | EF5% | Spearman (full test) |
|---|---:|---:|---:|---:|---:|
| RandomForest | 0.885 | 0.918 | 1.713 | 1.713 | 0.652 |
| XGBoost | 0.869 | 0.898 | 1.713 | 1.713 | 0.637 |
| Transformer | 0.793 | 0.832 | 1.713 | 1.594 | 0.472 |
| RF + Transformer ensemble | 0.873 | 0.914 | 1.713 | 1.713 | 0.626 |
| Random chance | 0.500 | 0.584 | 1.000 | 1.000 | 0.000 |

### How to read the table

* **ROC-AUC** measures ranking quality: 0.50 is random, 1.0 is perfect. A
  value above 0.8 on an extreme scaffold split is a strong result for a
  fingerprint model trained on only ~8,000 molecules.
* **PR-AUC** is the precision-recall curve area. Random chance is not 0.5
  here because the test set is enriched for actives (849/1454 = 58.4%
  active); the 0.918 PR-AUC means actives are recovered with far better
  precision than the base rate.
* **EF1% / EF5%** (enrichment factor) compare the active fraction in the top
  1% / 5% of ranked molecules with the active fraction in the whole test set.
  EF = 1.0 means no better than random; EF = 1.713 means the top 1% contains
  about 71% more actives than expected by chance.
* **Spearman (full test)** is the correlation between predicted and true
  pIC50 over all 2,033 scaffold test molecules, not just the binary subset.
  RandomForest reaches +0.652 on this deliberately hard split.

### Interpretation

The strongest and simplest model, RandomForest on Morgan fingerprints,
produces the best screening benchmark: ROC-AUC 0.885 and EF1% 1.713 on the
same scaffold split where regression R2 was only 0.40. This makes sense:
ranking is easier than exact pIC50 regression, and the binary active/inactive
question is closer to how virtual screening is actually used.

The equal-weight RF + Transformer ensemble is slightly below RandomForest
alone (ROC-AUC 0.873 vs 0.885). The ensemble was selected in M6.5 because it
generalized better on the external IC50 set; the retrospective benchmark is a
different stress test, and either model is defensible depending on the goal.

The Transformer alone is weaker (ROC-AUC 0.793) but still clearly better than
chance on unseen chemistry.

---

## P0-2 Docking pose validation

### Protocol

* Receptor: EGFR 1M17 (PDB).
* Crystal ligand: AQ4 (erlotinib).
* Docking: AutoDock Vina 1.2.7 with the same receptor file and box used in
  M7.
* The docked output is the M7 erlotinib result (`CHEMBL553_out.pdbqt`).
* Each of the 9 Vina models is compared with the crystal pose by heavy-atom
  RMSD.

### Results

| Check | Value |
|---|---:|
| Vina models scored | 9 |
| Models with RMSD < 2.0 A | 6 / 9 |
| Best affinity model | model 1, -7.014 kcal/mol, RMSD 2.494 A |
| Best RMSD model | model 6, RMSD 1.238 A, -6.783 kcal/mol |

### How to read it

* **RMSD** measures how close a docked pose is to the experimentally observed
  binding pose. Below 2.0 A is usually considered a successful redocking.
* 6 of 9 Vina models reproduce a near-native erlotinib pose, and the best
  sampled pose is 1.238 A from the crystal structure.
* The best-scoring pose (lowest Vina affinity) is not the most native-like
  pose (2.494 A), which is the standard AutoDock Vina behavior: the scoring
  function samples near-native poses but does not always rank them first.

### Interpretation

The docking setup is valid at the pose level: the pocket, receptor file, and
box are consistent with the known erlotinib binding mode. Affinity ranking
should still be treated as a screening approximation, but the geometry is
trustworthy enough to support the M7/M8 docking shortlist.

---

## Files

```text
src/models/m9_retrospective_benchmark.py
src/models/m9_docking_pose_validation.py

results/m9_retrospective_benchmark.json
results/m9_retrospective_predictions.csv
results/figures/m9_retrospective_roc.png

results/m9_pose_validation.json
results/m9_pose_validation.csv
results/figures/m9_pose_validation.png
```

## How to run

```bash
# Retrospective screening benchmark (project conda environment)
conda activate egfr-aidd
python -u src/models/m9_retrospective_benchmark.py

# Docking pose validation (needs openbabel via the docking environment)
conda activate egfr-aidd-dock
python src/models/m9_docking_pose_validation.py
```

## Caveats

* The scaffold test set is an extreme extrapolation benchmark: all 2,033 test
  molecules come from singleton scaffolds, and the test pIC50 distribution is
  shifted lower than training (mean 6.68 vs 7.01). The positive M9 numbers are
  therefore a strong but not "easy" result.
* The active/inactive threshold (7.0 / 6.0) is a project-defined educational
  choice.
* Enrichment is measured on the scaffold test set, not on a wet-lab
  confirmed library. It demonstrates retrospective screening power, not
  prospective experimental success.
* Only 20 molecules were docked in M7, so a docking-based enrichment curve
  would have too little power; pose validation is the appropriate docking
  sanity check at this scale.
* The receptor is rigid and the box is defined by the erlotinib co-crystal;
  this is standard for a pose-validation setup, not a full induced-fit study.
