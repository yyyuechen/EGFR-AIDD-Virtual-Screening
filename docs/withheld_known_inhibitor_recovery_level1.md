# Level 1: Exact-Molecule Withholding Recovery Benchmark

## Purpose

This benchmark asks whether known EGFR inhibitors that are completely
withheld as exact molecules from training are still preferentially ranked
near the top of a larger retrospective ranking pool.

Level 1 removes only exact molecules. Same-scaffold analogs and similar
molecules intentionally remain in training, so this is not a
scaffold-generalization benchmark.

## Frozen model

The locked final Morgan-RF configuration is used unchanged:

```text
Morgan radius = 2, nBits = 2048
RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1,
                      min_samples_leaf=1, max_features=0.20,
                      max_depth=None)
```

No parameter was changed based on recovery results.

## Positive set

Only four high-confidence named inhibitors could be identified from existing
repository evidence (M6.5/M7/M8 controls and the curated ChEMBL-linked
dataset):

| compound | ChEMBL ID | curated pIC50 | measurements |
|---|---|---:|---:|
| afatinib | CHEMBL1173655 | 8.979 | 106 |
| erlotinib | CHEMBL553 | 7.523 | 197 |
| gefitinib | CHEMBL939 | 7.638 | 228 |
| lapatinib | CHEMBL554 | 7.967 | 82 |

Limitation: the local data files contain no compound-name metadata and
osimertinib is documented as absent from the local ChEMBL export. No
additional labels were fabricated; this is the strongest defensible set.

## Withholding

* Original standard-scaffold train: 8,128 molecules
* Withheld positive structures: 4
* Final training set: 8,124 molecules
* Verified: `exact_training_match = False` for every positive after
  withholding (canonical SMILES check)

## Ranking pool

* Total: 475 molecules
* Positives: 4 (all withheld from training)
* Background: 471 held-out molecules (standard scaffold validation+test,
  pIC50 < 6)
* Positive prevalence: 0.84%
* The pIC50 6-7 middle band is excluded
* The pool consists entirely of molecules not used for model fitting

## Positive-level results

| compound | exp pIC50 | pred pIC50 | rank | percentile | top 1% | top 5% | top 10% | max sim | same scaffold in train | scaffold count |
|---|---:|---:|---:|---:|---|---|---|---:|---|---:|
| afatinib | 8.979 | 8.292 | 1 | 0.21% | yes | yes | yes | 1.0000 | yes | 28 |
| erlotinib | 7.523 | 7.976 | 3 | 0.63% | yes | yes | yes | 0.9038 | yes | 559 |
| gefitinib | 7.638 | 7.776 | 7 | 1.47% | no | yes | yes | 0.9194 | yes | 17 |
| lapatinib | 7.967 | 7.677 | 13 | 2.74% | no | yes | yes | 0.8734 | yes | 37 |

## Recovery and enrichment

| threshold | recovered | recall | EF | theoretical max EF |
|---:|---:|---:|---:|---:|
| top 1% | 2/4 | 0.50 | 47.5 | 100 |
| top 5% | 4/4 | 1.00 | 19.8 | 20 |
| top 10% | 4/4 | 1.00 | 9.9 | 10 |

Mean positive rank 6.0; median positive rank 5.0; median positive percentile
1.05%.

## Global ranking metrics

* ROC-AUC = 0.9926
* PR-AUC = 0.6007 (baseline prevalence 0.0084)
* Spearman(pred, experimental pIC50) = 0.4131

Caveat: the pool is threshold-constructed (positives >= 7 vs background < 6),
so the quantitative Spearman partly reflects that construction and should
not be compared with full-dataset regression Spearman.

## Applicability and same-scaffold diagnostics

All four withheld positives have `max_train_tanimoto >= 0.8`:

* >=0.8: 4/4
* 0.6-<0.8: 0
* 0.4-<0.6: 0
* <0.4: 0

All four positives have their Murcko scaffold present in training, with
17-559 training molecules sharing the scaffold (mean 160.25). Recovery is
therefore concentrated in a chemically easy regime: exact molecules are
hidden, but close analogs and the same scaffolds remain available.

## Comparison with the historical M6.5 control ranks

| compound | historical M6.5 rank | new withheld-model rank | max sim |
|---|---|---:|---:|
| afatinib | 1 | 1 | 1.0000 |
| erlotinib | 2 | 3 | 0.9038 |
| gefitinib | 3 | 7 | 0.9194 |
| lapatinib | 4 | 13 | 0.8734 |

The historical result came from a different model (M6.5 RF trained on a
candidate-like internal split with controls removed from the training pool)
and a different 64-molecule candidate set, so the two are not directly
comparable.

## Interpretation

**B. Moderate recovery (strong enrichment in a training-similar regime).**
The withheld model recovers 4/4 known inhibitors within the top 5% and
EF5% is near its theoretical maximum (19.8 of 20). However, every recovered
positive has max similarity >= 0.8 and its same scaffold present in
training, so the strong result reflects near-training chemistry rather than
novel-scaffold generalization.

## Is Level 2 justified?

Yes. Level 1 cannot distinguish "ranked highly because close chemistry is
known" from "ranked highly for genuinely new scaffolds". A Level 2
scaffold-withheld evaluation is scientifically necessary before any
novel-scaffold claim can be made.

## Files

```text
src/models/withheld_known_inhibitor_recovery_level1.py
results/withheld_known_inhibitor_positive_set.csv
results/withheld_known_inhibitor_ranking_pool.csv
results/withheld_known_inhibitor_rankings.csv
results/withheld_known_inhibitor_recovery_summary.json
results/figures/withheld_known_inhibitor_rank_distribution.png
results/figures/withheld_known_inhibitor_recovery.png
```

## Quality control

Verified: no withheld positive exact molecule remains in training; no
duplicate positive structures; no duplicate ranking-pool structures; frozen
RF configuration used unchanged; enrichment formulas use the standard EF
definition; pool prevalence and theoretical maximum EF are reported; no
parameter was selected from recovery metrics.
