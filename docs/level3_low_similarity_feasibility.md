# Level 3 Feasibility Audit (Phase 3A)

## Purpose

Before retraining any model, this audit measures how much of the original
standard-scaffold training set would be removed by the Level 3 filter:

remove every training molecule with Morgan Tanimoto >= 0.60 to any of the
four known positives (radius = 2, nBits = 2048), plus exact molecules and
same-Murcko-scaffold molecules.

No model was trained during this audit, and the threshold was not changed
after seeing the removal count.

## Per-positive removal counts

| compound | n_sim >= 0.60 | n_same_scaffold | scaffold overlap with sim set | n_exact |
|---|---:|---:|---:|---:|
| afatinib | 84 | 29 | 26 | 1 |
| erlotinib | 40 | 560 | 23 | 1 |
| gefitinib | 114 | 18 | 9 | 1 |
| lapatinib | 65 | 38 | 38 | 1 |

## Union statistics

* Original training N = 8,128
* Unique molecules removed by Tanimoto >= 0.60 union = 302
* Unique molecules removed by scaffold union = 645
* Total unique molecules removed (joint) = 810
* Removed fraction = 9.97%
* Final training N = 7,318

## Training distribution before and after

| statistic | before | after |
|---|---:|---:|
| pIC50 mean | 6.930 | 6.878 |
| pIC50 std | 1.356 | 1.372 |
| pIC50 median | 7.041 | 6.984 |
| active fraction (>=7) | 0.515 | 0.497 |

The removal modestly lowers mean activity and active fraction; the
distribution remains usable for regression training.

## Scaffold diversity before and after

| statistic | before | after |
|---|---:|---:|
| n_scaffolds | 2,858 | 2,785 |
| mean molecules/scaffold | 2.84 | 2.63 |
| median molecules/scaffold | 1 | 1 |
| max scaffold size | 560 | 125 |
| singleton scaffolds | 1,879 | 1,845 |

The maximum scaffold group shrinks from 560 to 125, but the dataset keeps
2,785 scaffolds and 7,318 molecules, so the remaining set is scientifically
usable.

## Conclusion

The remaining training set is usable. Level 3 should be described as a
severe but partial distribution-shift stress test: 9.97% of training data is
removed, activity distribution shifts only modestly, and scaffold diversity
remains high.
