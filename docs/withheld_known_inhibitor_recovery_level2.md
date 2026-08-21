# Level 2: Scaffold-Withheld Known-Inhibitor Recovery

## Purpose

Level 1 removed only the exact molecules of four known EGFR inhibitors and
showed strong recovery, but all positives still had `max_train_tanimoto >=
0.8` and their Murcko scaffolds remained in training. Level 2 removes both
the exact positive molecules and every training molecule sharing each
positive's Bemis-Murcko scaffold, then repeats the same frozen-model
ranking experiment on the identical 475-molecule pool.

## Frozen model

Unchanged locked configuration:

```text
Morgan radius = 2, nBits = 2048
RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1,
                      min_samples_leaf=1, max_features=0.20,
                      max_depth=None)
```

## Training removal (joint scaffold withholding)

| compound | positive Murcko scaffold | training molecules removed |
|---|---|---:|
| afatinib | `c1ccc(Nc2ncnc3cc(OC4CCOC4)ccc23)cc1` | 29 |
| erlotinib | `c1ccc(Nc2ncnc3ccccc23)cc1` | 560 |
| gefitinib | `c1ccc(Nc2ncnc3ccc(OCCCN4CCOCC4)cc23)cc1` | 18 |
| lapatinib | `c1ccc(COc2ccc(Nc3ncnc4ccc(-c5ccco5)cc34)cc2)cc1` | 38 |

* Original training N = 8,128
* Total unique molecules removed = 645
* Overlap between removed scaffold sets = 0
* Final training N = 7,483
* QC: `exact_training_match = False` and
  `same_scaffold_present_in_training = False` for all four positives

## Closest remaining training molecule per positive

| positive | Level 1 max sim | Level 2 max sim | closest remaining similarity |
|---|---:|---:|---:|
| afatinib | 1.0000 | 0.8049 | 0.8049 |
| erlotinib | 0.9038 | 0.7121 | 0.7121 |
| gefitinib | 0.9194 | 0.9194 | 0.9194 |
| lapatinib | 0.8734 | 0.8395 | 0.8395 |

Gefitinib's closest remaining molecule is a very close analog with a
one-carbon-shorter linker (`OCCN` vs `OCCCN`), which produces a different
Murcko scaffold but almost the same Morgan fingerprint.

## Positive-level Level 1 vs Level 2

| compound | exp pIC50 | L1 pred | L1 rank | L2 pred | L2 rank | rank change | pred change |
|---|---:|---:|---:|---:|---:|---:|---:|
| afatinib | 8.979 | 8.292 | 1 | 8.397 | 1 | 0 | +0.104 |
| erlotinib | 7.523 | 7.976 | 3 | 7.778 | 5 | +2 | -0.197 |
| gefitinib | 7.638 | 7.776 | 7 | 7.684 | 8 | +1 | -0.092 |
| lapatinib | 7.967 | 7.677 | 13 | 7.621 | 13 | 0 | -0.056 |

## Level 2 recovery metrics

| threshold | recovered | recall | EF | theoretical max EF |
|---:|---:|---:|---:|---:|
| top 1% | 2/4 | 0.50 | 47.5 | 100 |
| top 5% | 4/4 | 1.00 | 19.8 | 20 |
| top 10% | 4/4 | 1.00 | 9.9 | 10 |

ROC-AUC = 0.9910; PR-AUC = 0.5207 (pool prevalence 0.84%);
Spearman(pred, experimental pIC50) = 0.4192 (threshold-constructed pool).

## Level 1 vs Level 2 comparison

| metric | Level 1 | Level 2 | change |
|---|---:|---:|---:|
| Recall@1% | 0.50 | 0.50 | 0.00 |
| Recall@5% | 1.00 | 1.00 | 0.00 |
| Recall@10% | 1.00 | 1.00 | 0.00 |
| EF1% | 47.5 | 47.5 | 0.00 |
| EF5% | 19.8 | 19.8 | 0.00 |
| EF10% | 9.9 | 9.9 | 0.00 |
| ROC-AUC | 0.9926 | 0.9910 | -0.0016 |
| PR-AUC | 0.6007 | 0.5207 | -0.0801 |
| median positive rank | 5.0 | 6.5 | +1.5 |
| mean positive rank | 6.0 | 6.75 | +0.75 |
| median positive max sim | 0.912 | 0.822 | -0.089 |

## Applicability domain after scaffold withholding

* >=0.8: 3/4 positives (afatinib, gefitinib, lapatinib)
* 0.6-<0.8: 1/4 (erlotinib, 0.712)
* 0.4-<0.6: 0
* <0.4: 0

Scaffold removal lowered similarities, but no positive dropped below 0.6 and
three of four remained in the high-similarity bin. The benchmark still does
not test genuinely low-similarity chemistry.

## Individual-compound interpretation

* Afatinib: unchanged rank 1; predicted activity even rose slightly (+0.104)
  after its 29 scaffold analogs were removed.
* Erlotinib: most affected (prediction -0.197, rank 3 -> 5, max sim
  0.904 -> 0.712), yet it remains inside the top 1.1% of the pool.
* Gefitinib: rank 7 -> 8 despite its closest remaining analog having
  essentially identical chemistry (max sim 0.919, different scaffold via a
  one-carbon linker change); its recovery is not evidence of low-similarity
  generalization.
* Lapatinib: rank unchanged at 13; prediction fell only -0.056.

## Interpretation

**A. Strong scaffold-withheld recovery at the pool level.** All four
positives remain in the top 13 of 475 (top 2.7%) and Recall@5% stays 4/4
with EF5% = 19.8/20 after same-scaffold analogs are removed.

Important caveats: positive N is only 4; recovery is retrospective; the pool
is threshold-constructed; known inhibitors are well-characterized drugs, not
arbitrary novel chemistry; and most positives still sit at max similarity
>= 0.8 (erlotinib 0.71). Scaffold withholding is stronger than exact
withholding but still does not guarantee low structural similarity.

## Is Level 3 justified?

Yes. Even after scaffold withholding, no positive has max similarity < 0.6,
so the current benchmark cannot support a claim of low-similarity or
novel-chemotype generalization. A Level 3 analysis that additionally
requires low Tanimoto similarity would be the next meaningful test.

## Files

```text
src/models/scaffold_withheld_known_inhibitor_recovery_level2.py
results/scaffold_withheld_known_inhibitor_training_removals.csv
results/scaffold_withheld_known_inhibitor_rankings.csv
results/scaffold_withheld_known_inhibitor_summary.json
results/level1_vs_level2_known_inhibitor_comparison.csv
results/level1_vs_level2_known_inhibitor_metrics.csv
results/figures/level1_vs_level2_known_inhibitor_ranks.png
results/figures/level1_vs_level2_recovery.png
```

## Quality control

Verified: exact same 4 positives as Level 1; exact same 475 ranking
molecules; no positive exact molecule remains in training; no positive
Murcko scaffold remains in training; frozen RF configuration unchanged; no
tuning performed; Level 1 files were not overwritten.
