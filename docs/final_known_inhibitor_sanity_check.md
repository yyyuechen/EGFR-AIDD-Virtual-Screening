# Final Known-Inhibitor Sanity Check

## Purpose

This is a frozen-model biological sanity check, not hyperparameter tuning.
The locked tuned Morgan-RF (`max_features = 0.20`) and the original baseline
(`max_features = 1.0`) are evaluated on the same 64-molecule candidate set
used by the earlier M6.5 known-inhibitor experiment.

## Candidate set

* Source: `results/m65_improved_shortlist.csv`
* Total: 64 molecules = 60 candidate-like + 4 positive controls
  (afatinib, erlotinib, gefitinib, lapatinib)
* The canonical-SMILES set is identical to
  `results/m65_fair_comparison_predictions.csv`

Both models were trained only on the canonical standard-scaffold train
split (8,128 molecules). Ranks are deterministic: higher predicted pIC50
first, then canonical SMILES.

## Known-inhibitor results

| compound | exact training match | max_train_tanimoto | baseline pred | baseline rank | tuned pred | tuned rank | rank change |
|---|---|---:|---:|---:|---:|---:|---:|
| afatinib | True | 1.0000 | 8.6006 | 1 | 8.6071 | 1 | 0 |
| lapatinib | True | 1.0000 | 7.8683 | 2 | 7.8828 | 2 | 0 |
| gefitinib | True | 1.0000 | 7.6811 | 3 | 7.6665 | 3 | 0 |
| erlotinib | True | 1.0000 | 7.6463 | 4 | 7.6549 | 4 | 0 |

Top-rank recovery: 4/4 in top 5, 4/4 in top 10, and 4/4 in top 20 for both
models.

## Top 10 under the tuned model

| rank | name / id | tuned pred | max_train_tanimoto | exact train | known inhibitor |
|---:|---|---:|---:|---|---|
| 1 | afatinib | 8.6071 | 1.0000 | True | True |
| 2 | lapatinib | 7.8828 | 1.0000 | True | True |
| 3 | gefitinib | 7.6665 | 1.0000 | True | True |
| 4 | erlotinib | 7.6549 | 1.0000 | True | True |
| 5 | CHEMBL2316916 | 6.6392 | 0.3043 | False | False |
| 6 | CHEMBL5614250 | 6.1574 | 0.4304 | False | False |
| 7 | CHEMBL5564149 | 6.1192 | 0.3095 | False | False |
| 8 | CHEMBL13983 | 5.9326 | 0.6389 | False | False |
| 9 | CHEMBL554993 | 5.9147 | 0.3088 | False | False |
| 10 | CHEMBL512054 | 5.9027 | 0.3333 | False | False |

## Applicability-domain interpretation

All four known inhibitors have `max_train_tanimoto = 1.0` and an exact
canonical-SMILES match in the training set. Their high ranks are therefore
high but strongly training-like (interpolation/memory), not novel-discovery
evidence.

The first non-control candidates (ranks 5-10) lie mostly in the lower
similarity region (0.30-0.64), so the model is still able to rank
candidate-like molecules, but their predicted pIC50 values are clearly below
the known inhibitors.

## Comparison with the earlier "top 1-4" claim

The earlier M6.5 experiment used the same 64-molecule candidate set, but a
different model:

* earlier model: M6.5 improved RandomForest trained on a candidate-like
  internal split, with the four controls removed from the training pool;
* current model: locked standard-split RandomForest, in which all four
  controls are exact training molecules.

The current result numerically reproduces ranks 1-4, but the two checks are
not equivalent: the M6.5 check tested unseen controls, while the current
check mostly demonstrates that the model remembers its training chemistry.

## Interpretation

**A. Ranking preserved.** The locked tuned model still places all four
known inhibitors at ranks 1-4 in the same candidate set.

Important caveat: the preserved ranking is weakened by exact training-set
overlap for all four inhibitors. This check supports biological plausibility
and internal consistency, but it is not independent validation of
prospective discovery. The earlier M6.5 result remains the only control
ranking generated from genuinely unseen inhibitor structures.

## Files

```text
src/models/final_known_inhibitor_check.py
results/final_known_inhibitor_ranking.csv
results/final_known_inhibitor_summary.json
results/figures/final_known_inhibitor_ranks.png
```

No hyperparameter was changed, no ranking result was used for model
selection, and the locked final model remains `max_features = 0.20`.
