# Level 3: Low-Similarity Withheld Known-Inhibitor Recovery

## Purpose

Level 1 removed exact molecules and Level 2 also removed same-scaffold
training molecules. Level 3 additionally removes every training molecule
with Morgan Tanimoto >= 0.60 to any of the four known positives, so each
positive ends with `max_train_tanimoto < 0.60`.

The frozen final Morgan-RF configuration is used unchanged:

```text
Morgan radius = 2, nBits = 2048
RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1,
                      min_samples_leaf=1, max_features=0.20,
                      max_depth=None)
```

## Training-set construction

* Original training N = 8,128
* Unique removed by Tanimoto >= 0.60 = 302
* Unique removed by scaffold union = 645
* Total unique removed = 810 (9.97%)
* Final training N = 7,318
* QC: exact match = 0, same scaffold = 0, all positive max sim < 0.60
  (0.5955-0.5978)

See `docs/level3_low_similarity_feasibility.md` for the Phase 3A audit.

## Positive-level Level 1 / 2 / 3 comparison

| compound | exp pIC50 | L1 pred / rank | L2 pred / rank | L3 pred / rank | L3-L1 rank | L3-L2 rank | L1 max sim | L2 max sim | L3 max sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| afatinib | 8.979 | 8.292 / 1 | 8.397 / 1 | 8.087 / 3 | +2 | +2 | 1.0000 | 0.8049 | 0.5978 |
| erlotinib | 7.523 | 7.976 / 3 | 7.778 / 5 | 7.734 / 10 | +7 | +5 | 0.9038 | 0.7121 | 0.5972 |
| gefitinib | 7.638 | 7.776 / 7 | 7.684 / 8 | 7.610 / 15 | +8 | +7 | 0.9194 | 0.9194 | 0.5976 |
| lapatinib | 7.967 | 7.677 / 13 | 7.621 / 13 | 7.576 / 19 | +6 | +6 | 0.8734 | 0.8395 | 0.5955 |

## Level 3 recovery metrics

| threshold | recovered | recall | EF | theoretical max EF |
|---:|---:|---:|---:|---:|
| top 1% | 1/4 | 0.25 | 23.75 | 100 |
| top 5% | 4/4 | 1.00 | 19.79 | 20 |
| top 10% | 4/4 | 1.00 | 9.90 | 10 |

ROC-AUC = 0.9804; PR-AUC = 0.2360 (prevalence 0.84%);
Spearman(pred, experimental pIC50) = 0.4329 (threshold-constructed pool).

## Level comparison

| metric | Level 1 | Level 2 | Level 3 |
|---|---:|---:|---:|
| Recall@1% | 0.50 | 0.50 | 0.25 |
| Recall@5% | 1.00 | 1.00 | 1.00 |
| Recall@10% | 1.00 | 1.00 | 1.00 |
| EF1% | 47.5 | 47.5 | 23.75 |
| EF5% | 19.8 | 19.8 | 19.8 |
| EF10% | 9.9 | 9.9 | 9.9 |
| ROC-AUC | 0.9926 | 0.9910 | 0.9804 |
| PR-AUC | 0.6007 | 0.5207 | 0.2360 |
| median positive rank | 5.0 | 6.5 | 12.5 |
| median positive max sim | 0.912 | 0.822 | 0.597 |

## Chemical-novelty verification

All four positives fall in the 0.5-<0.6 similarity bin (0.5955-0.5978).
None reaches <0.5, so Level 3 is a boundary low-similarity test, not a test
of genuinely distant chemistry. The closest remaining training molecules are
still active EGFR compounds (pIC50 7.27-8.31) with different Murcko
scaffolds.

Use the wording "low-similarity relative to this Morgan fingerprint
representation", not "novel chemistry".

## Individual-compound interpretation

* Afatinib: strongest low-similarity recovery, rank 3 (top 0.6%); its
  prediction dropped 0.31 but it remains the top-known-inhibitor.
* Erlotinib: rank 10; prediction changed little (-0.044) but rank fell from
  5 to 10.
* Gefitinib: rank 15; the largest relative decline (rank 7 -> 15), consistent
  with heavy dependence on near-identical quinazoline analogs.
* Lapatinib: rank 19; still within top 4%, but recovery is weakest in
  absolute rank.

## Interpretation

**B. Moderate low-similarity recovery.** Recovery remains strongly above
chance (4/4 in top 5%, EF5% at the theoretical maximum), but it declines
materially relative to Levels 1-2: top-1% recall falls to 1/4, PR-AUC drops
from 0.52 to 0.24, and the median positive rank rises from 6.5 to 12.5.

## What this does and does not show

* Analog-level recovery: clearly demonstrated (Level 1).
* Scaffold-transferable recovery: partially demonstrated (Levels 2-3), with
  different scaffolds but still active close chemistry in training.
* Low-similarity transferability: only at the 0.5-0.6 boundary; no positive
  reaches <0.5, so stronger novelty is not demonstrated.

## Limitations

* Positive N = 4.
* Retrospective benchmark.
* Negatives are threshold-constructed (pIC50 < 6).
* Morgan Tanimoto is representation-dependent.
* Training-set removal itself alters the learning distribution.
* Low similarity does not prove a novel mechanism or chemotype.
* This is not prospective inhibitor discovery.

## Motivation for a pretrained molecular model

Yes, Level 3 provides a legitimate motivation for a next research question:
as similarity to training chemistry decreases, recovery weakens. Whether a
pretrained molecular representation would preserve enrichment in the
0.4-<0.6 or <0.4 regime is an empirical question worth testing, but the
current evidence is retrospective and based on only four positives.

## Files

```text
src/models/low_similarity_withheld_known_inhibitor_recovery_level3.py
results/level3_low_similarity_feasibility.csv
results/level3_low_similarity_training_removals.csv
results/low_similarity_withheld_known_inhibitor_rankings.csv
results/withheld_known_inhibitor_level3_summary.json
results/level1_level2_level3_comparison.csv
results/level1_level2_level3_metrics.csv
results/figures/level1_level2_level3_ranks.png
results/figures/level1_level2_level3_recovery.png
```

## Quality control

Verified: same four positives and same 475-molecule pool as Levels 1-2;
threshold remained fixed at 0.60; exact match and same-scaffold counts are
zero; all positive max sim < 0.60; frozen RF configuration unchanged; no
result was used for model selection.
