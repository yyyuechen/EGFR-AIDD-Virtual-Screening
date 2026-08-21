# M3 - Morgan baseline: random split vs scaffold split

## What this milestone does

M3 keeps the same Morgan fingerprint (`radius=2`, `nBits=2048`) and the same
RandomForest / XGBoost hyperparameters from M2, then compares two data splits:

```text
random split   -> sklearn train_test_split, seed=42
scaffold split -> Murcko scaffold groups kept intact
```

The scaffold split is the harder test: molecules in the test set come from
scaffolds that were not seen during training, which better mimics
generalization to new chemistry.

## Files added

```text
src/models/evaluate_splits.py
notebooks/03_morgan_split_comparison.ipynb
```

## Run

```bash
cd "/path/to/AIDD人工智能药物设计与深度学习蛋白质预测/egfr-aidd-project"
conda activate egfr-aidd

python src/models/evaluate_splits.py
```

## Expected outputs

```text
results/m3_morgan_split_comparison_results.json
results/figures/m3_random_vs_scaffold_split.png
```

The results JSON records audit information in addition to the metrics:
package versions, complete model parameters, train/test indices, pIC50
train/test distribution stats, scaffold assignments, and scaffold size
distributions.

## What to check

1. Random split R2 / RMSE should be the optimistic reference.
2. Scaffold split R2 / RMSE is expected to be worse; that is the point.
3. The gap shows how much the model relied on seeing similar scaffolds in
   training.
4. M4 (GNN) and M5 (Transformer) should be evaluated with the same scaffold
   split so the comparison is fair.

## Known boundaries of the scaffold split

The current scaffold split is deliberately strict, and that makes its
interpretation narrower than "molecules are harder":

* The test set contains almost entirely **single-molecule scaffolds**. In the
  current run, all 2,033 test scaffolds have size 1, while the train side
  contains scaffold families up to hundreds of molecules. M3 therefore tests
  **extrapolation to unseen scaffolds**, not generalization within a known
  scaffold family.
* The test pIC50 distribution is shifted relative to the train set (test mean
  is lower: train mean 7.010, test mean 6.676), so part of the R2 drop comes
  from a label-distribution shift and not only from chemistry being harder.
  Both effects are recorded in
  `results/m3_morgan_split_comparison_results.json` under
  `pIC50_train_stats` / `pIC50_test_stats`.

If M4/M5 should also demonstrate whether a learned representation captures
within-scaffold SAR, add a scaffold-balanced split (e.g. scaffold groups are
sampled across both train and test while keeping molecules of the same
scaffold together). Keep the current strict split as the "new scaffold"
boundary case.
