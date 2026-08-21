# M4 - Molecular graph baselines (GCN / GIN)

## What this milestone does

M4 converts the M1 molecules into molecular graphs and trains two graph
models for pIC50 regression:

```text
canonical_smiles
  -> RDKit Mol
  -> molecular graph (atom + bond features)
  -> GCN / GIN
  -> pIC50 prediction
```

The graph models use the **same Murcko scaffold split as M3**, so the test
molecules are exactly the same as the Morgan baseline test set. This keeps the
M3 vs M4 comparison fair.

## Files added

```text
src/models/graph_baseline.py
notebooks/04_molecular_graph_baseline.ipynb
```

## Run

```bash
cd "/path/to/AIDD人工智能药物设计与深度学习蛋白质预测/egfr-aidd-project"
conda activate egfr-aidd

KMP_DUPLICATE_LIB_OK=TRUE python src/models/graph_baseline.py
```

On macOS, `KMP_DUPLICATE_LIB_OK=TRUE` is needed because xgboost and PyTorch
both load an OpenMP runtime.

## Expected outputs

```text
results/m4_graph_baseline_results.json
results/figures/m4_graph_baseline_scaffold_comparison.png
```

## Current scaffold-split result

| Model | R2 | RMSE | MAE |
|---|---:|---:|---:|
| RandomForest (M3 Morgan) | 0.3977 | 0.9985 | 0.7588 |
| XGBoost (M3 Morgan) | 0.3237 | 1.0581 | 0.8116 |
| GCN (M4 graph, atom features only) | 0.1361 | 1.1958 | 0.9652 |
| GIN/GINE (M4 graph, atom + bond features) | 0.2178 | 1.1379 | 0.9136 |

The graph models are clearly worse than the Morgan baseline under the strict
scaffold split. This is expected for a first, lightly-tuned graph baseline and
is a useful result: learned representations do not automatically beat simple
fingerprints, especially on unseen scaffolds.

Implementation note: GCN uses atom features only; GIN is implemented with
`GINEConv`, so it actually consumes the bond features. The results JSON
records `edge_features_used`.

Known limitations: single seed, no validation-based early stopping, and the
M3 scaffold split's test set consists entirely of single-molecule scaffolds.
The scaffold test set also has a lower mean pIC50 than the train set, so part
of the drop is label-distribution shift.

The results JSON records versions, graph/model parameters, train/test
indices, scaffold assignments, scaffold sizes, and pIC50 distribution stats.
