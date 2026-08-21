# M5 - SMILES Transformer baseline

## What this milestone does

M5 trains a small SMILES Transformer encoder for pIC50 regression:

```text
canonical_smiles
  -> tokenized SMILES sequence
  -> Transformer encoder (learned sequence embedding)
  -> pIC50 prediction
```

The Transformer uses the **same Murcko scaffold split as M3 and M4**, so every
model sees exactly the same test molecules.

## Files added

```text
src/models/smiles_transformer.py
notebooks/05_smiles_transformer.ipynb
```

## Run

```bash
cd "/path/to/AIDD人工智能药物设计与深度学习蛋白质预测/egfr-aidd-project"
conda activate egfr-aidd

KMP_DUPLICATE_LIB_OK=TRUE python src/models/smiles_transformer.py
```

Training takes about 10 minutes on CPU with the default settings
(30 epochs, hidden size 64, 2 Transformer layers).

## Expected outputs

```text
results/m5_smiles_transformer_results.json
results/figures/m5_smiles_transformer_scaffold_comparison.png
```

## Current scaffold-split result

| Model | R2 | RMSE | MAE |
|---|---:|---:|---:|
| RandomForest (M3 Morgan) | 0.3977 | 0.9985 | 0.7588 |
| XGBoost (M3 Morgan) | 0.3237 | 1.0581 | 0.8116 |
| GCN (M4 graph) | 0.1361 | 1.1958 | 0.9652 |
| GIN/GINE (M4 graph) | 0.2178 | 1.1379 | 0.9136 |
| SMILES Transformer (M5) | 0.0438 | 1.2581 | 0.9871 |

The SMILES Transformer performs worse than both the Morgan baseline and the
graph models under the strict scaffold split. This is another useful negative
result: a small, lightly-tuned learned sequence representation does not beat
simple fingerprints on unseen scaffolds.

Interpretation caveats:

* R2 is low (0.044) while Pearson is moderate (0.477), which is typical of
  predictions compressed toward the mean. The model is a weak baseline, not a
  tuned production model.
* The test set is the same extreme "new scaffold" split as M3/M4: all test
  scaffolds are singletons, and the test pIC50 distribution is shifted lower.
* Training used one seed and no validation-based early stopping.

Reproduction: `src/models/smiles_transformer.py` is the script that trains
and saves the JSON. The notebook loads saved results for fast inspection and
verifies the split/tokenizer/model metadata against the JSON; set
`M5_RETRAIN=1` before running the notebook if you want it to retrain
(about 10 minutes on CPU).

The results JSON records tokenizer details, model parameters, package
versions, train/test indices, scaffold assignments, scaffold sizes, and pIC50
distribution stats.
