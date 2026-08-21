# M2 - Morgan fingerprint baseline

## What this milestone does

```text
canonical_smiles
  -> RDKit Mol
  -> Morgan fingerprint (radius 2, 2048 bits)
  -> RandomForest / XGBoost
  -> pIC50 regression
```

This is the first actual machine-learning baseline of the project. It uses a
**fixed, handcrafted molecular representation** (Morgan fingerprint), which
will later be compared with learned graph (M4) and sequence (M5)
representations.

## Files added

```text
src/models/morgan_baseline.py
notebooks/02_morgan_baseline.ipynb
```

## Install the new dependencies

```bash
conda activate egfr-aidd
conda install -c conda-forge xgboost
```

`scikit-learn` and `scipy` are now also listed in `environment.yml`; if your
environment was created before this update, install them too:

```bash
conda install -c conda-forge scikit-learn scipy
```

## Run

```bash
cd "/path/to/AIDD人工智能药物设计与深度学习蛋白质预测/egfr-aidd-project"
conda activate egfr-aidd

python src/models/morgan_baseline.py
```

Then open `notebooks/02_morgan_baseline.ipynb` in JupyterLab and run it.

## Expected outputs

```text
results/m2_morgan_baseline_results.json
results/figures/m2_morgan_baseline_predicted_vs_actual.png
```

The JSON contains MAE, RMSE, R2, Pearson r, and Spearman rho for both
RandomForest and XGBoost on the random-split test set.

## What to check before M3

1. Are RMSE and R2 reasonable for a pIC50 range of about 2-12?
2. Do predicted vs actual points follow the diagonal?
3. Which model looks better, and by how much?
4. Remember: random split overestimates real-world generalization; M3 adds a
   scaffold split to test "new chemistry".
