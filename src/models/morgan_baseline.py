#!/usr/bin/env python3
"""
M2: Morgan fingerprint baseline for EGFR pIC50 regression.

Pipeline
--------
canonical_smiles
  -> RDKit Mol
  -> Morgan fingerprint (fixed molecular representation)
  -> RandomForest / XGBoost
  -> pIC50 prediction

This milestone deliberately does NOT implement scaffold split or model
ensembles.  The goal is a simple, interpretable baseline that M3 (random vs
scaffold split) and M4/M5 (learned representations) can be compared against.

Usage (from the project root)
-----------------------------
    python src/models/morgan_baseline.py
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import scipy
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit(
        "xgboost is required for M2. Install it with:\n"
        "    conda activate egfr-aidd\n"
        "    conda install -c conda-forge xgboost"
    ) from exc


def print_step(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def smiles_to_morgan_fingerprint(
    smiles: str, radius: int = 2, n_bits: int = 2048
) -> np.ndarray | None:
    """
    Convert one SMILES string into a Morgan fingerprint bit vector.

    Input:
        A SMILES string such as "CCO".

    Output:
        A NumPy vector of length n_bits with 0/1 entries, or None if the
        SMILES cannot be parsed.

    Why:
        Machine-learning models need a fixed-size numeric vector.  Morgan
        fingerprints encode the local chemical environments around each atom,
        which is the same "fixed representation" concept used in Day02 of the
        AIDD course.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    bit_vector = generator.GetFingerprint(mol)
    return np.array(list(bit_vector), dtype=np.float32)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the regression metrics used to judge the baseline."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "Spearman_rho": float(spearmanr(y_true, y_pred)[0]),
    }


def distribution_stats(values: np.ndarray) -> dict:
    """Mean/std/min/median/max of a numeric array for audit output."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def json_safe(value):
    """Convert numpy/Python objects into JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def package_versions() -> dict:
    """Record package versions so results can be audited and reproduced."""
    return {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "matplotlib": matplotlib.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M2 Morgan fingerprint baseline")
    parser.add_argument(
        "--data-file",
        default="data/processed/egfr_activity_final.csv",
        help="M1 final molecule-level dataset.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Where to save the results JSON.",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/figures",
        help="Where to save the evaluation figures.",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--xgb-trees", type=int, default=300)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: load the M1 dataset
    # ------------------------------------------------------------------
    print_step("Step 1: load M1 dataset")
    df = pd.read_csv(args.data_file)
    required = ["canonical_smiles", "pIC50"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    print(f"  molecules: {len(df):,}")
    print(f"  pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")

    # ------------------------------------------------------------------
    # Step 2: SMILES -> RDKit Mol -> Morgan fingerprint
    # ------------------------------------------------------------------
    print_step("Step 2: build Morgan fingerprints")
    fingerprints = df["canonical_smiles"].map(
        lambda s: smiles_to_morgan_fingerprint(s, args.radius, args.n_bits)
    )
    invalid = int(fingerprints.isna().sum())
    if invalid:
        raise ValueError(f"{invalid} SMILES could not be parsed by RDKit")

    X = np.vstack(fingerprints.to_numpy())
    y = df["pIC50"].to_numpy(dtype=np.float32)
    print(f"  X shape: {X.shape}  (n_molecules x n_fingerprint_bits)")
    print(f"  y shape: {y.shape}")

    # ------------------------------------------------------------------
    # Step 3: random train/test split
    # ------------------------------------------------------------------
    print_step("Step 3: random train/test split")
    all_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=args.test_size, random_state=args.seed
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  train: {X_train.shape[0]:,}   test: {X_test.shape[0]:,}")

    # ------------------------------------------------------------------
    # Step 4: train RandomForest and XGBoost
    # ------------------------------------------------------------------
    print_step("Step 4: train models")
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=args.rf_trees,
            random_state=args.seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=args.xgb_trees,
            learning_rate=0.05,
            max_depth=6,
            random_state=args.seed,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"  training {name} ...")
        model.fit(X_train, y_train)
        trained[name] = model

    # ------------------------------------------------------------------
    # Step 5: evaluate on the held-out test set
    # ------------------------------------------------------------------
    print_step("Step 5: evaluation on test set")
    metrics = {}
    predictions = {}
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        metrics[name] = evaluate_regression(y_test, y_pred)
        print(f"\n  {name}:")
        for metric, value in metrics[name].items():
            print(f"    {metric}: {value:.4f}")

    # ------------------------------------------------------------------
    # Step 6: save results and figures
    # ------------------------------------------------------------------
    print_step("Step 6: save results and figures")
    summary = {
        "milestone": "M2",
        "fingerprint": {
            "type": "Morgan",
            "radius": args.radius,
            "n_bits": args.n_bits,
        },
        "split": {
            "type": "random",
            "test_size": args.test_size,
            "seed": args.seed,
            "test_frac_actual": float(len(test_idx) / len(df)),
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
        },
        "pIC50_train_stats": distribution_stats(y_train),
        "pIC50_test_stats": distribution_stats(y_test),
        "versions": package_versions(),
        "model_params": json_safe(
            {name: model.get_params() for name, model in models.items()}
        ),
        "n_molecules": int(len(df)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "metrics": metrics,
    }
    summary_path = results_dir / "m2_morgan_baseline_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved -> {summary_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_test, y_pred, s=8, alpha=0.4)
        limit = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(limit, limit, "r--", label="perfect prediction")
        ax.set_xlabel("observed pIC50")
        ax.set_ylabel("predicted pIC50")
        ax.set_title(
            f"{name}\nR2={metrics[name]['R2']:.3f} "
            f"RMSE={metrics[name]['RMSE']:.3f}"
        )
        ax.legend()
    fig.tight_layout()
    fig_path = figures_dir / "m2_morgan_baseline_predicted_vs_actual.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  saved -> {fig_path}")

    print_step("M2 complete")
    print("Next: M3 will compare this random-split baseline with a scaffold split.")


if __name__ == "__main__":
    main()
