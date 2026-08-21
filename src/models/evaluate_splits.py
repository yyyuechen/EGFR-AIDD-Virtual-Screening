#!/usr/bin/env python3
"""
M3: random split vs Murcko scaffold split for the Morgan baseline.

Why M3?
-------
M2 used a random train/test split, which is optimistic because similar
molecules can appear in both train and test. M3 repeats the same Morgan
fingerprint and the same RandomForest/XGBoost models under two splits:

  random   : sklearn train_test_split with the same seed as M2;
  scaffold : split by Murcko scaffold so that entire scaffolds stay
             together, meaning the test molecules come from scaffolds not
             seen in training.

The scaffold split is a harder and more realistic estimate of generalization
to new chemistry.

Usage (from the project root)
-----------------------------
    python src/models/evaluate_splits.py
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
from rdkit.Chem.Scaffolds import MurckoScaffold
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
        "xgboost is required for M3. Install it with:\n"
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

    This is the same fixed molecular representation used in M2. Morgan
    fingerprints are rule-based, not learned embeddings.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    bit_vector = generator.GetFingerprint(mol)
    return np.array(list(bit_vector), dtype=np.float32)


def scaffold_split_indices(
    smiles: list[str], test_size: float = 0.2
) -> tuple[list[int], list[int], dict]:
    """
    Split molecule indices by Murcko scaffold.

    Molecules sharing the same Murcko scaffold stay in the same split, so the
    test set contains scaffolds that were not observed during training.

    The scaffold groups are sorted largest-first, then by scaffold SMILES for
    deterministic ordering. Scaffold groups are greedily assigned to the train
    side until adding another group would exceed the requested train fraction.
    """
    scaffold_to_indices: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(
                f"RDKit could not parse SMILES at index {idx}: {smi!r}"
            )
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    groups = sorted(
        scaffold_to_indices.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )

    n = len(smiles)
    train_cutoff = int((1.0 - test_size) * n)
    train_idx: list[int] = []
    test_idx: list[int] = []

    for scaffold, indices in groups:
        if len(train_idx) + len(indices) <= train_cutoff:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)

    train_set = set(train_idx)
    train_scaffold_sizes: list[int] = []
    test_scaffold_sizes: list[int] = []
    scaffold_by_index: list[str | None] = [None] * n
    for scaffold, indices in groups:
        in_train = all(i in train_set for i in indices)
        if in_train:
            train_scaffold_sizes.append(len(indices))
        else:
            test_scaffold_sizes.append(len(indices))
        for i in indices:
            scaffold_by_index[i] = scaffold
    stats = {
        "n_scaffolds": len(groups),
        "scaffolds_train": len(train_scaffold_sizes),
        "scaffolds_test": len(test_scaffold_sizes),
        "train_scaffold_sizes": train_scaffold_sizes,
        "test_scaffold_sizes": test_scaffold_sizes,
        "scaffold_size_stats": {
            "train": scaffold_size_stats(train_scaffold_sizes),
            "test": scaffold_size_stats(test_scaffold_sizes),
        },
        "scaffold_by_index": scaffold_by_index,
    }
    return train_idx, test_idx, stats


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the same regression metrics used in M2."""
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


def scaffold_size_stats(sizes: list[int]) -> dict:
    """Summary of scaffold group sizes for one split."""
    arr = np.asarray(sizes, dtype=np.float64)
    if len(arr) == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "singletons": 0,
        }
    return {
        "count": int(len(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "singletons": int((arr == 1).sum()),
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


def make_models(rf_trees: int, xgb_trees: int, seed: int) -> dict:
    """Fresh model instances so the two splits use identical settings."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=rf_trees,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=xgb_trees,
            learning_rate=0.05,
            max_depth=6,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def run_split_evaluation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: dict,
) -> dict:
    """Train every model on one split and evaluate on its held-out test set."""
    metrics: dict = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics[name] = evaluate_regression(y_test, y_pred)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 random vs scaffold split")
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
    # Step 1: load the M1 dataset and build the Morgan feature matrix
    # ------------------------------------------------------------------
    print_step("Step 1: load M1 dataset and build Morgan fingerprints")
    df = pd.read_csv(args.data_file)
    required = ["canonical_smiles", "pIC50"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    print(f"  molecules: {len(df):,}")

    fingerprints = df["canonical_smiles"].map(
        lambda s: smiles_to_morgan_fingerprint(s, args.radius, args.n_bits)
    )
    invalid = int(fingerprints.isna().sum())
    if invalid:
        raise ValueError(f"{invalid} SMILES could not be parsed by RDKit")

    X = np.vstack(fingerprints.to_numpy())
    y = df["pIC50"].to_numpy(dtype=np.float32)
    print(f"  X shape: {X.shape}")

    # ------------------------------------------------------------------
    # Step 2: build both splits
    # ------------------------------------------------------------------
    print_step("Step 2: random split vs Murcko scaffold split")
    all_indices = np.arange(len(df))
    idx_r_train, idx_r_test = train_test_split(
        all_indices, test_size=args.test_size, random_state=args.seed
    )
    X_train_r, X_test_r = X[idx_r_train], X[idx_r_test]
    y_train_r, y_test_r = y[idx_r_train], y[idx_r_test]

    train_idx, test_idx, scaffold_stats = scaffold_split_indices(
        df["canonical_smiles"].tolist(), test_size=args.test_size
    )
    X_train_s, X_test_s = X[train_idx], X[test_idx]
    y_train_s, y_test_s = y[train_idx], y[test_idx]
    scaffold_test_frac = len(test_idx) / len(df)
    if abs(scaffold_test_frac - args.test_size) > 0.05:
        print(
            "  [warn] scaffold split test fraction "
            f"{scaffold_test_frac:.3f} deviates from requested "
            f"{args.test_size:.2f}"
        )

    splits = {
        "random": {
            "type": "random",
            "test_size": args.test_size,
            "seed": args.seed,
            "test_frac_actual": float(len(idx_r_test) / len(df)),
            "train_indices": idx_r_train.tolist(),
            "test_indices": idx_r_test.tolist(),
            "pIC50_train_stats": distribution_stats(y_train_r),
            "pIC50_test_stats": distribution_stats(y_test_r),
            "n_train": int(len(y_train_r)),
            "n_test": int(len(y_test_r)),
        },
        "scaffold": {
            "type": "murcko_scaffold",
            "test_size": args.test_size,
            "test_frac_actual": float(scaffold_test_frac),
            "train_indices": train_idx,
            "test_indices": test_idx,
            "pIC50_train_stats": distribution_stats(y_train_s),
            "pIC50_test_stats": distribution_stats(y_test_s),
            "n_train": int(len(y_train_s)),
            "n_test": int(len(y_test_s)),
            **scaffold_stats,
        },
    }
    print(f"  random   train={len(y_train_r):,}  test={len(y_test_r):,}")
    print(f"  scaffold train={len(y_train_s):,}  test={len(y_test_s):,}")
    print(
        f"  scaffold groups: {scaffold_stats['n_scaffolds']:,} "
        f"(train {scaffold_stats['scaffolds_train']:,}, "
        f"test {scaffold_stats['scaffolds_test']:,})"
    )
    print(f"  scaffold size stats: {scaffold_stats['scaffold_size_stats']}")

    # ------------------------------------------------------------------
    # Step 3: train and evaluate the same models under each split
    # ------------------------------------------------------------------
    print_step("Step 3: train RandomForest and XGBoost for each split")
    for split_name, (X_tr, y_tr, X_te, y_te) in {
        "random": (X_train_r, y_train_r, X_test_r, y_test_r),
        "scaffold": (X_train_s, y_train_s, X_test_s, y_test_s),
    }.items():
        models = make_models(args.rf_trees, args.xgb_trees, args.seed)
        metrics = run_split_evaluation(X_tr, y_tr, X_te, y_te, models)
        splits[split_name]["metrics"] = metrics
        print(f"\n  {split_name} split:")
        for model_name, metric in metrics.items():
            print(f"    {model_name}:")
            for key, value in metric.items():
                print(f"      {key}: {value:.4f}")

    # ------------------------------------------------------------------
    # Step 4: save results and comparison figure
    # ------------------------------------------------------------------
    print_step("Step 4: save results and figure")
    summary = {
        "milestone": "M3",
        "fingerprint": {
            "type": "Morgan",
            "radius": args.radius,
            "n_bits": args.n_bits,
        },
        "versions": package_versions(),
        "model_params": json_safe(
            {
                name: model.get_params()
                for name, model in make_models(
                    args.rf_trees, args.xgb_trees, args.seed
                ).items()
            }
        ),
        "split_test_size": args.test_size,
        "seed": args.seed,
        "splits": splits,
    }
    summary_path = results_dir / "m3_morgan_split_comparison_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved -> {summary_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    model_names = ["RandomForest", "XGBoost"]
    split_names = ["random", "scaffold"]
    x = np.arange(len(model_names))
    width = 0.35

    for ax, metric_name in zip(axes, ["R2", "RMSE"]):
        for j, split_name in enumerate(split_names):
            values = [
                splits[split_name]["metrics"][model][metric_name]
                for model in model_names
            ]
            ax.bar(x + (j - 0.5) * width, values, width, label=split_name)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} by split")
        ax.legend()
    fig.tight_layout()
    fig_path = figures_dir / "m3_random_vs_scaffold_split.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  saved -> {fig_path}")

    print_step("M3 complete")
    print(
        "Compare the two splits: random is optimistic, scaffold is the harder "
        "new-chemistry estimate."
    )


if __name__ == "__main__":
    main()
