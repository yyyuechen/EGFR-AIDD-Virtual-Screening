#!/usr/bin/env python3
"""
Controlled hyperparameter study: Random Forest max_depth.

This is the final RF hyperparameter investigated before final model
confirmation. Exactly one modeling decision changes: `max_depth`
(20, 40, 60, 80, None). Everything else stays fixed:

  - radius = 2, nBits = 2048 Morgan fingerprints (same implementation as M2/M3);
  - the standard 80/10/10 Bemis-Murcko scaffold split (seed 42);
  - training and validation molecules only (the standard-scaffold TEST set
    is never touched);
  - n_estimators = 300, random_state = 42, n_jobs = -1, min_samples_leaf = 1,
    max_features = 0.20, and all other RF parameters at current defaults.

Primary metric: validation Spearman. Secondary: validation R2 / RMSE / MAE.
Training metrics, generalization gaps, and tree-complexity diagnostics
(including how many trees actually reach the imposed depth limit) are
recorded so capacity effects can be interpreted directly.

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/max_depth_tuning_study.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from src.models.evaluate_splits import (
    evaluate_regression,
    json_safe,
    package_versions,
    smiles_to_morgan_fingerprint,
)
from src.models.portfolio_hardening_analysis import (
    standard_scaffold_split_indices,
)


RADIUS = 2
N_BITS = 2048
N_ESTIMATORS = 300
SEED = 42
MAX_FEATURES = 0.2
MAX_DEPTH_LIST = [20, 40, 60, 80, None]


def make_rf(max_depth: int | None) -> RandomForestRegressor:
    """RF with the current reference settings, only max_depth changed."""
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=1,
        max_features=MAX_FEATURES,
        max_depth=max_depth,
    )


def forest_complexity(
    rf: RandomForestRegressor,
    X_train: np.ndarray,
    max_depth: int | None,
) -> dict:
    """Depth/leaf statistics, samples per leaf, and depth-limit saturation."""
    depths = np.asarray([t.tree_.max_depth for t in rf.estimators_], dtype=int)
    n_leaves = np.asarray(
        [t.tree_.n_leaves for t in rf.estimators_], dtype=int
    )

    leaf_ids = rf.apply(X_train)
    samples_per_leaf = []
    for col in range(leaf_ids.shape[1]):
        _, counts = np.unique(leaf_ids[:, col], return_counts=True)
        samples_per_leaf.append(float(np.mean(counts)))
    samples_arr = np.asarray(samples_per_leaf, dtype=float)

    if max_depth is None:
        fraction_hitting = 0.0
    else:
        fraction_hitting = float(np.mean(depths >= max_depth))

    return {
        "mean_tree_depth": float(np.mean(depths)),
        "median_tree_depth": float(np.median(depths)),
        "max_tree_depth": int(np.max(depths)),
        "min_tree_depth": int(np.min(depths)),
        "mean_n_leaves": float(np.mean(n_leaves)),
        "median_n_leaves": float(np.median(n_leaves)),
        "max_n_leaves": int(np.max(n_leaves)),
        "mean_samples_per_leaf": float(np.mean(samples_arr)),
        "median_samples_per_leaf": float(np.median(samples_arr)),
        "fraction_trees_hitting_depth_limit": fraction_hitting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RandomForest max_depth study (validation only)"
    )
    parser.add_argument(
        "--data-file", default="data/processed/egfr_activity_final.csv"
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_file)
    smiles_all = df["canonical_smiles"].tolist()
    y_all = df["pIC50"].to_numpy(dtype=np.float32)

    std_train, std_val, std_test, _, split_meta = (
        standard_scaffold_split_indices(
            smiles_all,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            seed=42,
        )
    )
    assert len(std_test) > 0
    print(
        f"standard scaffold split: train={len(std_train)} "
        f"val={len(std_val)} test={len(std_test)} (test not used)"
    )

    smiles_train = [smiles_all[i] for i in std_train]
    smiles_val = [smiles_all[i] for i in std_val]
    y_train = y_all[std_train]
    y_val = y_all[std_val]

    print(f"fingerprint: Morgan radius={RADIUS}, nBits={N_BITS}")
    print(f"fixed RF: n_estimators={N_ESTIMATORS}, min_samples_leaf=1, "
          f"max_features={MAX_FEATURES}, random_state={SEED}")
    print("building train/validation fingerprint matrices...", flush=True)
    X_train = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in smiles_train]
    ).astype(np.float32)
    X_val = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in smiles_val]
    ).astype(np.float32)

    reference = make_rf(None)
    reference_params = reference.get_params()
    print("\nfull effective RandomForestRegressor parameter dictionary "
          "(reference, max_depth=None):")
    for key, value in sorted(reference_params.items()):
        print(f"  {key}: {value}")

    rows: list[dict] = []
    detail: dict = {}

    for md in MAX_DEPTH_LIST:
        label = "None" if md is None else str(md)
        print(f"\n--- max_depth={label} ---", flush=True)
        rf = make_rf(md)
        rf.fit(X_train, y_train)

        pred_train = rf.predict(X_train)
        pred_val = rf.predict(X_val)
        train_metrics = evaluate_regression(y_train, pred_train)
        val_metrics = evaluate_regression(y_val, pred_val)
        train_spearman = float(spearmanr(y_train, pred_train)[0])
        val_spearman = float(spearmanr(y_val, pred_val)[0])

        complexity = forest_complexity(rf, X_train, md)

        row = {
            "max_depth": "None" if md is None else md,
            "train_MAE": train_metrics["MAE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "train_Spearman": train_spearman,
            "val_MAE": val_metrics["MAE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_R2": val_metrics["R2"],
            "val_Spearman": val_spearman,
            "R2_gap": train_metrics["R2"] - val_metrics["R2"],
            "Spearman_gap": train_spearman - val_spearman,
            "RMSE_gap": val_metrics["RMSE"] - train_metrics["RMSE"],
            "mean_tree_depth": complexity["mean_tree_depth"],
            "median_tree_depth": complexity["median_tree_depth"],
            "max_tree_depth": complexity["max_tree_depth"],
            "mean_n_leaves": complexity["mean_n_leaves"],
            "median_n_leaves": complexity["median_n_leaves"],
            "mean_samples_per_leaf": complexity["mean_samples_per_leaf"],
            "fraction_trees_hitting_depth_limit": (
                complexity["fraction_trees_hitting_depth_limit"]
            ),
        }
        rows.append(row)
        detail[label] = {
            "max_depth": None if md is None else md,
            "train_metrics": {
                **train_metrics,
                "Spearman_rho": train_spearman,
            },
            "validation_metrics": {
                **val_metrics,
                "Spearman_rho": val_spearman,
            },
            "gaps": {
                "Spearman_gap_train_minus_val": train_spearman - val_spearman,
                "R2_gap_train_minus_val": (
                    train_metrics["R2"] - val_metrics["R2"]
                ),
                "RMSE_gap_val_minus_train": (
                    val_metrics["RMSE"] - train_metrics["RMSE"]
                ),
            },
            "tree_complexity": complexity,
            "rf_params": rf.get_params(),
        }
        print(
            f"  train: MAE={train_metrics['MAE']:.4f} "
            f"RMSE={train_metrics['RMSE']:.4f} "
            f"R2={train_metrics['R2']:.4f} "
            f"Spearman={train_spearman:.4f}"
        )
        print(
            f"  val:   MAE={val_metrics['MAE']:.4f} "
            f"RMSE={val_metrics['RMSE']:.4f} "
            f"R2={val_metrics['R2']:.4f} "
            f"Spearman={val_spearman:.4f}"
        )
        print(
            f"  gaps:  Spearman={row['Spearman_gap']:.4f} "
            f"R2={row['R2_gap']:.4f} RMSE={row['RMSE_gap']:.4f}"
        )
        print(
            f"  trees: mean_depth={complexity['mean_tree_depth']:.2f} "
            f"median_depth={complexity['median_tree_depth']:.1f} "
            f"max_depth={complexity['max_tree_depth']} "
            f"mean_leaves={complexity['mean_n_leaves']:.1f} "
            f"mean_samples_per_leaf={complexity['mean_samples_per_leaf']:.1f}"
        )
        print(
            f"  depth-limit saturation: "
            f"{100.0 * complexity['fraction_trees_hitting_depth_limit']:.1f}%"
        )

    out = pd.DataFrame(rows)
    csv_path = results_dir / "max_depth_tuning_results.csv"
    out.to_csv(csv_path, index=False)

    summary = {
        "milestone": "RandomForest max_depth tuning study",
        "validation_only": True,
        "test_set_used": False,
        "fingerprint": {
            "type": "Morgan",
            "radius": RADIUS,
            "nBits": N_BITS,
            "implementation": "AllChem.GetMorganGenerator(...).GetFingerprint(mol)",
        },
        "split": {
            **split_meta,
            "n_train": len(std_train),
            "n_validation": len(std_val),
            "n_test": len(std_test),
        },
        "rf_reference_params": json_safe(reference_params),
        "max_depth_values": [None if v is None else v for v in MAX_DEPTH_LIST],
        "primary_metric": "validation Spearman",
        "secondary_metrics": ["validation R2", "validation RMSE", "validation MAE"],
        "versions": package_versions(),
        "results": rows,
        "detail": detail,
    }
    json_path = results_dir / "max_depth_tuning_results.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {json_path}")

    # Figure 1: max_depth vs train/validation Spearman (categorical axis).
    labels = ["20", "40", "60", "80", "None"]
    x_pos = np.arange(len(labels))
    train_sp = [r["train_Spearman"] for r in rows]
    val_sp = [r["val_Spearman"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 4))
    ax.plot(
        x_pos,
        train_sp,
        marker="o",
        label="train",
        color="#C44E52",
    )
    ax.plot(
        x_pos,
        val_sp,
        marker="s",
        label="validation",
        color="#4C72B0",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Spearman")
    ax.set_title("Train vs validation Spearman")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig1_path = figures_dir / "max_depth_train_val.png"
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)

    # Figure 2: actual mean tree depth vs validation metrics.
    mean_depth = [r["mean_tree_depth"] for r in rows]
    val_r2 = [r["val_R2"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(
        mean_depth,
        val_sp,
        marker="o",
        color="#4C72B0",
    )
    axes[0].set_xlabel("actual mean tree depth")
    axes[0].set_ylabel("validation Spearman")
    axes[0].set_title("Validation Spearman vs depth")
    axes[0].grid(alpha=0.3)
    axes[1].plot(
        mean_depth,
        val_r2,
        marker="o",
        color="#55A868",
    )
    axes[1].set_xlabel("actual mean tree depth")
    axes[1].set_ylabel("validation R2")
    axes[1].set_title("Validation R2 vs depth")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig2_path = figures_dir / "max_depth_complexity.png"
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)

    print(f"saved -> {fig1_path}")
    print(f"saved -> {fig2_path}")


if __name__ == "__main__":
    main()
