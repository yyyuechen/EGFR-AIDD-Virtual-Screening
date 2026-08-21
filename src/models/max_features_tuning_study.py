#!/usr/bin/env python3
"""
Controlled hyperparameter study: Random Forest max_features.

This study changes exactly one modeling decision at a time: the RandomForest
`max_features` fraction (0.2, 0.4, 0.6, 0.8, 1.0). Everything else stays
fixed:

  - radius = 2, nBits = 2048 Morgan fingerprints (same implementation as M2/M3);
  - the standard 80/10/10 Bemis-Murcko scaffold split (seed 42);
  - training and validation molecules only (the standard-scaffold TEST set
    is never touched);
  - n_estimators = 300, random_state = 42, n_jobs = -1, min_samples_leaf = 1,
    max_depth = None, and all other RF parameters at current defaults.

Primary model-selection metric: validation Spearman. Secondary: validation
RMSE / MAE / R2. Training metrics and generalization gaps are diagnostic
only. Tree complexity and feature-importance diagnostics are recorded so the
effect of feature subsampling can be inspected.

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/max_features_tuning_study.py
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
MAX_FEATURES_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]


def make_rf(max_features: float) -> RandomForestRegressor:
    """RF with the reference settings, only max_features changed."""
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=1,
        max_features=max_features,
    )


def forest_complexity(rf: RandomForestRegressor) -> dict:
    """Mean/median/max depth and leaf counts across all trees."""
    depths = np.asarray([t.tree_.max_depth for t in rf.estimators_], dtype=int)
    n_leaves = np.asarray(
        [t.tree_.n_leaves for t in rf.estimators_], dtype=int
    )
    return {
        "mean_tree_depth": float(np.mean(depths)),
        "median_tree_depth": float(np.median(depths)),
        "max_tree_depth": int(np.max(depths)),
        "min_tree_depth": int(np.min(depths)),
        "mean_n_leaves": float(np.mean(n_leaves)),
        "median_n_leaves": float(np.median(n_leaves)),
        "max_n_leaves": int(np.max(n_leaves)),
    }


def feature_importance_stats(rf: RandomForestRegressor) -> dict:
    """Top-feature concentration and non-zero feature count."""
    importances = np.asarray(rf.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1]
    total = float(importances.sum())
    top10 = float(importances[order[:10]].sum() / total)
    top50 = float(importances[order[:50]].sum() / total)
    return {
        "top10_importance_fraction": top10,
        "top50_importance_fraction": top50,
        "n_features_with_nonzero_importance": int((importances > 0).sum()),
        "n_features_total": int(len(importances)),
        "top10_feature_indices": [int(i) for i in order[:10]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RandomForest max_features study (validation only)"
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
    print("building train/validation fingerprint matrices...", flush=True)
    X_train = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in smiles_train]
    ).astype(np.float32)
    X_val = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in smiles_val]
    ).astype(np.float32)

    reference = make_rf(1.0)
    reference_params = reference.get_params()
    print("\nfull effective RandomForestRegressor parameter dictionary "
          "(reference, max_features=1.0):")
    for key, value in sorted(reference_params.items()):
        print(f"  {key}: {value}")

    rows: list[dict] = []
    detail: dict = {}

    for mf in MAX_FEATURES_LIST:
        print(f"\n--- max_features={mf} ---", flush=True)
        rf = make_rf(mf)
        rf.fit(X_train, y_train)

        pred_train = rf.predict(X_train)
        pred_val = rf.predict(X_val)
        train_metrics = evaluate_regression(y_train, pred_train)
        val_metrics = evaluate_regression(y_val, pred_val)
        train_spearman = float(spearmanr(y_train, pred_train)[0])
        val_spearman = float(spearmanr(y_val, pred_val)[0])

        complexity = forest_complexity(rf)
        importances = feature_importance_stats(rf)

        row = {
            "max_features": mf,
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
            "mean_n_leaves": complexity["mean_n_leaves"],
            "top10_importance_fraction": importances["top10_importance_fraction"],
            "top50_importance_fraction": importances["top50_importance_fraction"],
            "n_features_with_nonzero_importance": (
                importances["n_features_with_nonzero_importance"]
            ),
        }
        rows.append(row)
        detail[str(mf)] = {
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
            "feature_importance": importances,
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
            f"mean_leaves={complexity['mean_n_leaves']:.1f}"
        )
        print(
            f"  importance: top10={importances['top10_importance_fraction']:.4f} "
            f"top50={importances['top50_importance_fraction']:.4f} "
            f"nonzero={importances['n_features_with_nonzero_importance']}"
        )

    out = pd.DataFrame(rows)
    csv_path = results_dir / "max_features_tuning_results.csv"
    out.to_csv(csv_path, index=False)

    summary = {
        "milestone": "RandomForest max_features tuning study",
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
        "max_features_values": MAX_FEATURES_LIST,
        "primary_metric": "validation Spearman",
        "secondary_metrics": ["validation RMSE", "validation MAE", "validation R2"],
        "versions": package_versions(),
        "results": rows,
        "detail": detail,
    }
    json_path = results_dir / "max_features_tuning_results.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {json_path}")

    # Figure 1: train vs validation Spearman, making the gap visible.
    mf_vals = [r["max_features"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.plot(
        mf_vals,
        [r["train_Spearman"] for r in rows],
        marker="o",
        label="train",
        color="#C44E52",
    )
    ax.plot(
        mf_vals,
        [r["val_Spearman"] for r in rows],
        marker="s",
        label="validation",
        color="#4C72B0",
    )
    ax.set_xlabel("max_features")
    ax.set_ylabel("Spearman")
    ax.set_title("Train vs validation Spearman")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig1_path = figures_dir / "max_features_train_val.png"
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)

    # Figure 2: validation R2 and Spearman together.
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.plot(
        mf_vals,
        [r["val_Spearman"] for r in rows],
        marker="o",
        label="Spearman",
        color="#4C72B0",
    )
    ax.plot(
        mf_vals,
        [r["val_R2"] for r in rows],
        marker="s",
        label="R2",
        color="#55A868",
    )
    ax.set_xlabel("max_features")
    ax.set_ylabel("validation metric")
    ax.set_title("Validation R2 / Spearman")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig2_path = figures_dir / "max_features_validation.png"
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)

    print(f"saved -> {fig1_path}")
    print(f"saved -> {fig2_path}")


if __name__ == "__main__":
    main()
