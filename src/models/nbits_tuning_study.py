#!/usr/bin/env python3
"""
Controlled hyperparameter study: Morgan fingerprint nBits.

This study changes exactly one modeling decision at a time: the Morgan
fingerprint dimensionality (512, 1024, 2048, 4096). Everything else stays
fixed:

  - radius = 2;
  - the standard 80/10/10 Bemis-Murcko scaffold split (seed 42);
  - training and validation molecules only (the standard-scaffold TEST set
    is never touched);
  - RandomForest hyperparameters from M2/M3 (300 trees, random_state 42);
  - preprocessing, endpoint definition, and all other settings.

Primary metric: validation Spearman. Secondary: validation RMSE / MAE / R2.

The study also reports fingerprint sparsity/density, fingerprint
uniqueness/collisions on the train+validation set, and validation
max_train_tanimoto versus the training set.

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/nbits_tuning_study.py
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

from src.models.evaluate_splits import (
    evaluate_regression,
    json_safe,
    make_models,
    package_versions,
    smiles_to_morgan_fingerprint,
)
from src.models.portfolio_hardening_analysis import (
    standard_scaffold_split_indices,
)
from src.models.radius_tuning_study import morgan_bit_vectors


RADIUS = 2
NBITS_LIST = [512, 1024, 2048, 4096]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Morgan nBits study (validation only)"
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
    smiles_eval = smiles_train + smiles_val
    y_train = y_all[std_train]
    y_val = y_all[std_val]

    n_eval = len(smiles_eval)
    n_dup_smiles = df["canonical_smiles"].duplicated().sum()
    assert n_dup_smiles == 0
    print(f"molecules evaluated for representation stats: {n_eval} "
          f"(train+val), duplicate canonical SMILES: {n_dup_smiles}")

    rf_params = make_models(300, 300, 42)["RandomForest"].get_params()
    rows: list[dict] = []
    detail: dict = {}

    for n_bits in NBITS_LIST:
        print(f"\n--- nBits={n_bits} ---", flush=True)
        X_train = np.vstack(
            [
                smiles_to_morgan_fingerprint(s, RADIUS, n_bits)
                for s in smiles_train
            ]
        ).astype(np.float32)
        X_val = np.vstack(
            [
                smiles_to_morgan_fingerprint(s, RADIUS, n_bits)
                for s in smiles_val
            ]
        ).astype(np.float32)
        X_eval = np.vstack([X_train, X_val])

        rf = make_models(300, 300, 42)["RandomForest"]
        rf.fit(X_train, y_train)
        pred = rf.predict(X_val)
        metrics = evaluate_regression(y_val, pred)
        spearman_val = float(spearmanr(y_val, pred)[0])

        on_bits = X_eval.sum(axis=1)
        densities = on_bits / n_bits
        mean_on_bits = float(np.mean(on_bits))
        median_on_bits = float(np.median(on_bits))
        mean_density = float(np.mean(densities))
        median_density = float(np.median(densities))

        unique_rows = np.unique(X_eval, axis=0)
        n_unique = int(unique_rows.shape[0])
        n_duplicated = n_eval - n_unique
        _, counts = np.unique(X_eval, axis=0, return_counts=True)
        collision_groups = int((counts > 1).sum())
        max_group = int(counts.max()) if len(counts) else 0

        sim = max_train_tanimoto_stats(smiles_train, smiles_val, n_bits)

        rows.append(
            {
                "nBits": n_bits,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Spearman": spearman_val,
                "mean_on_bits": mean_on_bits,
                "median_on_bits": median_on_bits,
                "mean_fingerprint_density": mean_density,
                "median_fingerprint_density": median_density,
                "n_molecules_evaluated": n_eval,
                "unique_fingerprints": n_unique,
                "duplicated_fingerprints": n_duplicated,
                "collision_groups": collision_groups,
                "max_collision_group_size": max_group,
                "mean_max_train_tanimoto": sim["mean"],
                "median_max_train_tanimoto": sim["median"],
                "min_max_train_tanimoto": sim["min"],
                "max_max_train_tanimoto": sim["max"],
            }
        )
        detail[str(n_bits)] = {
            "metrics": metrics,
            "spearman_validation": spearman_val,
            "on_bits": {
                "mean": mean_on_bits,
                "median": median_on_bits,
                "mean_density": mean_density,
                "median_density": median_density,
            },
            "uniqueness": {
                "n_molecules_evaluated": n_eval,
                "unique_fingerprints": n_unique,
                "duplicated_fingerprints": n_duplicated,
                "collision_groups": collision_groups,
                "max_collision_group_size": max_group,
                "duplicate_canonical_smiles_in_dataset": n_dup_smiles,
            },
            "validation_max_train_tanimoto": sim,
            "rf_params": rf.get_params(),
        }
        print(
            f"  MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
            f"R2={metrics['R2']:.4f} Spearman={spearman_val:.4f} "
            f"mean_on_bits={mean_on_bits:.2f} mean_density={mean_density:.4f} "
            f"unique={n_unique} collisions={collision_groups} "
            f"mean_max_sim={sim['mean']:.4f}"
        )

    out = pd.DataFrame(rows)
    csv_path = results_dir / "nbits_tuning_results.csv"
    out.to_csv(csv_path, index=False)

    summary = {
        "milestone": "nBits tuning study",
        "validation_only": True,
        "test_set_used": False,
        "fingerprint": {
            "type": "Morgan",
            "radius": RADIUS,
            "nBits_list": NBITS_LIST,
        },
        "split": {
            **split_meta,
            "n_train": len(std_train),
            "n_validation": len(std_val),
            "n_test": len(std_test),
        },
        "rf_params": json_safe(rf_params),
        "primary_metric": "validation Spearman",
        "secondary_metrics": ["validation RMSE", "validation MAE", "validation R2"],
        "versions": package_versions(),
        "results": rows,
        "detail": detail,
    }
    json_path = results_dir / "nbits_tuning_results.json"
    json_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {json_path}")

    # Simple validation-performance figure.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric in zip(axes, ["Spearman", "R2"]):
        ax.plot(
            [r["nBits"] for r in rows],
            [r[metric] for r in rows],
            marker="o",
            color="#4C72B0",
        )
        ax.set_xlabel("nBits")
        ax.set_ylabel(metric)
        ax.set_title(f"Validation {metric}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = figures_dir / "nbits_validation_performance.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {fig_path}")


def max_train_tanimoto_stats(
    smiles_train: list[str], smiles_val: list[str], n_bits: int
) -> dict:
    train_bits = morgan_bit_vectors(smiles_train, RADIUS, n_bits)
    val_bits = morgan_bit_vectors(smiles_val, RADIUS, n_bits)
    from rdkit.Chem import DataStructs

    maxima = np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(fp, train_bits))
            for fp in val_bits
        ],
        dtype=float,
    )
    return {
        "mean": float(np.mean(maxima)),
        "median": float(np.median(maxima)),
        "min": float(np.min(maxima)),
        "max": float(np.max(maxima)),
    }


if __name__ == "__main__":
    main()
