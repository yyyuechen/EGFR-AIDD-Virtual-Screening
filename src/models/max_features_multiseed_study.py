#!/usr/bin/env python3
"""
Multi-seed reproducibility study for the Random Forest max_features benefit.

This is NOT a hyperparameter search. The previous validation-only studies
found a broad optimum around max_features = 0.10-0.20 using random_state=42.
This study tests whether the feature-subsampling benefit is reproducible
across five RF random seeds while keeping everything else fixed:

  - the same standard 80/10/10 Bemis-Murcko scaffold split (seed 42 for the
    split itself; the split is NOT re-generated per RF seed);
  - Morgan radius = 2, nBits = 2048;
  - n_estimators = 300, min_samples_leaf = 1, max_depth = None, n_jobs = -1;
  - all other RF parameters unchanged.

max_features values: 1.00, 0.20, 0.15, 0.10.
RF random states: 42, 7, 123, 2024, 777.

The standard-scaffold TEST set is never touched.

Outputs
-------
results/max_features_multiseed_results.csv
results/max_features_multiseed_summary.json
results/figures/max_features_multiseed_spearman.png

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/max_features_multiseed_study.py
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
MAX_FEATURES_LIST = [1.0, 0.2, 0.15, 0.1]
RF_SEEDS = [42, 7, 123, 2024, 777]
METRIC_NAMES = ["MAE", "RMSE", "R2", "Spearman"]


def make_rf(max_features: float, seed: int) -> RandomForestRegressor:
    """RF with fixed reference settings; only max_features/seed vary."""
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=1,
        max_features=max_features,
    )


def summarize_metric(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def paired_improvement(
    low_values: list[float], baseline_values: list[float], seeds: list[int]
) -> dict:
    diffs = np.asarray(low_values, dtype=float) - np.asarray(
        baseline_values, dtype=float
    )
    return {
        "mean_improvement": float(np.mean(diffs)),
        "std_improvement": float(np.std(diffs, ddof=1))
        if len(diffs) > 1
        else 0.0,
        "min_improvement": float(np.min(diffs)),
        "max_improvement": float(np.max(diffs)),
        "n_seeds_positive": int((diffs > 0).sum()),
        "n_seeds_total": len(seeds),
        "per_seed_improvement": {
            str(seed): float(diff) for seed, diff in zip(seeds, diffs)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="max_features multi-seed reproducibility study"
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

    reference = make_rf(1.0, 42)
    reference_params = reference.get_params()
    print("\nfull effective RandomForestRegressor parameter dictionary "
          "(reference, max_features=1.0, random_state=42):")
    for key, value in sorted(reference_params.items()):
        print(f"  {key}: {value}")

    seed_rows: list[dict] = []
    predictions_by_mf: dict[str, dict[int, np.ndarray]] = {}
    metrics_by_mf: dict[str, dict[int, dict]] = {}

    for mf in MAX_FEATURES_LIST:
        predictions_by_mf[str(mf)] = {}
        metrics_by_mf[str(mf)] = {}
        for seed in RF_SEEDS:
            print(
                f"\n--- max_features={mf} random_state={seed} "
                f"({len(seed_rows) + 1}/{len(MAX_FEATURES_LIST) * len(RF_SEEDS)}) ---",
                flush=True,
            )
            rf = make_rf(mf, seed)
            rf.fit(X_train, y_train)
            pred_val = rf.predict(X_val)
            pred_val_full = rf.predict(X_train)
            metrics = evaluate_regression(y_val, pred_val)
            spearman_val = float(spearmanr(y_val, pred_val)[0])
            metrics_full = dict(metrics)
            metrics_full["Spearman_rho"] = spearman_val
            metrics_full["Spearman"] = spearman_val

            predictions_by_mf[str(mf)][seed] = pred_val.astype(np.float64)
            metrics_by_mf[str(mf)][seed] = metrics_full
            seed_rows.append(
                {
                    "max_features": mf,
                    "random_state": seed,
                    "val_MAE": metrics_full["MAE"],
                    "val_RMSE": metrics_full["RMSE"],
                    "val_R2": metrics_full["R2"],
                    "val_Spearman": metrics_full["Spearman_rho"],
                    "train_R2": float(
                        evaluate_regression(y_train, pred_val_full)["R2"]
                    ),
                }
            )
            print(
                f"  val: MAE={metrics_full['MAE']:.4f} "
                f"RMSE={metrics_full['RMSE']:.4f} "
                f"R2={metrics_full['R2']:.4f} "
                f"Spearman={metrics_full['Spearman_rho']:.4f}"
            )

    by_mf: dict[str, dict] = {}
    baseline = {
        metric: [metrics_by_mf["1.0"][seed][metric] for seed in RF_SEEDS]
        for metric in METRIC_NAMES
    }
    for mf in MAX_FEATURES_LIST:
        key = str(mf)
        metric_values = {
            metric: [
                metrics_by_mf[key][seed][metric] for seed in RF_SEEDS
            ]
            for metric in METRIC_NAMES
        }
        preds = np.vstack(
            [predictions_by_mf[key][seed] for seed in RF_SEEDS]
        )
        pred_std = np.std(preds, axis=0, ddof=1)
        by_mf[key] = {
            "max_features": mf,
            "metrics": {
                metric: summarize_metric(metric_values[metric])
                for metric in METRIC_NAMES
            },
            "prediction_stability_across_seeds": {
                "mean_pred_std": float(np.mean(pred_std)),
                "median_pred_std": float(np.median(pred_std)),
                "p95_pred_std": float(np.quantile(pred_std, 0.95)),
            },
            "n_seeds": len(RF_SEEDS),
        }
        if key != "1.0":
            by_mf[key]["paired_vs_1_0"] = {
                metric: paired_improvement(
                    metric_values[metric], baseline[metric], RF_SEEDS
                )
                for metric in METRIC_NAMES
            }

    seed_df = pd.DataFrame(seed_rows)
    csv_path = results_dir / "max_features_multiseed_results.csv"
    seed_df.to_csv(csv_path, index=False)

    summary = {
        "milestone": "max_features multi-seed reproducibility study",
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
            "split_fixed_across_rf_seeds": True,
        },
        "rf_reference_params": json_safe(reference_params),
        "max_features_values": MAX_FEATURES_LIST,
        "rf_random_states": RF_SEEDS,
        "primary_metric": "validation Spearman",
        "secondary_metrics": ["validation R2", "validation RMSE", "validation MAE"],
        "versions": package_versions(),
        "per_seed_results": seed_rows,
        "by_max_features": by_mf,
    }
    json_path = results_dir / "max_features_multiseed_summary.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {json_path}")

    # Figure: mean validation Spearman with +/- 1 SD across seeds.
    mf_vals = [r["max_features"] for r in [
        dict(by_mf["1.0"]),
        dict(by_mf["0.2"]),
        dict(by_mf["0.15"]),
        dict(by_mf["0.1"]),
    ]]
    mean_sp = [by_mf[str(m)]["metrics"]["Spearman"]["mean"] for m in mf_vals]
    std_sp = [by_mf[str(m)]["metrics"]["Spearman"]["std"] for m in mf_vals]
    fig, ax = plt.subplots(figsize=(6.0, 4))
    ax.errorbar(
        mf_vals,
        mean_sp,
        yerr=std_sp,
        marker="o",
        capsize=4,
        color="#4C72B0",
        label="mean +/- 1 SD",
    )
    ax.set_xlabel("max_features")
    ax.set_ylabel("validation Spearman")
    ax.set_title("Validation Spearman across RF seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = figures_dir / "max_features_multiseed_spearman.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {fig_path}")


if __name__ == "__main__":
    main()
