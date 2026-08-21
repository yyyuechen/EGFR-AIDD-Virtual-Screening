#!/usr/bin/env python3
"""
Repeated scaffold-split robustness study for the locked Morgan-RF models.

This is NOT hyperparameter tuning. The two locked configurations are:

  - baseline: max_features = 1.0;
  - tuned:    max_features = 0.20.

Everything else is identical:

  - Morgan radius = 2, nBits = 2048;
  - n_estimators = 300, min_samples_leaf = 1, max_depth = None, n_jobs = -1;
  - RF random_state = 42 (fixed, so scaffold-partition variation is isolated).

Five deterministic 80/10/10 whole-Murcko-scaffold partitions are generated
with split seeds 7, 21, 42, 123, and 2024 using the same algorithm as the
existing standard split. Seed 42 reproduces the canonical split. Every
configuration is evaluated on that split's TEST set; no parameter is changed
based on any result.

Outputs
-------
results/repeated_scaffold_split_results.csv
results/repeated_scaffold_split_summary.json
results/figures/repeated_scaffold_spearman.png
results/figures/repeated_scaffold_r2.png
results/figures/repeated_scaffold_similarity.png

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/repeated_scaffold_split_study.py
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
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.models.evaluate_splits import (
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
RF_SEED = 42
SPLIT_SEEDS = [7, 21, 42, 123, 2024]
BASELINE_MAX_FEATURES = 1.0
TUNED_MAX_FEATURES = 0.20
ACTIVE_THRESHOLD = 7.0


def make_rf(max_features: float) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RF_SEED,
        n_jobs=-1,
        min_samples_leaf=1,
        max_features=max_features,
        max_depth=None,
    )


def morgan_bit_vectors(smiles_list: list[str]) -> list:
    generator = AllChem.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred)[0]),
    }


def screening_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    active = (y_true >= ACTIVE_THRESHOLD).astype(int)
    n_pos = int(active.sum())
    n_neg = int((active == 0).sum())
    if n_pos < 10 or n_neg < 10:
        return {"ROC_AUC": None, "PR_AUC": None, "n_pos": n_pos, "n_neg": n_neg}
    return {
        "ROC_AUC": float(roc_auc_score(active, y_pred)),
        "PR_AUC": float(average_precision_score(active, y_pred)),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def tanimoto_stats(test_fps: list, train_fps: list) -> dict:
    maxima = np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
            for fp in test_fps
        ],
        dtype=float,
    )
    return {
        "mean": float(np.mean(maxima)),
        "median": float(np.median(maxima)),
        "frac_lt_0.4": float(np.mean(maxima < 0.4)),
        "frac_lt_0.6": float(np.mean(maxima < 0.6)),
        "frac_ge_0.8": float(np.mean(maxima >= 0.8)),
    }


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeated scaffold-split robustness study"
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

    print("building Morgan fingerprint matrix and bit vectors...", flush=True)
    X_all = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in smiles_all]
    ).astype(np.float32)
    bit_vectors_all = morgan_bit_vectors(smiles_all)

    rows: list[dict] = []
    for split_seed in SPLIT_SEEDS:
        print(f"\n--- scaffold split seed={split_seed} ---", flush=True)
        train_idx, val_idx, test_idx, scaffold_by_index, split_meta = (
            standard_scaffold_split_indices(
                smiles_all,
                train_frac=0.8,
                val_frac=0.1,
                test_frac=0.1,
                seed=split_seed,
            )
        )

        def scaffold_set(idx_list):
            return {scaffold_by_index[i] for i in idx_list}

        overlap = (
            len(scaffold_set(train_idx) & scaffold_set(test_idx))
            + len(scaffold_set(train_idx) & scaffold_set(val_idx))
            + len(scaffold_set(val_idx) & scaffold_set(test_idx))
        )
        assert overlap == 0
        print(
            f"  counts: train={len(train_idx)} val={len(val_idx)} "
            f"test={len(test_idx)} scaffold-overlap={overlap}"
        )

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]
        test_fps = [bit_vectors_all[i] for i in test_idx]
        train_fps = [bit_vectors_all[i] for i in train_idx]
        sim = tanimoto_stats(test_fps, train_fps)

        baseline = make_rf(BASELINE_MAX_FEATURES)
        tuned = make_rf(TUNED_MAX_FEATURES)
        print("  training baseline/tuned RF...", flush=True)
        baseline.fit(X_train, y_train)
        tuned.fit(X_train, y_train)
        p_base = baseline.predict(X_test)
        p_tuned = tuned.predict(X_test)

        m_base = {
            **regression_metrics(y_test, p_base),
            **screening_metrics(y_test, p_base),
        }
        m_tuned = {
            **regression_metrics(y_test, p_tuned),
            **screening_metrics(y_test, p_tuned),
        }
        active_frac = float((y_test >= ACTIVE_THRESHOLD).mean())
        row = {
            "split_seed": split_seed,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "test_n": len(test_idx),
            "train_scaffolds": len(scaffold_set(train_idx)),
            "val_scaffolds": len(scaffold_set(val_idx)),
            "test_scaffolds": len(scaffold_set(test_idx)),
            "test_mean_pIC50": float(np.mean(y_test)),
            "test_pIC50_std": float(np.std(y_test)),
            "test_active_fraction": active_frac,
            "mean_max_train_tanimoto": sim["mean"],
            "median_max_train_tanimoto": sim["median"],
            "frac_max_tanimoto_lt_0.4": sim["frac_lt_0.4"],
            "frac_max_tanimoto_lt_0.6": sim["frac_lt_0.6"],
            "frac_max_tanimoto_ge_0.8": sim["frac_ge_0.8"],
            "baseline_MAE": m_base["MAE"],
            "baseline_RMSE": m_base["RMSE"],
            "baseline_R2": m_base["R2"],
            "baseline_Spearman": m_base["Spearman"],
            "baseline_ROC_AUC": m_base["ROC_AUC"],
            "baseline_PR_AUC": m_base["PR_AUC"],
            "tuned_MAE": m_tuned["MAE"],
            "tuned_RMSE": m_tuned["RMSE"],
            "tuned_R2": m_tuned["R2"],
            "tuned_Spearman": m_tuned["Spearman"],
            "tuned_ROC_AUC": m_tuned["ROC_AUC"],
            "tuned_PR_AUC": m_tuned["PR_AUC"],
            "delta_MAE": m_tuned["MAE"] - m_base["MAE"],
            "delta_RMSE": m_tuned["RMSE"] - m_base["RMSE"],
            "delta_R2": m_tuned["R2"] - m_base["R2"],
            "delta_Spearman": m_tuned["Spearman"] - m_base["Spearman"],
        }
        rows.append(row)
        print(
            f"  baseline: MAE={m_base['MAE']:.4f} RMSE={m_base['RMSE']:.4f} "
            f"R2={m_base['R2']:.4f} Spearman={m_base['Spearman']:.4f}"
        )
        print(
            f"  tuned:   MAE={m_tuned['MAE']:.4f} RMSE={m_tuned['RMSE']:.4f} "
            f"R2={m_tuned['R2']:.4f} Spearman={m_tuned['Spearman']:.4f}"
        )
        print(
            f"  deltas:  R2={row['delta_R2']:+.4f} "
            f"Spearman={row['delta_Spearman']:+.4f} "
            f"RMSE={row['delta_RMSE']:+.4f} MAE={row['delta_MAE']:+.4f}"
        )
        print(
            f"  similarity: mean={sim['mean']:.4f} median={sim['median']:.4f} "
            f"<0.4={sim['frac_lt_0.4']:.3f} <0.6={sim['frac_lt_0.6']:.3f} "
            f">=0.8={sim['frac_ge_0.8']:.3f}"
        )

    out = pd.DataFrame(rows)
    csv_path = results_dir / "repeated_scaffold_split_results.csv"
    out.to_csv(csv_path, index=False)

    delta_spearman = [r["delta_Spearman"] for r in rows]
    delta_r2 = [r["delta_R2"] for r in rows]
    delta_rmse = [r["delta_RMSE"] for r in rows]
    delta_mae = [r["delta_MAE"] for r in rows]
    paired = {
        "delta_Spearman": {
            **summarize(delta_spearman),
            "n_splits_tuned_gt_baseline": int(
                np.sum(np.asarray(delta_spearman) > 0)
            ),
        },
        "delta_R2": {
            **summarize(delta_r2),
            "n_splits_tuned_gt_baseline": int(
                np.sum(np.asarray(delta_r2) > 0)
            ),
        },
        "delta_RMSE": {
            **summarize(delta_rmse),
            "n_splits_tuned_lt_baseline": int(
                np.sum(np.asarray(delta_rmse) < 0)
            ),
        },
        "delta_MAE": {
            **summarize(delta_mae),
            "n_splits_tuned_lt_baseline": int(
                np.sum(np.asarray(delta_mae) < 0)
            ),
        },
        "n_splits": len(rows),
        "interpretation_note": (
            "Paired deltas use the same split seed and the same RF "
            "random_state, so the comparison isolates scaffold-partition "
            "variation from RF stochasticity."
        ),
    }

    # Exploratory difficulty association (n=5, qualitative only).
    sim_mean = [r["mean_max_train_tanimoto"] for r in rows]
    difficulty = {
        "n_splits": len(rows),
        "note": "n=5; correlations are qualitative/exploratory only.",
        "Spearman_rho_mean_sim_vs_baseline_Spearman": float(
            spearmanr(sim_mean, [r["baseline_Spearman"] for r in rows])[0]
        ),
        "Spearman_rho_mean_sim_vs_tuned_Spearman": float(
            spearmanr(sim_mean, [r["tuned_Spearman"] for r in rows])[0]
        ),
        "Spearman_rho_mean_sim_vs_baseline_R2": float(
            spearmanr(sim_mean, [r["baseline_R2"] for r in rows])[0]
        ),
        "Spearman_rho_mean_sim_vs_tuned_R2": float(
            spearmanr(sim_mean, [r["tuned_R2"] for r in rows])[0]
        ),
    }

    summary = {
        "milestone": "repeated scaffold-split robustness study",
        "study_type": "data-partition robustness (not hyperparameter tuning)",
        "test_set_used": True,
        "fingerprint": {
            "type": "Morgan",
            "radius": RADIUS,
            "nBits": N_BITS,
        },
        "rf": {
            "n_estimators": N_ESTIMATORS,
            "random_state": RF_SEED,
            "min_samples_leaf": 1,
            "max_depth": None,
            "baseline_max_features": BASELINE_MAX_FEATURES,
            "tuned_max_features": TUNED_MAX_FEATURES,
        },
        "split_seeds": SPLIT_SEEDS,
        "canonical_split_note": (
            "split seed 42 reproduces the canonical standard scaffold split "
            "used for the locked headline result."
        ),
        "paired_analysis": paired,
        "difficulty_association": difficulty,
        "versions": package_versions(),
        "results": rows,
    }
    json_path = results_dir / "repeated_scaffold_split_summary.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {json_path}")

    # Paired plots.
    seeds = [r["split_seed"] for r in rows]
    base_sp = [r["baseline_Spearman"] for r in rows]
    tuned_sp = [r["tuned_Spearman"] for r in rows]
    base_r2 = [r["baseline_R2"] for r in rows]
    tuned_r2 = [r["tuned_R2"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for x, b, t in zip(seeds, base_sp, tuned_sp):
        ax.plot([x, x], [b, t], color="#999999", linewidth=1.5, zorder=1)
    ax.scatter(seeds, base_sp, marker="o", color="#C44E52", zorder=2, label="baseline (mf=1.0)")
    ax.scatter(seeds, tuned_sp, marker="s", color="#4C72B0", zorder=2, label="tuned (mf=0.20)")
    ax.set_xlabel("scaffold split seed")
    ax.set_ylabel("test Spearman")
    ax.set_title("Paired test Spearman across scaffold splits")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig1 = figures_dir / "repeated_scaffold_spearman.png"
    fig.savefig(fig1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for x, b, t in zip(seeds, base_r2, tuned_r2):
        ax.plot([x, x], [b, t], color="#999999", linewidth=1.5, zorder=1)
    ax.scatter(seeds, base_r2, marker="o", color="#C44E52", zorder=2, label="baseline (mf=1.0)")
    ax.scatter(seeds, tuned_r2, marker="s", color="#4C72B0", zorder=2, label="tuned (mf=0.20)")
    ax.set_xlabel("scaffold split seed")
    ax.set_ylabel("test R2")
    ax.set_title("Paired test R2 across scaffold splits")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig2 = figures_dir / "repeated_scaffold_r2.png"
    fig.savefig(fig2, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(sim_mean, base_sp, marker="o", color="#C44E52", label="baseline Spearman")
    ax.scatter(sim_mean, tuned_sp, marker="s", color="#4C72B0", label="tuned Spearman")
    for x, y_b, y_t, s in zip(sim_mean, base_sp, tuned_sp, seeds):
        ax.annotate(str(s), (x, y_b), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.annotate(str(s), (x, y_t), textcoords="offset points", xytext=(4, -9), fontsize=8)
    ax.set_xlabel("mean max_train_tanimoto")
    ax.set_ylabel("test Spearman")
    ax.set_title("Split difficulty vs test Spearman (n=5, exploratory)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig3 = figures_dir / "repeated_scaffold_similarity.png"
    fig.savefig(fig3, dpi=150)
    plt.close(fig)

    print(f"saved -> {fig1}")
    print(f"saved -> {fig2}")
    print(f"saved -> {fig3}")


if __name__ == "__main__":
    main()
