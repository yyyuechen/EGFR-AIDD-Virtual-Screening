#!/usr/bin/env python3
"""
Level 3 retrospective known-inhibitor recovery: low-similarity withholding.

Level 1 removed exact molecules; Level 2 additionally removed same-Murcko-
scaffold training molecules. Level 3 goes further: from the original
standard-scaffold training set, remove every molecule with Morgan Tanimoto
>= 0.60 to any of the four known positives (plus exact molecules and same-
scaffold molecules), so each positive ends with max_train_tanimoto < 0.60.

The benchmark is split into:

  Phase 3A: feasibility audit before training (no model trained yet);
  Phase 3B: frozen RF retraining and ranking of the identical 475-molecule
            pool.

The frozen final Morgan-RF configuration is used unchanged.

Outputs
-------
results/level3_low_similarity_feasibility.csv
results/level3_low_similarity_training_removals.csv
results/low_similarity_withheld_known_inhibitor_rankings.csv
results/withheld_known_inhibitor_level3_summary.json
results/level1_level2_level3_comparison.csv
results/figures/level1_level2_level3_ranks.png
results/figures/level1_level2_level3_recovery.png

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/low_similarity_withheld_known_inhibitor_recovery_level3.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import DataStructs
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.evaluate_splits import (
    json_safe,
    package_versions,
    smiles_to_morgan_fingerprint,
)
from src.models.portfolio_hardening_analysis import (
    standard_scaffold_split_indices,
)
from src.models.scaffold_withheld_known_inhibitor_recovery_level2 import (
    recovery_metrics,
)
from src.models.withheld_known_inhibitor_recovery_level1 import (
    make_rf,
    max_train_tanimoto,
    morgan_bit_vectors,
    murcko,
)


POSITIVE_SET_FILE = "results/withheld_known_inhibitor_positive_set.csv"
POOL_FILE = "results/withheld_known_inhibitor_ranking_pool.csv"
LEVEL1_SUMMARY_FILE = "results/withheld_known_inhibitor_recovery_summary.json"
LEVEL2_COMPARISON_FILE = "results/level1_vs_level2_known_inhibitor_comparison.csv"
LEVEL1_RANKINGS_FILE = "results/withheld_known_inhibitor_rankings.csv"
LEVEL2_RANKINGS_FILE = "results/scaffold_withheld_known_inhibitor_rankings.csv"
SIM_THRESHOLD = 0.60


def scaffold_diversity(scaffolds: list[str]) -> dict:
    counter = Counter(scaffolds)
    sizes = np.asarray(list(counter.values()), dtype=float)
    return {
        "n_scaffolds": len(counter),
        "mean_molecules_per_scaffold": float(np.mean(sizes)),
        "median_molecules_per_scaffold": float(np.median(sizes)),
        "max_molecules_per_scaffold": int(sizes.max()) if len(sizes) else 0,
        "n_singleton_scaffolds": int((sizes == 1).sum()),
    }


def distribution_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "active_fraction": float((values >= 7.0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Level 3 low-similarity withholding recovery benchmark"
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

    positives = pd.read_csv(POSITIVE_SET_FILE)
    pool = pd.read_csv(POOL_FILE)
    level1_summary = json.loads(Path(LEVEL1_SUMMARY_FILE).read_text())
    level2_comparison = pd.read_csv(LEVEL2_COMPARISON_FILE)
    level1_rankings = pd.read_csv(LEVEL1_RANKINGS_FILE)
    level2_rankings = pd.read_csv(LEVEL2_RANKINGS_FILE)
    assert len(positives) == 4 and len(pool) == 475
    print(
        f"reused positives n={len(positives)}, pool n={len(pool)} "
        f"(threshold={SIM_THRESHOLD})"
    )

    df = pd.read_csv(args.data_file)
    smiles_all = df["canonical_smiles"].tolist()
    y_all = df["pIC50"].to_numpy(dtype=np.float32)
    ids_all = df["molecule_chembl_id"].tolist()
    smiles_to_idx = {s: i for i, s in enumerate(smiles_all)}

    train_idx, _, _, _, split_meta = standard_scaffold_split_indices(
        smiles_all,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        seed=42,
    )
    train_idx = list(train_idx)
    original_train_n = len(train_idx)
    original_train_smiles = [smiles_all[i] for i in train_idx]
    original_train_y = np.asarray([y_all[i] for i in train_idx], dtype=float)
    original_train_scaffolds = [murcko(s) for s in original_train_smiles]
    original_train_fps = morgan_bit_vectors(original_train_smiles)

    positive_smiles = positives["canonical_smiles"].tolist()
    positive_fps = morgan_bit_vectors(positive_smiles)
    scaffold_cache = {s: murcko(s) for s in positive_smiles}

    # Phase 3A: per-positive feasibility.
    feasibility_rows = []
    sim_sets = []
    for s, fp in zip(positive_smiles, positive_fps):
        sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fp, original_train_fps)
        )
        sim_idx = {train_idx[i] for i in np.where(sims >= SIM_THRESHOLD)[0]}
        sc = scaffold_cache[s]
        scaffold_idx = {
            train_idx[i]
            for i, sc_i in enumerate(original_train_scaffolds)
            if sc_i == sc
        }
        sim_sets.append(sim_idx)
        feasibility_rows.append(
            {
                "compound": positives.loc[
                    positives["canonical_smiles"] == s, "compound_name_or_id"
                ].iloc[0],
                "n_train_total": original_train_n,
                "n_sim_ge_0.60": len(sim_idx),
                "n_same_scaffold": len(scaffold_idx),
                "n_scaffold_with_sim_ge_0.60": len(sim_idx & scaffold_idx),
                "n_exact": 1,
            }
        )
    union_sim = set().union(*sim_sets)
    union_scaffold = {
        train_idx[i]
        for i, sc_i in enumerate(original_train_scaffolds)
        if sc_i in scaffold_cache.values()
    }
    union_removed = union_sim | union_scaffold
    removed_n = len(union_removed)
    final_train = [i for i in train_idx if i not in union_removed]
    final_train_n = len(final_train)
    removed_fraction = removed_n / original_train_n

    final_train_smiles = [smiles_all[i] for i in final_train]
    final_train_set = set(final_train_smiles)
    final_train_y = np.asarray([y_all[i] for i in final_train], dtype=float)
    final_train_scaffolds = [murcko(s) for s in final_train_smiles]

    feasibility_summary = {
        "original_train_n": original_train_n,
        "removed_n": removed_n,
        "removed_fraction": removed_fraction,
        "final_train_n": final_train_n,
        "sim_threshold": SIM_THRESHOLD,
        "removed_by_sim_union_n": len(union_sim),
        "removed_by_scaffold_union_n": len(union_scaffold),
        "distribution_before": distribution_stats(original_train_y),
        "distribution_after": distribution_stats(final_train_y),
        "scaffold_diversity_before": scaffold_diversity(original_train_scaffolds),
        "scaffold_diversity_after": scaffold_diversity(final_train_scaffolds),
    }
    feasibility_df = pd.DataFrame(feasibility_rows)
    feasibility_csv = results_dir / "level3_low_similarity_feasibility.csv"
    feasibility_df.to_csv(feasibility_csv, index=False)
    print(f"saved -> {feasibility_csv}")
    print("Phase 3A feasibility:")
    print(pd.Series(feasibility_summary))
    print(feasibility_df)

    # Phase 3B verification: max_train_tanimoto < 0.60 for all positives.
    final_train_fps = morgan_bit_vectors(final_train_smiles)
    l3_sims = max_train_tanimoto(positive_fps, final_train_fps)
    assert all(s < SIM_THRESHOLD for s in l3_sims), "filtering violation"
    leftover_exact = [s for s in positive_smiles if s in final_train_set]
    assert not leftover_exact
    leftover_scaffolds = [
        sc for sc in scaffold_cache.values() if sc in set(final_train_scaffolds)
    ]
    assert not leftover_scaffolds
    print(
        f"Phase 3B: final_train={final_train_n} "
        f"positive max sims={[round(s, 4) for s in l3_sims]}"
    )

    # Closest remaining training molecule per positive.
    closest_rows = []
    for s, fp, sim_l3 in zip(positive_smiles, positive_fps, l3_sims):
        sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fp, final_train_fps)
        )
        best = int(np.argmax(sims))
        closest_smiles = final_train_smiles[best]
        closest_rows.append(
            {
                "canonical_smiles": s,
                "max_train_tanimoto_level3": float(sims[best]),
                "closest_remaining_smiles": closest_smiles,
                "closest_remaining_scaffold": final_train_scaffolds[best],
                "closest_remaining_pIC50": float(y_all[final_train[best]]),
            }
        )
    closest_df = pd.DataFrame(closest_rows)
    removal_df = feasibility_df.copy()
    removal_df["union_removed_after_joint_filtering"] = len(union_removed)
    removal_df["final_train_n"] = final_train_n
    removal_csv = results_dir / "level3_low_similarity_training_removals.csv"
    removal_df.to_csv(removal_csv, index=False)
    print(f"saved -> {removal_csv}")

    # Rank the identical pool with the frozen model.
    pool_smiles = pool["canonical_smiles"].tolist()
    pool_roles = pool["role"].tolist()
    pool_true = pool["true_pIC50"].to_numpy(dtype=float)
    n_pool = len(pool_smiles)
    n_pos = int((np.asarray(pool_roles) == "positive").sum())
    prevalence = n_pos / n_pool

    X_train = np.vstack(
        [smiles_to_morgan_fingerprint(s) for s in final_train_smiles]
    ).astype(np.float32)
    X_pool = np.vstack(
        [smiles_to_morgan_fingerprint(s) for s in pool_smiles]
    ).astype(np.float32)
    y_train = np.asarray([y_all[i] for i in final_train], dtype=np.float32)
    rf = make_rf()
    print("training frozen RF on low-similarity-withheld training set...",
          flush=True)
    rf.fit(X_train, y_train)
    pred_pool = rf.predict(X_pool)

    order = np.lexsort((np.asarray(pool_smiles), -pred_pool))
    rank = np.empty(n_pool, dtype=int)
    rank[order] = np.arange(1, n_pool + 1)
    percentile = 100.0 * rank / n_pool
    pool_fps = morgan_bit_vectors(pool_smiles)
    pool_sim = max_train_tanimoto(pool_fps, final_train_fps)
    pool_scaffolds = [murcko(s) for s in pool_smiles]
    name_map = {
        s: f"{r['name']} ({r['chembl_id']})"
        for s, r in zip(
            positives["canonical_smiles"],
            positives.to_dict(orient="records"),
        )
    }
    rankings = pd.DataFrame(
        {
            "compound_name_or_id": [
                name_map.get(s, str(ids_all[smiles_to_idx[s]]))
                for s in pool_smiles
            ],
            "canonical_smiles": pool_smiles,
            "experimental_pIC50": pool_true,
            "predicted_pIC50": pred_pool,
            "rank": rank,
            "percentile": percentile,
            "known_positive": [r == "positive" for r in pool_roles],
            "max_train_tanimoto": pool_sim,
            "exact_training_match": [s in final_train_set for s in pool_smiles],
            "same_scaffold_present_in_training": [
                sc in set(final_train_scaffolds) for sc in pool_scaffolds
            ],
            "murcko_scaffold": pool_scaffolds,
        }
    )
    rankings = rankings.sort_values("rank").reset_index(drop=True)
    ranking_csv = results_dir / "low_similarity_withheld_known_inhibitor_rankings.csv"
    rankings.to_csv(ranking_csv, index=False)
    print(f"saved -> {ranking_csv}")

    labels = rankings["known_positive"].astype(int).to_numpy()
    scores = rankings["predicted_pIC50"].to_numpy()
    metrics = recovery_metrics(labels, scores, prevalence)
    metrics["spearman_pred_vs_experimental_pic50"] = float(
        spearmanr(rankings["experimental_pIC50"].to_numpy(), scores)[0]
    )
    metrics["mean_positive_rank"] = float(
        np.mean(rankings.loc[rankings["known_positive"], "rank"])
    )
    metrics["median_positive_rank"] = float(
        np.median(rankings.loc[rankings["known_positive"], "rank"])
    )
    print("Level 3 recovery:", json_safe(metrics["recovery"]))
    print(
        f"Level 3 ROC-AUC={metrics['roc_auc']:.4f} "
        f"PR-AUC={metrics['pr_auc']:.4f} "
        f"Spearman={metrics['spearman_pred_vs_experimental_pic50']:.4f}"
    )

    # Three-level positive comparison.
    l1_pos = pd.DataFrame(level1_summary["positive_level"])
    l2_map = {
        r["compound"]: r
        for _, r in level2_comparison.iterrows()
    }
    comparison_rows = []
    for _, row in positives.iterrows():
        name = row["compound_name_or_id"].split(" (")[0]
        l1 = l1_pos[l1_pos["compound_name_or_id"].str.startswith(name)].iloc[0]
        l2 = l2_map[name]
        l3 = rankings[rankings["compound_name_or_id"].str.startswith(name)].iloc[0]
        comparison_rows.append(
            {
                "compound": name,
                "experimental_pIC50": float(l1["true_pIC50"]),
                "L1_pred": float(l1["predicted_pIC50"]),
                "L1_rank": int(l1["rank"]),
                "L1_percentile": float(l1["percentile_rank"]),
                "L1_max_sim": float(l1["max_train_tanimoto"]),
                "L2_pred": float(l2["level2_predicted_pIC50"]),
                "L2_rank": int(l2["level2_rank"]),
                "L2_percentile": float(l2["level2_percentile"]),
                "L2_max_sim": float(l2["level2_max_train_tanimoto"]),
                "L3_pred": float(l3["predicted_pIC50"]),
                "L3_rank": int(l3["rank"]),
                "L3_percentile": float(l3["percentile"]),
                "L3_max_sim": float(l3["max_train_tanimoto"]),
                "L3_minus_L2_rank": int(l3["rank"]) - int(l2["level2_rank"]),
                "L3_minus_L1_rank": int(l3["rank"]) - int(l1["rank"]),
                "L3_pred_minus_L2_pred": float(l3["predicted_pIC50"])
                - float(l2["level2_predicted_pIC50"]),
                "L3_pred_minus_L1_pred": float(l3["predicted_pIC50"])
                - float(l1["predicted_pIC50"]),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = results_dir / "level1_level2_level3_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"saved -> {comparison_csv}")

    # Aggregate level metrics.
    l1_rec = level1_summary["ranking"]["recovery"]
    l2_metrics = json.loads(
        Path(results_dir / "scaffold_withheld_known_inhibitor_summary.json").read_text()
    )["level2_metrics"]
    l2_rec = {
        key: value
        for key, value in l2_metrics["recovery"].items()
    }
    l2_pos_median = float(np.median(
        [r["level2_rank"] for r in l2_map.values()]
    ))
    l2_pos_median_sim = float(np.median(
        [r["level2_max_train_tanimoto"] for r in l2_map.values()]
    ))
    l1_median_rank = level1_summary["ranking"]["median_positive_rank"]
    l1_median_sim = float(np.median(
        [r["max_train_tanimoto"] for r in l1_pos.to_dict("records")]
    ))
    l3_median_rank = float(np.median(
        rankings.loc[rankings["known_positive"], "rank"]
    ))
    l3_median_sim = float(np.median(
        rankings.loc[rankings["known_positive"], "max_train_tanimoto"]
    ))

    metric_rows = []
    for frac, l1key, l2key in [
        ("1%", "0.01", "top_1%"),
        ("5%", "0.05", "top_5%"),
        ("10%", "0.1", "top_10%"),
    ]:
        for metric in ["recall", "enrichment_factor"]:
            metric_rows.append(
                {
                    "metric": f"{metric}_{frac}",
                    "level1": l1_rec[l1key][metric],
                    "level2": l2_rec[l2key][metric],
                    "level3": metrics["recovery"][l2key][metric],
                }
            )
    for metric in ["roc_auc", "pr_auc"]:
        metric_rows.append(
            {
                "metric": metric,
                "level1": level1_summary["ranking"][metric],
                "level2": l2_metrics[metric],
                "level3": metrics[metric],
            }
        )
    metric_rows.append(
        {
            "metric": "median_positive_rank",
            "level1": l1_median_rank,
            "level2": l2_pos_median,
            "level3": l3_median_rank,
        }
    )
    metric_rows.append(
        {
            "metric": "median_positive_max_train_tanimoto",
            "level1": l1_median_sim,
            "level2": l2_pos_median_sim,
            "level3": l3_median_sim,
        }
    )
    metric_df = pd.DataFrame(metric_rows)
    metric_csv = results_dir / "level1_level2_level3_metrics.csv"
    metric_df.to_csv(metric_csv, index=False)

    l3_sims_arr = np.asarray(l3_sims, dtype=float)
    l3_sim_bins = {
        "0.5-<0.6": int(((l3_sims_arr >= 0.5) & (l3_sims_arr < 0.6)).sum()),
        "0.4-<0.5": int(((l3_sims_arr >= 0.4) & (l3_sims_arr < 0.5)).sum()),
        "<0.4": int((l3_sims_arr < 0.4).sum()),
    }
    summary = {
        "milestone": "Level 3 low-similarity withholding recovery",
        "level": 3,
        "sim_threshold": SIM_THRESHOLD,
        "frozen_model": {
            "morgan": {"radius": 2, "nBits": 2048},
            "random_forest": {
                "n_estimators": 300,
                "random_state": 42,
                "n_jobs": -1,
                "min_samples_leaf": 1,
                "max_features": 0.20,
                "max_depth": None,
            },
        },
        "positive_set_reused": len(positives),
        "ranking_pool_reused": len(pool),
        "pool_prevalence": prevalence,
        "feasibility": feasibility_summary,
        "per_positive_feasibility": feasibility_df.to_dict(orient="records"),
        "training": {
            "original_train_n": original_train_n,
            "removed_n": removed_n,
            "removed_fraction": removed_fraction,
            "final_train_n": final_train_n,
            "exact_training_match_remaining": 0,
            "same_scaffold_remaining": 0,
            "all_positive_max_sim_lt_0.60": True,
        },
        "closest_remaining_training_molecule": closest_df.to_dict(orient="records"),
        "positive_similarity_bins_level3": l3_sim_bins,
        "positive_level_comparison": comparison_df.to_dict(orient="records"),
        "level3_metrics": metrics,
        "level1_level2_level3_metrics": metric_df.to_dict(orient="records"),
        "versions": package_versions(),
    }
    json_path = results_dir / "withheld_known_inhibitor_level3_summary.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"saved -> {json_path}")

    # Figure: three-level ranks per inhibitor.
    names = [r["compound"] for r in comparison_rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 4))
    for i, name in enumerate(names):
        r = comparison_rows[i]
        ax.plot(
            [0, 1, 2], [r["L1_rank"], r["L2_rank"], r["L3_rank"]],
            marker="o", color="#999999", linewidth=1.2, zorder=1,
        )
    ax.scatter(
        np.full(len(names), 0), [r["L1_rank"] for r in comparison_rows],
        marker="o", color="#C44E52", zorder=2, label="Level 1",
    )
    ax.scatter(
        np.full(len(names), 1), [r["L2_rank"] for r in comparison_rows],
        marker="s", color="#4C72B0", zorder=2, label="Level 2",
    )
    ax.scatter(
        np.full(len(names), 2), [r["L3_rank"] for r in comparison_rows],
        marker="D", color="#55A868", zorder=2, label="Level 3",
    )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["L1", "L2", "L3"])
    ax.set_ylabel("rank in 475-molecule pool")
    ax.set_title("Known-inhibitor ranks across withholding levels")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig1 = figures_dir / "level1_level2_level3_ranks.png"
    fig.savefig(fig1, dpi=150)
    plt.close(fig)

    # Figure: cumulative recovery curves.
    l1_cum = np.cumsum(
        level1_rankings.sort_values("rank")["known_positive"].astype(int)
    ) / n_pos
    l2_cum = np.cumsum(
        level2_rankings.sort_values("rank")["known_positive"].astype(int)
    ) / n_pos
    l3_cum = np.cumsum(labels[np.argsort(-scores)]) / n_pos
    xfrac = np.arange(1, n_pool + 1) / n_pool
    fig, ax = plt.subplots(figsize=(7.0, 4))
    ax.plot(xfrac, l1_cum, color="#C44E52", label="Level 1")
    ax.plot(xfrac, l2_cum, color="#4C72B0", label="Level 2")
    ax.plot(xfrac, l3_cum, color="#55A868", label="Level 3")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", label="chance")
    ax.set_xlabel("top fraction of pool")
    ax.set_ylabel("fraction of positives recovered")
    ax.set_title("Cumulative recovery by withholding level")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig2 = figures_dir / "level1_level2_level3_recovery.png"
    fig.savefig(fig2, dpi=150)
    plt.close(fig)
    print(f"saved -> {fig1}")
    print(f"saved -> {fig2}")


if __name__ == "__main__":
    main()
