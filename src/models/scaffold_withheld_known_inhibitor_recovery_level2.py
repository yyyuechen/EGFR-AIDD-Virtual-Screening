#!/usr/bin/env python3
"""
Level 2 retrospective known-inhibitor recovery: scaffold withholding.

Level 1 removed only the exact molecules of four known EGFR inhibitors and
showed strong recovery, but all four positives remained highly similar to
training chemistry (max_train_tanimoto >= 0.8) and their Murcko scaffolds
were still present in training.

Level 2 removes, for each positive:

  - the exact positive molecule; and
  - every training molecule sharing that positive's Bemis-Murcko scaffold.

Removal is performed jointly for the four positive scaffolds. The ranking
pool is exactly the Level 1 pool (475 molecules) so Level 1 vs Level 2
comparison is direct. The frozen final Morgan-RF configuration is reused
unchanged; no parameter is tuned.

Outputs
-------
results/scaffold_withheld_known_inhibitor_training_removals.csv
results/scaffold_withheld_known_inhibitor_rankings.csv
results/scaffold_withheld_known_inhibitor_summary.json
results/level1_vs_level2_known_inhibitor_comparison.csv
results/figures/level1_vs_level2_known_inhibitor_ranks.png
results/figures/level1_vs_level2_recovery.png

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/scaffold_withheld_known_inhibitor_recovery_level2.py
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
from sklearn.metrics import (
    average_precision_score,
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
from src.models.withheld_known_inhibitor_recovery_level1 import (
    make_rf,
    max_train_tanimoto,
    morgan_bit_vectors,
    murcko,
)


POSITIVE_SET_FILE = "results/withheld_known_inhibitor_positive_set.csv"
POOL_FILE = "results/withheld_known_inhibitor_ranking_pool.csv"
LEVEL1_RANKINGS_FILE = "results/withheld_known_inhibitor_rankings.csv"
LEVEL1_SUMMARY_FILE = "results/withheld_known_inhibitor_recovery_summary.json"


def recovery_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    prevalence: float,
) -> dict:
    n_pos = int(labels.sum())
    n_pool = len(labels)
    order = np.argsort(-scores)
    ranked_labels = labels[order]
    out = {}
    for fraction in [0.01, 0.05, 0.10]:
        k = int(math.ceil(fraction * n_pool))
        n_rec = int(ranked_labels[:k].sum())
        ef = (n_rec / k) / prevalence if k > 0 else 0.0
        theoretical_max = min(1.0 / prevalence, 1.0 / fraction)
        out[f"top_{int(fraction*100)}%"] = {
            "top_k": k,
            "n_recovered": n_rec,
            "recall": n_rec / n_pos,
            "enrichment_factor": ef,
            "theoretical_max_enrichment_factor": theoretical_max,
        }
    return {
        "n_pool": n_pool,
        "n_positive": n_pos,
        "prevalence": prevalence,
        "recovery": out,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Level 2 scaffold-withheld recovery benchmark"
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
    level1_rankings = pd.read_csv(LEVEL1_RANKINGS_FILE)

    assert len(positives) == 4
    assert len(pool) == 475
    assert pool["canonical_smiles"].is_unique
    print(
        f"reused positive set n={len(positives)}, ranking pool n={len(pool)} "
        f"(positives={int(pool['role'].eq('positive').sum())})"
    )

    df = pd.read_csv(args.data_file)
    smiles_all = df["canonical_smiles"].tolist()
    y_all = df["pIC50"].to_numpy(dtype=np.float32)
    ids_all = df["molecule_chembl_id"].tolist()
    smiles_to_idx = {s: i for i, s in enumerate(smiles_all)}

    train_idx, val_idx, test_idx, _, split_meta = (
        standard_scaffold_split_indices(
            smiles_all,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            seed=42,
        )
    )
    train_idx = list(train_idx)
    original_train_n = len(train_idx)

    # Scaffolds for all dataset molecules (cache).
    scaffold_cache: dict[str, str] = {}
    all_scaffolds = []
    for s in smiles_all:
        if s not in scaffold_cache:
            scaffold_cache[s] = murcko(s)
        all_scaffolds.append(scaffold_cache[s])
    train_scaffold_counter = Counter(
        all_scaffolds[i] for i in train_idx
    )

    # Joint scaffold removal.
    positive_smiles = positives["canonical_smiles"].tolist()
    removal_rows = []
    removed_sets = []
    for _, row in positives.iterrows():
        sc = scaffold_cache[row["canonical_smiles"]]
        remove_idx = [
            i for i in train_idx if all_scaffolds[i] == sc
        ]
        removed_sets.append(set(remove_idx))
        removal_rows.append(
            {
                "compound": row["compound_name_or_id"],
                "positive_Murcko_scaffold": sc,
                "training_molecules_removed_for_scaffold": len(remove_idx),
            }
        )
    union_removed = set().union(*removed_sets)
    overlap = {
        i: j
        for i in range(len(removed_sets))
        for j in range(i + 1, len(removed_sets))
        if removed_sets[i] & removed_sets[j]
    }
    final_train = [i for i in train_idx if i not in union_removed]
    final_train_smiles = [smiles_all[i] for i in final_train]
    final_train_set = set(final_train_smiles)
    final_train_scaffolds = [all_scaffolds[i] for i in final_train]
    final_train_scaffold_set = set(final_train_scaffolds)
    final_train_scaffold_counter = Counter(final_train_scaffolds)

    # QC: no exact positive or positive scaffold remains.
    leftover_exact = [s for s in positive_smiles if s in final_train_set]
    assert not leftover_exact
    leftover_scaffolds = [
        scaffold_cache[s] for s in positive_smiles
        if scaffold_cache[s] in final_train_scaffold_set
    ]
    assert not leftover_scaffolds

    removal_df = pd.DataFrame(removal_rows)
    removal_df["overlap_with_other_removed_scaffolds"] = [
        int(sum(1 for j in range(len(removed_sets)) if j != i and removed_sets[i] & removed_sets[j]))
        for i in range(len(removed_sets))
    ]
    removal_df["same_scaffold_remaining_after_removal"] = 0
    removal_df["exact_match_remaining"] = 0
    removal_csv = results_dir / "scaffold_withheld_known_inhibitor_training_removals.csv"
    removal_df.to_csv(removal_csv, index=False)
    print(
        f"removal: original_train={original_train_n} "
        f"unique_removed={len(union_removed)} final_train={len(final_train)} "
        f"overlap_pairs={len(overlap)}"
    )
    print(removal_df[["compound", "training_molecules_removed_for_scaffold"]])

    # Level 1 positive metrics for comparison.
    l1_pos = pd.DataFrame(level1_summary["positive_level"])
    l1_by_name = {
        r["compound_name_or_id"].split(" (")[0]: r
        for _, r in l1_pos.iterrows()
    }

    # New max_train_tanimoto to closest remaining training molecule.
    final_train_fps = morgan_bit_vectors(final_train_smiles)
    positive_fps = morgan_bit_vectors(positive_smiles)
    closest_rows = []
    for s, fp in zip(positive_smiles, positive_fps):
        sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fp, final_train_fps)
        )
        best = int(np.argmax(sims))
        closest_rows.append(
            {
                "canonical_smiles": s,
                "max_train_tanimoto_level2": float(sims[best]),
                "closest_remaining_smiles": final_train_smiles[best],
                "closest_remaining_scaffold": final_train_scaffolds[best],
            }
        )

    # Pool and model.
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
    print("training frozen RF on scaffold-withheld training set...", flush=True)
    rf.fit(X_train, y_train)
    pred_pool = rf.predict(X_pool)

    order = np.lexsort((np.asarray(pool_smiles), -pred_pool))
    rank = np.empty(n_pool, dtype=int)
    rank[order] = np.arange(1, n_pool + 1)
    percentile_rank = 100.0 * rank / n_pool
    pool_fps = morgan_bit_vectors(pool_smiles)
    tanimoto_l2 = max_train_tanimoto(pool_fps, final_train_fps)
    pool_scaffolds = [scaffold_cache[s] for s in pool_smiles]
    same_scaffold = [
        sc in final_train_scaffold_set for sc in pool_scaffolds
    ]
    same_scaffold_count = [
        final_train_scaffold_counter.get(sc, 0) for sc in pool_scaffolds
    ]

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
            "percentile_rank": percentile_rank,
            "known_positive": [r == "positive" for r in pool_roles],
            "max_train_tanimoto": tanimoto_l2,
            "exact_training_match": [s in final_train_set for s in pool_smiles],
            "same_scaffold_present_in_training": same_scaffold,
            "same_scaffold_train_count": same_scaffold_count,
            "murcko_scaffold": pool_scaffolds,
        }
    )
    rankings = rankings.sort_values("rank").reset_index(drop=True)
    ranking_csv = results_dir / "scaffold_withheld_known_inhibitor_rankings.csv"
    rankings.to_csv(ranking_csv, index=False)
    print(f"saved -> {ranking_csv}")

    pos = rankings[rankings["known_positive"]].copy()
    labels = rankings["known_positive"].astype(int).to_numpy()
    scores = rankings["predicted_pIC50"].to_numpy()
    metrics = recovery_metrics(labels, scores, prevalence)
    metrics["spearman_pred_vs_experimental_pic50"] = float(
        spearmanr(rankings["experimental_pIC50"].to_numpy(), scores)[0]
    )
    metrics["spearman_caveat"] = (
        "Pool is threshold-constructed (positives >=7 vs background <6)."
    )
    print("Level 2 recovery:", json_safe(metrics["recovery"]))
    print(
        f"Level 2 ROC-AUC={metrics['roc_auc']:.4f} "
        f"PR-AUC={metrics['pr_auc']:.4f} "
        f"Spearman={metrics['spearman_pred_vs_experimental_pic50']:.4f}"
    )

    # Positive-level Level1 vs Level2 table.
    comparison_rows = []
    for _, row in positives.iterrows():
        name = row["compound_name_or_id"].split(" (")[0]
        l1 = l1_by_name[name]
        l2 = pos[pos["compound_name_or_id"].str.startswith(name)].iloc[0]
        comparison_rows.append(
            {
                "compound": name,
                "experimental_pIC50": float(l1["true_pIC50"]),
                "level1_predicted_pIC50": float(l1["predicted_pIC50"]),
                "level1_rank": int(l1["rank"]),
                "level1_percentile": float(l1["percentile_rank"]),
                "level1_max_train_tanimoto": float(l1["max_train_tanimoto"]),
                "level2_predicted_pIC50": float(l2["predicted_pIC50"]),
                "level2_rank": int(l2["rank"]),
                "level2_percentile": float(l2["percentile_rank"]),
                "level2_max_train_tanimoto": float(l2["max_train_tanimoto"]),
                "rank_change_level2_minus_level1": int(l2["rank"]) - int(l1["rank"]),
                "predicted_pIC50_change": float(l2["predicted_pIC50"])
                - float(l1["predicted_pIC50"]),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = results_dir / "level1_vs_level2_known_inhibitor_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"saved -> {comparison_csv}")
    print(comparison_df)

    # Aggregate Level1 metrics from the Level1 summary JSON.
    l1_rec = level1_summary["ranking"]["recovery"]
    l1_agg = {
        "median_positive_rank": level1_summary["ranking"]["median_positive_rank"],
        "mean_positive_rank": level1_summary["ranking"]["mean_positive_rank"],
        "median_positive_max_train_tanimoto": float(
            np.median([r["max_train_tanimoto"] for r in l1_pos.to_dict("records")])
        ),
    }
    l2_agg = {
        "median_positive_rank": float(np.median(pos["rank"])),
        "mean_positive_rank": float(np.mean(pos["rank"])),
        "median_positive_max_train_tanimoto": float(
            np.median(pos["max_train_tanimoto"])
        ),
    }
    metric_rows = []
    for frac, key in [("0.01", "top_1%"), ("0.05", "top_5%"), ("0.1", "top_10%")]:
        for metric in ["recall", "enrichment_factor"]:
            l1v = l1_rec[frac][metric]
            l2v = metrics["recovery"][key][metric]
            metric_rows.append(
                {
                    "metric": f"{metric}_{int(float(frac)*100)}%",
                    "level1": l1v,
                    "level2": l2v,
                    "change": l2v - l1v,
                }
            )
    for metric in ["roc_auc", "pr_auc"]:
        l1v = level1_summary["ranking"][metric]
        l2v = metrics[metric]
        metric_rows.append(
            {
                "metric": metric,
                "level1": l1v,
                "level2": l2v,
                "change": l2v - l1v,
            }
        )
    for metric in ["median_positive_rank", "mean_positive_rank",
                   "median_positive_max_train_tanimoto"]:
        metric_rows.append(
            {
                "metric": metric,
                "level1": l1_agg[metric],
                "level2": l2_agg[metric],
                "change": l2_agg[metric] - l1_agg[metric],
            }
        )
    metric_df = pd.DataFrame(metric_rows)
    metric_csv = results_dir / "level1_vs_level2_known_inhibitor_metrics.csv"
    metric_df.to_csv(metric_csv, index=False)

    summary = {
        "milestone": "Level 2 scaffold-withheld known-inhibitor recovery",
        "level": 2,
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
        "training": {
            "original_train_n": original_train_n,
            "unique_molecules_removed": len(union_removed),
            "final_train_n": len(final_train),
            "removed_scaffold_overlap_pairs": len(overlap),
            "exact_training_match_remaining": 0,
            "same_scaffold_remaining": 0,
        },
        "training_removals": removal_df.to_dict(orient="records"),
        "closest_remaining_training_molecule": closest_rows,
        "positive_level_comparison": comparison_df.to_dict(orient="records"),
        "level2_metrics": metrics,
        "level1_vs_level2_metrics": metric_df.to_dict(orient="records"),
        "positive_similarity_bins_level2": {
            ">=0.8": int((pos["max_train_tanimoto"] >= 0.8).sum()),
            "0.6-<0.8": int(
                ((pos["max_train_tanimoto"] >= 0.6)
                 & (pos["max_train_tanimoto"] < 0.8)).sum()
            ),
            "0.4-<0.6": int(
                ((pos["max_train_tanimoto"] >= 0.4)
                 & (pos["max_train_tanimoto"] < 0.6)).sum()
            ),
            "<0.4": int((pos["max_train_tanimoto"] < 0.4).sum()),
        },
        "versions": package_versions(),
    }
    json_path = results_dir / "scaffold_withheld_known_inhibitor_summary.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"saved -> {json_path}")

    # Figure 1: paired ranks L1 vs L2.
    names = [r["compound"] for r in comparison_rows]
    l1r = [r["level1_rank"] for r in comparison_rows]
    l2r = [r["level2_rank"] for r in comparison_rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for xi, a, b in zip(x, l1r, l2r):
        ax.plot([xi, xi], [a, b], color="#999999", linewidth=1.5, zorder=1)
    ax.scatter(x, l1r, marker="o", color="#C44E52", zorder=2, label="Level 1")
    ax.scatter(x, l2r, marker="s", color="#4C72B0", zorder=2, label="Level 2")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("rank in 475-molecule pool")
    ax.set_title("Known-inhibitor ranks: Level 1 vs Level 2")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig1 = figures_dir / "level1_vs_level2_known_inhibitor_ranks.png"
    fig.savefig(fig1, dpi=150)
    plt.close(fig)

    # Figure 2: cumulative recovery curves L1 vs L2.
    l1_ranked = level1_rankings.sort_values("rank")
    l1_cum = np.cumsum(l1_ranked["known_positive"].astype(int).to_numpy()) / n_pos
    l2_cum = np.cumsum(labels[np.argsort(-scores)]) / n_pos
    xfrac = np.arange(1, n_pool + 1) / n_pool
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xfrac, l1_cum, color="#C44E52", linewidth=2, label="Level 1 (exact withholding)")
    ax.plot(xfrac, l2_cum, color="#4C72B0", linewidth=2, label="Level 2 (scaffold withholding)")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", label="chance")
    ax.set_xlabel("top fraction of pool")
    ax.set_ylabel("fraction of positives recovered")
    ax.set_title("Level 1 vs Level 2 cumulative recovery")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig2 = figures_dir / "level1_vs_level2_recovery.png"
    fig.savefig(fig2, dpi=150)
    plt.close(fig)
    print(f"saved -> {fig1}")
    print(f"saved -> {fig2}")


if __name__ == "__main__":
    main()
