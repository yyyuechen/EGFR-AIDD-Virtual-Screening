#!/usr/bin/env python3
"""
Level 1 retrospective known-inhibitor recovery benchmark.

Scientific question: can known EGFR inhibitors that are completely withheld
as exact molecules from training still be preferentially ranked near the top
of a larger candidate pool?

Protocol
--------
1. Select experimentally supported named known inhibitors already present in
   the curated project dataset (repository evidence: the four positive
   controls used in M6.5/M7/M8).
2. Remove their exact standardized molecules from the canonical standard
   scaffold TRAIN split (Level 1: exact-molecule withholding only; same
   scaffold analogs remain in training).
3. Train the frozen final Morgan-RF configuration on the remaining training
   molecules.
4. Build a ranking pool from held-out molecules not used for model fitting:
   validation+test molecules with pIC50 < 6 as background, plus the four
   withheld positives. The pIC50 6-7 middle band is excluded from the
   primary benchmark.
5. Rank the pool and compute recovery, enrichment, ROC-AUC, PR-AUC, and
   same-scaffold diagnostics.

No model parameter is changed based on recovery metrics.

Outputs
-------
results/withheld_known_inhibitor_positive_set.csv
results/withheld_known_inhibitor_ranking_pool.csv
results/withheld_known_inhibitor_rankings.csv
results/withheld_known_inhibitor_recovery_summary.json
results/figures/withheld_known_inhibitor_rank_distribution.png
results/figures/withheld_known_inhibitor_recovery.png

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/withheld_known_inhibitor_recovery_level1.py
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
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
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


RADIUS = 2
N_BITS = 2048
N_ESTIMATORS = 300
SEED = 42
TUNED_MAX_FEATURES = 0.20
ACTIVE_MIN = 7.0
NEGATIVE_MAX = 6.0

# Repository-evidenced named controls (M6.5/M7/M8). No additional names exist
# in the local export metadata; osimertinib is explicitly absent.
POSITIVE_DEFS = [
    {
        "name": "afatinib",
        "chembl_id": "CHEMBL1173655",
        "source_identifier": "results/m65_improved_shortlist.csv + data/processed/egfr_activity_final.csv",
        "selection_reason": (
            "named approved EGFR inhibitor used as positive control in M6.5; "
            "curated pIC50 >= 7 with many measurements"
        ),
    },
    {
        "name": "erlotinib",
        "chembl_id": "CHEMBL553",
        "source_identifier": "results/m65_improved_shortlist.csv + data/processed/egfr_activity_final.csv",
        "selection_reason": (
            "named approved EGFR inhibitor used as positive control in M6.5; "
            "curated pIC50 >= 7 with many measurements"
        ),
    },
    {
        "name": "gefitinib",
        "chembl_id": "CHEMBL939",
        "source_identifier": "results/m65_improved_shortlist.csv + data/processed/egfr_activity_final.csv",
        "selection_reason": (
            "named approved EGFR inhibitor used as positive control in M6.5; "
            "curated pIC50 >= 7 with many measurements"
        ),
    },
    {
        "name": "lapatinib",
        "chembl_id": "CHEMBL554",
        "source_identifier": "results/m65_improved_shortlist.csv + data/processed/egfr_activity_final.csv",
        "selection_reason": (
            "named approved EGFR inhibitor used as positive control in M6.5; "
            "curated pIC50 >= 7 with many measurements"
        ),
    },
]


def make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=1,
        max_features=TUNED_MAX_FEATURES,
        max_depth=None,
    )


def morgan_bit_vectors(smiles_list: list[str]) -> list:
    generator = AllChem.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def inchi_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles}")
    return Chem.MolToInchiKey(mol)


def murcko(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def max_train_tanimoto(test_fps: list, train_fps: list) -> list[float]:
    return [
        float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
        for fp in test_fps
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Level 1 exact-molecule withholding recovery benchmark"
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
    ids_all = df["molecule_chembl_id"].tolist()
    smiles_to_idx = {s: i for i, s in enumerate(smiles_all)}

    # Build positive set from repository-evidenced controls present in data.
    positive_rows = []
    for defn in POSITIVE_DEFS:
        row = df[df["molecule_chembl_id"] == defn["chembl_id"]]
        if row.empty:
            raise SystemExit(f"positive {defn['name']} not found in curated data")
        r = row.iloc[0]
        smiles = str(r["canonical_smiles"])
        pic50 = float(r["pIC50"])
        if pic50 < ACTIVE_MIN:
            raise SystemExit(
                f"positive {defn['name']} has pIC50 {pic50:.3f} < {ACTIVE_MIN}"
            )
        positive_rows.append(
            {
                "compound_name_or_id": f"{defn['name']} ({defn['chembl_id']})",
                "name": defn["name"],
                "chembl_id": defn["chembl_id"],
                "canonical_smiles": smiles,
                "inchi_key": inchi_key(smiles),
                "experimental_pIC50": pic50,
                "n_measurements": int(r["n_measurements"]),
                "source_identifier": defn["source_identifier"],
                "selection_reason": defn["selection_reason"],
            }
        )

    positive_df = pd.DataFrame(positive_rows)
    assert positive_df["canonical_smiles"].is_unique
    positive_csv = results_dir / "withheld_known_inhibitor_positive_set.csv"
    positive_df.to_csv(positive_csv, index=False)
    print(f"saved -> {positive_csv}")
    print(positive_df[["compound_name_or_id", "experimental_pIC50", "n_measurements"]])

    # Standard split and exact withholding.
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
    positive_indices = [smiles_to_idx[s] for s in positive_df["canonical_smiles"]]
    final_train = [i for i in train_idx if i not in set(positive_indices)]
    final_train_smiles = [smiles_all[i] for i in final_train]
    final_train_set = set(final_train_smiles)

    # QC: no positive remains in training.
    leftover = [
        s for s in positive_df["canonical_smiles"] if s in final_train_set
    ]
    assert not leftover, f"positive molecules still in training: {leftover}"
    print(
        f"withholding: original train={len(train_idx)} "
        f"withheld_positives={len(positive_indices)} "
        f"final_train={len(final_train)}"
    )

    # Ranking pool: held-out val+test pIC50<6 background + withheld positives.
    holdout_idx = list(set(val_idx) | set(test_idx))
    negative_idx = [
        i for i in holdout_idx if y_all[i] < NEGATIVE_MAX
    ]
    negative_idx = [i for i in negative_idx if i not in set(positive_indices)]
    pool_indices = negative_idx + positive_indices
    pool_smiles = [smiles_all[i] for i in pool_indices]
    assert len(set(pool_smiles)) == len(pool_smiles), "duplicate pool structures"
    pool_roles = ["background"] * len(negative_idx) + ["positive"] * len(positive_indices)
    n_pool = len(pool_indices)
    n_pos = len(positive_indices)
    prevalence = n_pos / n_pool
    print(
        f"ranking pool: n={n_pool} positives={n_pos} background={len(negative_idx)} "
        f"prevalence={prevalence:.4f}"
    )

    pool_df = pd.DataFrame(
        {
            "molecule_chembl_id": [ids_all[i] for i in pool_indices],
            "canonical_smiles": pool_smiles,
            "true_pIC50": [float(y_all[i]) for i in pool_indices],
            "role": pool_roles,
        }
    )
    pool_csv = results_dir / "withheld_known_inhibitor_ranking_pool.csv"
    pool_df.to_csv(pool_csv, index=False)
    print(f"saved -> {pool_csv}")

    # Fingerprints, model, predictions.
    X_train = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in final_train_smiles]
    ).astype(np.float32)
    X_pool = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in pool_smiles]
    ).astype(np.float32)
    y_final_train = np.asarray([y_all[i] for i in final_train], dtype=np.float32)

    rf = make_rf()
    print("training frozen tuned RF on withheld training set...", flush=True)
    rf.fit(X_train, y_final_train)
    pred_pool = rf.predict(X_pool)

    order = np.lexsort((np.asarray(pool_smiles), -pred_pool))
    rank = np.empty(n_pool, dtype=int)
    rank[order] = np.arange(1, n_pool + 1)
    percentile_rank = 100.0 * rank / n_pool

    train_fps = morgan_bit_vectors(final_train_smiles)
    pool_fps = morgan_bit_vectors(pool_smiles)
    tanimoto = max_train_tanimoto(pool_fps, train_fps)

    train_scaffold_counts = Counter(
        murcko(smiles_all[i]) for i in final_train
    )
    pool_scaffolds = [murcko(s) for s in pool_smiles]
    same_scaffold_present = [
        sc in train_scaffold_counts for sc in pool_scaffolds
    ]
    same_scaffold_train_count = [
        train_scaffold_counts.get(sc, 0) for sc in pool_scaffolds
    ]

    rankings = pd.DataFrame(
        {
            "compound_name_or_id": [str(ids_all[i]) for i in pool_indices],
            "canonical_smiles": pool_smiles,
            "true_pIC50": [float(y_all[i]) for i in pool_indices],
            "known_positive": [r == "positive" for r in pool_roles],
            "predicted_pIC50": pred_pool,
            "rank": rank,
            "percentile_rank": percentile_rank,
            "max_train_tanimoto": tanimoto,
            "exact_training_match": [s in final_train_set for s in pool_smiles],
            "murcko_scaffold": pool_scaffolds,
            "same_scaffold_present_in_training": same_scaffold_present,
            "same_scaffold_train_count": same_scaffold_train_count,
        }
    )
    # Attach readable names for positives.
    name_map = {
        s: f"{r['name']} ({r['chembl_id']})"
        for s, r in zip(positive_df["canonical_smiles"], positive_rows)
    }
    rankings["compound_name_or_id"] = [
        name_map.get(s, str(ids_all[pool_indices[i]]))
        for i, s in enumerate(pool_smiles)
    ]
    rankings = rankings.sort_values("rank").reset_index(drop=True)
    ranking_csv = results_dir / "withheld_known_inhibitor_rankings.csv"
    rankings.to_csv(ranking_csv, index=False)
    print(f"saved -> {ranking_csv}")

    pos = rankings[rankings["known_positive"]].copy()
    pos_ranks = pos["rank"].astype(int).tolist()
    pos_percentiles = pos["percentile_rank"].astype(float).tolist()
    mean_pos_rank = float(np.mean(pos_ranks))
    median_pos_rank = float(np.median(pos_ranks))
    median_pos_percentile = float(np.median(pos_percentiles))

    labels = rankings["known_positive"].astype(int).to_numpy()
    scores = rankings["predicted_pIC50"].to_numpy()
    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    quant_spearman = float(spearmanr(
        rankings["true_pIC50"].to_numpy(), scores
    )[0])

    recovery = {}
    for fraction in [0.01, 0.05, 0.10]:
        k = int(math.ceil(fraction * n_pool))
        top = rankings.head(k)
        n_rec = int(top["known_positive"].sum())
        ef = (n_rec / k) / prevalence if k > 0 else 0.0
        theoretical_max = min(1.0 / prevalence, 1.0 / fraction)
        recovery[str(fraction)] = {
            "top_fraction": fraction,
            "top_k": k,
            "n_recovered": n_rec,
            "recall": n_rec / n_pos,
            "enrichment_factor": ef,
            "theoretical_max_enrichment_factor": theoretical_max,
        }
        print(
            f"  top {fraction:.0%}: k={k} recovered={n_rec}/{n_pos} "
            f"recall={n_rec/n_pos:.3f} EF={ef:.3f} maxEF={theoretical_max:.1f}"
        )

    sim_bins = {
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
    }
    print("positive similarity bins:", sim_bins)
    print(
        "same-scaffold in training:",
        {
            "n_positives_with_scaffold_in_train": int(
                pos["same_scaffold_present_in_training"].sum()
            ),
            "mean_same_scaffold_train_count": float(
                np.mean(pos["same_scaffold_train_count"])
            ),
        },
    )

    # Historical M6.5 comparison for the four named controls.
    historical = pd.read_csv("results/m65_improved_shortlist.csv")
    hist_controls = historical[historical["is_positive_control"]].sort_values(
        "rank"
    )
    historical_ranks = {
        str(row["name"]): int(row["rank"])
        for _, row in hist_controls.iterrows()
    }
    control_table = []
    for _, row in pos.iterrows():
        name = row["compound_name_or_id"].split(" (")[0]
        control_table.append(
            {
                "compound": name,
                "historical_M65_rank": historical_ranks.get(name),
                "new_rank": int(row["rank"]),
                "rank_percentile": float(row["percentile_rank"]),
                "same_scaffold_present_in_training": bool(
                    row["same_scaffold_present_in_training"]
                ),
                "same_scaffold_train_count": int(row["same_scaffold_train_count"]),
                "max_train_tanimoto": float(row["max_train_tanimoto"]),
                "predicted_pIC50": float(row["predicted_pIC50"]),
                "experimental_pIC50": float(row["true_pIC50"]),
            }
        )

    summary = {
        "milestone": "Level 1 exact-molecule withholding recovery benchmark",
        "level": 1,
        "note": (
            "Only exact-molecule withholding. Same-scaffold analogs remain in "
            "training; this is not a scaffold-generalization benchmark."
        ),
        "frozen_model": {
            "morgan": {"radius": RADIUS, "nBits": N_BITS},
            "random_forest": {
                "n_estimators": N_ESTIMATORS,
                "random_state": SEED,
                "n_jobs": -1,
                "min_samples_leaf": 1,
                "max_features": TUNED_MAX_FEATURES,
                "max_depth": None,
            },
        },
        "positive_set": {
            "n_positives": n_pos,
            "limitation": (
                "Only four high-confidence named inhibitors could be "
                "identified from existing repository resources; the curated "
                "data files contain no compound-name metadata and osimertinib "
                "is documented as absent from the local export. No labels "
                "were fabricated."
            ),
            "rows": positive_rows,
        },
        "training": {
            "original_train_n": len(train_idx),
            "withheld_positive_structures": n_pos,
            "final_train_n": len(final_train),
            "exact_training_match_after_withholding": False,
        },
        "ranking_pool": {
            "n_total": n_pool,
            "n_positive": n_pos,
            "n_background": len(negative_idx),
            "positive_prevalence": prevalence,
            "background_definition": "pIC50 < 6",
            "excluded_band": "6 <= pIC50 < 7",
            "pool_uses_only_held_out_molecules": True,
            "source": "standard scaffold validation+test molecules (not used for fitting)",
        },
        "ranking": {
            "mean_positive_rank": mean_pos_rank,
            "median_positive_rank": median_pos_rank,
            "median_positive_percentile": median_pos_percentile,
            "recovery": recovery,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "pr_auc_baseline_prevalence": prevalence,
            "spearman_pred_vs_experimental_pic50": quant_spearman,
            "spearman_caveat": (
                "Pool is threshold-constructed (positives >=7 vs background "
                "<6), so the quantitative Spearman partly reflects this "
                "construction and should not be compared with full-dataset "
                "regression Spearman."
            ),
        },
        "positive_level": pos[
            [
                "compound_name_or_id",
                "true_pIC50",
                "predicted_pIC50",
                "rank",
                "percentile_rank",
                "max_train_tanimoto",
                "same_scaffold_present_in_training",
                "same_scaffold_train_count",
            ]
        ].to_dict(orient="records"),
        "positive_similarity_bins": sim_bins,
        "same_scaffold_stats": {
            "n_positives_with_scaffold_in_train": int(
                pos["same_scaffold_present_in_training"].sum()
            ),
            "mean_same_scaffold_train_count": float(
                np.mean(pos["same_scaffold_train_count"])
            ),
            "max_same_scaffold_train_count": int(
                pos["same_scaffold_train_count"].max()
            ),
        },
        "historical_comparison": {
            "historical_model": (
                "M6.5 improved RandomForest trained on a candidate-like "
                "internal split with controls removed from the training pool"
            ),
            "historical_candidate_set_n": len(historical),
            "same_candidate_pool": False,
            "controls": control_table,
        },
        "versions": package_versions(),
    }
    json_path = results_dir / "withheld_known_inhibitor_recovery_summary.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"\nsaved -> {json_path}")

    # Figure 1: percentile rank distribution of withheld positives.
    fig, ax = plt.subplots(figsize=(6.0, 4))
    names = [r["compound_name_or_id"].split(" (")[0] for _, r in pos.iterrows()]
    colors = ["#4C72B0" if p <= 10 else "#55A868" if p <= 25 else "#C44E52"
              for p in pos_percentiles]
    ax.bar(names, pos_percentiles, color=colors)
    ax.axhline(10, color="#4C72B0", linestyle="--", linewidth=1, label="top 10%")
    ax.axhline(50, color="#888888", linestyle=":", linewidth=1, label="50th percentile")
    ax.set_ylabel("rank percentile (lower = better)")
    ax.set_title("Withheld known-inhibitor rank percentiles")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig1 = figures_dir / "withheld_known_inhibitor_rank_distribution.png"
    fig.savefig(fig1, dpi=150)
    plt.close(fig)

    # Figure 2: cumulative recovery curve.
    sorted_labels = labels[np.argsort(-scores)]
    cum = np.cumsum(sorted_labels) / n_pos
    x = np.arange(1, n_pool + 1) / n_pool
    fig, ax = plt.subplots(figsize=(6.0, 4))
    ax.plot(x, cum, color="#4C72B0", linewidth=2)
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", label="chance")
    for frac in [0.01, 0.05, 0.10]:
        k = int(math.ceil(frac * n_pool))
        ax.axvline(frac, color="#C44E52", linestyle=":", linewidth=1)
        ax.scatter([frac], [cum[k - 1]], color="#C44E52", zorder=3)
    ax.set_xlabel("top fraction of pool")
    ax.set_ylabel("fraction of withheld positives recovered")
    ax.set_title("Level 1 cumulative recovery")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig2 = figures_dir / "withheld_known_inhibitor_recovery.png"
    fig.savefig(fig2, dpi=150)
    plt.close(fig)
    print(f"saved -> {fig1}")
    print(f"saved -> {fig2}")


if __name__ == "__main__":
    main()
