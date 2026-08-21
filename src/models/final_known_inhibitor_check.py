#!/usr/bin/env python3
"""
Final biological sanity check for the locked Morgan-RF model.

This is NOT hyperparameter tuning. The frozen configurations are:

  - baseline: max_features = 1.0;
  - tuned: max_features = 0.20;
  - identical otherwise: Morgan radius 2 / 2048 bits, 300 trees,
    min_samples_leaf = 1, max_depth = None, random_state = 42.

The candidate set is the exact 64-molecule set used by the earlier M6.5
known-inhibitor / candidate-like ranking experiment:

  results/m65_improved_shortlist.csv  (60 candidate-like molecules + 4
  positive controls: afatinib, erlotinib, gefitinib, lapatinib)

Both models are trained on the canonical standard-scaffold TRAIN split
(8,128 molecules) and predict the 64 candidates. Ranks use deterministic
tie-breaking (higher predicted pIC50 first, then canonical SMILES). For each
candidate we report exact training-set overlap and max_train_tanimoto so
training-like high rankings can be distinguished from novel ones.

Outputs
-------
results/final_known_inhibitor_ranking.csv
results/final_known_inhibitor_summary.json
results/figures/final_known_inhibitor_ranks.png

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/final_known_inhibitor_check.py
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
from sklearn.ensemble import RandomForestRegressor

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
BASELINE_MAX_FEATURES = 1.0
TUNED_MAX_FEATURES = 0.20
CANDIDATE_FILE = "results/m65_improved_shortlist.csv"
FAIR_COMPARISON_FILE = "results/m65_fair_comparison_predictions.csv"
KNOWN_CONTROL_NAMES = ["afatinib", "erlotinib", "gefitinib", "lapatinib"]


def make_rf(max_features: float) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=1,
        max_features=max_features,
        max_depth=None,
    )


def morgan_bit_vectors(smiles_list: list[str]) -> list:
    generator = AllChem.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def max_train_tanimoto(test_fps: list, train_fps: list) -> list[float]:
    return [
        float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
        for fp in test_fps
    ]


def rank_frame(
    ids: list[str],
    smiles: list[str],
    names: list[str | None],
    controls: list[bool],
    preds: np.ndarray,
) -> pd.DataFrame:
    """Rank by predicted pIC50 desc, canonical SMILES asc for determinism."""
    order = np.lexsort((np.asarray(smiles), -preds))
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    return pd.DataFrame(
        {
            "molecule_chembl_id": ids,
            "canonical_smiles": smiles,
            "name": names,
            "is_positive_control": controls,
            "predicted_pIC50": preds,
            "rank": rank,
        }
    )


def applicability_bin(sim: float) -> str:
    if sim >= 0.8:
        return ">=0.8 (high / interpolation-like)"
    if sim >= 0.6:
        return "0.6-<0.8 (moderate)"
    if sim >= 0.4:
        return "0.4-<0.6 (lower)"
    return "<0.4 (strong OOD warning)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final known-inhibitor sanity check"
    )
    parser.add_argument(
        "--data-file", default="data/processed/egfr_activity_final.csv"
    )
    parser.add_argument(
        "--candidate-file", default=CANDIDATE_FILE
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

    train_idx, val_idx, test_idx, _, split_meta = (
        standard_scaffold_split_indices(
            smiles_all,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            seed=42,
        )
    )
    train_set = set(smiles_all[i] for i in train_idx)
    smiles_train = [smiles_all[i] for i in train_idx]
    y_train = y_all[train_idx]
    print(
        f"standard scaffold split: train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)}"
    )

    cand = pd.read_csv(args.candidate_file)
    fair = pd.read_csv(FAIR_COMPARISON_FILE)
    same_set = set(cand["canonical_smiles"]) == set(fair["canonical_smiles"])
    print(f"candidate set identical to fair-comparison file: {same_set}")

    cand_ids = cand["molecule_chembl_id"].tolist()
    cand_smiles = cand["canonical_smiles"].tolist()
    cand_names = cand.get("name", [None] * len(cand)).tolist()
    cand_names = [None if pd.isna(x) else str(x) for x in cand_names]
    cand_controls = cand.get(
        "is_positive_control", [False] * len(cand)
    ).astype(bool).tolist()
    n_candidates = len(cand)
    print(
        f"candidate set: {args.candidate_file} n={n_candidates} "
        f"(controls={sum(cand_controls)})"
    )

    print("building fingerprints...", flush=True)
    X_train = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in smiles_train]
    ).astype(np.float32)
    X_cand = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in cand_smiles]
    ).astype(np.float32)
    train_fps = morgan_bit_vectors(smiles_train)
    cand_fps = morgan_bit_vectors(cand_smiles)
    tanimoto = max_train_tanimoto(cand_fps, train_fps)
    exact_match = [s in train_set for s in cand_smiles]

    baseline = make_rf(BASELINE_MAX_FEATURES)
    tuned = make_rf(TUNED_MAX_FEATURES)
    print("training baseline RF (max_features=1.0)...", flush=True)
    baseline.fit(X_train, y_train)
    print("training tuned RF (max_features=0.20)...", flush=True)
    tuned.fit(X_train, y_train)
    pred_base = baseline.predict(X_cand)
    pred_tuned = tuned.predict(X_cand)

    base_df = rank_frame(
        cand_ids, cand_smiles, cand_names, cand_controls, pred_base
    )
    tuned_df = rank_frame(
        cand_ids, cand_smiles, cand_names, cand_controls, pred_tuned
    )
    rank_map = dict(zip(tuned_df["molecule_chembl_id"], tuned_df["rank"]))
    base_rank_map = dict(zip(base_df["molecule_chembl_id"], base_df["rank"]))

    out = pd.DataFrame(
        {
            "molecule_chembl_id": cand_ids,
            "name": cand_names,
            "canonical_smiles": cand_smiles,
            "is_known_inhibitor": cand_controls,
            "exact_training_match": exact_match,
            "max_train_tanimoto": tanimoto,
            "applicability_bin": [applicability_bin(t) for t in tanimoto],
            "baseline_predicted_pIC50": pred_base,
            "baseline_rank": [base_rank_map[i] for i in cand_ids],
            "tuned_predicted_pIC50": pred_tuned,
            "tuned_rank": [rank_map[i] for i in cand_ids],
            "rank_change_tuned_minus_baseline": [
                rank_map[i] - base_rank_map[i] for i in cand_ids
            ],
        }
    )
    out = out.sort_values("tuned_rank").reset_index(drop=True)
    csv_path = results_dir / "final_known_inhibitor_ranking.csv"
    out.to_csv(csv_path, index=False)

    controls = out[out["is_known_inhibitor"]].copy()
    print("\nknown inhibitor summary:")
    for _, row in controls.iterrows():
        print(
            f"  {row['name']}: train_match={row['exact_training_match']} "
            f"sim={row['max_train_tanimoto']:.4f} "
            f"baseline={row['baseline_predicted_pIC50']:.4f} "
            f"rank={row['baseline_rank']} "
            f"tuned={row['tuned_predicted_pIC50']:.4f} "
            f"rank={row['tuned_rank']}"
        )

    def count_in_top(df_sub, n):
        return int((df_sub["tuned_rank"] <= n).sum()), int(
            (df_sub["baseline_rank"] <= n).sum()
        )

    top_counts = {
        f"top_{n}": {
            "tuned": count_in_top(controls, n)[0],
            "baseline": count_in_top(controls, n)[1],
        }
        for n in [5, 10, 20]
    }
    print("top-counts (tuned/baseline):", top_counts)

    top10 = out.head(10).copy()
    print("\ntop 10 tuned candidates:")
    for _, row in top10.iterrows():
        print(
            f"  rank={row['tuned_rank']} {row['name'] or row['molecule_chembl_id']} "
            f"pred={row['tuned_predicted_pIC50']:.4f} sim={row['max_train_tanimoto']:.4f} "
            f"exact={row['exact_training_match']} control={row['is_known_inhibitor']}"
        )

    # Compare with the earlier M6.5 top-1-4 claim.
    earlier = pd.read_csv(args.candidate_file)
    earlier_control = earlier[earlier["is_positive_control"]].copy()
    earlier_control = earlier_control.sort_values("rank").reset_index(drop=True)
    earlier_claim = {
        "source_file": args.candidate_file,
        "model": "M6.5 improved RandomForest (candidate-like internal split)",
        "n_candidates": len(earlier),
        "n_controls": len(earlier_control),
        "earlier_control_ranks": [
            {
                "name": row["name"],
                "rank": int(row["rank"]),
                "rf_pred": float(row["rf_pred"]),
            }
            for _, row in earlier_control.iterrows()
        ],
    }

    summary = {
        "milestone": "final known-inhibitor biological sanity check",
        "test_set_used": False,
        "locked_model": {
            "morgan": {"radius": RADIUS, "nBits": N_BITS},
            "random_forest": {
                "n_estimators": N_ESTIMATORS,
                "random_state": SEED,
                "n_jobs": -1,
                "min_samples_leaf": 1,
                "max_features": TUNED_MAX_FEATURES,
                "max_depth": None,
            },
            "baseline_max_features": BASELINE_MAX_FEATURES,
        },
        "split": {
            **split_meta,
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
            "n_test": len(test_idx),
        },
        "candidate_set": {
            "source_file": args.candidate_file,
            "n_total": n_candidates,
            "n_candidate_like": n_candidates - sum(cand_controls),
            "n_positive_controls": sum(cand_controls),
            "identical_to_earlier_ranking_experiment": True,
            "fair_comparison_file_matches": same_set,
            "note": (
                "The earlier M6.5 experiment used the same 64 molecules "
                "(60 candidate-like + 4 controls) but trained a different RF "
                "on a candidate-like internal split, so the ranks are not "
                "directly comparable model-to-model."
            ),
        },
        "known_inhibitors": controls.to_dict(orient="records"),
        "top_counts": top_counts,
        "top10_tuned": top10[
            [
                "tuned_rank",
                "name",
                "molecule_chembl_id",
                "tuned_predicted_pIC50",
                "max_train_tanimoto",
                "exact_training_match",
                "is_known_inhibitor",
            ]
        ].to_dict(orient="records"),
        "earlier_claim": earlier_claim,
        "interpretation_note": (
            "A high rank for an exact training molecule is not independent "
            "validation; distinguish 'known inhibitor ranked highly' from "
            "'unseen known inhibitor ranked highly'."
        ),
        "versions": package_versions(),
    }
    json_path = results_dir / "final_known_inhibitor_summary.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {json_path}")

    # Figure: baseline vs tuned ranks for the four known inhibitors.
    fig, ax = plt.subplots(figsize=(6.0, 4))
    names = [str(row["name"]) for _, row in controls.iterrows()]
    x = np.arange(len(names))
    base_ranks = controls["baseline_rank"].astype(int).tolist()
    tuned_ranks = controls["tuned_rank"].astype(int).tolist()
    for xi, b, t in zip(x, base_ranks, tuned_ranks):
        ax.plot([xi, xi], [b, t], color="#999999", linewidth=1.5, zorder=1)
    ax.scatter(x, base_ranks, marker="o", color="#C44E52", zorder=2, label="baseline (mf=1.0)")
    ax.scatter(x, tuned_ranks, marker="s", color="#4C72B0", zorder=2, label="tuned (mf=0.20)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("rank in 64-molecule candidate set")
    ax.set_title("Known-inhibitor ranks: baseline vs tuned")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = figures_dir / "final_known_inhibitor_ranks.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {fig_path}")


if __name__ == "__main__":
    main()
