#!/usr/bin/env python3
"""
Final candidate-level interpretation and shortlist prioritization.

This closes the current EGFR AIDD project with an evidence-based
prioritization of NON-TRAINING candidate molecules for potential
experimental validation.

Evidence sources are kept separate:

  1. canonical locked Morgan-RF predictions and rank;
  2. 25 frozen-model perturbation runs (5 scaffold partitions x 5 RF seeds)
     for candidate-level stability;
  3. applicability-domain (max_train_tanimoto) with historical error
     context;
  4. existing physicochemical descriptors (Lipinski/Veber);
  5. existing 1M17 Vina docking scores (score-based only);
  6. repository-level external evidence (not established for these
     candidate-like molecules).

Tiers are assigned with explicit, transparent rules, not with a weighted
score. The locked final Morgan-RF configuration is never changed.

Outputs
-------
results/final_candidate_evidence_matrix.csv
results/final_candidate_shortlist.csv
results/final_candidate_prioritization_summary.json
results/final_candidate_stability_25runs.csv
results/figures/final_candidate_potential_vs_confidence.png
results/figures/final_candidate_rank_stability.png

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/final_candidate_prioritization.py
"""

from __future__ import annotations

import argparse
import json
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
    morgan_bit_vectors,
    murcko,
)


CANDIDATE_RANKING_FILE = "results/final_known_inhibitor_ranking.csv"
PHYSCHEM_FILE = "results/m7_admet_descriptors.csv"
DOCKING_FILE = "results/m7_docking_results.csv"
M8_FILE = "results/m8_final_candidates.csv"
SPLIT_SEEDS = [7, 21, 42, 123, 2024]
RF_SEEDS = [42, 7, 123, 2024, 777]


def ad_category(sim: float) -> str:
    if sim >= 0.80:
        return "HIGH"
    if sim >= 0.60:
        return "MODERATE"
    if sim >= 0.40:
        return "LOWER"
    return "OOD"


def historical_error_context(sim: float) -> str:
    if sim < 0.4:
        return "MAE ~1.07 (high-error region)"
    if sim < 0.6:
        return "MAE ~0.82"
    if sim < 0.8:
        return "MAE ~0.63"
    return "MAE ~0.51"


def docking_status(affinity: float | None) -> str:
    if affinity is None or pd.isna(affinity):
        return "INCOMPLETE"
    if affinity <= -7.5:
        return "SUPPORTIVE"
    if affinity <= -6.0:
        return "MODERATE"
    return "CONCERN"


def potential_label(pred: float, rank: int) -> str:
    if pred >= 6.0 or rank <= 10:
        return "HIGH"
    if pred >= 5.6:
        return "MODERATE"
    return "LOW"


def stability_label(median_rank: float, top10_freq: float) -> str:
    if median_rank <= 15 and top10_freq >= 0.5:
        return "GOOD"
    if median_rank <= 30 and top10_freq >= 0.25:
        return "MODERATE"
    return "WEAK"


def assign_tier(
    potential: str,
    stability: str,
    ad: str,
    physchem_pass: bool,
    dock: str,
) -> str:
    ad_acceptable = ad in ("HIGH", "MODERATE")
    if (
        potential == "HIGH"
        and stability == "GOOD"
        and ad_acceptable
        and physchem_pass
        and dock in ("SUPPORTIVE", "MODERATE")
    ):
        return "Tier 1"
    if (
        potential == "HIGH"
        and stability in ("GOOD", "MODERATE")
        and not ad_acceptable
        and physchem_pass
        and dock != "CONCERN"
    ):
        return "Tier 2"
    if (
        potential == "HIGH"
        and stability == "GOOD"
        and ad_acceptable
        and physchem_pass
        and dock == "INCOMPLETE"
    ):
        return "Tier 2"
    if (
        potential == "HIGH"
        and stability in ("GOOD", "MODERATE")
        and ad_acceptable
        and physchem_pass
        and dock == "INCOMPLETE"
    ):
        return "Tier 2"
    if (
        potential == "MODERATE"
        and stability == "GOOD"
        and ad_acceptable
        and physchem_pass
        and dock in ("SUPPORTIVE", "MODERATE")
    ):
        return "Tier 2"
    if (
        potential == "MODERATE"
        and stability == "GOOD"
        and physchem_pass
        and dock in ("SUPPORTIVE", "MODERATE")
        and not ad_acceptable
    ):
        return "Tier 2"
    return "Tier 3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final candidate prioritization"
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

    ranking = pd.read_csv(CANDIDATE_RANKING_FILE)
    physchem = pd.read_csv(PHYSCHEM_FILE)
    docking = pd.read_csv(DOCKING_FILE)
    m8 = pd.read_csv(M8_FILE)

    eligible = ranking[
        ~ranking["is_known_inhibitor"]
        & ~ranking["exact_training_match"]
    ].copy()
    eligible = eligible.sort_values(
        ["tuned_predicted_pIC50", "canonical_smiles"],
        ascending=[False, True],
    ).reset_index(drop=True)
    eligible["rank_eligible"] = np.arange(1, len(eligible) + 1)
    n_eligible = len(eligible)
    print(f"eligible candidates: {n_eligible} "
          f"(controls excluded, exact train matches excluded)")

    df = pd.read_csv(args.data_file)
    smiles_all = df["canonical_smiles"].tolist()
    y_all = df["pIC50"].to_numpy(dtype=np.float32)

    train_idx, _, _, _, _ = standard_scaffold_split_indices(
        smiles_all, 0.8, 0.1, 0.1, seed=42
    )
    canonical_train_smiles = [smiles_all[i] for i in train_idx]
    canonical_train_scaffolds = set(murcko(s) for s in canonical_train_smiles)

    cand_smiles = eligible["canonical_smiles"].tolist()
    cand_scaffolds = [murcko(s) for s in cand_smiles]
    same_scaffold_canonical = [
        sc in canonical_train_scaffolds for sc in cand_scaffolds
    ]

    # Fingerprints and 25-run stability.
    print("building fingerprints...", flush=True)
    X_all = np.vstack(
        [smiles_to_morgan_fingerprint(s) for s in smiles_all]
    ).astype(np.float32)
    X_cand = np.vstack(
        [smiles_to_morgan_fingerprint(s) for s in cand_smiles]
    ).astype(np.float32)
    cand_smile_set = set(cand_smiles)

    run_rows = []
    for split_seed in SPLIT_SEEDS:
        tr_idx, _, _, _, _ = standard_scaffold_split_indices(
            smiles_all, 0.8, 0.1, 0.1, seed=split_seed
        )
        tr_set = set(smiles_all[i] for i in tr_idx)
        candidates_in_train = cand_smile_set & tr_set
        ranking_mask = [s not in candidates_in_train for s in cand_smiles]
        n_ranked = int(sum(ranking_mask))
        y_tr = np.asarray([y_all[i] for i in tr_idx], dtype=np.float32)
        for rf_seed in RF_SEEDS:
            print(
                f"stability run: split_seed={split_seed} rf_seed={rf_seed} "
                f"n_ranked={n_ranked}",
                flush=True,
            )
            rf = make_rf()
            rf.random_state = rf_seed
            rf.fit(X_all[tr_idx], y_tr)
            preds = rf.predict(X_cand)
            sub_idx = np.where(ranking_mask)[0]
            sub_smiles = [cand_smiles[i] for i in sub_idx]
            sub_preds = preds[sub_idx]
            order = np.lexsort((np.asarray(sub_smiles), -sub_preds))
            ranks = np.empty(len(sub_idx), dtype=int)
            ranks[order] = np.arange(1, len(sub_idx) + 1)
            for pos, orig_i in enumerate(sub_idx):
                run_rows.append(
                    {
                        "molecule_chembl_id": eligible.iloc[orig_i]["molecule_chembl_id"],
                        "split_seed": split_seed,
                        "rf_seed": rf_seed,
                        "predicted_pIC50": float(preds[orig_i]),
                        "rank": int(ranks[pos]),
                        "n_ranked_in_run": n_ranked,
                        "in_train_for_split": False,
                    }
                )
    stability_df = pd.DataFrame(run_rows)
    stability_csv = results_dir / "final_candidate_stability_25runs.csv"
    stability_df.to_csv(stability_csv, index=False)
    print(f"saved -> {stability_csv}")

    agg = (
        stability_df.groupby("molecule_chembl_id")
        .agg(
            pred_mean=("predicted_pIC50", "mean"),
            pred_sd=("predicted_pIC50", "std"),
            pred_median=("predicted_pIC50", "median"),
            rank_mean=("rank", "mean"),
            rank_median=("rank", "median"),
            rank_min=("rank", "min"),
            rank_max=("rank", "max"),
            n_valid=("rank", "count"),
        )
        .reset_index()
    )
    for frac in [5, 10, 20]:
        col = f"top{frac}_freq"
        agg[col] = (
            stability_df[stability_df["rank"] <= frac]
            .groupby("molecule_chembl_id")
            .size()
            .div(agg.set_index("molecule_chembl_id")["n_valid"])
            .reindex(agg["molecule_chembl_id"])
            .fillna(0.0)
            .to_numpy()
        )

    # Merge evidence.
    merged = eligible.merge(
        physchem[
            [
                "molecule_chembl_id",
                "MW",
                "cLogP",
                "TPSA",
                "HBD",
                "HBA",
                "QED",
                "ESOL_logS",
                "Lipinski_ok",
                "Veber_ok",
                "ADMET_ok",
            ]
        ],
        on="molecule_chembl_id",
        how="left",
    ).merge(
        docking[["molecule_chembl_id", "affinity_kcal_mol", "note"]],
        on="molecule_chembl_id",
        how="left",
    ).merge(
        m8[["molecule_chembl_id", "m7_rank", "docking_hit", "confidence_tier"]],
        on="molecule_chembl_id",
        how="left",
    )
    merged = merged.merge(agg, on="molecule_chembl_id", how="left")
    merged["murcko_scaffold"] = cand_scaffolds
    merged["same_scaffold_in_canonical_train"] = same_scaffold_canonical
    merged["AD_category"] = merged["max_train_tanimoto"].map(ad_category)
    merged["historical_error_context"] = merged["max_train_tanimoto"].map(
        historical_error_context
    )
    merged["docking_status"] = merged["affinity_kcal_mol"].map(docking_status)
    merged["potential"] = [
        potential_label(p, r)
        for p, r in zip(
            merged["tuned_predicted_pIC50"], merged["rank_eligible"]
        )
    ]
    merged["stability_label"] = [
        stability_label(m, f) if pd.notna(m) else "WEAK"
        for m, f in zip(merged["rank_median"], merged["top10_freq"])
    ]
    merged["physchem_pass"] = merged["ADMET_ok"].fillna(False).astype(bool)
    merged["tier"] = [
        assign_tier(p, s, ad, ph, d)
        for p, s, ad, ph, d in zip(
            merged["potential"],
            merged["stability_label"],
            merged["AD_category"],
            merged["physchem_pass"],
            merged["docking_status"],
        )
    ]
    merged["external_evidence"] = (
        "not established in repository; no curated EGFR IC50 or name metadata "
        "for this candidate-like molecule"
    )

    matrix_cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "tuned_predicted_pIC50",
        "rank_eligible",
        "max_train_tanimoto",
        "AD_category",
        "historical_error_context",
        "murcko_scaffold",
        "same_scaffold_in_canonical_train",
        "pred_mean",
        "pred_sd",
        "pred_median",
        "rank_mean",
        "rank_median",
        "rank_min",
        "rank_max",
        "top5_freq",
        "top10_freq",
        "top20_freq",
        "n_valid",
        "MW",
        "cLogP",
        "TPSA",
        "HBD",
        "HBA",
        "QED",
        "ESOL_logS",
        "Lipinski_ok",
        "Veber_ok",
        "ADMET_ok",
        "affinity_kcal_mol",
        "docking_status",
        "m7_rank",
        "docking_hit",
        "confidence_tier",
        "external_evidence",
        "potential",
        "stability_label",
        "tier",
    ]
    matrix = merged[matrix_cols].sort_values(
        ["tier", "rank_eligible"]
    )
    matrix_csv = results_dir / "final_candidate_evidence_matrix.csv"
    matrix.to_csv(matrix_csv, index=False)
    print(f"saved -> {matrix_csv}")

    shortlist = merged[merged["tier"].isin(["Tier 1", "Tier 2"])].copy()
    shortlist = shortlist.sort_values(
        ["tier", "rank_eligible", "top10_freq"],
        ascending=[True, True, False],
    )
    shortlist["experimental_validation_priority"] = [
        f"P{i}"
        for i in range(1, len(shortlist) + 1)
    ]
    shortlist["priority_rationale"] = shortlist.apply(
        lambda r: (
            f"pred={r['tuned_predicted_pIC50']:.2f}, rank={r['rank_eligible']}, "
            f"median rank={r['rank_median']:.0f} across {int(r['n_valid'])} runs, "
            f"sim={r['max_train_tanimoto']:.2f} ({r['AD_category']}), "
            f"physchem={'pass' if r['physchem_pass'] else 'fail'}, "
            f"docking={r['docking_status']}"
        ),
        axis=1,
    )
    shortlist_cols = [
        "molecule_chembl_id",
        "tuned_predicted_pIC50",
        "rank_eligible",
        "rank_median",
        "rank_min",
        "rank_max",
        "top10_freq",
        "max_train_tanimoto",
        "AD_category",
        "historical_error_context",
        "ADMET_ok",
        "docking_status",
        "affinity_kcal_mol",
        "external_evidence",
        "tier",
        "experimental_validation_priority",
        "priority_rationale",
    ]
    shortlist_csv = results_dir / "final_candidate_shortlist.csv"
    shortlist[shortlist_cols].to_csv(shortlist_csv, index=False)
    print(f"saved -> {shortlist_csv}")
    print(shortlist[["molecule_chembl_id", "tier", "rank_eligible", "rank_median"]])

    summary = {
        "milestone": "final candidate prioritization",
        "model_frozen": True,
        "eligible_n": n_eligible,
        "candidate_pool_source": "results/final_known_inhibitor_ranking.csv "
        "(64-pool; controls and exact train matches excluded)",
        "tier_rules": {
            "potential": "HIGH if canonical pred >= 6.0 or rank <= 10; "
            "MODERATE if pred >= 5.6; else LOW",
            "stability": "GOOD if median rank <= 15 and top10 freq >= 0.5; "
            "MODERATE if median rank <= 30 and top10 freq >= 0.25; else WEAK",
            "tier_1": "HIGH potential + GOOD stability + HIGH/MODERATE AD + "
            "physchem pass + supportive/moderate docking",
            "tier_2": "novel-but-supported, docking-pending, or "
            "moderate-potential-but-well-supported",
            "tier_3": "insufficient or concerning evidence",
        },
        "tier_counts": merged["tier"].value_counts().to_dict(),
        "shortlist": shortlist[shortlist_cols].to_dict(orient="records"),
        "stability": {
            "design": "5 scaffold partitions x 5 RF seeds = 25 frozen runs",
            "n_runs": len(stability_df),
        },
        "versions": package_versions(),
    }
    json_path = results_dir / "final_candidate_prioritization_summary.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"saved -> {json_path}")

    # Figure 1: potential vs confidence.
    color_map = {"Tier 1": "#4C72B0", "Tier 2": "#55A868", "Tier 3": "#BBBBBB"}
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for tier, color in color_map.items():
        sub = merged[merged["tier"] == tier]
        ax.scatter(
            sub["max_train_tanimoto"],
            sub["tuned_predicted_pIC50"],
            color=color,
            label=tier,
            s=28,
        )
    for _, r in shortlist.head(12).iterrows():
        ax.annotate(
            r["molecule_chembl_id"].replace("CHEMBL", ""),
            (r["max_train_tanimoto"], r["tuned_predicted_pIC50"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.axvline(0.6, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("max_train_tanimoto")
    ax.set_ylabel("canonical tuned predicted pIC50")
    ax.set_title("Candidate potential vs applicability-domain confidence")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig1 = figures_dir / "final_candidate_potential_vs_confidence.png"
    fig.savefig(fig1, dpi=150)
    plt.close(fig)

    # Figure 2: rank stability.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for tier, color in color_map.items():
        sub = merged[merged["tier"] == tier].dropna(subset=["rank_median"])
        ax.errorbar(
            sub["rank_median"],
            [tier] * len(sub),
            xerr=[
                sub["rank_median"] - sub["rank_min"],
                sub["rank_max"] - sub["rank_median"],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1,
            label=tier,
        )
    ax.set_xlabel("median rank across 25 frozen runs")
    ax.set_yticks(["Tier 1", "Tier 2", "Tier 3"])
    ax.set_title("Candidate rank stability")
    ax.grid(alpha=0.3, axis="x")
    ax.legend()
    fig.tight_layout()
    fig2 = figures_dir / "final_candidate_rank_stability.png"
    fig.savefig(fig2, dpi=150)
    plt.close(fig)
    print(f"saved -> {fig1}")
    print(f"saved -> {fig2}")


if __name__ == "__main__":
    main()
