#!/usr/bin/env python3
"""
M8: final analysis report and error analysis.

This script consolidates every milestone result (M2-M7) into one auditable
summary:

  - milestone regression metrics (random vs scaffold split)
  - external validation history (M6 original -> M6.5 improved -> fair
    comparison -> RF+Transformer ensemble)
  - model agreement on the 64-molecule scoring set
  - external error/bias analysis on the 59 raw-IC50 molecules
  - final candidate table with activity + consensus + ADMET + docking
  - confidence tiers for the docked shortlist

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/m8_final_report.py
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
from scipy.stats import pearsonr, spearmanr

from src.utils import json_safe


def flatten_metrics(
    metrics: dict, milestone: str, split: str
) -> list[dict]:
    """Flatten a {model: metrics} dict into report rows."""
    rows = []
    for model, values in metrics.items():
        rows.append(
            {
                "milestone": milestone,
                "split": split,
                "model": model,
                "MAE": round(float(values["MAE"]), 4),
                "RMSE": round(float(values["RMSE"]), 4),
                "R2": round(float(values["R2"]), 4),
                "Pearson": round(float(values["Pearson_r"]), 4),
                "Spearman": round(float(values["Spearman_rho"]), 4),
            }
        )
    return rows


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="M8 final analysis report")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    m2 = load_json(results_dir / "m2_morgan_baseline_results.json")
    m3 = load_json(results_dir / "m3_morgan_split_comparison_results.json")
    m4 = load_json(results_dir / "m4_graph_baseline_results.json")
    m5 = load_json(results_dir / "m5_smiles_transformer_results.json")
    m6_ext = load_json(results_dir / "m6_external_validation.json")
    m65_imp = load_json(results_dir / "m65_model_improvement.json")
    m65_fair = load_json(results_dir / "m65_fair_comparison.json")
    m65_ens = load_json(results_dir / "m65_rf_transformer_ensemble.json")
    m7_rank = load_json(results_dir / "m7_final_ranking.json")

    milestone_metrics = []
    milestone_metrics += flatten_metrics(m2["metrics"], "M2", "random")
    milestone_metrics += flatten_metrics(
        m3["splits"]["random"]["metrics"], "M3", "random"
    )
    milestone_metrics += flatten_metrics(
        m3["splits"]["scaffold"]["metrics"], "M3", "scaffold"
    )
    milestone_metrics += flatten_metrics(
        m4["splits"]["scaffold"]["metrics"], "M4", "scaffold"
    )
    milestone_metrics += flatten_metrics(
        {"Transformer": m5["splits"]["scaffold"]["metrics"]},
        "M5",
        "scaffold",
    )

    external_history = {
        "M6_original_ensemble": {
            "metrics": m6_ext["model_metrics"]["ensemble_pic50"],
            "n": m6_ext["n_with_external_ic50"],
        },
        "M6_original_rf": m6_ext["model_metrics"]["rf_pred"],
        "M6_original_xgb": m6_ext["model_metrics"]["xgb_pred"],
        "M6_original_gcn": m6_ext["model_metrics"]["gcn_pred"],
        "M6_original_gin": m6_ext["model_metrics"]["gin_pred"],
        "M6_original_transformer": m6_ext["model_metrics"]["transformer_pred"],
        "M65_improved_RF_selected": m65_imp["selected_model"],
        "M65_improved_RF_calibrated": m65_imp["external_metrics"][
            "linear_calibrated_RF"
        ],
        "M65_controls": m65_imp["control_ranks"],
    }
    for entry in m65_fair["models"]:
        external_history[f"fair_{entry['model']}"] = {
            "external_metrics": entry["external_metrics"],
            "control_ranks": entry["control_ranks"],
            "controls_in_top20": entry["controls_in_top20"],
        }
    external_history["RF_Transformer_equal_weight"] = {
        "external_metrics": m65_ens["equal_weight_0_5"]["external_metrics"],
        "control_ranks": m65_ens["equal_weight_0_5"]["control_ranks"],
    }

    fair_preds = pd.read_csv(
        results_dir / "m65_fair_comparison_predictions.csv"
    )
    ensemble_preds = pd.read_csv(
        results_dir / "m65_rf_transformer_ensemble_predictions.csv"
    )
    admet = pd.read_csv(results_dir / "m7_admet_descriptors.csv")
    docking = pd.read_csv(results_dir / "m7_docking_results.csv")
    final_rank = pd.read_csv(results_dir / "m7_final_ranking.csv")

    model_cols = [
        "RandomForest_pred",
        "XGBoost_pred",
        "GCN_pred",
        "GIN_pred",
        "Transformer_pred",
    ]
    agreement = fair_preds[model_cols].corr(method="spearman").round(3)

    fair_preds["RF_TF_diff"] = np.abs(
        fair_preds["RandomForest_pred"] - fair_preds["Transformer_pred"]
    )

    candidates = ensemble_preds.copy()
    candidates = candidates.merge(
        fair_preds[
            ["molecule_chembl_id"] + model_cols + ["RF_TF_diff"]
        ],
        on="molecule_chembl_id",
        how="left",
    )
    admet_cols = [
        "ADMET_ok",
        "MW",
        "cLogP",
        "TPSA",
        "HBD",
        "HBA",
        "QED",
        "ESOL_logS",
        "Lipinski_violations",
        "Veber_violations",
    ]
    candidates = candidates.merge(
        admet[["molecule_chembl_id"] + admet_cols],
        on="molecule_chembl_id",
        how="left",
    )
    candidates = candidates.merge(
        final_rank[
            ["molecule_chembl_id", "affinity_kcal_mol", "m7_score", "m7_rank"]
        ],
        on="molecule_chembl_id",
        how="left",
    )

    candidates["activity_high"] = (
        candidates["ensemble_equal_pred"] >= 6.5
    )
    candidates["docking_hit"] = (
        candidates["affinity_kcal_mol"].notna()
        & (candidates["affinity_kcal_mol"] <= -7.0)
    )
    candidates["consensus_ok"] = candidates["RF_TF_diff"] <= 1.0

    def confidence_tier(row) -> str | None:
        if pd.isna(row["affinity_kcal_mol"]):
            return None
        checks = [
            bool(row["activity_high"]),
            bool(row["docking_hit"]),
            bool(row["ADMET_ok"]),
            bool(row["consensus_ok"]),
        ]
        n = sum(checks)
        if n == 4:
            return "high"
        if n >= 2:
            return "mid"
        return "low"

    candidates["confidence_tier"] = candidates.apply(
        confidence_tier, axis=1
    )

    ext = pd.read_csv(results_dir / "m6_external_validation.csv")
    external_error = {}
    for col in [
        "rf_pred",
        "xgb_pred",
        "gcn_pred",
        "gin_pred",
        "transformer_pred",
        "ensemble_pic50",
    ]:
        y = ext["pIC50_external"].to_numpy()
        p = ext[col].to_numpy()
        residual = p - y
        external_error[col] = {
            "n": int(len(y)),
            "bias_mean": round(float(np.mean(residual)), 4),
            "MAE": round(float(np.mean(np.abs(residual))), 4),
            "RMSE": round(float(np.sqrt(np.mean(residual**2))), 4),
            "Spearman": round(float(spearmanr(y, p)[0]), 4),
        }

    top10 = candidates[
        candidates["m7_rank"].notna()
    ].sort_values("m7_rank").head(10)
    top10_records = top10[
        [
            "molecule_chembl_id",
            "name",
            "source",
            "ensemble_equal_pred",
            "affinity_kcal_mol",
            "ADMET_ok",
            "RF_TF_diff",
            "confidence_tier",
            "m7_rank",
        ]
    ].to_dict(orient="records")

    docked = candidates[candidates["m7_rank"].notna()]
    summary = {
        "milestone": "M8 final analysis report",
        "milestone_metrics": milestone_metrics,
        "external_history": external_history,
        "model_agreement_spearman": {
            "columns": model_cols,
            "matrix": agreement.values.tolist(),
        },
        "rf_transformer_consensus": {
            "mean_abs_diff": round(
                float(fair_preds["RF_TF_diff"].mean()), 4
            ),
            "max_abs_diff": round(float(fair_preds["RF_TF_diff"].max()), 4),
            "n_diff_gt_1": int((fair_preds["RF_TF_diff"] > 1.0).sum()),
        },
        "external_error_analysis": external_error,
        "final_shortlist": {
            "n_docked": int(docked["m7_rank"].notna().sum()),
            "n_admet_ok": int(docked["ADMET_ok"].sum()),
            "n_docking_hit": int(docked["docking_hit"].sum()),
            "n_high_confidence": int(
                (docked["confidence_tier"] == "high").sum()
            ),
            "n_mid_confidence": int(
                (docked["confidence_tier"] == "mid").sum()
            ),
            "n_low_confidence": int(
                (docked["confidence_tier"] == "low").sum()
            ),
            "top10": top10_records,
        },
        "notes": [
            "Model agreement is Spearman correlation on the 64-molecule"
            " scoring set from the M6.5 fair comparison.",
            "External error analysis uses the original M6 model predictions"
            " against the 59 non-nM ChEMBL IC50 records.",
            "Confidence tier counts activity_high, docking_hit, ADMET_ok and"
            " RF-Transformer consensus; 4/4 = high, >=2 = mid, else low.",
        ],
    }

    json_path = results_dir / "m8_final_analysis.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    candidates.to_csv(
        results_dir / "m8_final_candidates.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(agreement.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(model_cols)))
    ax.set_yticks(range(len(model_cols)))
    ax.set_xticklabels(
        [c.replace("_pred", "") for c in model_cols], rotation=45, ha="right"
    )
    ax.set_yticklabels([c.replace("_pred", "") for c in model_cols])
    for i in range(len(model_cols)):
        for j in range(len(model_cols)):
            ax.text(
                j,
                i,
                f"{agreement.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_title("Model agreement (Spearman, 64 scoring molecules)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures_dir / "m8_model_agreement.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(ext["pIC50_external"], ext["ensemble_pic50"], s=25)
    lo = min(ext["pIC50_external"].min(), ext["ensemble_pic50"].min())
    hi = max(ext["pIC50_external"].max(), ext["ensemble_pic50"].max())
    axes[0].plot([lo, hi], [lo, hi], "--", color="gray")
    axes[0].set_xlabel("external pIC50")
    axes[0].set_ylabel("M6 ensemble prediction")
    axes[0].set_title("M6 original ensemble vs external labels")
    residual = ext["ensemble_pic50"] - ext["pIC50_external"]
    axes[1].hist(residual, bins=15, color="#4C72B0")
    axes[1].axvline(
        residual.mean(), color="#C44E52", linestyle="--", label="mean bias"
    )
    axes[1].set_xlabel("prediction - external pIC50")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].set_title("M6 ensemble external residuals")
    fig.tight_layout()
    fig.savefig(figures_dir / "m8_external_error.png", dpi=150)
    plt.close(fig)

    plot_df = top10.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0" if ok else "#C44E52" for ok in plot_df["ADMET_ok"]]
    bars = ax.barh(
        np.arange(len(plot_df)),
        plot_df["m7_score"],
        color=colors,
    )
    ax.set_yticks(np.arange(len(plot_df)))
    labels = [
        (
            str(row["name"])
            if pd.notna(row["name"]) and row["name"]
            else str(row["molecule_chembl_id"])
        )
        for _, row in plot_df.iterrows()
    ]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("M7 composite score")
    ax.set_title("M8 final shortlist (top 10, ADMET color)")
    for bar, row in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{row[1]['affinity_kcal_mol']:.1f} kcal/mol",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "m8_final_shortlist.png", dpi=150)
    plt.close(fig)

    print("=" * 70)
    print("M8 final analysis report")
    print("=" * 70)
    print(f"  milestones summarized: {len(milestone_metrics)} metric rows")
    print(
        f"  docked shortlist: {summary['final_shortlist']['n_docked']}, "
        f"high/mid/low confidence: "
        f"{summary['final_shortlist']['n_high_confidence']}/"
        f"{summary['final_shortlist']['n_mid_confidence']}/"
        f"{summary['final_shortlist']['n_low_confidence']}"
    )
    print(f"  saved -> {json_path}")
    print(f"  saved -> {results_dir / 'm8_final_candidates.csv'}")
    for fig_name in [
        "m8_model_agreement.png",
        "m8_external_error.png",
        "m8_final_shortlist.png",
    ]:
        print(f"  saved -> {figures_dir / fig_name}")


if __name__ == "__main__":
    main()
