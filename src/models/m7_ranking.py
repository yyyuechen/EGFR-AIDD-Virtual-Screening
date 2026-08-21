#!/usr/bin/env python3
"""
M7: final ranking combining ensemble activity, ADMET, and docking.

Inputs:
  results/m65_rf_transformer_ensemble_predictions.csv
  results/m7_admet_descriptors.csv
  results/m7_docking_results.csv

The final score is a transparent weighted composite calculated only over
molecules that were successfully docked:

  score = 0.5 * z(ensemble pIC50)
        + 0.3 * z(-docking affinity)
        + 0.2 * ADMET_ok

Weights are educational defaults, not tuned on the external set.

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/m7_ranking.py
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

from src.utils import json_safe


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 final ranking")
    parser.add_argument(
        "--scoring-file",
        default="results/m65_rf_transformer_ensemble_predictions.csv",
    )
    parser.add_argument(
        "--admet-file", default="results/m7_admet_descriptors.csv"
    )
    parser.add_argument(
        "--docking-file", default="results/m7_docking_results.csv"
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ensemble = pd.read_csv(args.scoring_file)
    admet = pd.read_csv(args.admet_file)
    docking = pd.read_csv(args.docking_file)

    merged = ensemble.merge(
        admet[
            [
                "molecule_chembl_id",
                "MW",
                "cLogP",
                "TPSA",
                "HBD",
                "HBA",
                "QED",
                "ESOL_logS",
                "Lipinski_violations",
                "Veber_violations",
                "ADMET_ok",
            ]
        ],
        on="molecule_chembl_id",
        how="left",
    ).merge(
        docking[
            [
                "molecule_chembl_id",
                "affinity_kcal_mol",
                "affinity_rank",
            ]
        ],
        on="molecule_chembl_id",
        how="inner",
    )

    docked = merged[merged["affinity_kcal_mol"].notna()].copy()
    if docked.empty:
        raise SystemExit("No successfully docked molecules to rank.")

    docked["z_activity"] = (
        docked["ensemble_equal_pred"] - docked["ensemble_equal_pred"].mean()
    ) / docked["ensemble_equal_pred"].std(ddof=0)
    neg_affinity = -docked["affinity_kcal_mol"]
    docked["z_affinity"] = (
        neg_affinity - neg_affinity.mean()
    ) / neg_affinity.std(ddof=0)
    docked["ADMET_ok_int"] = docked["ADMET_ok"].astype(int)
    docked["m7_score"] = (
        0.5 * docked["z_activity"]
        + 0.3 * docked["z_affinity"]
        + 0.2 * docked["ADMET_ok_int"]
    )
    docked["m7_rank"] = docked["m7_score"].rank(ascending=False).astype(int)
    docked = docked.sort_values("m7_rank").reset_index(drop=True)

    out_columns = [
        "molecule_chembl_id",
        "name",
        "source",
        "ensemble_equal_pred",
        "affinity_kcal_mol",
        "ADMET_ok",
        "Lipinski_violations",
        "Veber_violations",
        "MW",
        "cLogP",
        "TPSA",
        "HBD",
        "HBA",
        "QED",
        "ESOL_logS",
        "z_activity",
        "z_affinity",
        "m7_score",
        "m7_rank",
    ]
    out_columns = [c for c in out_columns if c in docked.columns]
    final = docked[out_columns]
    csv_path = results_dir / "m7_final_ranking.csv"
    final.to_csv(csv_path, index=False)

    summary = {
        "milestone": "M7 final ranking",
        "score_formula": (
            "0.5*z(ensemble pIC50) + 0.3*z(-docking affinity) + 0.2*ADMET_ok"
        ),
        "n_ranked": int(len(final)),
        "n_admet_ok_in_shortlist": int(final["ADMET_ok"].sum()),
        "top10": final.head(10).to_dict(orient="records"),
        "notes": [
            "Only molecules with a successful docking result are ranked.",
            "z-scores are computed across this docked shortlist only.",
            "Weights are fixed educational defaults and were not tuned.",
        ],
    }
    json_path = results_dir / "m7_final_ranking.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )

    print("=" * 70)
    print("M7 final ranking")
    print("=" * 70)
    for _, row in final.head(12).iterrows():
        print(
            f"  #{int(row['m7_rank']):>2} {row['molecule_chembl_id']:<16} "
            f"score={row['m7_score']:+.2f} "
            f"pIC50={row['ensemble_equal_pred']:.2f} "
            f"affinity={row['affinity_kcal_mol']:.2f} "
            f"ADMET={row['ADMET_ok']}"
        )
    print(f"  saved -> {csv_path}")
    print(f"  saved -> {json_path}")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4C72B0" if ok else "#C44E52" for ok in final["ADMET_ok"]]
    ax.scatter(
        final["ensemble_equal_pred"],
        -final["affinity_kcal_mol"],
        s=60,
        c=colors,
        alpha=0.85,
    )
    for _, row in final.head(8).iterrows():
        label = row.get("name") if pd.notna(row.get("name")) and row.get("name") else str(row["molecule_chembl_id"])
        ax.annotate(
            label,
            (row["ensemble_equal_pred"], -row["affinity_kcal_mol"]),
            fontsize=8,
            textcoords="offset points",
            xytext=(4, 4),
        )
    ax.set_xlabel("ensemble predicted pIC50 (RF+Transformer)")
    ax.set_ylabel("-docking affinity (kcal/mol)")
    ax.set_title("M7 final shortlist: activity vs docking, ADMET color")
    fig.tight_layout()
    fig_path = figures_dir / "m7_final_ranking.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  saved -> {fig_path}")


if __name__ == "__main__":
    main()
