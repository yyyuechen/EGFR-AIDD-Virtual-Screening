#!/usr/bin/env python3
"""
Final portfolio figures: workflow diagram and key-results summary.

Reads only existing result files:
  results/scaffold_split_comparison.json
  results/m9_retrospective_benchmark.json
  results/m9_pose_validation.json

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/final_portfolio_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw_workflow(ax) -> None:
    """Draw a clean end-to-end pipeline diagram."""
    steps = [
        "ChEMBL EGFR",
        "Data curation\n10,161 molecules",
        "Representations\nMorgan / GNN / Transformer",
        "Scaffold-aware\nevaluation",
        "Applicability\ndomain",
        "Virtual\nscreening",
        "ADMET\nfiltering",
        "EGFR\ndocking",
        "Candidate\nprioritization",
    ]
    n = len(steps)
    for i, label in enumerate(steps):
        row = i // 5
        col = i % 5
        x = col * 0.20 + 0.10
        y = 0.72 - row * 0.52
        box = ax.add_patch(
            matplotlib.patches.FancyBboxPatch(
                (x - 0.085, y - 0.09),
                0.17,
                0.18,
                boxstyle="round,pad=0.01",
                fc="#EDF3FB",
                ec="#4C72B0",
                lw=1.2,
            )
        )
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
        )
        if col < 4 and i < n - 1:
            ax.annotate(
                "",
                xy=(x + 0.087, y),
                xytext=(x - 0.087 + 0.19, y),
                arrowprops=dict(arrowstyle="-|>", color="#4C72B0", lw=1.1),
            )
        elif row == 0 and col == 4 and i + 1 < n:
            ax.annotate(
                "",
                xy=(x, y - 0.26),
                xytext=(x, y + 0.26),
                arrowprops=dict(arrowstyle="-|>", color="#4C72B0", lw=1.1),
            )
    ax.text(
        0.5,
        0.93,
        "EGFR inhibitor discovery workflow",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final portfolio figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    splits = json.load(open(results_dir / "scaffold_split_comparison.json"))
    m9 = json.load(open(results_dir / "m9_retrospective_benchmark.json"))
    pose = json.load(open(results_dir / "m9_pose_validation.json"))

    # Figure 1: workflow.
    fig, ax = plt.subplots(figsize=(11, 4.2))
    draw_workflow(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "final_workflow.png", dpi=150)
    plt.close(fig)

    # Figure 2: key results.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    split_names = ["Random", "Standard\nscaffold", "Hard scaffold\nOOD"]
    keys = ["random", "standard_scaffold", "hard_scaffold_ood"]
    r2 = [
        splits["split_evaluation"][k]["RandomForest"]["regression"]["R2"]
        for k in keys
    ]
    rho = [
        splits["split_evaluation"][k]["RandomForest"]["regression"][
            "Spearman_rho"
        ]
        for k in keys
    ]
    x = np.arange(3)
    width = 0.36
    axes[0].bar(
        x - width / 2, r2, width, label="R2", color="#4C72B0"
    )
    axes[0].bar(
        x + width / 2, rho, width, label="Spearman", color="#55A868"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(split_names, fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Morgan-RF metric")
    axes[0].set_title("Scaffold-aware validation")
    axes[0].legend(fontsize=8)
    for xi, (r2v, rhov) in enumerate(zip(r2, rho)):
        axes[0].text(
            xi - width / 2, r2v + 0.02, f"{r2v:.3f}", ha="center", fontsize=8
        )
        axes[0].text(
            xi + width / 2, rhov + 0.02, f"{rhov:.3f}", ha="center", fontsize=8
        )

    bins = sorted(
        splits["applicability"]["bins"], key=lambda b: b["mean_similarity"]
    )
    labels = [b["bin"] for b in bins]
    mae = [b["MAE"] for b in bins]
    axes[1].bar(labels, mae, color="#DD8452")
    for i, v in enumerate(mae):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    axes[1].set_ylabel("MAE (pIC50)")
    axes[1].set_xlabel("max train Tanimoto bin")
    axes[1].set_title("Error vs chemical similarity")
    axes[1].tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "EGFR AIDD portfolio key results",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures_dir / "final_key_results.png", dpi=150)
    plt.close(fig)

    print("saved ->", figures_dir / "final_workflow.png")
    print("saved ->", figures_dir / "final_key_results.png")
    print(
        "annotations: ROC-AUC=%.3f, PR-AUC=%.3f, best RMSD=%.3f A"
        % (
            m9["metrics"]["RandomForest"]["ROC_AUC"],
            m9["metrics"]["RandomForest"]["PR_AUC"],
            pose["best_rmsd_model"]["rmsd_to_crystal_A"],
        )
    )


if __name__ == "__main__":
    main()
