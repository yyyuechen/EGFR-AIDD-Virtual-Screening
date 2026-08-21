#!/usr/bin/env python3
"""
Portfolio hardening: standard scaffold split benchmark + applicability domain.

This script adds two focused analyses without touching the existing
milestones:

1. A conventional 80/10/10 Bemis-Murcko scaffold split, allocated at the
   scaffold-group level with a deterministic seed, compared against the
   existing random split and the existing hard scaffold OOD split.
2. A chemical applicability-domain analysis: for every standard-scaffold
   test molecule, compute the maximum Morgan Tanimoto similarity to any
   training molecule and relate it to prediction error.

Models are the existing Morgan + RandomForest / XGBoost baselines with the
exact parameter settings used by M2/M3 (RF 300 trees, XGB 300 trees,
lr=0.05, depth=6, seed=42).

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/portfolio_hardening_analysis.py
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
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.models.evaluate_splits import (
    distribution_stats,
    evaluate_regression,
    json_safe,
    make_models,
    package_versions,
    print_step,
    scaffold_split_indices,
    smiles_to_morgan_fingerprint,
)
from src.models.m9_retrospective_benchmark import benchmark_metrics


def murcko_groups(smiles_list: list[str]) -> list[tuple[str, list[int]]]:
    """Group molecule indices by Bemis-Murcko scaffold, deterministic order."""
    scaffold_to_indices: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES at index {idx}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffold_to_indices.setdefault(scaffold, []).append(idx)
    return sorted(
        scaffold_to_indices.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )


def standard_scaffold_split_indices(
    smiles_list: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int], list[str | None], dict]:
    """
    Deterministic 80/10/10 split at the scaffold-group level.

    Scaffold groups are sorted largest-first (then by scaffold SMILES), the
    group order is shuffled with the recorded seed, and each complete group
    is assigned to the split with the largest remaining deficit. Molecules
    sharing a scaffold can never cross splits.
    """
    groups = murcko_groups(smiles_list)
    n = len(smiles_list)
    targets = {
        "train": int(round(train_frac * n)),
        "val": int(round(val_frac * n)),
    }
    targets["test"] = n - targets["train"] - targets["val"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    counts = {"train": 0, "val": 0, "test": 0}
    split_for_scaffold: dict[str, str] = {}
    for gi in order:
        scaffold, indices = groups[gi]
        split = max(counts, key=lambda s: targets[s] - counts[s])
        split_for_scaffold[scaffold] = split
        counts[split] += len(indices)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    scaffold_by_index: list[str | None] = [None] * n
    for scaffold, indices in groups:
        split = split_for_scaffold[scaffold]
        {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }[split].extend(indices)
        for i in indices:
            scaffold_by_index[i] = scaffold

    metadata = {
        "type": "standard_murcko_scaffold",
        "train_frac_target": train_frac,
        "val_frac_target": val_frac,
        "test_frac_target": test_frac,
        "seed": seed,
        "target_molecule_counts": targets,
        "actual_molecule_counts": dict(counts),
    }
    return train_idx, val_idx, test_idx, scaffold_by_index, metadata


def split_qc(
    df: pd.DataFrame,
    idx_list: list[int],
    scaffold_by_index: list[str | None],
    name: str,
    total_n: int,
) -> dict:
    idx = np.asarray(idx_list, dtype=int)
    scaffolds = [scaffold_by_index[i] for i in idx]
    sizes = pd.Series(scaffolds).value_counts()
    pic50 = df.iloc[idx]["pIC50"].to_numpy(dtype=float)
    return {
        "name": name,
        "n_compounds": int(len(idx)),
        "pct_compounds": round(100.0 * len(idx) / total_n, 2),
        "n_scaffolds": int(sizes.shape[0]),
        "largest_scaffold_group": int(sizes.max()) if sizes.shape[0] else 0,
        "median_scaffold_group": (
            float(sizes.median()) if sizes.shape[0] else None
        ),
        "n_singleton_scaffolds": int((sizes == 1).sum()),
        "pIC50_mean": float(np.mean(pic50)),
        "pIC50_std": float(np.std(pic50)),
        "pIC50_median": float(np.median(pic50)),
        "n_active_m1_class": int((df.iloc[idx]["activity_class"] == 1).sum()),
        "n_inactive_m1_class": int((df.iloc[idx]["activity_class"] == 0).sum()),
        "n_active_screen": int((pic50 >= 7.0).sum()),
        "n_inactive_screen": int((pic50 < 6.0).sum()),
        "n_middle_band": int(((pic50 >= 6.0) & (pic50 < 7.0)).sum()),
    }


def leakage_checks(
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    scaffold_by_index: list[str | None],
) -> dict:
    def scaffold_set(idx_list):
        return {scaffold_by_index[i] for i in idx_list}

    tr = scaffold_set(train_idx)
    va = scaffold_set(val_idx)
    te = scaffold_set(test_idx)
    return {
        "train_cap_test": len(tr & te),
        "train_cap_val": len(tr & va),
        "val_cap_test": len(va & te),
    }


def morgan_bit_vectors(
    smiles_list: list[str], radius: int = 2, n_bits: int = 2048
) -> list:
    """Return RDKit bit-vector objects for bulk Tanimoto similarity."""
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def assign_bin(sim: float) -> str:
    if sim < 0.2:
        return "<0.2"
    if sim < 0.4:
        return "0.2-<0.4"
    if sim < 0.6:
        return "0.4-<0.6"
    if sim < 0.8:
        return "0.6-<0.8"
    return ">=0.8"


def merge_small_bins(
    df: pd.DataFrame, min_n: int = 10
) -> tuple[pd.DataFrame, list[str]]:
    """Merge adjacent similarity bins with fewer than min_n molecules.

    Intervals are merged with their nearest neighbour and relabelled so the
    combined bin reflects the actual similarity range.
    """
    intervals = [
        ("<0.2", 0.0, 0.2),
        ("0.2-<0.4", 0.2, 0.4),
        ("0.4-<0.6", 0.4, 0.6),
        ("0.6-<0.8", 0.6, 0.8),
        (">=0.8", 0.8, 1.0001),
    ]
    out = df.copy()
    out["bin"] = out["max_train_tanimoto"].map(assign_bin)
    while True:
        counts = out["bin"].value_counts().to_dict()
        small = [label for label, _, _ in intervals if counts.get(label, 0) < min_n]
        if not small:
            break
        target = small[0]
        pos = next(
            i for i, (label, _, _) in enumerate(intervals) if label == target
        )
        if pos == len(intervals) - 1:
            neighbor_pos = pos - 1
        else:
            neighbor_pos = pos + 1
        target_lo = intervals[pos][1]
        target_hi = intervals[pos][2]
        neighbor_lo = intervals[neighbor_pos][1]
        neighbor_hi = intervals[neighbor_pos][2]
        lo = min(target_lo, neighbor_lo)
        hi = max(target_hi, neighbor_hi)
        if lo <= 0.0:
            new_label = f"<{hi:.1f}"
        elif hi >= 1.0:
            new_label = f">={lo:.1f}"
        else:
            new_label = f"{lo:.1f}-<{hi:.1f}"
        out.loc[
            out["bin"].isin([intervals[pos][0], intervals[neighbor_pos][0]]),
            "bin",
        ] = new_label
        merged = (new_label, lo, hi)
        low_pos = min(pos, neighbor_pos)
        intervals[low_pos : low_pos + 2] = [merged]
    return out, [label for label, _, _ in intervals]


def bin_metrics(binned: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for label in sorted(binned["bin"].unique()):
        sub = binned[binned["bin"] == label]
        n = int(len(sub))
        row = {
            "bin": label,
            "n": n,
            "mean_similarity": round(float(sub["max_train_tanimoto"].mean()), 4),
            "MAE": round(float(mean_absolute_error(sub["true_pIC50"], sub["predicted_pIC50"])), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(sub["true_pIC50"], sub["predicted_pIC50"]))), 4),
        }
        if n >= 10:
            row["R2"] = round(float(r2_score(sub["true_pIC50"], sub["predicted_pIC50"])), 4)
            rho = spearmanr(sub["max_train_tanimoto"], sub["absolute_error"])
            row["Spearman_sim_vs_error"] = round(float(rho.statistic), 4)
            row["Spearman_p"] = round(float(rho.pvalue), 4)
        else:
            row["R2"] = None
            row["Spearman_sim_vs_error"] = None
            row["Spearman_p"] = None
            row["note"] = "too few molecules for stable correlation metrics"
        rows.append(row)
    return rows


def ranking_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> dict:
    n = int(mask.sum())
    result = {"group": label, "n": n}
    if n < 20:
        result["note"] = "too few molecules for stable ranking metrics"
        return result
    rho = spearmanr(y_true[mask], y_pred[mask])
    result["Spearman"] = round(float(rho.statistic), 4)
    result["Spearman_p"] = round(float(rho.pvalue), 4)
    active_mask = mask & ((y_true >= 7.0) | (y_true < 6.0))
    if active_mask.sum() >= 20:
        binary = (y_true[active_mask] >= 7.0).astype(int)
        scores = y_pred[active_mask]
        result["ROC_AUC"] = round(float(roc_auc_score(binary, scores)), 4)
        result["PR_AUC"] = round(float(average_precision_score(binary, scores)), 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio hardening: scaffold splits + applicability domain"
    )
    parser.add_argument(
        "--data-file", default="data/processed/egfr_activity_final.csv"
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--xgb-trees", type=int, default=300)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print_step("Step 1: load dataset and build Morgan fingerprints")
    df = pd.read_csv(args.data_file)
    smiles_all = df["canonical_smiles"].tolist()
    y_all = df["pIC50"].to_numpy(dtype=np.float32)
    n = len(df)
    all_idx = np.arange(n)
    print(f"  molecules: {n:,}")

    X = np.vstack(
        [smiles_to_morgan_fingerprint(s) for s in smiles_all]
    ).astype(np.float32)
    print(f"  X shape: {X.shape}")

    print_step("Step 2: build three splits")
    r_train, r_test = train_test_split(
        all_idx, test_size=0.2, random_state=args.seed
    )

    hard_train, hard_test, hard_scaffold_stats = scaffold_split_indices(
        smiles_all, test_size=0.2
    )
    hard_groups = murcko_groups(smiles_all)
    hard_scaffold_by_index: list[str | None] = [None] * n
    for scaffold, indices in hard_groups:
        for i in indices:
            hard_scaffold_by_index[i] = scaffold

    (
        std_train,
        std_val,
        std_test,
        std_scaffold_by_index,
        std_meta,
    ) = standard_scaffold_split_indices(
        smiles_all,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        seed=args.seed,
    )

    splits_def = {
        "random": {
            "type": "random",
            "train_indices": r_train.tolist(),
            "test_indices": r_test.tolist(),
            "seed": args.seed,
        },
        "standard_scaffold": {
            **std_meta,
            "train_indices": std_train,
            "val_indices": std_val,
            "test_indices": std_test,
        },
        "hard_scaffold_ood": {
            "type": "hard_murcko_scaffold_ood",
            "test_size_target": 0.2,
            "algorithm": (
                "scaffold groups sorted largest-first, greedily assigned to"
                " train until int(0.8*n); remainder becomes test (no"
                " validation split)"
            ),
            "train_indices": hard_train,
            "test_indices": hard_test,
            "scaffold_size_stats": hard_scaffold_stats["scaffold_size_stats"],
        },
    }

    print_step("Step 3: split QC and leakage checks")

    def qc_simple(idx_list, name):
        idx = np.asarray(idx_list, dtype=int)
        pic50 = df.iloc[idx]["pIC50"].to_numpy(dtype=float)
        return {
            "name": name,
            "n_compounds": int(len(idx)),
            "pct_compounds": round(100.0 * len(idx) / n, 2),
            "pIC50_mean": float(np.mean(pic50)),
            "pIC50_std": float(np.std(pic50)),
            "pIC50_median": float(np.median(pic50)),
            "n_active_m1_class": int((df.iloc[idx]["activity_class"] == 1).sum()),
            "n_inactive_m1_class": int((df.iloc[idx]["activity_class"] == 0).sum()),
            "n_active_screen": int((pic50 >= 7.0).sum()),
            "n_inactive_screen": int((pic50 < 6.0).sum()),
            "n_middle_band": int(((pic50 >= 6.0) & (pic50 < 7.0)).sum()),
        }

    # Random split has no scaffold assignment; the scaffold-based splits use
    # their Murcko scaffold mapping for QC.
    qc = {
        "random_train": qc_simple(r_train.tolist(), "random_train"),
        "random_test": qc_simple(r_test.tolist(), "random_test"),
        "standard_train": split_qc(df, std_train, std_scaffold_by_index, "standard_train", n),
        "standard_val": split_qc(df, std_val, std_scaffold_by_index, "standard_val", n),
        "standard_test": split_qc(df, std_test, std_scaffold_by_index, "standard_test", n),
        "hard_train": split_qc(df, hard_train, hard_scaffold_by_index, "hard_train", n),
        "hard_test": split_qc(df, hard_test, hard_scaffold_by_index, "hard_test", n),
    }
    leakage = {
        "standard": leakage_checks(std_train, std_val, std_test, std_scaffold_by_index),
        "hard": leakage_checks(hard_train, [], hard_test, hard_scaffold_by_index),
    }
    for key, checks in leakage.items():
        print(f"  {key} leakage (scaffold intersections): {checks}")
    for name in ["standard_test", "hard_test"]:
        print(
            f"  {name}: n={qc[name]['n_compounds']}, "
            f"scaffolds={qc[name]['n_scaffolds']}, "
            f"singletons={qc[name]['n_singleton_scaffolds']}"
        )

    print_step("Step 4: train Morgan-RF / XGBoost on every split")
    models = make_models(args.rf_trees, args.xgb_trees, args.seed)
    split_eval: dict = {}
    for split_name, train_list, test_list, val_list in [
        ("random", r_train.tolist(), r_test.tolist(), None),
        ("standard_scaffold", std_train, std_test, std_val),
        ("hard_scaffold_ood", hard_train, hard_test, None),
    ]:
        X_tr, y_tr = X[train_list], y_all[train_list]
        X_te, y_te = X[test_list], y_all[test_list]
        trained = {}
        for model_name, model in models.items():
            model.fit(X_tr, y_tr)
            trained[model_name] = model.predict(X_te)
        split_eval[split_name] = {}
        for model_name, y_pred in trained.items():
            entry = {
                "regression": evaluate_regression(y_te, y_pred),
                "screening": benchmark_metrics(
                    y_pred,
                    y_te,
                    active_mask=(y_te >= 7.0) | (y_te < 6.0),
                ),
            }
            split_eval[split_name][model_name] = entry
        if val_list:
            X_va, y_va = X[val_list], y_all[val_list]
            split_eval[split_name]["validation"] = {}
            for model_name, y_pred in trained.items():
                split_eval[split_name]["validation"][model_name] = (
                    evaluate_regression(y_va, models[model_name].predict(X_va))
                )
        print(
            f"  {split_name}: "
            f"RF R2={split_eval[split_name]['RandomForest']['regression']['R2']:.3f} "
            f"MAE={split_eval[split_name]['RandomForest']['regression']['MAE']:.3f} "
            f"Spearman={split_eval[split_name]['RandomForest']['regression']['Spearman_rho']:.3f}"
        )

    print_step("Step 5: applicability domain (standard scaffold test)")
    train_bits = morgan_bit_vectors([smiles_all[i] for i in std_train])
    test_bits = morgan_bit_vectors([smiles_all[i] for i in std_test])
    max_tanimoto = np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(fp, train_bits))
            for fp in test_bits
        ],
        dtype=float,
    )
    # Refit RF on the standard scaffold train split so the applicability
    # predictions use exactly that train set as the reference chemistry.
    std_rf = make_models(args.rf_trees, args.xgb_trees, args.seed)[
        "RandomForest"
    ]
    std_rf.fit(X[std_train], y_all[std_train])
    std_rf_pred = std_rf.predict(X[std_test])
    true = y_all[std_test]
    abs_err = np.abs(true - std_rf_pred)
    activity_class = np.where(
        true >= 7.0, "active", np.where(true < 6.0, "inactive", "middle")
    )
    app_df = pd.DataFrame(
        {
            "canonical_smiles": [smiles_all[i] for i in std_test],
            "true_pIC50": true,
            "predicted_pIC50": std_rf_pred,
            "absolute_error": abs_err,
            "max_train_tanimoto": max_tanimoto,
            "scaffold": [std_scaffold_by_index[i] for i in std_test],
            "activity_class": activity_class,
        }
    )
    app_csv = results_dir / "applicability_domain_predictions.csv"
    app_df.to_csv(app_csv, index=False)

    rho_err = spearmanr(max_tanimoto, abs_err)
    pearson_err = pearsonr(max_tanimoto, abs_err)
    binned, bin_order = merge_small_bins(app_df)
    bins = bin_metrics(binned)
    print(
        f"  Spearman(max_train_tanimoto, abs_error)="
        f"{rho_err.statistic:.4f} (p={rho_err.pvalue:.4f})"
    )
    print(f"  max_train_tanimoto stats: {distribution_stats(max_tanimoto)}")
    print(f"  saved -> {app_csv}")

    ranking_groups = {}
    for label, mask in [
        ("high_similarity_ge_0.6", max_tanimoto >= 0.6),
        ("low_similarity_lt_0.6", max_tanimoto < 0.6),
    ]:
        ranking_groups[label] = ranking_group_metrics(
            true, std_rf_pred, mask, label
        )

    print_step("Step 6: save JSON and figures")
    summary = {
        "milestone": "portfolio_hardening",
        "purpose": (
            "standard scaffold split benchmark + chemical applicability"
            " domain analysis"
        ),
        "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 2048},
        "versions": package_versions(),
        "model_params": json_safe(
            {name: m.get_params() for name, m in models.items()}
        ),
        "active_inactive_definition": (
            "screening binary: active pIC50 >= 7.0, inactive pIC50 < 6.0,"
            " middle band excluded; M1 activity_class (>=7 vs <7) reported"
            " separately in QC"
        ),
        "splits": splits_def,
        "qc": qc,
        "leakage": leakage,
        "split_evaluation": split_eval,
        "applicability": {
            "split": "standard_scaffold",
            "n_test": int(len(std_test)),
            "max_train_tanimoto_stats": distribution_stats(max_tanimoto),
            "spearman_sim_vs_abs_error": {
                "rho": float(rho_err.statistic),
                "p": float(rho_err.pvalue),
            },
            "pearson_sim_vs_abs_error": {
                "r": float(pearson_err.statistic),
                "p": float(pearson_err.pvalue),
            },
            "bins": bins,
            "ranking_by_similarity_group": ranking_groups,
        },
    }
    summary_path = results_dir / "scaffold_split_comparison.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2))
    print(f"  saved -> {summary_path}")

    # Figure 1: metrics across splits.
    split_names = ["Random", "Standard scaffold", "Hard scaffold OOD"]
    key_map = ["random", "standard_scaffold", "hard_scaffold_ood"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(split_names))
    for ax, metric in zip(axes, ["R2", "MAE"]):
        for model_name, color in [("RandomForest", "#4C72B0"), ("XGBoost", "#DD8452")]:
            values = [
                split_eval[k][model_name]["regression"][metric]
                for k in key_map
            ]
            ax.plot(x, values, marker="o", label=model_name, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(split_names, fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"Morgan baseline {metric}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig1 = figures_dir / "portfolio_split_comparison.png"
    fig.savefig(fig1, dpi=150)
    print(f"  saved -> {fig1}")

    # Figure 2: Tanimoto vs absolute error.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(max_tanimoto, abs_err, s=14, alpha=0.55, color="#4C72B0")
    coeffs = np.polyfit(max_tanimoto, abs_err, 1)
    xs = np.linspace(max_tanimoto.min(), max_tanimoto.max(), 50)
    ax.plot(xs, np.polyval(coeffs, xs), color="#C44E52", linewidth=1.6)
    ax.set_xlabel("max train Tanimoto (Morgan, radius=2, 2048 bits)")
    ax.set_ylabel("absolute pIC50 error")
    ax.set_title(
        f"Applicability domain (standard scaffold test)\n"
        f"Spearman={rho_err.statistic:.3f}, p={rho_err.pvalue:.3f}"
    )
    fig.tight_layout()
    fig2 = figures_dir / "portfolio_applicability_scatter.png"
    fig.savefig(fig2, dpi=150)
    print(f"  saved -> {fig2}")

    # Figure 3: MAE by similarity bin.
    labels = [row["bin"] for row in bins]
    mae = [row["MAE"] for row in bins]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, mae, color="#55A868")
    for i, v in enumerate(mae):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("max train Tanimoto bin")
    ax.set_ylabel("MAE (pIC50)")
    ax.set_title("Prediction error by chemical similarity to training set")
    fig.tight_layout()
    fig3 = figures_dir / "portfolio_applicability_bins.png"
    fig.savefig(fig3, dpi=150)
    print(f"  saved -> {fig3}")

    print_step("Portfolio hardening analysis complete")
    print(
        "Interpretation guide: Random is the optimistic in-distribution"
        " estimate; standard scaffold is a realistic scaffold-aware estimate;"
        " hard scaffold OOD is the extreme extrapolation test."
    )


if __name__ == "__main__":
    main()
