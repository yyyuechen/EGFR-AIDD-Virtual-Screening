#!/usr/bin/env python3
"""
P0-3b: multi-seed reproducibility for the final screening models.

The M6.5 fair comparison and RF + Transformer ensemble were trained once with
seed 42. This script retrains the final screening models across several seeds
and reports the spread of external Spearman, p-values, control ranks, and
top-20 stability. The external labels are the P0-3a clean labels (median
across exact IC50 records).

Models covered per seed:
  - RandomForest (M6.5 selected params, random_state = seed)
  - XGBoost (M6.5 selected params, random_state = seed, for reference)
  - SMILES Transformer (best-epoch selection, torch seed = seed)
  - Equal-weight RF + Transformer ensemble

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/p03_multiseed_reproducibility.py --seeds 42,7,123,2024,777
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
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.models.external_candidate_validation import regression_metrics
from src.models.m65_fair_comparison import (
    control_rank_summary,
    print_step,
    train_transformer_with_epoch_selection,
)
from src.models.m65_model_improvement import (
    CONTROL_IDS,
    candidate_like_validation_mask,
    morgan_fingerprints,
)
from src.models.m65_rf_transformer_ensemble import (
    RF_PARAMS,
    predict_transformer,
)
from src.models.p03_external_label_audit import (
    build_clean_labels,
    build_external_records,
)
from src.utils import json_safe


XGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 8,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "min_child_weight": 3,
}


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_positive": int((arr > 0).sum()),
    }


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_iter: int = 1000, seed: int = 12345
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rhos = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        rhos.append(float(spearmanr(y_true[idx], y_pred[idx])[0]))
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return {
        "lo_2_5": float(lo),
        "hi_97_5": float(hi),
        "n_iter": n_iter,
    }


def permutation_pvalue(
    y_true: np.ndarray, y_pred: np.ndarray, n_perm: int = 1000, seed: int = 54321
) -> dict:
    rng = np.random.default_rng(seed)
    observed = float(spearmanr(y_true, y_pred)[0])
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(y_true)
        if spearmanr(perm, y_pred)[0] >= observed:
            count += 1
    return {
        "observed": observed,
        "p_value": float((count + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-3 multi-seed reproducibility")
    parser.add_argument("--data-file", default="data/processed/egfr_activity_final.csv")
    parser.add_argument("--candidates-file", default="data/candidates/egfr_candidate_library.csv")
    parser.add_argument("--activities-file", default="data/raw/egfr_chembl_activities_raw.csv")
    parser.add_argument("--seeds", default="42,7,123,2024,777")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print_step("Step 1: data (clean external labels + controls + scoring)")
    final = pd.read_csv(args.data_file)
    candidates = pd.read_csv(args.candidates_file)
    activities = pd.read_csv(args.activities_file)

    controls = final[final["molecule_chembl_id"].isin(CONTROL_IDS)].copy()
    controls["name"] = controls["molecule_chembl_id"].map(CONTROL_IDS)
    controls["source"] = "positive_control"
    controls["pIC50_reference"] = final.set_index("molecule_chembl_id").loc[
        controls["molecule_chembl_id"], "pIC50"
    ].to_numpy()

    records = build_external_records(candidates, activities)
    clean = build_clean_labels(records)
    clean_map = dict(zip(clean["molecule_chembl_id"], clean["pIC50_median"]))
    external_candidates = candidates[
        candidates["molecule_chembl_id"].map(clean_map).notna()
    ].copy()
    external_candidates["pIC50_external_clean"] = external_candidates[
        "molecule_chembl_id"
    ].map(clean_map)

    control_smiles = set(controls["canonical_smiles"])
    training_pool = final[~final["canonical_smiles"].isin(control_smiles)].copy()
    scoring = pd.concat(
        [
            candidates[["molecule_chembl_id", "canonical_smiles", "source"]],
            controls[["molecule_chembl_id", "canonical_smiles", "source"]],
        ],
        ignore_index=True,
    )
    scoring["is_positive_control"] = scoring["source"] == "positive_control"
    scoring["name"] = scoring["molecule_chembl_id"].map(CONTROL_IDS)

    print(f"  training pool: {len(training_pool):,}")
    print(f"  clean external test: {len(external_candidates)}")
    print(f"  scoring set: {len(scoring)}")

    print_step("Step 2: fingerprints + candidate-like split (same across seeds)")
    smiles_pool = training_pool["canonical_smiles"].tolist()
    y_pool = training_pool["pIC50"].to_numpy(dtype=np.float32)
    smiles_external = external_candidates["canonical_smiles"].tolist()
    y_external = external_candidates["pIC50_external_clean"].to_numpy(
        dtype=np.float32
    )
    smiles_scoring = scoring["canonical_smiles"].tolist()

    X_pool = morgan_fingerprints(smiles_pool)
    X_candidate_library = morgan_fingerprints(
        candidates["canonical_smiles"].tolist()
    )
    X_external = morgan_fingerprints(smiles_external)
    X_scoring = morgan_fingerprints(smiles_scoring)

    validation_mask = candidate_like_validation_mask(
        X_pool, X_candidate_library, fraction=0.25
    )
    train_idx = np.where(~validation_mask)[0]
    val_idx = np.where(validation_mask)[0]
    smiles_train = [smiles_pool[i] for i in train_idx]
    smiles_val = [smiles_pool[i] for i in val_idx]
    y_train = y_pool[train_idx]
    y_val = y_pool[val_idx]
    print(
        f"  train={len(train_idx):,} candidate-like val={len(val_idx):,} "
        f"external={len(y_external):,}"
    )

    print_step(f"Step 3: multi-seed training ({len(seeds)} seeds)")
    per_seed: list[dict] = []
    prediction_rows: list[dict] = []
    external_ensemble_preds: dict[int, np.ndarray] = {}
    for seed in seeds:
        print(
            f"\n--- seed {seed}: RF / XGB / Transformer / equal ensemble ---",
            flush=True,
        )
        torch.manual_seed(seed)

        rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **RF_PARAMS)
        rf.fit(X_pool[train_idx], y_train)
        rf_ext_pred = rf.predict(X_external)
        rf_score_pred = rf.predict(X_scoring)
        rf_val_rho = float(spearmanr(y_val, rf.predict(X_pool[val_idx]))[0])

        xgb = XGBRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **XGB_PARAMS,
        )
        xgb.fit(X_pool[train_idx], y_train)
        xgb_ext_pred = xgb.predict(X_external)
        xgb_score_pred = xgb.predict(X_scoring)

        transformer, vocab, max_len, tf_best, _ = (
            train_transformer_with_epoch_selection(
                smiles_train,
                y_train,
                smiles_val,
                y_val,
                seed=seed,
            )
        )
        tf_ext_pred = predict_transformer(
            transformer, smiles_external, vocab, max_len
        )
        tf_score_pred = predict_transformer(
            transformer, smiles_scoring, vocab, max_len
        )

        ens_ext_pred = 0.5 * rf_ext_pred + 0.5 * tf_ext_pred
        ens_score_pred = 0.5 * rf_score_pred + 0.5 * tf_score_pred
        control_ranks, controls_in_top20 = control_rank_summary(
            ens_score_pred, scoring
        )

        entry = {
            "seed": seed,
            "RandomForest": regression_metrics(y_external, rf_ext_pred),
            "XGBoost": regression_metrics(y_external, xgb_ext_pred),
            "Transformer": regression_metrics(y_external, tf_ext_pred),
            "Ensemble_equal_weight": regression_metrics(
                y_external, ens_ext_pred
            ),
            "rf_validation_Spearman": rf_val_rho,
            "transformer_selection": {
                "best_epoch": tf_best["epoch"],
                "validation_Spearman": tf_best["rho"],
                "vocab_size": len(vocab),
                "max_len": max_len,
            },
            "control_ranks_ensemble": control_ranks,
            "controls_in_top20_ensemble": controls_in_top20,
        }
        per_seed.append(entry)
        print(
            f"  seed {seed}: RF={entry['RandomForest']['Spearman_rho']:+.3f} "
            f"XGB={entry['XGBoost']['Spearman_rho']:+.3f} "
            f"TF={entry['Transformer']['Spearman_rho']:+.3f} "
            f"ens={entry['Ensemble_equal_weight']['Spearman_rho']:+.3f} "
            f"controls={control_ranks}",
            flush=True,
        )

        ranked = scoring.copy()
        ranked["ensemble_pred"] = ens_score_pred
        ranked["rank"] = ranked["ensemble_pred"].rank(
            ascending=False
        ).astype(int)
        for _, row in ranked.iterrows():
            prediction_rows.append(
                {
                    "seed": seed,
                    "molecule_chembl_id": row["molecule_chembl_id"],
                    "canonical_smiles": row["canonical_smiles"],
                    "is_positive_control": row["is_positive_control"],
                    "ensemble_pred": float(row["ensemble_pred"]),
                }
            )
        external_ensemble_preds[seed] = ens_ext_pred

    print_step("Step 4: summarize")
    models = ["RandomForest", "XGBoost", "Transformer", "Ensemble_equal_weight"]
    stats = {
        name: summarize(
            [entry[name]["Spearman_rho"] for entry in per_seed]
        )
        for name in models
    }
    p_values = {
        name: [entry[name]["Spearman_p"] for entry in per_seed]
        for name in models
    }

    top20_by_seed = {}
    for seed in seeds:
        rows = [
            r
            for r in prediction_rows
            if r["seed"] == seed
        ]
        top20_by_seed[seed] = {
            r["molecule_chembl_id"]
            for r in sorted(rows, key=lambda r: r["ensemble_pred"], reverse=True)[
                :20
            ]
        }

    seed42_top20 = top20_by_seed.get(seeds[0], set())
    top20_overlap = {
        seed: int(len(seed42_top20 & top20_by_seed[seed]))
        for seed in seeds
    }

    ens_boot: dict | None = None
    ens_perm: dict | None = None
    first_seed_preds = external_ensemble_preds.get(seeds[0])
    if first_seed_preds is not None and len(first_seed_preds) == len(y_external):
        ens_boot = bootstrap_ci(y_external, first_seed_preds)
        ens_perm = permutation_pvalue(y_external, first_seed_preds)

    summary = {
        "milestone": "P0-3b multi-seed reproducibility",
        "seeds": seeds,
        "external_labels": (
            "P0-3a clean labels (median across exact IC50 records)"
        ),
        "n_external": int(len(y_external)),
        "models": models,
        "per_seed": per_seed,
        "summary_stats": stats,
        "spearman_p_values_by_seed": p_values,
        "sign_consistency": {
            name: bool(all(entry[name]["Spearman_rho"] > 0 for entry in per_seed))
            for name in models
        },
        "top20_overlap_with_first_seed": top20_overlap,
        "bootstrap_95ci_ensemble_first_seed": ens_boot,
        "permutation_pvalue_ensemble_first_seed": ens_perm,
        "caveats": [
            "The candidate-like split is deterministic (Morgan-based), so"
            " seed variation only affects model fitting.",
            "GCN / GIN remain single-seed educational baselines; this script"
            " covers the final screening models.",
            "Transformer training is the dominant runtime; the same"
            " best-epoch-on-validation selection is used for every seed.",
        ],
    }
    summary_path = results_dir / "p03_multiseed_reproducibility.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )

    per_seed_rows = []
    for entry in per_seed:
        row = {"seed": entry["seed"]}
        for name in models:
            row[f"{name}_Spearman"] = entry[name]["Spearman_rho"]
            row[f"{name}_p"] = entry[name]["Spearman_p"]
            row[f"{name}_MAE"] = entry[name]["MAE"]
        row["controls_in_top20_ensemble"] = entry[
            "controls_in_top20_ensemble"
        ]
        per_seed_rows.append(row)
    per_seed_path = results_dir / "p03_multiseed_per_seed.csv"
    pd.DataFrame(per_seed_rows).to_csv(per_seed_path, index=False)
    print(f"  saved -> {summary_path}")
    print(f"  saved -> {per_seed_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["RandomForest", "XGBoost", "Transformer", "Ensemble_equal_weight"]
    means = [stats[n]["mean"] for n in names]
    stds = [stats[n]["std"] for n in names]
    colors = ["#4C72B0", "#DD8452", "#8172B3", "#55A868"]
    ax.barh(
        np.arange(len(names)),
        means,
        xerr=stds,
        color=colors,
        capsize=4,
    )
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("external Spearman (mean over seeds)")
    ax.set_title("P0-3 multi-seed external ranking")
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            mean + std + 0.01,
            i,
            f"{mean:+.3f} +/- {std:.3f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig_path = figures_dir / "p03_multiseed_external_rho.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  saved -> {fig_path}")

    print_step("P0-3b multi-seed reproducibility complete")
    for name in models:
        s = stats[name]
        print(
            f"  {name:<22} mean={s['mean']:+.3f} "
            f"std={s['std']:.3f} min={s['min']:+.3f} max={s['max']:+.3f}"
        )
    print(f"  equal-weight ensemble p-values: {p_values['Ensemble_equal_weight']}")


if __name__ == "__main__":
    main()
