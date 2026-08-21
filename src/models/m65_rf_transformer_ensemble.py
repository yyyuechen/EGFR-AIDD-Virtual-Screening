#!/usr/bin/env python3
"""
M6.5: RF + Transformer weighted ensemble validation.

The fair five-model comparison showed RandomForest and the SMILES Transformer
are the two best external ranking models (+0.233 and +0.244 Spearman). This
script validates a weighted mean of the two under the same M6.5 protocol:

- Same training pool (M1 minus four positive controls).
- Same candidate-like validation split (25% of M1 most similar to the
  screening library).
- The weight is selected by candidate-like validation Spearman only; the 59
  external molecules are used exactly once as the final held-out check.

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/m65_rf_transformer_ensemble.py
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

from src.models.external_candidate_validation import (
    build_external_table,
    regression_metrics,
)
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
from src.models.smiles_transformer import (
    encode_smiles,
    predict as transformer_predict,
)


RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": 1.0,
}


def predict_transformer(
    model,
    smiles_list: list[str],
    vocab: dict[str, int],
    max_len: int,
    batch_size: int = 128,
) -> np.ndarray:
    """Encode SMILES with the training vocab and predict."""
    ids = [encode_smiles(s, vocab, max_len) for s in smiles_list]
    dataset = TensorDataset(
        torch.tensor(ids, dtype=torch.long),
        torch.zeros(len(ids), dtype=torch.float),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return transformer_predict(model, loader)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M6.5 RF + Transformer weighted ensemble"
    )
    parser.add_argument("--data-file", default="data/processed/egfr_activity_final.csv")
    parser.add_argument("--candidates-file", default="data/candidates/egfr_candidate_library.csv")
    parser.add_argument("--activities-file", default="data/raw/egfr_chembl_activities_raw.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print_step("Step 1: data (external set + positive controls + scoring)")
    final = pd.read_csv(args.data_file)
    candidates = pd.read_csv(args.candidates_file)
    activities = pd.read_csv(args.activities_file)

    controls = final[final["molecule_chembl_id"].isin(CONTROL_IDS)].copy()
    controls["name"] = controls["molecule_chembl_id"].map(CONTROL_IDS)
    controls["source"] = "positive_control"
    controls = controls[
        ["molecule_chembl_id", "name", "canonical_smiles", "source"]
    ].copy()
    controls["pIC50_reference"] = final.set_index("molecule_chembl_id").loc[
        controls["molecule_chembl_id"], "pIC50"
    ].to_numpy()

    external = build_external_table(candidates, activities)
    external_map = dict(zip(external["molecule_chembl_id"], external["pIC50_external"]))
    external_candidates = candidates[
        candidates["molecule_chembl_id"].map(external_map).notna()
    ].copy()
    external_candidates["pIC50_external"] = external_candidates[
        "molecule_chembl_id"
    ].map(external_map)

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
    print(f"  external test: {len(external_candidates)}")
    print(f"  scoring set:   {len(scoring)}")

    print_step("Step 2: candidate-like split + fingerprints")
    smiles_pool = training_pool["canonical_smiles"].tolist()
    y_pool = training_pool["pIC50"].to_numpy(dtype=np.float32)
    smiles_external = external_candidates["canonical_smiles"].tolist()
    y_external = external_candidates["pIC50_external"].to_numpy(dtype=np.float32)
    smiles_scoring = scoring["canonical_smiles"].tolist()

    X_pool = morgan_fingerprints(smiles_pool)
    X_candidate_library = morgan_fingerprints(candidates["canonical_smiles"].tolist())
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
        f"  train={len(train_idx):,}  candidate-like val={len(val_idx):,}  "
        f"external={len(y_external):,}"
    )

    print_step("Step 3: train RF (M6.5 selected params)")
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, **RF_PARAMS)
    rf_model.fit(X_pool[train_idx], y_train)
    rf_val_pred = rf_model.predict(X_pool[val_idx])
    rf_ext_pred = rf_model.predict(X_external)
    rf_score_pred = rf_model.predict(X_scoring)
    print(
        f"  RF validation Spearman="
        f"{spearmanr(y_val, rf_val_pred)[0]:+.3f}"
    )

    print_step("Step 4: train Transformer (best-epoch selection)")
    transformer, vocab, max_len, tf_best, tf_history = (
        train_transformer_with_epoch_selection(
            smiles_train,
            y_train,
            smiles_val,
            y_val,
        )
    )
    tf_val_pred = predict_transformer(
        transformer, smiles_val, vocab, max_len
    )
    tf_ext_pred = predict_transformer(
        transformer, smiles_external, vocab, max_len
    )
    tf_score_pred = predict_transformer(
        transformer, smiles_scoring, vocab, max_len
    )
    print(
        f"  Transformer validation Spearman="
        f"{spearmanr(y_val, tf_val_pred)[0]:+.3f}"
    )

    print_step("Step 5: weight sweep (selection on candidate-like validation)")
    weights = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    sweep: list[dict] = []
    for w_rf in weights:
        val_ens = w_rf * rf_val_pred + (1.0 - w_rf) * tf_val_pred
        ext_ens = w_rf * rf_ext_pred + (1.0 - w_rf) * tf_ext_pred
        score_ens = w_rf * rf_score_pred + (1.0 - w_rf) * tf_score_pred
        val_rho = float(spearmanr(y_val, val_ens)[0])
        ext_metrics = regression_metrics(y_external, ext_ens)
        ranks, in_top20 = control_rank_summary(score_ens, scoring)
        sweep.append(
            {
                "weight_rf": float(w_rf),
                "weight_transformer": float(round(1.0 - w_rf, 2)),
                "validation_Spearman": val_rho,
                "external_metrics": ext_metrics,
                "control_ranks": ranks,
                "controls_in_top20": in_top20,
            }
        )
        print(
            f"  w_rf={w_rf:.2f} val_rho={val_rho:+.3f} "
            f"ext_rho={ext_metrics['Spearman_rho']:+.3f}",
            flush=True,
        )

    best_val = max(entry["validation_Spearman"] for entry in sweep)
    best_candidates = [
        entry
        for entry in sweep
        if abs(entry["validation_Spearman"] - best_val) < 1e-9
    ]
    selected = min(
        best_candidates, key=lambda entry: abs(entry["weight_rf"] - 0.5)
    )
    by_weight = {entry["weight_rf"]: entry for entry in sweep}
    equal = by_weight[0.5]
    rf_only = by_weight[1.0]
    tf_only = by_weight[0.0]
    print(
        f"  selected w_rf={selected['weight_rf']:.2f} "
        f"val_rho={selected['validation_Spearman']:+.3f} "
        f"ext_rho={selected['external_metrics']['Spearman_rho']:+.3f}"
    )

    selected_score_pred = (
        selected["weight_rf"] * rf_score_pred
        + selected["weight_transformer"] * tf_score_pred
    )
    equal_score_pred = 0.5 * rf_score_pred + 0.5 * tf_score_pred

    print_step("Step 6: save results")
    summary = {
        "milestone": "M6.5 RF + Transformer weighted ensemble",
        "weight_selection": "candidate-like validation Spearman",
        "selected_weight_rf": selected["weight_rf"],
        "selected_weight_transformer": selected["weight_transformer"],
        "rf_only": {
            "external_metrics": rf_only["external_metrics"],
            "control_ranks": rf_only["control_ranks"],
        },
        "transformer_only": {
            "external_metrics": tf_only["external_metrics"],
            "control_ranks": tf_only["control_ranks"],
        },
        "equal_weight_0_5": {
            "external_metrics": equal["external_metrics"],
            "control_ranks": equal["control_ranks"],
        },
        "selected_ensemble": {
            "weight_rf": selected["weight_rf"],
            "weight_transformer": selected["weight_transformer"],
            "validation_Spearman": selected["validation_Spearman"],
            "external_metrics": selected["external_metrics"],
            "control_ranks": selected["control_ranks"],
            "controls_in_top20": selected["controls_in_top20"],
        },
        "transformer_selection": {
            "best_epoch": tf_best["epoch"],
            "validation_Spearman": tf_best["rho"],
            "vocab_size": len(vocab),
            "max_len": max_len,
        },
        "sweep": sweep,
        "caveats": [
            "Weight is selected on candidate-like validation only; the"
            " external 59 molecules are a single final held-out check.",
            "Transformer is retrained on the same split as M6.5 fair"
            " comparison and uses best-epoch selection.",
        ],
    }
    summary_path = results_dir / "m65_rf_transformer_ensemble.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    out = scoring.copy()
    out["rf_pred"] = rf_score_pred
    out["transformer_pred"] = tf_score_pred
    out["ensemble_equal_pred"] = equal_score_pred
    out["ensemble_selected_pred"] = selected_score_pred
    out["rank_selected"] = out["ensemble_selected_pred"].rank(
        ascending=False
    ).astype(int)
    out = out.sort_values("rank_selected").reset_index(drop=True)
    out.to_csv(
        results_dir / "m65_rf_transformer_ensemble_predictions.csv", index=False
    )
    print(f"  saved -> {summary_path}")
    print(f"  saved -> {results_dir / 'm65_rf_transformer_ensemble_predictions.csv'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    w_vals = [entry["weight_rf"] for entry in sweep]
    val_rhos = [entry["validation_Spearman"] for entry in sweep]
    ext_rhos = [entry["external_metrics"]["Spearman_rho"] for entry in sweep]
    axes[0].plot(w_vals, val_rhos, marker="o", label="validation Spearman")
    axes[0].plot(w_vals, ext_rhos, marker="s", label="external Spearman")
    axes[0].axvline(
        selected["weight_rf"], color="gray", linestyle="--", linewidth=1
    )
    axes[0].set_xlabel("weight_rf")
    axes[0].set_ylabel("Spearman")
    axes[0].legend()
    axes[0].set_title("RF + Transformer weight sweep")

    ctrl = out[out["is_positive_control"]]
    axes[1].barh(
        np.arange(len(ctrl)),
        ctrl["rank_selected"],
        color="#C44E52",
    )
    axes[1].set_yticks(np.arange(len(ctrl)))
    axes[1].set_yticklabels(ctrl["name"], fontsize=9)
    axes[1].invert_xaxis()
    axes[1].set_xlabel("rank (1 = top)")
    axes[1].set_title("Positive controls (selected ensemble)")
    fig.tight_layout()
    fig_path = figures_dir / "m65_rf_transformer_ensemble.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  saved -> {fig_path}")

    print_step("M6.5 RF + Transformer ensemble complete")
    print(
        f"  selected w_rf={selected['weight_rf']:.2f}, "
        f"external Spearman="
        f"{selected['external_metrics']['Spearman_rho']:+.3f}, "
        f"controls={selected['control_ranks']}"
    )


if __name__ == "__main__":
    main()
