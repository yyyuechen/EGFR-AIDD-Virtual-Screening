#!/usr/bin/env python3
"""
M6.5 fair comparison: retrain all five models under the same protocol.

The original M6.5 improvement loop tuned RF / XGB and left the M4/M5 deep
models on their old training protocol. This script makes the comparison fair:

- Same training pool (M1 minus the four local positive controls).
- Same candidate-like validation split (25% of M1 most similar to the
  screening library).
- Same selection principle: models are chosen by candidate-like validation
  Spearman. Trees use a small hyperparameter grid; GCN / GIN / Transformer
  use best-epoch selection over 30 epochs.
- Same external test set: the 59 molecules with non-nM ChEMBL IC50 records.
- Same scoring set: 60 candidates plus the four positive controls.

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/m65_fair_comparison.py
"""

from __future__ import annotations

import argparse
import copy
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
from torch_geometric.loader import DataLoader as GraphDataLoader
from xgboost import XGBRegressor

from src.models.external_candidate_validation import (
    build_external_table,
    regression_metrics,
)
from src.models.graph_baseline import (
    EDGE_FEATURE_DIM,
    GCNRegressor,
    GINRegressor,
    predict as graph_predict,
    smiles_to_graph,
    train_epoch as graph_train_epoch,
)
from src.models.m65_model_improvement import (
    CONTROL_IDS,
    RF_GRID,
    XGB_GRID,
    candidate_like_validation_mask,
    morgan_fingerprints,
)
from src.models.morgan_baseline import smiles_to_morgan_fingerprint
from src.models.smiles_transformer import (
    SMILESTransformerRegressor,
    build_vocab,
    encode_smiles,
    predict as transformer_predict,
    tokenize_smiles,
    train_epoch as transformer_train_epoch,
)


def print_step(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70, flush=True)


def train_graph_with_epoch_selection(
    model_name: str,
    factory,
    graphs_train: list,
    y_train: np.ndarray,
    graphs_val: list,
    y_val: np.ndarray,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 256,
) -> tuple[torch.nn.Module, dict, list]:
    """
    Train a GCN/GIN for 30 epochs and keep the best-epoch checkpoint by
    candidate-like validation Spearman, matching the tree grid-selection
    principle (select on internal validation, then evaluate externally).
    """
    torch.manual_seed(seed)
    model = factory()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    train_loader = GraphDataLoader(graphs_train, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(graphs_val, batch_size=batch_size, shuffle=False)

    history: list[dict] = []
    best: dict | None = None
    for epoch in range(1, epochs + 1):
        loss = graph_train_epoch(model, train_loader, optimizer)
        val_pred = graph_predict(model, val_loader)
        val_rho = float(spearmanr(y_val, val_pred)[0])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss),
                "validation_Spearman": val_rho,
            }
        )
        if best is None or val_rho > best["rho"]:
            best = {
                "epoch": epoch,
                "rho": val_rho,
                "state": copy.deepcopy(model.state_dict()),
            }
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"    {model_name} epoch {epoch:02d}/{epochs} "
                f"loss={loss:.4f} val_rho={val_rho:+.3f}",
                flush=True,
            )

    model.load_state_dict(best["state"])
    print(f"  {model_name}: best epoch {best['epoch']} val_rho={best['rho']:+.3f}")
    return model, best, history


def train_transformer_with_epoch_selection(
    smiles_train: list[str],
    y_train: np.ndarray,
    smiles_val: list[str],
    y_val: np.ndarray,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 128,
) -> tuple[SMILESTransformerRegressor, dict[str, int], int, dict, list]:
    """
    Train the M5 SMILES Transformer with best-epoch selection on the same
    candidate-like validation set. Vocabulary and max_len come from the
    training split only.
    """
    torch.manual_seed(seed)
    vocab = build_vocab(smiles_train)
    train_lengths = [len(tokenize_smiles(s)) for s in smiles_train]
    max_len = min(max(train_lengths), 200)

    train_ids = [
        encode_smiles(s, vocab, max_len) for s in smiles_train
    ]
    val_ids = [encode_smiles(s, vocab, max_len) for s in smiles_val]
    train_dataset = TensorDataset(
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_ids, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.float),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SMILESTransformerRegressor(
        vocab_size=len(vocab),
        max_len=max_len,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4
    )

    history: list[dict] = []
    best: dict | None = None
    for epoch in range(1, epochs + 1):
        loss = transformer_train_epoch(model, train_loader, optimizer)
        val_pred = transformer_predict(model, val_loader)
        val_rho = float(spearmanr(y_val, val_pred)[0])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss),
                "validation_Spearman": val_rho,
            }
        )
        if best is None or val_rho > best["rho"]:
            best = {
                "epoch": epoch,
                "rho": val_rho,
                "state": copy.deepcopy(model.state_dict()),
            }
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"    Transformer epoch {epoch:02d}/{epochs} "
                f"loss={loss:.4f} val_rho={val_rho:+.3f}",
                flush=True,
            )

    model.load_state_dict(best["state"])
    print(
        f"  Transformer: best epoch {best['epoch']} "
        f"val_rho={best['rho']:+.3f}, vocab={len(vocab)}, max_len={max_len}"
    )
    return model, vocab, max_len, best, history


def control_rank_summary(
    predictions: np.ndarray,
    scoring: pd.DataFrame,
) -> tuple[dict, bool]:
    """Rank the full scoring set and return positive-control ranks."""
    ranked = scoring.copy()
    ranked["prediction"] = predictions
    ranked["rank"] = ranked["prediction"].rank(ascending=False).astype(int)
    control_rows = ranked[ranked["is_positive_control"]]
    ranks = {
        str(row["molecule_chembl_id"]): int(row["rank"])
        for _, row in control_rows.iterrows()
    }
    return ranks, all(rank <= 20 for rank in ranks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="M6.5 fair comparison")
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
    print(f"  scoring set:   {len(scoring)} (60 candidates + {len(controls)} controls)")

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

    print_step("Step 3: tree models (same grid + validation selection)")
    model_summaries: list[dict] = []
    prediction_columns: dict[str, np.ndarray] = {}

    rf_results: list[tuple[dict, RandomForestRegressor, np.ndarray, np.ndarray]] = []
    for params in RF_GRID:
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        model.fit(X_pool[train_idx], y_train)
        val_pred = model.predict(X_pool[val_idx])
        ext_pred = model.predict(X_external)
        rf_results.append((params, model, val_pred, ext_pred))
    simple_rf = [
        (params, model, val_pred, ext_pred)
        for params, model, val_pred, ext_pred in rf_results
        if params["max_features"] == 1.0 and params["max_depth"] is None
    ]
    selected_rf = max(
        simple_rf,
        key=lambda item: float(spearmanr(y_val, item[2])[0]),
    )
    rf_params, rf_model, rf_val_pred, rf_ext_pred = selected_rf
    print(
        f"  RF selected: {rf_params} val_rho="
        f"{spearmanr(y_val, rf_val_pred)[0]:+.3f}"
    )

    xgb_results: list[tuple[dict, XGBRegressor, np.ndarray, np.ndarray]] = []
    for params in XGB_GRID:
        xgb_params = {
            "n_estimators": 500,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
            **params,
        }
        model = XGBRegressor(**xgb_params)
        model.fit(X_pool[train_idx], y_train)
        val_pred = model.predict(X_pool[val_idx])
        ext_pred = model.predict(X_external)
        xgb_results.append((params, model, val_pred, ext_pred))
    selected_xgb = max(
        xgb_results,
        key=lambda item: float(spearmanr(y_val, item[2])[0]),
    )
    xgb_params, xgb_model, xgb_val_pred, xgb_ext_pred = selected_xgb
    print(
        f"  XGB selected: {xgb_params} val_rho="
        f"{spearmanr(y_val, xgb_val_pred)[0]:+.3f}"
    )

    def add_tree_summary(
        name: str,
        representation: str,
        params: dict,
        val_pred: np.ndarray,
        ext_pred: np.ndarray,
        score_pred: np.ndarray,
        model_type: str,
    ) -> None:
        ranks, in_top20 = control_rank_summary(score_pred, scoring)
        entry = {
            "model": name,
            "representation": representation,
            "selection": {
                "type": "grid",
                "model_type": model_type,
                "params": params,
                "validation_Spearman": float(spearmanr(y_val, val_pred)[0]),
            },
            "external_metrics": regression_metrics(y_external, ext_pred),
            "control_ranks": ranks,
            "controls_in_top20": in_top20,
        }
        model_summaries.append(entry)
        prediction_columns[name] = score_pred
        print(
            f"  {name}: external Spearman="
            f"{entry['external_metrics']['Spearman_rho']:+.3f}, "
            f"control ranks={ranks}"
        )

    rf_score_pred = rf_model.predict(X_scoring)
    xgb_score_pred = xgb_model.predict(X_scoring)
    add_tree_summary(
        "RandomForest", "Morgan fingerprint (radius=2, 2048 bits)",
        rf_params, rf_val_pred, rf_ext_pred, rf_score_pred, "RF",
    )
    add_tree_summary(
        "XGBoost", "Morgan fingerprint (radius=2, 2048 bits)",
        xgb_params, xgb_val_pred, xgb_ext_pred, xgb_score_pred, "XGB",
    )

    print_step("Step 4: graph models (same split, best-epoch selection)")
    graphs_train = [smiles_to_graph(s, float(y)) for s, y in zip(smiles_train, y_train)]
    graphs_val = [smiles_to_graph(s, 0.0) for s in smiles_val]
    graphs_external = [smiles_to_graph(s, 0.0) for s in smiles_external]
    graphs_scoring = [smiles_to_graph(s, 0.0) for s in smiles_scoring]
    if any(g is None for g in graphs_train + graphs_val + graphs_external + graphs_scoring):
        raise ValueError("Failed to convert a SMILES to graph")

    node_feature_dim = graphs_train[0].x.shape[1]
    external_graph_loader = GraphDataLoader(
        graphs_external, batch_size=256, shuffle=False
    )
    scoring_graph_loader = GraphDataLoader(
        graphs_scoring, batch_size=256, shuffle=False
    )

    gcn, gcn_best, gcn_history = train_graph_with_epoch_selection(
        "GCN",
        lambda: GCNRegressor(node_feature_dim),
        graphs_train,
        y_train,
        graphs_val,
        y_val,
    )
    gcn_ext_pred = graph_predict(gcn, external_graph_loader)
    gcn_score_pred = graph_predict(gcn, scoring_graph_loader)
    gcn_ranks, gcn_top20 = control_rank_summary(gcn_score_pred, scoring)
    model_summaries.append(
        {
            "model": "GCN",
            "representation": "molecular graph (atom features)",
            "selection": {
                "type": "best_epoch",
                "epoch": gcn_best["epoch"],
                "validation_Spearman": gcn_best["rho"],
                "history": gcn_history,
            },
            "external_metrics": regression_metrics(y_external, gcn_ext_pred),
            "control_ranks": gcn_ranks,
            "controls_in_top20": gcn_top20,
        }
    )
    prediction_columns["GCN"] = gcn_score_pred
    print(
        f"  GCN: external Spearman="
        f"{model_summaries[-1]['external_metrics']['Spearman_rho']:+.3f}, "
        f"control ranks={gcn_ranks}"
    )

    gin, gin_best, gin_history = train_graph_with_epoch_selection(
        "GIN",
        lambda: GINRegressor(
            node_feature_dim, edge_feature_dim=EDGE_FEATURE_DIM
        ),
        graphs_train,
        y_train,
        graphs_val,
        y_val,
    )
    gin_ext_pred = graph_predict(gin, external_graph_loader)
    gin_score_pred = graph_predict(gin, scoring_graph_loader)
    gin_ranks, gin_top20 = control_rank_summary(gin_score_pred, scoring)
    model_summaries.append(
        {
            "model": "GIN",
            "representation": "molecular graph (atom + bond features)",
            "selection": {
                "type": "best_epoch",
                "epoch": gin_best["epoch"],
                "validation_Spearman": gin_best["rho"],
                "history": gin_history,
            },
            "external_metrics": regression_metrics(y_external, gin_ext_pred),
            "control_ranks": gin_ranks,
            "controls_in_top20": gin_top20,
        }
    )
    prediction_columns["GIN"] = gin_score_pred
    print(
        f"  GIN: external Spearman="
        f"{model_summaries[-1]['external_metrics']['Spearman_rho']:+.3f}, "
        f"control ranks={gin_ranks}"
    )

    print_step("Step 5: SMILES Transformer (same split, best-epoch selection)")
    transformer, vocab, max_len, transformer_best, transformer_history = (
        train_transformer_with_epoch_selection(
            smiles_train,
            y_train,
            smiles_val,
            y_val,
        )
    )
    external_ids = [
        encode_smiles(s, vocab, max_len) for s in smiles_external
    ]
    scoring_ids = [encode_smiles(s, vocab, max_len) for s in smiles_scoring]
    external_dataset = TensorDataset(
        torch.tensor(external_ids, dtype=torch.long),
        torch.zeros(len(external_ids), dtype=torch.float),
    )
    scoring_dataset = TensorDataset(
        torch.tensor(scoring_ids, dtype=torch.long),
        torch.zeros(len(scoring_ids), dtype=torch.float),
    )
    transformer_ext_pred = transformer_predict(
        transformer,
        DataLoader(external_dataset, batch_size=128, shuffle=False),
    )
    transformer_score_pred = transformer_predict(
        transformer,
        DataLoader(scoring_dataset, batch_size=128, shuffle=False),
    )
    transformer_ranks, transformer_top20 = control_rank_summary(
        transformer_score_pred, scoring
    )
    model_summaries.append(
        {
            "model": "Transformer",
            "representation": "SMILES token sequence",
            "selection": {
                "type": "best_epoch",
                "epoch": transformer_best["epoch"],
                "validation_Spearman": transformer_best["rho"],
                "history": transformer_history,
            },
            "tokenizer": {"vocab_size": len(vocab), "max_len": max_len},
            "external_metrics": regression_metrics(
                y_external, transformer_ext_pred
            ),
            "control_ranks": transformer_ranks,
            "controls_in_top20": transformer_top20,
        }
    )
    prediction_columns["Transformer"] = transformer_score_pred
    print(
        f"  Transformer: external Spearman="
        f"{model_summaries[-1]['external_metrics']['Spearman_rho']:+.3f}, "
        f"control ranks={transformer_ranks}"
    )

    print_step("Step 6: save results")
    summary = {
        "milestone": "M6.5 fair comparison",
        "split": {
            "type": "candidate-like Tanimoto holdout",
            "fraction": 0.25,
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "n_external": int(len(y_external)),
        },
        "external_labels": "non-nM ChEMBL IC50 records converted to pIC50",
        "models": model_summaries,
        "caveats": [
            "All models were retrained on the same pool / split / seed 42.",
            "Trees use grid selection; deep models use best-epoch selection,"
            " both by candidate-like validation Spearman.",
            "Osimertinib is not in the local ChEMBL export, so positive"
            " controls are erlotinib, gefitinib, afatinib, lapatinib.",
        ],
    }
    summary_path = results_dir / "m65_fair_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    out = scoring.copy()
    for name, pred in prediction_columns.items():
        out[f"{name}_pred"] = pred
    out.to_csv(results_dir / "m65_fair_comparison_predictions.csv", index=False)
    print(f"  saved -> {summary_path}")
    print(f"  saved -> {results_dir / 'm65_fair_comparison_predictions.csv'}")

    names = [entry["model"] for entry in model_summaries]
    ext_rhos = [entry["external_metrics"]["Spearman_rho"] for entry in model_summaries]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0" if n == "RandomForest" else "#55A868" for n in names]
    bars = ax.barh(np.arange(len(names)), ext_rhos, color=colors)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("external Spearman")
    ax.set_title("M6.5 fair comparison (same split / tuning principle)")
    for bar, rho in zip(bars, ext_rhos):
        ax.text(
            rho + (0.01 if rho >= 0 else -0.06),
            bar.get_y() + bar.get_height() / 2,
            f"{rho:+.3f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig_path = figures_dir / "m65_fair_comparison.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  saved -> {fig_path}")

    print_step("M6.5 fair comparison complete")
    for entry in model_summaries:
        print(
            f"  {entry['model']:<12} ext_rho="
            f"{entry['external_metrics']['Spearman_rho']:+.3f} "
            f"controls={entry['control_ranks']}"
        )


if __name__ == "__main__":
    main()
