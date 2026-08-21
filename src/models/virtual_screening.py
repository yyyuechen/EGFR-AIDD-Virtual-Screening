#!/usr/bin/env python3
"""
M6: Ensemble virtual screening for EGFR inhibitors.

Pipeline
--------
M1 dataset (full) is used to train:

  Morgan fingerprint   -> RandomForest / XGBoost
  molecular graph      -> GCN / GIN
  SMILES sequence      -> Transformer encoder

The three representations are then used to score a candidate library of
molecules that were NOT part of the M1 modeling dataset (excluded ChEMBL
records). Predictions are averaged into an ensemble pIC50 and ranked.

Usage (from the project root)
-----------------------------
    python src/models/virtual_screening.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GraphDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess_egfr import standardize_molecule
from src.models.graph_baseline import (
    GCNRegressor,
    GINRegressor,
    EDGE_FEATURE_DIM,
    predict as graph_predict,
    smiles_to_graph,
    train_epoch as graph_train_epoch,
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
    print("=" * 70)


def build_candidate_library(
    data_dir: Path,
    candidates_dir: Path,
) -> pd.DataFrame:
    """
    Build a local candidate library from ChEMBL molecules that were not used
    in the M1 modeling dataset.
    """
    molecules = pd.read_csv(data_dir / "raw" / "egfr_molecules_smiles.csv")
    final = pd.read_csv(data_dir / "processed" / "egfr_activity_final.csv")
    final_smiles = set(final["canonical_smiles"])

    candidates = molecules[~molecules["canonical_smiles"].isin(final_smiles)].copy()
    candidates = candidates.dropna(subset=["canonical_smiles"])
    candidates = candidates.drop_duplicates(subset=["molecule_chembl_id"])

    standardized = candidates["canonical_smiles"].map(standardize_molecule)
    candidates["canonical_smiles"] = [
        x[0] if x is not None else None for x in standardized
    ]
    candidates = candidates.dropna(subset=["canonical_smiles"])
    candidates = candidates[~candidates["canonical_smiles"].isin(final_smiles)]
    candidates = candidates.drop_duplicates(subset=["canonical_smiles"])
    candidates["source"] = "chembl_nonmodeled"

    candidates_dir.mkdir(parents=True, exist_ok=True)
    out_path = candidates_dir / "egfr_candidate_library.csv"
    candidates[["molecule_chembl_id", "canonical_smiles", "source"]].to_csv(
        out_path, index=False
    )
    print(f"  candidate library saved -> {out_path}")
    return candidates


def train_morgan_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    rf_trees: int,
    xgb_trees: int,
) -> dict:
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=rf_trees, random_state=seed, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=xgb_trees,
            learning_rate=0.05,
            max_depth=6,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        ),
    }
    for name, model in models.items():
        print(f"  training {name} ...")
        model.fit(X_train, y_train)
    return models


def train_graph_model(
    name: str,
    factory,
    graphs_train: list,
    y_train: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = factory()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loader = GraphDataLoader(graphs_train, batch_size=batch_size, shuffle=True)
    print(f"  training {name} ...")
    for epoch in range(1, epochs + 1):
        loss = graph_train_epoch(model, loader, optimizer)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"    epoch {epoch:02d}/{epochs} train_loss={loss:.4f}")
    return model


def train_transformer(
    smiles_train: list[str],
    y_train: np.ndarray,
    seed: int,
    max_len_arg: int,
    epochs: int,
    batch_size: int,
) -> tuple[SMILESTransformerRegressor, dict[str, int], int]:
    torch.manual_seed(seed)
    vocab = build_vocab(smiles_train)
    train_lengths = [len(tokenize_smiles(s)) for s in smiles_train]
    max_len = max(train_lengths)
    if max_len > max_len_arg:
        print(f"  [warn] training max_len {max_len} > --max-len {max_len_arg}")
    max_len = min(max_len, max_len_arg)

    ids = [encode_smiles(s, vocab, max_len) for s in smiles_train]
    dataset = TensorDataset(
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SMILESTransformerRegressor(
        vocab_size=len(vocab),
        max_len=max_len,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    print("  training Transformer ...")
    for epoch in range(1, epochs + 1):
        loss = transformer_train_epoch(model, loader, optimizer)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"    epoch {epoch:02d}/{epochs} train_loss={loss:.4f}")
    return model, vocab, max_len


def main() -> None:
    parser = argparse.ArgumentParser(description="M6 ensemble virtual screening")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--candidates-dir", default="data/candidates")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--models-dir", default="results/models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--xgb-trees", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    candidates_dir = Path(args.candidates_dir)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    models_dir = Path(args.models_dir)
    for d in (candidates_dir, results_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Step 1: load M1 dataset and build candidate library
    # ------------------------------------------------------------------
    print_step("Step 1: load M1 dataset and candidate library")
    final_df = pd.read_csv(data_dir / "processed" / "egfr_activity_final.csv")
    candidates = build_candidate_library(data_dir, candidates_dir)
    print(f"  training molecules: {len(final_df):,}")
    print(f"  candidates: {len(candidates):,}")

    train_smiles = final_df["canonical_smiles"].tolist()
    y_train = final_df["pIC50"].to_numpy(dtype=np.float32)
    candidate_smiles = candidates["canonical_smiles"].tolist()

    # ------------------------------------------------------------------
    # Step 2: Morgan fingerprints and models
    # ------------------------------------------------------------------
    print_step("Step 2: Morgan fingerprint models (full dataset)")
    X_train = np.vstack(
        [smiles_to_morgan_fingerprint(s, radius=2, n_bits=2048) for s in train_smiles]
    ).astype(np.float32)
    X_candidates = np.vstack(
        [
            smiles_to_morgan_fingerprint(s, radius=2, n_bits=2048)
            for s in candidate_smiles
        ]
    ).astype(np.float32)
    morgan_models = train_morgan_models(
        X_train,
        y_train,
        seed=args.seed,
        rf_trees=args.rf_trees,
        xgb_trees=args.xgb_trees,
    )
    joblib.dump(morgan_models["RandomForest"], models_dir / "rf_egfr.joblib")
    joblib.dump(morgan_models["XGBoost"], models_dir / "xgb_egfr.joblib")

    rf_pred = morgan_models["RandomForest"].predict(X_candidates)
    xgb_pred = morgan_models["XGBoost"].predict(X_candidates)
    morgan_pred = 0.5 * (rf_pred + xgb_pred)

    # ------------------------------------------------------------------
    # Step 3: graph models (full dataset)
    # ------------------------------------------------------------------
    print_step("Step 3: graph models (full dataset)")
    graphs_train = [smiles_to_graph(s, float(y)) for s, y in zip(train_smiles, y_train)]
    if any(g is None for g in graphs_train):
        raise ValueError("Failed to convert a training SMILES to graph")
    graphs_candidates = [smiles_to_graph(s, 0.0) for s in candidate_smiles]
    if any(g is None for g in graphs_candidates):
        raise ValueError("Failed to convert a candidate SMILES to graph")

    node_feature_dim = graphs_train[0].x.shape[1]
    gcn = train_graph_model(
        "GCN",
        lambda: GCNRegressor(node_feature_dim),
        graphs_train,
        y_train,
        args.seed,
        args.epochs,
        args.batch_size,
    )
    gin = train_graph_model(
        "GIN",
        lambda: GINRegressor(
            node_feature_dim, edge_feature_dim=EDGE_FEATURE_DIM
        ),
        graphs_train,
        y_train,
        args.seed,
        args.epochs,
        args.batch_size,
    )
    torch.save(gcn.state_dict(), models_dir / "gcn_egfr.pt")
    torch.save(gin.state_dict(), models_dir / "gin_egfr.pt")

    cand_graph_loader = GraphDataLoader(
        graphs_candidates, batch_size=args.batch_size
    )
    gcn_pred = graph_predict(gcn, cand_graph_loader)
    gin_pred = graph_predict(gin, cand_graph_loader)
    graph_pred = 0.5 * (gcn_pred + gin_pred)

    # ------------------------------------------------------------------
    # Step 4: Transformer model (full dataset)
    # ------------------------------------------------------------------
    print_step("Step 4: SMILES Transformer (full dataset)")
    transformer, vocab, max_len = train_transformer(
        train_smiles,
        y_train,
        seed=args.seed,
        max_len_arg=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    torch.save(transformer.state_dict(), models_dir / "transformer_egfr.pt")
    (models_dir / "smiles_vocab.json").write_text(json.dumps(vocab))

    cand_ids = [encode_smiles(s, vocab, max_len) for s in candidate_smiles]
    cand_dataset = TensorDataset(
        torch.tensor(cand_ids, dtype=torch.long),
        torch.zeros(len(cand_ids), dtype=torch.float),
    )
    cand_loader = DataLoader(cand_dataset, batch_size=args.batch_size)
    transformer_pred = transformer_predict(transformer, cand_loader)

    # ------------------------------------------------------------------
    # Step 5: ensemble ranking and outputs
    # ------------------------------------------------------------------
    print_step("Step 5: ensemble ranking and outputs")
    ensemble_pic50 = (morgan_pred + graph_pred + transformer_pred) / 3.0
    out = candidates[["molecule_chembl_id", "canonical_smiles", "source"]].copy()
    out["rf_pred"] = rf_pred
    out["xgb_pred"] = xgb_pred
    out["morgan_pred"] = morgan_pred
    out["gcn_pred"] = gcn_pred
    out["gin_pred"] = gin_pred
    out["graph_pred"] = graph_pred
    out["transformer_pred"] = transformer_pred
    out["ensemble_pic50"] = ensemble_pic50
    out["ensemble_rank"] = out["ensemble_pic50"].rank(ascending=False).astype(int)
    out = out.sort_values("ensemble_rank").reset_index(drop=True)

    shortlist = out.head(args.top_n)
    shortlist.to_csv(results_dir / "m6_virtual_screening_shortlist.csv", index=False)

    summary = {
        "milestone": "M6",
        "ensemble": {
            "strategy": "mean of Morgan / graph / Transformer pIC50 predictions",
            "morgan_models": ["RandomForest", "XGBoost"],
            "graph_models": ["GCN", "GIN"],
            "transformer_model": "SMILES Transformer",
            "top_n": args.top_n,
        },
        "candidate_library": str(candidates_dir / "egfr_candidate_library.csv"),
        "n_candidates": int(len(out)),
        "n_training_molecules": int(len(final_df)),
        "seed": args.seed,
        "model_configs": {
            "morgan": {
                "radius": 2,
                "n_bits": 2048,
                "RandomForest": {"n_estimators": args.rf_trees},
                "XGBoost": {
                    "n_estimators": args.xgb_trees,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                },
            },
            "graph": {
                "node_feature_dim": int(node_feature_dim),
                "edge_feature_dim": EDGE_FEATURE_DIM,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
            "transformer": {
                "vocab_size": int(len(vocab)),
                "max_len": int(max_len),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
        },
        "candidate_predictions": out.to_dict(orient="records"),
        "shortlist": shortlist.to_dict(orient="records"),
    }
    summary_path = results_dir / "m6_virtual_screening_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved -> {summary_path}")
    print(f"  saved -> {results_dir / 'm6_virtual_screening_shortlist.csv'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(out["ensemble_pic50"], bins=20, color="#4C72B0")
    axes[0].set_xlabel("ensemble predicted pIC50")
    axes[0].set_ylabel("number of candidates")
    axes[0].set_title("Ensemble score distribution")

    top = out.head(20).iloc[::-1]
    labels = [s[:25] + ("..." if len(s) > 25 else "") for s in top["canonical_smiles"]]
    axes[1].barh(np.arange(len(top)), top["ensemble_pic50"], color="#55A868")
    axes[1].set_yticks(np.arange(len(top)))
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xlabel("ensemble predicted pIC50")
    axes[1].set_title("Top 20 candidates")
    fig.tight_layout()
    fig_path = figures_dir / "m6_virtual_screening_shortlist.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  saved -> {fig_path}")

    print_step("M6 complete")
    print(f"  top candidate: {shortlist.iloc[0]['molecule_chembl_id']}")


if __name__ == "__main__":
    main()
