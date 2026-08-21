#!/usr/bin/env python3
"""
M4: molecular graph baselines (GCN and GIN) for EGFR pIC50 regression.

Same experiment design as M3:

  canonical_smiles
    -> RDKit Mol
    -> molecular graph (atom features + bond features)
    -> GCN / GIN
    -> pIC50 regression

The graph models use the same Murcko scaffold split as M3, so their
new-chemistry test set is exactly the same as the Morgan baseline test set.

Usage (from the project root)
-----------------------------
    python src/models/graph_baseline.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# macOS conda environments often load two OpenMP runtimes (xgboost + torch).
# Set this before importing either library.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GINEConv,
    global_max_pool,
    global_mean_pool,
)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_splits import (
    distribution_stats,
    evaluate_regression,
    json_safe,
    package_versions as m3_package_versions,
    scaffold_split_indices,
)


ATOMIC_NUMS = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
EDGE_FEATURE_DIM = 7


def print_step(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def one_hot(value, choices: list) -> list[float]:
    """One-hot encoding with an extra 'other' slot for unseen values."""
    return [float(value == choice) for choice in choices] + [0.0]


def atom_features(atom) -> list[float]:
    """Small, interpretable RDKit atom feature vector."""
    features: list[float] = []
    features += one_hot(atom.GetAtomicNum(), ATOMIC_NUMS)
    features += one_hot(int(atom.GetDegree()), list(range(6)))
    features += one_hot(int(atom.GetFormalCharge()), [-2, -1, 0, 1, 2])
    features += one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.GetTotalNumHs()))
    features.append(float(atom.IsInRing()))
    return features


def bond_features(bond) -> list[float]:
    """Small, interpretable RDKit bond feature vector."""
    features: list[float] = []
    features += one_hot(bond.GetBondType(), BOND_TYPES)
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))
    return features


def smiles_to_graph(smiles: str, y: float) -> Data | None:
    """Convert a SMILES string into a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    edge_index: list[tuple[int, int]] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))
        bf = bond_features(bond)
        edge_attr.append(bf)
        edge_attr.append(bf)

    if edge_index:
        edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index_t = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index_t,
        edge_attr=edge_attr_t,
        y=torch.tensor([y], dtype=torch.float),
        smiles=smiles,
    )


class GCNRegressor(nn.Module):
    """Two-layer GCN using atom features only, with mean + max pooling."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_channels * 2, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        pooled = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=1,
        )
        return self.lin(pooled).squeeze(-1)


class GINRegressor(nn.Module):
    """Two-layer GINE (edge-feature-aware GIN) with mean + max pooling."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.1,
        edge_feature_dim: int = EDGE_FEATURE_DIM,
    ):
        super().__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINEConv(mlp1, edge_dim=edge_feature_dim)
        self.conv2 = GINEConv(mlp2, edge_dim=edge_feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_channels * 2, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index, data.edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, data.edge_attr))
        pooled = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=1,
        )
        return self.lin(pooled).squeeze(-1)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    for data in loader:
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, data.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for data in loader:
            predictions.append(model(data).cpu().numpy())
    return np.concatenate(predictions)


def graph_package_versions() -> dict:
    versions = m3_package_versions()
    versions.update(
        {
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        }
    )
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 GCN/GIN graph baseline")
    parser.add_argument(
        "--data-file",
        default="data/processed/egfr_activity_final.csv",
        help="M1 final molecule-level dataset.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Where to save the results JSON.",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/figures",
        help="Where to save the evaluation figures.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Step 1: load the M1 dataset and reuse the M3 scaffold split
    # ------------------------------------------------------------------
    print_step("Step 1: load M1 dataset and build molecular graphs")
    df = pd.read_csv(args.data_file)
    required = ["canonical_smiles", "pIC50"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    smiles_list = df["canonical_smiles"].tolist()
    y = df["pIC50"].to_numpy(dtype=np.float32)

    train_idx, test_idx, scaffold_stats = scaffold_split_indices(
        smiles_list, test_size=args.test_size
    )
    print(f"  train={len(train_idx):,}  test={len(test_idx):,}")
    print(
        f"  scaffolds: {scaffold_stats['n_scaffolds']:,} "
        f"(train {scaffold_stats['scaffolds_train']:,}, "
        f"test {scaffold_stats['scaffolds_test']:,})"
    )

    graphs = []
    invalid = 0
    for smi, target in zip(smiles_list, y):
        graph = smiles_to_graph(smi, float(target))
        if graph is None:
            invalid += 1
            continue
        graphs.append(graph)
    if invalid:
        raise ValueError(f"{invalid} SMILES could not be converted to graphs")

    node_feature_dim = graphs[0].x.shape[1]
    n_nodes_mean = float(np.mean([g.num_nodes for g in graphs]))
    n_edges_mean = float(np.mean([g.num_edges for g in graphs]))
    print(f"  graphs: {len(graphs):,}")
    print(f"  node feature dim: {node_feature_dim}")
    print(f"  mean nodes/graph: {n_nodes_mean:.1f}, mean edges/graph: {n_edges_mean:.1f}")

    graphs_train = [graphs[i] for i in train_idx]
    graphs_test = [graphs[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # ------------------------------------------------------------------
    # Step 2: train GCN and GIN under the same scaffold split
    # ------------------------------------------------------------------
    print_step("Step 2: train GCN and GIN")
    model_factories = {
        "GCN": lambda: GCNRegressor(
            in_channels=node_feature_dim,
            hidden_channels=args.hidden,
            dropout=args.dropout,
        ),
        "GIN": lambda: GINRegressor(
            in_channels=node_feature_dim,
            hidden_channels=args.hidden,
            dropout=args.dropout,
            edge_feature_dim=EDGE_FEATURE_DIM,
        ),
    }

    metrics: dict = {}
    for name, factory in model_factories.items():
        torch.manual_seed(args.seed)
        model = factory()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        train_loader = DataLoader(
            graphs_train,
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            graphs_test,
            batch_size=args.batch_size,
            shuffle=False,
        )

        start = time.time()
        print(f"\n  training {name} ...")
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer)
            if epoch % 10 == 0 or epoch == args.epochs:
                print(f"    epoch {epoch:02d}/{args.epochs} train_loss={loss:.4f}")

        y_pred = predict(model, test_loader)
        metrics[name] = evaluate_regression(y_test, y_pred)
        elapsed = time.time() - start
        print(f"  {name} done in {elapsed:.1f}s")
        for key, value in metrics[name].items():
            print(f"    {key}: {value:.4f}")

    # ------------------------------------------------------------------
    # Step 3: save results
    # ------------------------------------------------------------------
    print_step("Step 3: save results and comparison figure")
    summary = {
        "milestone": "M4",
        "representation": "molecular graph (GCN node-only / GIN with bond features)",
        "graph_params": {
            "node_feature_dim": int(node_feature_dim),
            "edge_feature_dim": EDGE_FEATURE_DIM,
            "edge_features_used": {"GCN": False, "GIN": True},
            "hidden_channels": args.hidden,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
        },
        "graph_stats": {
            "n_graphs": len(graphs),
            "n_nodes_mean": n_nodes_mean,
            "n_edges_mean": n_edges_mean,
        },
        "versions": graph_package_versions(),
        "split_test_size": args.test_size,
        "seed": args.seed,
        "splits": {
            "scaffold": {
                "type": "murcko_scaffold",
                "test_size": args.test_size,
                "test_frac_actual": float(len(test_idx) / len(df)),
                "train_indices": train_idx,
                "test_indices": test_idx,
                "pIC50_train_stats": distribution_stats(y_train),
                "pIC50_test_stats": distribution_stats(y_test),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                **scaffold_stats,
                "metrics": metrics,
            }
        },
    }
    summary_path = results_dir / "m4_graph_baseline_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved -> {summary_path}")

    # Compare with M3 Morgan scaffold baseline.
    m3_path = results_dir / "m3_morgan_split_comparison_results.json"
    if m3_path.exists():
        m3 = json.loads(m3_path.read_text())
        m3_scaffold_metrics = m3["splits"]["scaffold"]["metrics"]
        model_names = ["RandomForest", "XGBoost", "GCN", "GIN"]
        r2_values = [
            m3_scaffold_metrics["RandomForest"]["R2"],
            m3_scaffold_metrics["XGBoost"]["R2"],
            metrics["GCN"]["R2"],
            metrics["GIN"]["R2"],
        ]
        rmse_values = [
            m3_scaffold_metrics["RandomForest"]["RMSE"],
            m3_scaffold_metrics["XGBoost"]["RMSE"],
            metrics["GCN"]["RMSE"],
            metrics["GIN"]["RMSE"],
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(model_names))
        width = 0.55
        axes[0].bar(x, r2_values, width, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names)
        axes[0].set_ylabel("R2")
        axes[0].set_title("Scaffold split R2 (M3 Morgan vs M4 graph)")

        axes[1].bar(x, rmse_values, width, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names)
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("Scaffold split RMSE (M3 Morgan vs M4 graph)")

        fig.tight_layout()
        fig_path = figures_dir / "m4_graph_baseline_scaffold_comparison.png"
        fig.savefig(fig_path, dpi=150)
        print(f"  saved -> {fig_path}")

    print_step("M4 complete")
    print("Compare GCN/GIN with RandomForest/XGBoost on the same scaffold split.")


if __name__ == "__main__":
    main()
