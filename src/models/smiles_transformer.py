#!/usr/bin/env python3
"""
M5: SMILES Transformer baseline for EGFR pIC50 regression.

Same experiment design as M3/M4:

  canonical_smiles
    -> tokenized SMILES sequence
    -> Transformer encoder
    -> pIC50 regression

The Transformer uses the same Murcko scaffold split as M3 and M4, so all
milestones are compared on exactly the same test molecules.

Usage (from the project root)
-----------------------------
    python src/models/smiles_transformer.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

# macOS conda environments often load two OpenMP runtimes (xgboost + torch).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_splits import (
    distribution_stats,
    evaluate_regression,
    json_safe,
    package_versions as m3_package_versions,
    scaffold_split_indices,
)


TWO_CHAR_ELEMENTS = {
    "Al", "As", "Au", "Ba", "Be", "Bi", "Br", "Ca", "Cd", "Ce", "Cl",
    "Co", "Cr", "Cs", "Cu", "Dy", "Er", "Eu", "Fe", "Ga", "Gd", "Ge",
    "He", "Hf", "Hg", "Ho", "In", "Ir", "La", "Li", "Lu", "Mg", "Mn",
    "Mo", "Na", "Nb", "Nd", "Ne", "Ni", "Os", "Pb", "Pd", "Pm", "Po",
    "Pr", "Pt", "Rb", "Re", "Rh", "Rn", "Ru", "Sb", "Sc", "Se", "Si",
    "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm",
    "Xe", "Yb", "Zn", "Zr",
}


def print_step(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def tokenize_smiles(smiles: str) -> list[str]:
    """
    Tokenize a SMILES string into atom-level and punctuation tokens.

    Examples:
        CCO -> ["C", "C", "O"]
        CCl -> ["C", "Cl"]
        c1ccccc1 -> ["c", "1", "c", "c", "c", "c", "c", "1"]
    """
    tokens: list[str] = []
    i = 0
    while i < len(smiles):
        if smiles[i] == "[":
            end = smiles.find("]", i)
            if end == -1:
                end = len(smiles) - 1
            tokens.append(smiles[i : end + 1])
            i = end + 1
        elif i + 1 < len(smiles) and smiles[i : i + 2] in TWO_CHAR_ELEMENTS:
            tokens.append(smiles[i : i + 2])
            i += 2
        else:
            tokens.append(smiles[i])
            i += 1
    return tokens


def build_vocab(smiles_list: list[str]) -> dict[str, int]:
    """Build a deterministic token vocabulary from training SMILES only."""
    counter: Counter = Counter()
    for smiles in smiles_list:
        counter.update(tokenize_smiles(smiles))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token in sorted(counter):
        vocab[token] = len(vocab)
    return vocab


def encode_smiles(
    smiles: str,
    vocab: dict[str, int],
    max_len: int,
) -> list[int]:
    """Encode one SMILES into a padded integer sequence."""
    tokens = tokenize_smiles(smiles)
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids + [vocab["<pad>"]] * (max_len - len(ids))


class SMILESTransformerRegressor(nn.Module):
    """
    Small Transformer encoder for pIC50 regression.

    Token and position embeddings are summed, encoded by a Transformer
    encoder, pooled over non-padding tokens, and mapped to one scalar.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = 0
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.position_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids == self.pad_idx
        positions = torch.arange(
            input_ids.size(1),
            device=input_ids.device,
        ).unsqueeze(0)
        x = self.token_embed(input_ids) + self.position_embed(positions)
        x = self.transformer(x, src_key_padding_mask=mask)

        valid = ~mask
        x = x * valid.unsqueeze(-1)
        pooled = x.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    for input_ids, targets in loader:
        optimizer.zero_grad()
        pred = model(input_ids)
        loss = F.mse_loss(pred, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
    return total_loss / len(loader.dataset)


def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, _ in loader:
            predictions.append(model(input_ids).cpu().numpy())
    return np.concatenate(predictions)


def m5_package_versions() -> dict:
    versions = m3_package_versions()
    versions["torch"] = torch.__version__
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 SMILES Transformer baseline")
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
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    print_step("Step 1: load M1 dataset and reuse scaffold split")
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
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  train={len(train_idx):,}  test={len(test_idx):,}")
    print(
        f"  scaffolds: {scaffold_stats['n_scaffolds']:,} "
        f"(train {scaffold_stats['scaffolds_train']:,}, "
        f"test {scaffold_stats['scaffolds_test']:,})"
    )

    # ------------------------------------------------------------------
    # Step 2: tokenize SMILES and build vocabulary from training set
    # ------------------------------------------------------------------
    print_step("Step 2: tokenize SMILES")
    train_smiles = [smiles_list[i] for i in train_idx]
    vocab = build_vocab(train_smiles)

    token_lengths = [len(tokenize_smiles(s)) for s in smiles_list]
    actual_max_len = max(token_lengths)
    max_len = args.max_len
    if actual_max_len > max_len:
        print(
            f"  [warn] longest SMILES has {actual_max_len} tokens; "
            f"truncating to max_len={max_len}"
        )
    else:
        max_len = actual_max_len

    all_ids = [
        encode_smiles(s, vocab, max_len)
        for s in smiles_list
    ]
    train_ids = [all_ids[i] for i in train_idx]
    test_ids = [all_ids[i] for i in test_idx]

    train_dataset = TensorDataset(
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_ids, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.float),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"  vocab size: {len(vocab):,}")
    print(f"  max token length: {max_len}")
    print(f"  token length mean: {np.mean(token_lengths):.1f}")

    # ------------------------------------------------------------------
    # Step 3: train the Transformer
    # ------------------------------------------------------------------
    print_step("Step 3: train SMILES Transformer")
    torch.manual_seed(args.seed)
    model = SMILESTransformerRegressor(
        vocab_size=len(vocab),
        max_len=max_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer)
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:02d}/{args.epochs} train_loss={loss:.4f}")

    y_pred = predict(model, test_loader)
    metrics = evaluate_regression(y_test, y_pred)
    elapsed = time.time() - start
    print(f"  Transformer done in {elapsed:.1f}s")
    for key, value in metrics.items():
        print(f"    {key}: {value:.4f}")

    n_params = sum(p.numel() for p in model.parameters())

    # ------------------------------------------------------------------
    # Step 4: save results and comparison figure
    # ------------------------------------------------------------------
    print_step("Step 4: save results and comparison figure")
    summary = {
        "milestone": "M5",
        "representation": "SMILES Transformer (learned sequence embedding)",
        "tokenizer": {
            "type": "atom-level SMILES tokens",
            "vocab_size": int(len(vocab)),
            "max_len": int(max_len),
            "actual_max_token_length": int(actual_max_len),
        },
        "model_params": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "n_parameters": int(n_params),
        },
        "versions": m5_package_versions(),
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
    summary_path = results_dir / "m5_smiles_transformer_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved -> {summary_path}")

    # Compare with M3 Morgan and M4 graph baselines.
    model_names = ["RandomForest", "XGBoost", "GCN", "GIN", "Transformer"]
    r2_values: list[float] = []
    rmse_values: list[float] = []

    m3_path = results_dir / "m3_morgan_split_comparison_results.json"
    if m3_path.exists():
        m3 = json.loads(m3_path.read_text())["splits"]["scaffold"]["metrics"]
        r2_values += [m3["RandomForest"]["R2"], m3["XGBoost"]["R2"]]
        rmse_values += [m3["RandomForest"]["RMSE"], m3["XGBoost"]["RMSE"]]
    else:
        model_names = model_names[2:]

    m4_path = results_dir / "m4_graph_baseline_results.json"
    if m4_path.exists():
        m4 = json.loads(m4_path.read_text())["splits"]["scaffold"]["metrics"]
        r2_values += [m4["GCN"]["R2"], m4["GIN"]["R2"]]
        rmse_values += [m4["GCN"]["RMSE"], m4["GIN"]["RMSE"]]
    else:
        model_names = [n for n in model_names if n not in {"GCN", "GIN"}]

    r2_values.append(metrics["R2"])
    rmse_values.append(metrics["RMSE"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(model_names))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    axes[0].bar(x, r2_values, 0.55, color=colors[: len(model_names)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].set_ylabel("R2")
    axes[0].set_title("Scaffold split R2 (M3/M4/M5)")

    axes[1].bar(x, rmse_values, 0.55, color=colors[: len(model_names)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Scaffold split RMSE (M3/M4/M5)")
    fig.tight_layout()
    fig_path = figures_dir / "m5_smiles_transformer_scaffold_comparison.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  saved -> {fig_path}")

    print_step("M5 complete")
    print(
        "Compare the SMILES Transformer with Morgan, GCN, and GIN on the "
        "same scaffold split."
    )


if __name__ == "__main__":
    main()
