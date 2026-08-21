#!/usr/bin/env python3
"""
M9 P0-1: retrospective virtual-screening benchmark.

This turns the pipeline from a reproduction exercise into a measurable
research question:

  Can the RF + Transformer screening ensemble rank known EGFR-active
  molecules above weak/inactive molecules on molecules it has never seen?

Protocol
--------
1. Use the M3 Murcko scaffold split (8,128 train / 2,033 test) so every test
   scaffold is unseen.
2. Retrain RF, XGBoost and the SMILES Transformer on the scaffold train
   split only. The Transformer uses an internal random 10% of the train for
   best-epoch selection, never the test split.
3. On the test split, define:
      active   : pIC50 >= 7.0
      inactive : pIC50 < 6.0
   (the middle band is excluded from binary metrics).
4. Report ROC-AUC, PR-AUC (average precision), enrichment factors EF1%/EF5%,
   and Spearman on the full test set.

Run with the project conda environment (rdkit + torch + sklearn):

    conda activate egfr-aidd
    python src/models/m9_retrospective_benchmark.py

Usage (from the project root)
-----------------------------
    python src/models/m9_retrospective_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.evaluate_splits import scaffold_split_indices
from src.models.m65_fair_comparison import (
    train_transformer_with_epoch_selection,
)
from src.models.m65_rf_transformer_ensemble import (
    RF_PARAMS,
    predict_transformer,
)
from src.models.morgan_baseline import smiles_to_morgan_fingerprint


XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "min_child_weight": 3,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "verbosity": 0,
}


def morgan_matrix(smiles_list: list[str]) -> np.ndarray:
    return np.vstack(
        [smiles_to_morgan_fingerprint(s) for s in smiles_list]
    ).astype(np.float32)


def enrichment_factor(scores: np.ndarray, y: np.ndarray, fraction: float) -> float:
    order = np.argsort(-scores)
    n_select = max(1, int(fraction * len(y)))
    selected_pos = int(y[order[:n_select]].sum())
    total_pos = int(y.sum())
    if total_pos == 0:
        return 0.0
    return (selected_pos / n_select) / (total_pos / len(y))


def benchmark_metrics(
    scores_full: np.ndarray,
    y_full: np.ndarray,
    active_mask: np.ndarray,
) -> dict:
    scores = scores_full[active_mask]
    y = (y_full[active_mask] >= 7.0).astype(int)
    return {
        "n_active": int(y.sum()),
        "n_inactive": int((y == 0).sum()),
        "ROC_AUC": round(float(roc_auc_score(y, scores)), 4),
        "PR_AUC": round(float(average_precision_score(y, scores)), 4),
        "EF1%": round(float(enrichment_factor(scores, y, 0.01)), 3),
        "EF5%": round(float(enrichment_factor(scores, y, 0.05)), 3),
        "Spearman_full_test": round(
            float(spearmanr(y_full, scores_full)[0]), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 retrospective benchmark")
    parser.add_argument("--data-file", default="data/processed/egfr_activity_final.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_file)
    smiles_all = df["canonical_smiles"].tolist()
    y_all = df["pIC50"].to_numpy(dtype=np.float32)
    train_idx, test_idx, _ = scaffold_split_indices(
        smiles_all, test_size=0.2
    )
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    print(f"scaffold split: train={len(train_idx)}, test={len(test_idx)}")

    smiles_train = [smiles_all[i] for i in train_idx]
    smiles_test = [smiles_all[i] for i in test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    X_train = morgan_matrix(smiles_train)
    X_test = morgan_matrix(smiles_test)

    print("training RF and XGBoost ...")
    rf = RandomForestRegressor(random_state=args.seed, n_jobs=-1, **RF_PARAMS)
    rf.fit(X_train, y_train)
    rf_test = rf.predict(X_test)

    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X_train, y_train)
    xgb_test = xgb.predict(X_test)

    # Internal random 10% validation for Transformer epoch selection only.
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(train_idx))
    n_val = max(1, int(0.1 * len(train_idx)))
    val_pos = perm[:n_val]
    tr_pos = perm[n_val:]
    smiles_tr = [smiles_train[i] for i in tr_pos]
    y_tr = y_train[tr_pos]
    smiles_val = [smiles_train[i] for i in val_pos]
    y_val = y_train[val_pos]

    print("training Transformer (best-epoch on internal validation) ...")
    transformer, vocab, max_len, _, _ = (
        train_transformer_with_epoch_selection(
            smiles_tr,
            y_tr,
            smiles_val,
            y_val,
            seed=args.seed,
        )
    )
    transformer_test = predict_transformer(
        transformer, smiles_test, vocab, max_len
    )

    ensemble_test = 0.5 * (rf_test + transformer_test)

    active_mask = (y_test >= 7.0) | (y_test < 6.0)
    y_binary = (y_test[active_mask] >= 7.0).astype(int)

    predictions = pd.DataFrame(
        {
            "molecule_chembl_id": [df.iloc[i]["molecule_chembl_id"] for i in test_idx],
            "canonical_smiles": smiles_test,
            "pIC50": y_test,
            "active_label": np.where(y_test >= 7.0, 1, np.where(y_test < 6.0, 0, np.nan)),
            "RF_pred": rf_test,
            "XGB_pred": xgb_test,
            "Transformer_pred": transformer_test,
            "RF_Transformer_ensemble_pred": ensemble_test,
        }
    )
    predictions.to_csv(
        results_dir / "m9_retrospective_predictions.csv", index=False
    )

    models = {
        "RandomForest": rf_test,
        "XGBoost": xgb_test,
        "Transformer": transformer_test,
        "RF_Transformer_ensemble": ensemble_test,
    }
    metrics = {
        name: benchmark_metrics(scores, y_test, active_mask)
        for name, scores in models.items()
    }
    metrics["Random_chance"] = {
        "n_active": int(y_binary.sum()),
        "n_inactive": int((y_binary == 0).sum()),
        "ROC_AUC": 0.5,
        "PR_AUC": round(
            float(y_binary.mean()), 4
        ),
        "EF1%": 1.0,
        "EF5%": 1.0,
        "Spearman_full_test": 0.0,
    }

    summary = {
        "milestone": "M9 P0-1 retrospective screening benchmark",
        "split": {
            "type": "Murcko scaffold",
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "binary_definition": {
            "active": "pIC50 >= 7.0",
            "inactive": "pIC50 < 6.0",
            "excluded": "6.0 <= pIC50 < 7.0",
        },
        "n_binary_test": int(active_mask.sum()),
        "metrics": metrics,
        "models": list(models.keys()),
    }
    json_path = results_dir / "m9_retrospective_benchmark.json"
    json_path.write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7, 7))
    for name, scores in models.items():
        fpr, tpr, _ = roc_curve(y_binary, scores[active_mask])
        ax.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['ROC_AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("M9 retrospective screening benchmark (scaffold test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig_path = figures_dir / "m9_retrospective_roc.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print("=" * 70)
    print("M9 P0-1 retrospective screening benchmark")
    print("=" * 70)
    print(f"  binary test molecules: {int(active_mask.sum())}")
    for name, m in metrics.items():
        print(
            f"  {name:<24} AUC={m['ROC_AUC']:.3f} PR-AUC={m['PR_AUC']:.3f} "
            f"EF1%={m['EF1%']:.2f} EF5%={m['EF5%']:.2f} "
            f"Spearman={m['Spearman_full_test']:.3f}"
        )
    print(f"  saved -> {json_path}")
    print(f"  saved -> {results_dir / 'm9_retrospective_predictions.csv'}")
    print(f"  saved -> {fig_path}")


if __name__ == "__main__":
    main()
