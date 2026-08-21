#!/usr/bin/env python3
"""
Controlled hyperparameter study: Morgan fingerprint radius.

This study changes exactly one modeling decision at a time: the Morgan
fingerprint radius (1, 2, 3). Everything else stays fixed:

  - the standard 80/10/10 Bemis-Murcko scaffold split (seed 42);
  - training and validation molecules only (the standard-scaffold TEST set
    is never touched);
  - nBits = 2048;
  - RandomForest hyperparameters from M2/M3 (300 trees, random_state 42);
  - preprocessing, endpoint definition, and all other settings.

Models are selected on the validation split with Spearman as the primary
metric; RMSE / MAE / R2 are secondary. The study also reports fingerprint
sparsity and validation-set maximum Tanimoto similarity to the training set.

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/radius_tuning_study.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from scipy.stats import spearmanr

from src.models.evaluate_splits import (
    evaluate_regression,
    json_safe,
    make_models,
    package_versions,
    smiles_to_morgan_fingerprint,
)
from src.models.portfolio_hardening_analysis import (
    standard_scaffold_split_indices,
)


N_BITS = 2048
RADII = [1, 2, 3]


def morgan_bit_vectors(
    smiles_list: list[str], radius: int, n_bits: int = N_BITS
) -> list:
    """ExplicitBitVectors for bulk Tanimoto similarity at one radius."""
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def max_train_tanimoto_stats(
    smiles_train: list[str], smiles_val: list[str], radius: int
) -> dict:
    train_bits = morgan_bit_vectors(smiles_train, radius)
    val_bits = morgan_bit_vectors(smiles_val, radius)
    maxima = np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(fp, train_bits)) for fp in val_bits],
        dtype=float,
    )
    return {
        "mean": float(np.mean(maxima)),
        "median": float(np.median(maxima)),
        "min": float(np.min(maxima)),
        "max": float(np.max(maxima)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Morgan fingerprint radius study (validation only)"
    )
    parser.add_argument(
        "--data-file", default="data/processed/egfr_activity_final.csv"
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-bits", type=int, default=N_BITS)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_file)
    smiles_all = df["canonical_smiles"].tolist()
    y_all = df["pIC50"].to_numpy(dtype=np.float32)

    std_train, std_val, std_test, _, split_meta = (
        standard_scaffold_split_indices(
            smiles_all,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            seed=42,
        )
    )
    assert len(std_test) > 0
    print(
        f"standard scaffold split: train={len(std_train)} "
        f"val={len(std_val)} test={len(std_test)} (test not used)"
    )

    smiles_train = [smiles_all[i] for i in std_train]
    smiles_val = [smiles_all[i] for i in std_val]
    y_train = y_all[std_train]
    y_val = y_all[std_val]

    rf_params = make_models(300, 300, 42)["RandomForest"].get_params()
    rows: list[dict] = []
    detail: dict = {}

    for radius in RADII:
        print(f"\n--- radius={radius} ---", flush=True)
        X_train = np.vstack(
            [smiles_to_morgan_fingerprint(s, radius, args.n_bits) for s in smiles_train]
        ).astype(np.float32)
        X_val = np.vstack(
            [smiles_to_morgan_fingerprint(s, radius, args.n_bits) for s in smiles_val]
        ).astype(np.float32)

        rf = make_models(300, 300, 42)["RandomForest"]
        rf.fit(X_train, y_train)
        pred = rf.predict(X_val)
        metrics = evaluate_regression(y_val, pred)
        spearman_val = float(spearmanr(y_val, pred)[0])

        on_bits_train = X_train.sum(axis=1)
        mean_on_bits = float(np.mean(on_bits_train))
        median_on_bits = float(np.median(on_bits_train))
        density = mean_on_bits / args.n_bits

        sim = max_train_tanimoto_stats(smiles_train, smiles_val, radius)

        rows.append(
            {
                "radius": radius,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Spearman": spearman_val,
                "mean_on_bits": mean_on_bits,
                "median_on_bits": median_on_bits,
                "fingerprint_density": density,
                "mean_max_train_tanimoto": sim["mean"],
                "median_max_train_tanimoto": sim["median"],
                "min_max_train_tanimoto": sim["min"],
                "max_max_train_tanimoto": sim["max"],
            }
        )
        detail[str(radius)] = {
            "metrics": metrics,
            "spearman_validation": spearman_val,
            "fingerprint_stats": {
                "mean_on_bits": mean_on_bits,
                "median_on_bits": median_on_bits,
                "density": density,
            },
            "validation_max_train_tanimoto": sim,
            "rf_params": rf.get_params(),
        }
        print(
            f"  MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
            f"R2={metrics['R2']:.4f} Spearman={spearman_val:.4f} "
            f"mean_on_bits={mean_on_bits:.2f} "
            f"mean_max_train_tanimoto={sim['mean']:.4f}"
        )

    out = pd.DataFrame(rows)
    csv_path = results_dir / "radius_tuning_results.csv"
    out.to_csv(csv_path, index=False)

    summary = {
        "milestone": "radius tuning study",
        "validation_only": True,
        "test_set_used": False,
        "fingerprint": {"type": "Morgan", "n_bits": args.n_bits, "radii": RADII},
        "split": {
            **split_meta,
            "n_train": len(std_train),
            "n_validation": len(std_val),
            "n_test": len(std_test),
        },
        "rf_params": json_safe(rf_params),
        "primary_metric": "validation Spearman",
        "secondary_metrics": ["validation RMSE", "validation MAE", "validation R2"],
        "versions": package_versions(),
        "results": rows,
        "detail": detail,
    }
    json_path = results_dir / "radius_tuning_results.json"
    json_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {json_path}")


if __name__ == "__main__":
    main()
