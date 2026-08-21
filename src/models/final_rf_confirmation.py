#!/usr/bin/env python3
"""
Final locked Random Forest configuration confirmation.

This script marks the end of hyperparameter tuning for the Morgan-RF model.
It trains the two locked configurations on the standard-scaffold TRAIN split
only and evaluates them on the standard-scaffold TEST split:

  - original baseline: max_features = 1.0;
  - locked tuned model: max_features = 0.20.

All other settings are identical:

  - Morgan radius = 2, nBits = 2048;
  - n_estimators = 300, random_state = 42, n_jobs = -1, min_samples_leaf = 1,
    max_depth = None;
  - standard 80/10/10 Bemis-Murcko scaffold split, seed 42.

Reported test metrics: MAE, RMSE, R2, Spearman, and screening metrics
ROC-AUC / PR-AUC using the established activity definition active =
pIC50 >= 7. The screening metrics are benchmark-specific and are kept
separate from the M9 retrospective screening benchmark. Paired bootstrap
(1,000 resamples) provides 95% confidence intervals for the metrics and for
tuned-minus-baseline differences. No parameter is changed after seeing test
results.

Outputs
-------
results/final_rf_configuration.json
results/final_rf_test_results.json
results/final_rf_test_predictions.csv

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/final_rf_confirmation.py
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.models.evaluate_splits import (
    json_safe,
    package_versions,
    smiles_to_morgan_fingerprint,
)
from src.models.portfolio_hardening_analysis import (
    standard_scaffold_split_indices,
)


RADIUS = 2
N_BITS = 2048
N_ESTIMATORS = 300
SEED = 42
MIN_SAMPLES_LEAF = 1
MAX_DEPTH = None
BASELINE_MAX_FEATURES = 1.0
TUNED_MAX_FEATURES = 0.20
N_BOOTSTRAP = 1000
ACTIVE_THRESHOLD = 7.0


def make_rf(max_features: float) -> RandomForestRegressor:
    """Locked RandomForest with only max_features differing."""
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=max_features,
        max_depth=MAX_DEPTH,
    )


def morgan_bit_vectors(smiles_list: list[str]) -> list:
    generator = AllChem.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def max_train_tanimoto(
    smiles_train: list[str], smiles_test: list[str]
) -> list[float]:
    train_bits = morgan_bit_vectors(smiles_train)
    test_bits = morgan_bit_vectors(smiles_test)
    return [
        float(max(DataStructs.BulkTanimotoSimilarity(fp, train_bits)))
        for fp in test_bits
    ]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred)[0]),
    }


def screening_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    active = (y_true >= ACTIVE_THRESHOLD).astype(int)
    if active.sum() == 0 or (active == 0).sum() == 0:
        return {"ROC_AUC": None, "PR_AUC": None}
    return {
        "ROC_AUC": float(roc_auc_score(active, y_pred)),
        "PR_AUC": float(average_precision_score(active, y_pred)),
    }


def bootstrap_ci(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final locked RF confirmation (test evaluation)"
    )
    parser.add_argument(
        "--data-file", default="data/processed/egfr_activity_final.csv"
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_file)
    smiles_all = df["canonical_smiles"].tolist()
    y_all = df["pIC50"].to_numpy(dtype=np.float32)
    ids_all = df["molecule_chembl_id"].tolist()

    std_train, std_val, std_test, _, split_meta = (
        standard_scaffold_split_indices(
            smiles_all,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            seed=42,
        )
    )
    print(
        f"standard scaffold split: train={len(std_train)} "
        f"val={len(std_val)} test={len(std_test)}"
    )

    smiles_train = [smiles_all[i] for i in std_train]
    smiles_test = [smiles_all[i] for i in std_test]
    y_train = y_all[std_train]
    y_test = y_all[std_test]

    print("building fingerprints (train/test)...", flush=True)
    X_train = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in smiles_train]
    ).astype(np.float32)
    X_test = np.vstack(
        [smiles_to_morgan_fingerprint(s, RADIUS, N_BITS) for s in smiles_test]
    ).astype(np.float32)

    print("computing max_train_tanimoto for test molecules...", flush=True)
    tanimoto_test = max_train_tanimoto(smiles_train, smiles_test)

    baseline = make_rf(BASELINE_MAX_FEATURES)
    tuned = make_rf(TUNED_MAX_FEATURES)
    print("training baseline RF (max_features=1.0)...", flush=True)
    baseline.fit(X_train, y_train)
    print("training tuned RF (max_features=0.20)...", flush=True)
    tuned.fit(X_train, y_train)

    pred_baseline = baseline.predict(X_test)
    pred_tuned = tuned.predict(X_test)

    m_baseline = {
        **regression_metrics(y_test, pred_baseline),
        **screening_metrics(y_test, pred_baseline),
    }
    m_tuned = {
        **regression_metrics(y_test, pred_tuned),
        **screening_metrics(y_test, pred_tuned),
    }
    metrics = ["MAE", "RMSE", "R2", "Spearman", "ROC_AUC", "PR_AUC"]
    deltas = {
        metric: (
            m_tuned[metric] - m_baseline[metric]
            if m_tuned[metric] is not None and m_baseline[metric] is not None
            else None
        )
        for metric in metrics
    }

    print("\nbaseline test metrics:")
    for metric in metrics:
        val = m_baseline[metric]
        print(f"  {metric}: {'n/a' if val is None else round(val, 4)}")
    print("tuned test metrics:")
    for metric in metrics:
        val = m_tuned[metric]
        print(f"  {metric}: {'n/a' if val is None else round(val, 4)}")
    print("tuned - baseline:")
    for metric in metrics:
        val = deltas[metric]
        print(f"  {metric}: {'n/a' if val is None else round(val, 4)}")

    # Paired bootstrap on test molecules.
    rng = np.random.default_rng(args.bootstrap_seed)
    n_test = len(y_test)
    boot_metrics: dict[str, dict[str, np.ndarray]] = {
        metric: {"baseline": np.zeros(args.n_bootstrap), "tuned": np.zeros(args.n_bootstrap)}
        for metric in metrics
    }
    boot_delta: dict[str, np.ndarray] = {
        metric: np.zeros(args.n_bootstrap) for metric in metrics
    }
    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n_test, size=n_test)
        y_b = y_test[idx]
        p_b = pred_baseline[idx]
        p_t = pred_tuned[idx]
        rm_b = {
            **regression_metrics(y_b, p_b),
            **screening_metrics(y_b, p_b),
        }
        rm_t = {
            **regression_metrics(y_b, p_t),
            **screening_metrics(y_b, p_t),
        }
        for metric in metrics:
            if rm_b[metric] is not None and rm_t[metric] is not None:
                boot_metrics[metric]["baseline"][b] = rm_b[metric]
                boot_metrics[metric]["tuned"][b] = rm_t[metric]
                boot_delta[metric][b] = rm_t[metric] - rm_b[metric]
            else:
                boot_delta[metric][b] = np.nan

    bootstrap_out: dict = {}
    delta_out: dict = {}
    for metric in metrics:
        mask = ~np.isnan(boot_delta[metric])
        if mask.sum() < 50:
            bootstrap_out[metric] = {
                "baseline": None,
                "tuned": None,
                "n_valid_resamples": int(mask.sum()),
            }
            delta_out[metric] = None
            continue
        bootstrap_out[metric] = {
            "baseline": bootstrap_ci(
                boot_metrics[metric]["baseline"][mask]
            ),
            "tuned": bootstrap_ci(boot_metrics[metric]["tuned"][mask]),
            "n_valid_resamples": int(mask.sum()),
        }
        delta_out[metric] = bootstrap_ci(boot_delta[metric][mask])

    print("\nbootstrap 95% CI (tuned - baseline):")
    for metric in metrics:
        if delta_out[metric] is None:
            print(f"  {metric}: n/a")
        else:
            d = delta_out[metric]
            print(
                f"  {metric}: mean={d['mean']:.4f} "
                f"CI95=[{d['ci95_low']:.4f}, {d['ci95_high']:.4f}]"
            )

    # Machine-readable locked configuration.
    configuration = {
        "tuning_complete": True,
        "note": (
            "Locked final Morgan-RF configuration. The standard scaffold "
            "test set was not used for model selection during the controlled "
            "hyperparameter study."
        ),
        "morgan": {"type": "Morgan", "radius": RADIUS, "nBits": N_BITS},
        "random_forest": {
            "n_estimators": N_ESTIMATORS,
            "random_state": SEED,
            "n_jobs": -1,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": TUNED_MAX_FEATURES,
            "max_depth": None,
            "baseline_max_features": BASELINE_MAX_FEATURES,
        },
        "split": {
            **split_meta,
            "n_train": len(std_train),
            "n_validation": len(std_val),
            "n_test": len(std_test),
        },
        "primary_selection_metric": "validation Spearman",
        "secondary_selection_metrics": [
            "validation R2",
            "validation RMSE",
            "validation MAE",
        ],
        "versions": package_versions(),
    }
    config_path = results_dir / "final_rf_configuration.json"
    config_path.write_text(
        json.dumps(json_safe(configuration), indent=2, allow_nan=False)
    )

    test_results = {
        "milestone": "final locked RF test confirmation",
        "test_set_used": True,
        "split": {
            **split_meta,
            "n_train": len(std_train),
            "n_validation": len(std_val),
            "n_test": len(std_test),
        },
        "locked_configuration": configuration,
        "activity_definition": f"active = pIC50 >= {ACTIVE_THRESHOLD}",
        "test_n": n_test,
        "test_active_fraction": float((y_test >= ACTIVE_THRESHOLD).mean()),
        "benchmark_note": (
            "ROC-AUC / PR-AUC are reported on the standard-scaffold test "
            "split and must not be mixed with the M9 retrospective screening "
            "benchmark, which uses a different external screening set."
        ),
        "metric_interpretation": {
            "MAE": "lower is better",
            "RMSE": "lower is better",
            "R2": "higher is better",
            "Spearman": "higher is better",
            "ROC_AUC": "higher is better",
            "PR_AUC": "higher is better",
            "delta_direction": "tuned minus baseline",
        },
        "baseline": m_baseline,
        "tuned": m_tuned,
        "delta_tuned_minus_baseline": deltas,
        "bootstrap": {
            "method": "paired resampling of test molecules with replacement",
            "n_resamples": args.n_bootstrap,
            "seed": args.bootstrap_seed,
            "metrics": bootstrap_out,
            "delta_tuned_minus_baseline_ci95": delta_out,
        },
        "versions": package_versions(),
    }
    results_path = results_dir / "final_rf_test_results.json"
    results_path.write_text(
        json.dumps(json_safe(test_results), indent=2, allow_nan=False)
    )

    pred_df = pd.DataFrame(
        {
            "molecule_chembl_id": [ids_all[i] for i in std_test],
            "canonical_smiles": smiles_test,
            "true_pIC50": y_test,
            "baseline_prediction": pred_baseline,
            "tuned_prediction": pred_tuned,
            "abs_error_baseline": np.abs(pred_baseline - y_test),
            "abs_error_tuned": np.abs(pred_tuned - y_test),
            "max_train_tanimoto": tanimoto_test,
        }
    )
    pred_path = results_dir / "final_rf_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print(f"\nsaved -> {config_path}")
    print(f"saved -> {results_path}")
    print(f"saved -> {pred_path}")


if __name__ == "__main__":
    main()
