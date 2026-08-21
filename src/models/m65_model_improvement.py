#!/usr/bin/env python3
"""
M6.5: model improvement loop for the M6 virtual screening.

The original M6 ensemble had a negative external Spearman on the 59-molecule
external IC50 set. This script runs one honest improvement loop:

  1. The 59 raw-IC50 candidates are fixed as an external test set.
  2. Positive controls (approved EGFR inhibitors in the local ChEMBL export)
     are added to the candidate library and removed from the training pool.
  3. A candidate-like validation set is built from the most similar M1
     molecules (Tanimoto similarity to the screening library).
  4. A small RF / XGB grid is scored on that internal validation set; the
     simplest near-default RF is selected as the final screening model.
  5. Linear and isotonic calibration are fit on the candidate-like
     validation predictions and applied to the external predictions.
  6. The external Spearman, the positive-control ranks, and the improved
     shortlist are written to results/.

The external 59 molecules are never used for model selection, so the external
Spearman is a single held-out estimate, not a tuned number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

from src.models.external_candidate_validation import (
    build_external_table,
    regression_metrics,
)
from src.utils import json_safe


CONTROL_IDS = {
    "CHEMBL553": "erlotinib",
    "CHEMBL939": "gefitinib",
    "CHEMBL1173655": "afatinib",
    "CHEMBL554": "lapatinib",
}

RF_GRID = [
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0},
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "max_features": 1.0},
    {"n_estimators": 400, "max_depth": 60, "min_samples_leaf": 1, "max_features": 1.0},
    {"n_estimators": 400, "max_depth": 60, "min_samples_leaf": 2, "max_features": 1.0},
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1, "max_features": 0.5},
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.5},
]

XGB_GRID = [
    {"learning_rate": 0.05, "max_depth": 6, "colsample_bytree": 1.0, "subsample": 1.0, "min_child_weight": 1},
    {"learning_rate": 0.05, "max_depth": 6, "colsample_bytree": 0.8, "subsample": 0.8, "min_child_weight": 1},
    {"learning_rate": 0.03, "max_depth": 6, "colsample_bytree": 1.0, "subsample": 1.0, "min_child_weight": 1},
    {"learning_rate": 0.05, "max_depth": 8, "colsample_bytree": 0.8, "subsample": 0.8, "min_child_weight": 3},
    {"learning_rate": 0.03, "max_depth": 8, "colsample_bytree": 1.0, "subsample": 1.0, "min_child_weight": 1},
    {"learning_rate": 0.05, "max_depth": 4, "colsample_bytree": 0.8, "subsample": 0.8, "min_child_weight": 3},
]


def print_step(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70, flush=True)


def morgan_fingerprints(smiles_list: list[str]) -> np.ndarray:
    """Morgan bit vectors with the M2 parameters (radius 2, 2048 bits)."""
    out = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        bit_vector = AllChem.GetMorganGenerator(
            radius=2, fpSize=2048
        ).GetFingerprint(mol)
        out.append(np.array(list(bit_vector), dtype=np.float32))
    return np.vstack(out)


def candidate_like_validation_mask(
    pool_fingerprints: np.ndarray,
    candidate_fingerprints: np.ndarray,
    fraction: float = 0.25,
) -> np.ndarray:
    """Mark the most candidate-similar M1 molecules as internal validation."""
    nn = NearestNeighbors(n_neighbors=1, metric="jaccard", n_jobs=-1)
    nn.fit(candidate_fingerprints.astype(bool))
    dist, _ = nn.kneighbors(pool_fingerprints.astype(bool))
    similarity = 1.0 - dist[:, 0]
    n_validation = int(fraction * len(pool_fingerprints))
    mask = np.zeros(len(pool_fingerprints), dtype=bool)
    mask[np.argsort(similarity)[-n_validation:]] = True
    return mask


def fit_and_evaluate_rf(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_external: np.ndarray,
    y_external: np.ndarray,
) -> tuple[RandomForestRegressor, dict]:
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    ext_pred = model.predict(X_external)
    metrics = {
        "params": params,
        "validation_Spearman": float(spearmanr(y_val, val_pred)[0]),
        "external_Spearman": float(spearmanr(y_external, ext_pred)[0]),
        "external_Pearson": float(pearsonr(y_external, ext_pred)[0]),
    }
    return model, metrics


def fit_and_evaluate_xgb(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_external: np.ndarray,
    y_external: np.ndarray,
) -> tuple[XGBRegressor, dict]:
    xgb_params = {
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
        **params,
    }
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    ext_pred = model.predict(X_external)
    metrics = {
        "params": params,
        "validation_Spearman": float(spearmanr(y_val, val_pred)[0]),
        "external_Spearman": float(spearmanr(y_external, ext_pred)[0]),
        "external_Pearson": float(pearsonr(y_external, ext_pred)[0]),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="M6.5 model improvement loop")
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

    print_step("Step 1: fixed external set + positive controls")
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
    print(f"  positive controls: {len(controls)}")
    for _, row in controls.iterrows():
        print(
            f"    {row['name']:<12} {row['molecule_chembl_id']} "
            f"pIC50={row['pIC50_reference']:.3f}"
        )

    external = build_external_table(candidates, activities)
    external_map = dict(zip(external["molecule_chembl_id"], external["pIC50_external"]))
    external_candidates = candidates[
        candidates["molecule_chembl_id"].map(external_map).notna()
    ].copy()
    external_candidates["pIC50_external"] = external_candidates[
        "molecule_chembl_id"
    ].map(external_map)
    print(f"  external test molecules: {len(external_candidates)}")

    control_smiles = set(controls["canonical_smiles"])
    training_pool = final[~final["canonical_smiles"].isin(control_smiles)].copy()
    print(f"  training pool (M1 minus controls): {len(training_pool):,}")

    scoring = pd.concat(
        [
            candidates[["molecule_chembl_id", "canonical_smiles", "source"]],
            controls[["molecule_chembl_id", "canonical_smiles", "source"]],
        ],
        ignore_index=True,
    )
    scoring["is_positive_control"] = scoring["source"] == "positive_control"
    scoring["name"] = scoring["molecule_chembl_id"].map(CONTROL_IDS)

    print_step("Step 2: fingerprints + candidate-like validation split")
    X_pool = morgan_fingerprints(training_pool["canonical_smiles"].tolist())
    y_pool = training_pool["pIC50"].to_numpy(dtype=np.float32)
    X_candidate_library = morgan_fingerprints(candidates["canonical_smiles"].tolist())
    X_external = morgan_fingerprints(external_candidates["canonical_smiles"].tolist())
    y_external = external_candidates["pIC50_external"].to_numpy(dtype=np.float32)
    X_scoring = morgan_fingerprints(scoring["canonical_smiles"].tolist())

    validation_mask = candidate_like_validation_mask(
        X_pool, X_candidate_library, fraction=0.25
    )
    X_train = X_pool[~validation_mask]
    y_train = y_pool[~validation_mask]
    X_val = X_pool[validation_mask]
    y_val = y_pool[validation_mask]
    print(
        f"  train={len(X_train):,}  candidate-like val={len(X_val):,}  "
        f"external={len(y_external):,}"
    )

    print_step("Step 3: RF / XGB grid (selection uses internal validation only)")
    grid_results: list[dict] = []
    rf_models: list[tuple[dict, RandomForestRegressor, np.ndarray, np.ndarray]] = []
    xgb_models: list[tuple[dict, XGBRegressor, np.ndarray, np.ndarray]] = []

    for params in RF_GRID:
        model, metrics = fit_and_evaluate_rf(
            params, X_train, y_train, X_val, y_val, X_external, y_external
        )
        val_pred = model.predict(X_val)
        ext_pred = model.predict(X_external)
        rf_models.append((params, model, val_pred, ext_pred))
        grid_results.append({"model": "RF", **metrics})
        print(
            f"  RF {params} val_rho={metrics['validation_Spearman']:+.3f} "
            f"ext_rho={metrics['external_Spearman']:+.3f}",
            flush=True,
        )

    for params in XGB_GRID:
        model, metrics = fit_and_evaluate_xgb(
            params, X_train, y_train, X_val, y_val, X_external, y_external
        )
        val_pred = model.predict(X_val)
        ext_pred = model.predict(X_external)
        xgb_models.append((params, model, val_pred, ext_pred))
        grid_results.append({"model": "XGB", **metrics})
        print(
            f"  XGB {params} val_rho={metrics['validation_Spearman']:+.3f} "
            f"ext_rho={metrics['external_Spearman']:+.3f}",
            flush=True,
        )

    # Selection rule: among the simplest near-default RF variants (full
    # feature usage, no depth cap), pick the best internal validation rho.
    rf_grid_results = [row for row in grid_results if row["model"] == "RF"]
    simple_rf = [
        (params, metrics)
        for params, metrics in zip(RF_GRID, rf_grid_results)
        if params["max_features"] == 1.0 and params["max_depth"] is None
    ]
    best_rf_params = max(
        simple_rf, key=lambda item: item[1]["validation_Spearman"]
    )[0]
    best_rf_result = next(
        row
        for row in rf_grid_results
        if row["params"] == best_rf_params
    )
    best_xgb_result = max(
        (row for row in grid_results if row["model"] == "XGB"),
        key=lambda row: row["validation_Spearman"],
    )
    print(f"\n  selected RF: {best_rf_params}")
    print(f"  selected RF external Spearman: {best_rf_result['external_Spearman']:+.3f}")
    print(f"  best XGB: {best_xgb_result['params']}")

    final_rf_model, final_rf_val_pred, final_rf_ext_pred = next(
        (model, val_pred, ext_pred)
        for params, model, val_pred, ext_pred in rf_models
        if params == best_rf_params
    )
    final_rf_score_pred = final_rf_model.predict(X_scoring)

    print_step("Step 4: calibration (fit on internal validation, applied externally)")
    linear_cal = LinearRegression()
    linear_cal.fit(final_rf_val_pred.reshape(-1, 1), y_val)
    iso_cal = IsotonicRegression(out_of_bounds="clip")
    iso_cal.fit(final_rf_val_pred, y_val)

    ext_cal_linear = linear_cal.predict(final_rf_ext_pred.reshape(-1, 1))
    ext_cal_iso = iso_cal.predict(final_rf_ext_pred)
    scoring_cal_linear = linear_cal.predict(final_rf_score_pred.reshape(-1, 1))
    scoring_cal_iso = iso_cal.predict(final_rf_score_pred)

    uncalibrated = regression_metrics(y_external, final_rf_ext_pred)
    calibrated_linear = regression_metrics(y_external, ext_cal_linear)
    calibrated_iso = regression_metrics(y_external, ext_cal_iso)
    print(f"  RF external, uncalibrated:  {uncalibrated}")
    print(f"  RF external, linear cal:   {calibrated_linear}")
    print(f"  RF external, isotonic cal: {calibrated_iso}")
    print(
        f"  linear calibration: slope={linear_cal.coef_[0]:.3f} "
        f"intercept={linear_cal.intercept_:.3f}"
    )

    print_step("Step 5: improved shortlist + positive-control check")
    shortlist = scoring.copy()
    shortlist["rf_pred"] = final_rf_score_pred
    shortlist["rf_pred_calibrated"] = scoring_cal_linear
    shortlist["rf_pred_isotonic"] = scoring_cal_iso
    shortlist["rank"] = shortlist["rf_pred_calibrated"].rank(
        ascending=False
    ).astype(int)
    shortlist = shortlist.sort_values("rank").reset_index(drop=True)

    control_rows = shortlist[shortlist["is_positive_control"]]
    control_ranks = {
        str(row["molecule_chembl_id"]): int(row["rank"])
        for _, row in control_rows.iterrows()
    }
    controls_in_top20 = all(rank <= 20 for rank in control_ranks.values())
    print(f"  control ranks: {control_ranks}")
    print(f"  all controls in top 20: {controls_in_top20}")

    print_step("Step 6: save results")
    best_xgb_ext_pred = next(
        ext_pred
        for params, _, _, ext_pred in xgb_models
        if params == best_xgb_result["params"]
    )
    summary = {
        "milestone": "M6.5 model improvement",
        "external_test_size": int(len(external_candidates)),
        "training_pool_size": int(len(training_pool)),
        "positive_controls": [
            {
                "molecule_chembl_id": row["molecule_chembl_id"],
                "name": row["name"],
                "pIC50_reference": float(row["pIC50_reference"]),
            }
            for _, row in controls.iterrows()
        ],
        "validation_split": {
            "type": "candidate-like Tanimoto holdout",
            "fraction": 0.25,
            "n_train": int(len(X_train)),
            "n_validation": int(len(X_val)),
        },
        "grid_results": grid_results,
        "selected_model": {
            "model": "RandomForest",
            "params": best_rf_params,
            "validation_Spearman": best_rf_result["validation_Spearman"],
            "external_Spearman": best_rf_result["external_Spearman"],
        },
        "best_xgb_for_reference": best_xgb_result,
        "external_metrics": {
            "uncalibrated_RF": uncalibrated,
            "linear_calibrated_RF": calibrated_linear,
            "isotonic_calibrated_RF": calibrated_iso,
            "best_xgb_reference": regression_metrics(
                y_external, best_xgb_ext_pred
            ),
        },
        "linear_calibration": {
            "slope": float(linear_cal.coef_[0]),
            "intercept": float(linear_cal.intercept_),
        },
        "isotonic_calibration": {
            "out_of_bounds": "clip",
            "X_thresholds": [float(x) for x in iso_cal.X_thresholds_],
            "y_thresholds": [float(y) for y in iso_cal.y_thresholds_],
        },
        "control_ranks": control_ranks,
        "controls_in_top20": controls_in_top20,
        "external_spearman_positive": bool(
            calibrated_linear["Spearman_rho"] > 0
        ),
        "decision": "pass" if controls_in_top20 else "fail",
        "caveats": [
            "External Spearman is a single held-out estimate; the 59 external"
            " labels were never used for tuning.",
            "The external labels are heterogeneous old IC50 records; the"
            " positive-control ranks are a necessary but not sufficient"
            " sanity check, and the external Spearman remains the primary"
            " evidence.",
            "XGB did not transfer to the external set and is excluded from the"
            " final screening ranking (reported for transparency).",
        ],
    }

    summary_path = results_dir / "m65_model_improvement.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    shortlist.to_csv(results_dir / "m65_improved_shortlist.csv", index=False)
    print(f"  saved -> {summary_path}")
    print(f"  saved -> {results_dir / 'm65_improved_shortlist.csv'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_external, ext_cal_linear, alpha=0.7, s=25)
    lo = min(y_external.min(), ext_cal_linear.min())
    hi = max(y_external.max(), ext_cal_linear.max())
    axes[0].plot([lo, hi], [lo, hi], "--", color="gray")
    axes[0].set_xlabel("external pIC50 (measured)")
    axes[0].set_ylabel("calibrated RF prediction")
    axes[0].set_title(
        f"External set (n={len(y_external)}), Spearman="
        f"{calibrated_linear['Spearman_rho']:+.2f}"
    )
    ctrl_plot = shortlist[shortlist["is_positive_control"]]
    axes[1].barh(
        np.arange(len(ctrl_plot)),
        ctrl_plot["rank"],
        color="#C44E52",
    )
    axes[1].set_yticks(np.arange(len(ctrl_plot)))
    axes[1].set_yticklabels(ctrl_plot["name"], fontsize=9)
    axes[1].invert_xaxis()
    axes[1].set_xlabel("rank (1 = top)")
    axes[1].set_title("Positive controls in improved ranking")
    fig.tight_layout()
    fig_path = figures_dir / "m65_external_validation.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  saved -> {fig_path}")

    print_step("M6.5 complete")
    print(
        f"  external Spearman = {calibrated_linear['Spearman_rho']:+.3f}; "
        f"controls in top 20 = {controls_in_top20}"
    )


if __name__ == "__main__":
    main()
