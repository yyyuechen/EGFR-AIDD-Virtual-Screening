#!/usr/bin/env python3
"""
M6 external candidate validation.

The M6 candidate library was built from ChEMBL EGFR IC50 records that were
excluded from the M1 modeling dataset because their units are not nM
(mostly ug/mL, plus a few uM and one "10^3 uM" record). Those raw records
are still usable as a crude external check: convert each IC50 to pIC50,
then compare the M6 ensemble predictions with measured activity.

Usage (from the project root)
-----------------------------
    python src/models/external_candidate_validation.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr, spearmanr


PREDICTION_COLUMNS = [
    "rf_pred",
    "xgb_pred",
    "morgan_pred",
    "gcn_pred",
    "gin_pred",
    "graph_pred",
    "transformer_pred",
    "ensemble_pic50",
]


def ic50_to_pic50(value: float, units: str, mol) -> float | None:
    """Convert one ChEMBL IC50 record into pIC50."""
    mw = Descriptors.MolWt(mol)
    if not mw or mw <= 0:
        return None
    unit = units.strip()
    if unit == "nM":
        ic50_nm = value
    elif unit in ("uM", "µM", "μM", "/uM"):
        ic50_nm = value * 1e3
    elif unit in ("ug.mL-1", "ug/mL"):
        # ug/mL == mg/L; mg/L / (g/mol) gives mmol/L == mM.
        ic50_mm = value / mw
        ic50_nm = ic50_mm * 1e6
    elif unit == "10^3 uM":
        # Literal ChEMBL unit: 10^3 uM == 1 mM.
        ic50_nm = value * 1e6
    else:
        return None
    if ic50_nm <= 0:
        return None
    return 9.0 - math.log10(ic50_nm)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pearson = pearsonr(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "MAE": mae,
        "RMSE": rmse,
        "Pearson_r": float(pearson.statistic),
        "Pearson_p": float(pearson.pvalue),
        "Spearman_rho": float(spearman.statistic),
        "Spearman_p": float(spearman.pvalue),
    }


def build_external_table(
    candidates: pd.DataFrame,
    activities: pd.DataFrame,
) -> pd.DataFrame:
    """Convert raw IC50 records into a best-value-per-molecule table."""
    smiles_by_id = dict(
        zip(candidates["molecule_chembl_id"], candidates["canonical_smiles"])
    )
    rows: list[dict] = []
    numeric = activities.copy()
    numeric["standard_value_numeric"] = pd.to_numeric(
        numeric["standard_value"], errors="coerce"
    )
    for record in numeric.dropna(subset=["standard_value_numeric"]).itertuples():
        mol = Chem.MolFromSmiles(smiles_by_id.get(record.molecule_chembl_id, ""))
        if mol is None:
            continue
        pic50 = ic50_to_pic50(
            float(record.standard_value_numeric),
            str(record.standard_units),
            mol,
        )
        if pic50 is None:
            continue
        ic50_nm = 10 ** (9.0 - pic50)
        rows.append(
            {
                "molecule_chembl_id": record.molecule_chembl_id,
                "pIC50_external": pic50,
                "IC50_nM_external": ic50_nm,
                "external_units": record.standard_units,
                "external_year": record.document_year,
            }
        )
    external = pd.DataFrame(rows)
    if external.empty:
        return external
    # Keep the most potent measurement when multiple records exist.
    return (
        external.sort_values("pIC50_external", ascending=False)
        .drop_duplicates("molecule_chembl_id", keep="first")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="M6 external validation")
    parser.add_argument(
        "--candidates-file",
        default="data/candidates/egfr_candidate_library.csv",
    )
    parser.add_argument(
        "--activities-file",
        default="data/raw/egfr_chembl_activities_raw.csv",
    )
    parser.add_argument(
        "--m6-results-file",
        default="results/m6_virtual_screening_results.json",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates_file)
    activities = pd.read_csv(args.activities_file)
    with open(args.m6_results_file) as fh:
        m6 = json.load(fh)
    predictions = pd.DataFrame(m6["candidate_predictions"])

    external = build_external_table(candidates, activities)
    merged = predictions.merge(external, on="molecule_chembl_id", how="inner")
    if len(merged) < 10:
        raise ValueError(
            "External merge too small; check candidate IDs or unit handling."
        )

    summary = {
        "milestone": "M6 external validation",
        "n_candidates": int(len(predictions)),
        "n_with_external_ic50": int(len(merged)),
        "external_source": "raw ChEMBL IC50 records excluded from M1 by units",
        "external_pIC50_stats": merged["pIC50_external"]
        .describe()
        .round(4)
        .to_dict(),
        "model_metrics": {},
        "notes": [
            "ug/mL IC50 converted through RDKit molecular weight.",
            "10^3 uM is treated literally as 1 mM; sensitivity is reported below.",
            "These measurements come from several old assays and are a crude,"
            " not a gold-standard, external check.",
        ],
    }
    for column in PREDICTION_COLUMNS:
        summary["model_metrics"][column] = regression_metrics(
            merged["pIC50_external"].to_numpy(),
            merged[column].to_numpy(),
        )

    # Sensitivity check: reinterpret "10^3 uM" as plain uM.
    if (merged["external_units"].astype(str) == "10^3 uM").any():
        sensitivity = merged.copy()
        mask = sensitivity["external_units"].astype(str) == "10^3 uM"
        sensitivity.loc[mask, "pIC50_external"] = 9.0 - math.log10(
            sensitivity.loc[mask, "IC50_nM_external"].iloc[0] / 1000.0
        )
        summary["sensitivity_10^3_uM_as_uM"] = regression_metrics(
            sensitivity["pIC50_external"].to_numpy(),
            sensitivity["ensemble_pic50"].to_numpy(),
        )

    out_csv = results_dir / "m6_external_validation.csv"
    merged.sort_values("ensemble_rank").to_csv(out_csv, index=False)
    out_json = results_dir / "m6_external_validation.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("=" * 70)
    print("M6 external validation")
    print("=" * 70)
    print(f"  candidates with external IC50: {len(merged)} / {len(predictions)}")
    for column, metrics in summary["model_metrics"].items():
        print(
            f"  {column:<18} Pearson={metrics['Pearson_r']:+.3f} "
            f"Spearman={metrics['Spearman_rho']:+.3f} "
            f"MAE={metrics['MAE']:.3f} RMSE={metrics['RMSE']:.3f}"
        )
    print(f"  saved -> {out_csv}")
    print(f"  saved -> {out_json}")


if __name__ == "__main__":
    main()
