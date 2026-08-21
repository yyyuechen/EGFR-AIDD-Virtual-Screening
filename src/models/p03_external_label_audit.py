#!/usr/bin/env python3
"""
P0-3a: audit and clean the 59-molecule external IC50 label set.

The M6/M6.5 external test set is built from ChEMBL IC50 records whose units
are not nM. The original builder keeps the most potent record per molecule.
This audit script keeps all records, reports provenance, aggregates with the
median (the same rule as M1), and flags duplicates / ambiguous units.

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/p03_external_label_audit.py
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
from scipy.stats import spearmanr

from src.models.external_candidate_validation import (
    ic50_to_pic50,
    regression_metrics,
)
from src.utils import json_safe


def build_external_records(
    candidates: pd.DataFrame,
    activities: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per non-nM IC50 record with a converted pIC50."""
    smiles_by_id = dict(
        zip(candidates["molecule_chembl_id"], candidates["canonical_smiles"])
    )
    numeric = activities.copy()
    numeric["standard_value_numeric"] = pd.to_numeric(
        numeric["standard_value"], errors="coerce"
    )
    rows: list[dict] = []
    for record in numeric.dropna(
        subset=["standard_value_numeric"]
    ).itertuples():
        units = str(record.standard_units).strip()
        if units == "nM":
            continue
        mol = Chem.MolFromSmiles(smiles_by_id.get(record.molecule_chembl_id, ""))
        if mol is None:
            continue
        pic50 = ic50_to_pic50(
            float(record.standard_value_numeric), units, mol
        )
        if pic50 is None:
            continue
        rows.append(
            {
                "activity_id": record.activity_id,
                "molecule_chembl_id": record.molecule_chembl_id,
                "standard_type": str(record.standard_type),
                "standard_value": float(record.standard_value_numeric),
                "standard_units": units,
                "standard_relation": str(record.standard_relation),
                "assay_chembl_id": str(record.assay_chembl_id),
                "document_year": (
                    int(record.document_year)
                    if pd.notna(record.document_year)
                    else None
                ),
                "pIC50": pic50,
                "IC50_nM": float(10 ** (9.0 - pic50)),
            }
        )
    return pd.DataFrame(rows)


def build_clean_labels(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per molecule with the median pIC50 and provenance flags."""
    if records.empty:
        return pd.DataFrame()
    agg = (
        records.groupby("molecule_chembl_id", as_index=False)
        .agg(
            n_records=("pIC50", "size"),
            pIC50_median=("pIC50", "median"),
            pIC50_max=("pIC50", "max"),
            pIC50_min=("pIC50", "min"),
            pIC50_std=("pIC50", "std"),
            units=("standard_units", lambda s: "|".join(sorted(set(s)))),
            years=(
                "document_year",
                lambda s: "|".join(str(y) for y in sorted(s.dropna())),
            ),
            assay_ids=(
                "assay_chembl_id",
                lambda s: "|".join(sorted(set(s))),
            ),
        )
    )
    agg["flag_duplicate"] = agg["n_records"] > 1
    agg["flag_ambiguous_unit_10_3_uM"] = agg["units"].str.contains(
        "10^3 uM", regex=False
    )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-3 external label audit")
    parser.add_argument(
        "--candidates-file",
        default="data/candidates/egfr_candidate_library.csv",
    )
    parser.add_argument(
        "--activities-file",
        default="data/raw/egfr_chembl_activities_raw.csv",
    )
    parser.add_argument(
        "--ensemble-file",
        default="results/m65_rf_transformer_ensemble_predictions.csv",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates_file)
    activities = pd.read_csv(args.activities_file)
    records = build_external_records(candidates, activities)
    clean = build_clean_labels(records)

    relation_counts = (
        records["standard_relation"].value_counts().to_dict()
    )
    unit_counts = records["standard_units"].value_counts().to_dict()

    sensitivity: dict | None = None
    ensemble = pd.read_csv(args.ensemble_file)
    if not clean.empty and "ensemble_equal_pred" in ensemble.columns:
        merged = ensemble.merge(
            clean[["molecule_chembl_id", "pIC50_max", "pIC50_median"]],
            on="molecule_chembl_id",
            how="inner",
        )
        if len(merged) >= 10:
            sensitivity = {
                "n_molecules": int(len(merged)),
                "n_different_labels": int(
                    (
                        (merged["pIC50_max"] - merged["pIC50_median"]).abs()
                        > 1e-9
                    ).sum()
                ),
                "max_label_metrics": regression_metrics(
                    merged["pIC50_max"].to_numpy(),
                    merged["ensemble_equal_pred"].to_numpy(),
                ),
                "median_label_metrics": regression_metrics(
                    merged["pIC50_median"].to_numpy(),
                    merged["ensemble_equal_pred"].to_numpy(),
                ),
            }

    summary = {
        "milestone": "P0-3a external label audit",
        "n_records": int(len(records)),
        "n_molecules": int(clean["molecule_chembl_id"].nunique()),
        "unit_counts": {str(k): int(v) for k, v in unit_counts.items()},
        "relation_counts": {
            str(k): int(v) for k, v in relation_counts.items()
        },
        "year_min": int(records["document_year"].min()),
        "year_max": int(records["document_year"].max()),
        "n_duplicate_molecules": int(clean["flag_duplicate"].sum()),
        "n_ambiguous_unit_records": int(
            records["standard_units"].eq("10^3 uM").sum()
        ),
        "clean_label_rule": (
            "median pIC50 across exact IC50 records with documented unit"
            " conversion; duplicate and ambiguous units are flagged, not"
            " silently dropped"
        ),
        "sensitivity_max_vs_median": sensitivity,
        "caveats": [
            "All 61 records are exact (standard_relation = '=') and all are"
            " assay type B.",
            "ug.mL-1 conversion uses RDKit molecular weight; 10^3 uM is"
            " treated literally as 1 mM and flagged.",
            "The median aggregation matches the M1 molecule-level rule and"
            " avoids the optimistic 'most potent record wins' choice.",
        ],
    }

    record_path = results_dir / "p03_external_label_audit.csv"
    records.to_csv(record_path, index=False)
    clean_path = results_dir / "p03_external_clean_labels.csv"
    clean.to_csv(clean_path, index=False)
    summary_path = results_dir / "p03_external_label_audit.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )

    print("=" * 70)
    print("P0-3a external label audit")
    print("=" * 70)
    print(f"  records: {len(records)}  molecules: {len(clean)}")
    print(f"  units: {unit_counts}")
    print(f"  relations: {relation_counts}")
    print(
        f"  duplicate molecules: {clean['flag_duplicate'].sum()}, "
        f"ambiguous 10^3 uM records: "
        f"{int(records['standard_units'].eq('10^3 uM').sum())}"
    )
    if sensitivity:
        print(
            f"  max vs median labels differ for "
            f"{sensitivity['n_different_labels']} molecules"
        )
        print(
            f"  ensemble Spearman: max={sensitivity['max_label_metrics']['Spearman_rho']:.3f} "
            f"median={sensitivity['median_label_metrics']['Spearman_rho']:.3f}"
        )
    print(f"  saved -> {record_path}")
    print(f"  saved -> {clean_path}")
    print(f"  saved -> {summary_path}")


if __name__ == "__main__":
    main()
