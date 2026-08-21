#!/usr/bin/env python3
"""
preprocess_egfr.py
==================

Milestone 1, Stage 2: scientific preprocessing.

Responsibilities:
    1. Load the raw ChEMBL activity table and molecule SMILES table.
    2. Report and filter activity types (keep IC50 as the modeling endpoint).
    3. Handle standard_relation conservatively (keep exact "=" records for
       the primary regression dataset).
    4. Normalize IC50 units to nM.
    5. Merge molecule structures, validate SMILES with RDKit, and apply a
       conservative standardization (largest fragment, canonical SMILES;
       stereochemistry preserved, tautomers not canonicalized).
    6. Calculate pIC50 = 9 - log10(IC50_nM).
    7. Aggregate repeated measurements to a molecule-level dataset using
       median pIC50.
    8. Save intermediate tables, the final dataset, and a processing summary.

Usage (from the project root):
    python src/data/preprocess_egfr.py
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
from rdkit.Chem.MolStandardize import rdMolStandardize


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
# All IC50 values are converted to nM before pIC50 is calculated:
#
#   1 M   = 1e9 nM
#   1 mM  = 1e6 nM
#   1 uM  = 1e3 nM   (ChEMBL may use uM, µM, or μM)
#   1 nM  = 1
#   1 pM  = 1e-3 nM
#
# If an unknown unit appears we report it and exclude the row rather than
# guessing a conversion factor.
UNIT_TO_NM = {
    "M": 1e9,
    "mM": 1e6,
    "uM": 1e3,
    "µM": 1e3,
    "μM": 1e3,
    "nM": 1.0,
    "pM": 1e-3,
}

# The project-defined classification threshold (educational, not universal):
# pIC50 >= 7 -> active (1), pIC50 < 7 -> inactive (0).
ACTIVITY_THRESHOLD = 7.0


def print_step(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def report_removal(label: str, before: int, after: int) -> None:
    removed = before - after
    print(f"  {label}: {before:,} -> {after:,}  (removed {removed:,})")


def convert_ic50_to_pic50(ic50_nm: float) -> float:
    """
    Convert an IC50 in nM to pIC50.

    pIC50 = -log10(IC50 in molar units)

    IC50_nM * 1e-9 = IC50 in M

    so

    pIC50 = -log10(IC50_nM * 1e-9)
          = 9 - log10(IC50_nM)

    Sanity example:
        IC50 = 10 nM
        pIC50 = 9 - log10(10) = 9 - 1 = 8
    """
    return 9.0 - math.log10(ic50_nm)


def standardize_molecule(smiles: str) -> tuple[str, int] | None:
    """
    Conservative molecular standardization for M1.

    Steps performed:
      1. Parse SMILES into an RDKit Mol.
      2. Keep the largest fragment (removes salts/counterions).
      3. Write a canonical, isomeric SMILES string from the parent Mol.

    Steps intentionally NOT performed in M1:
      - tautomer canonicalization (can change the representation of a molecule
        in ways that deserve explicit discussion before adoption);
      - charge neutralization / reionization (charges can be biologically
        relevant and should be reviewed, not silently removed);
      - full "normalize everything" MolStandardize pipelines.

    Returns (canonical_smiles, num_fragments) or None if processing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        num_fragments = len(Chem.GetMolFrags(mol))
        # rdMolStandardize.FragmentParent() is a Boost.Python function that
        # takes the Mol directly and returns the parent (largest) fragment.
        parent_mol = rdMolStandardize.FragmentParent(mol)
        canonical = Chem.MolToSmiles(
            parent_mol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None

    return canonical, num_fragments


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw EGFR ChEMBL data.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Project data directory (default: data).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Project results directory (default: results).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    results_dir = Path(args.results_dir)

    for d in (interim_dir, processed_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: load raw data
    # ------------------------------------------------------------------
    print_step("Step 1: load raw ChEMBL data")
    activities = pd.read_csv(raw_dir / "egfr_chembl_activities_raw.csv")
    molecules = pd.read_csv(raw_dir / "egfr_molecules_smiles.csv")

    print(f"  raw activity rows : {len(activities):,}")
    print(f"  molecule rows     : {len(molecules):,}")
    print(f"  raw columns       : {list(activities.columns)}")

    summary: dict = {
        "target_info": json.loads(
            (raw_dir / "egfr_target_info.json").read_text()
        ),
        "activity_threshold_pIC50": ACTIVITY_THRESHOLD,
    }

    # ------------------------------------------------------------------
    # Step 2: inspect activity types, then keep IC50 only
    # ------------------------------------------------------------------
    print_step("Step 2: inspect activity types and keep IC50")
    type_counts = activities["standard_type"].fillna("missing").value_counts()
    print("  activity type counts:")
    for activity_type, count in type_counts.items():
        print(f"    {activity_type}: {count:,}")
    summary["activity_type_counts"] = {
        str(k): int(v) for k, v in type_counts.items()
    }

    before = len(activities)
    ic50_df = activities[activities["standard_type"] == "IC50"].copy()
    report_removal("IC50 filter", before, len(ic50_df))
    summary["raw_records"] = before
    summary["ic50_records"] = len(ic50_df)

    ic50_df.to_csv(interim_dir / "egfr_ic50_records.csv", index=False)

    # ------------------------------------------------------------------
    # Step 3: numeric value sanity checks
    # ------------------------------------------------------------------
    print_step("Step 3: numeric IC50 values")
    ic50_df["standard_value_numeric"] = pd.to_numeric(
        ic50_df["standard_value"], errors="coerce"
    )
    before = len(ic50_df)
    numeric_df = ic50_df[
        ic50_df["standard_value_numeric"].notna()
        & (ic50_df["standard_value_numeric"] > 0)
    ].copy()
    report_removal("numeric, >0 filter", before, len(numeric_df))
    summary["excluded_non_numeric_or_nonpositive"] = before - len(numeric_df)
    summary["numeric_records"] = len(numeric_df)

    # ------------------------------------------------------------------
    # Step 4: handle standard_relation conservatively
    # ------------------------------------------------------------------
    print_step("Step 4: standard_relation handling")
    relation_counts = (
        numeric_df["standard_relation"].fillna("missing").value_counts()
    )
    print("  relation counts:")
    for relation, count in relation_counts.items():
        print(f"    {relation}: {count:,}")
    summary["relation_counts"] = {
        str(k): int(v) for k, v in relation_counts.items()
    }

    before = len(numeric_df)
    exact_df = numeric_df[numeric_df["standard_relation"] == "="].copy()
    report_removal("exact '=' only (primary regression set)", before, len(exact_df))
    summary["exact_relation_records"] = len(exact_df)
    summary["excluded_censored_measurements"] = before - len(exact_df)

    exact_df.to_csv(interim_dir / "egfr_exact_ic50.csv", index=False)

    # ------------------------------------------------------------------
    # Step 5: normalize units to nM
    # ------------------------------------------------------------------
    print_step("Step 5: unit normalization to nM")
    unit_counts = exact_df["standard_units"].fillna("missing").value_counts()
    print("  unit counts:")
    for unit, count in unit_counts.items():
        print(f"    {unit}: {count:,}")
    summary["unit_counts"] = {str(k): int(v) for k, v in unit_counts.items()}

    supported_mask = exact_df["standard_units"].map(UNIT_TO_NM).notna()
    before = len(exact_df)
    supported_df = exact_df[supported_mask].copy()
    report_removal("supported units", before, len(supported_df))
    summary["supported_unit_records"] = len(supported_df)
    summary["excluded_unsupported_units"] = before - len(supported_df)

    supported_df["IC50_nM"] = (
        supported_df["standard_value_numeric"]
        * supported_df["standard_units"].map(UNIT_TO_NM)
    )

    # Sanity: conversion must never produce non-positive values.
    invalid_ic50 = supported_df["IC50_nM"] <= 0
    if invalid_ic50.any():
        print(f"  [warn] removing {invalid_ic50.sum():,} rows with IC50_nM <= 0")
        supported_df = supported_df[~invalid_ic50]
    summary["excluded_nonpositive_after_conversion"] = int(invalid_ic50.sum())

    # ------------------------------------------------------------------
    # Step 6: merge molecular structures
    # ------------------------------------------------------------------
    print_step("Step 6: merge molecular structures")
    smiles_map = (
        molecules.drop_duplicates(subset="molecule_chembl_id")
        .set_index("molecule_chembl_id")["canonical_smiles"]
    )

    # The v3 activity download no longer includes canonical_smiles; if the
    # column is absent, create it as NaN so SMILES are filled from the
    # molecule endpoint instead.
    if "canonical_smiles" not in supported_df.columns:
        supported_df["canonical_smiles"] = np.nan

    # Prefer the SMILES directly returned by the activity endpoint; fill
    # missing values from the molecule endpoint.
    supported_df["source_smiles"] = supported_df["canonical_smiles"].fillna(
        supported_df["molecule_chembl_id"].map(smiles_map)
    )
    supported_df["smiles_source"] = np.where(
        supported_df["canonical_smiles"].notna(),
        "activity",
        np.where(
            supported_df["molecule_chembl_id"].map(smiles_map).notna(),
            "molecule",
            "missing",
        ),
    )

    before = len(supported_df)
    structure_df = supported_df[supported_df["source_smiles"].notna()].copy()
    report_removal("with structures", before, len(structure_df))
    summary["with_structures_records"] = len(structure_df)
    summary["excluded_missing_structures"] = before - len(structure_df)

    structure_df.to_csv(interim_dir / "egfr_with_structures.csv", index=False)

    # ------------------------------------------------------------------
    # Step 7: RDKit validation and molecular standardization
    # ------------------------------------------------------------------
    print_step("Step 7: RDKit validation and standardization")
    parsed = structure_df["source_smiles"].map(Chem.MolFromSmiles)
    valid_mask = parsed.notna()
    before = len(structure_df)
    valid_df = structure_df[valid_mask].copy()
    report_removal("RDKit-valid SMILES", before, len(valid_df))
    summary["valid_smiles_records"] = len(valid_df)
    summary["excluded_invalid_smiles"] = before - len(valid_df)

    standardized = valid_df["source_smiles"].map(standardize_molecule)
    valid_std_mask = standardized.notna()
    std_df = valid_df[valid_std_mask].copy()
    report_removal("standardization succeeded", len(valid_df), len(std_df))
    summary["standardized_records"] = len(std_df)
    summary["excluded_standardization_failures"] = int((~valid_std_mask).sum())

    std_df["canonical_smiles"] = [x[0] for x in standardized[valid_std_mask]]
    std_df["num_fragments"] = [x[1] for x in standardized[valid_std_mask]]

    # ------------------------------------------------------------------
    # Step 8: calculate pIC50
    # ------------------------------------------------------------------
    print_step("Step 8: calculate pIC50")
    std_df["pIC50"] = std_df["IC50_nM"].map(convert_ic50_to_pic50)
    print("  sanity check: IC50=10 nM -> pIC50=8")
    print(
        "  observed example: pIC50="
        f"{convert_ic50_to_pic50(10.0):.2f} for IC50_nM=10"
    )
    std_df.to_csv(interim_dir / "egfr_standardized.csv", index=False)

    # ------------------------------------------------------------------
    # Step 9: aggregate repeated measurements at molecule level
    # ------------------------------------------------------------------
    print_step("Step 9: aggregate repeated measurements (median pIC50)")

    grouped = (
        std_df.groupby("canonical_smiles")
        .agg(
            n_measurements=("pIC50", "size"),
            pIC50=("pIC50", "median"),
            pIC50_std=(
                "pIC50",
                lambda x: x.std(ddof=1) if len(x) > 1 else np.nan,
            ),
            pIC50_min=("pIC50", "min"),
            pIC50_max=("pIC50", "max"),
        )
        .reset_index()
    )

    # Representative ChEMBL molecule ID: the ID with the most measurements.
    rep_id = (
        std_df.groupby("canonical_smiles")["molecule_chembl_id"]
        .agg(lambda x: x.value_counts().idxmax())
        .rename("molecule_chembl_id")
    )
    all_ids = (
        std_df.groupby("canonical_smiles")["molecule_chembl_id"]
        .agg(
            lambda x: "|".join(
                sorted({str(v) for v in x if pd.notna(v)})
            )
        )
        .rename("molecule_chembl_ids")
    )
    n_molecules = (
        std_df.groupby("canonical_smiles")["molecule_chembl_id"]
        .agg(lambda x: x.nunique())
        .rename("n_molecules")
    )

    final_df = grouped.merge(rep_id, on="canonical_smiles")
    final_df = final_df.merge(all_ids, on="canonical_smiles")
    final_df = final_df.merge(n_molecules, on="canonical_smiles")

    # Back-transform the aggregated pIC50 so IC50_nM is unambiguously linked
    # to the aggregated pIC50:
    #   IC50_nM = 10^(9 - pIC50)
    final_df["IC50_nM"] = 10 ** (9 - final_df["pIC50"])
    final_df["activity_class"] = (
        final_df["pIC50"] >= ACTIVITY_THRESHOLD
    ).astype(int)

    final_df = final_df[
        [
            "canonical_smiles",
            "pIC50",
            "IC50_nM",
            "activity_class",
            "n_measurements",
            "pIC50_std",
            "pIC50_min",
            "pIC50_max",
            "molecule_chembl_id",
            "molecule_chembl_ids",
            "n_molecules",
        ]
    ].sort_values("canonical_smiles").reset_index(drop=True)

    print(f"  unique standardized molecules: {len(final_df):,}")
    print(f"  measurement rows aggregated  : {len(std_df):,}")
    summary["unique_final_molecules"] = len(final_df)
    summary["measurement_rows_after_standardization"] = len(std_df)

    # ------------------------------------------------------------------
    # Step 10: quality checks and outputs
    # ------------------------------------------------------------------
    print_step("Step 10: quality checks")
    checks = {
        "no_missing_canonical_smiles": final_df["canonical_smiles"].notna().all(),
        "canonical_smiles_unique": final_df["canonical_smiles"].is_unique,
        "pIC50_finite": np.isfinite(final_df["pIC50"]).all(),
        "IC50_nM_positive": (final_df["IC50_nM"] > 0).all(),
        "n_measurements_ge_1": (final_df["n_measurements"] >= 1).all(),
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if not passed:
            raise RuntimeError(f"Quality check failed: {name}")

    processed_path = processed_dir / "egfr_activity_final.csv"
    final_df.to_csv(processed_path, index=False)
    print(f"  saved -> {processed_path}")

    summary["pIC50_min"] = float(final_df["pIC50"].min())
    summary["pIC50_max"] = float(final_df["pIC50"].max())
    summary["pIC50_mean"] = float(final_df["pIC50"].mean())
    summary["pIC50_median"] = float(final_df["pIC50"].median())
    summary["n_active"] = int((final_df["activity_class"] == 1).sum())
    summary["n_inactive"] = int((final_df["activity_class"] == 0).sum())

    summary_path = results_dir / "data_processing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  saved -> {summary_path}")

    print("\nFinal dataset preview:")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
