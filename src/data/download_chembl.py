#!/usr/bin/env python3
"""
download_chembl.py
==================

Milestone 1, Stage 1: data acquisition.

Responsibilities:
    1. Identify the ChEMBL target corresponding to EGFR / UniProt P00533.
    2. Retrieve EGFR activity records from ChEMBL.
    3. Retrieve molecular structures (SMILES) for the molecules appearing
       in those activity records.
    4. Save the raw data under data/raw/.

This script intentionally does NOT do scientific filtering, unit conversion,
pIC50 calculation, or molecular standardization.  Those steps belong to
preprocess_egfr.py so that acquisition and preprocessing stay auditable.

Usage example (from the project root):
    python src/data/download_chembl.py

Optional flags:
    --target-chembl-id CHEMBLxxx   skip target selection and use this ID
    --limit 500                    only download first 500 activities (testing)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Biological constants
# ---------------------------------------------------------------------------
# The course material works with EGFR through UniProt ID P00533
# (see day01-AIDD/code/08-chembl.ipynb).  We deliberately start from the
# UniProt accession instead of hard-coding a ChEMBL target ID, so the link
# UniProt P00533 -> target_chembl_id is shown and can be verified.
UNIPROT_ID = "P00533"
ORGANISM = "Homo sapiens"

# ChEMBL stores UniProt accessions inside target_components, so the correct
# REST filter is the nested field "target_components.accession" (not
# "uniprot_accession", which the client silently ignores and would return
# every target in the database).
NESTED_UNIPROT_FILTER = "target_components__accession"

# Human-readable hints used to pick the wild-type EGFR target among the
# targets that ChEMBL associates with P00533.
NAME_HINT = "epidermal growth factor receptor"
MUTANT_HINT = re.compile(
    r"mutant|variant|deletion|truncat|fusion|isoform|domain|chimera",
    flags=re.IGNORECASE,
)

# Columns that are useful to keep in the raw activity table for later
# inspection.  ChEMBL may not return every column; we keep the intersection.
RAW_ACTIVITY_COLUMNS = [
    "activity_id",
    "molecule_chembl_id",
    "target_chembl_id",
    "target_pref_name",
    "target_organism",
    "standard_type",
    "standard_value",
    "standard_units",
    "standard_relation",
    "pchembl_value",
    "assay_chembl_id",
    "assay_type",
    "assay_organism",
    "assay_description",
    "document_journal",
    "document_year",
    "canonical_smiles",
]


def print_step(title: str) -> None:
    """Small visual separator so the pipeline is easy to follow in the terminal."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def select_egfr_target(target_rows: list[dict]) -> dict:
    """
    Select the ChEMBL target that best represents wild-type human EGFR.

    Input
    -----
    target_rows : list of dicts returned by
                  new_client.target.filter(uniprot_accession="P00533")

    Output
    ------
    The selected target dict.

    Why this is needed
    ------------------
    A single UniProt accession can be attached to several ChEMBL targets
    (wild-type protein, mutants, complexes, isoforms).  For an EGFR inhibitor
    dataset we want the wild-type human EGFR target.  The selection rule is
    printed so that the user can verify it before trusting the dataset.
    """

    if not target_rows:
        raise RuntimeError(
            f"No ChEMBL target found for UniProt accession {UNIPROT_ID!r}."
        )

    # Keep every candidate visible in the log: we do not silently hide targets.
    candidates = pd.DataFrame(target_rows)
    display_cols = [
        c for c in ["target_chembl_id", "pref_name", "organism", "target_type"]
        if c in candidates.columns
    ]
    print("\nCandidates returned by ChEMBL for UniProt P00533:")
    print(candidates[display_cols].to_string(index=False))

    # Rule 1: only human targets.
    human = candidates[
        (candidates.get("organism", "").astype(str) == ORGANISM)
        | (candidates.get("tax_id", 0).astype(str) == "9606")
    ]
    if human.empty:
        raise RuntimeError(
            "No human target found for UniProt P00533. "
            "Inspect candidates manually and use --target-chembl-id."
        )

    # Rule 2: prefer a target whose name looks like wild-type EGFR.
    def _name_score(row: pd.Series) -> int:
        name = str(row.get("pref_name", "")).lower()
        score = 0
        if NAME_HINT in name:
            score += 2
        if not MUTANT_HINT.search(name):
            score += 1
        return score

    human = human.copy()
    human["_name_score"] = human.apply(_name_score, axis=1)
    human = human.sort_values(
        ["_name_score", "target_chembl_id"], ascending=[False, True]
    )

    # Rule 3 (tie-breaker): prefer SINGLE PROTEIN targets.
    selected = human[human["_name_score"] > 0].head(1)
    if selected.empty:
        selected = human.head(1)

    selected_row = selected.iloc[0].to_dict()
    selected_row.pop("_name_score", None)

    print("\nSelected ChEMBL target:")
    print(f"  target_chembl_id : {selected_row.get('target_chembl_id')}")
    print(f"  pref_name        : {selected_row.get('pref_name')}")
    print(f"  organism         : {selected_row.get('organism')}")
    print(f"  target_type      : {selected_row.get('target_type')}")
    print(
        "  NOTE: if your tutor expects a different EGFR target "
        "(e.g. a specific mutant), rerun with --target-chembl-id."
    )
    return selected_row


def fetch_p00533_targets() -> list[dict]:
    """
    Fetch ChEMBL targets whose components contain UniProt P00533.

    The webresource client is used first, but with a hard row cap so that a
    silently ignored filter cannot cause a full-table download.  A target
    search is used as a fallback, and every candidate is verified locally
    against target_components before it is returned.
    """
    from chembl_webresource_client.new_client import new_client

    target_api = new_client.target

    rows = []
    try:
        # Ask for at most 100 rows.  If the nested filter is silently ignored,
        # this still prevents the script from downloading the entire table.
        rows = list(
            target_api.filter(**{NESTED_UNIPROT_FILTER: UNIPROT_ID})[:100]
        )
    except Exception as exc:
        print(f"  [warn] target filter request failed: {exc}")

    if len(rows) >= 100:
        print(
            "  [warn] target filter appears to have been ignored "
            f"({len(rows)} rows); falling back to target search."
        )
        rows = []

    if not rows:
        try:
            rows = list(target_api.search(UNIPROT_ID))
        except Exception as exc:
            print(f"  [warn] target search failed: {exc}")

    verified = []
    for row in rows:
        components = row.get("target_components") or []
        accessions = {c.get("accession") for c in components}
        if UNIPROT_ID in accessions:
            verified.append(row)

    if not verified:
        raise RuntimeError(
            f"Could not retrieve any ChEMBL target containing UniProt "
            f"{UNIPROT_ID!r}. Inspect target_components manually and rerun "
            "with --target-chembl-id."
        )

    if len(verified) < len(rows):
        print(
            f"  [warn] removed {len(rows) - len(verified)} target rows that "
            "did not contain P00533 in target_components"
        )

    return verified


# Common activity endpoint names used only for the "before filtering"
# summary.  The actual modeling dataset uses IC50 exact binding records.
ACTIVITY_TYPE_COUNTS = [
    "IC50", "Ki", "Kd", "EC50", "GI50", "Potency", "Inhibition",
    "AC50", "CC50", "LD50",
]


def fetch_activity_type_counts(target_chembl_id: str) -> dict:
    """
    Fetch total counts per activity type without downloading every row.

    len(query) asks ChEMBL for only the first page and reads page_meta
    total_count, so this is cheap even for large targets.
    """
    from chembl_webresource_client.new_client import new_client

    activity_api = new_client.activity
    counts: dict = {}
    total = len(
        activity_api.filter(target_chembl_id=target_chembl_id).only("activity_id")
    )
    counts["total"] = int(total)
    for activity_type in ACTIVITY_TYPE_COUNTS:
        try:
            n = len(
                activity_api.filter(
                    target_chembl_id=target_chembl_id, type=activity_type
                ).only("activity_id")
            )
            counts[activity_type] = int(n)
        except Exception as exc:
            counts[activity_type] = f"error: {exc}"
    return counts


def fetch_activities(target_chembl_id: str, limit: int | None = None) -> pd.DataFrame:
    """
    Retrieve EGFR IC50 activity records from ChEMBL via direct REST paging.

    Query (same as the course):
        target_chembl_id + type="IC50" + relation="=" + assay_type="B"

    Why direct REST?
    The webresource client returned only 270 rows for this query even though
    the API reports 17,686 total records.  Direct GET requests with explicit
    limit/offset parameters are transparent and easy to debug.
    """
    import requests

    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    fields = [
        "activity_id",
        "molecule_chembl_id",
        "target_chembl_id",
        "target_pref_name",
        "target_organism",
        "standard_type",
        "standard_value",
        "standard_units",
        "standard_relation",
        "pchembl_value",
        "assay_chembl_id",
        "assay_type",
        "assay_organism",
        "document_year",
    ]

    page_size = 250
    offset = 0
    rows = []
    use_only = True
    total = None

    while True:
        params = {
            "target_chembl_id": target_chembl_id,
            "type": "IC50",
            "relation": "=",
            "assay_type": "B",
            "limit": page_size,
            "offset": offset,
        }
        if use_only:
            params["only"] = ",".join(fields)

        page = None
        for attempt in range(1, 4):
            try:
                response = requests.get(base_url, params=params, timeout=60)
                if response.status_code == 400 and use_only:
                    # Some field in "only" may be rejected by this API version.
                    print("  [warn] REST 'only' rejected; retrying with full columns")
                    use_only = False
                    break
                response.raise_for_status()
                data = response.json()
                page = data.get("activities", [])
                total = data["page_meta"]["total_count"]
                break
            except Exception as exc:
                print(
                    f"  [warn] activity page {offset // page_size + 1} "
                    f"attempt {attempt} failed: {exc}"
                )
                if attempt < 3:
                    import time

                    time.sleep(5 * attempt)
                else:
                    raise

        if page is None:
            continue  # "only" was rejected; retry the same offset without it

        rows.extend(page)
        print(f"  fetched {len(rows):,} / {total:,} IC50 binding records", flush=True)

        if limit is not None and len(rows) >= limit:
            rows = rows[:limit]
            print(f"  [dev] limited to {limit} rows")
            break
        if not page or offset + len(page) >= total:
            break
        offset += page_size

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[[c for c in fields if c in df.columns]]


def fetch_molecule_smiles(
    molecule_ids: list[str],
    chunk_size: int = 250,
    sleep_seconds: float = 0.3,
) -> pd.DataFrame:
    """
    Fetch canonical SMILES from the ChEMBL molecule endpoint.

    Unique molecule IDs are queried in chunks so the same molecule is never
    requested twice.  Progress is printed so long downloads stay inspectable.
    """
    from chembl_webresource_client.new_client import new_client

    unique_ids = sorted({str(x) for x in molecule_ids if pd.notna(x)})
    molecule_api = new_client.molecule
    records: list[dict] = []

    for start in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[start : start + chunk_size]
        chunk_records = []
        for attempt in range(1, 4):
            try:
                chunk_records = list(
                    molecule_api.filter(molecule_chembl_id__in=chunk).only(
                        "molecule_chembl_id", "molecule_structures"
                    )
                )
                break
            except Exception as exc:
                print(f"  [warn] molecule chunk attempt {attempt} failed: {exc}")
                if attempt < 3:
                    time.sleep(3 * attempt)
                else:
                    chunk_records = []

        for record in chunk_records:
            structure = record.get("molecule_structures") or {}
            smiles = structure.get("canonical_smiles")
            records.append(
                {
                    "molecule_chembl_id": record.get("molecule_chembl_id"),
                    "canonical_smiles": smiles,
                }
            )

        if len(records) % (chunk_size * 10) < chunk_size:
            print(
                f"  molecules with SMILES fetched: {len(records):,}/{len(unique_ids):,}",
                flush=True,
            )

        time.sleep(sleep_seconds)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw EGFR ChEMBL data.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Project data directory (default: data).",
    )
    parser.add_argument(
        "--target-chembl-id",
        default=None,
        help="Optional ChEMBL target ID override (e.g. CHEMBL1829).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="DEV ONLY: download at most N activity rows for connectivity tests.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Number of molecule IDs per molecule API request.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print_step("Step 1: identify EGFR target from UniProt P00533")
    target_rows = fetch_p00533_targets()

    # Save the full candidate list so the target decision is auditable.
    with open(raw_dir / "egfr_targets_candidates.json", "w") as handle:
        json.dump(target_rows, handle, indent=2, default=str)

    if args.target_chembl_id:
        selected = {
            "target_chembl_id": args.target_chembl_id,
            "selection": "manual override",
        }
        print(f"Using manual target override: {args.target_chembl_id}")
    else:
        selected = select_egfr_target(target_rows)

    with open(raw_dir / "egfr_target_info.json", "w") as handle:
        json.dump(selected, handle, indent=2, default=str)

    print_step("Step 2: inspect activity types, then retrieve IC50 records")
    type_counts = fetch_activity_type_counts(selected["target_chembl_id"])
    print("  activity type counts (from ChEMBL):")
    for key, value in type_counts.items():
        print(f"    {key}: {value}")
    with open(raw_dir / "egfr_activity_type_counts.json", "w") as handle:
        json.dump(type_counts, handle, indent=2)
    print(f"  saved -> {raw_dir / 'egfr_activity_type_counts.json'}")
    activities = fetch_activities(selected["target_chembl_id"], limit=args.limit)
    if activities.empty:
        raise RuntimeError(
            "No activity records returned. "
            "Check target selection or network connectivity."
        )

    print(f"  activity records: {len(activities):,}")
    print(f"  unique molecules: {activities['molecule_chembl_id'].nunique():,}")
    print(f"  columns: {list(activities.columns)}")

    activities.to_csv(raw_dir / "egfr_chembl_activities_raw.csv", index=False)
    print(f"  saved -> {raw_dir / 'egfr_chembl_activities_raw.csv'}")

    print_step("Step 3: retrieve molecular structures (SMILES)")
    molecules = fetch_molecule_smiles(
        activities["molecule_chembl_id"].tolist(),
        chunk_size=args.chunk_size,
    )
    molecules = molecules.drop_duplicates(subset="molecule_chembl_id").reset_index(
        drop=True
    )
    molecules.to_csv(raw_dir / "egfr_molecules_smiles.csv", index=False)
    print(f"  molecule records: {len(molecules):,}")
    print(f"  molecules with SMILES: {molecules['canonical_smiles'].notna().sum():,}")
    print(f"  saved -> {raw_dir / 'egfr_molecules_smiles.csv'}")

    print_step("Download complete")
    print("Next step: python src/data/preprocess_egfr.py")


if __name__ == "__main__":
    main()
