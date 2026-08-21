#!/usr/bin/env python3
"""
Audit validation molecules whose max_train_tanimoto equals 1.0.

Reference fingerprint: Morgan radius 2, nBits 2048, binary bit vectors,
using the exact generator settings of the project
(includeChirality=False, useBondTypes=True, countSimulation=False).

The audit uses the standard 80/10/10 Murcko scaffold split (seed 42) and
compares every validation molecule against every training molecule. Test
molecules are never inspected or evaluated.

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/tanimoto_one_audit.py
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
from rdkit.Chem import AllChem, DataStructs

from src.models.portfolio_hardening_analysis import (
    standard_scaffold_split_indices,
)
from src.models.radius_tuning_study import morgan_bit_vectors


EPS = 1e-9


def inchikey(mol: Chem.Mol) -> str | None:
    try:
        return Chem.InchiToInchiKey(Chem.MolToInchi(mol))
    except Exception:
        return None


def classify_pair(
    val_smiles: str,
    tr_smiles: str,
    val_ik: str | None,
    tr_ik: str | None,
) -> str:
    if val_smiles == tr_smiles:
        return "identical_structure"
    if val_ik is not None and val_ik == tr_ik:
        return "same_connectivity_or_tautomer"
    val_mol = Chem.MolFromSmiles(val_smiles)
    tr_mol = Chem.MolFromSmiles(tr_smiles)
    if val_mol is not None and tr_mol is not None:
        val_noniso = Chem.MolToSmiles(val_mol, isomericSmiles=False)
        tr_noniso = Chem.MolToSmiles(tr_mol, isomericSmiles=False)
        if val_noniso == tr_noniso and val_ik != tr_ik:
            return "likely_stereochemistry_only"
    return "different_structure"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tanimoto=1 audit")
    parser.add_argument(
        "--data-file", default="data/processed/egfr_activity_final.csv"
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_file)
    smiles_all = df["canonical_smiles"].tolist()
    ids = df["molecule_chembl_id"].tolist()

    std_train, std_val, std_test, scaffold_by_index, split_meta = (
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
        f"val={len(std_val)} test={len(std_test)} (test not inspected)"
    )

    smiles_train = [smiles_all[i] for i in std_train]
    smiles_val = [smiles_all[i] for i in std_val]
    ids_train = [ids[i] for i in std_train]
    ids_val = [ids[i] for i in std_val]

    train_bits = morgan_bit_vectors(smiles_train, radius=2, n_bits=2048)
    val_bits = morgan_bit_vectors(smiles_val, radius=2, n_bits=2048)

    mol_cache: dict[str, Chem.Mol | None] = {}

    def get_mol(smi: str) -> Chem.Mol | None:
        if smi not in mol_cache:
            mol_cache[smi] = Chem.MolFromSmiles(smi)
        return mol_cache[smi]

    rows: list[dict] = []
    affected_val_idx: list[int] = []

    for vi, vfp in enumerate(val_bits):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(vfp, train_bits))
        max_sim = float(sims.max())
        if max_sim < 1.0 - EPS:
            continue
        affected_val_idx.append(vi)
        val_smiles = smiles_val[vi]
        val_mol = get_mol(val_smiles)
        val_ik = inchikey(val_mol) if val_mol is not None else None
        val_scaffold = scaffold_by_index[std_val[vi]]
        for tj in np.flatnonzero(sims >= 1.0 - EPS):
            tr_smiles = smiles_train[int(tj)]
            tr_mol = get_mol(tr_smiles)
            tr_ik = inchikey(tr_mol) if tr_mol is not None else None
            tr_scaffold = scaffold_by_index[std_train[int(tj)]]
            rows.append(
                {
                    "validation_canonical_smiles": val_smiles,
                    "training_canonical_smiles": tr_smiles,
                    "validation_molecule_chembl_id": ids_val[vi],
                    "training_molecule_chembl_id": ids_train[int(tj)],
                    "validation_inchikey": val_ik,
                    "training_inchikey": tr_ik,
                    "validation_scaffold": val_scaffold,
                    "training_scaffold": tr_scaffold,
                    "tanimoto": float(sims[int(tj)]),
                    "smiles_identical": val_smiles == tr_smiles,
                    "inchikey_identical": (
                        val_ik is not None and val_ik == tr_ik
                    ),
                    "scaffolds_identical": val_scaffold == tr_scaffold,
                    "stereochemistry_difference": classify_pair(
                        val_smiles, tr_smiles, val_ik, tr_ik
                    ),
                }
            )

    audit_df = pd.DataFrame(rows)
    csv_path = results_dir / "tanimoto_one_audit.csv"
    audit_df.to_csv(csv_path, index=False)

    n_affected = len(set(affected_val_idx))
    summary = {
        "milestone": "Tanimoto=1 audit",
        "split": {
            **split_meta,
            "n_train": len(std_train),
            "n_validation": len(std_val),
            "n_test": len(std_test),
        },
        "fingerprint_generator": {
            "api": "rdkit.Chem.AllChem.GetMorganGenerator",
            "radius": 2,
            "fpSize": 2048,
            "includeChirality": False,
            "useBondTypes": True,
            "countSimulation": False,
            "onlyNonzeroInvariants": False,
            "includeRingMembership": True,
            "fingerprint_type": "ExplicitBitVect (binary bit vector)",
            "counts_used": False,
        },
        "n_validation_molecules": len(smiles_val),
        "n_affected_validation_molecules": n_affected,
        "n_pairs": len(rows),
        "affected_validation_indices": affected_val_idx,
        "summary_by_classification": (
            audit_df["stereochemistry_difference"].value_counts().to_dict()
            if not audit_df.empty
            else {}
        ),
        "exact_molecule_leakage_detected": bool(
            not audit_df.empty
            and audit_df["smiles_identical"].any()
        ),
    }
    json_path = results_dir / "tanimoto_one_audit.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    print(f"affected validation molecules: {n_affected} / {len(smiles_val)}")
    print(f"pairs with Tanimoto = 1.0: {len(rows)}")
    if not audit_df.empty:
        print(audit_df["stereochemistry_difference"].value_counts().to_string())
        print(audit_df[["validation_molecule_chembl_id", "training_molecule_chembl_id"]].head(20).to_string(index=False))
    print(f"saved -> {csv_path}")
    print(f"saved -> {json_path}")


if __name__ == "__main__":
    main()
