#!/usr/bin/env python3
"""
M7: ADMET descriptor calculation for the M6.5 scoring set.

The M6.5 RF + Transformer ensemble produced a ranked scoring set of 64
molecules (60 candidates + 4 approved EGFR inhibitor controls). This script
computes the standard, reproducible ADMET-style descriptors for each molecule
and flags simple drug-likeness rules:

  Lipinski   : MW <= 500, cLogP <= 5, HBD <= 5, HBA <= 10
  Veber      : TPSA <= 140, rotatable bonds <= 10
  ESOL       : Delaney aqueous solubility estimate (logS)

Usage (from the project root)
-----------------------------
    conda activate egfr-aidd
    python src/models/m7_admet.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors


def esol_log_s(mol, logp: float) -> float:
    """Delaney ESOL estimate: logS = f(cLogP, MW, rotors, aromatic fraction)."""
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    aromatic_frac = aromatic / heavy if heavy else 0.0
    mw = Descriptors.MolWt(mol)
    rotors = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return (
        0.16
        - 0.63 * logp
        - 0.0062 * mw
        + 0.066 * rotors
        - 0.74 * aromatic_frac
    )


def admet_row(row: pd.Series) -> dict | None:
    mol = Chem.MolFromSmiles(row["canonical_smiles"])
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotors = rdMolDescriptors.CalcNumRotatableBonds(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    qed = QED.qed(mol)

    lipinski_violations = int(
        (mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)
    )
    veber_violations = int((tpsa > 140) or (rotors > 10))

    return {
        "molecule_chembl_id": row["molecule_chembl_id"],
        "name": row.get("name"),
        "source": row["source"],
        "canonical_smiles": row["canonical_smiles"],
        "MW": round(mw, 3),
        "cLogP": round(logp, 3),
        "TPSA": round(tpsa, 3),
        "HBD": int(hbd),
        "HBA": int(hba),
        "RotatableBonds": int(rotors),
        "AromaticRings": int(aromatic_rings),
        "HeavyAtoms": int(heavy),
        "QED": round(qed, 4),
        "ESOL_logS": round(esol_log_s(mol, logp), 3),
        "Lipinski_violations": lipinski_violations,
        "Veber_violations": veber_violations,
        "Lipinski_ok": bool(lipinski_violations <= 1),
        "Veber_ok": bool(veber_violations == 0),
        "ADMET_ok": bool(lipinski_violations <= 1 and veber_violations == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 ADMET descriptors")
    parser.add_argument(
        "--scoring-file",
        default="results/m65_rf_transformer_ensemble_predictions.csv",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    scoring = pd.read_csv(args.scoring_file)
    rows = [admet_row(row) for _, row in scoring.iterrows()]
    rows = [row for row in rows if row is not None]
    table = pd.DataFrame(rows)

    out_csv = results_dir / "m7_admet_descriptors.csv"
    table.to_csv(out_csv, index=False)

    summary = {
        "milestone": "M7 ADMET descriptors",
        "n_molecules": int(len(table)),
        "n_admet_ok": int(table["ADMET_ok"].sum()),
        "n_lipinski_ok": int(table["Lipinski_ok"].sum()),
        "n_veber_ok": int(table["Veber_ok"].sum()),
        "lipinski_violation_counts": table["Lipinski_violations"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "descriptor_means": {
            col: float(table[col].mean())
            for col in ["MW", "cLogP", "TPSA", "HBD", "HBA", "QED", "ESOL_logS"]
        },
    }
    out_json = results_dir / "m7_admet_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("=" * 70)
    print("M7 ADMET descriptors")
    print("=" * 70)
    print(f"  molecules: {summary['n_molecules']}")
    print(f"  ADMET ok (Lipinski <=1 + Veber): {summary['n_admet_ok']}")
    print(f"  Lipinski ok: {summary['n_lipinski_ok']}")
    print(f"  Veber ok: {summary['n_veber_ok']}")
    print(f"  saved -> {out_csv}")
    print(f"  saved -> {out_json}")


if __name__ == "__main__":
    main()
