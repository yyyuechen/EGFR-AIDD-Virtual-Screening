#!/usr/bin/env python3
"""
M7: AutoDock Vina docking of the M6.5 RF + Transformer shortlist.

Receptor: EGFR kinase domain (PDB 1M17) prepared as a rigid PDBQT.
Pocket: centered on the co-crystallized erlotinib (AQ4).
Ligands: top-N molecules from the M6.5 equal-weight ensemble plus the four
positive controls, prepared with RDKit 3D conformers and Meeko PDBQT output.

Docking environment (rdkit + meeko + openbabel):

    conda env create -f environment-docking.yml
    conda activate egfr-aidd-dock
    python src/models/m7_docking.py

Usage (from the project root)
-----------------------------
    python src/models/m7_docking.py --top-n 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from meeko import MoleculePreparation, PDBQTWriterLegacy
from src.utils import json_safe


def prepare_ligand_pdbqt(
    smiles: str,
    out_path: Path,
    seed: int = 42,
) -> tuple[bool, str]:
    """Generate a 3D conformer and write a Meeko PDBQT file."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "invalid SMILES"
    mol = Chem.AddHs(mol)
    ok = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if ok != 0:
        ok = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
    if ok != 0:
        return False, "3D embedding failed"
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    pdbqt_string, is_ok, error = PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        return False, error or "Meeko PDBQT write failed"
    out_path.write_text(pdbqt_string)
    return True, ""


def parse_best_affinity(out_path: Path) -> float | None:
    """Read the first Vina result line from an output PDBQT."""
    if not out_path.exists():
        return None
    for line in out_path.read_text().splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            parts = line.split()
            try:
                return float(parts[3])
            except (IndexError, ValueError):
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 AutoDock Vina docking")
    parser.add_argument("--vina-bin", default="data/docking/vina")
    parser.add_argument("--box-file", default="data/docking/vina_box.json")
    parser.add_argument(
        "--scoring-file",
        default="results/m65_rf_transformer_ensemble_predictions.csv",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--ligands-dir", default="data/docking/ligands")
    parser.add_argument("--out-dir", default="results/m7_docking_out")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vina_bin = Path(args.vina_bin)
    if not vina_bin.exists():
        raise SystemExit(f"Vina binary not found: {vina_bin}")
    box = json.loads(Path(args.box_file).read_text())
    receptor = Path(box["receptor_pdbqt"])
    if not receptor.exists():
        raise SystemExit(f"Receptor PDBQT not found: {receptor}")

    results_dir = Path(args.results_dir)
    ligands_dir = Path(args.ligands_dir)
    out_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ligands_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    scoring = pd.read_csv(args.scoring_file)
    scoring["is_positive_control"] = scoring["source"] == "positive_control"
    top = scoring.nlargest(args.top_n, "ensemble_equal_pred")
    controls = scoring[scoring["is_positive_control"]]
    shortlist = pd.concat([controls, top]).drop_duplicates(
        subset="molecule_chembl_id"
    )
    print(
        f"Docking {len(shortlist)} molecules "
        f"({len(controls)} controls + top {args.top_n})"
    )

    rows: list[dict] = []
    for _, row in shortlist.iterrows():
        chembl_id = str(row["molecule_chembl_id"])
        smiles = str(row["canonical_smiles"])
        ligand_path = ligands_dir / f"{chembl_id}.pdbqt"
        ok, error = prepare_ligand_pdbqt(smiles, ligand_path, seed=args.seed)
        if not ok:
            rows.append(
                {
                    "molecule_chembl_id": chembl_id,
                    "name": row.get("name", ""),
                    "source": row["source"],
                    "affinity_kcal_mol": None,
                    "note": f"ligand prep failed: {error}",
                }
            )
            print(f"  {chembl_id}: ligand prep failed ({error})")
            continue

        out_path = out_dir / f"{chembl_id}_out.pdbqt"
        cmd = [
            str(vina_bin),
            "--receptor", str(receptor),
            "--ligand", str(ligand_path),
            "--center_x", str(box["center"][0]),
            "--center_y", str(box["center"][1]),
            "--center_z", str(box["center"][2]),
            "--size_x", str(box["size"][0]),
            "--size_y", str(box["size"][1]),
            "--size_z", str(box["size"][2]),
            "--exhaustiveness", str(args.exhaustiveness),
            "--cpu", str(args.cpu),
            "--seed", str(args.seed),
            "--out", str(out_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        affinity = parse_best_affinity(out_path)
        if affinity is None:
            note = (result.stderr or result.stdout).strip().splitlines()
            note = " ".join(note[-2:]) if note else "no affinity parsed"
        else:
            note = ""
        rows.append(
            {
                "molecule_chembl_id": chembl_id,
                "name": row.get("name", ""),
                "source": row["source"],
                "affinity_kcal_mol": affinity,
                "note": note,
            }
        )
        print(
            f"  {chembl_id}: affinity="
            f"{affinity if affinity is None else round(affinity, 3)}"
        )

    table = pd.DataFrame(rows)
    table = table.sort_values(
        "affinity_kcal_mol", na_position="last"
    ).reset_index(drop=True)
    table["affinity_rank"] = (
        table["affinity_kcal_mol"].rank(ascending=True).astype("Int64")
    )
    csv_path = results_dir / "m7_docking_results.csv"
    table.to_csv(csv_path, index=False)

    summary = {
        "milestone": "M7 AutoDock Vina docking",
        "receptor_pdbqt": str(receptor),
        "box": box,
        "vina_version": "1.2.7",
        "exhaustiveness": args.exhaustiveness,
        "cpu": args.cpu,
        "seed": args.seed,
        "n_docked": int(table["affinity_kcal_mol"].notna().sum()),
        "n_failed": int(table["affinity_kcal_mol"].isna().sum()),
        "results": table.to_dict(orient="records"),
        "notes": [
            "Rigid receptor docking; receptor PDBQT prepared with Open Babel.",
            "Ligands prepared with RDKit 3D conformers + Meeko.",
            "Affinity is the best Vina mode in kcal/mol (more negative = stronger).",
        ],
    }
    json_path = results_dir / "m7_docking_summary.json"
    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )

    print("=" * 70)
    print("M7 docking complete")
    print(f"  docked: {summary['n_docked']}  failed: {summary['n_failed']}")
    print(f"  saved -> {csv_path}")
    print(f"  saved -> {json_path}")


if __name__ == "__main__":
    main()
