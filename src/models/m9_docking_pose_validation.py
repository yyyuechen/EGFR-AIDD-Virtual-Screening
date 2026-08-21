#!/usr/bin/env python3
"""
M9 P0-2: docking pose validation with the erlotinib co-crystal.

The M7 docking used EGFR PDB 1M17, which contains erlotinib as the crystal
ligand (residue AQ4). This script redocks the same ligand (using the saved
M7 Vina output), converts every Vina model to a 3D structure, and computes
the heavy-atom RMSD of each model against the crystal pose.

A RMSD below 2.0 Angstrom for at least one near-top model is the standard
sanity check that the docking setup can reproduce the known binding pose.

Docking environment (rdkit + numpy + matplotlib + openbabel):

    conda env create -f environment-docking.yml
    conda activate egfr-aidd-dock
    python src/models/m9_docking_pose_validation.py

Usage (from the project root)
-----------------------------
    python src/models/m9_docking_pose_validation.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem


def extract_hetatm(pdb_path: Path, resname: str) -> str:
    lines = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith("HETATM") and line[17:20] == resname:
            lines.append(line)
    if not lines:
        raise ValueError(f"No HETATM residue {resname} found in {pdb_path}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def run_obabel(obabel_bin: Path, args: list[str]) -> None:
    cmd = [str(obabel_bin)] + args
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"obabel failed: {result.stderr}")


def parse_affinities(pdbqt_path: Path) -> list[float]:
    affinities = []
    for line in pdbqt_path.read_text().splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            affinities.append(float(line.split()[3]))
    return affinities


def heavy_atom_rmsd(docked: Chem.Mol, crystal: Chem.Mol, template: Chem.Mol) -> float:
    """Kabsch RMSD using a template SMILES to map atoms between two conformers."""
    m1 = docked.GetSubstructMatch(template)
    m2 = crystal.GetSubstructMatch(template)
    if len(m1) != template.GetNumAtoms() or len(m2) != template.GetNumAtoms():
        raise ValueError("Template match failed")

    def coords(mol, mapping):
        conf = mol.GetConformer()
        return np.array(
            [list(conf.GetAtomPosition(i)) for i in mapping], dtype=float
        )

    P = coords(docked, m1)
    Q = coords(crystal, m2)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, sign]) @ U.T
    aligned = Pc @ R.T
    return float(np.sqrt(np.mean(np.sum((aligned - Qc) ** 2, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser(description="M9 docking pose validation")
    parser.add_argument("--pdb", default="data/docking/1M17.pdb")
    parser.add_argument(
        "--docked-pdbqt",
        default="results/m7_docking_out/CHEMBL553_out.pdbqt",
    )
    parser.add_argument("--ligand-smiles", default="C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1")
    parser.add_argument("--obabel-bin", default="obabel")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    obabel_candidate = shutil.which(args.obabel_bin)
    if obabel_candidate is None and Path(args.obabel_bin).exists():
        obabel_candidate = args.obabel_bin
    if obabel_candidate is None:
        raise SystemExit(
            "obabel not found on PATH; install openbabel with "
            "environment-docking.yml and try again"
        )
    obabel = Path(obabel_candidate)

    template = Chem.MolFromSmiles(args.ligand_smiles)
    if template is None:
        raise SystemExit("Invalid ligand SMILES")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        crystal_pdb = tmp / "aq4_crystal.pdb"
        crystal_pdb.write_text(extract_hetatm(Path(args.pdb), "AQ4"))
        crystal_sdf = tmp / "aq4_crystal.sdf"
        docked_sdf = tmp / "erlotinib_docked.sdf"

        run_obabel(obabel, [str(crystal_pdb), "-O", str(crystal_sdf)])
        run_obabel(
            obabel,
            [str(args.docked_pdbqt), "-O", str(docked_sdf)],
        )

        crystal = Chem.MolFromMolFile(
            str(crystal_sdf), removeHs=True, sanitize=True
        )
        affinities = parse_affinities(Path(args.docked_pdbqt))
        supplier = Chem.SDMolSupplier(
            str(docked_sdf), removeHs=True, sanitize=True
        )

        rows = []
        for i, mol in enumerate(supplier):
            if mol is None:
                continue
            rmsd = heavy_atom_rmsd(mol, crystal, template)
            affinity = affinities[i] if i < len(affinities) else None
            rows.append(
                {
                    "model": i + 1,
                    "affinity_kcal_mol": affinity,
                    "rmsd_to_crystal_A": round(rmsd, 3),
                }
            )

    best_affinity = min(rows, key=lambda r: r["affinity_kcal_mol"])
    best_rmsd = min(rows, key=lambda r: r["rmsd_to_crystal_A"])
    n_good = sum(1 for r in rows if r["rmsd_to_crystal_A"] < 2.0)

    summary = {
        "milestone": "M9 P0-2 docking pose validation",
        "receptor_pdb": args.pdb,
        "docked_pdbqt": args.docked_pdbqt,
        "crystal_ligand": "AQ4 (erlotinib)",
        "n_models": len(rows),
        "n_models_rmsd_below_2A": n_good,
        "best_affinity_model": best_affinity,
        "best_rmsd_model": best_rmsd,
        "results": rows,
        "interpretation": (
            "Best-scoring pose may not be the most native-like; a standard"
            " pose validation reports the best RMSD among near-top modes."
        ),
    }
    json_path = results_dir / "m9_pose_validation.json"
    json_path.write_text(json.dumps(summary, indent=2))

    table = [rows[i] for i in range(len(rows))]
    import csv

    csv_path = results_dir / "m9_pose_validation.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "affinity_kcal_mol", "rmsd_to_crystal_A"])
        writer.writeheader()
        writer.writerows(table)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"model {r['model']}" for r in rows]
    rmsds = [r["rmsd_to_crystal_A"] for r in rows]
    colors = ["#4C72B0" if r < 2.0 else "#C44E52" for r in rmsds]
    bars = ax.bar(labels, rmsds, color=colors)
    ax.axhline(2.0, color="gray", linestyle="--", label="2 Angstrom threshold")
    ax.set_ylabel("heavy-atom RMSD to crystal (Angstrom)")
    ax.set_title("Erlotinib docking pose validation (1M17)")
    ax.legend()
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{row['affinity_kcal_mol']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig_path = figures_dir / "m9_pose_validation.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print("=" * 70)
    print("M9 P0-2 docking pose validation")
    print("=" * 70)
    print(f"  models: {len(rows)}, RMSD<2A: {n_good}")
    print(
        f"  best affinity model: model {best_affinity['model']}, "
        f"{best_affinity['affinity_kcal_mol']:.3f}, "
        f"RMSD {best_affinity['rmsd_to_crystal_A']:.3f}"
    )
    print(
        f"  best RMSD model: model {best_rmsd['model']}, "
        f"RMSD {best_rmsd['rmsd_to_crystal_A']:.3f}, "
        f"affinity {best_rmsd['affinity_kcal_mol']:.3f}"
    )
    print(f"  saved -> {json_path}")
    print(f"  saved -> {csv_path}")
    print(f"  saved -> {fig_path}")


if __name__ == "__main__":
    main()
