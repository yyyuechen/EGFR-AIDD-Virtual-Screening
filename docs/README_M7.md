# M7 - ADMET + docking

## What this milestone does

M7 filters and ranks the M6.5 RF + Transformer shortlist with two additional,
orthogonal signals:

1. **ADMET / drug-likeness** from RDKit descriptors (Lipinski, Veber, ESOL,
   QED).
2. **Structural docking** with AutoDock Vina against the EGFR kinase domain
   (PDB 1M17, co-crystallized with erlotinib).

The final M7 ranking combines ensemble pIC50, docking affinity, and ADMET
into one transparent score.

## ADMET summary

64 scoring molecules were described with RDKit:

| Check | Passed |
|---|---:|
| Lipinski (<=1 violation) | 60 |
| Veber (TPSA <= 140, rotors <= 10) | 54 |
| Both (ADMET_ok) | 52 |

Descriptors saved per molecule: MW, cLogP, TPSA, HBD, HBA, rotatable bonds,
aromatic rings, QED, ESOL logS, Lipinski/Veber flags.

## Docking setup

- Receptor: EGFR kinase domain, PDB `1M17` (chain A, water and ligand
  removed), converted to rigid PDBQT with Open Babel.
- Pocket: centered on the co-crystallized erlotinib (AQ4), 24 x 24 x 24 Ang.
- Tools: AutoDock Vina 1.2.7 (macOS arm64 binary in `data/docking/vina`),
  RDKit 3D conformers, Meeko PDBQT writer.
- Shortlist: top 20 molecules by equal-weight RF + Transformer ensemble plus
  the four positive controls (deduplicated to 20 molecules).
- Result: all 20 molecules docked successfully; 0 failures.

Sanity check with positive controls:

| Control | Vina affinity (kcal/mol) |
|---|---:|
| Afatinib | -8.42 |
| Lapatinib | -8.82 |
| Gefitinib | -8.19 |
| Erlotinib | -7.01 |

This is a necessary but not sufficient sanity check: the controls are much
more active than the candidate library (reference pIC50 7.5-9.0 vs external
mean 4.72), so ranking them high is expected even for weak models. Docking
alone also does not reproduce the control activity order: afatinib (reference
pIC50 8.98) is affinity rank 10 and erlotinib (reference pIC50 7.52) is
affinity rank 17, which is why the final ranking combines pIC50, docking, and
ADMET instead of relying on docking alone.

## Final ranking

Score formula (fixed educational weights, not tuned):

```text
m7_score = 0.5 * z(ensemble pIC50)
         + 0.3 * z(-docking affinity)
         + 0.2 * ADMET_ok
```

Top 10 of the M7 final ranking:

| Rank | ChEMBL ID | Name | pIC50 | Affinity | ADMET_ok |
|---:|---|---|---:|---:|---|
| 1 | CHEMBL1173655 | afatinib | 7.95 | -8.42 | True |
| 2 | CHEMBL554 | lapatinib | 7.76 | -8.82 | False |
| 3 | CHEMBL939 | gefitinib | 7.59 | -8.19 | True |
| 4 | CHEMBL553 | erlotinib | 7.70 | -7.01 | True |
| 5 | CHEMBL243664 | - | 6.92 | -8.64 | True |
| 6 | CHEMBL521470 | - | 6.54 | -9.52 | True |
| 7 | CHEMBL13983 | - | 6.39 | -9.22 | True |
| 8 | CHEMBL5564149 | - | 6.31 | -9.08 | True |
| 9 | CHEMBL554993 | - | 6.16 | -9.94 | False |
| 10 | CHEMBL2316916 | - | 7.01 | -7.74 | False |

The four approved EGFR inhibitors occupy ranks 1-4. This is a necessary but
not sufficient sanity check; it confirms the pipeline is not broken, but the
activity gap between controls and the weak candidate library makes high
control ranks expected. The external Spearman and the M9 retrospective
benchmark are the primary evidence of ranking quality.

## Run

ADMET and ranking use the project conda environment:

```bash
conda activate egfr-aidd
python src/models/m7_admet.py
python src/models/m7_ranking.py
```

Docking needs rdkit + meeko + openbabel. Create the docking environment once:

```bash
conda env create -f environment-docking.yml
conda activate egfr-aidd-dock
python src/models/m7_docking.py --top-n 20
```

## Outputs

```text
results/m7_admet_descriptors.csv
results/m7_admet_summary.json
results/m7_docking_results.csv
results/m7_docking_summary.json
results/m7_final_ranking.csv
results/m7_final_ranking.json
results/figures/m7_final_ranking.png
data/docking/1M17.pdb
data/docking/1m17_receptor.pdbqt
data/docking/vina_box.json
data/docking/ligands/
results/m7_docking_out/
```

## Caveats

- Rigid-receptor docking is a screening approximation, not a binding free
  energy calculation.
- The docking box is defined by the erlotinib co-crystal and is appropriate
  for this pocket, but alternative pockets would need separate boxes.
- The composite weights (0.5 / 0.3 / 0.2) are educational defaults; they were
  not tuned and should be treated as an explicit modeling choice.
- ADMET_ok is a rule-based flag, not an experimental ADME measurement.
- Docking ranks do not match experimental activity order even for the
  approved controls (e.g., afatinib affinity rank 10, erlotinib rank 17),
  which is expected for rigid-receptor screening approximations.
