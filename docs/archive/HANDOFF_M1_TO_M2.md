# EGFR AIDD Project — M1 → M2 Handoff

## Project goal

Educational, portfolio-quality AIDD project that builds an EGFR inhibitor
pipeline and compares three molecular representations for pIC50 prediction:

1. Morgan fingerprint (fixed / handcrafted representation)
2. GNN embedding (learned graph representation)
3. SMILES Transformer embedding (learned sequence representation)

## Course reference location

```text
/path/to/AIDD人工智能药物设计与深度学习蛋白质预测
```

Course material is read-only reference. All new code lives in the project
repository below.

## Project repository location

```text
/path/to/AIDD人工智能药物设计与深度学习蛋白质预测/egfr-aidd-project
```

Note: the repository is now under Git; earlier milestones were reconstructed
from files, timestamps, and saved outputs rather than commits.

## Milestone roadmap

```text
M1. EGFR activity dataset construction   [COMPLETE]
M2. Morgan fingerprint baseline          [COMPLETE]
M3. Model evaluation (random vs scaffold split)
M4. Molecular graph models (GCN / MPNN)
M5. SMILES sequence model (Transformer)
M6. Virtual screening
M7. ADMET + docking
M8. Candidate ranking + final analysis
```

## M1 completed work

M1 was implemented end to end:

* ChEMBL target selection from UniProt P00533, resolving to CHEMBL203
  (wild-type human EGFR, SINGLE PROTEIN).
* Raw activity retrieval and molecule SMILES retrieval.
* IC50-only endpoint filtering, numeric-value sanity checks, and exact
  `standard_relation == "="` filtering.
* Unit normalization to nM.
* SMILES merging, RDKit validation, largest-fragment standardization, and
  canonical SMILES generation.
* `pIC50 = 9 - log10(IC50_nM)`.
* Median pIC50 aggregation over repeated measurements.
* Optional `activity_class` threshold at pIC50 >= 7.
* QC notebook and processing summary.

Key files:

* `src/data/download_chembl.py`
* `src/data/preprocess_egfr.py`
* `notebooks/01_egfr_dataset_qc.ipynb`
* `results/data_processing_summary.json`

## M1 scientific decisions

* Only IC50 is used; Ki, Kd, EC50, and other endpoint types are excluded.
* Only exact `"="` measurements are used for regression. Censored values
  (`<`, `>`) are reported and excluded, not converted.
* Units are converted to nM with explicit factors:
  `M=1e9`, `mM=1e6`, `uM/µM/μM=1e3`, `nM=1`, `pM=1e-3`.
* SMILES are validated by RDKit, reduced to the largest fragment
  (salts/counterions removed), and written as canonical isomeric SMILES.
* Tautomer canonicalization and charge neutralization are intentionally not
  applied in M1.
* Repeated measurements are aggregated with the median pIC50; the final
  `IC50_nM` is back-transformed from the aggregated pIC50.
* The activity threshold is a documented educational choice, not a universal
  biological definition.

## M1 final dataset

Path:

```text
data/processed/egfr_activity_final.csv
```

Shape: `(10161, 11)`

Columns:

```text
canonical_smiles, pIC50, IC50_nM, activity_class, n_measurements,
pIC50_std, pIC50_min, pIC50_max, molecule_chembl_id,
molecule_chembl_ids, n_molecules
```

Observed values from `results/data_processing_summary.json`:

* pIC50 min: 1.602
* pIC50 max: 11.523
* pIC50 mean: 6.943
* pIC50 median: 7.047
* active (`pIC50 >= 7`): 5,261
* inactive: 4,900

## M2 objective

Build the first machine-learning baseline:

```text
canonical_smiles
  -> RDKit Mol
  -> Morgan fingerprint
  -> RandomForest / XGBoost
  -> pIC50 regression
```

M2 deliberately implements only regression and only a random split. It does
not yet implement scaffold split, validation splitting, or classification.

## M2 representation

```text
SMILES
  -> Chem.MolFromSmiles -> RDKit Mol
  -> AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
  -> fixed 2048-bit 0/1 vector
  -> np.vstack -> X matrix with shape (n_molecules, 2048)
```

Important concept: the Morgan fingerprint is a fixed, rule-based molecular
representation, not a learned neural-network embedding. The fingerprint is
computed from local atomic neighborhoods by hashing, and it is expanded into
2048 numeric model features per molecule.

* `radius` controls the neighborhood size: `radius=2` corresponds
  conceptually to ECFP4-style neighborhoods (two bond steps from each atom).
* `nBits=2048` is the fixed fingerprint length, which becomes the number of
  input features. A dataframe column containing fingerprint arrays does not
  mean the model has one feature; each bit is one feature.

## M2 parameters

Extracted from `src/models/morgan_baseline.py`:

```text
fingerprint type: Morgan (bit vector)
radius: 2
n_bits: 2048
RDKit API: AllChem.GetMorganGenerator
split type: random train/test only
test_size: 0.2
seed: 42
n_molecules: 10161
n_train: 8128
n_test: 2033
RandomForest: n_estimators=300, random_state=42, n_jobs=-1
XGBoost: n_estimators=300, learning_rate=0.05, max_depth=6,
         random_state=42, n_jobs=-1, verbosity=0
```

Regression targets: `pIC50` only.

## M2 current implementation status

Implemented:

* `src/models/morgan_baseline.py`
* `notebooks/02_morgan_baseline.ipynb`
* `results/m2_morgan_baseline_results.json`
* `results/figures/m2_morgan_baseline_predicted_vs_actual.png`
* `docs/README_M2.md`

The script covers loading the M1 dataset, SMILES -> RDKit Mol -> Morgan
fingerprint, feature matrix construction, random train/test split, RF/XGBoost
training, regression evaluation, result JSON saving, and a predicted-vs-actual
figure.

The fingerprint step uses the current RDKit API:
`AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)`.

## M2 current test status

The script and notebook were updated from the deprecated
`GetMorganFingerprintAsBitVect` to `MorganGenerator`. A full rerun of
`src/models/morgan_baseline.py` in the `egfr-aidd` conda environment
reproduced the saved metrics exactly with clean output, and
`results/m2_morgan_baseline_results.json` plus the predicted-vs-actual figure
were regenerated. The notebook was then re-executed headlessly and its
outputs were refreshed; no deprecation warnings remain.

## Existing results

From `results/m2_morgan_baseline_results.json`:

```text
RandomForest:
  MAE: 0.5361
  RMSE: 0.7379
  R2: 0.7010
  Pearson_r: 0.8378
  Spearman_rho: 0.8370

XGBoost:
  MAE: 0.6236
  RMSE: 0.8163
  R2: 0.6340
  Pearson_r: 0.7987
  Spearman_rho: 0.8027
```

## Unresolved issues

1. RESOLVED: RDKit deprecation noise from
   `GetMorganFingerprintAsBitVect`. The script and notebook now use
   `MorganGenerator`; reruns produced clean output and identical metrics.
2. No validation split: the milestone description mentions
   train/validation/test, but M2 currently performs only a random
   train/test split. This is a design decision to review before M3.
3. No classification model: `activity_class` exists in M1 but M2 does not
   train a classifier. Regression is the implemented M2 scope.
4. Educational clarity: the M2 notebook is code-first. The conceptual points
   in this handoff (radius meaning, ECFP4, nBits as feature count, expansion
   of fingerprint arrays into X) are not yet written into the notebook in
   detail.
5. Outdated status docs: `../README.md` and `project_roadmap.md` still mark
   M1 as current and M2 as not implemented.
6. No tests directory or unit/smoke tests exist for M1 or M2.

## Exact next step

Recommended single next step: M3 - compare the random-split Morgan baseline
with a scaffold split, using the same fingerprint parameters
(`radius=2`, `nBits=2048`). Do not start M3 until approved.
