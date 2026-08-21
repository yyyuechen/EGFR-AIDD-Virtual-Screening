# M6 - Ensemble virtual screening

## What this milestone does

M6 combines the three molecular representations trained in M2-M5 into an
ensemble and scores a local candidate library:

```text
Morgan fingerprint  -> RandomForest / XGBoost
molecular graph     -> GCN / GIN
SMILES sequence     -> Transformer

mean predicted pIC50 -> ensemble rank -> shortlist
```

The candidate library is built from ChEMBL molecules that were **not** part
of the M1 modeling dataset, so the models have not been trained on these
compounds.

## Files added

```text
src/models/virtual_screening.py
notebooks/06_virtual_screening.ipynb
data/candidates/egfr_candidate_library.csv
results/models/                 # trained RF/XGB/GCN/GIN/Transformer artifacts
```

## Run

```bash
cd "/path/to/AIDD人工智能药物设计与深度学习蛋白质预测/egfr-aidd-project"
conda activate egfr-aidd

KMP_DUPLICATE_LIB_OK=TRUE python src/models/virtual_screening.py
```

Training all models on the full M1 dataset takes about 12 minutes on CPU.

## Expected outputs

```text
results/m6_virtual_screening_results.json
results/m6_virtual_screening_shortlist.csv
results/figures/m6_virtual_screening_shortlist.png
```

## Top 10 candidates

| Rank | ChEMBL ID | ensemble pIC50 |
|---:|---|---:|
| 1 | CHEMBL2316916 | 7.234 |
| 2 | CHEMBL4531006 | 6.253 |
| 3 | CHEMBL5614250 | 6.196 |
| 4 | CHEMBL4539558 | 6.082 |
| 5 | CHEMBL13983 | 5.993 |
| 6 | CHEMBL2315696 | 5.964 |
| 7 | CHEMBL2315698 | 5.942 |
| 8 | CHEMBL2315409 | 5.914 |
| 9 | CHEMBL484321 | 5.844 |
| 10 | CHEMBL5564149 | 5.834 |

## Interpretation and limitations

* The ensemble is a simple mean of pIC50 predictions from the three
  representation families.
* The candidate library is small (60 molecules) and comes from ChEMBL
  records that were excluded from M1 modeling. It is a demonstration library,
  not a full virtual-screening deck.
* These are computational rankings only. M7 will add ADMET filters and
  docking before anything is called a hit.
* Model artifacts are saved under `results/models/` for reuse in M7/M8.
