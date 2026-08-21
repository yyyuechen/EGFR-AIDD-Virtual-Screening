# Morgan Fingerprint nBits Study

## Purpose and protocol

This controlled study changes exactly one modeling decision at a time: the
Morgan fingerprint dimensionality (`nBits` = 512, 1024, 2048, 4096).

Everything else is fixed:

* radius = 2;
* standard 80/10/10 Bemis-Murcko scaffold split, seed 42;
* training set 8,128 molecules; validation set 1,015 molecules;
* **the standard-scaffold test set (1,018 molecules) is never used**;
* RandomForest from M2/M3: 300 trees, random_state 42, n_jobs = -1;
* preprocessing, endpoint definition, and all other settings unchanged.

Primary metric: validation Spearman. Secondary metrics: validation RMSE,
MAE, R2.

## Results (validation split only)

| nBits | MAE | RMSE | R2 | Spearman | mean on-bits | fingerprint density | unique fingerprints | duplicated fingerprints | mean max train Tanimoto |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.5891 | 0.7684 | 0.6060 | **0.7748** | 59.91 | 0.1170 | 8,778 | 365 | 0.7361 |
| 1024 | 0.5980 | 0.7812 | 0.5928 | 0.7678 | 61.61 | 0.0602 | 8,780 | 363 | 0.7249 |
| 2048 | 0.5947 | 0.7785 | 0.5956 | 0.7689 | 62.44 | 0.0305 | 8,780 | 363 | 0.7198 |
| 4096 | 0.5885 | 0.7744 | 0.5999 | 0.7712 | 62.99 | 0.0154 | 8,780 | 363 | 0.7168 |

Full density and Tanimoto statistics:

| nBits | median on-bits | median density | median max sim | min max sim | max max sim | collision groups | max group size |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 62.0 | 0.1211 | 0.7612 | 0.1977 | 1.000 | 287 | 10 |
| 1024 | 63.0 | 0.0615 | 0.7500 | 0.1974 | 1.000 | 286 | 10 |
| 2048 | 64.0 | 0.0313 | 0.7500 | 0.1646 | 1.000 | 286 | 10 |
| 4096 | 65.0 | 0.0159 | 0.7442 | 0.1750 | 1.000 | 286 | 10 |

## What nBits means

Morgan fingerprints are fixed-length binary bit vectors. Each circular
environment is hashed into one or more of the `nBits` positions, so `nBits`
controls the resolution and density of the hashed fingerprint space. It does
not change what chemical environment radius 2 captures; it changes how those
environments are projected into a fixed-size binary vector.

## What fingerprint collisions are

A collision occurs when two different molecules produce the identical binary
fingerprint vector. In this dataset all canonical SMILES are unique, so
identical fingerprint vectors always correspond to different canonical
structures (verified: 0 duplicate canonical SMILES in the 10,161-molecule
dataset; 9,143 train+validation molecules used for the representation
statistics).

## Why smaller nBits could lose information

With a smaller bit space, more distinct environments must share the same
bits, which can in principle collapse different chemistry into the same
vector. In this study, however, the number of collisions barely changes with
`nBits` (287 groups at 512 bits vs 286 at 1024-4096 bits), so hashing
dimension is not the main source of collisions here.

## Why larger nBits are not guaranteed to improve prediction

More bits reduce density but also make fingerprints sparser and more
sensitive to individual bits. A larger vector does not add new chemistry;
it only changes the projection. The validation results show that 4096 bits
are not clearly better than 512 bits.

## How sparsity/density changes with dimensionality

The absolute number of on-bits stays roughly constant (mean 59.9 to 63.0),
which is expected because radius-2 Morgan environments have a similar number
of features per molecule. Density therefore falls by about half with each
doubling of `nBits`: 0.117 -> 0.060 -> 0.031 -> 0.015. This is exactly the
"same number of bits in a larger vector" effect.

## Does predictive performance improve, saturate, or worsen?

Performance is essentially flat. `nBits = 512` has the highest validation
Spearman (0.7748) and the best R2 (0.6060); `nBits = 1024` is slightly worst
(0.7678 / 0.5928); 2048 and 4096 sit in between. The total Spearman spread is
only 0.007 and R2 spread is 0.013, so the differences are small relative to
model noise.

## Does apparent chemical-space similarity change?

Yes, modestly. `mean_max_train_tanimoto` decreases from 0.7361 (512 bits) to
0.7168 (4096 bits), and the median decreases from 0.761 to 0.744. This is a
consequence of lower density, not a chemical change in the molecules
themselves.

## Does collision frequency actually decrease with nBits?

Almost not. 512 bits has 287 collision groups (365 duplicated fingerprints);
1024, 2048, and 4096 bits each have 286 groups (363 duplicated
fingerprints). The maximum group size is 10 at every setting. Increasing the
bit space beyond 1024 does not materially reduce collisions, because the
observed collisions are dominated by identical radius-2 environments between
different structures (the same representation-collapse phenomenon seen in
the Tanimoto=1 audit), not by hashing overflow.

## Do regression and ranking metrics agree?

Yes. For all four metrics the ordering is identical:

```text
512 > 4096 > 2048 > 1024
```

where ">" means better validation performance. There is no conflict between
regression calibration and ranking in this study.

## Is the difference large enough to change the 2048-bit reference?

No. Although 512 bits has the numerically best validation metrics, the
differences are small (Spearman range 0.007, R2 range 0.013) and well within
the expected run-to-run variation of a single-seed evaluation. The 2048-bit
reference remains a stable, standard configuration and matches all earlier
milestone results. No model or README headline metric is changed.

## Decision classification

**B. Performance plateau.** The 512-4096 bit range performs effectively
equivalently on this validation split. A 512-bit fingerprint shows a slight
numerical edge and 4096 is close to it, but neither is a material
improvement. The current 2048-bit reference is retained.

## Files

```text
src/models/nbits_tuning_study.py
results/nbits_tuning_results.csv
results/nbits_tuning_results.json
results/figures/nbits_validation_performance.png
```

## Reproducibility

```bash
conda activate egfr-aidd
python src/models/nbits_tuning_study.py
```

The JSON records the split, RandomForest parameters, package versions, and
all per-nBits statistics. The test split was not evaluated.
